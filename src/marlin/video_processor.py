"""Long-video chunking and grounding for Marlin-2B.

Pure chunking library — no model loading, no file I/O beyond temp chunks,
no CLI. Splits a video into overlapping chunks, calls a user-supplied
ground_fn on each, normalises local timestamps to global, deduplicates
overlap events, and returns the combined result in memory.

    from marlin.ffmpeg import probe_duration
    from .video_processor import find_in_long_video

    duration = probe_duration(path)
    if duration > 30.0:
        result = find_in_long_video(path, query, m.ground)
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging import get_logger

logger = get_logger("chunk")

# ── constants ─────────────────────────────────────────────────────────────────

from .constants import (
    CHUNK_SECONDS,
    OVERLAP_SECONDS,
    DEDUP_TOLERANCE_SECONDS,
    DEDUP_IOU_THRESHOLD,
    MIN_CHUNK_SECONDS,
    VIDEO_MAX_PIXELS,
    VIDEO_FPS,
)


# ── exceptions ────────────────────────────────────────────────────────────────


class VideoChunkingError(Exception):
    """Base exception for chunking errors."""


class FFmpegError(VideoChunkingError):
    """FFmpeg or ffprobe subprocess failed."""


# ── data models ───────────────────────────────────────────────────────────────
@dataclass
class VideoChunk:
    """One time-window inside the source video."""

    chunk_id: int
    start: float
    end: float
    duration: float
    path: Path | None = None


@dataclass
class GroundingHit:
    """A single grounding match mapped to global time."""

    chunk_id: int
    local_start: float
    local_end: float
    global_start: float
    global_end: float
    description: str
    tier: str


@dataclass
class LongVideoFindResult:
    """Aggregated result of grounding across all chunks."""

    video_path: Path
    duration_seconds: float
    chunk_seconds: float
    overlap_seconds: float
    query: str
    hits: list[GroundingHit]


# ── helpers ───────────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format seconds into MM:SS.ff for log alignment."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


# ── phase 1: probe & plan ────────────────────────────────────────────────────


def generate_chunks(
    duration_seconds: float,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[VideoChunk]:
    """Build a list of overlapping VideoChunk windows across the video."""
    if duration_seconds <= 0:
        raise VideoChunkingError("duration_seconds must be > 0")
    if chunk_seconds <= 0:
        raise VideoChunkingError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise VideoChunkingError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise VideoChunkingError("overlap_seconds must be < chunk_seconds")

    step = chunk_seconds - overlap_seconds
    chunks: list[VideoChunk] = []
    start = 0.0
    chunk_id = 0

    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        if chunks and abs(end - chunks[-1].end) < 1e-6:
            break
        chunks.append(
            VideoChunk(
                chunk_id=chunk_id,
                start=round(start, 6),
                end=round(end, 6),
                duration=round(end - start, 6),
            )
        )
        chunk_id += 1
        if end >= duration_seconds:
            break
        start += step

    # Avoid a uselessly short trailing chunk: fold it into its predecessor so we
    # never spend a model pass on a sub-second tail.
    if len(chunks) > 1 and chunks[-1].duration < MIN_CHUNK_SECONDS:
        tail = chunks.pop()
        prev = chunks[-1]
        prev.end = tail.end
        prev.duration = round(prev.end - prev.start, 6)

    return chunks


# ── phase 2: extract ─────────────────────────────────────────────────────────


def extract_chunk(
    input_video: Path,
    chunk: VideoChunk,
    output_dir: Path,
) -> VideoChunk:
    """Extract a single chunk from the source video via FFmpeg.

    Re-encodes (does NOT stream-copy) to guarantee frame-accuracy. To optimize
    performance and avoid redundant transcoding passes, this also performs
    client-side downscaling (resolution and frame rate) to the model's target
    budget during the extraction step.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(chunk.start * 1000)
    end_ms = int(chunk.end * 1000)
    filename = f"chunk_{chunk.chunk_id:04d}_{start_ms}_{end_ms}.mp4"
    output_path = output_dir / filename

    from .ffmpeg import extract_segment, probe_dimensions, calculate_target_resolution

    # Retrieve source dimensions to calculate target budget
    dims = probe_dimensions(input_video)
    width, height = None, None
    if dims:
        w, h = dims
        tgt = calculate_target_resolution(w, h, max_pixels=VIDEO_MAX_PIXELS)
        if tgt:
            width, height = tgt

    t0 = time.monotonic()
    success = extract_segment(
        input_video,
        start=chunk.start,
        duration=chunk.duration,
        dest=output_path,
        reencode=True,
        with_audio=False,
        reset_timestamps=True,
        width=width,
        height=height,
        fps=VIDEO_FPS,
    )
    elapsed = time.monotonic() - t0
    if not success or not output_path.exists() or output_path.stat().st_size == 0:
        raise FFmpegError(f"FFmpeg failed for chunk {chunk.chunk_id}")

    chunk.path = output_path
    logger.debug("extracted chunk {} in {:.1f}s", chunk.chunk_id, elapsed)
    return chunk


# ── phase 3: dedup ────────────────────────────────────────────────────────────


def dedup_hits(
    hits: list[GroundingHit],
    tolerance: float = DEDUP_TOLERANCE_SECONDS,
    iou_threshold: float = DEDUP_IOU_THRESHOLD,
) -> list[GroundingHit]:
    """Merge near-duplicate hits from overlapping chunks.

    Two hits are treated as the same physical event (and the longer span kept)
    when their spans overlap substantially — intersection-over-union >=
    *iou_threshold*, or they overlap at all and start within *tolerance*
    seconds. Distinct events that merely sit close together (no overlap) are
    preserved, which a start-proximity-only rule would wrongly collapse.
    """
    if not hits:
        return []

    sorted_hits = sorted(hits, key=lambda h: h.global_start)
    kept: list[GroundingHit] = [sorted_hits[0]]

    for candidate in sorted_hits[1:]:
        prev = kept[-1]
        inter = max(
            0.0,
            min(candidate.global_end, prev.global_end)
            - max(candidate.global_start, prev.global_start),
        )
        span = max(candidate.global_end, prev.global_end) - min(
            candidate.global_start, prev.global_start
        )
        iou = inter / span if span > 0 else 0.0
        close_start = abs(candidate.global_start - prev.global_start) <= tolerance

        if iou >= iou_threshold or (inter > 0 and close_start):
            cand_dur = candidate.global_end - candidate.global_start
            prev_dur = prev.global_end - prev.global_start
            if cand_dur > prev_dur:
                kept[-1] = candidate
        else:
            kept.append(candidate)

    return kept


# ── phase 4: main pipeline ───────────────────────────────────────────────────

# Type alias for the callable the CLI passes in (backend.Marlin.ground).
# Signature: ground_fn(video: Path, query: str) -> ((start, end), tier)
GroundFn = Callable[[Path, str], tuple[tuple[float, float], str]]


def find_in_long_video(
    video_path: Path,
    query: str,
    ground_fn: GroundFn,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    dedup_tolerance: float = DEDUP_TOLERANCE_SECONDS,
    on_chunk_start: Callable[[int, int, float, float], Any] | None = None,
) -> LongVideoFindResult:
    """Chunk a long video, run *ground_fn* on each chunk, and return merged hits.

    Parameters
    ----------
    video_path
        Path to the source video file.
    query
        Natural-language description of the event to locate.
    ground_fn
        Callable with signature ``(video: Path, query: str) -> ((start, end), tier)``.
        Typically ``backend.Marlin.ground``.
    chunk_seconds
        Duration per chunk window.
    overlap_seconds
        Overlap between consecutive chunks.
    dedup_tolerance
        Start-time tolerance for deduplication.
    on_chunk_start
        Optional callback ``(chunk_idx, total_chunks, start_sec, end_sec)``
        invoked before each chunk is processed (for progress UI).

    Returns
    -------
    LongVideoFindResult
        Aggregated, deduplicated grounding hits in global time.

    Raises
    ------
    VideoChunkingError
        If the video is missing, or every chunk fails to extract/ground (so an
        all-failure run is never silently reported as "not found").
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoChunkingError(f"Video file not found: {video_path}")

    from .ffmpeg import probe_duration
    duration = probe_duration(video_path)
    if duration == 0.0:
        raise VideoChunkingError(f"Could not probe duration for: {video_path}")
    chunks = generate_chunks(duration, chunk_seconds, overlap_seconds)
    total = len(chunks)

    logger.info(
        "chunking {} ({:.1f}s) into {} chunks ({}s window, {}s overlap)",
        video_path.name,
        duration,
        total,
        chunk_seconds,
        overlap_seconds,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="marlin_chunks_"))
    raw_hits: list[GroundingHit] = []
    errored = 0

    try:
        for chunk in chunks:
            if on_chunk_start:
                on_chunk_start(chunk.chunk_id, total, chunk.start, chunk.end)

            try:
                extract_chunk(video_path, chunk, temp_dir)
                (local_start, local_end), tier = ground_fn(chunk.path, query)
            except Exception as exc:
                errored += 1
                logger.warning("chunk {} failed: {}", chunk.chunk_id, exc)
                continue
            finally:
                # Don't let chunk files accumulate across a multi-hour video.
                if chunk.path is not None:
                    Path(chunk.path).unlink(missing_ok=True)

            if tier == "no_match":
                continue

            # Clamp to chunk boundaries first, then drop empty/inverted spans.
            local_start = max(0.0, min(local_start, chunk.duration))
            local_end = max(0.0, min(local_end, chunk.duration))
            if local_end <= local_start:
                continue

            raw_hits.append(
                GroundingHit(
                    chunk_id=chunk.chunk_id,
                    local_start=round(local_start, 2),
                    local_end=round(local_end, 2),
                    global_start=round(chunk.start + local_start, 2),
                    global_end=round(chunk.start + local_end, 2),
                    description=query,
                    tier=tier,
                )
            )

        if total > 0 and errored == total:
            raise VideoChunkingError(
                f"all {total} chunks failed to extract/ground — see warnings above"
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    deduped = dedup_hits(raw_hits, tolerance=dedup_tolerance)

    logger.info(
        "chunking complete: {} raw hits → {} after dedup",
        len(raw_hits),
        len(deduped),
    )

    return LongVideoFindResult(
        video_path=video_path,
        duration_seconds=duration,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        query=query,
        hits=deduped,
    )


# ── serialisation (for visualizer) ────────────────────────────────────────────


def hits_to_visualizer_events(result: LongVideoFindResult) -> list[dict]:
    """Convert hits to the dict format expected by the HTML visualizer."""
    return [
        {
            "global_start": h.global_start,
            "global_end": h.global_end,
            "description": h.description,
            "chunk_id": h.chunk_id,
        }
        for h in result.hits
    ]
