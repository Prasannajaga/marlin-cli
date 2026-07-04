"""ffmpeg chunking — the pattern proven in sentrysearch, tuned for Marlin.

Stream-copy 30s chunks with 5s overlap; skip parked-camera/still chunks via
a 3-frame JPEG-entropy check; re-encode a small 480p proxy for the API
payload (raw chunk keeps its audio for optional STT).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .models import Chunk
from .ffmpeg import (
    has_ffmpeg,
    probe_duration as ffmpeg_probe_duration,
    extract_segment,
    extract_proxy,
    extract_frame,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def check_ffmpeg() -> bool:
    """Return whether both ``ffmpeg`` and ``ffprobe`` are available."""
    return has_ffmpeg()


def probe_duration(path: Path) -> float:
    """Return the media duration in seconds.

    Parameters
    ----------
    path
        Video file to inspect.
    """
    return ffmpeg_probe_duration(path)


def chunk_spans(
    duration: float, chunk: float = 30.0, overlap: float = 5.0
) -> list[tuple[float, float]]:
    """Return deterministic overlapping chunk spans.

    Parameters
    ----------
    duration
        Source duration in seconds.
    chunk
        Desired chunk length in seconds.
    overlap
        Overlap between adjacent chunks in seconds.
    """
    if duration <= 0:
        return []
    if duration <= chunk:
        return [(0.0, duration)]
    step = max(chunk - overlap, 1.0)
    spans = []
    t = 0.0
    while t < duration:
        end = min(t + chunk, duration)
        spans.append((round(t, 2), round(end, 2)))
        if end >= duration:
            break
        t += step
    return spans


def extract_chunk(source: Path, start: float, end: float, workdir: Path) -> Chunk | None:
    """Extract a raw chunk and model proxy.

    Parameters
    ----------
    source
        Source video.
    start
        Start time in seconds.
    end
        End time in seconds.
    workdir
        Temporary directory for extracted files.

    Returns
    -------
    Chunk | None
        Extracted chunk metadata, or ``None`` when ffmpeg fails.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    dur = end - start
    raw = workdir / f"raw_{start:.0f}.mp4"

    # Attempt fast stream-copy first (reencode=False) and fall back to re-encoding
    success = extract_segment(
        source,
        start=start,
        duration=dur,
        dest=raw,
        reencode=False,
        with_audio=True,
        reset_timestamps=False,
    )
    if not success or not raw.exists() or raw.stat().st_size == 0:
        return None

    proxy = workdir / f"proxy_{start:.0f}.mp4"
    success = extract_proxy(raw, proxy, width=-2, height=480, fps=5)
    if not success or not proxy.exists() or proxy.stat().st_size == 0:
        return None

    return Chunk(source=source, start=start, end=end, raw=raw, proxy=proxy)


def is_still_chunk(chunk: Chunk, threshold: float = 0.98) -> bool:
    """Return whether a chunk appears visually static.

    Parameters
    ----------
    chunk
        Extracted chunk to inspect.
    threshold
        Minimum ratio between sampled JPEG frame sizes.
    """
    sizes = []
    with tempfile.TemporaryDirectory() as td:
        for i, frac in enumerate((0.1, 0.5, 0.9)):
            out = Path(td) / f"f{i}.jpg"
            success = extract_frame(chunk.proxy, chunk.duration * frac, out)
            if success and out.exists():
                sizes.append(out.stat().st_size)

    if len(sizes) < 3 or min(sizes) == 0:
        return False
    return (min(sizes) / max(sizes)) >= threshold
