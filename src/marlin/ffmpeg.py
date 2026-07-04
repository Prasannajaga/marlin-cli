"""Unified wrapper for all FFmpeg and FFprobe operations in Marlin."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from .logging import get_logger

logger = get_logger("ffmpeg")

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


class FFmpegExecutionError(Exception):
    """Raised when an FFmpeg or FFprobe subprocess execution fails."""


def has_ffmpeg() -> bool:
    """Return whether both ffmpeg and ffprobe are installed and available."""
    return shutil.which(FFMPEG_BIN) is not None and shutil.which(FFPROBE_BIN) is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess command and capture its output."""
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def probe_duration(video_path: Path | str) -> float:
    """Return the media duration of a video file in seconds using ffprobe.

    Parameters
    ----------
    video_path
        Path to the video file to inspect.

    Returns
    -------
    float
        Duration of the video in seconds, or 0.0 if unable to probe.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return 0.0

    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    r = _run(cmd)
    try:
        data = json.loads(r.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0


def extract_segment(
    source: Path,
    start: float,
    duration: float,
    dest: Path,
    *,
    reencode: bool = True,
    with_audio: bool = True,
    reset_timestamps: bool = True,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> bool:
    """Extract a time-accurate segment from a video file.

    Parameters
    ----------
    source
        Path to the source video.
    start
        Start offset in seconds.
    duration
        Segment length in seconds.
    dest
        Path to the output segment file.
    reencode
        If True, re-encodes the stream. If False, attempts a fast stream-copy
        first and falls back to re-encoding.
    with_audio
        Whether to preserve the audio stream.
    reset_timestamps
        Whether to reset frame timestamps in the output to start from zero.
    width
        Optional downscaled width target.
    height
        Optional downscaled height target.
    fps
        Optional frame-rate target.

    Returns
    -------
    bool
        True if extraction succeeded, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Build video filter if scaling or frame-rate adjustment is requested
    vf_filters = []
    if width is not None and height is not None:
        vf_filters.append(f"scale={width}:{height}")
    if fps is not None:
        vf_filters.append(f"fps={fps}")
    vf_arg = ["-vf", ",".join(vf_filters)] if vf_filters else []

    if not reencode and not vf_filters:
        # 1. Attempt fast stream-copy (only if no filtering/scaling is requested)
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(source),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(dest),
        ]
        r = _run(cmd)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True

        # 2. Fall back to re-encoding
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
        ]
        if with_audio:
            cmd += ["-c:a", "aac"]
        else:
            cmd += ["-an"]
        cmd += vf_arg
        cmd.append(str(dest))
        r = _run(cmd)
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0

    else:
        # Frame-accurate grounding extraction
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-map",
            "0:v:0",
        ]
        if not with_audio:
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
        cmd += vf_arg
        if reset_timestamps:
            cmd += ["-reset_timestamps", "1"]
        cmd.append(str(dest))

        r = _run(cmd)
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def probe_dimensions(video_path: Path | str) -> tuple[int, int] | None:
    """Return width and height of a video file via ffprobe."""
    try:
        video_path = Path(video_path)
        cmd = [
            FFPROBE_BIN,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(video_path),
        ]
        r = _run(cmd)
        out = r.stdout.strip()
        w, h = (int(x) for x in out.split("x")[:2])
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def calculate_target_resolution(w: int, h: int, max_pixels: int, factor: int = 32) -> tuple[int, int] | None:
    """Return the model-compatible downscaled resolution or None if no downscaling is needed."""
    if w * h <= max_pixels:
        return None
    beta = math.sqrt(w * h / max_pixels)
    w2 = max(factor, math.floor(w / beta / factor) * factor)
    h2 = max(factor, math.floor(h / beta / factor) * factor)
    return w2, h2


def extract_proxy(
    source: Path,
    dest: Path,
    *,
    width: int = -2,
    height: int = 480,
    fps: int = 5,
) -> bool:
    """Create a lower resolution, low frame rate proxy of a video segment.

    Parameters
    ----------
    source
        Path to the high-res chunk segment.
    dest
        Path to save the low-res proxy.
    width
        Proxy width constraint.
    height
        Proxy height constraint.
    fps
        Proxy output frames per second.

    Returns
    -------
    bool
        True if the proxy creation succeeded, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height},fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-an",
        str(dest),
    ]
    r = _run(cmd)
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def extract_frame(source: Path, time_offset: float, dest: Path) -> bool:
    """Extract a single frame from the video at the given time offset.

    Parameters
    ----------
    source
        Path to the video file.
    time_offset
        Offset in seconds where the frame is located.
    dest
        Output JPEG file path.

    Returns
    -------
    bool
        True if the frame was successfully extracted, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{time_offset:.2f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "5",
        str(dest),
    ]
    r = _run(cmd)
    return r.returncode == 0 and dest.exists()


def trim_clip(source: Path, start: float, duration: float, dest: Path) -> bool:
    """Trim a video segment (trying stream-copy first, falling back to re-encoding).

    Parameters
    ----------
    source
        Path to the source video.
    start
        Padded start offset in seconds.
    duration
        Padded duration in seconds.
    dest
        Path to save the output clip.

    Returns
    -------
    bool
        True if trimming succeeded, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try copy
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(source),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(dest),
    ]
    r = _run(cmd)
    if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
        return True

    # 2. Try reencode
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(dest),
    ]
    r = _run(cmd)
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
