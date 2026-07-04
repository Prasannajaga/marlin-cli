"""Unit tests for the centralized marlin.ffmpeg API."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure tests fallback to testing env
os.environ.setdefault("MARLIN_ENV", "testing")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin.ffmpeg import (
    has_ffmpeg,
    probe_duration,
    extract_segment,
    extract_proxy,
    extract_frame,
    trim_clip,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import have_ffmpeg as fixture_have_ffmpeg, make_sample_video


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_has_ffmpeg():
    assert has_ffmpeg() is True


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_probe_duration():
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "test.mp4"
        make_sample_video(video_path, duration=5.0)
        assert abs(probe_duration(video_path) - 5.0) < 0.2


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_extract_segment_reencode():
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "test.mp4"
        make_sample_video(video_path, duration=10.0)

        dest_path = Path(td) / "segment.mp4"
        # Grounding mode: reencode=True, with_audio=False
        success = extract_segment(
            video_path,
            start=2.0,
            duration=3.0,
            dest=dest_path,
            reencode=True,
            with_audio=False,
            reset_timestamps=True,
        )
        assert success is True
        assert dest_path.exists()
        assert dest_path.stat().st_size > 0
        assert abs(probe_duration(dest_path) - 3.0) < 0.2


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_extract_segment_copy():
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "test.mp4"
        make_sample_video(video_path, duration=10.0)

        dest_path = Path(td) / "segment_copy.mp4"
        # Indexing mode: reencode=False, with_audio=True
        success = extract_segment(
            video_path,
            start=1.0,
            duration=4.0,
            dest=dest_path,
            reencode=False,
            with_audio=True,
        )
        assert success is True
        assert dest_path.exists()
        assert dest_path.stat().st_size > 0
        # Stream copy snaps to GOP boundaries, so duration will be at least 4.0s
        assert probe_duration(dest_path) >= 4.0


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_extract_proxy():
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "test.mp4"
        make_sample_video(video_path, duration=5.0)

        proxy_path = Path(td) / "proxy.mp4"
        success = extract_proxy(video_path, proxy_path)
        assert success is True
        assert proxy_path.exists()
        assert proxy_path.stat().st_size > 0


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_extract_frame():
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "test.mp4"
        make_sample_video(video_path, duration=5.0)

        frame_path = Path(td) / "frame.jpg"
        success = extract_frame(video_path, time_offset=2.5, dest=frame_path)
        assert success is True
        assert frame_path.exists()
        assert frame_path.stat().st_size > 0


@pytest.mark.skipif(not fixture_have_ffmpeg(), reason="ffmpeg or ffprobe not available")
def test_trim_clip():
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "test.mp4"
        make_sample_video(video_path, duration=8.0)

        trim_path = Path(td) / "trim.mp4"
        success = trim_clip(video_path, start=1.5, duration=3.0, dest=trim_path)
        assert success is True
        assert trim_path.exists()
        assert trim_path.stat().st_size > 0
        # Stream copy snaps to GOP boundaries, so duration will be at least 3.0s
        assert probe_duration(trim_path) >= 3.0
