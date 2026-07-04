#!/usr/bin/env python3
"""Side-by-side benchmark comparing legacy FFmpeg functions with new marlin.ffmpeg API."""

from __future__ import annotations

import os
import sys
import time
import tempfile
import json
import subprocess
from pathlib import Path

# Ensure tests fallback to testing env
os.environ.setdefault("MARLIN_ENV", "testing")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import marlin.ffmpeg as new_ffmpeg
import marlin.video_processor as legacy_vp
import marlin.chunker as legacy_ck
from marlin.video_processor import VideoChunk
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import have_ffmpeg


def run_benchmark():
    if not have_ffmpeg():
        print("ERROR: ffmpeg and ffprobe are required to run the benchmark.")
        sys.exit(1)

    print("=" * 70)
    print(" FFmpeg Wrapper Side-by-Side Performance & Correctness Benchmark")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        source_video = Path("/data/nemostation/data/10-minutes-goals.mp4")
        if not source_video.is_file():
            print(f"ERROR: Benchmark source video not found at {source_video}")
            sys.exit(1)
        print(f"Using video: {source_video.name} ({source_video.stat().st_size / (1024 * 1024):.1f} MB)\n")

        # ── Test 1: Duration Probing ──────────────────────────────────────────
        print("1. [Benchmarking] Duration Probing")
        
        # Legacy
        def legacy_probe_duration_seconds(video_path):
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            return float(json.loads(r.stdout)["format"]["duration"])

        t0 = time.perf_counter()
        legacy_dur = legacy_probe_duration_seconds(source_video)
        legacy_probe_time = (time.perf_counter() - t0) * 1000
        
        # New
        t0 = time.perf_counter()
        new_dur = new_ffmpeg.probe_duration(source_video)
        new_probe_time = (time.perf_counter() - t0) * 1000
        
        print(f"   Legacy probe: {legacy_dur:.2f}s (took {legacy_probe_time:.2f} ms)")
        print(f"   New probe:    {new_dur:.2f}s (took {new_probe_time:.2f} ms)")
        assert abs(legacy_dur - new_dur) < 1e-3, "Duration mismatch!"
        print("   Status: [OK] Durations match perfectly.\n")

        # ── Test 2: Grounding Segment Extraction ──────────────────────────────
        print("2. [Benchmarking] Grounding Chunk Extraction (Re-encode, Video-only)")
        chunk = VideoChunk(chunk_id=1, start=5.0, end=15.0, duration=10.0)
        
        # Legacy
        legacy_ground_dest = temp_dir / "legacy_ground.mp4"
        chunk_legacy = VideoChunk(chunk_id=1, start=5.0, end=15.0, duration=10.0)
        t0 = time.perf_counter()
        legacy_vp.extract_chunk(source_video, chunk_legacy, temp_dir)
        # Rename to distinct file
        if chunk_legacy.path:
            chunk_legacy.path.rename(legacy_ground_dest)
        legacy_ground_time = time.perf_counter() - t0
        legacy_ground_sz = legacy_ground_dest.stat().st_size if legacy_ground_dest.exists() else 0

        # New
        new_ground_dest = temp_dir / "new_ground.mp4"
        t0 = time.perf_counter()
        new_ffmpeg.extract_segment(
            source_video,
            start=5.0,
            duration=10.0,
            dest=new_ground_dest,
            reencode=True,
            with_audio=False,
            reset_timestamps=True,
        )
        new_ground_time = time.perf_counter() - t0
        new_ground_sz = new_ground_dest.stat().st_size if new_ground_dest.exists() else 0

        print(f"   Legacy: {legacy_ground_time:.3f}s (size: {legacy_ground_sz / 1024:.1f} KB)")
        print(f"   New:    {new_ground_time:.3f}s (size: {new_ground_sz / 1024:.1f} KB)")
        assert new_ground_dest.exists() and new_ground_sz > 0, "New extraction output missing/empty!"
        print("   Status: [OK] Grounding segments generated successfully.\n")

        # ── Test 3: Chunker Index Extraction (Copy Mode) ──────────────────────
        print("3. [Benchmarking] Index Chunk Extraction (Stream Copy Mode)")
        
        # Legacy
        t0 = time.perf_counter()
        legacy_chunk_meta = legacy_ck.extract_chunk(source_video, start=10.0, end=20.0, workdir=temp_dir)
        legacy_index_time = time.perf_counter() - t0
        legacy_index_sz = legacy_chunk_meta.raw.stat().st_size if legacy_chunk_meta and legacy_chunk_meta.raw.exists() else 0
        
        # New
        new_index_dest = temp_dir / "new_index_raw.mp4"
        t0 = time.perf_counter()
        new_ffmpeg.extract_segment(
            source_video,
            start=10.0,
            duration=10.0,
            dest=new_index_dest,
            reencode=False,
            with_audio=True,
            reset_timestamps=True,
        )
        new_index_time = time.perf_counter() - t0
        new_index_sz = new_index_dest.stat().st_size if new_index_dest.exists() else 0
        
        print(f"   Legacy: {legacy_index_time:.3f}s (size: {legacy_index_sz / 1024:.1f} KB)")
        print(f"   New:    {new_index_time:.3f}s (size: {new_index_sz / 1024:.1f} KB)")
        assert new_index_dest.exists() and new_index_sz > 0, "New index extraction output missing/empty!"
        print("   Status: [OK] Chunks extracted successfully.\n")

        # ── Summary Report ────────────────────────────────────────────────────
        print("=" * 70)
        print(" SUMMARY COMPARISON REPORT")
        print("=" * 70)
        print(f" Probing:          Legacy: {legacy_probe_time:6.2f} ms | New: {new_probe_time:6.2f} ms")
        print(f" Grounding Extract: Legacy: {legacy_ground_time:6.3f} s  | New: {new_ground_time:6.3f} s")
        print(f" Chunker Extract:   Legacy: {legacy_index_time:6.3f} s  | New: {new_index_time:6.3f} s")
        print("=" * 70)
        print(" All comparison tests passed cleanly.")
        print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
