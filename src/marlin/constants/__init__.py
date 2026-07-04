"""Marlin constant settings, protocol limits, and fixed identifiers.

This module houses all non-configurable system constants and default values,
ensuring no magic numbers exist in views, controllers, or business logic.
"""
from __future__ import annotations

import os
from marlin.config import CONFIG_DIR

# --- Platform & Model Defaults ---
DEFAULT_MODEL = "NemoStation/Marlin-2B"
DEFAULT_LOCAL_URL = "http://localhost:8000/v1"
NO_KEY = "no-key-required"
DEFAULT_MLX_WEIGHTS = "NemoStation/Marlin-2B-MLX-8bit"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Video Grounding Limits ---
# Marlin's per-frame budget (modeling_marlin.py: VIDEO_MAX_PIXELS=200704, 2 fps,
# patch×merge = 16×2 = 32). The server already smart_resizes to this, but only
# after decoding full-res frames — so we downscale on the client to the same
# target first: same frames the model would see, far cheaper to decode + much
# less memory (a 4K decode is the thing that OOMs weak machines).
VIDEO_MAX_PIXELS = 200704
VIDEO_FPS = 2.0
FACTOR = 32

# --- Video Chunking Parameters ---
# 30s/5s matches Config.chunk_seconds/chunk_overlap and the existing chunker.py.
# It is a correctness requirement, not just a cost choice: vLLM's Qwen3-VL path
# compresses timestamps on long clips (vllm#30847), so chunks must stay <=30s to
# ground inside the model's training distribution. See backend.py module docstring.
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 5.0
DEDUP_TOLERANCE_SECONDS = 5.0
DEDUP_IOU_THRESHOLD = 0.5
# Fold a trailing chunk shorter than this into its predecessor.
MIN_CHUNK_SECONDS = 2.0

# --- Video Mime Types Mapping ---
MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
}

# --- Engine Setup Constants ---
# Public MLX weights (Apple Silicon) — same repo string as cfg.mlx_weights (HF
# canonical / fallback).
MLX_REPO = "NemoStation/Marlin-2B-MLX-8bit"
# Apple-Silicon multimodal SGLang support lives on this fork branch until upstream.
SGLANG_FORK = "https://github.com/Itssshikhar/sglang"
SGLANG_BRANCH = "codex/marlin-mlx-mm-support"
LOCAL_PORT = 8000

# Validated SGLang-MLX video sampling (matches Marlin's training distribution)
# and the architecture override (config.json brands the arch as a Marlin
# subclass of Qwen3_5ForConditionalGeneration).
MM_CONFIG = '{"video":{"fps":2.0,"min_frames":4,"max_frames":240,"max_pixels":200704}}'
ARCH_OVERRIDE = '{"architectures":["Qwen3_5ForConditionalGeneration"]}'

# --- Weights Signatures ---
WEIGHT_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "model.safetensors",
    "model.safetensors.index.json",
    "modeling_marlin.py",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

# Pinned SHA256 of each weight file (the known-good model). The engine runs the
# downloaded modeling_marlin.py via --trust-remote-code, so we verify every file
# after download: a tampered mirror, a hijacked MARLIN_MLX_WEIGHTS_URL, or a
# transient CDN corruption all fail closed → fall back to Hugging Face.
WEIGHT_SHA256 = {
    "config.json": "d6ab48208818ce26017d65ab64b30f0c419227d4d219c69999305507e3584554",
    "generation_config.json": "0d54a28c36c5143413aa18a910f661d17483909e87321f34ad58859b66cf25b4",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "model.safetensors": "0575deb6f9e9f68405979f4e7d4cf42f0361774f722a480180dff95d414466bf",
    "model.safetensors.index.json": (
        "83ee945a045012893ec93b0074510f4e581ac4130b7284307137ffeb35ba0ef2"
    ),
    "modeling_marlin.py": "baec8f0a2cabf5d64a883d8b5ac1881a9119f91f2251e6964f45447d24b19733",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "processor_config.json": "566b6b6ff98913f6b18295b03c03244875bf6c15f96db586d8b024060e54f0c3",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
}


# Fast weights mirror — Azure Blob (anonymous, no HF rate limit). The CLI curls
# these into WEIGHTS_DIR and serves the engine from there; HF is the fallback.
# Override the mirror with MARLIN_MLX_WEIGHTS_URL.
MLX_WEIGHTS_URL = os.environ.get(
    "MARLIN_MLX_WEIGHTS_URL", "https://marlinweights-d7btf0h4c5dxbzdk.z01.azurefd.net/weights"
).rstrip("/")
WEIGHTS_DIR = CONFIG_DIR / "weights" / "marlin-2b-mlx-8bit"
_WEIGHTS_DONE = WEIGHTS_DIR / ".complete"  # Azure mirror fetched + verified here
_HF_DONE = WEIGHTS_DIR / ".hf_complete"  # HF (token) fetched into the HF cache

ENGINES_DIR = CONFIG_DIR / "engines"
MLX_ENGINE_DIR = ENGINES_DIR / "sglang-mlx"