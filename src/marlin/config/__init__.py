"""Configuration resolution: env vars > ~/.marlin/config.json > defaults.

Local mode needs no credentials at all. Hosted mode needs MARLIN_API_KEY
(or the key stored by `marlin setup`). The OpenAI SDK requires a non-empty
api_key string, so local uses the "no-key-required" placeholder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_DIR = Path(os.environ.get("MARLIN_HOME", Path.home() / ".marlin"))
CONFIG_FILE = CONFIG_DIR / "config.json"
JOBS_DIR = CONFIG_DIR / "jobs"
DOWNLOADS_DIR = CONFIG_DIR / "downloads"
DB_DIR = CONFIG_DIR / "index.lancedb"

DEFAULT_MODEL = "NemoStation/Marlin-2B"
DEFAULT_LOCAL_URL = "http://localhost:8000/v1"
NO_KEY = "no-key-required"


def _default_db_path() -> str:
    """Return the default LanceDB path under the active Marlin home."""
    home = Path(os.environ.get("MARLIN_HOME", Path.home() / ".marlin"))
    return str(home / "index.lancedb")


class Config(BaseModel):
    """Runtime configuration for local or hosted Marlin inference.

    Attributes
    ----------
    mode
        Backend mode: ``local`` or ``hosted``.
    base_url
        OpenAI-compatible API base URL.
    api_key
        Hosted API key. Local mode uses a placeholder when this is empty.
    model
        Served model name.
    engine
        Concrete local engine name or ``auto``.
    mlx_weights
        MLX weights repository or local path.
    embed_model
        Sentence embedding model used by experimental indexing.
    chunk_seconds
        Experimental index chunk duration.
    chunk_overlap
        Experimental index chunk overlap.
    db_path
        LanceDB path for the experimental index.
    extra
        Forward-compatible extension bag.
    """

    model_config = ConfigDict(validate_assignment=True)

    mode: str = "local"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "NemoStation/Marlin-2B"
    engine: str = "auto"
    mlx_weights: str = "NemoStation/Marlin-2B-MLX-8bit"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_seconds: float = 30.0
    chunk_overlap: float = 5.0
    db_path: str = Field(default_factory=_default_db_path)
    extra: dict = Field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        """Return a non-empty API key string for OpenAI-compatible clients."""
        return self.api_key or "no-key-required"


class LoggingConfig(BaseModel):
    """Resolved production logging settings.

    Attributes
    ----------
    stderr_enabled
        Whether records are emitted to stderr.
    file_enabled
        Whether records are emitted to the rotating log file.
    stderr_level
        Minimum stderr level.
    file_level
        Minimum file sink level.
    log_file
        Path to the active log file.
    rotation
        Loguru rotation policy for the file sink.
    retention
        Loguru retention policy for rotated files.
    compression
        Compression format for rotated files.
    serialize
        Whether sinks should use Loguru JSON serialization.
    diagnose
        Whether Loguru should include local variable diagnostics in tracebacks.
    enqueue
        Whether records are queued for non-blocking and multiprocess-safe writes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stderr_enabled: bool
    file_enabled: bool
    stderr_level: str
    file_level: str
    log_file: Path
    rotation: str
    retention: str
    compression: str
    serialize: bool
    diagnose: bool
    enqueue: bool


def load() -> Config:
    """Load configuration from disk, then apply environment overrides."""
    cfg = Config()
    if CONFIG_FILE.is_file():
        try:
            cfg = Config.model_validate(json.loads(CONFIG_FILE.read_text()))
        except (json.JSONDecodeError, ValidationError):
            pass
    # Env always wins — the non-interactive/agent path.
    if os.environ.get("MARLIN_BASE_URL"):
        cfg.base_url = os.environ["MARLIN_BASE_URL"].rstrip("/")
        cfg.mode = "local" if _looks_local(cfg.base_url) else "hosted"
    if os.environ.get("MARLIN_API_KEY"):
        cfg.api_key = os.environ["MARLIN_API_KEY"]
    if os.environ.get("MARLIN_MODEL"):
        cfg.model = os.environ["MARLIN_MODEL"]
    if os.environ.get("MARLIN_ENGINE"):
        cfg.engine = os.environ["MARLIN_ENGINE"]
    if os.environ.get("MARLIN_MLX_WEIGHTS"):
        cfg.mlx_weights = os.environ["MARLIN_MLX_WEIGHTS"]
    return cfg


def save(cfg: Config) -> Path:
    """Persist configuration to ``~/.marlin/config.json``."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg.model_dump(), indent=2) + "\n")
    CONFIG_FILE.chmod(0o600)
    return CONFIG_FILE


def configured() -> bool:
    """Return whether Marlin has explicit local configuration."""
    return CONFIG_FILE.is_file() or bool(os.environ.get("MARLIN_BASE_URL"))


def _looks_local(url: str) -> bool:
    """Return whether a base URL points at a local server."""
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))
