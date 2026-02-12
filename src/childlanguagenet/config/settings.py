"""Typed settings for ChildLanguageNet RAG system.

Loads configuration from ``config.toml`` (committed to git) with optional
environment-variable overrides for Docker / CI.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Resolve project root: walk up from this file until we find pyproject.toml
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()


def _find_project_root() -> Path:
    """Walk up from this file to locate the directory containing pyproject.toml."""
    current = _THIS_FILE.parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: assume CWD
    return Path.cwd()


PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------------------
# Load config.toml — env vars always win (for Docker / CI overrides)
# ---------------------------------------------------------------------------
_CONFIG: dict[str, Any] = {}
_config_path = PROJECT_ROOT / "config.toml"
if _config_path.exists():
    with open(_config_path, "rb") as fp:
        _CONFIG = tomllib.load(fp)


def _cfg(env_key: str, *toml_keys: str, default: str = "") -> str:
    """Return env var > nested TOML value > default."""
    val = os.getenv(env_key)
    if val is not None:
        return val
    # Walk nested dict
    node: Any = _CONFIG
    for k in toml_keys:
        if isinstance(node, dict):
            node = node.get(k)
        else:
            node = None
            break
    if node is not None:
        return str(node)
    return default


@dataclass(frozen=True)
class Settings:
    """Immutable, typed configuration for the RAG system."""

    # --- Paths -----------------------------------------------------------
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    metadata_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "metadata.json")
    pdf_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "pdf")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "index" / "faiss")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")

    # --- Provider selection ("ollama" or "openai") -------------------------
    llm_provider: str = field(
        default_factory=lambda: _cfg("LLM_PROVIDER", "llm", "provider", default="ollama")
    )
    embedding_provider: str = field(
        default_factory=lambda: _cfg(
            "EMBEDDING_PROVIDER", "embeddings", "provider", default="ollama"
        )
    )

    # --- API keys (only needed when provider=openai) ----------------------
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # --- Ollama ----------------------------------------------------------
    ollama_base_url: str = field(
        default_factory=lambda: _cfg(
            "OLLAMA_BASE_URL", "ollama", "base_url", default="http://localhost:11434"
        )
    )

    # --- LLM -------------------------------------------------------------
    llm_model: str = field(
        default_factory=lambda: _cfg("LLM_MODEL", "llm", "model", default="llama3.2")
    )
    temperature: float = field(
        default_factory=lambda: float(
            _cfg("TEMPERATURE", "llm", "temperature", default="0.2")
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            _cfg("MAX_TOKENS", "llm", "max_tokens", default="1024")
        )
    )

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _cfg(
            "EMBEDDING_MODEL", "embeddings", "model", default="nomic-embed-text"
        )
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(
            _cfg("EMBEDDING_BATCH_SIZE", "embeddings", "batch_size", default="64")
        )
    )
    use_local_embeddings: bool = field(
        default_factory=lambda: _cfg(
            "USE_LOCAL_EMBEDDINGS", "embeddings", "use_local", default="false"
        ).lower() == "true"
    )

    # --- Chunking --------------------------------------------------------
    chunk_size: int = field(
        default_factory=lambda: int(
            _cfg("CHUNK_SIZE", "chunking", "chunk_size", default="500")
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(
            _cfg("CHUNK_OVERLAP", "chunking", "chunk_overlap", default="50")
        )
    )

    # --- Vectorstore / Retrieval -----------------------------------------
    faiss_index_type: str = field(
        default_factory=lambda: _cfg("FAISS_INDEX_TYPE", default="Flat")
    )
    faiss_normalize: bool = field(
        default_factory=lambda: _cfg("FAISS_NORMALIZE", default="true").lower() == "true"
    )
    default_top_k: int = field(
        default_factory=lambda: int(
            _cfg("DEFAULT_TOP_K", "retrieval", "default_top_k", default="8")
        )
    )

    # --- UI --------------------------------------------------------------
    ui_title: str = field(
        default_factory=lambda: _cfg(
            "UI_TITLE", "ui", "title", default="Child Language Research RAG"
        )
    )
    max_sources_display: int = field(
        default_factory=lambda: int(
            _cfg("MAX_SOURCES_DISPLAY", "ui", "max_sources_display", default="10")
        )
    )

    # --- Telemetry -------------------------------------------------------
    log_level: str = field(
        default_factory=lambda: _cfg("LOG_LEVEL", "telemetry", "log_level", default="INFO")
    )
    metrics_enabled: bool = field(
        default_factory=lambda: _cfg(
            "METRICS_ENABLED", "telemetry", "metrics_enabled", default="true"
        ).lower() == "true"
    )

    # --- Helpers ---------------------------------------------------------
    @property
    def uses_openai(self) -> bool:
        """Return True if either LLM or embedding provider is openai."""
        return self.llm_provider == "openai" or self.embedding_provider == "openai"

    def require_openai_key(self) -> str:
        """Return the OpenAI API key or raise with a clear message."""
        if not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Set it as an environment variable."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
