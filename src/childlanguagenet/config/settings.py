"""Typed settings for ChildLanguageNet RAG system.

Loads from environment variables / .env file with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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

# Load .env from project root (if present)
load_dotenv(PROJECT_ROOT / ".env")


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
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "ollama")
    )

    # --- API keys --------------------------------------------------------
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # --- Ollama ----------------------------------------------------------
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # --- LLM -------------------------------------------------------------
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024"))
    )

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    )
    use_local_embeddings: bool = field(
        default_factory=lambda: os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    )

    # --- Chunking --------------------------------------------------------
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )

    # --- Vectorstore / Retrieval -----------------------------------------
    faiss_index_type: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_TYPE", "Flat")
    )
    faiss_normalize: bool = field(
        default_factory=lambda: os.getenv("FAISS_NORMALIZE", "true").lower() == "true"
    )
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "8"))
    )

    # --- UI --------------------------------------------------------------
    ui_title: str = field(
        default_factory=lambda: os.getenv("UI_TITLE", "Child Language Research RAG")
    )
    max_sources_display: int = field(
        default_factory=lambda: int(os.getenv("MAX_SOURCES_DISPLAY", "10"))
    )

    # --- Telemetry -------------------------------------------------------
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true"
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
                "Set it in your environment or in a .env file at project root."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
