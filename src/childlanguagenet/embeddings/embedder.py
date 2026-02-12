"""Embedding helper — Ollama (default), OpenAI, or local sentence-transformers."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from childlanguagenet.config.settings import Settings


def get_embeddings(settings: "Settings"):
    """Return an embedding model instance based on settings.

    Providers (``settings.embedding_provider``):
    * ``"ollama"`` (default) — uses Ollama with ``nomic-embed-text``
    * ``"openai"`` — uses OpenAI ``text-embedding-3-small`` (requires API key)
    * If ``settings.use_local_embeddings`` is ``True``, uses a local
      sentence-transformer model (requires ``sentence-transformers``).
    """
    # --- Local sentence-transformers (legacy flag) -----------------------
    if settings.use_local_embeddings:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=settings.embedding_model)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    # --- Ollama (default) ------------------------------------------------
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    # --- OpenAI ----------------------------------------------------------
    if settings.embedding_provider == "openai":
        settings.require_openai_key()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown embedding_provider '{settings.embedding_provider}'. "
        "Use 'ollama' or 'openai'."
    )


def embedding_cache_key(text: str, model_name: str) -> str:
    """Deterministic cache key for an embedding: SHA-256(text || model_name)."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()
