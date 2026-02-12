"""Embedding helper — OpenAI (default) or local sentence-transformers (flag)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from langchain_openai import OpenAIEmbeddings

if TYPE_CHECKING:
    from childlanguagenet.config.settings import Settings


def get_embeddings(settings: "Settings"):
    """Return an embedding model instance based on settings.

    * Default: OpenAI ``text-embedding-3-small``
    * If ``settings.use_local_embeddings`` is ``True``, uses a local
      sentence-transformer model (requires ``sentence-transformers``).
    """
    settings.require_openai_key()

    if settings.use_local_embeddings:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=settings.embedding_model)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def embedding_cache_key(text: str, model_name: str) -> str:
    """Deterministic cache key for an embedding: SHA-256(text || model_name)."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()
