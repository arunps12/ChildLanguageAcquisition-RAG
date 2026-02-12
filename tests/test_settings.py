"""Smoke tests for settings loading."""

from childlanguagenet.config.settings import Settings, get_settings


def test_settings_defaults():
    """Settings can be instantiated with defaults."""
    s = Settings()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert s.default_top_k == 8
    assert s.llm_provider in ("ollama", "openai")
    assert s.embedding_provider in ("ollama", "openai")
    assert s.data_dir.name == "data"
    assert s.index_dir.name == "faiss"


def test_get_settings_cached():
    """get_settings returns a cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
