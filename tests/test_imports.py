"""Smoke tests for import sanity."""


def test_import_package():
    import childlanguagenet
    assert hasattr(childlanguagenet, "__version__")


def test_import_settings():
    from childlanguagenet.config.settings import Settings
    assert Settings is not None


def test_import_metadata_registry():
    from childlanguagenet.ingestion.metadata_registry import validate_metadata
    assert callable(validate_metadata)


def test_import_chunking():
    from childlanguagenet.ingestion.chunking import chunk_documents
    assert callable(chunk_documents)


def test_import_faiss_store():
    from childlanguagenet.vectorstore.faiss_store import FAISSStore
    assert FAISSStore is not None


def test_import_rag_graph():
    from childlanguagenet.graph.rag_graph import build_rag_graph
    assert callable(build_rag_graph)


def test_import_cite():
    from childlanguagenet.citations.cite import Citation, format_sources_section
    assert Citation is not None
    assert callable(format_sources_section)


def test_import_telemetry():
    from childlanguagenet.telemetry.logging import get_logger
    from childlanguagenet.telemetry.metrics import get_metrics
    assert callable(get_logger)
    assert callable(get_metrics)
