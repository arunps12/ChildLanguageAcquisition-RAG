"""CLI entry points for ChildLanguageNet RAG.

Scripts:
    childrag        — main CLI (help + subcommands)
    childrag-index  — build/update FAISS index
    childrag-serve  — run Streamlit app
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from childlanguagenet.config.settings import get_settings


# ── helpers ────────────────────────────────────────────────────────────────

def _build_index(settings=None):
    """Build FAISS index from metadata registry."""
    from childlanguagenet.ingestion.metadata_registry import validate_metadata
    from childlanguagenet.ingestion.loaders import load_all_papers
    from childlanguagenet.ingestion.chunking import chunk_documents
    from childlanguagenet.embeddings.embedder import get_embeddings
    from childlanguagenet.vectorstore.faiss_store import FAISSStore
    from childlanguagenet.telemetry.logging import get_logger

    settings = settings or get_settings()
    logger = get_logger(__name__)

    logger.info("Validating metadata registry …")
    records = validate_metadata(settings.metadata_file)
    logger.info("Loaded %d records from metadata.json", len(records))

    logger.info("Loading documents …")
    docs = load_all_papers(records, data_dir=settings.data_dir)
    logger.info("Loaded %d raw document pages", len(docs))

    logger.info("Chunking documents …")
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Created %d chunks", len(chunks))

    logger.info("Building FAISS index …")
    embeddings = get_embeddings(settings)
    store = FAISSStore(
        index_dir=settings.index_dir,
        embedding=embeddings,
        k_default=settings.default_top_k,
    )
    store.build(chunks)
    store.save()
    logger.info("Index saved to %s", settings.index_dir)

    # Write build manifest
    _write_manifest(settings, len(records), len(chunks))
    logger.info("Done.")


def _write_manifest(settings, n_docs: int, n_chunks: int):
    """Write build_manifest.json alongside the index."""
    import datetime

    manifest = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "n_documents": n_docs,
        "n_chunks": n_chunks,
    }
    try:
        import subprocess as _sp

        sha = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(settings.project_root),
            stderr=_sp.DEVNULL,
        ).decode().strip()
        manifest["git_commit"] = sha
    except Exception:
        pass

    path = settings.index_dir / "build_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _validate_metadata_cmd(args):
    """CLI handler for validate-metadata sub-command."""
    from childlanguagenet.ingestion.metadata_registry import validate_metadata
    from childlanguagenet.telemetry.logging import get_logger

    settings = get_settings()
    logger = get_logger(__name__)
    meta_path = Path(args.metadata) if args.metadata else settings.metadata_file

    logger.info("Validating %s …", meta_path)
    records = validate_metadata(meta_path)
    logger.info("✓ %d valid records", len(records))

    # Write validation artifact
    out_path = Path(args.out) if args.out else settings.artifacts_dir / "metadata_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"valid": True, "n_records": len(records)}, indent=2))
    logger.info("Wrote validation report to %s", out_path)


def _ingest_cmd(args):
    """CLI handler for ingest sub-command."""
    from childlanguagenet.ingestion.metadata_registry import validate_metadata
    from childlanguagenet.ingestion.loaders import load_all_papers
    from childlanguagenet.ingestion.chunking import chunk_documents
    from childlanguagenet.telemetry.logging import get_logger

    settings = get_settings()
    logger = get_logger(__name__)
    meta_path = Path(args.metadata) if args.metadata else settings.metadata_file

    records = validate_metadata(meta_path)
    docs = load_all_papers(records, data_dir=settings.data_dir)

    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    out_dir = Path(args.out) if args.out else settings.artifacts_dir / "ingested_texts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save chunks as JSONL
    out_file = out_dir / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ch in chunks:
            record = {
                "chunk_id": ch.metadata.get("chunk_id", ""),
                "paper_id": ch.metadata.get("paper_id", ""),
                "title": ch.metadata.get("title", ""),
                "text": ch.page_content,
                "metadata": {k: v for k, v in ch.metadata.items()
                             if k not in ("chunk_id", "paper_id", "title")},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d chunks to %s", len(chunks), out_file)


def _build_index_cmd(args):
    """CLI handler for build-index sub-command."""
    from childlanguagenet.ingestion.metadata_registry import validate_metadata
    from childlanguagenet.ingestion.loaders import load_all_papers
    from childlanguagenet.ingestion.chunking import chunk_documents
    from childlanguagenet.embeddings.embedder import get_embeddings
    from childlanguagenet.vectorstore.faiss_store import FAISSStore
    from childlanguagenet.telemetry.logging import get_logger

    settings = get_settings()
    logger = get_logger(__name__)

    # If --inputs provided, read chunks from JSONL; else build from metadata
    if args.inputs:
        from langchain_core.documents import Document

        chunks_file = Path(args.inputs) / "chunks.jsonl"
        if not chunks_file.exists():
            logger.error("chunks.jsonl not found in %s", args.inputs)
            sys.exit(1)

        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                meta = obj.get("metadata", {})
                meta["chunk_id"] = obj.get("chunk_id", "")
                meta["paper_id"] = obj.get("paper_id", "")
                meta["title"] = obj.get("title", "")
                chunks.append(Document(page_content=obj["text"], metadata=meta))
        logger.info("Loaded %d chunks from %s", len(chunks), chunks_file)
    else:
        records = validate_metadata(settings.metadata_file)
        docs = load_all_papers(records, data_dir=settings.data_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    out_dir = Path(args.out) if args.out else settings.index_dir
    embeddings = get_embeddings(settings)
    store = FAISSStore(index_dir=out_dir, embedding=embeddings, k_default=settings.default_top_k)
    store.build(chunks)
    store.save()
    _write_manifest(settings, n_docs=0, n_chunks=len(chunks))
    logger.info("Index saved to %s (%d chunks)", out_dir, len(chunks))


# ── entry points ───────────────────────────────────────────────────────────

def main():
    """Top-level CLI entry point (childrag)."""
    parser = argparse.ArgumentParser(
        prog="childrag",
        description="ChildLanguageNet RAG — CLI tools",
    )
    sub = parser.add_subparsers(dest="command")

    # validate-metadata
    p_val = sub.add_parser("validate-metadata", help="Validate metadata.json")
    p_val.add_argument("--metadata", default=None, help="Path to metadata.json")
    p_val.add_argument("--out", default=None, help="Output path for validation report")

    # ingest
    p_ing = sub.add_parser("ingest", help="Ingest documents → chunked JSONL")
    p_ing.add_argument("--metadata", default=None, help="Path to metadata.json")
    p_ing.add_argument("--out", default=None, help="Output directory for ingested texts")

    # build-index
    p_idx = sub.add_parser("build-index", help="Build FAISS index")
    p_idx.add_argument("--inputs", default=None, help="Path to ingested_texts/ directory")
    p_idx.add_argument("--out", default=None, help="Output directory for FAISS index")

    # serve
    sub.add_parser("serve", help="Run Streamlit app")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "validate-metadata":
        _validate_metadata_cmd(args)
    elif args.command == "ingest":
        _ingest_cmd(args)
    elif args.command == "build-index":
        _build_index_cmd(args)
    elif args.command == "serve":
        serve_streamlit()
    else:
        parser.print_help()


def build_index():
    """Entry point for childrag-index."""
    _build_index()


def serve_streamlit():
    """Entry point for childrag-serve."""
    settings = get_settings()
    app_path = settings.project_root / "apps" / "streamlit_app.py"
    if not app_path.exists():
        print(f"ERROR: Streamlit app not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
