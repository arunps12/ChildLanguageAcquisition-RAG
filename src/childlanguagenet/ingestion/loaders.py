"""Document loaders — PDF and URL with metadata propagation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

from childlanguagenet.ingestion.metadata_registry import PaperRecord


def _normalize_text(text: str) -> str:
    """Strip repeated whitespace / page headers."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_paper(
    rec: PaperRecord,
    data_dir: Union[str, Path],
) -> List[Document]:
    """Load a single paper (PDF or URL) and attach paper-level metadata.

    Parameters
    ----------
    rec : PaperRecord
        Validated metadata record.
    data_dir : Path
        Root data directory (used to resolve ``pdf_file`` paths).
    """
    base_meta = rec.to_dict()
    data_dir = Path(data_dir)

    if rec.source_type == "pdf" and rec.path_or_url:
        pdf_path = data_dir / rec.path_or_url
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found for '{rec.id}': {pdf_path}"
            )
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for d in docs:
            d.page_content = _normalize_text(d.page_content)
            d.metadata.update(base_meta)
            d.metadata["source_kind"] = "pdf"
            d.metadata["source_path"] = str(pdf_path)
        return docs

    if rec.source_type == "url" and rec.path_or_url:
        loader = WebBaseLoader(rec.path_or_url)
        docs = loader.load()
        for d in docs:
            d.page_content = _normalize_text(d.page_content)
            d.metadata.update(base_meta)
            d.metadata["source_kind"] = "url"
            d.metadata["source_path"] = rec.path_or_url
        return docs

    raise ValueError(
        f"Paper '{rec.id}': cannot load — source_type={rec.source_type}, "
        f"path_or_url={rec.path_or_url}"
    )


def load_all_papers(
    records: List[PaperRecord],
    data_dir: Union[str, Path],
) -> List[Document]:
    """Load all papers from validated records, propagating metadata."""
    all_docs: List[Document] = []
    for rec in records:
        try:
            all_docs.extend(load_paper(rec, data_dir=data_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load paper_id={rec.id}: {exc}"
            ) from exc
    return all_docs
