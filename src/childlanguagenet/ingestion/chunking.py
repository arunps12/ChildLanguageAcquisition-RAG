"""Deterministic document chunking with stable chunk IDs and metadata propagation."""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split *documents* into chunks with stable IDs and full metadata.

    Each chunk receives:
    * ``chunk_index`` — zero-based index within its parent document
    * ``chunk_id`` — ``{paper_id}::chunk_{index:05d}``
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    paper_chunk_counts: Dict[str, int] = {}
    for ch in chunks:
        pid = ch.metadata.get("paper_id", "unknown_paper")
        idx = paper_chunk_counts.get(pid, 0)
        paper_chunk_counts[pid] = idx + 1
        ch.metadata["chunk_index"] = idx
        ch.metadata["chunk_id"] = f"{pid}::chunk_{idx:05d}"

    return chunks
