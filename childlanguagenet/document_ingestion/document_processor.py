"""
Document processing module for metadata-first loading and splitting documents.

Reads paper registry from data/metadata.json, loads each paper (PDF or URL),
attaches paper-level metadata to each Document, then splits into chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader


@dataclass(frozen=True)
class PaperRecord:
    """Paper-level metadata as defined in metadata.json."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    journal_or_venue: Optional[str] = None
    doi: Optional[str] = None
    publisher: Optional[str] = None
    paper_type: Optional[str] = None
    open_access: Optional[bool] = None
    source_url: Optional[str] = None
    pdf_file: Optional[str] = None  # path relative to data_dir, e.g. "pdf/foo.pdf"


class DocumentProcessor:
    """Handles metadata-first document loading and processing."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        metadata_filename: str = "metadata.json",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / metadata_filename

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # -----------------------
    # Metadata registry
    # -----------------------
    def load_metadata_registry(self) -> List[PaperRecord]:
        """Load paper registry from data/metadata.json."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at: {self.metadata_path}. "
                "Expected it under the data folder."
            )

        with self.metadata_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(
                f"metadata.json must be a JSON array (list of paper objects), got: {type(raw)}"
            )

        records: List[PaperRecord] = []
        for item in raw:
            # Minimal validation / required fields
            for k in ("paper_id", "title", "authors", "year"):
                if k not in item:
                    raise ValueError(f"Missing required field '{k}' in metadata.json entry: {item}")

            records.append(
                PaperRecord(
                    paper_id=item["paper_id"],
                    title=item["title"],
                    authors=item["authors"],
                    year=int(item["year"]),
                    journal_or_venue=item.get("journal_or_venue"),
                    doi=item.get("doi"),
                    publisher=item.get("publisher"),
                    paper_type=item.get("paper_type"),
                    open_access=item.get("open_access"),
                    source_url=item.get("source_url"),
                    pdf_file=item.get("pdf_file"),
                )
            )
        return records

    # -----------------------
    # Loaders (PDF / URL)
    # -----------------------
    def _paper_metadata_dict(self, rec: PaperRecord) -> Dict[str, Any]:
        """Convert PaperRecord to metadata dict stored on every Document."""
        return {
            "paper_id": rec.paper_id,
            "title": rec.title,
            "authors": rec.authors,
            "year": rec.year,
            "journal_or_venue": rec.journal_or_venue,
            "doi": rec.doi,
            "publisher": rec.publisher,
            "paper_type": rec.paper_type,
            "open_access": rec.open_access,
            "source_url": rec.source_url,
            "pdf_file": rec.pdf_file,
        }

    def load_paper(self, rec: PaperRecord) -> List[Document]:
        """
        Load a single paper using:
        1) local PDF path (data_dir / pdf_file) if available
        2) else source_url (web loader)
        Returns Documents with paper metadata attached.
        """
        base_meta = self._paper_metadata_dict(rec)

        # Prefer local PDF if present
        if rec.pdf_file:
            pdf_path = self.data_dir / rec.pdf_file
            if pdf_path.exists():
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()

                # attach metadata
                for d in docs:
                    d.metadata.update(base_meta)
                    d.metadata["source_kind"] = "pdf"
                    d.metadata["source_path"] = str(pdf_path)
                return docs

        # Fallback to URL if provided
        if rec.source_url:
            loader = WebBaseLoader(rec.source_url)
            docs = loader.load()

            for d in docs:
                d.metadata.update(base_meta)
                d.metadata["source_kind"] = "url"
                d.metadata["source_path"] = rec.source_url
            return docs

        raise ValueError(
            f"Paper '{rec.paper_id}' has neither an existing local PDF nor a source_url. "
            f"pdf_file={rec.pdf_file}, source_url={rec.source_url}"
        )

    def load_all_papers(self, records: List[PaperRecord]) -> List[Document]:
        """Load all papers from the registry and attach metadata."""
        all_docs: List[Document] = []
        for rec in records:
            try:
                all_docs.extend(self.load_paper(rec))
            except Exception as e:
                raise RuntimeError(f"Failed to load paper_id={rec.paper_id}: {e}") from e
        return all_docs

    # -----------------------
    # Splitting
    # -----------------------
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while keeping metadata."""
        chunks = self.splitter.split_documents(documents)

        # Add chunk-level fields to metadata
        paper_chunk_counts: Dict[str, int] = {}
        for ch in chunks:
            pid = ch.metadata.get("paper_id", "unknown_paper")
            idx = paper_chunk_counts.get(pid, 0)
            paper_chunk_counts[pid] = idx + 1

            ch.metadata["chunk_index"] = idx
            ch.metadata["chunk_id"] = f"{pid}::chunk_{idx:05d}"
        return chunks

    # -----------------------
    # pipeline method
    # -----------------------
    def process_from_metadata(self) -> List[Document]:
        """
        Full pipeline:
        metadata.json -> load per paper -> split into chunks
        """
        records = self.load_metadata_registry()
        docs = self.load_all_papers(records)
        return self.split_documents(docs)
