"""Metadata registry — schema validation for data/metadata.json.

Validates structure, detects duplicate IDs, verifies PDF existence, checks URL schemes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PaperRecord:
    """Validated paper-level metadata record."""

    id: str
    title: str
    source_type: str  # "pdf" | "url"
    path_or_url: str
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    license: Optional[str] = None
    # extra fields from the existing schema
    journal_or_venue: Optional[str] = None
    doi: Optional[str] = None
    publisher: Optional[str] = None
    paper_type: Optional[str] = None
    open_access: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return metadata as a plain dict (for embedding in Document.metadata)."""
        return {
            "paper_id": self.id,
            "title": self.title,
            "authors": self.authors or [],
            "year": self.year,
            "source_type": self.source_type,
            "path_or_url": self.path_or_url,
            "journal_or_venue": self.journal_or_venue,
            "doi": self.doi,
            "publisher": self.publisher,
            "paper_type": self.paper_type,
            "open_access": self.open_access,
            "tags": self.tags or [],
        }


_REQUIRED_FIELDS = {"paper_id", "title"}


def validate_metadata(
    metadata_path: Path,
    data_dir: Optional[Path] = None,
) -> List[PaperRecord]:
    """Validate ``metadata.json`` and return a list of :class:`PaperRecord`.

    Validation rules
    ----------------
    * Fail fast on missing required fields (``paper_id``, ``title``).
    * Detect duplicate IDs.
    * For PDFs: verify the file exists (relative to *data_dir*).
    * For URLs: verify the scheme is ``http`` or ``https``.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(f"metadata.json must be a JSON array, got {type(raw).__name__}")

    if data_dir is None:
        data_dir = metadata_path.parent  # assume metadata sits inside data/

    seen_ids: set[str] = set()
    records: List[PaperRecord] = []
    errors: List[str] = []

    for idx, item in enumerate(raw):
        # --- required fields ---
        for key in ("paper_id", "title"):
            if key not in item:
                errors.append(f"Entry {idx}: missing required field '{key}'")

        pid = item.get("paper_id", f"__missing_{idx}")

        # --- duplicate IDs ---
        if pid in seen_ids:
            errors.append(f"Entry {idx}: duplicate paper_id '{pid}'")
        seen_ids.add(pid)

        # --- determine source_type + path_or_url ---
        pdf_file = item.get("pdf_file")
        source_url = item.get("source_url")

        if pdf_file:
            source_type = "pdf"
            path_or_url = pdf_file
            # verify file exists
            full_path = data_dir / pdf_file
            if not full_path.exists():
                errors.append(
                    f"Entry {idx} ({pid}): PDF file not found: {full_path}"
                )
        elif source_url:
            source_type = "url"
            path_or_url = source_url
            if not (source_url.startswith("http://") or source_url.startswith("https://")):
                errors.append(
                    f"Entry {idx} ({pid}): URL must start with http:// or https://, got: {source_url}"
                )
        else:
            source_type = "pdf"
            path_or_url = ""
            errors.append(
                f"Entry {idx} ({pid}): must have either 'pdf_file' or 'source_url'"
            )

        # --- build record even if there are non-fatal errors ---
        authors = item.get("authors")
        if isinstance(authors, str):
            authors = [authors]

        year = item.get("year")
        if year is not None:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        records.append(
            PaperRecord(
                id=pid,
                title=item.get("title", ""),
                source_type=source_type,
                path_or_url=path_or_url,
                authors=authors,
                year=year,
                tags=item.get("tags"),
                notes=item.get("notes"),
                license=item.get("license"),
                journal_or_venue=item.get("journal_or_venue"),
                doi=item.get("doi"),
                publisher=item.get("publisher"),
                paper_type=item.get("paper_type"),
                open_access=item.get("open_access"),
            )
        )

    if errors:
        msg = "Metadata validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ValueError(msg)

    return records
