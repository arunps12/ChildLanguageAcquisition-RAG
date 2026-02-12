"""Citation model and formatting helpers."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    """Paper-level citation derived from chunk metadata."""

    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    journal_or_venue: Optional[str] = None
    doi: Optional[str] = None
    source_url: Optional[str] = None

    def format_apa_like(self) -> str:
        """Return a short APA-like citation string."""
        parts: List[str] = []
        if self.authors:
            parts.append(", ".join(self.authors))
        if self.year:
            parts.append(f"({self.year})")
        if self.title:
            parts.append(self.title)
        if self.journal_or_venue:
            parts.append(f"*{self.journal_or_venue}*")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        return ". ".join(parts) if parts else self.paper_id


def format_sources_section(citations: List[Citation]) -> str:
    """Format a Markdown-style 'Sources' section from a list of citations."""
    if not citations:
        return ""
    lines = ["**Sources:**", ""]
    for i, c in enumerate(citations, 1):
        lines.append(f"{i}. {c.format_apa_like()}")
    return "\n".join(lines)
