"""RAG state definition for LangGraph (metadata-first, paper-level citations)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.schema import Document

from childlanguagenet.config.config import Config
class Citation(BaseModel):
    """Paper-level citation info derived from retrieved chunk metadata."""
    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    journal_or_venue: Optional[str] = None
    doi: Optional[str] = None
    source_url: Optional[str] = None


class RAGState(BaseModel):
    """State object for RAG workflow."""

    # input
    question: str

    # retrieval controls
    k: int = Config.DEFAULT_TOP_K

    # retrieval results
    retrieved_docs: List[Document] = Field(default_factory=list)

    # store unique paper ids used by retrieved_docs
    retrieved_paper_ids: List[str] = Field(default_factory=list)

    # paper-level citations extracted from retrieved_docs
    citations: List[Citation] = Field(default_factory=list)

    # generation output
    answer: str = ""

    # debugging / transparency
    debug: Dict[str, Any] = Field(default_factory=dict)
