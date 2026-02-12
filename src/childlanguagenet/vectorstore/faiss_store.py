"""FAISS vector store — build, save, load, retrieve."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class FAISSStore:
    """Manages FAISS index lifecycle: build → save → load → retrieve."""

    def __init__(
        self,
        index_dir: Union[str, Path],
        embedding,
        k_default: int = 8,
    ):
        self.index_dir = Path(index_dir)
        self.embedding = embedding
        self.k_default = k_default
        self._vectorstore: Optional[FAISS] = None
        self._retriever = None

    # ── build ──────────────────────────────────────────────────────────────

    def build(self, documents: List[Document]) -> None:
        """Create a FAISS vectorstore from chunked documents."""
        if not documents:
            raise ValueError("No documents provided to build the index.")
        self._vectorstore = FAISS.from_documents(documents, self.embedding)
        self._retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.k_default},
        )

    # ── persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist FAISS index + docstore to disk."""
        if self._vectorstore is None:
            raise ValueError("No vectorstore to save. Call build() first.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_dir))

    def load(self) -> None:
        """Load FAISS index from disk."""
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_dir}'. "
                "Build it first with childrag-index."
            )
        self._vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embedding,
            allow_dangerous_deserialization=True,
        )
        self._retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.k_default},
        )

    # ── retrieval ──────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None

    def get_retriever(self, k: Optional[int] = None):
        """Return a LangChain retriever."""
        if self._vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load() or build() first.")
        if k is not None:
            return self._vectorstore.as_retriever(search_kwargs={"k": k})
        return self._retriever

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve top-k chunks for *query*."""
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)
