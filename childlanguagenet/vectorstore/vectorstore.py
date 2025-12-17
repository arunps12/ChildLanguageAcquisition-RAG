"""Vector store module for document embedding and retrieval (metadata-aware)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

#from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from childlanguagenet.config.config import Config


class VectorStore:
    """Manages vector store operations for RAG (FAISS + embeddings)."""

    def __init__(
        self,
        index_dir: Optional[Union[str, Path]] = None,
        k_default: Optional[int] = None,
    ):
        """
        Args:
            index_dir: Folder where the FAISS index will be saved/loaded.
                       Defaults to Config.INDEX_DIR.
            k_default: Default number of chunks to retrieve.
                       Defaults to Config.DEFAULT_TOP_K.
        """
        self.index_dir = Path(index_dir) if index_dir is not None else Path(Config.INDEX_DIR)
        self.k_default = k_default if k_default is not None else int(Config.DEFAULT_TOP_K)

        
        self.embedding = OpenAIEmbeddings()

        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

    # -----------------------
    # Build / Save / Load
    # -----------------------
    def create_vectorstore(self, documents: List[Document], save: bool = True) -> None:
        """
        Create vector store from chunked documents.

        Notes:
        - Expects documents already have metadata (paper_id, title, year, doi, etc.)
        - FAISS preserves Document.metadata for retrieval/citations.

        Args:
            documents: Chunked documents from DocumentProcessor.process_from_metadata()
            save: Whether to save the index to disk
        """
        if not documents:
            raise ValueError("No documents provided to create_vectorstore().")

        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_default})

        if save:
            self.save()

    def save(self) -> None:
        """Save FAISS index to disk."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))

    def load(self) -> None:
        """Load FAISS index from disk."""
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_dir}'. "
                "Build it first with create_vectorstore()."
            )

        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embedding,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_default})

    # -----------------------
    # Retrieval
    # -----------------------
    def get_retriever(self, k: Optional[int] = None):
        """Get retriever instance (optionally override k)."""
        if self.vectorstore is None or self.retriever is None:
            raise ValueError("Vector store not initialized. Call load() or create_vectorstore() first.")

        if k is not None:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})

        return self.retriever

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            k: Number of chunks to retrieve (defaults to self.k_default)

        Returns:
            List of Documents (chunks) with metadata preserved.
        """
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)
