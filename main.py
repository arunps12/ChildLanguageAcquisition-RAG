"""
Main application entry point for ChildLanguageNet (metadata-first PDF/URL RAG).

- Loads paper registry from data/metadata.json
- Builds/loads FAISS index (Config.INDEX_DIR)
- Runs a simple LangGraph workflow: retrieve -> generate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from childlanguagenet.config.config import Config
from childlanguagenet.document_ingestion.document_processor import DocumentProcessor
from childlanguagenet.vectorstore.vectorstore import VectorStore
from childlanguagenet.graph_builder.graph_builder import GraphBuilder


class ChildLanguageNetRAG:
    """Main RAG application wrapper."""

    def __init__(self, rebuild_index: bool = False):
        """
        Args:
            rebuild_index: If True, rebuild FAISS index from metadata registry
                           even if an index already exists on disk.
        """
        print("Initializing ChildLanguageNet RAG...")

        self.llm = Config.get_llm()

        # Document processing (metadata.json -> load PDF/URL -> split)
        self.doc_processor = DocumentProcessor(
            data_dir=Config.DATA_DIR,
            metadata_path=Config.METADATA_FILE,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        # Vector store (FAISS)
        self.vector_store = VectorStore(index_dir=Config.INDEX_DIR, k_default=Config.DEFAULT_TOP_K)

        # Setup vector store (load or build)
        self._setup_vectorstore(rebuild_index=rebuild_index)

        # Graph (LangGraph)
        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm,
        )
        self.graph_builder.build()

        print("System initialized successfully!\n")

    def _setup_vectorstore(self, rebuild_index: bool = False) -> None:
        """Load FAISS index if present, otherwise build from metadata registry."""
        index_dir = Path(Config.INDEX_DIR)

        if (not rebuild_index) and index_dir.exists():
            try:
                print(f"Loading existing FAISS index from: {index_dir}")
                self.vector_store.load()
                print("Index loaded.\n")
                return
            except Exception as e:
                print(f"Failed to load index ({e}). Rebuilding...\n")

        print("Building documents from metadata registry...")
        chunks = self.doc_processor.process_from_metadata()
        print(f"Created {len(chunks)} document chunks")
        print("Creating FAISS vector store...")
        self.vector_store.create_vectorstore(chunks, save=True)
        print(f"Index saved to: {index_dir}\n")

    def ask(self, question: str) -> str:
        """Ask a question and return the answer (prints answer + citations if present)."""
        question = question.strip()
        if not question:
            return ""

        print(f"Question: {question}\n")
        print("Processing...\n")

        result = self.graph_builder.run(question)

        answer = result.get("answer", "")
        citations = result.get("citations", None)

        print(f"Answer:\n{answer}\n")

        # Print citations if available
        if citations:
            print("Citations:")
            if isinstance(citations, list):
                for c in citations:
                    print(f" - {c}")
            else:
                print(citations)
            print()

        return answer

    def interactive_mode(self) -> None:
        """Run an interactive CLI loop."""
        print("Interactive Mode (type 'quit' to exit)\n")
        while True:
            q = input("Enter your question: ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            self.ask(q)
            print("-" * 90)


def main(rebuild_index: bool = False) -> None:
    """Main entry point."""
    rag = ChildLanguageNetRAG(rebuild_index=rebuild_index)

    # Example questions (edit/remove)
    example_questions = [
        "Summarize the main themes across the papers in the registry.",
        "Which papers discuss infant-directed speech (IDS) and what are their key findings?",
        "What methods are used for measuring early language development in these studies?",
    ]

    print("=" * 90)
    print("Running example questions:")
    print("=" * 90 + "\n")

    for q in example_questions:
        rag.ask(q)
        print("=" * 90 + "\n")

    user_input = input("Would you like to enter interactive mode? (y/n): ").strip().lower()
    if user_input == "y":
        rag.interactive_mode()


if __name__ == "__main__":
    
    main(rebuild_index=True)
