"""LangGraph nodes for RAG workflow (child language acquisition research)."""

from typing import List

#from langchain.schema import Document
from langchain_core.documents import Document

from childlanguagenet.state.rag_state import RAGState, Citation


class RAGNodes:
    """Contains node functions for the RAG workflow."""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes.

        Args:
            retriever: Retriever instance from VectorStore
            llm: Language model instance (e.g., from Config.get_llm())
        """
        self.retriever = retriever
        self.llm = llm

    # -----------------------
    # Retrieval node
    # -----------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents.

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved documents
        """
        docs: List[Document] = self.retriever.invoke(state.question)

        # Collect unique paper_ids for transparency/debugging
        paper_ids = []
        for d in docs:
            pid = d.metadata.get("paper_id")
            if pid and pid not in paper_ids:
                paper_ids.append(pid)

        return RAGState(
            question=state.question,
            k=state.k,
            retrieved_docs=docs,
            retrieved_paper_ids=paper_ids,
        )

    # -----------------------
    # Generation node
    # -----------------------
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate an answer grounded in retrieved documents.

        Args:
            state: Current RAG state with retrieved documents

        Returns:
            Updated RAG state with generated answer and citations
        """
        if not state.retrieved_docs:
            return RAGState(
                question=state.question,
                k=state.k,
                retrieved_docs=[],
                answer="I could not find relevant documents to answer this question.",
            )

        # ---- Build context ----
        context_blocks = []
        for d in state.retrieved_docs:
            pid = d.metadata.get("paper_id", "unknown")
            context_blocks.append(f"[{pid}]\n{d.page_content}")

        context = "\n\n".join(context_blocks)

        # ---- Prompt ----
        prompt = f"""
You are a research assistant for child language acquisition.

Answer the question strictly using the provided context.
If the answer is not supported by the context, say so clearly.
Do not hallucinate citations.

Context:
{context}

Question:
{state.question}

Answer:
""".strip()

        response = self.llm.invoke(prompt)

        # ---- Extract citations (paper-level, deduplicated) ----
        citations = []
        seen = set()

        for d in state.retrieved_docs:
            pid = d.metadata.get("paper_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            citations.append(
                Citation(
                    paper_id=pid,
                    title=d.metadata.get("title"),
                    authors=d.metadata.get("authors"),
                    year=d.metadata.get("year"),
                    journal_or_venue=d.metadata.get("journal_or_venue"),
                    doi=d.metadata.get("doi"),
                    source_url=d.metadata.get("source_url"),
                )
            )

        return RAGState(
            question=state.question,
            k=state.k,
            retrieved_docs=state.retrieved_docs,
            retrieved_paper_ids=state.retrieved_paper_ids,
            citations=citations,
            answer=response.content,
        )
