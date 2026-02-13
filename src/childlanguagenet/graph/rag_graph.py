"""LangGraph RAG pipeline — retrieve → generate with citation-aware output."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from childlanguagenet.citations.cite import Citation, format_sources_section

# ── State ──────────────────────────────────────────────────────────────────


class RAGState(BaseModel):
    """State flowing through the RAG graph."""

    user_query: str = ""
    retrieved_chunks: List[Document] = Field(default_factory=list)
    answer_text: str = ""
    citations: List[Citation] = Field(default_factory=list)
    debug_info: Dict[str, Any] = Field(default_factory=dict)

    # retrieval controls
    k: int = 8


# ── Nodes ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a research assistant specialising in child language acquisition.\n"
    "Answer the user's question using ONLY the evidence provided below.\n"
    "Cite sources by their paper_id (e.g. [vandam_2016]).\n"
    "If the evidence is insufficient, say so clearly.\n"
    "Do not invent facts or citations. Keep answers concise and research-oriented."
)


class RAGNodes:
    """Retrieval + generation nodes for the graph."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    # -- retrieval -------------------------------------------------------

    def retrieve(self, state: RAGState) -> RAGState:
        docs: List[Document] = self.retriever.invoke(state.user_query)
        return RAGState(
            user_query=state.user_query,
            k=state.k,
            retrieved_chunks=docs,
        )

    # -- generation (context-stuffing) -----------------------------------

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        """Build a numbered evidence block from retrieved chunks."""
        parts: list[str] = []
        for i, d in enumerate(docs[:12], 1):
            meta = d.metadata or {}
            pid = meta.get("paper_id", f"doc_{i}")
            title = meta.get("title", "Unknown")
            year = meta.get("year", "n.d.")
            header = f"[{i}] {pid} | {title} ({year})"
            doi = meta.get("doi")
            if doi:
                header += f" | DOI: {doi}"
            parts.append(f"{header}\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    def generate(self, state: RAGState) -> RAGState:
        """Generate answer from pre-retrieved chunks, then extract citations."""
        context = self._format_context(state.retrieved_chunks)

        user_msg = (
            f"## Evidence\n\n{context}\n\n---\n\n"
            f"## Question\n\n{state.user_query}\n\n"
            "Provide a detailed answer based on the evidence above, "
            "citing sources by their paper_id."
        )

        response = self.llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        answer = response.content if hasattr(response, "content") else str(response)

        # Extract deduplicated citations from retrieved chunks
        citations: List[Citation] = []
        seen: set[str] = set()
        for d in state.retrieved_chunks:
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
                    source_url=d.metadata.get("path_or_url")
                    or d.metadata.get("source_url"),
                )
            )

        # Append sources section to answer
        sources_text = format_sources_section(citations)
        full_answer = f"{answer}\n\n{sources_text}" if sources_text else answer

        return RAGState(
            user_query=state.user_query,
            k=state.k,
            retrieved_chunks=state.retrieved_chunks,
            answer_text=full_answer,
            citations=citations,
        )


# ── Graph builder ──────────────────────────────────────────────────────────


def build_rag_graph(retriever, llm):
    """Build and compile the LangGraph RAG workflow.

    Returns a compiled graph that accepts and returns :class:`RAGState`.
    """
    nodes = RAGNodes(retriever=retriever, llm=llm)

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("generate", nodes.generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


def run_query(graph, question: str, k: int = 8) -> Dict[str, Any]:
    """Convenience wrapper to run a query through the compiled graph."""
    initial = RAGState(user_query=question, k=k)
    return graph.invoke(initial)
