"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_answer (childlanguagenet)."""

from __future__ import annotations

from typing import List, Optional

from langchain.schema import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from childlanguagenet.state.rag_state import RAGState, Citation


class RAGNodes:
    """Contains node functions for RAG workflow (ReAct-style agent for answering)."""

    def __init__(self, retriever, llm):
        """
        Args:
            retriever: Retriever instance (e.g., FAISS retriever)
            llm: Chat LLM instance
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy init

    # -----------------------
    # Retriever node 
    # -----------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant chunks for the question."""
        docs: List[Document] = self.retriever.invoke(state.question)

        # Track unique paper IDs for transparency
        paper_ids: List[str] = []
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
    # Tools
    # -----------------------
    def _build_tools(self) -> List[Tool]:
        """Build tools (retriever tool only; no external web tools for research reproducibility)."""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found in the local research corpus."

            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {}) or {}
                pid = meta.get("paper_id", f"doc_{i}")
                title = meta.get("title", "Unknown title")
                year = meta.get("year", "n.d.")
                doi = meta.get("doi")

                header = f"[{i}] {pid} | {title} ({year})"
                if doi:
                    header += f" | DOI: {doi}"

                merged.append(f"{header}\n{d.page_content}")

            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description=(
                "Fetch passages from the local child language acquisition research corpus. "
                "Use this tool to ground answers and find evidence."
            ),
            func=retriever_tool_fn,
        )

        return [retriever_tool]

    def _build_agent(self) -> None:
        """Build a ReAct agent with the retriever tool."""
        tools = self._build_tools()

        system_prompt = (
            "You are a research assistant for child language acquisition.\n"
            "You have access to a 'retriever' tool that searches a local corpus of papers.\n"
            "Rules:\n"
            "1) Use the retriever tool to gather evidence before answering.\n"
            "2) If evidence is insufficient, say so clearly.\n"
            "3) Do not invent claims or citations.\n"
            "4) Keep answers concise and research-oriented."
        )

        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    # -----------------------
    # Answer node (ReAct agent)
    # -----------------------
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using a ReAct agent (grounded by the local retriever tool)."""
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        messages = result.get("messages", [])

        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        # Deduplicate paper-level citations from retrieved_docs already in state
        citations: List[Citation] = []
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
            answer=answer or "Could not generate an answer from the available evidence.",
        )
