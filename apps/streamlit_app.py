"""Streamlit UI for ChildLanguageNet RAG.

Run with:
    childrag-serve
    streamlit run apps/streamlit_app.py
"""

from __future__ import annotations

import time

import streamlit as st

from childlanguagenet.config.settings import get_settings
from childlanguagenet.embeddings.embedder import get_embeddings
from childlanguagenet.graph.rag_graph import build_rag_graph, run_query
from childlanguagenet.telemetry.logging import get_logger
from childlanguagenet.telemetry.metrics import get_metrics
from childlanguagenet.vectorstore.faiss_store import FAISSStore

logger = get_logger(__name__)
metrics = get_metrics()

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Child Language Research RAG", layout="centered")
st.markdown(
    "<style>.stButton > button { width: 100%; font-weight: 700; }</style>",
    unsafe_allow_html=True,
)


# ── Cached resources ──────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading FAISS index …")
def _load_store():
    """Load or build the FAISS store (cached across reruns)."""
    settings = get_settings()
    embeddings = get_embeddings(settings)
    store = FAISSStore(
        index_dir=settings.index_dir,
        embedding=embeddings,
        k_default=settings.default_top_k,
    )
    built_chunks = 0
    try:
        store.load()
        logger.info("Loaded existing FAISS index from %s", settings.index_dir)
    except FileNotFoundError:
        logger.warning("No index found — building from metadata …")
        from childlanguagenet.ingestion.chunking import chunk_documents
        from childlanguagenet.ingestion.loaders import load_all_papers
        from childlanguagenet.ingestion.metadata_registry import validate_metadata

        records = validate_metadata(settings.metadata_file, data_dir=settings.data_dir)
        docs = load_all_papers(records, data_dir=settings.data_dir)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        built_chunks = len(chunks)
        store.build(chunks)
        store.save()
        logger.info("Built and saved index (%d chunks)", built_chunks)
    return store, built_chunks


@st.cache_resource(show_spinner="Initializing LLM …")
def _get_llm():
    settings = get_settings()

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )

    # OpenAI provider
    settings.require_openai_key()
    import os

    os.environ["OPENAI_API_KEY"] = settings.openai_api_key  # type: ignore[arg-type]
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def _rebuild_index():
    """Force-rebuild the FAISS index (clears cache)."""
    _load_store.clear()
    st.rerun()


# ── Render helpers ─────────────────────────────────────────────────────────


def _render_citations(result: dict):
    citations = result.get("citations") or []
    if not citations:
        return
    st.subheader("Sources")
    for c in citations:
        if isinstance(c, dict):
            pid, title, year, doi, url = (
                c.get("paper_id"),
                c.get("title"),
                c.get("year"),
                c.get("doi"),
                c.get("source_url"),
            )
        else:
            pid, title, year, doi, url = (
                getattr(c, "paper_id", None),
                getattr(c, "title", None),
                getattr(c, "year", None),
                getattr(c, "doi", None),
                getattr(c, "source_url", None),
            )
        line = f"- {title or pid or 'Unknown'}"
        if year:
            line += f" ({year})"
        if doi:
            line += f" | DOI: {doi}"
        st.markdown(line)
        if url:
            st.caption(url)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    settings = get_settings()

    # Session defaults
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title(settings.ui_title)
    st.write("Question answering over a curated child-language research corpus.")

    # ── Load index ──────────────────────────────────────────────────────
    try:
        store, built_chunks = _load_store()
    except Exception as exc:
        st.error(f"Failed to load/build index: {exc}")
        logger.exception("Index load error")
        return

    if built_chunks > 0:
        st.success(f"Built index from {built_chunks} chunks.")
    else:
        st.success("Index loaded from disk.")

    # ── Build / Refresh button ──────────────────────────────────────────
    if st.button("🔄 Build / Refresh Index"):
        _rebuild_index()

    st.divider()

    # ── QA form ─────────────────────────────────────────────────────────
    with st.form("qa_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What mechanisms explain why infant-directed speech supports learning?",
        )
        submit = st.form_submit_button("Ask")

    if submit and question:
        try:
            llm = _get_llm()
        except Exception as exc:
            st.error(f"LLM initialization failed: {exc}")
            return

        graph = build_rag_graph(retriever=store.get_retriever(), llm=llm)

        with st.spinner("Retrieving and generating …"):
            start = time.time()
            metrics.inc("queries_total")
            with metrics.timer("generation_latency_seconds"):
                result = run_query(graph, question, k=settings.default_top_k)
            elapsed = time.time() - start

        answer = result.get("answer_text", "No answer returned.")
        st.session_state.history.append(
            {"question": question, "answer": answer, "time": elapsed}
        )
        st.session_state.history = st.session_state.history[-10:]

        st.subheader("Answer")
        st.write(answer)

        _render_citations(result)

        with st.expander("Retrieved chunks (debug)"):
            docs = result.get("retrieved_chunks") or []
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata or {}
                pid = meta.get("paper_id", "unknown")
                title = meta.get("title", "Unknown")
                st.markdown(f"**Chunk {i}:** {pid} | {title}")
                txt = doc.page_content
                if len(txt) > 1500:
                    txt = txt[:1500] + "…"
                st.text_area(f"chunk_{i}", txt, height=160)

        st.caption(f"Response time: {elapsed:.2f}s")

    # ── Recent history ──────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        st.subheader("Recent questions")
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**Q:** {item['question']}")
            ans = item["answer"]
            st.markdown(f"**A:** {ans[:300]}{'…' if len(ans) > 300 else ''}")
            st.caption(f"Time: {item['time']:.2f}s")
            st.markdown("")


if __name__ == "__main__":
    main()
