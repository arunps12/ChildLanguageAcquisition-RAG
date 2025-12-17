"""Streamlit UI for childlanguagenet (metadata-first PDF RAG)."""

import time
from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on PYTHONPATH so `childlanguagenet.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from childlanguagenet.config.config import Config
from childlanguagenet.document_ingestion.document_processor import DocumentProcessor
from childlanguagenet.vectorstore.vectorstore import VectorStore
from childlanguagenet.graph_builder.graph_builder import GraphBuilder


st.set_page_config(
    page_title="Child Language Research RAG",
    page_icon=None,
    layout="centered",
)

st.markdown(
    """
    <style>
    .stButton > button { width: 100%; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():
    """
    Initialize the system using local PDFs and metadata.json only.

    Workflow:
    - Load metadata.json
    - Load local PDFs referenced by pdf_file
    - Chunk documents
    - Load FAISS index if present; otherwise build and save it
    - Build LangGraph workflow
    """
    llm = Config.get_llm()

    doc_processor = DocumentProcessor(
        data_dir=Config.DATA_DIR,
        metadata_path=Config.METADATA_FILE,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    vector_store = VectorStore(
        index_dir=Config.INDEX_DIR,
        k_default=Config.DEFAULT_TOP_K,
    )

    built_chunks = 0
    try:
        vector_store.load()
    except Exception:
        chunks = doc_processor.process_from_metadata()
        built_chunks = len(chunks)
        vector_store.create_vectorstore(chunks, save=True)

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm,
    )
    graph_builder.build()

    return graph_builder, built_chunks


def render_citations(result: dict):
    citations = result.get("citations") or []
    if not citations:
        return

    st.subheader("Sources")
    for c in citations:
        if isinstance(c, dict):
            pid = c.get("paper_id")
            title = c.get("title")
            year = c.get("year")
            doi = c.get("doi")
            url = c.get("source_url")
        else:
            pid = getattr(c, "paper_id", None)
            title = getattr(c, "title", None)
            year = getattr(c, "year", None)
            doi = getattr(c, "doi", None)
            url = getattr(c, "source_url", None)

        line = f"- {title or pid or 'Unknown'}"
        if year:
            line += f" ({year})"
        if doi:
            line += f" | DOI: {doi}"
        st.markdown(line)

        if url:
            st.caption(url)


def main():
    init_session_state()

    st.title("Child Language Research RAG")
    st.write("Question answering over a local research paper corpus (PDFs + metadata.json).")

    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            try:
                rag_system, built_chunks = initialize_rag()
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True

                if built_chunks > 0:
                    st.success(f"System ready. Built index from {built_chunks} chunks.")
                else:
                    st.success("System ready. Loaded existing FAISS index from disk.")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
                return

    st.divider()

    with st.form("qa_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What mechanisms explain why infant-directed speech supports learning?",
        )
        submit = st.form_submit_button("Ask")

    if submit and question:
        with st.spinner("Retrieving and generating answer..."):
            start = time.time()
            result = st.session_state.rag_system.run(question)
            elapsed = time.time() - start

        st.session_state.history.append(
            {"question": question, "answer": result.get("answer", ""), "time": elapsed}
        )
        st.session_state.history = st.session_state.history[-10:]

        st.subheader("Answer")
        st.write(result.get("answer", "No answer returned."))

        render_citations(result)

        with st.expander("Retrieved chunks (debug)"):
            docs = result.get("retrieved_docs", []) or []
            for i, doc in enumerate(docs, 1):
                meta = getattr(doc, "metadata", {}) or {}
                pid = meta.get("paper_id", "unknown_paper")
                title = meta.get("title", "Unknown title")
                st.markdown(f"Chunk {i}: {pid} | {title}")

                txt = doc.page_content
                if len(txt) > 1500:
                    txt = txt[:1500] + "..."
                st.text_area(f"chunk_{i}", txt, height=160)

        st.caption(f"Response time: {elapsed:.2f} seconds")

    if st.session_state.history:
        st.divider()
        st.subheader("Recent questions")
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"Question: {item['question']}")
            ans = item["answer"]
            st.markdown(f"Answer: {ans[:250]}{'...' if len(ans) > 250 else ''}")
            st.caption(f"Time: {item['time']:.2f}s")
            st.markdown("")


if __name__ == "__main__":
    main()
