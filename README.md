# ChildLanguageAcquisition-RAG

A **metadata-first, agentic Retrieval-Augmented Generation (RAG)** system for **child language acquisition research**, built with **LangChain, LangGraph, FAISS, OpenAI, and Streamlit**.

The system enables researchers to query a curated corpus of academic papers (PDFs and URLs) and obtain **grounded, citation-aware answers** through an agentic workflow.

---

## Key Features

- **Metadata-first document ingestion**
  - Central paper registry (`data/metadata.json`)
  - Supports local PDFs and web URLs
- **Agentic RAG workflow**
  - LangGraph pipeline: `retrieve → generate`
- **FAISS vector store**
  - Efficient similarity search
  - Metadata preserved for citations
- **Interactive Streamlit interface**
- **Reproducible dependency management with `uv`** (`uv.lock` committed)

---

## Project Structure

```
ChildLanguageAcquisition-RAG/
├── app.py                      # Streamlit application
├── main.py                     # CLI entry point
├── childlanguagenet/
│   ├── config/
│   │   └── config.py
│   ├── document_ingestion/
│   │   └── document_processor.py
│   ├── vectorstore/
│   │   └── vectorstore.py
│   ├── graph_builder/
│   │   └── graph_builder.py
│   ├── node/
│   │   └── react_node.py
│   └── state/
│       └── rag_state.py
├── data/
│   ├── metadata.json            # Paper registry
│   ├── pdf/                     # Local PDFs
│   └── index/faiss/             # FAISS index
├── pyproject.toml
├── uv.lock                      # Locked, reproducible dependencies
├── .env
└── README.md
```

---

## How It Works

1. **Metadata registry** defines the research corpus (`metadata.json`)
2. PDFs / URLs are loaded and split into chunks
3. Chunks are embedded and indexed with FAISS
4. LangGraph orchestrates retrieval and answer generation
5. Streamlit provides an interactive research UI

---

## Installation (Recommended: `uv`)

This project uses **`uv`** for **fast, reproducible Python dependency management**.  
The committed `uv.lock` ensures identical environments across machines.

### 1️⃣ Install `uv`

```bash
pip install uv
```

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### 2️⃣ Create environment & install dependencies

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

---

### 3️⃣ Environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
USER_AGENT=ChildLanguageAcquisitionRAG/0.1 (University of Oslo)
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

### Remote server (e.g. UiO ml2)

```bash
ssh -N -L 8501:localhost:8501 username@server
```

Then open locally:

```
http://localhost:8501
```

---

## 🖥 Run from CLI (optional)

```bash
python main.py
```

---

## Example Questions

- What are the main characteristics of infant-directed speech discussed across the papers in this corpus?
- How does infant-directed speech differ from adult-directed speech according to these studies?
- What experimental or computational methods are used to study early language development?
- Which papers analyze prosodic or phonetic exaggeration in infant-directed speech?

---

## Rebuilding the FAISS Index

```bash
rm -rf data/index/faiss
streamlit run app.py
```

---

## 🧩 Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- FAISS
- OpenAI
- uv

---

## 📌 Use Cases

- Child language acquisition research
- Literature review automation
- IDS vs ADS analysis
- Research-grade RAG systems
- Academic demos and teaching

---

## 🙌 Acknowledgements

Developed at the **University of Oslo**  
Department of Linguistics and Scandinavian Studies

---

## 📬 Contact

**Arun Prakash Singh**  
University of Oslo  
📧 a.p.singh@iln.uio.no
