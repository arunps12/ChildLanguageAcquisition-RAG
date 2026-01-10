# ChildLanguageAcquisition-RAG

# ChildLanguageAcquisition-RAG

A **metadata-first, agentic Retrieval-Augmented Generation (RAG)** system for **child language acquisition research**, built with **LangChain, LangGraph, FAISS, OpenAI, and Streamlit**.

The system enables researchers to query a curated corpus of academic papers (PDFs and web-based sources) and obtain **grounded, citation-aware answers** through an agentic retrieval and generation workflow.

All data processing steps including document ingestion, chunking, vector index construction, and index persistence are tracked using **DVC (Data Version Control)**, ensuring **end-to-end reproducibility**, versioned datasets, and transparent dependency management across research and deployment environments.

The project also includes a **production-oriented CI/CD pipeline** using **Jenkins and Docker**, enabling automated builds, containerization, and deployment of the Streamlit application to **AWS EC2 via Amazon ECR**, bridging research workflows and real-world deployment.

The diagram below illustrates the end-to-end CI/CD workflow of **ChildLanguageAcquisition-RAG**.
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
├── childlanguagenet/                 # Core RAG library (Python package)
│   ├── __init__.py                   # Package marker / public API
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                 # Central configuration
│   │
│   ├── document_ingestion/
│   │   ├── __init__.py
│   │   └── document_processor.py     # PDF / URL ingestion & metadata parsing
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── vectorstore.py            # FAISS vector store logic
│   │
│   ├── graph_builder/
│   │   ├── __init__.py
│   │   └── graph_builder.py          # LangGraph agentic pipeline
│   │
│   ├── node/
│   │   ├── __init__.py
│   │   ├── react_node.py             # ReAct-style agent reasoning node
│   │   └── rag_node.py               # Retrieval–Augmented Generation node
│   │
│   └── state/
│       ├── __init__.py
│       └── rag_state.py              # Shared RAG state definition
│
├── streamlit_app.py                  # Streamlit UI entry point
├── main.py                           # CLI / programmatic entry point
│
├── data/
│   ├── metadata.json                 # Central paper registry
│   ├── pdf/                          # Local PDF corpus
│   └── index/
│       └── faiss/                    # Persisted FAISS indices
│
├── ci-cd-logs/                       # Jenkins / CI build logs (artifacts)
│
├── .github/                          # GitHub Actions CI workflows
│   └── workflows/
│       └── main.yml                  # CI pipeline (lint, test, build)
│
├── .jenkins/                         # Jenkins pipeline definitions
│   └── Jenkinsfile                  # Jenkins CI/CD pipeline
│
├── Dockerfile                        # Production Docker image
├── docker-compose.yml                # Local multi-service orchestration
│
├── pyproject.toml                    # Project metadata (uv-managed)
├── uv.lock                           # Fully locked, reproducible dependencies
├── requirements.txt                  # Runtime-only dependencies (deployment)
│
├── .env                              # Environment variables (NOT committed)
├── LICENSE                           # Open-source license
└── README.md                         # Project documentation
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
