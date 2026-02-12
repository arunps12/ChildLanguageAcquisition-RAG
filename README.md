# ChildLanguageAcquisition-RAG

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-2D9CDB)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-Package_Manager-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/Ruff-Linter-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **metadata-first, agentic Retrieval-Augmented Generation (RAG)** system for **child language acquisition research**, built with **LangChain, LangGraph, FAISS, OpenAI, and Streamlit**.

The system enables researchers to query a curated corpus of academic papers (PDFs and web-based sources) and obtain **grounded, citation-aware answers** through an agentic retrieval and generation workflow.

All data processing steps — document ingestion, chunking, vector index construction, and index persistence — are tracked using **DVC (Data Version Control)**, ensuring **end-to-end reproducibility**.

The project also includes a **production-oriented CI/CD pipeline** (GitHub Actions + Jenkins/Docker), enabling automated builds, containerization, and deployment to **AWS EC2 via Amazon ECR**.

![CI/CD Pipeline](docs/image/ci_cd_pipeline.png)

---

## Project Structure

```
ChildLanguageAcquisition-RAG/
├── src/
│   └── childlanguagenet/
│       ├── __init__.py
│       ├── cli.py                     # CLI entry points
│       ├── config/
│       │   └── settings.py            # Typed settings (.env support)
│       ├── ingestion/
│       │   ├── metadata_registry.py   # Schema validation
│       │   ├── loaders.py             # PDF + URL loaders
│       │   └── chunking.py            # Deterministic chunking
│       ├── embeddings/
│       │   └── embedder.py            # OpenAI / local embeddings
│       ├── vectorstore/
│       │   ├── faiss_store.py         # FAISS build / load / retrieve
│       │   └── persistence.py         # Chunks JSONL + manifest
│       ├── graph/
│       │   └── rag_graph.py           # LangGraph RAG pipeline
│       ├── citations/
│       │   └── cite.py                # Citation model + formatting
│       └── telemetry/
│           ├── logging.py             # Structured logging
│           └── metrics.py             # Counters + histograms
│
├── apps/
│   └── streamlit_app.py              # Streamlit UI
│
├── data/
│   ├── metadata.json                  # Paper registry
│   ├── pdf/                           # Local PDF corpus
│   └── index/faiss/                   # Persisted FAISS index
│
├── artifacts/                         # Reports, metrics, ingested texts
├── tests/                             # Pytest test suite
│
├── dvc.yaml                           # DVC pipeline stages
├── Dockerfile                         # Production Docker image
├── docker-compose.yml                 # Local dev + deployment
├── .github/workflows/ci.yml           # GitHub Actions CI
├── .jenkins/Jenkinsfile               # Jenkins CI/CD pipeline
├── pyproject.toml                     # Project metadata + entry points
├── uv.lock                            # Locked dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies (uv recommended)

```bash
pip install uv
uv sync
source .venv/bin/activate
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Build the FAISS index

```bash
childrag-index
```

### 4. Run the Streamlit app

```bash
childrag-serve
# or: streamlit run apps/streamlit_app.py
```

Open http://localhost:8501

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `childrag --help` | Show available subcommands |
| `childrag-index` | Build/update FAISS index from metadata.json |
| `childrag-serve` | Launch Streamlit app |
| `childrag validate-metadata` | Validate metadata.json schema |
| `childrag ingest` | Ingest documents → chunked JSONL |
| `childrag build-index` | Build FAISS index from ingested chunks |

---

## DVC Pipeline

```bash
dvc repro
```

Stages: `validate_metadata` → `ingest` → `build_index`

---

## Docker

```bash
docker build -t childlanguagenet-rag .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... childlanguagenet-rag
```

Or with docker-compose:

```bash
OPENAI_API_KEY=sk-... docker compose up
```

---

## Example Questions

- What are the main characteristics of infant-directed speech discussed across these papers?
- How does infant-directed speech differ from adult-directed speech?
- What experimental methods are used to study early language development?
- Which papers analyze prosodic or phonetic exaggeration in infant-directed speech?

---

## Tech Stack

- **LangChain** + **LangGraph** — RAG orchestration
- **FAISS** — vector similarity search
- **OpenAI** — embeddings + LLM
- **Streamlit** — interactive UI
- **DVC** — data/pipeline versioning
- **uv** — fast dependency management
- **Docker** — containerization
- **GitHub Actions** + **Jenkins** — CI/CD

---

## Acknowledgements

Developed at the **University of Oslo**
Department of Linguistics and Scandinavian Studies

---

## Contact

**Arun Prakash Singh**
University of Oslo
📧 a.p.singh@iln.uio.no

---

## License

See [LICENSE](LICENSE).
