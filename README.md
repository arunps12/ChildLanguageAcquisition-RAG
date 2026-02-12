# ChildLanguageAcquisition-RAG

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-2D9CDB)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-Package_Manager-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/Ruff-Linter-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **metadata-first, agentic Retrieval-Augmented Generation (RAG)** system for **child language acquisition research**, built with **LangChain, LangGraph, FAISS, Ollama, and Streamlit**.

> **No API key required.** The app runs entirely locally using [Ollama](https://ollama.com/) for both embeddings and LLM inference.

The system enables researchers to query a curated corpus of academic papers (PDFs and web-based sources) and obtain **grounded, citation-aware answers** through an agentic retrieval and generation workflow.

All data processing steps — document ingestion, chunking, vector index construction, and index persistence — are tracked using **DVC (Data Version Control)**, ensuring **end-to-end reproducibility**.

The project also includes a **production-oriented CI/CD pipeline** (GitHub Actions + Jenkins/Docker), enabling automated builds, containerization, and deployment to **AWS EC2 via Amazon ECR**.


---

## How It Works

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion Pipeline"]
        A["metadata.json"] --> B["PDF / URL Loaders"]
        B --> C["Deterministic Chunker"]
    end

    subgraph Indexing["Indexing"]
        C --> D["Ollama Embeddings"]
        D --> E["FAISS\nVector Index"]
    end

    subgraph RAG["Agentic RAG - LangGraph"]
        F["User Query"] --> G["Retrieve Top-K Chunks"]
        G --> H["ReAct Agent - Ollama LLM"]
        H --> I["Citation-Aware Answer"]
    end

    E -.-> G

    subgraph Serve["Delivery"]
        I --> J["Streamlit UI"]
        I --> K["CLI"]
    end

    subgraph Ops["Ops"]
        L["DVC Reproducibility"] -.-> Ingestion
        L -.-> Indexing
        M["Docker + CI/CD"] -.-> Serve
    end

    style Ingestion fill:#e8f4fd,stroke:#2196F3
    style Indexing fill:#e8f5e9,stroke:#4CAF50
    style RAG fill:#fff3e0,stroke:#FF9800
    style Serve fill:#fce4ec,stroke:#E91E63
    style Ops fill:#f3e5f5,stroke:#9C27B0
```

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
│       │   └── embedder.py            # Ollama / local embeddings
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

## Usage

### For Researchers (end users)

#### Option A — Run locally without Docker

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/ChildLanguageAcquisition-RAG.git
cd ChildLanguageAcquisition-RAG

# 2. Install Ollama (one-time) — https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Install dependencies
pip install uv && uv sync && source .venv/bin/activate

# 4. Launch the app (index auto-builds on first run)
cp .env.example .env   # defaults to Ollama — no API key needed
childrag-serve
```

Open http://localhost:8501 — **no API key required**.

#### Option B — Run locally with Docker (one command)

If the app is **not** deployed, you can run it on your own machine using Docker. You only need [Docker](https://docs.docker.com/get-docker/) and [Ollama](https://ollama.com/download) installed.

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/ChildLanguageAcquisition-RAG.git
cd ChildLanguageAcquisition-RAG

# 2. Ensure Ollama is running with required models
ollama pull llama3.2 && ollama pull nomic-embed-text

# 3. Launch (builds the image + starts the app)
docker compose up
```

Open http://localhost:8501 — the app auto-builds the index on first launch.

> **No API key, no Python setup required** — Docker + Ollama handle everything.

---

### For Developers (local setup)

#### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/ChildLanguageAcquisition-RAG.git
cd ChildLanguageAcquisition-RAG
```

#### 2. Install dependencies (uv recommended)

```bash
pip install uv
uv sync --extra dev
source .venv/bin/activate
```

#### 3. Install Ollama models

```bash
# One-time setup — https://ollama.com/download
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### 4. Environment variables

```bash
cp .env.example .env
# Defaults to Ollama — no API key needed
```

#### 4. Build the FAISS index

```bash
childrag-index
```

> **Don't want to build the index yourself?** You have two options:
>
> 1. **Pull the pre-built index** — if a [DVC remote](https://dvc.org/doc/user-guide/data-management/remote-storage) is configured, just run:
>
>    ```bash
>    dvc pull
>    ```
>
>    This downloads the ready-to-use FAISS index. **No PDFs needed** — go straight to Step 5.
>
> 2. **Auto-build on first launch** — if no index is found, the Streamlit app automatically builds one from `data/pdf/` + `metadata.json` when you open it.
>
> In both cases, Ollama must be running locally.

#### 5. Run the Streamlit app

```bash
childrag-serve
# or: streamlit run apps/streamlit_app.py
```

Open http://localhost:8501

---

### Who needs what?

| Scenario | Clone repo? | Docker? | Ollama? | PDFs? | Build index? |
|---|---|---|---|---|---|
| **Researcher** — local without Docker | Yes | No | Yes | No | No (auto) |
| **Researcher** — local with Docker | Yes | Yes | Yes | No | No (auto) |
| **Developer** with `dvc pull` | Yes | No | Yes | No | No |
| **Developer** building from scratch | Yes | No | Yes | Yes | Yes |

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
docker run -p 8501:8501 childlanguagenet-rag
```

Or with docker-compose:

```bash
docker compose up
```

> Ollama must be running on the host. The container connects to `host.docker.internal:11434` by default.

---

## Example Questions

- What are the main characteristics of infant-directed speech discussed across these papers?
- How does infant-directed speech differ from adult-directed speech?
- What experimental methods are used to study early language development?
- Which papers analyze prosodic or phonetic exaggeration in infant-directed speech?

---

## Tech Stack

- **Ollama** — local LLM + embeddings (llama3.2, nomic-embed-text)
- **LangChain** + **LangGraph** — RAG orchestration
- **FAISS** — vector similarity search
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
