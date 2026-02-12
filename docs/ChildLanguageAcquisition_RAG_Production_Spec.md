# ChildLanguageAcquisition-RAG — Production Architecture & Packaging Specification

Generated on: 2026-02-12 (Europe/Oslo)

------------------------------------------------------------------------

# OBJECTIVE

Refactor and harden **ChildLanguageAcquisition-RAG** into a production-grade,
research-friendly RAG system that:

1. Uses **src-layout** Python packaging
2. Ships as a **reusable library** + a **runnable app**
3. Implements a **metadata-first ingestion pipeline** (PDF + URL)
4. Persists **FAISS indices** with metadata for citation grounding
5. Tracks data & indices with **DVC** for end-to-end reproducibility
6. Provides **Streamlit UI** + optional **CLI** entry points
7. Adds **observability** (structured logging + basic metrics)
8. Adds **CI/CD** (GitHub Actions) and optional **Jenkins** pipeline
9. Supports **Docker** for local + AWS deployment (ECR → EC2)
10. Preserves existing behavior (do not break core user flows)

This document should be treated as the SINGLE SOURCE OF TRUTH for implementing
the production refactor, in the same spirit as the packaging/MLOps spec you
shared. fileciteturn0file0

------------------------------------------------------------------------

# SCOPE

**In scope**
- Project structure refactor (src-layout, import cleanup)
- Library boundaries + public API
- Ingestion, chunking, embedding, indexing, persistence
- Agentic workflow (LangGraph) wiring
- Streamlit app hardening (config, caching, errors, citations)
- DVC pipeline (stage-wise)
- CI (lint/test/build), Docker build, release artifacts
- Optional Jenkinsfile compatible with AWS ECR/EC2 deployment

**Out of scope**
- Changing the research design (metadata-first + citation grounding stays)
- Switching vector DB away from FAISS (unless explicitly required later)

------------------------------------------------------------------------

# PHASE 0 — STRUCTURE AUDIT (MANDATORY FIRST STEP)

Before coding:
1. Inspect the full repository tree.
2. Identify:
   - Dead code / unused notebooks
   - Redundant folders or duplicate logic
   - Hardcoded paths (local-only, machine-specific)
   - Mixed entry points (e.g., `app.py` vs `streamlit_app.py`)
   - Untracked data/index artifacts inside git
3. Propose the **minimal** deletions/moves to reach the target structure.
4. Execute the refactor while preserving working behavior.

Deliverables:
- A short `docs/STRUCTURE_AUDIT.md` summarizing changes and rationale.
- Git commits per phase (one per phase minimum).

------------------------------------------------------------------------

# PHASE 1 — PACKAGING (src-layout + Public API + Entry Points)

## 1.1 Target Structure

Target project structure:

ChildLanguageAcquisition-RAG/
│
├── src/
│   └── childlanguagenet/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── metadata_registry.py
│       │   ├── loaders.py
│       │   └── chunking.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── embedder.py
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── faiss_store.py
│       │   └── persistence.py
│       ├── graph/
│       │   ├── __init__.py
│       │   └── rag_graph.py
│       ├── citations/
│       │   ├── __init__.py
│       │   └── cite.py
│       ├── telemetry/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   └── metrics.py
│       └── cli.py
│
├── apps/
│   └── streamlit_app.py
│
├── data/
│   ├── metadata.json
│   ├── pdf/
│   └── index/
│       └── faiss/
│
├── artifacts/                 # reports, logs, drift/quality checks, etc.
├── tests/
│
├── dvc.yaml
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .jenkins/Jenkinsfile       # optional but supported
│
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE

Notes:
- Keep **data/** and **artifacts/** out of python package.
- Keep Streamlit app in **apps/** to separate UI from library.

## 1.2 Update Imports

- Replace all `from childlanguagenet...` imports to match `src/` layout.
- Remove any relative imports that break packaging.
- Ensure `python -m childlanguagenet` is not required (use scripts instead).

## 1.3 Entry Points

In `pyproject.toml`, define scripts:

[project.scripts]
childrag = "childlanguagenet.cli:main"
childrag-index = "childlanguagenet.cli:build_index"
childrag-serve = "childlanguagenet.cli:serve_streamlit"

Required behaviors:
- `childrag-index` builds/updates FAISS index from `data/metadata.json`
- `childrag-serve` runs Streamlit app (local)
- `childrag` provides a minimal CLI (help + subcommands)

------------------------------------------------------------------------

# PHASE 2 — CONFIGURATION (Typed Settings + .env)

Implement `src/childlanguagenet/config/settings.py` with:
- Data paths (metadata file, pdf dir, index dir)
- Chunking params (chunk size, overlap)
- Embedding params (model name, batch size)
- Vectorstore params (FAISS index type, normalize, distance metric)
- LLM params (model name, temperature, max tokens)
- UI params (title, max sources to show)
- Telemetry params (log level, metrics enabled)

Rules:
- Support `.env` via `python-dotenv`
- No hardcoded absolute paths
- Defaults should work for local development

Provide `.env.example` in repo.

------------------------------------------------------------------------

# PHASE 3 — METADATA-FIRST INGESTION

## 3.1 Metadata Registry

`data/metadata.json` remains the single registry.

Implement a schema (validated at runtime):
- id (string, unique)
- title
- authors (optional)
- year (optional)
- source_type: "pdf" | "url"
- path_or_url
- tags (optional list)
- notes (optional)
- license (optional)

Validation requirements:
- Fail fast on missing required fields
- Detect duplicate IDs
- For PDFs: verify file exists
- For URLs: verify scheme (http/https)

## 3.2 Loaders

Implement:
- PDF loader (robust to common PDF parsing errors)
- URL loader (simple HTTP fetch; optionally reader-mode extraction)
- Normalize text (strip repeated whitespace, remove page headers if possible)

## 3.3 Chunking

Implement a deterministic chunker:
- Stable splitting strategy
- Chunk IDs include document ID + chunk number
- Preserve metadata on each chunk (doc id, title, source, page if available)

------------------------------------------------------------------------

# PHASE 4 — EMBEDDING + FAISS VECTORSTORE

## 4.1 Embedder

Implement `embedder.py` supporting:
- OpenAI embeddings (default)
- Optionally local sentence-transformers (behind a flag)

Requirements:
- Batch embedding
- Deterministic caching key (text hash + model name)
- Store embedding metadata alongside chunks (for reproducibility)

## 4.2 FAISS Store

Implement:
- Build index from chunk embeddings
- Save index + chunk metadata to disk under `data/index/faiss/`
- Load existing index if present (fast startup)

Persistence artifacts:
- `index.faiss`
- `chunks.parquet` (or jsonl) containing chunk text + metadata
- `build_manifest.json` (timestamp, git commit, embedding model, counts)

------------------------------------------------------------------------

# PHASE 5 — AGENTIC RAG (LangGraph)

Implement `rag_graph.py` with a clear state model:
- user_query
- retrieved_chunks (with metadata)
- answer_text
- citations (doc id/title + quote spans if available)
- debug_info (optional)

Graph: `retrieve → generate`

Requirements:
- Retrieval step returns top-k chunks with similarity scores
- Generation step produces:
  - Direct answer
  - “Sources” section listing citations
- Provide “citation-aware” formatting helpers in `citations/cite.py`

------------------------------------------------------------------------

# PHASE 6 — STREAMLIT APP HARDENING

Move UI entry point to `apps/streamlit_app.py`.

Requirements:
- A single, stable command to run (used in Docker/CI):
  - `streamlit run apps/streamlit_app.py`
- App should:
  - Load index on startup (or prompt to build)
  - Provide a “Build/Refresh Index” button
  - Show retrieved sources with metadata
  - Display grounded citations clearly
  - Handle errors gracefully (PDF parse failure, missing API key)

Caching:
- Use Streamlit caching for index load
- Do not cache responses in a way that mixes users unintentionally

------------------------------------------------------------------------

# PHASE 7 — DVC PIPELINE (REPRODUCIBLE STAGES)

Create a stage-wise `dvc.yaml`:

stages:
  validate_metadata:
    cmd: python -m childlanguagenet.cli validate-metadata --metadata data/metadata.json
    deps:
      - data/metadata.json
    outs:
      - artifacts/metadata_validation.json

  ingest:
    cmd: python -m childlanguagenet.cli ingest --metadata data/metadata.json --out artifacts/ingested_texts/
    deps:
      - data/metadata.json
      - data/pdf/
    outs:
      - artifacts/ingested_texts/

  build_index:
    cmd: python -m childlanguagenet.cli build-index --inputs artifacts/ingested_texts/ --out data/index/faiss/
    deps:
      - artifacts/ingested_texts/
    outs:
      - data/index/faiss/

Rules:
- Keep large artifacts tracked by DVC, not git.
- Ensure `dvc repro` can rebuild the index from scratch.

------------------------------------------------------------------------

# PHASE 8 — OBSERVABILITY (Logging + Basic Metrics)

## 8.1 Structured Logging

Implement `telemetry/logging.py`:
- JSON logs (optional) or consistent formatter
- Include:
  - app version (git SHA if available)
  - embedding model
  - index build id
- Log key events: index load/build, retrieval count, errors

## 8.2 Metrics (Optional but recommended)

Implement `telemetry/metrics.py`:
- Counters:
  - queries_total
  - index_build_total
  - errors_total
- Histograms:
  - retrieval_latency_seconds
  - generation_latency_seconds

Expose metrics:
- If running behind FastAPI, expose `/metrics`
- For Streamlit-only, write periodic metrics snapshots to `artifacts/metrics/`

(Keep this lightweight; do not over-engineer.)

------------------------------------------------------------------------

# PHASE 9 — CI/CD

## 9.1 GitHub Actions (Required)

`.github/workflows/ci.yml` must:
- Set up Python (pinned)
- `uv sync`
- Run:
  - lint (ruff)
  - type check (optional: mypy)
  - tests (pytest)
- Build Docker image
- (Optional) run `dvc repro` in CI if small test corpus is available

## 9.2 Jenkins (Optional but supported)

Provide `.jenkins/Jenkinsfile` that:
- Builds Docker image
- Pushes to Amazon ECR
- Deploys to EC2 (pull + restart container)
- Uses environment variables for AWS credentials and region

------------------------------------------------------------------------

# PHASE 10 — DOCKER (Local + AWS)

## 10.1 Dockerfile

Must:
- Install dependencies via `uv` (preferred) or `pip` fallback
- Copy `apps/streamlit_app.py` and `src/`
- Expose Streamlit port 8501
- Entry:
  - `streamlit run apps/streamlit_app.py --server.port=8501 --server.address=0.0.0.0`

## 10.2 docker-compose.yml

Provide local services:
- rag-app (Streamlit)
- (optional) a volume for `data/index/` and `artifacts/`

------------------------------------------------------------------------

# SECURITY & COMPLIANCE NOTES

- Never commit `.env` or API keys.
- Provide `.env.example` and document required vars.
- Avoid logging raw user queries in production logs unless required.
- Respect copyright: store only what is needed for indexing; limit redisplay.

------------------------------------------------------------------------

# ACCEPTANCE CHECKLIST

The refactor is complete when:

- `pip install -e .` installs the package from `src/`
- `childrag-index` builds FAISS index and persists it
- `childrag-serve` runs the Streamlit UI successfully
- `dvc repro` runs end-to-end on a small sample corpus
- CI passes lint + tests + Docker build
- Docker image runs locally and serves Streamlit on :8501
- Citations show doc metadata and are grounded in retrieved chunks

------------------------------------------------------------------------

# END OF SPECIFICATION
