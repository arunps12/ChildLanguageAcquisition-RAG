# Structure Audit — ChildLanguageAcquisition-RAG

**Date:** 2026-02-12  
**Phase:** 0

---

## Findings

### Dead Code / Unused Modules
| Item | Reason | Action |
|------|--------|--------|
| `childlanguagenet/node/rag_nodes.py` | Not imported anywhere — `graph_builder.py` imports `react_node.py` instead | **Removed** (logic preserved in new `graph/rag_graph.py`) |
| `main.py` (root) | Replaced by CLI entry points (`childrag`, `childrag-index`, `childrag-serve`) | **Removed** after migration |
| `streamlit_app.py` (root) | Migrated to `apps/streamlit_app.py` per spec | **Moved** |

### Redundant / Leftover Artifacts
| Item | Reason | Action |
|------|--------|--------|
| `ci-cd-logs/jenkins-console-2026-01-09.txt` | Build log artifact checked into git | **Deleted** |
| `data/index/faiss/index.pkl` | Stale pickle file; index will be rebuilt | **Deleted** |
| `docs/image/placeholder` | Empty placeholder file | **Deleted** |
| `requirements.txt` | Redundant with `pyproject.toml` + `uv.lock` | **Deleted** (deps fully managed via pyproject.toml) |

### Hardcoded Paths
| Location | Issue | Fix |
|----------|-------|-----|
| `config/config.py` `PROJECT_ROOT = Path(__file__).resolve().parents[2]` | Assumes nesting depth; breaks under src-layout | Replaced with `settings.py` using environment-aware defaults |
| `main.py` sys.path manipulation | Fragile `sys.path.insert(0, ...)` | Eliminated via proper packaging (src-layout + `pip install -e .`) |
| `streamlit_app.py` sys.path manipulation | Same issue | Eliminated |

### Security Concerns
| Item | Issue | Action |
|------|-------|-----|
| `.env` contains real API key | Should never be committed | Already in `.gitignore`; added `.env.example` template |

### Missing Directories (per spec)
- `src/childlanguagenet/` — src-layout package root
- `apps/` — Streamlit UI entry point
- `artifacts/` — reports, logs, metrics
- `tests/` — test suite

### Structural Changes Summary
1. **Moved** `childlanguagenet/` → `src/childlanguagenet/` (src-layout)
2. **Reorganized** modules per spec target structure:
   - `document_ingestion/` → `ingestion/` (metadata_registry.py, loaders.py, chunking.py)
   - `graph_builder/` + `node/` + `state/` → `graph/rag_graph.py` + `citations/cite.py`
   - `vectorstore/` → `vectorstore/` (faiss_store.py, persistence.py)
   - `config/config.py` → `config/settings.py`
   - Added `embeddings/embedder.py`
   - Added `telemetry/logging.py` + `telemetry/metrics.py`
   - Added `cli.py` with entry points
3. **Moved** `streamlit_app.py` → `apps/streamlit_app.py`
4. **Removed** `main.py` (replaced by CLI)
5. **Removed** `requirements.txt` (redundant)
6. **Removed** `ci-cd-logs/` directory
7. **Created** `artifacts/`, `tests/` directories
8. **Updated** `pyproject.toml` for src-layout + entry points
9. **Updated** `Dockerfile` and `docker-compose.yml`
10. **Added** `.github/workflows/ci.yml` (real CI pipeline)
11. **Updated** `.jenkins/Jenkinsfile`
12. **Added** `.env.example`
