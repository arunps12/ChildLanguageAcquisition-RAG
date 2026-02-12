"""Persistence helpers for FAISS store artifacts.

Complementary to ``faiss_store.py`` — handles chunks.jsonl and build_manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document


def save_chunks_jsonl(chunks: List[Document], path: Path) -> None:
    """Write chunk metadata + text to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            record = {
                "chunk_id": ch.metadata.get("chunk_id", ""),
                "paper_id": ch.metadata.get("paper_id", ""),
                "title": ch.metadata.get("title", ""),
                "text": ch.page_content,
                "metadata": {
                    k: v
                    for k, v in ch.metadata.items()
                    if k not in ("chunk_id", "paper_id", "title")
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: Path) -> List[Document]:
    """Read a JSONL file back into LangChain Documents."""
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            meta["chunk_id"] = obj.get("chunk_id", "")
            meta["paper_id"] = obj.get("paper_id", "")
            meta["title"] = obj.get("title", "")
            docs.append(Document(page_content=obj["text"], metadata=meta))
    return docs


def read_build_manifest(index_dir: Path) -> Dict[str, Any]:
    """Read build_manifest.json from index directory."""
    path = index_dir / "build_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
