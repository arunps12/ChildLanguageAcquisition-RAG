"""Lightweight metrics counters and histograms for ChildLanguageNet."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Optional


class Metrics:
    """In-process counters + latency histograms.

    Not a full Prometheus setup — just good enough for Streamlit-only mode.
    Periodically call :meth:`snapshot` to persist to ``artifacts/metrics/``.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, list] = defaultdict(list)

    # ── counters ───────────────────────────────────────────────────────

    def inc(self, name: str, n: int = 1) -> None:
        self._counters[name] += n

    def get_counter(self, name: str) -> int:
        return self._counters[name]

    # ── histograms ─────────────────────────────────────────────────────

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        start = time.monotonic()
        yield
        elapsed = time.monotonic() - start
        self._histograms[name].append(elapsed)

    # ── snapshot ───────────────────────────────────────────────────────

    def snapshot(self, path: Optional[Path] = None) -> Dict:
        """Return (and optionally persist) a JSON-safe snapshot."""
        data = {
            "counters": dict(self._counters),
            "histograms": {k: {"count": len(v), "mean": sum(v) / len(v) if v else 0}
                           for k, v in self._histograms.items()},
        }
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data


# Module-level singleton
_metrics = Metrics()


def get_metrics() -> Metrics:
    """Return the global metrics singleton."""
    return _metrics
