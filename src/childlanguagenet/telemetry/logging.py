"""Structured logging for ChildLanguageNet."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent format."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger, configuring the root logger on first call."""
    # Lazy-configure with default level; callers can reconfigure.
    configure_logging()
    return logging.getLogger(name or "childlanguagenet")
