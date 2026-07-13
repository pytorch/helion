"""Tiny stdlib helpers shared across helion_rag. No heavy deps here."""

from __future__ import annotations

import os
import sys
from typing import NoReturn

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
INDEX_FILE = "index.faiss"
DEFAULT_SIM_THRESHOLD = 0.85
DEFAULT_TOP_N = 10


def _log(msg: str) -> None:
    """Print to stderr with prefix."""
    print(f"[helion_rag] {msg}", file=sys.stderr)


def _die(msg: str) -> NoReturn:
    """Log error and exit. Use for unrecoverable config or data issues."""
    _log(f"error: {msg}")
    raise SystemExit(1)


def _sim_threshold() -> float:
    """Read HELION_RAG_SIM_THRESHOLD or fall back to default."""
    v = os.environ.get("HELION_RAG_SIM_THRESHOLD", "").strip()
    return (
        float(v)
        if v.replace(".", "", 1).lstrip("-").isdigit()
        else DEFAULT_SIM_THRESHOLD
    )
