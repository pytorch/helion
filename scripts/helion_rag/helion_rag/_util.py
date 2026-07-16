"""Tiny stdlib helpers shared across helion_rag."""

from __future__ import annotations

import math
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


def _parse_sim_threshold(raw: str) -> float:
    """Parse a finite similarity in [0, 1], else use the tested default."""
    try:
        value = float(raw.strip())
    except ValueError:
        return DEFAULT_SIM_THRESHOLD
    return (
        value if math.isfinite(value) and 0.0 <= value <= 1.0 else DEFAULT_SIM_THRESHOLD
    )


def _sim_threshold() -> float:
    """Read HELION_RAG_SIM_THRESHOLD as a finite similarity in [0, 1]."""
    return _parse_sim_threshold(os.environ.get("HELION_RAG_SIM_THRESHOLD", ""))
