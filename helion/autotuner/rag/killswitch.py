"""RAG kill switch — the first thing evaluated in the runtime order (§2 step 1).

When RAG is disabled the caller must return immediately to the unchanged baseline
without verifying manifests, prewarming providers, or importing/loading Qwen,
FAISS, manifests, or indexes. This module deliberately imports nothing heavy so
the check itself is free.
"""

from __future__ import annotations

import os
from typing import Protocol


class _HasRagFlag(Protocol):
    autotune_rag_enabled: bool


def rag_enabled(settings: _HasRagFlag) -> bool:
    """True when the opt-in RAG policy is enabled for these settings."""
    return bool(settings.autotune_rag_enabled)


def rag_enabled_env() -> bool:
    """Env-only kill switch for call sites that run before ``Settings`` exists.

    Mirrors the ``HELION_RAG_ENABLED`` default used by
    ``Settings.autotune_rag_enabled``.
    """
    return os.environ.get("HELION_RAG_ENABLED", "").strip().lower() not in {
        "",
        "0",
        "false",
    }
