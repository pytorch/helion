"""Typed probe of Helion's real exact autotune cache (§2 step 2).

Consulting the exact cache returns exactly one of :class:`ExactHit`,
:class:`ExactMiss`, or :class:`ExactReadError`. The distinction matters: a read
error is recorded separately, excluded from incremental RAG-coverage
denominators, and routed to the frozen fail-closed baseline — it must never be
reclassified as a miss (§2, §20.1).

The cache is any object exposing ``get_or_raise() -> Config | None`` (returning
``None`` only for a genuine miss and raising on a read error), which
:class:`~helion.autotuner.local_cache.LocalAutotuneCache` provides.
"""

from __future__ import annotations

from typing import Protocol

from .types import ExactHit
from .types import ExactMiss
from .types import ExactProbeResult
from .types import ExactReadError


class _ExactCache(Protocol):
    def get_or_raise(self) -> object | None: ...


def probe_exact_cache(cache: _ExactCache) -> ExactProbeResult:
    """Return the typed outcome of reading Helion's real exact cache."""
    try:
        config = cache.get_or_raise()
    except Exception as exc:
        return ExactReadError(error=f"{type(exc).__name__}: {exc}")
    if config is None:
        return ExactMiss()
    return ExactHit(config=config)
