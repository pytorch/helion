"""Optional output-tensor pool for Helion kernels.

For small kernels, allocating a fresh output tensor with
``torch.empty_like(x)`` is a measurable per-call cost (~1.5 μs on H100
for a 4 KB tensor). This module provides ``empty_like`` — a drop-in
replacement that, when ``HELION_OUTPUT_POOL=1`` is set in the
environment, reuses a small fixed-size cache of buffers keyed on
``(dtype, shape, stride, device)``.

When the env var is NOT set (the default), ``empty_like`` is a thin
pass-through to ``torch.empty_like`` and behaves exactly like the
PyTorch builtin.

Semantics caveat
----------------
Pooled buffers are RECYCLED across calls. Callers MUST consume the
returned tensor before the next call to ``empty_like`` with the same
key, or copy the data out. The buffer returned this call may be
re-handed to the next call — holding a reference for later use will
see its contents clobbered. This is why the pool is opt-in: the
``torch.empty_like`` semantics (fresh allocation) are the default to
avoid silent aliasing bugs.

Layout
------
``_POOLS`` is a process-wide dict mapping a tensor signature
(``(dtype, sizes, strides, device)``) to a small ring buffer (list)
of preallocated tensors. Each ``empty_like`` call rotates through the
ring, returning the next slot. Ring length is fixed at module-init.

The pool is intentionally simple: no LRU eviction, no maximum total
memory cap. The expected use case is hot inference loops that call
the same kernels with the same shapes repeatedly, so the pool grows
to its steady-state size after warmup and stays there.

Disabling at runtime
--------------------
Reading the env var once at import time (cached in ``_POOL_ENABLED``)
keeps ``empty_like`` allocation-free in the disabled path. To change
the setting after import, call :func:`set_pool_enabled`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Hashable


# Ring-buffer length per (dtype, sizes, strides, device) key. Two
# slots is enough to cover the common case (one buffer being written
# while the previous is still being read on another stream); going
# higher just wastes memory.
_POOL_DEPTH = 2

_POOLS: dict[Hashable, list[torch.Tensor]] = {}
_POOL_INDEX: dict[Hashable, int] = {}


def _env_enabled() -> bool:
    return os.environ.get("HELION_OUTPUT_POOL", "").lower() in ("1", "true", "yes")


_POOL_ENABLED: bool = _env_enabled()


def set_pool_enabled(enabled: bool) -> None:
    """Programmatic toggle for the pool, overriding the env var.

    Useful for tests and for libraries that want to opt in/out
    independently of the user's environment.
    """
    global _POOL_ENABLED
    _POOL_ENABLED = enabled


def is_pool_enabled() -> bool:
    return _POOL_ENABLED


def clear_pool() -> None:
    """Drop all pooled buffers. Useful for tests and for callers that
    want to reclaim memory after a workload finishes."""
    _POOLS.clear()
    _POOL_INDEX.clear()


def empty_like(template: torch.Tensor) -> torch.Tensor:
    """Return an uninitialized tensor with the same dtype/shape/stride/device
    as ``template``.

    When :func:`is_pool_enabled` is ``True``, returns a recycled buffer
    from a per-signature ring. When ``False``, falls through to
    ``torch.empty_like``.

    Callers must consume the returned tensor before the next call with
    the same signature, or copy the data out — see the module
    docstring.
    """
    if not _POOL_ENABLED:
        return torch.empty_like(template)
    key = (
        template.dtype,
        tuple(template.size()),
        tuple(template.stride()),
        template.device,
    )
    ring = _POOLS.get(key)
    if ring is None:
        ring = [torch.empty_like(template) for _ in range(_POOL_DEPTH)]
        _POOLS[key] = ring
        _POOL_INDEX[key] = 0
    idx = _POOL_INDEX[key]
    _POOL_INDEX[key] = (idx + 1) % _POOL_DEPTH
    return ring[idx]


__all__ = [
    "clear_pool",
    "empty_like",
    "is_pool_enabled",
    "set_pool_enabled",
]
