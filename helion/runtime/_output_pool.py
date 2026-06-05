"""Optional output-tensor pool for Helion kernels.

For small kernels, allocating a fresh output tensor with
``torch.empty_like(x)`` is a measurable per-call cost (~1.5 μs on H100
for a 4 KB tensor). This module provides ``_empty_like`` — a drop-in
replacement that, when ``HELION_OUTPUT_POOL=1`` is set in the
environment, caches one buffer per ``(dtype, shape, device, _slot)``
key and returns the cached buffer on subsequent calls with the same
key.

When the env var is NOT set (the default), ``_empty_like`` is a thin
pass-through to ``torch.empty_like`` and behaves exactly like the
PyTorch builtin.

Everything in this module is private (``_``-prefixed): the pool is
internal infrastructure consumed by Helion's codegen and autotuner,
not part of the public ``helion.runtime`` surface.

Semantics caveat
----------------
Pooled buffers are RECYCLED across calls. The same call site with the
same ``(dtype, shape, device, _slot)`` key gets the **same Python
tensor object** every call — its contents are overwritten by the
kernel each iteration. Callers MUST consume the returned tensor before
the next call with the same key, or copy the data out: holding a
reference for later use will see its contents clobbered. This is why
the pool is opt-in: ``torch.empty_like`` semantics (fresh allocation)
are the default to avoid silent aliasing bugs.

Thread safety
-------------
The enabled flag and the cache are both **per-thread** (backed by
``threading.local``). Calling :func:`_set_pool_enabled` /
:func:`_enable_pool` / :func:`_empty_like` from one thread has no
effect on what other threads see. This matches the autotune use case
(one tuner thread scopes the pool to its bench loop without affecting
concurrent inference threads) and avoids the bookkeeping that a
process-global shared cache would need.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Hashable


def _env_enabled() -> bool:
    return os.environ.get("HELION_OUTPUT_POOL", "").lower() in ("1", "true", "yes")


# Default flag, captured once at import. Each thread inherits this
# value on first read but can override via ``_set_pool_enabled``.
_DEFAULT_ENABLED: bool = _env_enabled()

# Per-thread state. ``_state.enabled`` (bool) and ``_state.cache``
# (dict) are lazily initialized on first access via the helpers below.
_state: threading.local = threading.local()


def _cache() -> dict[Hashable, torch.Tensor]:
    cache = getattr(_state, "cache", None)
    if cache is None:
        cache = {}
        _state.cache = cache
    return cache


def _set_pool_enabled(enabled: bool) -> None:
    """Programmatic toggle for the pool, overriding the env var.

    Per-thread: only affects the calling thread.
    """
    _state.enabled = enabled


def _is_pool_enabled() -> bool:
    """Whether the pool is currently enabled on the calling thread."""
    return getattr(_state, "enabled", _DEFAULT_ENABLED)


def _clear_pool() -> None:
    """Drop all pooled buffers for the calling thread.

    Useful for tests and for callers that want to reclaim memory after
    a workload finishes.
    """
    _cache().clear()


@contextlib.contextmanager
def _enable_pool() -> Generator[None, None, None]:
    """Context manager: enable the pool for the duration of the
    ``with`` block on the calling thread, then disable and clear the
    cache on exit.

    NOT re-entrant. Calling :func:`_enable_pool` while the pool is
    already enabled on the same thread raises ``RuntimeError``. The
    autotuner is the only caller today and never nests; we surface the
    constraint loudly so accidental nesting (which would clobber
    another scope's cached buffers via the exit-time ``_clear_pool``)
    fails immediately rather than silently.

    Per-thread: only affects the calling thread.
    """
    if _is_pool_enabled():
        raise RuntimeError(
            "helion.runtime._output_pool._enable_pool() is not re-entrant; "
            "pool is already enabled on this thread"
        )
    _set_pool_enabled(True)
    try:
        yield
    finally:
        _set_pool_enabled(False)
        _clear_pool()


def _empty_like(template: torch.Tensor, _slot: int = 0) -> torch.Tensor:
    """Return an uninitialized tensor with the same dtype/shape/device
    as ``template``.

    When :func:`_is_pool_enabled` is ``True``, returns the cached buffer
    for the ``(dtype, shape, device, _slot)`` key, allocating one on
    first use. The same Python tensor object is returned across calls
    with the same key. When ``False``, falls through to
    ``torch.empty_like``.

    ``_slot`` disambiguates multiple kernel-output allocations in the
    same generated wrapper that share a ``(dtype, shape, device)``
    triple. The codegen rewrite (``_rewrite_output_allocs_for_pool``)
    assigns a unique ``_slot`` per call site so that, e.g., a two-output
    kernel emitting two ``empty_like(x)`` allocations gets two distinct
    cached buffers rather than collapsing both onto one (which would
    silently alias them). Default ``_slot=0`` covers the single-output
    case and direct callers.

    Callers must consume the returned tensor before the next call with
    the same signature, or copy the data out — see the module
    docstring.
    """
    if not _is_pool_enabled():
        return torch.empty_like(template)
    cache = _cache()
    key = (template.dtype, tuple(template.size()), template.device, _slot)
    buf = cache.get(key)
    if buf is None:
        buf = torch.empty_like(template)
        cache[key] = buf
    return buf


__all__ = [
    "_clear_pool",
    "_empty_like",
    "_enable_pool",
    "_is_pool_enabled",
    "_set_pool_enabled",
]
