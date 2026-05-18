"""Output buffer pool for kernel-output ``torch.empty`` allocations.

The generated host wrapper's ``torch.empty(...)`` calls that produce
kernel-output tensors are routed through ``_output_pool_alloc(...)``
(see ``helion/_compiler/generate_ast.py``), which reuses buffers across
calls to amortize the per-call ``cudaMalloc`` cost (~5 µs per launch on
small kernels).

Activation:

* **Default (env unset).** Pooling is active for every kernel call. The
  first time the pool actually reuses a buffer, a one-time INFO log line
  describes what's pooled and how to opt out.
* **``HELION_REUSE_OUTPUT_BUFFERS=0``** (or ``off``/``false``/``no``).
  The pool is disabled — every call falls through to ``torch.empty``.
  Use this if you keep raw ``data_ptr()`` to outputs across calls,
  share a single ``Kernel`` across Python threads, or otherwise need
  pre-pool semantics.
* **``HELION_REUSE_OUTPUT_BUFFERS=debug``.** Pool active and every pool
  decision (hit, refcount miss, capture bypass, fresh alloc) is logged
  at INFO. Use this to diagnose suspected output corruption.

When pooling is active, three signals guard against silent aliasing
(they cover essentially all real usage):

  1. Tensor reference count: a cached tensor has exactly 3 refs (dict
     slot + the local variable holding the lookup result + the
     ``sys.getrefcount`` arg) when no external code holds it. If the
     refcount exceeds 3 — either because the user kept the previous
     output, took a view (PyTorch view ops bump the parent's refcount
     via ``_base`` for autograd version tracking), or autograd saved
     it for backward — we allocate fresh instead.

  2. CUDA stream tagging: the cache key includes the current CUDA
     stream id. A buffer first produced on stream A is never returned
     to a caller running on stream B, so cross-stream consumers don't
     observe stale data.

  3. CUDA graph capture: when ``torch.cuda.is_current_stream_capturing()``
     is True, the pool is bypassed entirely (always ``torch.empty``).
     CUDA graph buffer reuse is decided by the graph builder, not by
     runtime liveness, so the pool would interfere otherwise.

The ``_pool_active`` context manager no longer toggles activation — it
exists purely as a clear-on-exit scope used by the autotuner so per-trial
cached buffers don't leak past the trial boundary.

Concurrency note: the safety guarantees here are per Python process
(single-Python-thread), not thread-safe across multiple Python threads
that share a Kernel. The refcount-based liveness check between the
``dict.get`` and the ``return`` is not atomic across a thread switch.
Real Helion workloads are GPU-driven on a single Python thread, so this
matches existing usage; thread-sharing callers should opt out via
``HELION_REUSE_OUTPUT_BUFFERS=0``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing_extensions import Self

import torch

log: logging.Logger = logging.getLogger(__name__)

_REUSE_OUTPUT_BUFFERS_ENV = "HELION_REUSE_OUTPUT_BUFFERS"

# Tri-state mode resolved from the env var on first hot-path call:
# 0 = disabled (passthrough), 1 = enabled, 2 = enabled + debug logging.
_MODE_DISABLED = 0
_MODE_ENABLED = 1
_MODE_DEBUG = 2
_pool_mode: int | None = None

# One-time INFO log fired the first time the pool returns a reused buffer.
_pool_intro_logged = False

# Bind the low-level C entry points once at import time so the hot path
# avoids the Python wrapper layer. ``torch.cuda.current_stream(device)`` is
# ~10× slower than ``torch._C._cuda_getCurrentStream(device_index)`` and
# the public ``is_current_stream_capturing`` wrapper has a similar
# overhead profile.
_cuda_is_capturing = getattr(torch._C, "_cuda_isCurrentStreamCapturing", lambda: False)
_cuda_get_current_stream = getattr(
    torch._C, "_cuda_getCurrentStream", lambda _idx: (0, 0, 1)
)

# Cache keyed on (dtype, shape, device, stream_id). Values are strong
# references to the cached tensor; the liveness check uses
# ``sys.getrefcount`` to decide whether external code also holds the
# tensor (in which case we allocate fresh instead of reusing).
_output_pool_safe_cache: dict[
    tuple[torch.dtype, tuple[int, ...], torch.device, int], torch.Tensor
] = {}

# ``sys.getrefcount(x)`` always counts the temporary binding it makes for
# its argument, so a value held by exactly ``(dict slot, local var)``
# reports a refcount of 3. Anything higher means external code holds it.
_OUTPUT_POOL_BASELINE_REFCOUNT = 3


class _pool_active:
    """Clear-on-exit scope for the output pool.

    Activation no longer depends on this scope (the pool is on by default
    unless ``HELION_REUSE_OUTPUT_BUFFERS=0``). The autotuner still wraps
    each trial in ``_pool_active()`` so cached buffers from per-trial
    measurement don't leak across trial sets — entering is a no-op,
    exiting drops the cache.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        _output_pool_safe_cache.clear()


def _resolve_pool_mode() -> int:
    """Read ``HELION_REUSE_OUTPUT_BUFFERS`` once and cache the result."""
    global _pool_mode
    if _pool_mode is None:
        raw = os.environ.get(_REUSE_OUTPUT_BUFFERS_ENV, "").strip().lower()
        if raw in ("0", "off", "false", "no"):
            _pool_mode = _MODE_DISABLED
        elif raw == "debug":
            _pool_mode = _MODE_DEBUG
        else:
            _pool_mode = _MODE_ENABLED
    return _pool_mode


def _maybe_log_pool_intro() -> None:
    """Fire a one-time INFO line the first time the pool reuses a buffer."""
    global _pool_intro_logged
    if _pool_intro_logged:
        return
    _pool_intro_logged = True
    log.info(
        "Helion is reusing kernel-output buffers across calls to lower "
        "launch overhead. Set HELION_REUSE_OUTPUT_BUFFERS=0 to disable "
        "(needed if you hold raw .data_ptr() to outputs across calls or "
        "share a Kernel across Python threads), or =debug to log each "
        "pool decision."
    )


def _output_pool_alloc(
    *shape_args: object,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    **extra_kwargs: object,
) -> torch.Tensor:
    """Pooled replacement for ``torch.empty(*shape, dtype=..., device=...)``.

    Matches ``torch.empty``'s overloaded shape signature: callers can pass
    a single tuple/list ``([1024, 1024])`` or multiple positional dims
    ``(M, N)``. Generated host wrappers emit whichever the user wrote, so
    both shapes need to work transparently. ``dtype`` and ``device`` are
    optional to mirror ``torch.empty``'s defaults; ``extra_kwargs`` is a
    catch-all (e.g. ``pin_memory``) forwarded to ``torch.empty`` on the
    miss path.

    Pooling is on by default; ``HELION_REUSE_OUTPUT_BUFFERS=0`` disables
    it (passthrough to ``torch.empty``) and ``=debug`` adds INFO logging
    for every pool decision. See the module docstring for full details.
    Returns a tensor whose contents are undefined (same contract as
    ``torch.empty``); the kernel is expected to fully write it before the
    caller reads.
    """
    mode = _resolve_pool_mode()
    if mode == _MODE_DISABLED:
        # Fast path: pool disabled. Pure passthrough to ``torch.empty`` so
        # the user sees exact pre-pool semantics.
        kwargs: dict[str, object] = dict(extra_kwargs)
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        return torch.empty(*shape_args, **kwargs)  # type: ignore[arg-type]
    debug = mode == _MODE_DEBUG
    # Normalize ``shape_args`` → a flat tuple of ints (or a tuple containing
    # one tuple/list/torch.Size that we flatten). Mirrors torch.empty's
    # accepted overloads.
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list, torch.Size)):
        shape_t = tuple(shape_args[0])
    else:
        shape_t = tuple(shape_args)  # type: ignore[arg-type]
    # Resolve defaults the same way torch.empty does so the cache key
    # captures the actual realized dtype/device. Use ``get_default_device``
    # rather than ``torch.tensor(0.0).device`` so the device-defaulting
    # branch (theoretical — generated wrappers always pass ``device=``)
    # doesn't allocate a throwaway tensor per call.
    resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
    resolved_device = device if device is not None else torch.get_default_device()
    if isinstance(resolved_device, str):
        resolved_device = torch.device(resolved_device)
    if extra_kwargs:
        # Uncommon: extra kwargs (e.g. ``layout``, ``pin_memory``) change
        # storage characteristics, so don't pool — just delegate.
        if debug:
            log.info(
                "output pool: fresh alloc (extra kwargs %s) shape=%s dtype=%s device=%s",
                sorted(extra_kwargs),
                shape_t,
                resolved_dtype,
                resolved_device,
            )
        # pyrefly: ignore [no-matching-overload]
        return torch.empty(
            shape_t, dtype=resolved_dtype, device=resolved_device, **extra_kwargs
        )
    # Three guards: CG capture bypass, stream-keyed cache, refcount check.
    if resolved_device.type == "cuda":
        # Use the lowest-level C APIs available — the public Python
        # wrappers (``torch.cuda.is_current_stream_capturing``,
        # ``torch.cuda.current_stream``) are ~10× slower per call.
        if _cuda_is_capturing():
            # CUDA graph capture handles buffer reuse at graph-build time;
            # the runtime pool must not interfere.
            if debug:
                log.info(
                    "output pool: bypass (CUDA graph capture) shape=%s dtype=%s device=%s",
                    shape_t,
                    resolved_dtype,
                    resolved_device,
                )
            # pyrefly: ignore [no-matching-overload]
            return torch.empty(shape_t, dtype=resolved_dtype, device=resolved_device)
        device_index = resolved_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        # ``_cuda_getCurrentStream`` returns ``(stream_id, device, raw_id)``;
        # we use the first element as the stream tag.
        stream_id = _cuda_get_current_stream(device_index)[0]
    else:
        # Non-CUDA devices: the stream concept doesn't apply, but we still
        # want pooling per device. Use a constant stream tag.
        stream_id = 0
    key_safe = (resolved_dtype, shape_t, resolved_device, stream_id)
    # pyrefly: ignore [bad-argument-type]
    cached = _output_pool_safe_cache.get(key_safe)
    if cached is not None:
        # sys.getrefcount returns one more than the "real" count because
        # of the temporary binding it creates for its argument. Our pool
        # contributes (dict slot, ``cached`` local) → 2 + 1 (getrefcount
        # arg) = 3 baseline. Anything higher means at least one external
        # holder (direct ref, view via ``_base``, or autograd save).
        if sys.getrefcount(cached) <= _OUTPUT_POOL_BASELINE_REFCOUNT:
            if debug:
                log.info(
                    "output pool: hit shape=%s dtype=%s device=%s stream=%s",
                    shape_t,
                    resolved_dtype,
                    resolved_device,
                    stream_id,
                )
            else:
                _maybe_log_pool_intro()
            return cached
        if debug:
            log.info(
                "output pool: refcount miss (external holder) shape=%s "
                "dtype=%s device=%s stream=%s",
                shape_t,
                resolved_dtype,
                resolved_device,
                stream_id,
            )
    elif debug:
        log.info(
            "output pool: cold miss shape=%s dtype=%s device=%s stream=%s",
            shape_t,
            resolved_dtype,
            resolved_device,
            stream_id,
        )
    # pyrefly: ignore [no-matching-overload]
    fresh = torch.empty(shape_t, dtype=resolved_dtype, device=resolved_device)
    # pyrefly: ignore [unsupported-operation]
    _output_pool_safe_cache[key_safe] = fresh
    return fresh


def output_pool_clear() -> None:
    """Drop all pooled output buffers (for tests that need fresh storage)."""
    _output_pool_safe_cache.clear()


def _reset_output_pool_user_opt_in_cache() -> None:
    """Reset the cached env-var lookup so tests can flip the mode mid-process."""
    global _pool_mode, _pool_intro_logged
    _pool_mode = None
    _pool_intro_logged = False
    output_pool_clear()
