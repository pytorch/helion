"""Optional native (Rust) accelerators for Helion's hot paths.

This package is the integration point for a compiled extension that
re-implements a handful of per-call Python operations in Rust (built
with `PyO3 <https://pyo3.rs>`_) — most notably the launcher dispatch
itself. None of those accelerators are in tree yet; this module is the
stable Python-side contract they will plug into.

Layout
------
- ``_launcher`` (added in a later commit) — the compiled Rust
  extension.
- This ``__init__.py`` exposes ``AVAILABLE`` and the public function
  names. When the extension is importable, the names resolve to its
  Rust implementations. When it isn't, they resolve to ``None`` and
  callers must use the pure-Python fallback.

Caller pattern
--------------
::

    from helion import _native

    if _native.CompiledLauncher is not None:
        launcher = _native.CompiledLauncher()
        launcher.prime(...)
        # install in place of the Python ``default_launcher``
    # otherwise the Python ``default_launcher`` keeps running.

This two-level guard (module ``AVAILABLE`` flag plus per-symbol
``None`` sentinel) lets the native path bail out on unsupported
configurations (e.g. ABI mismatch, missing toolchain) without losing
the per-call speedup on the common path.

Why an optional extension
-------------------------
Helion ships as a pure-Python wheel via ``hatchling``. A failed Rust
build, an unsupported PyTorch ABI, or a non-CPython interpreter must
not break ``import helion``. The Python fallback is the source of
truth; the Rust accelerator is a strict subset that fast-paths the
common case.

Why Rust (not C)
----------------
The launcher is a small, performance-sensitive piece of code with a
narrow Python C-API surface. Rust + PyO3 gives the same access to the
CPython C API with:

- Memory safety (no leaked references on error paths — PyO3's
  ``PyResult`` / ``Bound`` types handle that).
- A standard build system (``cargo``) instead of an ad-hoc
  ``gcc`` one-liner.

Per-call performance is essentially identical to hand-written C: both
go through the same CPython entry points for tensor metadata reads
and Triton kernel dispatch, and PyO3's macro-generated trampolines
add negligible overhead on the hot path.
"""

from __future__ import annotations

import logging

log: logging.Logger = logging.getLogger(__name__)

# Set to ``True`` after a successful import of the compiled extension;
# ``False`` when it is missing or failed to load. Callers should check
# this AND the individual symbol (which may be ``None`` for functions
# the extension didn't provide).
AVAILABLE: bool = False

# Public function slots. Re-bound below if the extension imports.
# Each is either a callable (Rust-backed) or ``None`` (use Python).

# Reserved slot for a future per-tensor specialization-key
# accelerator. The earlier C experiment found only ~110 ns per call
# of saving (within run-to-run variance for typical kernels), so this
# slot is intentionally left unused until a path through the PyTorch
# stable ABI can do better. ``None`` means: use the pure-Python
# ``_tensor_key`` in ``helion/runtime/kernel.py``.
tensor_key = None

# Minimal Chunk-E Rust launcher: a Python type with a ``__call__``
# slot that dispatches a Triton kernel directly into
# ``compiled_kernel.run``, bypassing both the Python
# ``default_launcher`` frame AND Triton's ``JITFunction.run``
# pipeline. Caller is responsible for priming the launcher via
# ``CompiledLauncher.prime(triton_kernel, grid, args, ...)`` before
# installing it in place of the wrapper's ``_default_launcher``
# kwdefault. ``None`` if the extension isn't built — callers must use
# the pure-Python ``default_launcher`` in that case.
CompiledLauncher = None

try:
    from . import _launcher  # type: ignore[attr-defined]
except ImportError as e:
    log.debug("helion._native._launcher not available: %s", e)
else:
    AVAILABLE = True
    CompiledLauncher = getattr(_launcher, "CompiledLauncher", None)


__all__ = ["AVAILABLE", "CompiledLauncher", "tensor_key"]
