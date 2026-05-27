"""Optional native (C) accelerators for Helion's hot paths.

This package is the integration point for a compiled C extension that
re-implements a handful of per-call Python operations in C — most
notably the per-tensor specialization-key build used by
:meth:`Kernel.bind` and the launcher dispatch itself. None of those
C accelerators are in tree yet; this module is the stable Python-side
contract they will plug into.

Layout
------
- ``_native`` (to be added in a later commit) — the compiled extension.
- This ``__init__.py`` exposes ``AVAILABLE`` and the public function
  names. When ``_native`` is importable, the names resolve to its C
  implementations. When it isn't, they resolve to ``None`` and callers
  must use the pure-Python fallback.

Caller pattern
--------------
::

    from helion import _C

    def tensor_key(fn, obj):
        if _C.tensor_key is not None:
            key = _C.tensor_key(obj, fn.settings.static_shapes)
            if key is not None:
                return key
        # ... fall back to pure-Python implementation ...

This two-level guard (module ``AVAILABLE`` flag plus per-call
``None`` sentinel) lets the C path bail out on unsupported inputs
(SymInts, fake tensors) without losing the per-call C speedup on the
common path.

Why an optional extension
-------------------------
Helion ships as a pure-Python wheel via ``hatchling``. A failed C
build, an unsupported PyTorch ABI, or a non-CPython interpreter must
not break ``import helion``. The Python fallback is the source of
truth; the C accelerator is a strict subset that fast-paths the
common case.
"""

from __future__ import annotations

import logging

log: logging.Logger = logging.getLogger(__name__)

# Set to ``True`` after a successful import of ``_native``; ``False``
# when the extension is missing or failed to load. Callers should
# check this AND the individual symbol (which may be ``None`` for
# functions the extension didn't provide).
AVAILABLE: bool = False

# Public function slots. Re-bound below if the extension imports.
# Each is either a callable (C-backed) or ``None`` (use Python path).
tensor_key = None

# Minimal Chunk-E C launcher: a Python type with a ``tp_call`` slot
# that dispatches a Triton kernel directly into ``compiled_kernel.run``,
# bypassing both the Python ``default_launcher`` frame AND Triton's
# ``JITFunction.run`` pipeline. Caller is responsible for priming the
# launcher via ``CompiledLauncher.prime(triton_kernel, grid, args, ...)``
# before installing it in place of the wrapper's ``_default_launcher``
# kwdefault. ``None`` if the extension isn't built — callers must use
# the pure-Python ``default_launcher`` in that case.
CompiledLauncher = None

try:
    from . import _native  # type: ignore[attr-defined]
except ImportError as e:
    log.debug("helion._C._native not available: %s", e)
else:
    AVAILABLE = True
    tensor_key = getattr(_native, "tensor_key", None)

try:
    from . import _launcher  # type: ignore[attr-defined]
except ImportError as e:
    log.debug("helion._C._launcher not available: %s", e)
else:
    CompiledLauncher = getattr(_launcher, "CompiledLauncher", None)


__all__ = ["AVAILABLE", "CompiledLauncher", "tensor_key"]
