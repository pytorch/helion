"""The CuTe backend's torch -> CUTLASS dtype table, declared once.

``CUTLASS_DTYPE_NAME`` is the spelling generated code gives a register
value of each torch dtype (``CuteBackend.dtype_str`` agrees on every dtype
both know).  ``CUTLASS_POINTER_DTYPE_NAME`` is the element type the launcher
builds a global-memory pointer with; it differs on ``torch.bool`` only: CuTe
has no i1 global-memory tensors and torch stores a bool in one byte, so bool
tensor pointers travel as ``Uint8`` and the load lowering converts nonzero
bytes back to ``cutlass.Boolean`` registers.  Both tables agree that the
packed ``float4_e2m1fn_x2`` shell dtype is raw byte storage and that
``uint64`` travels as ``Int64`` (CuTe has no unsigned 64-bit pointer type).

The tables hold names, not ``cutlass`` objects, so importing this module
never needs the CuTe DSL; ``cutlass_pointer_dtype`` resolves the object
lazily.  The Helion-free standalone launcher carries a frozen copy of the
pointer table keyed by ``str(dtype)``
(``helion/runtime/cute/standalone_launcher.py``); a test keeps the two equal.
"""

from __future__ import annotations

import torch

from ... import exc

CUTLASS_DTYPE_NAME: dict[torch.dtype, str] = {
    torch.float16: "cutlass.Float16",
    torch.float32: "cutlass.Float32",
    torch.float64: "cutlass.Float64",
    torch.bfloat16: "cutlass.BFloat16",
    torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
    torch.float8_e5m2: "cutlass.Float8E5M2",
    torch.float8_e8m0fnu: "cutlass.Float8E8M0FNU",
    torch.float4_e2m1fn_x2: "cutlass.Uint8",
    torch.bool: "cutlass.Boolean",
    torch.int8: "cutlass.Int8",
    torch.int16: "cutlass.Int16",
    torch.int32: "cutlass.Int32",
    torch.int64: "cutlass.Int64",
    torch.uint8: "cutlass.Uint8",
    torch.uint32: "cutlass.Uint32",
    torch.uint64: "cutlass.Int64",
}

CUTLASS_POINTER_DTYPE_NAME: dict[torch.dtype, str] = {
    **CUTLASS_DTYPE_NAME,
    torch.bool: "cutlass.Uint8",
}

_CUTLASS_POINTER_DTYPES: dict[torch.dtype, object] | None = None


def cutlass_pointer_dtype(dtype: torch.dtype) -> object:
    """The ``cutlass`` element type of a global-memory pointer to ``dtype``."""
    global _CUTLASS_POINTER_DTYPES
    mapping = _CUTLASS_POINTER_DTYPES
    if mapping is None:
        import cutlass

        mapping = {
            dtype: getattr(cutlass, name.removeprefix("cutlass."))
            for dtype, name in CUTLASS_POINTER_DTYPE_NAME.items()
        }
        _CUTLASS_POINTER_DTYPES = mapping
    cutlass_dtype = mapping.get(dtype)
    if cutlass_dtype is None:
        raise exc.BackendUnsupported("cute", f"dtype: {dtype}")
    return cutlass_dtype
