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

``promotion_operand`` and ``narrow_common_dtype`` say which dtype torch
computes a binary op in, for the fused-body emitters that must round operands
to a 16-bit common dtype as torch does, ``round_scalar`` rounds a Python
scalar to a dtype at codegen time, and ``float_literal`` spells such a value
in generated source.
"""

from __future__ import annotations

import math

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Node

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

# The integer and boolean scalar types as generated code spells them: every
# type above that is not a float, and the unsigned 16- and 64-bit carriers no
# torch dtype maps to.
CUTLASS_INTEGER_TYPE_NAMES = frozenset(
    f"cutlass.{name}"
    for name in (
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
    )
)

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


def promotion_operand(arg: object) -> torch.Tensor | bool | int | float | None:
    """``arg``, an FX node or a Python value, as ``torch.result_type`` sees
    it: the node's tensor, or a number (a symbolic scalar counts as one of
    its category); None for anything else, such as a list of dims."""
    value = arg.meta.get("val") if isinstance(arg, Node) else arg
    if isinstance(value, (torch.Tensor, bool, int, float)):
        return value
    if isinstance(value, torch.SymBool):
        return True
    if isinstance(value, torch.SymInt):
        return 1
    if isinstance(value, torch.SymFloat):
        return 1.0
    return None


def narrow_common_dtype(lhs: object, rhs: object) -> torch.dtype | None:
    """The 16-bit float dtype torch computes a binary op of ``lhs`` and ``rhs``
    (FX nodes or Python values) in, else None (another dtype, or arguments
    that are not a tensor beside a tensor or number).  Torch converts a
    tensor operand of another dtype to it first (a 0-d fp32 tensor beside
    bf16), and a compare converts a Python scalar too (``bf16 > 0.3`` tests
    against ``bf16(0.3)``)."""
    a = promotion_operand(lhs)
    b = promotion_operand(rhs)
    if (
        a is None
        or b is None
        or not (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor))
    ):
        return None
    common = torch.result_type(a, b)
    return common if common in (torch.float16, torch.bfloat16) else None


def round_scalar(value: float, dtype: torch.dtype) -> float:
    """``value`` rounded to ``dtype`` at codegen time, to nearest even as the
    device conversion rounds (inf past the type's range).  The tensor is a
    real CPU one even while a kernel binds, under fake-tensor mode and a
    meta default device, where its ``item()`` would be an unbacked symbol
    or fail."""
    with unset_fake_temporarily():
        return float(torch.tensor(float(value), dtype=dtype, device="cpu").item())


def float_literal(value: float) -> str:
    """Python source for the float ``value``: ``repr`` spells inf and nan as
    bare names, undefined in the generated module, so those are
    ``float(...)`` calls."""
    return repr(value) if math.isfinite(value) else f"float('{value}')"
