# pyrefly: ignore-errors
"""Vector pack/store helpers for CuTe codegen.

Called during ``@cute.kernel`` tracing (plain Python; builds MLIR IR
directly), so no ``@cute.jit`` wrapper is needed.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import vector as _vector_dialect
import cutlass.cute as cute


def store_u16_vec(ptr: object, vals: list) -> None:
    """Pack ``len(vals)`` ``cutlass.Uint16`` scalars into one vector value
    and store it through ``ptr`` (a ``cute.Pointer``), emitting a single
    ST.32/ST.64/ST.128 instead of per-element 2-byte stores.

    ``vals`` is a compile-time Python list collected across an unrolled
    ``cutlass.range_constexpr(V)`` lane loop.
    """
    vecty = ir.VectorType.get([len(vals)], cutlass.Uint16.mlir_type)
    packed = _vector_dialect.from_elements(vecty, [v.ir_value() for v in vals])
    cute.arch.store(ptr, packed)


def store_u32_vec(ptr: object, vals: list) -> None:
    """``store_u16_vec`` for ``cutlass.Uint32`` lanes (fp32 stores bitcast
    their values to Uint32 first): one ST.64/ST.128 instead of per-element
    4-byte stores."""
    vecty = ir.VectorType.get([len(vals)], cutlass.Uint32.mlir_type)
    packed = _vector_dialect.from_elements(vecty, [v.ir_value() for v in vals])
    cute.arch.store(ptr, packed)
