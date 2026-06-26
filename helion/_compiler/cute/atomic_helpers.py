# pyrefly: ignore-errors
"""Float atomic max/min helpers for the CuTe backend.

NVVM/PTX has no native ``atom.max``/``atom.min`` for floating-point types
(only integer types support these). The CUTLASS DSL's ``cute.arch.atomic_max``
/ ``cute.arch.atomic_min`` therefore lower a float operand to an invalid
``nvvm.atomicrmw (f32, max)`` combination, which aborts the process with
``LLVM ERROR: Invalid AtomicType and Op combination for nvvm.atomicrmw``.

The standard workaround (used by CUDA's ``atomicMax(float*)`` emulation and by
CUTLASS's own :func:`cute.arch.atomic_fmax`) reinterprets the float bit pattern
as an integer and dispatches to the *integer* atomics, which are natively
supported:

  * For a non-negative ``val`` (sign bit clear) the IEEE-754 bit pattern is
    monotonic when read as a signed ``i32``, so a signed integer atomic
    reproduces the float comparison.
  * For a negative ``val`` (sign bit set) the ordering inverts, so we flip
    max<->min and operate on the *unsigned* integer representation.

This handles mixed-sign contents of the target cell correctly (the same
property that makes the CUDA idiom correct), and emits only integer atomics
that the CuTe backend already supports.

Only float32 is implemented here; that is what Helion's float atomic
max/min lowering needs for the test suite. 16-bit float atomics would need a
sub-word RMW (CAS on the enclosing 32-bit word) and are intentionally left
unsupported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
from cutlass import Float32
from cutlass import Int32
from cutlass import Uint32
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from cutlass._mlir import ir


@dsl_user_op
def atomic_max_float32(
    ptr: cute.Pointer,
    val: Float32,
    *,
    sem: str = "relaxed",
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    """Atomic float32 max via integer-bitcast dispatch to integer atomics.

    Returns the previous float32 value stored at ``ptr``.
    """
    intval = llvm.bitcast(Int32.mlir_type, val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

    def neg_body() -> Int32:
        # Negative val: float order inverts, use unsigned min.
        return Int32(
            Uint32(
                cute.arch.atomic_min(ptr, Uint32(intval), sem=sem, loc=loc, ip=ip)
            ).ir_value(loc=loc, ip=ip)
        )

    def nonneg_body() -> Int32:
        # Non-negative val: signed order matches float order, use signed max.
        return cute.arch.atomic_max(ptr, Int32(intval), sem=sem, loc=loc, ip=ip)

    old_intval = cutlass.cutlass_dsl.if_generate(
        Int32(intval) < Int32(0),
        neg_body,
        nonneg_body,
        [],
        [Int32],
        loc=loc,
        ip=ip,
    )
    assert not isinstance(old_intval, list)
    return Float32(
        llvm.bitcast(
            Float32.mlir_type, old_intval.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def atomic_min_float32(
    ptr: cute.Pointer,
    val: Float32,
    *,
    sem: str = "relaxed",
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    """Atomic float32 min via integer-bitcast dispatch to integer atomics.

    Returns the previous float32 value stored at ``ptr``.
    """
    intval = llvm.bitcast(Int32.mlir_type, val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

    def neg_body() -> Int32:
        # Negative val: float order inverts, use unsigned max.
        return Int32(
            Uint32(
                cute.arch.atomic_max(ptr, Uint32(intval), sem=sem, loc=loc, ip=ip)
            ).ir_value(loc=loc, ip=ip)
        )

    def nonneg_body() -> Int32:
        # Non-negative val: signed order matches float order, use signed min.
        return cute.arch.atomic_min(ptr, Int32(intval), sem=sem, loc=loc, ip=ip)

    old_intval = cutlass.cutlass_dsl.if_generate(
        Int32(intval) < Int32(0),
        neg_body,
        nonneg_body,
        [],
        [Int32],
        loc=loc,
        ip=ip,
    )
    assert not isinstance(old_intval, list)
    return Float32(
        llvm.bitcast(
            Float32.mlir_type, old_intval.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )
