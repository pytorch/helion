# pyrefly: ignore-errors
"""Device helpers for the SIMT matmul fold over a synthetic-lane K axis.

``emit_cute_synthetic_lane_fold_mm`` lowers a matmul whose contraction axis is
split across a per-thread lane loop by re-reading both operands over the full
K extent.  Emitting that K loop inline makes the enclosing lane loop's body a
nested loop, which the lane-reduction splitter (``tile_strategy``) cannot
stage: it accepts straight-line assignments only.  The helper below owns the
K loop, so the fold is one assignment at the call site, and gives the raw
e4m3 byte decode one home.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .quantized_helpers import fp8e4m3fn_to_float32


@cute.jit
def fold_dot(
    lhs_ptr: cute.Pointer,
    lhs_step: cutlass.Integer,
    rhs_ptr: cute.Pointer,
    rhs_step: cutlass.Integer,
    valid: cutlass.Boolean,
    identity: cute.Numeric,
    *,
    k: cutlass.Constexpr[int],
    lhs_e4m3: cutlass.Constexpr[bool],
    rhs_e4m3: cutlass.Constexpr[bool],
) -> cute.Numeric:
    """``sum_i lhs[i] * rhs[i]`` over ``k`` elements in ``identity``'s dtype.

    ``lhs_ptr``/``rhs_ptr`` address element 0 of each operand's K axis and
    ``*_step`` is that axis's element stride in the kernel's index type
    (``Int32``, or ``Int64`` once an input passes 2**31 elements).  Nothing is
    read when ``valid``
    is false (a lane past a free-axis extent): the result is ``identity``
    (zero).  e4m3 storage is read as raw bytes and decoded bit-exactly; other
    element types widen with the ordinary numeric cast.
    """
    dtype = type(identity)
    acc = identity
    if valid:
        for i in range(k):
            if cutlass.const_expr(lhs_e4m3):
                a = dtype(
                    fp8e4m3fn_to_float32(
                        cute.arch.load(lhs_ptr + i * lhs_step, cutlass.Uint8)
                    )
                )
            else:
                a = dtype((lhs_ptr + i * lhs_step).load())
            if cutlass.const_expr(rhs_e4m3):
                b = dtype(
                    fp8e4m3fn_to_float32(
                        cute.arch.load(rhs_ptr + i * rhs_step, cutlass.Uint8)
                    )
                )
            else:
                b = dtype((rhs_ptr + i * rhs_step).load())
            acc = acc + a * b
    return acc
