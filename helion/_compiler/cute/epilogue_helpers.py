from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from cutlass import Float32
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from helion.language._gelu_tanh_approx import GELU_ERF_INV_SQRT2

if TYPE_CHECKING:
    from ._mlir_compat import ir

# Heuristic compile-time cap: the target tcgen05 epilogue fragments are
# well below this, while larger pointwise tiles should avoid thousands
# of Python-unrolled MLIR ops.
_GELU_ERF_PACKED_MAX_UNROLLED_ELEMENTS = 512
_LOG2_E = 1.4426950408889634


def _rcp_approx_ftz_impl(
    x: object,
    *,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x).ir_value(  # pyrefly: ignore[bad-argument-type]
                    loc=loc, ip=ip
                )
            ],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp_approx_ftz(
    x: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    """Approximate f32 reciprocal using the native CUDA instruction."""
    return _rcp_approx_ftz_impl(x, loc=loc, ip=ip)


@dsl_user_op
def sigmoid_approx_ftz_f32(
    x: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    """FP32 sigmoid with the native approximate reciprocal instruction."""
    value = Float32(x)  # pyrefly: ignore[bad-argument-type]
    return _rcp_approx_ftz_impl(
        1.0 + cute.math.exp2(-value * _LOG2_E),
        loc=loc,
        ip=ip,
    )


def _gelu_erf_exact_expr(x: Float32 | cute.TensorSSA) -> Float32 | cute.TensorSSA:
    return 0.5 * x * (1.0 + cute.math.erf(x * GELU_ERF_INV_SQRT2))


@dsl_user_op
def gelu_erf_exact_f32x2(
    x: Float32 | cute.TensorSSA,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32 | cute.TensorSSA:
    """Exact GELU with packed f32x2 arithmetic for fp32 TensorSSA inputs."""
    if const_expr(not isinstance(x, cute.TensorSSA)):
        return _gelu_erf_exact_expr(x)

    # pyrefly does not narrow through ``const_expr``; the runtime guard
    # above keeps the generated path on TensorSSA values only.
    x_tensor = cast("cute.TensorSSA", x)
    element_count = cute.size(x_tensor.shape)
    if const_expr(
        x_tensor.dtype is not Float32
        or element_count < 2
        or element_count % 2 != 0
        or element_count > _GELU_ERF_PACKED_MAX_UNROLLED_ELEMENTS
    ):
        return _gelu_erf_exact_expr(x_tensor)

    values = []
    for i in range(element_count // 2):
        x_pair = (x_tensor[2 * i], x_tensor[2 * i + 1])
        scaled = cute.arch.mul_packed_f32x2(
            x_pair, (GELU_ERF_INV_SQRT2, GELU_ERF_INV_SQRT2)
        )
        erf_scaled = (cute.math.erf(scaled[0]), cute.math.erf(scaled[1]))
        fused = cute.arch.fma_packed_f32x2(erf_scaled, x_pair, x_pair)
        out_pair = cute.arch.mul_packed_f32x2((0.5, 0.5), fused)
        values.extend((out_pair[0], out_pair[1]))

    vec = vector.from_elements(
        T.vector(element_count, T.f32()),
        tuple(value.ir_value(loc=loc, ip=ip) for value in values),
        loc=loc,
        ip=ip,
    )
    # pyrefly: ignore [unexpected-keyword]
    return cute.TensorSSA(vec, x_tensor.shape, dtype=Float32, loc=loc, ip=ip)
