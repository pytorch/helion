from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from cutlass import Float32
from cutlass import Int8
from cutlass import Int16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from cutlass._mlir import ir


def _as_i16(
    value: object,
    *,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> ir.Value:
    raw_value = getattr(value, "value", None)
    if hasattr(raw_value, "type"):
        ir_value = cast("Any", raw_value)
    else:
        ir_value = cast("Any", value).ir_value(loc=loc, ip=ip)
    ir_type = str(cast("Any", ir_value).type)
    if ir_type == "i8":
        return llvm.zext(Int16.mlir_type, ir_value, loc=loc, ip=ip)
    if ir_type.startswith("f8"):
        as_i8 = llvm.bitcast(Int8.mlir_type, ir_value, loc=loc, ip=ip)
        return llvm.zext(Int16.mlir_type, as_i8, loc=loc, ip=ip)
    if ir_type == "i16":
        return ir_value
    raise TypeError(f"unsupported quantized scalar type: {ir_type}")


@dsl_user_op
def fp8e4m3fn_to_float32(
    value: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    value_i16 = _as_i16(value, loc=loc, ip=ip)
    result = llvm.inline_asm(
        Float32.mlir_type,
        [value_i16],
        """
        {
          .reg .b16 scale_lo, scale_hi;
          .reg .b32 scale_h2;
          cvt.rn.f16x2.e4m3x2 scale_h2, $1;
          mov.b32 {scale_lo, scale_hi}, scale_h2;
          cvt.f32.f16 $0, scale_lo;
        }
        """,
        "=f,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def float4_e2m1fn_x2_to_float32(
    value: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[Float32, Float32]:
    value_i16 = _as_i16(value, loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [Float32.mlir_type, Float32.mlir_type]
        ),
        [value_i16],
        """
        {
          .reg .b8 v;
          .reg .b16 v_lo, v_hi;
          .reg .b32 v_h2;
          mov.b16 {v, _}, $2;
          cvt.rn.f16x2.e2m1x2 v_h2, v;
          mov.b32 {v_lo, v_hi}, v_h2;
          cvt.f32.f16 $0, v_lo;
          cvt.f32.f16 $1, v_hi;
        }
        """,
        "=f,=f,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(Float32.mlir_type, result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(Float32.mlir_type, result, [1], loc=loc, ip=ip)),
    )
