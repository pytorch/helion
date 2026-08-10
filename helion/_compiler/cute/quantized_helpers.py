from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from cutlass import Float16
from cutlass import Float32
from cutlass import Int8
from cutlass import Int16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from ._mlir_compat import ir


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


def _as_i64(
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
    if str(cast("Any", ir_value).type) != "i64":
        raise TypeError(f"expected packed i64 value, got {ir_value.type}")
    return ir_value


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
def fp8e4m3fn_x2_to_float32(
    value: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[Float32, Float32]:
    """Decode two packed e4m3 fp8 bytes (low 16 bits) to (lo_f32, hi_f32).

    Uses a single ``cvt.rn.f16x2.e4m3x2`` to convert both bytes at once, which
    is ~2x cheaper than two scalar ``fp8e4m3fn_to_float32`` calls.
    """
    value_i16 = _as_i16(value, loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [Float32.mlir_type, Float32.mlir_type]
        ),
        [value_i16],
        """
        {
          .reg .b16 v_lo, v_hi;
          .reg .b32 v_h2;
          cvt.rn.f16x2.e4m3x2 v_h2, $2;
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


@dsl_user_op
def float4_e2m1fn_x16_to_float16(
    value: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[Float16, ...]:
    """Decode one packed 64-bit E2M1 group into sixteen FP16 values."""
    value_i64 = _as_i64(value, loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [Float16.mlir_type] * 16
        ),
        [value_i64],
        """
        {
          .reg .b32 lo, hi;
          .reg .b8 c0,c1,c2,c3,c4,c5,c6,c7;
          .reg .b32 h0,h1,h2,h3,h4,h5,h6,h7;
          mov.b64 {lo, hi}, $16;
          mov.b32 {c0,c1,c2,c3}, lo;
          cvt.rn.f16x2.e2m1x2 h0, c0; cvt.rn.f16x2.e2m1x2 h1, c1;
          cvt.rn.f16x2.e2m1x2 h2, c2; cvt.rn.f16x2.e2m1x2 h3, c3;
          mov.b32 {c4,c5,c6,c7}, hi;
          cvt.rn.f16x2.e2m1x2 h4, c4; cvt.rn.f16x2.e2m1x2 h5, c5;
          cvt.rn.f16x2.e2m1x2 h6, c6; cvt.rn.f16x2.e2m1x2 h7, c7;
          mov.b32 {$0,$1}, h0; mov.b32 {$2,$3}, h1;
          mov.b32 {$4,$5}, h2; mov.b32 {$6,$7}, h3;
          mov.b32 {$8,$9}, h4; mov.b32 {$10,$11}, h5;
          mov.b32 {$12,$13}, h6; mov.b32 {$14,$15}, h7;
        }
        """,
        ",".join(["=h"] * 16 + ["l"]),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float16(llvm.extractvalue(Float16.mlir_type, result, [i], loc=loc, ip=ip))
        for i in range(16)
    )


@dsl_user_op
def bfloat16_x16_to_float16(
    qword0: object,
    qword1: object,
    qword2: object,
    qword3: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[Float16, ...]:
    """Convert four packed BF16 qwords into sixteen FP16 values."""
    values_i64 = [
        _as_i64(value, loc=loc, ip=ip) for value in (qword0, qword1, qword2, qword3)
    ]
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [Float16.mlir_type] * 16
        ),
        values_i64,
        """
        {
          .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7;
          .reg .b16 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;
          mov.b64 {w0,w1}, $16; mov.b64 {w2,w3}, $17;
          mov.b64 {w4,w5}, $18; mov.b64 {w6,w7}, $19;
          mov.b32 {b0,b1}, w0; mov.b32 {b2,b3}, w1;
          mov.b32 {b4,b5}, w2; mov.b32 {b6,b7}, w3;
          mov.b32 {b8,b9}, w4; mov.b32 {b10,b11}, w5;
          mov.b32 {b12,b13}, w6; mov.b32 {b14,b15}, w7;
          cvt.rn.f16.bf16 $0, b0; cvt.rn.f16.bf16 $1, b1;
          cvt.rn.f16.bf16 $2, b2; cvt.rn.f16.bf16 $3, b3;
          cvt.rn.f16.bf16 $4, b4; cvt.rn.f16.bf16 $5, b5;
          cvt.rn.f16.bf16 $6, b6; cvt.rn.f16.bf16 $7, b7;
          cvt.rn.f16.bf16 $8, b8; cvt.rn.f16.bf16 $9, b9;
          cvt.rn.f16.bf16 $10, b10; cvt.rn.f16.bf16 $11, b11;
          cvt.rn.f16.bf16 $12, b12; cvt.rn.f16.bf16 $13, b13;
          cvt.rn.f16.bf16 $14, b14; cvt.rn.f16.bf16 $15, b15;
        }
        """,
        ",".join(["=h"] * 16 + ["l"] * 4),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float16(llvm.extractvalue(Float16.mlir_type, result, [i], loc=loc, ip=ip))
        for i in range(16)
    )
