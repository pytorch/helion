"""Original half-to-Float32 conversion behind an instruction-selection boundary.

The i16 bitcast supplies the inline-assembly half-register ABI, unchanged bit for
bit. Only cvt performs a numeric conversion. In particular this is not an FMA,
reassociation, FTZ, approximate conversion, or NaN-payload preservation policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from cutlass._mlir import ir


@dsl_user_op
def bfloat16_to_float32(
    value: cutlass.BFloat16,
    *,
    loc: ir.Location | None = None,  # pyrefly: ignore [missing-attribute]
    ip: ir.InsertionPoint | None = None,  # pyrefly: ignore [missing-attribute]
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                llvm.bitcast(
                    cutlass.Int16.mlir_type,
                    value.ir_value(loc=loc, ip=ip),
                    loc=loc,
                    ip=ip,
                )
            ],
            "cvt.f32.bf16 $0, $1;",
            "=f,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def float16_to_float32(
    value: cutlass.Float16,
    *,
    loc: ir.Location | None = None,  # pyrefly: ignore [missing-attribute]
    ip: ir.InsertionPoint | None = None,  # pyrefly: ignore [missing-attribute]
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                llvm.bitcast(
                    cutlass.Int16.mlir_type,
                    value.ir_value(loc=loc, ip=ip),
                    loc=loc,
                    ip=ip,
                )
            ],
            "cvt.f32.f16 $0, $1;",
            "=f,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
