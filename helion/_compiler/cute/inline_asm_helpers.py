from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from cutlass._mlir import ir


@dsl_user_op
def inline_asm_elementwise(
    args: tuple[object, ...],
    *,
    asm: str = "",
    constraints: str = "",
    dtype: object = None,
    is_pure: bool = True,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> object:
    """CuTe scalar helper for ``hl.inline_asm_elementwise``.

    Helion's CuTe SIMT lowering operates on one scalar per thread.  The public
    API's tensor elementwise semantics are therefore implemented by calling this
    helper once per generated scalar lane.
    """

    operands = []
    for arg in args:
        operand = cast("Any", arg).ir_value(loc=loc, ip=ip)
        if str(operand.type) in {"i1", "i8", "i16"}:
            operand = llvm.zext(Int32.mlir_type, operand, loc=loc, ip=ip)
        operands.append(operand)
    if isinstance(dtype, tuple):
        result = llvm.inline_asm(
            llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
                [cast("Any", dt).mlir_type for dt in dtype]
            ),
            operands,
            asm,
            constraints,
            has_side_effects=not is_pure,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return tuple(
            cast("Any", dt)(
                llvm.extractvalue(
                    cast("Any", dt).mlir_type, result, [i], loc=loc, ip=ip
                )
            )
            for i, dt in enumerate(dtype)
        )
    dtype = cast("Any", dtype)
    return dtype(
        llvm.inline_asm(
            dtype.mlir_type,
            operands,
            asm,
            constraints,
            has_side_effects=not is_pure,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
