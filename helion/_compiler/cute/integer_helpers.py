from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from cutlass import Integer
from cutlass._mlir.dialects import arith
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from ._mlir_compat import ir


@dsl_user_op
def python_mod(
    lhs: object,
    rhs: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> object:
    """Integer remainder with the divisor's sign, without signed overflow.

    CuTe's integer ``%`` emits truncating ``arith.remsi``. Python and Torch
    remainder instead follow floor division. Preserve CuTe's operand promotion,
    then correct only a nonzero remainder whose sign differs from the divisor.
    """
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs % rhs
    # Use the public arithmetic promotion rule, without narrowing a Python
    # constant or a wider operand. This unused addition is removed by lowering.
    promoted = cast("Any", lhs) + rhs
    if not isinstance(promoted, Integer):
        raise TypeError("integer modulo requires integer operands")
    dtype = type(promoted)
    left, right = dtype(lhs), dtype(rhs)
    if not dtype.signed:
        return left % right
    # INT_MIN % -1 is mathematically zero, but raw remsi can be poison. Using
    # +1 for this divisor avoids the exceptional division before any select.
    safe_right = dtype(
        arith.select(
            (right == dtype(-1)).ir_value(loc=loc, ip=ip),
            dtype(1).ir_value(loc=loc, ip=ip),
            right.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )
    remainder = left % safe_right
    correct = (remainder != dtype(0)) & ((remainder < dtype(0)) != (right < dtype(0)))
    # When selected, operands have opposite signs, so this addition cannot
    # overflow. Its unselected value has ordinary wrapping addi semantics.
    return dtype(
        arith.select(
            correct.ir_value(loc=loc, ip=ip),
            (remainder + right).ir_value(loc=loc, ip=ip),
            remainder.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )
