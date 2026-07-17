"""CuTe-backend codegen for ops defined in ``helion.language.quantized_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``quantized_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(float4_e2m1fn_x2_to_float32, "cute")
def _(state: CodegenState) -> list[ast.AST]:
    call = expr_from_string(
        "_cute_float4_e2m1fn_x2_to_float32({value})",
        value=state.ast_arg(0),
    )
    result = state.codegen.lift(call, dce=True, prefix="fp4_pair")
    return [
        expr_from_string(f"{result.id}[0]"),
        expr_from_string(f"{result.id}[1]"),
    ]
