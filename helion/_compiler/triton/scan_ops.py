"""Triton-backend codegen for ops defined in ``helion.language.scan_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``scan_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ...language import _decorators
from ...language.scan_ops import _associative_scan

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(_associative_scan, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate code for associative scan with combine function."""
    dim = state.proxy_arg(2)
    reverse = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    input_tensor = state.ast_arg(1)
    helper_func_name = state.register_helper_function(0)
    scan_expr = _create_scan_expression(
        input_tensor, cast("int", dim), helper_func_name, bool(reverse)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, scan_expr)
    return scan_expr


def _create_scan_expression(
    input_tensor: ast.AST, dim: int, helper_func_name: str, reverse: bool
) -> ast.AST:
    """Create the tl.associative_scan expression."""
    from ..ast_extension import expr_from_string

    template = (
        f"tl.associative_scan({{input_tensor}}, {{dim_value}}, {helper_func_name}, reverse=True)"
        if reverse
        else f"tl.associative_scan({{input_tensor}}, {{dim_value}}, {helper_func_name})"
    )
    return expr_from_string(
        template,
        input_tensor=input_tensor,
        dim_value=ast.Constant(value=dim),
    )


def _create_tuple_result_expressions(
    state: CodegenState, scan_expr: ast.AST
) -> list[ast.AST]:
    """Create getitem expressions for tuple results."""
    from ..ast_extension import expr_from_string

    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(
            "{scan_result}[{index}]", scan_result=scan_expr, index=ast.Constant(value=i)
        )
        for i in range(num_elements)
    ]
