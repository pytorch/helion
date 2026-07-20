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

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    reverse = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    input_tensor = _get_input_tensor_ast(state, bool(is_tuple_input))
    helper_func_name = _register_helper_function(state, cast("int", combine_graph_id))
    scan_expr = _create_scan_expression(
        input_tensor, cast("int", dim), helper_func_name, bool(reverse)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, scan_expr)
    return scan_expr


def _get_input_tensor_ast(state: CodegenState, is_tuple_input: bool) -> ast.AST:
    """Get the input tensor AST, handling tuple inputs specially."""
    if not is_tuple_input:
        return state.ast_arg(1)

    raw_input = state.ast_args[1]
    if isinstance(raw_input, tuple):
        from ..ast_extension import create

        tuple_elts = [
            elt if isinstance(elt, ast.AST) else ast.Constant(value=elt)
            for elt in raw_input
        ]
        return create(ast.Tuple, elts=tuple_elts, ctx=ast.Load())
    return state.ast_arg(1)


def _register_helper_function(state: CodegenState, combine_graph_id: int) -> str:
    """Register the helper function and return its final name."""
    from ..device_ir import HelperFunctionGraphInfo

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    state.codegen.device_function.register_helper_function(helper_graph_info)
    # Get the final name from the helper manager (which uses the namespace)
    return state.codegen.device_function.helper_manager.get_final_name(
        helper_graph_info
    )


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
