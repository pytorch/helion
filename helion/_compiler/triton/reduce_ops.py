"""Triton-backend codegen for ops defined in ``helion.language.reduce_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``reduce_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ...language import _decorators
from ...language.reduce_ops import _reduce

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(_reduce, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate code for reduce with combine function."""

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    keep_dims = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    # Input tensor is already masked, so we can use it directly
    if is_tuple_input:
        # For tuple inputs, we need to handle the tuple structure
        input_tensor = state.ast_args[1]
        if isinstance(input_tensor, tuple):
            from ..ast_extension import create

            input_tensor = create(ast.Tuple, elts=list(input_tensor), ctx=ast.Load())
        else:
            input_tensor = state.ast_arg(1)
    else:
        input_tensor = state.ast_arg(1)
    helper_func_name = _register_helper_function(state, cast("int", combine_graph_id))
    reduce_expr = _create_reduce_expression(
        input_tensor, dim, helper_func_name, bool(keep_dims)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, reduce_expr)
    return reduce_expr


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


def _create_reduce_expression(
    input_tensor: ast.AST, dim: object, helper_func_name: str, keep_dims: bool
) -> ast.AST:
    """Create the tl.reduce expression."""
    from ..ast_extension import expr_from_string

    if dim is None:
        # Reduce all dimensions
        if keep_dims:
            template = (
                f"tl.reduce({{input_tensor}}, None, {helper_func_name}, keep_dims=True)"
            )
        else:
            template = f"tl.reduce({{input_tensor}}, None, {helper_func_name})"
        return expr_from_string(
            template,
            input_tensor=input_tensor,
        )
    # Reduce specific dimension
    if keep_dims:
        template = f"tl.reduce({{input_tensor}}, {{dim_value}}, {helper_func_name}, keep_dims=True)"
    else:
        template = f"tl.reduce({{input_tensor}}, {{dim_value}}, {helper_func_name})"
    return expr_from_string(
        template,
        input_tensor=input_tensor,
        # pyrefly: ignore [bad-argument-type]
        dim_value=ast.Constant(value=dim),
    )


def _create_tuple_result_expressions(
    state: CodegenState, reduce_expr: ast.AST
) -> list[ast.AST]:
    """Create getitem expressions for tuple results."""
    from ..ast_extension import expr_from_string

    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(
            "{reduce_result}[{index}]",
            reduce_result=reduce_expr,
            index=ast.Constant(value=i),
        )
        for i in range(num_elements)
    ]
