"""Triton-backend codegen for ops defined in ``helion.language.scan_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``scan_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import cast

from ...language import _decorators
from ...language.scan_ops import _associative_scan
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .tracing_ops import _tile_mask_expr

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
    valid = _scan_valid_lanes(state, cast("int", dim), bool(reverse))
    if (
        valid is not None
        and not is_tuple_input
        and (identity := _combine_identity(state, cast("int", combine_graph_id)))
        is not None
    ):
        # An add or a multiply skips a lane that holds its identity.
        input_tensor = _identity_outside(state, input_tensor, valid, identity)
        valid = None
    if valid is not None:
        # A flag beside each value keeps the lanes that hold no element out
        # of the combine.
        input_tensor, helper_func_name = _with_validity(
            state, input_tensor, valid, helper_func_name, bool(is_tuple_input)
        )
    scan_expr = _create_scan_expression(
        input_tensor,
        state.tile_strategy.physical_dim(
            [*state.proxy_tensor_arg(1).size()], cast("int", dim)
        ),
        helper_func_name,
        bool(reverse),
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, scan_expr)
    if valid is not None:
        return expr_from_string("{scan}[0]", scan=scan_expr)
    return scan_expr


def _scanned_lanes_valid(state: CodegenState, dim: int) -> str | None:
    """The lanes along ``dim`` that hold elements, broadcast to the input's
    shape, or None when every lane does: those inside the mask of the dim's
    block (a reduction dim padded to a power of two, a partial last tile)."""
    tensor = state.proxy_tensor_arg(1)
    env = CompileEnvironment.current()
    block = env.resolve_block_id(tensor.size(dim))
    if block is None:
        return None
    return _tile_mask_expr(
        state.device_function, tensor.size(), (env.canonical_block_id(block),)
    )


def _with_validity(
    state: CodegenState,
    input_tensor: ast.AST,
    valid: str,
    helper_func_name: str,
    is_tuple: bool,
) -> tuple[ast.AST, str]:
    """The scan operands with ``valid`` appended, and a combine function that
    applies ``helper_func_name`` where both sides are valid and otherwise
    passes the valid side through: an invalid lane acts as an identity."""
    if is_tuple:
        assert isinstance(input_tensor, (ast.Tuple, ast.List))
        values = list(input_tensor.elts)
    else:
        values = [input_tensor]
    count = len(values)
    left = [f"a{i}" for i in range(count)]
    right = [f"b{i}" for i in range(count)]
    combined = [f"c{i}" for i in range(count)]
    # A tuple scan's traced combine returns a tuple, of one value too.
    results = f"({', '.join(combined)},)" if is_tuple else combined[0]
    selected = ", ".join(
        f"tl.where(both, {c}, tl.where(a_valid, {a}, {b}))"
        for a, b, c in zip(left, right, combined, strict=True)
    )
    name = state.device_function.new_var(f"{helper_func_name}_valid_lanes")
    state.codegen.module_statements.append(
        statement_from_string(
            "@triton.jit\n"
            f"def {name}({', '.join([*left, 'a_valid', *right, 'b_valid'])}):\n"
            f"    {results} = {helper_func_name}({', '.join([*left, *right])})\n"
            "    both = a_valid & b_valid\n"
            f"    return {selected}, a_valid | b_valid\n"
        )
    )
    operands = create(
        ast.Tuple, elts=[*values, expr_from_string(valid)], ctx=ast.Load()
    )
    return operands, name


def _scan_valid_lanes(state: CodegenState, dim: int, reverse: bool) -> str | None:
    """The lanes along ``dim`` that hold elements, where a scan in that
    direction could fold the others in, or None.

    A reverse scan folds the lanes after each element, which run past the end
    of the dim's block (``_scanned_lanes_valid``).  A jagged tile whose rows
    run along ``dim`` (its mask spans the dim) leaves lanes without elements
    between the others, so a scan in either direction skips the lanes outside
    that tile's mask too.
    """
    import torch

    value = state.proxy_arg(1)
    tensor = value[0] if isinstance(value, (tuple, list)) else value
    assert isinstance(tensor, torch.Tensor)
    # ``dim=-1`` names the last dim, which the jagged dims below must skip.
    dim %= tensor.ndim
    env = CompileEnvironment.current()
    masks = [
        _scanned_lanes_valid(state, other)
        for other, size in enumerate(tensor.size())
        if other != dim
        and (block := env.resolve_block_id(size)) is not None
        and env.is_jagged_tile(block)
        and any(
            env.known_equal(mask_size, tensor.size(dim))
            for mask_size in env.jagged_tile_mask_shapes[block]
        )
    ]
    if reverse:
        masks.append(_scanned_lanes_valid(state, dim))
    present = [mask for mask in masks if mask is not None]
    return " & ".join(f"({mask})" for mask in present) if present else None


def _identity_outside(
    state: CodegenState, input_tensor: ast.AST, valid: str, identity: int
) -> ast.AST:
    """``input_tensor`` with ``identity`` in the lanes outside ``valid``."""
    import torch

    tensor = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(
        f"tl.where({valid}, {{value}}, tl.full([], {identity}, {dtype}))",
        value=input_tensor,
    )


def _combine_identity(state: CodegenState, combine_graph_id: int) -> int | None:
    """The identity of a combine function that adds or multiplies its two
    operands, or None."""
    import torch

    from ..device_ir import HelperFunctionGraphInfo

    helper = state.get_graph(combine_graph_id)
    assert isinstance(helper, HelperFunctionGraphInfo)
    calls = [node for node in helper.graph.nodes if node.op == "call_function"]
    operands = helper.graph.find_nodes(op="placeholder")
    if len(calls) != 1 or len(operands) != 2:
        return None
    (call,) = calls
    if (
        call.kwargs
        or sorted(map(id, call.args)) != sorted(map(id, operands))
        or helper.graph.output_node().args[0] is not call
    ):
        # Another form, or a traced but unused add (``a + b; return b``).
        return None
    if call.target in (operator.add, torch.add, torch.ops.aten.add.Tensor):
        return 0
    if call.target in (operator.mul, torch.mul, torch.ops.aten.mul.Tensor):
        return 1
    return None


def _get_input_tensor_ast(state: CodegenState, is_tuple_input: bool) -> ast.AST:
    """Get the input tensor AST, handling tuple inputs specially."""
    if not is_tuple_input:
        return state.ast_arg(1)

    raw_input = state.ast_args[1]
    if isinstance(raw_input, tuple):
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
    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(
            "{scan_result}[{index}]", scan_result=scan_expr, index=ast.Constant(value=i)
        )
        for i in range(num_elements)
    ]
