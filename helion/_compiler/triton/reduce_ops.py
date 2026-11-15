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

import torch
from torch._inductor.codegen.simd import constant_repr

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
    reduced_dim = dim
    if dim is not None:
        dim = state.tile_strategy.physical_dim(
            [*state.proxy_tensor_arg(1).size()], cast("int", dim)
        )

    # The inputs come masked on their tiles' masks; the lanes of a padded
    # hl.arange take their masked value below (_mask_padded_range_lanes).
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
    input_tensor = _mask_padded_range_lanes(state, input_tensor, reduced_dim)
    helper_func_name = _register_helper_function(state, cast("int", combine_graph_id))
    reduce_expr = _create_reduce_expression(
        input_tensor, dim, helper_func_name, bool(keep_dims)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, reduce_expr)
    return reduce_expr


def _mask_padded_range_lanes(
    state: CodegenState, value: ast.AST, dim: object
) -> ast.AST:
    """``value``, the reduction's input (a tuple of inputs), with the lanes
    past the length of a padded ``hl.arange`` on the reduced dims set to each
    input's ``other``.

    The inputs come masked on their tiles' masks (``_mask_to``), but
    ``hl.load(x, [i, hl.arange(200)])`` has 256 lanes, which no tile's mask
    covers: ``hl.reduce(fn, row + 1.0, dim=0)`` reduced the 56 past the range
    too.  An input left without its ``_mask_to`` holds ``other`` in those
    lanes already (``remove_unnecessary_masking``)."""
    nodes = state.fx_node.args[1] if state.fx_node is not None else None
    if isinstance(value, ast.Tuple):
        assert isinstance(nodes, (list, tuple))
        value.elts = [
            cast("ast.expr", _mask_input(state, elt, node, dim))
            for elt, node in zip(value.elts, nodes, strict=True)
        ]
        return value
    return _mask_input(state, value, nodes, dim)


def _mask_input(
    state: CodegenState, value: ast.AST, node: object, dim: object
) -> ast.AST:
    """One input of ``_mask_padded_range_lanes``.  As in
    ``ReductionLowering._mask_padded_range_lanes``, a reduced dim of the
    range's own length (a tile kernel's reduction dim) is left alone: the
    ``_mask_to`` masks its lanes already.  hl.reduce is never rolled, so its
    reduction is persistent."""
    from ...language._tracing_ops import _mask_to
    from ..ast_extension import expr_from_string
    from ..compile_environment import CompileEnvironment
    from ..indexing_strategy import _padded_iota_dims

    if not isinstance(node, torch.fx.Node) or node.target is not _mask_to:
        return value
    source, other = node.args
    fake = node.meta["val"]
    assert isinstance(other, (int, float, bool))
    found = _padded_iota_dims(source, through_loads=True)
    if found is None or not isinstance(fake, torch.Tensor):
        return value
    env = CompileEnvironment.current()
    sizes = [*fake.size()]
    # The mask indexes the value's physical dims: a flattened tile's dims
    # share one.
    physical_dims = state.tile_strategy.physical_dims(sizes)
    offset = fake.ndim - len(found)
    for reduced in range(fake.ndim) if dim is None else [cast("int", dim) % fake.ndim]:
        length = found[reduced - offset] if reduced >= offset else None
        block_id = env.resolve_block_id(sizes[reduced])
        if length is None or (
            block_id is not None and env.block_sizes[block_id].size == length
        ):
            continue
        (physical,) = [
            index for index, dims in enumerate(physical_dims) if reduced in dims
        ]
        masked = env.backend.mask_padded_range_lanes(
            "{value}", length, physical, len(physical_dims), constant_repr(other)
        )
        if masked is not None:
            value = expr_from_string(masked, value=value)
    return value


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
