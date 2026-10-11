"""CuTe-backend codegen for ops defined in ``helion.language.reduce_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``reduce_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr

from ... import exc
from ...language import _decorators
from ...language.reduce_ops import _fake_reduce_tensor
from ...language.reduce_ops import _reduce

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import HelperFunctionGraphInfo
    from ..inductor_lowering import CodegenState
    from ..reduction_strategy import ReductionStrategy


def _infer_builtin_reduction_type_for_cute(
    state: CodegenState, combine_graph_id: int
) -> str | None:
    helper_graph_info = _get_helper_graph_info(state, combine_graph_id)
    output_values = _helper_graph_output_values(helper_graph_info)
    if output_values is None or len(output_values) != 1:
        return None
    output_node = output_values[0]
    placeholders = list(helper_graph_info.graph.find_nodes(op="placeholder"))
    if len(placeholders) != 2:
        return None
    if not _combines_operands(output_node, *placeholders):
        return None
    return _target_to_builtin_reduction(output_node.target)


def _combines_operands(
    node: torch.fx.Node, left: torch.fx.Node, right: torch.fx.Node
) -> bool:
    """Whether ``node`` applies its op to exactly ``left`` and ``right``.

    A combine that transforms an operand (``torch.maximum(a, b * 2)``),
    repeats one, or passes an extra argument (``alpha=``) is not the built-in
    reduction its op names.  The built-in ops are commutative, so either
    operand order matches.
    """
    return (
        node.op == "call_function"
        and not node.kwargs
        and node.args in ((left, right), (right, left))
    )


def _target_to_builtin_reduction(target: object) -> str | None:
    # torch.maximum/minimum propagate NaN, as the built-in max/min do.
    if target == torch.ops.aten.add.Tensor:
        return "sum"
    if target == torch.ops.aten.maximum.default:
        return "max"
    if target == torch.ops.aten.minimum.default:
        return "min"
    if target == torch.ops.aten.mul.Tensor:
        return "prod"
    return None


def _get_helper_graph_info(
    state: CodegenState, combine_graph_id: int
) -> HelperFunctionGraphInfo:
    from ..device_ir import HelperFunctionGraphInfo

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    return helper_graph_info


def _helper_graph_output_values(
    helper_graph_info: HelperFunctionGraphInfo,
) -> list[torch.fx.Node] | None:
    output_nodes = list(helper_graph_info.graph.find_nodes(op="output"))
    if len(output_nodes) != 1:
        return None
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, torch.fx.Node):
        return [output_value]
    if not isinstance(output_value, (tuple, list)):
        return None
    nodes: list[torch.fx.Node] = []
    for node in output_value:
        if not isinstance(node, torch.fx.Node):
            return None
        nodes.append(node)
    return nodes


def _infer_tuple_builtin_reduction_types_for_cute(
    state: CodegenState, combine_graph_id: int, tuple_arity: int
) -> tuple[str, ...] | None:
    helper_graph_info = _get_helper_graph_info(state, combine_graph_id)
    output_values = _helper_graph_output_values(helper_graph_info)
    if output_values is None or len(output_values) != tuple_arity:
        return None

    placeholders = list(helper_graph_info.graph.find_nodes(op="placeholder"))
    if len(placeholders) != 2 * tuple_arity:
        return None

    reduction_types: list[str] = []
    for i, output_node in enumerate(output_values):
        if not _combines_operands(
            output_node, placeholders[i], placeholders[i + tuple_arity]
        ):
            return None
        reduction_type = _target_to_builtin_reduction(output_node.target)
        if reduction_type is None:
            return None
        reduction_types.append(reduction_type)
    return tuple(reduction_types)


class _ArgReduce(NamedTuple):
    reduction_type: str
    # A tie keeps the right operand: a left fold over the elements
    # (``hl.reduce``'s reference order) then ends on the last tied element
    # instead of the first.
    tie_keeps_right: bool
    # The ``torch.where`` keeps the left operand when its comparison is false,
    # as every comparison with a NaN is.
    nan_keeps_left: bool


def _infer_tuple_argreduce_type_for_cute(
    state: CodegenState, combine_graph_id: int
) -> _ArgReduce | None:
    """Match ``(value, index)`` combines that keep the extreme value and its
    index, with ``torch.where`` over one ``>``/``<``/``>=``/``<=`` test."""
    helper_graph_info = _get_helper_graph_info(state, combine_graph_id)
    output_values = _helper_graph_output_values(helper_graph_info)
    if output_values is None or len(output_values) != 2:
        return None
    value_where_node, index_where_node = output_values
    if (
        value_where_node.op != "call_function"
        or value_where_node.target != torch.ops.aten.where.self
    ):
        return None
    if (
        index_where_node.op != "call_function"
        or index_where_node.target != torch.ops.aten.where.self
    ):
        return None
    if len(value_where_node.args) != 3 or len(index_where_node.args) != 3:
        return None

    compare_node = value_where_node.args[0]
    if compare_node is not index_where_node.args[0]:
        return None
    if (
        not isinstance(compare_node, torch.fx.Node)
        or compare_node.op != "call_function"
        or len(compare_node.args) != 2
    ):
        return None
    compare_target = compare_node.target
    compares = {
        torch.ops.aten.gt.Tensor: operator.gt,
        torch.ops.aten.lt.Tensor: operator.lt,
        torch.ops.aten.ge.Tensor: operator.ge,
        torch.ops.aten.le.Tensor: operator.le,
    }
    if compare_target not in compares:
        return None
    compare = compares[compare_target]

    placeholders = list(helper_graph_info.graph.find_nodes(op="placeholder"))
    if len(placeholders) != 4:
        return None
    left_value, left_index, right_value, right_index = placeholders
    if value_where_node.args[1:] not in {
        (right_value, left_value),
        (left_value, right_value),
    }:
        return None
    if index_where_node.args[1:] not in {
        (right_index, left_index),
        (left_index, right_index),
    }:
        return None
    if value_where_node.args[1:] == (right_value, left_value):
        choose_right_when_true = True
    elif value_where_node.args[1:] == (left_value, right_value):
        choose_right_when_true = False
    else:
        return None
    expected_index_branches = (
        (right_index, left_index)
        if choose_right_when_true
        else (left_index, right_index)
    )
    if index_where_node.args[1:] != expected_index_branches:
        return None

    compare_lhs, compare_rhs = compare_node.args
    if compare_lhs not in (left_value, right_value):
        return None
    if compare_rhs not in (left_value, right_value):
        return None
    if compare_lhs is compare_rhs:
        return None

    def keeps_right(left: int, right: int) -> bool:
        lhs = left if compare_lhs is left_value else right
        rhs = left if compare_rhs is left_value else right
        return compare(lhs, rhs) == choose_right_when_true

    tie_keeps_right = keeps_right(0, 0)
    if keeps_right(0, 1) and not keeps_right(1, 0):
        return _ArgReduce("argmax", tie_keeps_right, choose_right_when_true)
    if keeps_right(1, 0) and not keeps_right(0, 1):
        return _ArgReduce("argmin", tie_keeps_right, choose_right_when_true)
    return None


def _dim_reduction_strategy(
    state: CodegenState, fake_input: torch.Tensor, dim: int
) -> ReductionStrategy:
    """The strategy an aten reduction over ``dim`` of ``fake_input`` uses."""
    from ..compile_environment import CompileEnvironment
    from ..reduction_strategy import BlockReductionStrategy

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(fake_input.size(dim))
    if block_id is None:
        raise exc.BackendUnsupported(
            "cute", "hl.reduce over a dim that is not a tile or reduction dim"
        )
    if env.block_sizes[block_id].reduction:
        strategy = state.device_function.tile_strategy.get_reduction_strategy(block_id)
    else:
        strategy = BlockReductionStrategy(state, block_id)
    env.backend.validate_reduction_input(strategy.block_index, fake_input)
    return strategy


def _reduce_along(
    state: CodegenState,
    value: ast.AST,
    reduction_type: str,
    dims: list[int],
    fake_input: torch.Tensor,
    prefix: str,
) -> ast.AST:
    """Reduce ``value`` over ``dims`` with the strategies that own them.

    Each dim goes through the strategy an aten reduction over it uses
    (``ReductionLowering.codegen``): a block wider than a warp combines its
    warps' partials through shared memory, which a bare warp reduction does
    not.  Several dims (``dim=None``) reduce one at a time, innermost first.
    """
    from ..compile_environment import CompileEnvironment
    from ..reduction_strategy import cute_mark_cross_block_vec_lanes

    env = CompileEnvironment.current()
    for dim in sorted(dims, reverse=True):
        strategy = _dim_reduction_strategy(state, fake_input, dim)
        cute_mark_cross_block_vec_lanes(state, strategy.block_index)
        with env.fake_mode:
            fake_output = _fake_reduce_tensor(fake_input, dim, keep_dims=False)
        input_name = state.codegen.lift(value, dce=True, prefix=prefix).id
        value = env.backend.cast_ast(
            strategy.codegen_reduction(
                state, input_name, reduction_type, dim, fake_input, fake_output
            ),
            fake_output.dtype,
        )
        fake_input = fake_output
    return value


@_decorators.codegen(_reduce, "cute")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from ..ast_extension import expr_from_string
    from ..compile_environment import CompileEnvironment

    combine_graph_id = cast("int", state.proxy_arg(0))
    dim = state.proxy_arg(2)
    is_tuple_input = bool(state.proxy_arg(4))
    proxy_input = state.proxy_arg(1)
    first_input = (
        cast("Sequence[object]", proxy_input)[0] if is_tuple_input else proxy_input
    )
    assert isinstance(first_input, torch.Tensor)
    dims = [*range(first_input.ndim)] if dim is None else [cast("int", dim)]

    if not is_tuple_input:
        reduction_type = _infer_builtin_reduction_type_for_cute(state, combine_graph_id)
        if reduction_type is None:
            raise exc.BackendUnsupported(
                "cute",
                "hl.reduce custom combine function",
            )
        return _reduce_along(
            state,
            state.ast_arg(1),
            reduction_type,
            dims,
            first_input,
            "reduce_input",
        )

    ast_input = state.ast_args[1]
    if not isinstance(proxy_input, (tuple, list)) or not isinstance(
        ast_input, (tuple, list)
    ):
        raise exc.BackendUnsupported("cute", "hl.reduce tuple inputs")
    tuple_arity = len(proxy_input)
    if len(ast_input) != tuple_arity:
        raise exc.BackendUnsupported("cute", "hl.reduce tuple inputs")

    if reduction_types := _infer_tuple_builtin_reduction_types_for_cute(
        state, combine_graph_id, tuple_arity
    ):
        result_exprs: list[ast.AST] = []
        for i, reduction_type in enumerate(reduction_types):
            input_node = ast_input[i]
            assert isinstance(input_node, ast.AST), input_node
            result_exprs.append(
                _reduce_along(
                    state,
                    input_node,
                    reduction_type,
                    dims,
                    proxy_input[i],
                    f"reduce_input_{i}",
                )
            )
        return result_exprs

    argreduce = _infer_tuple_argreduce_type_for_cute(state, combine_graph_id)
    if argreduce is None:
        raise exc.BackendUnsupported("cute", "hl.reduce tuple custom combine function")
    if tuple_arity != 2:
        raise exc.BackendUnsupported(
            "cute",
            "hl.reduce tuple arg-reductions require 2 tuple elements",
        )
    if not isinstance(proxy_input[0], torch.Tensor) or not isinstance(
        proxy_input[1], torch.Tensor
    ):
        raise exc.BackendUnsupported("cute", "hl.reduce tuple arg-reduction inputs")
    if not isinstance(ast_input[0], ast.AST) or not isinstance(ast_input[1], ast.AST):
        raise exc.BackendUnsupported("cute", "hl.reduce tuple arg-reduction inputs")
    if len(dims) != 1:
        raise exc.BackendUnsupported(
            "cute", "hl.reduce tuple arg-reduction over several dims"
        )
    (dim_int,) = dims

    env = CompileEnvironment.current()
    backend = env.backend
    value_name = state.codegen.lift(ast_input[0], dce=True, prefix="reduce_value").id
    index_name = state.codegen.lift(ast_input[1], dce=True, prefix="reduce_index").id
    value_dtype = proxy_input[0].dtype
    index_dtype = proxy_input[1].dtype
    value_reduction = "max" if argreduce.reduction_type == "argmax" else "min"
    strategy = _dim_reduction_strategy(state, proxy_input[0], dim_int)
    mask = strategy.mask_var(strategy.block_index)
    int32 = backend.index_type_str(torch.int32)
    position = backend.cast_expr(strategy.index_var(strategy.block_index), int32)
    with env.fake_mode:
        fake_positions = torch.empty_like(proxy_input[0], dtype=torch.int32)

    def constant(value: float, dtype: torch.dtype) -> str:
        return backend.cast_expr(constant_repr(value), backend.dtype_str(dtype))

    def identity(reduction_type: str, dtype: torch.dtype) -> str:
        value = ir.Reduction.default_accumulator(reduction_type, dtype)
        assert isinstance(value, (float, int))
        return constant(value, dtype)

    def select_in_range(condition: str | None, value: str, otherwise: str) -> str:
        conditions = [c for c in (condition, mask) if c is not None]
        if not conditions:
            return value
        return f"({value}) if ({' and '.join(conditions)}) else ({otherwise})"

    def reduce(expr: str, reduction_type: str, fake: torch.Tensor, prefix: str) -> str:
        reduced = _reduce_along(
            state, expr_from_string(expr), reduction_type, dims, fake, f"{prefix}_input"
        )
        return state.codegen.lift(reduced, dce=True, prefix=prefix).id

    # Emulate the left fold over the elements (``hl.reduce``'s reference
    # order): find the position of the element it ends on, then take that
    # element's index operand (which need not increase with the position).
    # Without NaNs that is the first element holding the extreme value, or
    # the last one when a tie keeps the right operand.
    int32_max = backend.cast_expr(repr(torch.iinfo(torch.int32).max), int32)
    minus_one = backend.cast_expr("-1", int32)
    no_position = minus_one if argreduce.tie_keeps_right else int32_max
    value_input = value_name
    # Where the fold ends if it ends on a NaN, and (when the fold keeps the
    # right operand of a NaN comparison) the elements after the last NaN.
    nan_position: str | None = None
    after_nan: str | None = None
    if value_dtype.is_floating_point:
        is_nan = f"(({value_name}) != ({value_name}))"
        value_identity = identity(value_reduction, value_dtype)
        if argreduce.nan_keeps_left:
            # A NaN accumulator is kept to the end and a NaN right operand is
            # skipped: the fold ends on the first element if it is a NaN, and
            # otherwise on an element that is not.  Only the first element's
            # NaN reaches the value reduction, which propagates it.
            if env.block_sizes[strategy.block_index].reduction:
                nan_position = backend.cast_expr("0", int32)
            else:
                nan_position = reduce(
                    select_in_range(None, position, int32_max),
                    "min",
                    fake_positions,
                    "first_position",
                )
            value_input = (
                f"({value_identity}) if ({is_nan} and {position} != {nan_position}) "
                f"else ({value_name})"
            )
        else:
            # A NaN right operand replaces the accumulator and the next
            # element replaces a NaN accumulator: the fold restarts after the
            # last NaN, and ends on it when no element follows.
            nan_position = reduce(
                select_in_range(is_nan, position, minus_one),
                "max",
                fake_positions,
                "last_nan",
            )
            after_nan = f"({position} > ({nan_position}))"
            value_input = f"({value_name}) if {after_nan} else ({value_identity})"
    reduced_value = reduce(
        value_input, value_reduction, proxy_input[0], "reduced_value"
    )
    holds = f"(({value_name}) == ({reduced_value}))"
    if after_nan is not None:
        holds = f"{holds} and {after_nan}"
    chosen = reduce(
        select_in_range(holds, position, no_position),
        "max" if argreduce.tie_keeps_right else "min",
        fake_positions,
        "reduced_position",
    )
    result_value = reduced_value
    if nan_position is not None:
        if after_nan is None:
            ends_on_nan = f"(({reduced_value}) != ({reduced_value}))"
        else:
            # No element after the last NaN.
            ends_on_nan = f"(({chosen}) == ({no_position}))"
            nan = constant(float("nan"), value_dtype)
            result_value = f"({nan}) if {ends_on_nan} else ({reduced_value})"
        chosen = state.codegen.lift(
            expr_from_string(f"({nan_position}) if {ends_on_nan} else ({chosen})"),
            dce=True,
            prefix="fold_position",
        ).id
    reduced_index = _reduce_along(
        state,
        expr_from_string(
            f"({index_name}) if ({position} == {chosen}) "
            f"else ({identity('min', index_dtype)})"
        ),
        "min",
        dims,
        proxy_input[1],
        "reduce_index_candidate",
    )
    return [expr_from_string(result_value), reduced_index]
