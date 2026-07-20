"""CuTe-backend codegen for ops defined in ``helion.language.reduce_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``reduce_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.reduce_ops import _reduce

if TYPE_CHECKING:
    from ..device_ir import HelperFunctionGraphInfo
    from ..inductor_lowering import CodegenState


def _infer_builtin_reduction_type_for_cute(
    state: CodegenState, combine_graph_id: int
) -> str | None:
    helper_graph_info = _get_helper_graph_info(state, combine_graph_id)
    output_values = _helper_graph_output_values(helper_graph_info)
    if output_values is None or len(output_values) != 1:
        return None
    output_node = output_values[0]
    if output_node.op != "call_function":
        return None
    return _target_to_builtin_reduction(output_node.target)


def _target_to_builtin_reduction(target: object) -> str | None:
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
        if output_node.op != "call_function":
            return None
        reduction_type = _target_to_builtin_reduction(output_node.target)
        if reduction_type is None:
            return None
        left_node = placeholders[i]
        right_node = placeholders[i + tuple_arity]
        if output_node.args != (left_node, right_node):
            return None
        reduction_types.append(reduction_type)
    return tuple(reduction_types)


def _infer_tuple_argreduce_type_for_cute(
    state: CodegenState, combine_graph_id: int
) -> str | None:
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
    if compare_target not in {torch.ops.aten.gt.Tensor, torch.ops.aten.lt.Tensor}:
        return None

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

    def selected_for_pair(left: int, right: int) -> int:
        lhs = left if compare_lhs is left_value else right
        rhs = left if compare_rhs is left_value else right
        if compare_target == torch.ops.aten.gt.Tensor:
            cond = lhs > rhs
        else:
            cond = lhs < rhs
        if cond:
            return right if choose_right_when_true else left
        return left if choose_right_when_true else right

    selected_01 = selected_for_pair(0, 1)
    selected_10 = selected_for_pair(1, 0)
    if selected_01 == 1 and selected_10 == 1:
        return "argmax"
    if selected_01 == 0 and selected_10 == 0:
        return "argmin"
    return None


@_decorators.codegen(_reduce, "cute")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from ..ast_extension import expr_from_string

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    is_tuple_input = bool(state.proxy_arg(4))

    if dim is None:
        raise exc.BackendUnsupported("cute", "hl.reduce(..., dim=None)")

    from ..compile_environment import CompileEnvironment

    backend = CompileEnvironment.current().backend
    dim_int = cast("int", dim)
    combine_graph_id_int = cast("int", combine_graph_id)

    if not is_tuple_input:
        reduction_type = _infer_builtin_reduction_type_for_cute(
            state, combine_graph_id_int
        )
        if reduction_type is None:
            raise exc.BackendUnsupported(
                "cute",
                "hl.reduce custom combine function",
            )
        input_name = state.codegen.lift(
            state.ast_arg(1), dce=True, prefix="reduce_input"
        ).id
        return expr_from_string(
            backend.reduction_expr(
                input_name,
                reduction_type,
                dim_int,
            )
        )

    proxy_input = state.proxy_arg(1)
    ast_input = state.ast_args[1]
    if not isinstance(proxy_input, (tuple, list)) or not isinstance(
        ast_input, (tuple, list)
    ):
        raise exc.BackendUnsupported("cute", "hl.reduce tuple inputs")
    tuple_arity = len(proxy_input)
    if len(ast_input) != tuple_arity:
        raise exc.BackendUnsupported("cute", "hl.reduce tuple inputs")

    if reduction_types := _infer_tuple_builtin_reduction_types_for_cute(
        state, combine_graph_id_int, tuple_arity
    ):
        result_exprs: list[ast.AST] = []
        for i, reduction_type in enumerate(reduction_types):
            input_node = ast_input[i]
            assert isinstance(input_node, ast.AST), input_node
            input_name = state.codegen.lift(
                input_node, dce=True, prefix=f"reduce_input_{i}"
            ).id
            result_exprs.append(
                expr_from_string(
                    backend.reduction_expr(
                        input_name,
                        reduction_type,
                        dim_int,
                    )
                )
            )
        return result_exprs

    argreduce_type = _infer_tuple_argreduce_type_for_cute(state, combine_graph_id_int)
    if argreduce_type is None:
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

    value_name = state.codegen.lift(ast_input[0], dce=True, prefix="reduce_value").id
    index_name = state.codegen.lift(ast_input[1], dce=True, prefix="reduce_index").id
    index_dtype = proxy_input[1].dtype
    value_reduction = "max" if argreduce_type == "argmax" else "min"
    reduced_value_expr = expr_from_string(
        backend.reduction_expr(
            value_name,
            value_reduction,
            dim_int,
        )
    )
    reduced_index_expr = expr_from_string(
        backend.argreduce_result_expr(
            value_name,
            index_name,
            argreduce_type,
            dim_int,
            index_dtype,
            index_dtype=index_dtype,
        )
    )
    return [reduced_value_expr, reduced_index_expr]
