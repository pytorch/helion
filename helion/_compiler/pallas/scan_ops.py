"""Pallas-backend codegen for associative scan operations."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.scan_ops import _associative_scan
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..scan_ops import SCAN_ARITHMETIC_OPS
from ..scan_ops import SCAN_ATEN_COMPARISON_OPS
from ..scan_ops import SCAN_CAST_OP
from ..scan_ops import SCAN_MAX_OPS
from ..scan_ops import SCAN_MIN_OPS
from ..scan_ops import SCAN_PYTHON_COMPARISON_OPS
from ..scan_ops import SCAN_WHERE_OPS
from ..scan_ops import scan_combine_arg
from ..scan_ops import scan_combine_binary_expression
from ..scan_ops import scan_combine_check_extra_args
from ..scan_ops import scan_combine_check_kwargs
from .backend import _JAX_UNSUPPORTED_DTYPES

if TYPE_CHECKING:
    from ..device_ir import HelperFunctionGraphInfo
    from ..inductor_lowering import CodegenState


_PALLAS_SCAN_BINARY_OPS: dict[object, str] = {
    **SCAN_ARITHMETIC_OPS,
    **SCAN_PYTHON_COMPARISON_OPS,
    **SCAN_ATEN_COMPARISON_OPS,
}


@_decorators.codegen(_associative_scan, "pallas")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate a conservative static serial Pallas/JAX scan."""
    from ..device_ir import HelperFunctionGraphInfo

    combine_graph_id = cast("int", state.proxy_arg(0))
    dim = cast("int", state.proxy_arg(2))
    reverse = bool(state.proxy_arg(3))
    is_tuple_input = bool(state.proxy_arg(4))
    if reverse:
        raise exc.BackendUnsupported("pallas", "reverse associative_scan")

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)

    input_values, input_asts = _pallas_scan_inputs(state, is_tuple_input)
    result_exprs = _pallas_emit_serial_scan(
        state,
        helper_graph_info,
        input_values,
        input_asts,
        dim,
    )
    return result_exprs if is_tuple_input else result_exprs[0]


def _pallas_scan_inputs(
    state: CodegenState, is_tuple_input: bool
) -> tuple[list[torch.Tensor], list[ast.AST]]:
    input_proxy = state.proxy_arg(1)
    raw_input_ast = state.ast_args[1]

    if is_tuple_input:
        if not isinstance(input_proxy, (tuple, list)) or not isinstance(
            raw_input_ast, (tuple, list)
        ):
            raise exc.BackendUnsupported("pallas", "tuple associative_scan input")
        input_values = list(input_proxy)
        input_asts = list(raw_input_ast)
    else:
        input_values = [input_proxy]
        input_asts = [state.ast_arg(1)]

    if not input_values or not all(isinstance(t, torch.Tensor) for t in input_values):
        raise exc.BackendUnsupported("pallas", "associative_scan input")
    if not all(isinstance(arg, ast.AST) for arg in input_asts):
        raise exc.BackendUnsupported("pallas", "associative_scan input expression")
    return (
        cast("list[torch.Tensor]", input_values),
        cast("list[ast.AST]", input_asts),
    )


def _pallas_emit_serial_scan(
    state: CodegenState,
    helper_graph_info: HelperFunctionGraphInfo,
    input_values: list[torch.Tensor],
    input_asts: list[ast.AST],
    dim: int,
) -> list[ast.AST]:
    """Emit a static JAX loop, inline the combine graph, and stack its items."""
    first = input_values[0]
    ndim = first.ndim
    if ndim == 0:
        raise exc.BackendUnsupported("pallas", "scalar associative_scan input")
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise exc.BackendUnsupported("pallas", "associative_scan dim")

    first_shape = tuple(str(s) for s in first.shape)
    for value in input_values:
        if value.ndim != ndim or tuple(str(s) for s in value.shape) != first_shape:
            raise exc.BackendUnsupported("pallas", "tuple associative_scan shapes")

    env = CompileEnvironment.current()
    fake_extent: object = first.shape[dim]
    if isinstance(fake_extent, torch.SymInt):
        fake_extent = env.size_hint(fake_extent)
    if not isinstance(fake_extent, int) or fake_extent <= 0:
        raise exc.BackendUnsupported("pallas", "dynamic associative_scan extent")

    input_names = [
        state.codegen.lift(input_ast, dce=True, prefix="scan_input").id
        for input_ast in input_asts
    ]
    result_vars = [state.device_function.new_var("scan_out") for _ in input_names]
    acc_vars = [state.device_function.new_var("scan_acc") for _ in input_names]
    item_vars = [state.device_function.new_var("scan_items") for _ in input_names]
    extent_var = state.device_function.new_var("scan_extent")

    state.codegen.add_statement(
        statement_from_string(f"{extent_var} = {input_names[0]}.shape[{dim}]")
    )
    for item_var in item_vars:
        state.codegen.add_statement(
            statement_from_string(f"{item_var} = [None] * {extent_var}")
        )

    seed_index = _pallas_scan_index(ndim, dim, "0")
    for acc_var, item_var, input_name in zip(
        acc_vars, item_vars, input_names, strict=True
    ):
        state.codegen.add_statement(
            statement_from_string(f"{acc_var} = {input_name}[{seed_index}]")
        )
        state.codegen.add_statement(statement_from_string(f"{item_var}[0] = {acc_var}"))

    scan_i = state.device_function.new_var("scan_i")
    value_index = _pallas_scan_index(ndim, dim, scan_i)
    value_exprs = [f"{input_name}[{value_index}]" for input_name in input_names]
    combine_exprs = _pallas_combine_result_expressions(
        helper_graph_info, [*acc_vars, *value_exprs]
    )
    if len(combine_exprs) != len(acc_vars):
        raise exc.BackendUnsupported("pallas", "associative_scan combine output")
    next_vars = [state.device_function.new_var("scan_next") for _ in acc_vars]
    lines = [f"for {scan_i} in range(1, {extent_var}):"]
    lines.extend(
        f"    {next_var} = {combine_expr}"
        for next_var, combine_expr in zip(next_vars, combine_exprs, strict=True)
    )
    lines.extend(
        f"    {acc_var} = {next_var}"
        for acc_var, next_var in zip(acc_vars, next_vars, strict=True)
    )
    lines.extend(
        f"    {item_var}[{scan_i}] = {acc_var}"
        for item_var, acc_var in zip(item_vars, acc_vars, strict=True)
    )
    state.codegen.add_statement(statement_from_string("\n".join(lines)))

    for result_var, item_var in zip(result_vars, item_vars, strict=True):
        state.codegen.add_statement(
            statement_from_string(f"{result_var} = jnp.stack({item_var}, axis={dim})")
        )

    return [expr_from_string(result_var) for result_var in result_vars]


def _pallas_scan_index(ndim: int, dim: int, pos: str) -> str:
    parts = [":" for _ in range(ndim)]
    parts[dim] = pos
    return ", ".join(parts)


def _pallas_combine_result_expressions(
    helper_graph_info: HelperFunctionGraphInfo,
    arg_exprs: list[str],
) -> list[str]:
    """Inline the combine graph as JAX expressions for Pallas serial scan."""
    env = CompileEnvironment.current()
    graph = helper_graph_info.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(arg_exprs):
        raise exc.BackendUnsupported("pallas", "associative_scan combine arity")

    env_map: dict[object, str] = dict(zip(placeholders, arg_exprs, strict=True))

    def operand(value: object) -> str:
        if isinstance(value, torch.fx.Node):
            if value not in env_map:
                raise exc.BackendUnsupported(
                    "pallas", f"associative_scan combine dependency: {value}"
                )
            return env_map[value]
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        raise exc.BackendUnsupported(
            "pallas", f"associative_scan combine operand {value!r}"
        )

    for node in graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function":
            raise exc.BackendUnsupported(
                "pallas", f"associative_scan combine op {node.op}"
            )
        target = node.target
        if target in _PALLAS_SCAN_BINARY_OPS:
            lhs = operand(scan_combine_arg(node, 0, "input", "pallas"))
            rhs = operand(scan_combine_arg(node, 1, "other", "pallas"))
            env_map[node] = scan_combine_binary_expression(
                node, _PALLAS_SCAN_BINARY_OPS[target], lhs, rhs, operand, "pallas"
            )
        elif target in SCAN_WHERE_OPS:
            scan_combine_check_extra_args(node, 3, "pallas")
            scan_combine_check_kwargs(node, "pallas", ("condition", "input", "other"))
            cond = operand(scan_combine_arg(node, 0, "condition", "pallas"))
            tval = operand(scan_combine_arg(node, 1, "input", "pallas"))
            fval = operand(scan_combine_arg(node, 2, "other", "pallas"))
            env_map[node] = f"jnp.where({cond}, {tval}, {fval})"
        elif target in SCAN_MIN_OPS:
            scan_combine_check_extra_args(node, 2, "pallas")
            scan_combine_check_kwargs(node, "pallas", ("input", "other"))
            lhs = operand(scan_combine_arg(node, 0, "input", "pallas"))
            rhs = operand(scan_combine_arg(node, 1, "other", "pallas"))
            env_map[node] = f"jnp.minimum({lhs}, {rhs})"
        elif target in SCAN_MAX_OPS:
            scan_combine_check_extra_args(node, 2, "pallas")
            scan_combine_check_kwargs(node, "pallas", ("input", "other"))
            lhs = operand(scan_combine_arg(node, 0, "input", "pallas"))
            rhs = operand(scan_combine_arg(node, 1, "other", "pallas"))
            env_map[node] = f"jnp.maximum({lhs}, {rhs})"
        elif target is SCAN_CAST_OP:
            scan_combine_check_extra_args(node, 2, "pallas")
            scan_combine_check_kwargs(node, "pallas", ("a", "dtype"))
            dtype = scan_combine_arg(node, 1, "dtype", "pallas")
            if not isinstance(dtype, torch.dtype):
                raise exc.BackendUnsupported(
                    "pallas", f"associative_scan combine dtype {dtype!r}"
                )
            if dtype in _JAX_UNSUPPORTED_DTYPES:
                raise exc.BackendUnsupported(
                    "pallas", f"associative_scan combine dtype {dtype!r}"
                )
            env_map[node] = env.backend.cast_expr(
                operand(scan_combine_arg(node, 0, "a", "pallas")),
                env.backend.dtype_str(dtype),
            )
        else:
            raise exc.BackendUnsupported(
                "pallas", f"associative_scan combine function: {target}"
            )

    output_nodes = [n for n in graph.nodes if n.op == "output"]
    if len(output_nodes) != 1:
        raise exc.BackendUnsupported("pallas", "associative_scan combine output")
    outputs = output_nodes[0].args[0]
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    return [operand(output) for output in outputs]
