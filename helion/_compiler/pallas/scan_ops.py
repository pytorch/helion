"""Pallas code generation for prefix scans."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.scan_ops import _associative_scan
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..device_ir import HelperFunctionGraphInfo
    from ..inductor_lowering import CodegenState


def _is_add_scan(helper: HelperFunctionGraphInfo) -> bool:
    add_targets = (
        operator.add,
        torch.add,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
    )
    calls = [node for node in helper.graph.nodes if node.op == "call_function"]
    return len(calls) == 1 and calls[0].target in add_targets


@_decorators.codegen(_associative_scan, "pallas")
def _(state: CodegenState) -> ast.AST:
    from ..device_ir import HelperFunctionGraphInfo

    combine_graph_id = cast("int", state.proxy_arg(0))
    dim = cast("int", state.proxy_arg(2))
    reverse = bool(state.proxy_arg(3))
    is_tuple_input = bool(state.proxy_arg(4))
    if is_tuple_input:
        raise exc.BackendUnsupported("pallas", "tuple associative_scan input")

    helper = state.get_graph(combine_graph_id)
    assert isinstance(helper, HelperFunctionGraphInfo)
    if not _is_add_scan(helper):
        raise exc.BackendUnsupported("pallas", "non-add associative_scan")

    fx_node = state.fx_node
    if fx_node is None:
        raise exc.BackendUnsupported("pallas", "associative_scan without FX node")
    value = fx_node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise exc.BackendUnsupported("pallas", "associative_scan input")
    if dim < 0:
        dim += value.ndim
    if dim != value.ndim - 1:
        raise exc.BackendUnsupported("pallas", "associative_scan on a non-last axis")

    extent = value.shape[dim]
    if isinstance(extent, torch.SymInt):
        extent = CompileEnvironment.current().size_hint(extent)
    if not isinstance(extent, int):
        raise exc.BackendUnsupported("pallas", "dynamic associative_scan extent")

    result = state.codegen.device_function.new_var("scan")
    state.codegen.add_statement(
        statement_from_string(f"{result} = {{value}}", value=state.ast_arg(1))
    )
    offset = 1
    while offset < extent:
        next_result = state.codegen.device_function.new_var("scan")
        combined = f"jnp.add({result}[..., :-{offset}], {result}[..., {offset}:])"
        if reverse:
            expression = (
                f"jnp.concatenate(({combined}, {result}[..., -{offset}:]), axis=-1)"
            )
        else:
            expression = (
                f"jnp.concatenate(({result}[..., :{offset}], {combined}), axis=-1)"
            )
        state.codegen.add_statement(
            statement_from_string(f"{next_result} = {expression}")
        )
        result = next_result
        offset *= 2
    return expr_from_string(result)
