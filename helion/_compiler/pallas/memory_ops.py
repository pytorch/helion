"""Pallas-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language.memory_ops import _maybe_materialize_tile_index_load
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from . import codegen as pallas_codegen

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    name = state.device_function.tensor_arg(tensor).name
    name = pallas_codegen.vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    parts, none_dims = pallas_codegen.index_parts(state, subscript, tensor)
    value = pallas_codegen.sliced_value_for_store(
        state, tensor, subscript, parts, value
    )
    parts, value = pallas_codegen.widen_barrier_temp_store_indices(
        state, tensor, subscript, parts, value, name
    )
    idx_str = ", ".join(parts)
    from .gather import emit_scatter_store
    from .tensorcore_plan import TENSORCORE_PLAN_META
    from .tensorcore_plan import OneHotScatterPlan

    plan = state.fx_node.meta.get(TENSORCORE_PLAN_META) if state.fx_node else None
    is_scatter = isinstance(plan, OneHotScatterPlan)
    if is_scatter:
        value = emit_scatter_store(state, plan.plan, name, idx_str, value, none_dims)
    from .ordered_carry import emit_carry_store

    if not is_scatter and state.device_function.carry_tiles:
        if emit_carry_store(state, tensor, subscript, name, parts, value):
            pallas_codegen.mark_dma_scratch_initialized(state, tensor)
            return
    pallas_codegen.reject_partial_last_two_dim_direct_store(
        state, tensor, subscript, parts
    )
    if tensor.dtype is torch.bool:
        value = CompileEnvironment.current().backend.cast_ast(value, torch.bool)
    state.codegen.add_statement(
        statement_from_string(f"{name}[{idx_str}] = {{value}}", value=value)
    )
    if not is_scatter:
        pallas_codegen.mark_dma_scratch_initialized(state, tensor)


@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
    if tile_index_result is not None:
        return tile_index_result

    return pallas_codegen.load_expr(state, list(subscript), tensor)
