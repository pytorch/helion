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
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from . import codegen as pallas_codegen

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    from ... import exc
    from .dma import emit_immediate_indirect_transfer
    from .tensorcore_plan import TENSORCORE_PLAN_META
    from .tensorcore_plan import DmaScatterPlan
    from .tensorcore_plan import OneHotScatterPlan

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    unclamped_value = value
    assert isinstance(tensor, torch.Tensor)
    arg_name = state.device_function.tensor_arg(tensor).name
    name = state.device_function.pallas_tensor_ref_name(tensor)
    name = pallas_codegen.vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    value = pallas_codegen.sliced_value_for_store(
        state, tensor, subscript, parts, value
    )
    idx_str = ", ".join(parts)
    plan = state.fx_node.meta.get(TENSORCORE_PLAN_META) if state.fx_node else None
    if isinstance(plan, DmaScatterPlan):
        dma_ref = pallas_codegen.memory_op_dma_scratch(state)
        if dma_ref is None:
            raise exc.InvalidConfig(
                "indirect DMA store was not admitted by the active scheduler"
            )
        state.codegen.add_statement(
            statement_from_string(f"{dma_ref}[...] = {{value}}", value=value)
        )
        # The fori scheduler emits the writeback after the body. Root grids
        # have no enclosing scheduler, so this call emits it immediately.
        emit_immediate_indirect_transfer(state, plan, arg_name)
        return
    from .gather import emit_scatter_store

    is_scatter = isinstance(plan, OneHotScatterPlan)
    if is_scatter:
        value = emit_scatter_store(state, plan.plan, name, idx_str, value)
    elif state.proxy_arg(3) is not None:
        from ..tile_strategy import EmitPipelineLoopState

        loop, _ = pallas_codegen._find_dma_scratch_loop(state, arg_name)
        if loop is not None:
            if (
                not isinstance(loop, EmitPipelineLoopState)
                or state.fx_node not in loop._bounded_store_masks
            ):
                raise exc.BackendUnsupported(
                    "pallas", "masked DMA stores require an aligned bounded prefix"
                )
            # The prefix DMA already consumes this mask. Its untouched output
            # stage must not be read to apply the same mask a second time.
        else:
            mask = expr_from_string(
                "jnp.broadcast_to({mask}, jnp.shape({value}))",
                mask=state.ast_arg(3),
                value=unclamped_value,
            )
            mask = pallas_codegen.sliced_value_for_store(
                state, tensor, subscript, parts, mask
            )
            value = expr_from_string(
                f"jnp.where({{mask}}, {{value}}, {name}[{idx_str}])",
                mask=mask,
                value=value,
            )
    from .ordered_carry import emit_carry_store

    if not is_scatter and state.device_function.carry_tiles:
        if emit_carry_store(state, tensor, subscript, name, idx_str, value):
            return
    state.codegen.add_statement(
        statement_from_string(f"{name}[{idx_str}] = {{value}}", value=value)
    )


@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .view_ops import _resident_plan

    assert state.fx_node is not None
    if _resident_plan(state.fx_node) is not None:
        return _codegen_resident_load(state)

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
    if tile_index_result is not None:
        return tile_index_result

    result = pallas_codegen.load_expr(state, list(subscript), tensor)
    if state.proxy_arg(2) is not None:
        mask = state.ast_arg(2)
        # A bounded DMA may leave the padded part of its VMEM stage untouched.
        # The load API still promises zero for every masked-out element.
        expression = (
            "jnp.logical_and({mask}, {result})"
            if tensor.dtype is torch.bool
            else "jnp.where({mask}, {result}, 0)"
        )
        result = expr_from_string(expression, mask=mask, result=result)
    return result


def _codegen_resident_load(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    return pallas_codegen.resident_ref_load_expr(state, list(subscript), tensor)
