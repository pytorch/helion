"""Triton-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language.memory_ops import _maybe_materialize_tile_index_load
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


# Map short config names to full Triton API names for eviction policies
_EVICTION_POLICY_MAP = {
    "": None,
    "first": "evict_first",
    "last": "evict_last",
}


@_decorators.codegen(store, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        fx_node = state.fx_node
        assert fx_node is not None
        epilogue_subtile_group_id = fx_node.meta.get("epilogue_subtile_group_id")
        if epilogue_subtile_group_id is None:
            indexing_idx = device_fn.allocate_store_index()
        elif fx_node.meta.get("epilogue_subtile_primary_output", False):
            indexing_idx = device_fn.allocate_store_index()
            device_fn.epilogue_subtile_store_indices[epilogue_subtile_group_id] = (
                indexing_idx
            )
        else:
            indexing_idx = device_fn.epilogue_subtile_store_indices[
                epilogue_subtile_group_id
            ]
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        cache_modifier = None
        if state.codegen.on_device:
            modifier_idx = device_fn.device_store_cache_modifier_index
            device_fn.device_store_cache_modifier_index += 1
            modifiers = state.config.store_cache_modifiers
            if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
                cache_modifier = ast.Constant(value=modifiers[modifier_idx])

        if state.codegen.store_transform is not None:
            return state.codegen.store_transform(
                state,
                tensor,
                [*subscript],
                value,
                extra_mask,
                cache_modifier,
                strategy.codegen_store,
            )

        return strategy.codegen_store(
            state, tensor, [*subscript], value, extra_mask, cache_modifier
        )
    if isinstance(tensor, tuple):
        from ..indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack stores (multi-tensor device pointers);
        # fall through to the unfused path regardless of store_transform.
        device_fn = state.device_function
        device_fn.allocate_store_index()
        cache_modifier = None
        if state.codegen.on_device:
            modifier_idx = device_fn.device_store_cache_modifier_index
            device_fn.device_store_cache_modifier_index += 1
            modifiers = state.config.store_cache_modifiers
            if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
                cache_modifier = ast.Constant(value=modifiers[modifier_idx])
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask, cache_modifier
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.codegen(load, "triton")
def _(state: CodegenState) -> ast.AST:
    # A store to this tensor earlier in the same loop body followed by this load
    # is a cross-thread read-after-write on global memory; emit a barrier so the
    # store is visible before the read (see mark_intra_loop_raw_barriers).
    from ..loop_dependency_checker import INTRA_LOOP_RAW_BARRIER_META

    if state.fx_node is not None and state.fx_node.meta.get(
        INTRA_LOOP_RAW_BARRIER_META
    ):
        state.add_statement(statement_from_string("tl.debug_barrier()"))

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None

    device_fn = state.device_function
    load_idx = device_fn.device_load_index
    device_fn.device_load_index += 1

    # If no explicit eviction_policy and we're in device code, use tunable
    if eviction_policy is None and state.codegen.on_device:
        policies = state.config.load_eviction_policies
        if load_idx < len(policies):
            policy_value = policies[load_idx]
            eviction_policy = _EVICTION_POLICY_MAP.get(policy_value, policy_value)

    if eviction_policy is not None:
        assert isinstance(eviction_policy, str)
        eviction_policy = ast.Constant(value=eviction_policy)

    cache_modifier = None
    if state.codegen.on_device:
        modifier_idx = device_fn.device_load_cache_modifier_index
        device_fn.device_load_cache_modifier_index += 1
        modifiers = state.config.load_cache_modifiers
        if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
            cache_modifier = ast.Constant(value=modifiers[modifier_idx])

    if isinstance(tensor, torch.Tensor):
        tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
        if tile_index_result is not None:
            return tile_index_result

        # Use the shared memory op index for indexing strategy
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)

        if state.codegen.load_transform is not None:
            return state.codegen.load_transform(
                state,
                tensor,
                [*subscript],
                extra_mask,
                eviction_policy,
                cache_modifier,
                strategy.codegen_load,
            )

        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, cache_modifier
        )
    if isinstance(tensor, tuple):
        from ..indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack loads (multi-tensor device pointers);
        # fall through to the unfused path regardless of load_transform.
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state,
            tensor,
            dev_ptrs_ast,
            [*subscript],
            extra_mask,
            eviction_policy,
            cache_modifier,
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")
