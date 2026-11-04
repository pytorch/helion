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
from ...language.memory_ops import prefetch
from ...language.memory_ops import prefetch_is_bulk
from ...language.memory_ops import store
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import SubscriptIndexing
from ..indexing_strategy import _resolve_codegen_block_id
from .distributed_ops import inband_load_codegen
from .distributed_ops import inband_store_codegen

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

        codegen_store = inband_store_codegen(state, strategy.codegen_store)
        if state.codegen.store_transform is not None:
            return state.codegen.store_transform(
                state,
                tensor,
                [*subscript],
                value,
                extra_mask,
                cache_modifier,
                codegen_store,
            )

        return codegen_store(
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
        for statement in state.device_function.cta_barrier():
            state.add_statement(statement)

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None

    device_fn = state.device_function

    # If no explicit eviction_policy and we're in device code, use tunable.
    # Only such loads own a load_eviction_policies slot (see
    # _load_needs_eviction_tunable).
    if eviction_policy is None and state.codegen.on_device:
        load_idx = device_fn.device_load_index
        device_fn.device_load_index += 1
        policies = state.config.load_eviction_policies
        if isinstance(policies, str):
            policy_value = policies
        elif load_idx < len(policies):
            policy_value = policies[load_idx]
        else:
            policy_value = None
        if policy_value is not None:
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

        codegen_load = inband_load_codegen(state, strategy.codegen_load)
        if state.codegen.load_transform is not None:
            return state.codegen.load_transform(
                state,
                tensor,
                [*subscript],
                extra_mask,
                eviction_policy,
                cache_modifier,
                codegen_load,
            )

        return codegen_load(
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


def _prefetch_asm(*, bulk: bool, scalar: bool) -> str:
    """PTX for one L2 prefetch per element of the pointer operand ($1).

    The plain form issues ``prefetch.global.L2`` unless the int32 predicate
    ($2) is 0.  The bulk form's $2 is a byte count instead, skipped unless
    positive, and widens [$1, $1 + $2) to the 16-byte granules
    ``cp.async.bulk`` requires (a granule never crosses a page).  A scalar
    prefetch is issued by thread 0 alone instead of by every thread.
    """
    if bulk:
        regs = ".reg .pred p; .reg .b32 t, n; .reg .b64 a, e;"
        test = "setp.gt.s32 p, $2, 0;"
        issue = (
            "and.b64 a, $1, -16; cvt.s64.s32 e, $2; add.s64 e, e, $1; "
            "add.s64 e, e, 15; and.b64 e, e, -16; sub.s64 e, e, a; "
            "cvt.u32.u64 n, e; @p cp.async.bulk.prefetch.L2.global [a], n;"
        )
    else:
        regs = ".reg .pred p; .reg .b32 t;"
        test = "setp.ne.b32 p, $2, 0;"
        issue = "@p prefetch.global.L2 [$1];"
    if scalar:
        test += " mov.u32 t, %tid.x; setp.eq.and.u32 p, t, 0, p;"
    return f"{{ {regs} {test} {issue} }} mov.u32 $0, 0;"


def _prefetch_row_segment(
    state: CodegenState, tensor: torch.Tensor, last: object
) -> tuple[str, str] | None:
    """(first element, element count) of the row segment a bulk prefetch's
    last index covers, or None when its tile has no block size variable."""
    size = state.device_function.tensor_size(tensor, tensor.ndim - 1).name
    if isinstance(last, slice):
        return "0", size
    assert isinstance(last, torch.SymInt)
    block_id = CompileEnvironment.current().get_block_id(last)
    assert block_id is not None
    block_id = _resolve_codegen_block_id(state, block_id)
    block = state.device_function.block_size_var(block_id)
    if block is None:
        return None
    start = state.codegen.offset_var(block_id)
    if state.codegen.mask_var(block_id) is None:
        return start, block
    return start, f"tl.minimum({block}, {size} - {start})"


@_decorators.codegen(prefetch, "triton")
def _(state: CodegenState) -> None:
    if CompileEnvironment.current().ptx_capability is None:
        return None  # both prefetch instructions are PTX
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    name = state.device_function.tensor_arg(tensor).name
    segment = None
    if prefetch_is_bulk(tensor, [*subscript], extra_mask is not None):
        segment = _prefetch_row_segment(state, tensor, subscript[-1])
    if segment is not None:
        # One bulk prefetch per element of the leading indices.
        start, count = segment
        index = [*subscript[:-1], 0]
        indexing = SubscriptIndexing.create(state, tensor, index)
        nbytes = f"({count}) * {tensor.element_size()}"
        operand = f"tl.where({{mask}}, {nbytes}, 0)" if indexing.has_mask() else nbytes
        pointer = f"{name} + ({{offset}} + {start})"
    else:
        index = [*subscript]
        indexing = SubscriptIndexing.create(state, tensor, index, extra_mask)
        operand = "{mask}" if indexing.has_mask() else "1"
        pointer = f"{name} + {{offset}}"
    scalar = extra_mask is None and not SubscriptIndexing.compute_shape(
        tensor, index, state
    )
    asm = _prefetch_asm(bulk=segment is not None, scalar=scalar)
    state.add_statement(
        statement_from_string(
            f"tl.inline_asm_elementwise({asm!r}, '=r,l,r', "
            f"[{pointer}, tl.cast({operand}, tl.int32)], "
            "dtype=tl.int32, is_pure=False, pack=1)",
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )
    )
    return None
