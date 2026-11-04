"""Metal-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "metal")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import SubscriptIndexing

if TYPE_CHECKING:
    from ..indexing_strategy import PerDimIndexing
    from ..inductor_lowering import CodegenState


def _position_lane_indices(
    state: CodegenState, tensor: torch.Tensor, subscript: list[object]
) -> dict[int, str]:
    """Tensor dim -> index of each slice dim of an access that sits at a
    position lane of its block (``repeated_blocks``); empty for none.

    Such an access is emitted here, not by ``PointerIndexingStrategy``, so
    the strategy's index for that dim can be replaced.  What it adds besides
    can't apply: the store's scalar-pointer reshape (a position dim keeps the
    offset block-shaped), FP8's float ``other`` (Metal has no FP8), and the
    eviction and contiguity hints (Metal's walker drops them).

    The shared mask still tests the block's thread index on a position dim.
    That only drops repeated work: a thread past the block's extent has every
    access to the block masked, and the threads within it cover every lane.
    """
    from .reduction_strategy import MetalPersistentReductionStrategy
    from .repeated_blocks import slice_output_dims
    from .repeated_blocks import slice_start_step

    env = CompileEnvironment.current()
    shape = SubscriptIndexing.compute_shape(tensor, subscript, state)

    def lane(out_dim: int) -> str | None:
        block_id = env.get_block_id(shape[out_dim])
        if block_id is None:
            return None
        strategy = state.tile_strategy.block_id_to_strategy.get((block_id,))
        if not isinstance(strategy, MetalPersistentReductionStrategy):
            return None
        return strategy.position_lane(len(shape) - 1 - out_dim)

    dims = slice_output_dims(subscript)
    if dims is None:
        if any(lane(out_dim) for out_dim in range(len(shape))):
            raise exc.BackendUnsupported(
                "metal",
                f"a tensor index beside a repeated `:` slice block: {subscript}",
            )
        return {}
    indices: dict[int, str] = {}
    for out_dim, (tensor_dim, k) in dims.items():
        if (lane_var := lane(out_dim)) is not None:
            start, step = (
                state.device_function.literal_expr(v) for v in slice_start_step(k)
            )
            indices[tensor_dim] = f"({start} + ({lane_var}) * {step})"
    return indices


def _position_lane_offset(
    state: CodegenState,
    tensor: torch.Tensor,
    per_dim: PerDimIndexing,
    lanes: dict[int, str],
) -> ast.AST:
    """``SubscriptIndexing.create``'s offset, with the position lanes' indices."""
    env = CompileEnvironment.current()
    return expr_from_string(
        "+".join(
            f"{lanes.get(i, index)} * "
            f"{state.device_function.tensor_stride(tensor, i).name}"
            for i, index in enumerate(per_dim.dim_index_exprs)
            if not env.known_equal(tensor.size(i), 1)
        )
    )


def _position_lane_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
    lanes: dict[int, str],
) -> ast.AST:
    per_dim = SubscriptIndexing.compute_per_dim_indexing(
        state, tensor, subscript, extra_mask
    )
    loaded = expr_from_string(
        f"tl.load({state.device_function.tensor_arg(tensor).name} + {{offset}}, "
        "{mask}, other=0)",
        offset=_position_lane_offset(state, tensor, per_dim, lanes),
        mask=per_dim.mask_expr,
    )
    if per_dim.broadcast_mask_expr is None:
        return loaded
    backend = CompileEnvironment.current().backend
    return expr_from_string(
        backend.where_expr(
            "{mask}", "{loaded}", backend.full_expr([], "0", tensor.dtype)
        ),
        mask=per_dim.broadcast_mask_expr,
        loaded=loaded,
    )


def _position_lane_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    lanes: dict[int, str],
) -> ast.AST:
    per_dim = SubscriptIndexing.compute_per_dim_indexing(
        state, tensor, subscript, extra_mask, for_store=True
    )
    return expr_from_string(
        f"tl.store({state.device_function.tensor_arg(tensor).name} + {{offset}}, "
        "{value}, {mask})",
        offset=_position_lane_offset(state, tensor, per_dim, lanes),
        value=value,
        mask=per_dim.mask_expr,
    )


@_decorators.codegen(store, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton, except
    # at a position lane.  This produces tl.store(ptr + offset, val, mask);
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_store_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        if lanes := _position_lane_indices(state, tensor, [*subscript]):
            return _position_lane_store(
                state, tensor, [*subscript], value, extra_mask, lanes
            )
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_store(
            state, tensor, [*subscript], value, extra_mask, None
        )
    raise exc.BackendUnsupported("metal", f"store target type: {type(tensor)}")


@_decorators.codegen(load, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton, except
    # at a position lane.  This produces tl.load(ptr + offset, mask, other=0);
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None
    assert isinstance(eviction_policy, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_load_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        if lanes := _position_lane_indices(state, tensor, [*subscript]):
            return _position_lane_load(state, tensor, [*subscript], extra_mask, lanes)
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, None
        )
    raise exc.BackendUnsupported("metal", f"load tensor type: {type(tensor)}")
