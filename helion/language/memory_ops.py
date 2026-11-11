from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.utils import normalize_index_slices
from . import _decorators
from .stack_tensor import StackTensor

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    from .._compiler.inductor_lowering import CodegenState

# TileBeginWithOffset removed - using TileBeginWithOffsetPattern instead

__all__ = ["load", "store"]

log = logging.getLogger(__name__)


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def store(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value to a tensor using a list of indices.

    This function is equivalent to `tensor[index] = value` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor / stack tensor to store to
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | tuple,
    list[object],
    torch.Tensor | torch.SymInt | float | int,
    torch.Tensor | None,
]:
    from .tile_proxy import Tile

    if isinstance(value, torch.Tensor) and value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = Tile._tiles_to_sizes_for_index(index)

    if isinstance(tensor, StackTensor):
        index = normalize_index_slices(tensor.tensor_like.shape, index)
        return (tuple(tensor), index, value, extra_mask)

    if isinstance(tensor, torch.Tensor):
        index = normalize_index_slices(tensor.shape, index)
        return (tensor, index, value, extra_mask)

    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    # Allocate the reduction dims of full slices as ``tensor[index] = value``
    # and ``hl.load`` do, so the slots of the store have blocks to address
    # them by.
    target = tensor if isinstance(tensor, torch.Tensor) else tensor[0]
    assert isinstance(target, torch.Tensor)
    SubscriptIndexing.compute_shape(target, index)
    return None


def _record_pad_info(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    block_id: int,
    extra_pad: int = 0,
) -> None:
    """Record that a tensor dimension uses pl.ds() and may need host-side padding.

    *extra_pad* accounts for non-zero loop begins: 0 when the loop starts
    at offset 0, ``begin % block_size`` for a constant begin, or
    ``block_size - 1`` for a data-dependent begin.

    Every block_id that tiles the dim is kept: two inner loops can tile the
    same dim with different block sizes, and the pad must cover both.
    """
    pad_info = state.device_function.pallas_pad_info
    blocks = pad_info.setdefault(id(tensor), {}).setdefault(tensor_dim, {})
    blocks[block_id] = max(blocks.get(block_id, 0), extra_pad)


def _ref_masked_slice_indices(
    tensor: torch.Tensor, indices: list[object]
) -> list[object]:
    """Include sliced axes in Cartesian indexing and mask broadcasting."""
    index_dtype = next(
        (index.dtype for index in indices if isinstance(index, torch.Tensor)),
        torch.int64,
    )
    return [
        torch.arange(tensor.size(dim), device=tensor.device, dtype=index_dtype)[index]
        if isinstance(index, slice)
        else index
        for dim, index in enumerate(indices)
    ]


# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    from ..runtime.ref_mode import DEVICE_LOAD_NAME
    from ..runtime.ref_mode import is_device_load
    from .ref_tile import RefTile

    if is_device_load(tensor):
        raise exc.DeviceTensorSubscriptAssignmentNotAllowed(DEVICE_LOAD_NAME)

    # Normalize indices and identify tensor indices
    indices = [idx.index if isinstance(idx, RefTile) else idx for idx in index]
    if extra_mask is not None:
        indices = _ref_masked_slice_indices(tensor, indices)
    tensor_idx_positions = [
        i for i, idx in enumerate(indices) if isinstance(idx, torch.Tensor)
    ]

    # Handle broadcasting for multiple tensor indices
    if len(tensor_idx_positions) > 1:
        grids = torch.meshgrid(
            # pyrefly: ignore [bad-argument-type]
            *(indices[i] for i in tensor_idx_positions),
            indexing="ij",
        )
        for i, grid in zip(tensor_idx_positions, grids, strict=False):
            # pyrefly: ignore [unsupported-operation]
            indices[i] = grid

    if extra_mask is not None:
        mask = extra_mask.to(torch.bool)

        # Check bounds for tensor indices
        for i, idx in enumerate(indices):
            if isinstance(idx, torch.Tensor):
                mask = mask & (idx >= 0) & (idx < tensor.shape[i])
        mask_count = int(mask.sum().item())
        if mask_count == 0:
            return

        # Use index_put_ for masked stores
        valid_indices = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                valid_indices.append(idx[mask].long())
            else:
                idx_val = int(idx) if isinstance(idx, torch.SymInt) else idx
                valid_indices.append(
                    # pyrefly: ignore [no-matching-overload]
                    torch.full(
                        (mask_count,), idx_val, dtype=torch.long, device=tensor.device
                    )
                )

        if isinstance(value, torch.Tensor):
            values = value[mask]
        else:
            val = int(value) if isinstance(value, torch.SymInt) else value
            values = torch.full(
                (mask_count,), val, dtype=tensor.dtype, device=tensor.device
            )

        # Check for duplicate indices - this is undefined behavior in Triton
        if valid_indices:
            stacked = torch.stack(valid_indices, dim=1)
            unique_count = stacked.unique(dim=0).size(0)
            if unique_count < stacked.size(0):
                raise exc.DuplicateStoreIndicesError(
                    "hl.store with duplicate indices has undefined behavior in compiled mode. "
                    "The order in which values are written to the same memory location is "
                    "non-deterministic and may vary between Triton versions and backends."
                )

        tensor.index_put_(tuple(valid_indices), values, accumulate=False)
        return

    # Simple assignment
    tensor[tuple(indices)] = (  # pyrefly: ignore[unsupported-operation]
        int(value) if isinstance(value, torch.SymInt) else value
    )


@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def load(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    This function is equivalent to `tensor[index]` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range. It also accepts an optional
    `eviction_policy` which is forwarded to the underlying Triton `tl.load`
    call to control the cache eviction behavior (e.g., "evict_last").

    Args:
        tensor: The tensor / stack tensor to load from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
        eviction_policy: Optional Triton load eviction policy to hint cache behavior
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(load)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> tuple[torch.Tensor | tuple, list[object], torch.Tensor | None, str | None]:
    from .tile_proxy import Tile

    index = Tile._tiles_to_sizes_for_index(index)
    if isinstance(tensor, StackTensor):
        index = normalize_index_slices(tensor.tensor_like.shape, index)
        return (tuple(tensor), index, extra_mask, eviction_policy)
    assert isinstance(tensor, torch.Tensor)
    index = normalize_index_slices(tensor.shape, index)
    return (tensor, index, extra_mask, eviction_policy)


@_decorators.register_fake(load)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        target_shape = SubscriptIndexing.compute_shape(tensor, index)
        env = CompileEnvironment.current()
        env.backend.process_fake_tensor_load(tensor, index)
        return env.new_index_result(tensor, target_shape)
    if isinstance(tensor, tuple):
        tensor_like, dev_ptrs = tensor
        assert isinstance(tensor_like, torch.Tensor)
        assert isinstance(dev_ptrs, torch.Tensor)
        tensor_shape = SubscriptIndexing.compute_shape(tensor_like, index)
        target_shape = list(dev_ptrs.size()) + tensor_shape
        return tensor_like.new_empty(target_shape)
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


def _maybe_materialize_tile_index_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> ast.AST | None:
    """If this load is on a ``tile.index`` value (e.g. ``tile_m.index[:, None]``),
    emit the inline ``indices_<bid>[<sub>]`` expression and return it.
    Returns ``None`` otherwise.

    ``tile.index`` tensors are synthesized inside the kernel — they aren't
    registered in ``tensor_to_origin`` — so the regular load path's
    ``tensor_arg`` lookup would ``KeyError``.  Supported subscript entries
    are ``None`` (new axis) and ``slice(None)`` (full slice).
    """
    from ..language import tile_index

    tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
    if not (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        return None

    env = CompileEnvironment.current()
    block_id = env.get_block_id(tensor.size(0))
    assert block_id is not None
    base_var = state.codegen.index_var(block_id)

    parts = []
    for idx in subscript:
        if idx is None:
            parts.append("None")
        elif idx == slice(None):
            parts.append(":")
        else:
            raise AssertionError(f"Unexpected index type in tile_index load: {idx}")
    return expr_from_string(f"{base_var}[{', '.join(parts)}]")


@_decorators.get_masked_value(load)
def _(node: torch.fx.Node) -> int:
    return 0  # loads are always masked to 0


# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(load)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    from ..runtime.ref_mode import device_load
    from .ref_tile import RefTile

    if extra_mask is None:
        # Convert RefTiles to indices
        indices = [idx.index if isinstance(idx, RefTile) else idx for idx in index]
        # Use meshgrid for Cartesian product when we have multiple tensor indices
        tensor_idxs = [
            i for i, idx in enumerate(indices) if isinstance(idx, torch.Tensor)
        ]
        if len(tensor_idxs) > 1:
            # pyrefly: ignore [bad-argument-type]
            grids = torch.meshgrid(*(indices[i] for i in tensor_idxs), indexing="ij")
            for i, grid in zip(tensor_idxs, grids, strict=False):
                indices[i] = grid
        # Integer and slice indices give a view.
        return device_load(
            # pyrefly: ignore [bad-argument-type, bad-index]
            tensor[tuple(indices)]
        )

    # Create zero result matching mask shape
    result = torch.zeros(extra_mask.shape, dtype=tensor.dtype, device=tensor.device)

    index = _ref_masked_slice_indices(
        tensor, [idx.index if isinstance(idx, RefTile) else idx for idx in index]
    )
    # Process indices: clamp tensor indices
    orig_indices, safe_indices, is_tensor_mask = [], [], []
    for i, idx in enumerate(index):
        if isinstance(idx, torch.Tensor):
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            orig_indices.append(idx)
            safe_indices.append(torch.clamp(idx, 0, dim_size - 1))
            is_tensor_mask.append(True)
        else:
            orig_indices.append(idx)
            safe_indices.append(idx)
            is_tensor_mask.append(False)

    # Apply broadcasting if we have multiple tensor indices
    tensor_positions = [i for i, is_tensor in enumerate(is_tensor_mask) if is_tensor]

    if len(tensor_positions) > 1:
        # Add unsqueeze operations for broadcasting
        broadcast_indices = []
        for i, (idx, is_tensor) in enumerate(
            zip(safe_indices, is_tensor_mask, strict=False)
        ):
            if is_tensor:
                new_idx = idx
                # Add dimension for each other tensor index
                for j, other_pos in enumerate(tensor_positions):
                    if other_pos != i:
                        new_idx = new_idx.unsqueeze(j if other_pos < i else -1)
                broadcast_indices.append(new_idx)
            else:
                broadcast_indices.append(idx)
        values = tensor[tuple(broadcast_indices)]
    else:
        values = tensor[tuple(safe_indices)]

    # Build validity mask
    valid_mask = extra_mask.clone()
    for i, (orig_idx, is_tensor) in enumerate(
        zip(orig_indices, is_tensor_mask, strict=False)
    ):
        if is_tensor:
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            in_bounds = (orig_idx >= 0) & (orig_idx < dim_size)
            # Broadcast to match mask shape by adding dimensions
            # Count how many tensor indices come before and after this one
            n_before = sum(1 for j in range(i) if is_tensor_mask[j])
            n_after = sum(
                1 for j in range(i + 1, len(is_tensor_mask)) if is_tensor_mask[j]
            )

            # Add dimensions: n_after dimensions at the end, n_before at the beginning
            for _ in range(n_after):
                in_bounds = in_bounds.unsqueeze(-1)
            for _ in range(n_before):
                in_bounds = in_bounds.unsqueeze(0)
            valid_mask = valid_mask & in_bounds

    return torch.where(valid_mask, values, result)


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def prefetch(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Hint that ``tensor[index]`` will be read soon by warming the L2 cache.

    ``index`` and ``extra_mask`` follow :func:`load` (tiles, integers,
    slices, ``hl.arange`` / integer-tensor gathers).  Nothing is returned and
    nothing is ordered: a prefetch neither waits for earlier stores nor makes
    later loads see anything, so dropping it never changes a result.  Unlike
    :func:`load` it takes no ``indexing``, eviction-policy or cache-modifier
    config slot, so adding one leaves every load's tunables where they were.

    On NVIDIA GPUs (Triton backend), when the last index is a full slice or a
    tile of a stride-1 dim and there is no ``extra_mask``, each element of the
    leading indices issues one ``cp.async.bulk.prefetch.L2`` covering its row
    segment (sm_90+).  Any other prefetch issues ``prefetch.global.L2`` per
    element (one 128-byte line each).  Other targets drop it.

    Args:
        tensor: The tensor to prefetch from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(prefetch)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[object], torch.Tensor | None]:
    from .tile_proxy import Tile

    if not isinstance(tensor, torch.Tensor):
        raise NotImplementedError(f"Cannot prefetch from type: {type(tensor)}")
    index = Tile._tiles_to_sizes_for_index(index)
    return (tensor, normalize_index_slices(tensor.shape, index), extra_mask)


def prefetch_is_bulk(
    tensor: torch.Tensor, index: Sequence[object], has_extra_mask: bool
) -> bool:
    """Whether ``hl.prefetch(tensor, index)`` issues one bulk L2 prefetch per
    element of its leading indices: the last index is a full slice or a tile
    of a stride-1 dim, there is no extra mask, and the target is sm_90+."""
    env = CompileEnvironment.current()
    capability = env.ptx_capability
    if capability is None or capability < (9, 0) or has_extra_mask:
        return False
    if tensor.ndim == 0 or len(index) != tensor.ndim:
        return False
    if any(idx is None or idx is Ellipsis for idx in index):
        return False
    stride = tensor.stride(-1)
    if not (isinstance(stride, int) and stride == 1):
        return False
    if env.known_equal(tensor.size(-1), 1):
        return False
    last = index[-1]
    if isinstance(last, slice):
        return last.start is None and last.stop is None and last.step is None
    return isinstance(last, torch.SymInt) and env.get_block_id(last) is not None


@_decorators.register_fake(prefetch)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
) -> None:
    # Allocate the reduction dims of full slices as ``hl.load`` does, except
    # a bulk prefetch's row segment, which its byte count covers.
    if prefetch_is_bulk(tensor, index, extra_mask is not None):
        index = [*index[:-1], 0]
    SubscriptIndexing.compute_shape(tensor, index)
    return None


@_decorators.codegen(prefetch, "common")
def _(state: CodegenState) -> None:
    return None  # a hint: targets without a prefetch drop it


@_decorators.ref(prefetch)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
) -> None:
    return None
