from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = ["atomic_add", "load", "store"]


# Helper functions for ref mode implementations
def _normalize_indices(indices: slice | list | tuple) -> slice | tuple:
    if isinstance(indices, slice):
        return slice(indices.start, indices.stop)
    if isinstance(indices, (list, tuple)):
        return tuple(
            slice(idx.start, idx.stop) if isinstance(idx, slice) else idx
            for idx in indices
        )
    return indices


def _combine_masks(
    mask1: torch.Tensor | None, mask2: torch.Tensor | None
) -> torch.Tensor | None:
    if mask1 is not None and mask2 is not None:
        return mask1 & mask2
    return mask1 or mask2


def _apply_mask(
    result: torch.Tensor, mask: torch.Tensor | None, other: Any = 0
) -> torch.Tensor:
    if mask is None:
        return result

    # Handle shape mismatch
    if result.shape != mask.shape:
        if mask.numel() == 0 or result.numel() == 0:
            return torch.zeros(mask.shape, dtype=result.dtype, device=result.device)
        # Let torch handle broadcasting

    return torch.where(mask, result, other)


def _handle_single_tensor_index(
    tensor: torch.Tensor, idx_tensor: torch.Tensor, extra_mask: torch.Tensor | None
) -> torch.Tensor:
    """Handle indexing with a single tensor index (jagged array case)."""
    flat_indices = idx_tensor.flatten()
    clamped_indices = torch.clamp(flat_indices, 0, tensor.shape[0] - 1)

    if extra_mask is None:
        return tensor[clamped_indices].reshape(idx_tensor.shape)

    # Apply mask to filter valid indices
    valid_mask = extra_mask.flatten()
    gathered = tensor[clamped_indices]
    result = torch.zeros(idx_tensor.shape, dtype=tensor.dtype, device=tensor.device)
    result_flat = result.flatten()
    result_flat = torch.where(valid_mask, gathered, result_flat)
    return result_flat.reshape(idx_tensor.shape)


def _handle_mixed_indices(
    tensor: torch.Tensor, indices: tuple, extra_mask: torch.Tensor | None
) -> torch.Tensor:
    """Handle mixed indexing with slices and tensors."""
    expected_shape = []
    actual_indices = []
    tensor_shape = tensor.shape

    # Build expected output shape and process indices
    for i, idx in enumerate(indices):
        if isinstance(idx, slice):
            # Handle slice indices
            if idx.start is None and idx.stop is None:
                # Full slice like `:` 
                shape_size = tensor_shape[i] if i < len(tensor_shape) else 1
            else:
                start = idx.start or 0
                stop = idx.stop or (tensor_shape[i] if i < len(tensor_shape) else 1)
                shape_size = stop - start
            expected_shape.append(shape_size)
            actual_indices.append(idx)
        elif isinstance(idx, torch.Tensor):
            # Handle tensor indices - clamp to valid range
            expected_shape.extend(idx.shape)
            max_index = tensor_shape[i] - 1 if i < len(tensor_shape) else 0
            clamped_idx = torch.clamp(idx, 0, max_index)
            actual_indices.append(clamped_idx)
        else:
            # Regular integer index
            actual_indices.append(idx)

    # Perform indexing with error handling
    try:
        result = tensor[tuple(actual_indices)]

        # Handle shape mismatch when using extra_mask
        if extra_mask is not None and result.shape != tuple(expected_shape):
            result = _pad_result_to_expected_shape(
                result, expected_shape, tensor.dtype, tensor.device
            )

        return result
    except (RuntimeError, IndexError):
        # Return zeros if indexing fails (e.g., negative indices)
        return torch.zeros(expected_shape, dtype=tensor.dtype, device=tensor.device)


def _pad_result_to_expected_shape(
    result: torch.Tensor,
    expected_shape: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Pad result tensor with zeros to match expected shape."""
    padded_result = torch.zeros(expected_shape, dtype=dtype, device=device)

    if result.numel() > 0:
        # Copy valid data to padded result
        slices = [slice(0, s) for s in result.shape]
        padded_result[tuple(slices)] = result

    return padded_result


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def store(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value to a tensor using a list of indices.

    This function is equivalent to `tensor[index] = value` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor to store to
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, list[object], torch.Tensor | torch.SymInt | float, torch.Tensor | None
]:
    from .tile_proxy import Tile

    if isinstance(value, torch.Tensor) and value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = Tile._tiles_to_sizes(index)
    return (tensor, index, value, extra_mask)


@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    return None


@_decorators.codegen(store)
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    return state.device_function.indexing_strategy.codegen_store(
        state, tensor, [*subscript], value, extra_mask
    )


@_decorators.ref(store)
def _(
    tensor: torch.Tensor,
    indices: list[object],
    value: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> None:
    # Convert RefTile objects to slices
    from .tile_proxy import RefTile
    processed_indices = []
    for idx in indices:
        if isinstance(idx, RefTile):
            processed_indices.append(idx._slice)
        else:
            processed_indices.append(idx)
    indices = processed_indices
    
    normalized_indices = _normalize_indices(indices)

    if extra_mask is not None:
        current = tensor[normalized_indices]
        tensor[normalized_indices] = torch.where(extra_mask, value, current)
    else:
        tensor[normalized_indices] = value


@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def load(
    tensor: torch.Tensor, index: list[object], extra_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    This function is equivalent to `tensor[index]` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor to load from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(load)
def _(
    tensor: torch.Tensor, index: list[object], extra_mask: torch.Tensor | None = None
) -> torch.Tensor:
    return tensor.new_empty(SubscriptIndexing.compute_shape(tensor, index))


@_decorators.codegen(load)
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    return state.device_function.indexing_strategy.codegen_load(
        state, tensor, [*subscript], extra_mask
    )


@_decorators.get_masked_value(load)
def _(node: torch.fx.Node) -> int:
    return 0  # loads are always masked to 0


@_decorators.ref(load)
def _(
    tensor: torch.Tensor,
    indices: list[object],
    extra_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Combined mask handling is done inside load logic
    mask = None  # No base mask for ref mode
    other = 0

    assert isinstance(indices, (list, tuple))
    
    # Convert RefTile objects to slices
    from .tile_proxy import RefTile
    processed_indices = []
    for idx in indices:
        if isinstance(idx, RefTile):
            processed_indices.append(idx._slice)
        else:
            processed_indices.append(idx)
    indices = processed_indices

    # Case 1: Single tensor index (jagged indexing)
    if len(indices) == 1 and isinstance(indices[0], torch.Tensor):
        result = _handle_single_tensor_index(tensor, indices[0], extra_mask)

    # Case 2: Mixed indices containing slices (tiles)
    elif any(isinstance(idx, slice) for idx in indices):
        result = _handle_mixed_indices(tensor, tuple(indices), extra_mask)
    else:
        raise exc.InvalidIndexingType(
            f"Invalid indices type: {indices}. Expected a list of slices or tensors."
        )

    # Apply mask
    combined_mask = _combine_masks(mask, extra_mask)
    return _apply_mask(result, combined_mask, other)


@has_side_effect
@_decorators.api(allow_host_tensor=True)
def atomic_add(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> None:
    """
    Atomically add a value to a target tensor.

    Performs an atomic read-modify-write operation that adds value to target[index].
    This is safe for concurrent access from multiple threads/blocks.

    Args:
        target: The tensor to add to
        index: Indices into target for accumulating values
        value: The value to add (tensor or scalar)
        sem: Memory ordering semantics (default: 'relaxed')
            - 'relaxed': No ordering constraints
            - 'acquire': Acquire semantics
            - 'release': Release semantics
            - 'acq_rel': Acquire-release semantics

    Returns:
        None

    Examples:
        .. code-block:: python

            @helion.kernel
            def global_sum(x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
                # Each tile computes local sum, then atomically adds to global
                for tile in hl.tile(x.size(0)):
                    local_data = x[tile]
                    local_sum = local_data.sum()
                    hl.atomic_add(result, [0], local_sum)

                return result

    See Also:
        - :func:`~helion.language.store`: For non-atomic stores
        - :func:`~helion.language.load`: For atomic loads

    Note:
        - Required for race-free accumulation across parallel execution
        - Performance depends on memory access patterns and contention
        - Consider using regular operations when atomicity isn't needed
        - Higher memory semantics (acquire/release) have performance overhead
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, torch.Tensor | float | int, str]:
    from .tile_proxy import Tile

    valid_sems = {"relaxed", "acquire", "release", "acq_rel"}
    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    return (target, index, value, sem)


@_decorators.register_fake(atomic_add)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> None:
    return None


@_decorators.codegen(atomic_add)
def _(state: CodegenState) -> ast.AST:
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(3)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    value_expr = state.ast_args[2]
    if isinstance(value_expr, (int, float, bool)):
        value_expr = expr_from_string(constant_repr(value_expr))
    assert isinstance(value_expr, ast.AST)
    return expr_from_string(
        f"tl.atomic_add({name} + offset, value, mask=mask, sem=sem)",
        value=value_expr,
        offset=indices.index_expr,
        mask=indices.mask_expr,
        sem=sem,
    )


@_decorators.ref(atomic_add)
def _(
    tensor: torch.Tensor,
    indices: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> None:
    # Convert RefTile objects to slices
    from .tile_proxy import RefTile
    processed_indices = []
    for idx in indices:
        if isinstance(idx, RefTile):
            processed_indices.append(idx._slice)
        else:
            processed_indices.append(idx)
    indices = processed_indices
    
    # Special handling for scatter-add pattern (`tensor[tensor_idx, slice] += value`)
    if isinstance(indices, (list, tuple)) and len(indices) == 2:
        idx0, idx1 = indices
        if isinstance(idx0, torch.Tensor) and isinstance(idx1, slice):
            # This is the pattern: output[idxs, tile_f] += segment_vals
            start = idx1.start or 0
            stop = idx1.stop or tensor.shape[1]
            tensor_view = tensor[:, start:stop]
            tensor_view.index_add_(0, idx0, value)
            return

    # Default case
    normalized_indices = _normalize_indices(indices)
    tensor[normalized_indices] += value
