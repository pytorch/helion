from __future__ import annotations

import ast
from typing import TYPE_CHECKING

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


# Constants for ref mode
_MASKED_LOAD_VALUE = 0  # Default value for masked-out loads
_CLAMP_TO_BOUNDS = True  # Whether to clamp indices to tensor bounds


# Helper functions for ref mode implementations
def _convert_tiles_to_slices(indices: list[object]) -> list[object]:
    """Convert Tile objects to standard slices for PyTorch indexing."""
    from .ref_tile import RefTile

    return [idx._slice if isinstance(idx, RefTile) else idx for idx in indices]


def _normalize_indices(indices: slice | list | tuple) -> slice | tuple:
    """Normalize indices to a consistent format for PyTorch operations."""
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
    """Combine two masks using logical AND, handling None values."""
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 & mask2


def _apply_mask(
    result: torch.Tensor,
    mask: torch.Tensor | None,
    masked_value: float | torch.Tensor = _MASKED_LOAD_VALUE,
) -> torch.Tensor:
    """Apply a mask to a tensor, replacing masked-out values."""
    if mask is None:
        return result

    # Handle empty tensors
    if mask.numel() == 0 or result.numel() == 0:
        return torch.zeros(mask.shape, dtype=result.dtype, device=result.device)

    # Apply mask with broadcasting support
    return torch.where(mask, result, masked_value)


def _handle_single_tensor_index(
    tensor: torch.Tensor, idx_tensor: torch.Tensor, extra_mask: torch.Tensor | None
) -> torch.Tensor:
    """Handle indexing with a single tensor index (jagged array case).

    Args:
        tensor: Source tensor to index into
        idx_tensor: Tensor containing indices
        extra_mask: Optional mask to apply to results

    Returns:
        Indexed tensor with same shape as idx_tensor
    """
    # Flatten for easier processing
    flat_indices = idx_tensor.flatten()

    # Clamp indices to valid range
    if _CLAMP_TO_BOUNDS and tensor.shape[0] > 0:
        clamped_indices = torch.clamp(flat_indices, 0, tensor.shape[0] - 1)
    else:
        clamped_indices = flat_indices

    # Gather values
    gathered = tensor[clamped_indices]

    # Simple case: no masking needed
    if extra_mask is None:
        return gathered.reshape(idx_tensor.shape)

    # Apply mask and reshape
    flat_mask = extra_mask.flatten()
    result = torch.zeros_like(gathered)
    result = torch.where(flat_mask, gathered, result)
    return result.reshape(idx_tensor.shape)


def _compute_slice_shape(idx: slice, dim_size: int) -> int:
    """Compute the output size for a slice index."""
    if idx.start is None and idx.stop is None:
        return dim_size
    start = idx.start or 0
    stop = idx.stop or dim_size
    return max(0, stop - start)


def _handle_mixed_indices(
    tensor: torch.Tensor, indices: tuple, extra_mask: torch.Tensor | None
) -> torch.Tensor:
    """Handle mixed indexing with slices and tensors.

    Args:
        tensor: Source tensor to index
        indices: Tuple of indices (slices, tensors, or integers)
        extra_mask: Optional mask to apply

    Returns:
        Indexed tensor result
    """
    expected_shape = []
    processed_indices = []

    # Process each index and compute expected output shape
    for i, idx in enumerate(indices):
        dim_size = tensor.shape[i] if i < len(tensor.shape) else 1

        if isinstance(idx, slice):
            # Slice index: contributes one dimension to output
            expected_shape.append(_compute_slice_shape(idx, dim_size))
            processed_indices.append(idx)

        elif isinstance(idx, torch.Tensor):
            # Tensor index: contributes its shape to output
            expected_shape.extend(idx.shape)
            if _CLAMP_TO_BOUNDS and dim_size > 0:
                clamped_idx = torch.clamp(idx, 0, dim_size - 1)
                processed_indices.append(clamped_idx)
            else:
                processed_indices.append(idx)

        else:
            # Integer index: doesn't contribute to output shape
            processed_indices.append(idx)

    # Perform indexing with error handling
    try:
        result = tensor[tuple(processed_indices)]

        # Pad result if needed for shape consistency
        if extra_mask is not None and result.shape != tuple(expected_shape):
            result = _pad_result_to_expected_shape(
                result, expected_shape, tensor.dtype, tensor.device
            )

        return result

    except (RuntimeError, IndexError):
        # Return zeros for invalid indexing
        return torch.zeros(expected_shape, dtype=tensor.dtype, device=tensor.device)


def _pad_result_to_expected_shape(
    result: torch.Tensor,
    expected_shape: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Pad result tensor with zeros to match expected shape.

    Args:
        result: Tensor to pad
        expected_shape: Target shape
        dtype: Data type for padded tensor
        device: Device for padded tensor

    Returns:
        Padded tensor matching expected_shape
    """
    if list(result.shape) == expected_shape:
        return result

    padded = torch.zeros(expected_shape, dtype=dtype, device=device)

    if result.numel() > 0:
        # Copy valid data to padded result
        valid_slices = tuple(
            slice(0, min(s, e))
            for s, e in zip(result.shape, expected_shape, strict=True)
        )
        padded[valid_slices] = result[: padded[valid_slices].shape[0]]

    return padded


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
    """Reference implementation of store operation.

    Performs in-place storage with optional masking.
    Masked-out elements retain their original values.
    """
    # Convert Tile objects to slices and normalize
    indices = _convert_tiles_to_slices(indices)
    normalized_indices = _normalize_indices(indices)

    # Perform the store operation
    _perform_masked_store(tensor, normalized_indices, value, extra_mask)


def _perform_masked_store(
    tensor: torch.Tensor,
    indices: slice | tuple,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    """Execute the store operation with optional masking."""
    if mask is None:
        # Simple store without masking
        tensor[indices] = value
    else:
        # Masked store: preserve original values where mask is False
        current_values = tensor[indices]
        tensor[indices] = torch.where(mask, value, current_values)


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
    """Reference implementation of load operation.

    Handles various indexing patterns including:
    - Single tensor index (jagged arrays)
    - Mixed indices with slices and tensors
    - Masked loading with zero-fill for out-of-bounds
    """
    assert isinstance(indices, (list, tuple)), "Indices must be a list or tuple"

    # Convert Tile objects to slices
    indices = _convert_tiles_to_slices(indices)

    # Determine indexing pattern and load data
    result = _perform_indexed_load(tensor, indices, extra_mask)

    # Apply masking if needed
    return _apply_mask(result, extra_mask, _MASKED_LOAD_VALUE)


def _perform_indexed_load(
    tensor: torch.Tensor, indices: list[object], extra_mask: torch.Tensor | None
) -> torch.Tensor:
    """Perform the actual indexed load based on index types."""
    # Single tensor index case (jagged indexing)
    if _is_single_tensor_index(indices):
        return _handle_single_tensor_index(tensor, indices[0], extra_mask)

    # Mixed indices case (slices, tensors, integers)
    if _has_slice_indices(indices):
        return _handle_mixed_indices(tensor, tuple(indices), extra_mask)

    # Invalid indexing pattern
    raise exc.InvalidIndexingType(
        f"Invalid indices: {indices}. Expected slices or tensors."
    )


def _is_single_tensor_index(indices: list[object]) -> bool:
    """Check if indices represent a single tensor index pattern."""
    return len(indices) == 1 and isinstance(indices[0], torch.Tensor)


def _has_slice_indices(indices: list[object]) -> bool:
    """Check if indices contain any slice objects."""
    return any(isinstance(idx, slice) for idx in indices)


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
    """Reference implementation of atomic add operation.

    Performs atomic addition with special optimizations for
    scatter-add patterns commonly used in segment operations.

    Note: In ref mode, atomicity is simulated; true atomicity
    requires the compiled kernel.
    """
    # Convert Tile objects to slices
    indices = _convert_tiles_to_slices(indices)

    # Try optimized scatter-add pattern first
    if _try_scatter_add_optimization(tensor, indices, value):
        return

    # Fall back to standard atomic add
    _perform_standard_atomic_add(tensor, indices, value)


def _try_scatter_add_optimization(
    tensor: torch.Tensor, indices: list[object], value: torch.Tensor | float
) -> bool:
    """Attempt to use optimized scatter-add for specific patterns.

    Returns:
        True if optimization was applied, False otherwise
    """
    # Check for scatter-add pattern: tensor[tensor_idx, slice] += value
    if not _is_scatter_add_pattern(indices):
        return False

    tensor_idx, slice_idx = indices

    # Extract the slice bounds
    start = slice_idx.start or 0
    stop = slice_idx.stop or tensor.shape[1]

    # Create view and perform optimized addition
    tensor_view = tensor[:, start:stop]

    if isinstance(value, (float, int)):
        # Scalar value: create tensor and use index_add_
        value_tensor = torch.full(
            (1, tensor_view.shape[1]), value, dtype=tensor.dtype, device=tensor.device
        )
        tensor_view.index_add_(
            0, tensor_idx, value_tensor.expand(tensor_idx.shape[0], -1)
        )
    else:
        # Tensor value: use index_add_ directly
        tensor_view.index_add_(0, tensor_idx, value)

    return True


def _is_scatter_add_pattern(indices: list[object]) -> bool:
    """Check if indices match the scatter-add pattern."""
    return (
        isinstance(indices, (list, tuple))
        and len(indices) == 2
        and isinstance(indices[0], torch.Tensor)
        and isinstance(indices[1], slice)
    )


def _perform_standard_atomic_add(
    tensor: torch.Tensor, indices: list[object], value: torch.Tensor | float
) -> None:
    """Perform standard atomic addition without optimization."""
    normalized_indices = _normalize_indices(indices)
    tensor[normalized_indices] += value
