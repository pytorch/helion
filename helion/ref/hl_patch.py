"""Patched Helion language APIs (hl.*) used in ref mode"""

from __future__ import annotations

# Import builtins
import builtins
import itertools
from typing import Any
from typing import Callable
from typing import Iterator

import torch
from torch import Tensor

from helion import exc

_builtin_int = builtins.int
_builtin_list = builtins.list
_builtin_tuple = builtins.tuple
_builtin_range = builtins.range
_builtin_min = builtins.min
_builtin_max = builtins.max
_builtin_any = builtins.any


def _normalize_shape(shape: list[int | slice]) -> list[int]:
    return [s.stop - s.start if isinstance(s, slice) else s for s in shape]


def _normalize_indices(indices: Any) -> Any:  # noqa: ANN401
    if isinstance(indices, slice):
        return slice(indices.start, indices.stop)
    if isinstance(indices, (_builtin_list, _builtin_tuple)):
        return _builtin_tuple(
            slice(idx.start, idx.stop) if isinstance(idx, slice) else idx
            for idx in indices
        )
    return indices


def _apply_mask(result: Tensor, mask: Tensor | None, other: Any = 0) -> Tensor:  # noqa: ANN401
    if mask is None:
        return result

    # Handle shape mismatch
    if result.shape != mask.shape:
        if mask.numel() == 0 or result.numel() == 0:
            return torch.zeros(mask.shape, dtype=result.dtype, device=result.device)
        # Let torch handle broadcasting

    return torch.where(mask, result, other)


def _combine_masks(mask1: Tensor | None, mask2: Tensor | None) -> Tensor | None:
    if mask1 is not None and mask2 is not None:
        return mask1 & mask2
    return mask1 or mask2


def zeros(shape: list[int | slice], dtype: torch.dtype = torch.float32) -> Tensor:
    processed_shape = _normalize_shape(shape)
    return torch.zeros(processed_shape, dtype=dtype, device="cuda")


def full(
    shape: list[int | slice],
    value: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    processed_shape = _normalize_shape(shape)
    return torch.full(processed_shape, value, dtype=dtype, device="cuda")


def arange(*args: int, dtype: torch.dtype | None = None, **kwargs: Any) -> Tensor:  # noqa: ANN401
    if dtype is None:
        dtype = torch.int64
    return torch.arange(*args, dtype=dtype, device="cuda", **kwargs)


def tile(
    begin_or_end: int | Tensor | list[int | Tensor],
    end_or_none: int | Tensor | list[int | Tensor] | None = None,
    /,
    block_size: int | Tensor | list[int | Tensor] | None = None,
) -> Iterator[slice | tuple[slice, ...]]:
    # Step 1: Normalize begin and end values based on the number of arguments
    if end_or_none is not None:
        # Two positional args: begin_or_end is begin, end_or_none is end
        begin = begin_or_end
        end = end_or_none
    else:
        # One positional arg: begin_or_end is end, begin defaults to 0
        end = begin_or_end
        # Create begin with same structure as end, but all zeros
        if isinstance(end, (_builtin_list, _builtin_tuple)):
            begin = [0] * len(end)
        else:
            begin = 0

    # Step 2: Convert inputs to lists for uniform handling
    begin_list = _normalize_to_list(begin)  # type: ignore[arg-type]
    end_list = _normalize_to_list(end)  # type: ignore[arg-type]

    # Validate that begin and end have matching dimensions
    if len(begin_list) != len(end_list):
        raise ValueError(
            f"Mismatched dimensions: begin has {len(begin_list)}, end has {len(end_list)}"
        )

    # Step 3: Handle block_size (in ref mode, full dim size is always used as block_size)
    block_size_list = [None] * len(end_list)

    # Step 4: Generate tiles for each dimension
    dim_tiles = []
    for begin_val, end_val, _bs in zip(
        begin_list, end_list, block_size_list, strict=False
    ):
        # Convert to int for ref mode
        # This handles both regular ints and dynamic tensor values
        if isinstance(begin_val, torch.Tensor):
            assert begin_val.numel() == 1, (
                "begin_val tensor must have a single element in ref mode"
            )
            begin_val = int(begin_val.item())
        else:
            begin_val = int(begin_val)

        if isinstance(end_val, torch.Tensor):
            assert end_val.numel() == 1, (
                "end_val tensor must have a single element in ref mode"
            )
            end_val = int(end_val.item())
        else:
            end_val = int(end_val)

        if begin_val < end_val:
            tiles = [slice(begin_val, end_val)]
            dim_tiles.append(tiles)

    # If no valid tiles were created (all ranges were empty), return early
    if not dim_tiles:
        return

    # Step 5: Yield tiles based on dimensionality
    if len(dim_tiles) == 1:
        # Single dimension - yield slices directly
        yield from dim_tiles[0]
    else:
        # Multi-dimensional - yield tuples of slices
        yield from itertools.product(*dim_tiles)


def _normalize_to_list(
    value: int | Tensor | list[int | Tensor],
) -> list[int | Tensor]:
    """Convert various input types to a list for uniform handling."""
    if isinstance(value, (_builtin_list, _builtin_tuple)):
        return _builtin_list(value)
    # Single value - wrap in list
    return [value]


def grid(*args: Any) -> range | Iterator[tuple[int, ...]]:  # noqa: ANN401
    # Convert all arguments to integers
    int_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            int_args.append(arg.item())  # Convert tensor to Python int
        else:
            int_args.append(_builtin_int(arg))

    if len(int_args) == 1:
        # Single dimension grid
        return _builtin_range(int_args[0])
    # Multi-dimensional grid - return cartesian product
    ranges = [_builtin_range(arg) for arg in int_args]
    return itertools.product(*ranges)


def reduce(
    combine_fn: Callable,
    input_tensor: Tensor | tuple[Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    if dim is None:
        result = combine_fn(input_tensor)
        if keep_dims:
            # Keep all dimensions as 1
            shape = [1] * len(input_tensor.shape)  # type: ignore[attr-defined]
            result = result.reshape(shape)
    else:
        result = combine_fn(input_tensor, dim=dim, keepdim=keep_dims)

    return result


def specialize(value: Any) -> Any:  # noqa: ANN401
    return value


def constexpr(value: Any) -> Any:  # noqa: ANN401
    return value


def register_block_size(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    pass


def register_reduction_dim(dim: int) -> int:
    return dim


def register_tunable(name: str, fragment: Any) -> int:  # noqa: ANN401
    return 1


def _build_indices(shape: tuple[int, ...], dim: int, idx: int) -> tuple:
    """Build indexing tuple for accessing position idx along dimension dim."""
    indices = [slice(None)] * len(shape)
    indices[dim] = idx  # type: ignore[call-overload]
    return _builtin_tuple(indices)


def _iterate_scan_dimension(
    scan_size: int, reverse: bool
) -> Iterator[tuple[int, int, bool]]:
    """
    Generate iteration indices for scan operation.

    Yields:
        Tuple of (iteration_index, actual_index, is_first_element)
    """
    for i in _builtin_range(scan_size):
        # Calculate current index based on scan direction
        idx = (scan_size - 1 - i) if reverse else i

        # Check if this is the first element in the scan
        is_first = (i == 0 and not reverse) or (i == scan_size - 1 and reverse)

        yield i, idx, is_first


def _get_prev_index(idx: int, reverse: bool) -> int:
    """Get the previous index in the scan sequence."""
    return (idx + 1) if reverse else (idx - 1)


def _scan_single_tensor(
    combine_fn: Callable, input_tensor: Tensor, dim: int, reverse: bool
) -> Tensor:
    """Helper function to perform scan on a single tensor."""
    result = torch.empty_like(input_tensor)
    scan_size = input_tensor.shape[dim]

    # Iterate through the dimension to scan
    for _i, idx, is_first in _iterate_scan_dimension(scan_size, reverse):
        # Build indexing tuple to access elements at position idx along dim
        indices = _build_indices(input_tensor.shape, dim, idx)

        if is_first:
            # First element: copy input directly
            result[indices] = input_tensor[indices]
        else:
            # Combine with previous accumulated value
            prev_idx = _get_prev_index(idx, reverse)
            prev_indices = _build_indices(input_tensor.shape, dim, prev_idx)

            # Apply the combine function
            result[indices] = combine_fn(result[prev_indices], input_tensor[indices])

    return result


def _scan_tuple_tensors(
    combine_fn: Callable, input_tuple: tuple[Tensor, ...], dim: int, reverse: bool
) -> tuple[Tensor, ...]:
    """Helper function to perform scan on a tuple of tensors."""
    tensors = _builtin_list(input_tuple)
    scan_size = tensors[0].shape[dim]

    # Initialize result tensors
    results = [torch.empty_like(t) for t in tensors]

    # Iterate through the dimension to scan
    for _i, idx, is_first in _iterate_scan_dimension(scan_size, reverse):
        # Build indexing tuple
        indices = _build_indices(tensors[0].shape, dim, idx)

        if is_first:
            # First element: copy inputs directly
            for j, tensor in enumerate(tensors):
                results[j][indices] = tensor[indices]
        else:
            # Combine with previous accumulated values
            prev_idx = _get_prev_index(idx, reverse)
            prev_indices = _build_indices(tensors[0].shape, dim, prev_idx)

            # Gather values for combination
            current_vals = _builtin_tuple(t[indices] for t in tensors)
            prev_vals = _builtin_tuple(r[prev_indices] for r in results)

            # Apply combine function with unpacked arguments
            combined = combine_fn(*prev_vals, *current_vals)

            # Store results (handle both single and tuple returns)
            if isinstance(combined, _builtin_tuple):
                for j, val in enumerate(combined):
                    results[j][indices] = val
            else:
                # Single result case
                results[0][indices] = combined

    return _builtin_tuple(results)


def associative_scan(
    combine_fn: Callable,
    input_data: Tensor | tuple[Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    if isinstance(input_data, _builtin_tuple):
        return _scan_tuple_tensors(combine_fn, input_data, dim, reverse)
    return _scan_single_tensor(combine_fn, input_data, dim, reverse)


def cumsum(
    x: Tensor, axis: int, reverse: bool = False, keep_dims: bool = False
) -> Tensor:
    import operator

    add_fn = operator.add

    result = associative_scan(add_fn, x, axis, reverse=reverse)
    assert isinstance(result, Tensor)  # cumsum always returns a single tensor
    if keep_dims:
        return result
    return result


def cumprod(
    x: Tensor, axis: int, reverse: bool = False, keep_dims: bool = False
) -> Tensor:
    import operator

    mul_fn = operator.mul

    result = associative_scan(mul_fn, x, axis, reverse=reverse)
    assert isinstance(result, Tensor)  # cumprod always returns a single tensor
    if keep_dims:
        return result
    return result


def tile_begin(tile: slice) -> int | Tensor:
    return tile.start if tile.start is not None else 0


def tile_end(tile: slice) -> int | Tensor:
    return tile.stop


def tile_block_size(tile: slice) -> int | Tensor:
    start = tile.start if tile.start is not None else 0
    return tile.stop - start


def tile_id(tile: slice) -> int:
    # Always return 0 in ref mode since we always use full dim size as block size
    return 0


def tile_index(tile: slice) -> Tensor:
    return torch.arange(tile.start, tile.stop, device="cuda")


def device_print(prefix: str, *args: Any) -> None:  # noqa: ANN401
    print(prefix, *args)


def _load_slice(tensor: Tensor, indices: slice) -> Tensor:
    begin = indices.start or 0
    end = indices.stop
    # For torch.compile compatibility, we can't call min() with tensor and int
    # Just use the slice directly and let PyTorch handle bounds checking
    return tensor[begin:end]


def _handle_single_tensor_index(
    tensor: Tensor, idx_tensor: Tensor, extra_mask: Tensor | None
) -> Tensor:
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
    tensor: Tensor, indices: tuple, extra_mask: Tensor | None
) -> Tensor:
    """Handle mixed indexing with slices and tensors."""
    expected_shape = []
    actual_indices = []
    tensor_shape = tensor.shape

    # Build expected output shape and process indices
    for i, idx in enumerate(indices):
        if isinstance(idx, slice):
            # Handle slice indices
            shape_size = idx.stop - idx.start
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
        result = tensor[_builtin_tuple(actual_indices)]

        # Handle shape mismatch when using extra_mask
        if extra_mask is not None and result.shape != _builtin_tuple(expected_shape):
            result = _pad_result_to_expected_shape(
                result, expected_shape, tensor.dtype, tensor.device
            )

        return result
    except (RuntimeError, IndexError):
        # Return zeros if indexing fails (e.g., negative indices)
        return torch.zeros(expected_shape, dtype=tensor.dtype, device=tensor.device)


def _pad_result_to_expected_shape(
    result: Tensor, expected_shape: list[int], dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Pad result tensor with zeros to match expected shape."""
    padded_result = torch.zeros(expected_shape, dtype=dtype, device=device)

    if result.numel() > 0:
        # Copy valid data to padded result
        slices = [slice(0, s) for s in result.shape]
        padded_result[_builtin_tuple(slices)] = result

    return padded_result


def load(
    tensor: Tensor,
    indices: Any,  # noqa: ANN401
    mask: Tensor | None = None,
    other: Any | None = None,  # noqa: ANN401
    extra_mask: Any | None = None,  # noqa: ANN401
) -> Tensor:
    # Combine masks first
    combined_mask = _combine_masks(mask, extra_mask)

    assert isinstance(indices, (_builtin_list, _builtin_tuple))

    # Case 1: Single tensor index (jagged indexing)
    if len(indices) == 1 and isinstance(indices[0], torch.Tensor):
        result = _handle_single_tensor_index(tensor, indices[0], extra_mask)

    # Case 2: Mixed indices containing slices (tiles)
    elif _builtin_any(isinstance(idx, slice) for idx in indices):
        result = _handle_mixed_indices(tensor, _builtin_tuple(indices), extra_mask)  # type: ignore[arg-type]
    else:
        raise exc.InvalidIndexingType(
            f"Invalid indices type: {indices}. Expected a list of slices or tensors."
        )

    # Apply mask
    return _apply_mask(result, combined_mask, other or 0)


def store(
    tensor: Tensor,
    indices: Any,  # noqa: ANN401
    value: Tensor,
    mask: Tensor | None = None,
) -> None:
    normalized_indices = _normalize_indices(indices)

    if mask is not None:
        current = tensor[normalized_indices]
        tensor[normalized_indices] = torch.where(mask, value, current)
    else:
        tensor[normalized_indices] = value


def atomic_add(tensor: Tensor, indices: Any, value: Tensor) -> None:  # noqa: ANN401
    # Special handling for scatter-add pattern (`tensor[tensor_idx, slice] += value`)
    if isinstance(indices, (_builtin_list, _builtin_tuple)) and len(indices) == 2:
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


def static_range(start: int, end: int, step: int = 1) -> range:
    return _builtin_range(start, end, step)


def subscript(tensor: Tensor, indices: Any) -> Tensor:  # noqa: ANN401
    return load(tensor, indices)


def inline_asm_elementwise(
    asm_str: str,
    constraints: str,
    args: list[Any],
    dtype: torch.dtype,
    is_pure: bool = True,
    pack: int = 1,
) -> Tensor:
    raise NotImplementedError("inline_asm_elementwise is not supported in ref mode")


def signal(handle: Any, signal_id: Any) -> None:  # noqa: ANN401
    raise NotImplementedError("hl.signal is not supported in ref mode")


def wait(
    handle: Any,  # noqa: ANN401
    signal_id: Any,  # noqa: ANN401
    signal: int = 1,
    update: Any = None,  # noqa: ANN401
    op: str = "ld",
    scope: str = "gpu",
    sem: str = "acquire",
) -> None:
    raise NotImplementedError("hl.wait is not supported in ref mode")


class HLPatchContext:
    def __init__(self) -> None:
        self._original_ops = {}

    def __enter__(self) -> HLPatchContext:  # noqa: PYI034
        # Save original operations and apply patches
        import helion.language as hl

        # Operations to skip from patching requirement
        skip_list = {
            # hl.Tile is a non-public API used internally by Helion compilation process
            "Tile"
        }

        # Get all callable operations from hl module
        hl_ops = {
            name
            for name in dir(hl)
            if not name.startswith("_") and callable(getattr(hl, name, None))
        }

        # Remove skipped operations
        hl_ops_to_patch = hl_ops - skip_list

        # Get all available patches in this module
        patched_ops = {
            name
            for name in globals()
            if not name.startswith("_") and callable(globals().get(name))
        }

        # Assert that all helion.language ops have corresponding patches
        missing_patches = hl_ops_to_patch - patched_ops
        if missing_patches:
            raise AssertionError(
                f"The following hl.* ops do not have patches in hl_patch.py: {sorted(missing_patches)}"
            )

        # Apply patches for all hl operations (excluding skip list)
        for op_name in hl_ops_to_patch:
            self._original_ops[op_name] = getattr(hl, op_name)
            # Get the patched version from this module
            setattr(hl, op_name, globals()[op_name])

        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        # Restore original operations
        import helion.language as hl

        for op_name, original_op in self._original_ops.items():
            setattr(hl, op_name, original_op)

        self._original_ops.clear()

        return False


# Global instance for helion language patching
hl_patch_context = HLPatchContext()
