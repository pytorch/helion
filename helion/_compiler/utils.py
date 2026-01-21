from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def compute_slice_size(
    slice_obj: slice, original_size: int | torch.SymInt
) -> int | torch.SymInt:
    """
    Compute the size of a slice operation.

    Args:
        slice_obj: The slice object with start, stop, and step attributes
        original_size: The size of the dimension being sliced

    Returns:
        The size of the resulting sliced dimension
    """
    if slice_obj.step is not None and slice_obj.step != 1:
        # Calculate size based on step
        start = slice_obj.start if slice_obj.start is not None else 0
        stop = slice_obj.stop if slice_obj.stop is not None else original_size
        step = slice_obj.step
        return (stop - start + step - 1) // step
    # Full slice or slice without step
    start = slice_obj.start if slice_obj.start is not None else 0
    stop = slice_obj.stop if slice_obj.stop is not None else original_size
    return stop - start


def get_broadcast_slice(dim_index: int, total_dims: int) -> str:
    """Generate broadcast indexing syntax for N-dimensional tensors.

    Args:
        dim_index: Which dimension to keep (0-indexed)
        total_dims: Total number of dimensions in output

    Returns:
        String like "[:, None]", "[None, :]", "[:, None, None]", etc.

    Examples:
        get_broadcast_slice(0, 1) -> "[:]"
        get_broadcast_slice(0, 2) -> "[:, None]"
        get_broadcast_slice(1, 2) -> "[None, :]"
        get_broadcast_slice(0, 3) -> "[:, None, None]"
        get_broadcast_slice(1, 3) -> "[None, :, None]"
        get_broadcast_slice(2, 3) -> "[None, None, :]"
    """
    broadcast_keys = ["None"] * total_dims
    broadcast_keys[dim_index] = ":"
    return f"[{', '.join(broadcast_keys)}]"
