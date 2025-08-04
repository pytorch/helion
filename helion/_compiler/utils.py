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
    # Calculate slice size based on start/stop
    start = slice_obj.start if slice_obj.start is not None else 0
    stop = slice_obj.stop if slice_obj.stop is not None else original_size
    return stop - start


def get_slice_start(slice_obj: slice) -> int:
    """Get the start index of a slice, defaulting to 0."""
    return slice_obj.start if slice_obj.start is not None else 0
