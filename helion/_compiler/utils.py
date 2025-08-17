from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def compute_slice_size(
    slice_obj: slice, original_size: int | torch.SymInt
) -> int | torch.SymInt:
    """Compute the size of a slice given the original dimension size."""
    # Handle SliceProxy that somehow got through
    from ..language.slice_proxy import SliceProxy

    if isinstance(slice_obj, SliceProxy):
        slice_obj = slice_obj.to_slice()

    start = slice_obj.start if slice_obj.start is not None else 0
    stop = slice_obj.stop if slice_obj.stop is not None else original_size
    step = slice_obj.step if slice_obj.step is not None else 1

    # Handle negative indices
    if isinstance(start, int) and start < 0:
        start += original_size
    if isinstance(stop, int) and stop < 0:
        stop += original_size

    # Single formula for all cases
    return (stop - start + step - 1) // step
