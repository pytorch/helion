"""Type utilities for handling slices and other common type operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..language.slice_proxy import SliceProxy

if TYPE_CHECKING:
    from collections.abc import Sequence


def is_regular_tensor(obj: object) -> bool:
    """Check if an object is a regular tensor (not a tensor subclass like SliceProxy or Tile)."""
    if not isinstance(obj, torch.Tensor):
        return False
    from ..language.tile_proxy import Tile
    return not isinstance(obj, (SliceProxy, Tile))


def compute_slice_size(
    slice_obj: slice | SliceProxy, original_size: int | torch.SymInt
) -> int | torch.SymInt:
    """Compute the size of a slice given the original dimension size."""
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


def normalize_slice_bounds(
    slice_obj: slice | SliceProxy, size: int | torch.SymInt | None = None
) -> tuple[object, object, object]:
    """Normalize slice bounds to handle None values.
    
    Returns (start, stop, step) with None values replaced by defaults.
    """
    if isinstance(slice_obj, SliceProxy):
        slice_obj = slice_obj.to_slice()
    
    start = slice_obj.start if slice_obj.start is not None else 0
    stop = slice_obj.stop if slice_obj.stop is not None else size
    step = slice_obj.step if slice_obj.step is not None else 1
    
    return start, stop, step


def is_full_slice(slice_obj: slice | SliceProxy) -> bool:
    """Check if a slice represents a full slice [:] with no bounds."""
    if isinstance(slice_obj, SliceProxy):
        slice_obj = slice_obj.to_slice()
    
    return (
        slice_obj.start is None
        and slice_obj.stop is None
        and slice_obj.step is None
    )