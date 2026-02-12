from __future__ import annotations

import collections
import functools
from typing import Sequence

counters: collections.defaultdict[str, collections.Counter[str]] = (
    collections.defaultdict(collections.Counter)
)


def cdiv(a: int, b: int) -> int:
    """Ceiling division: returns ceil(a / b)."""
    return (a + b - 1) // b


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@functools.cache
def triton_is_available() -> bool:
    """Return True if triton is installed and importable."""
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def create_shape_matching_slices(
    shape1: Sequence[int], shape2: Sequence[int]
) -> tuple[slice, ...]:
    """Create slices to match the smaller of two shapes.

    This is used for masking tensors to compatible shapes by taking the
    minimum size in each dimension.

    Args:
        shape1: First shape (can be torch.Size or any sequence of ints)
        shape2: Second shape (can be torch.Size or any sequence of ints)

    Returns:
        Tuple of slices that can be used to index a tensor
    """
    return tuple(slice(0, min(d1, d2)) for d1, d2 in zip(shape1, shape2, strict=False))


def convert_size_arg(size: object) -> object:
    """Convert a size argument that may contain RefTile objects.

    Handles:
    - Single RefTile -> int (block_size)
    - List/tuple containing RefTiles -> list with converted sizes
    - Other values -> unchanged
    """
    # Import here to avoid circular dependency
    from .language.ref_tile import RefTile

    if isinstance(size, (list, tuple)):
        return [convert_size_arg(item) for item in size]
    if isinstance(size, RefTile):
        return size._block_size
    return size


def convert_tile_indices_to_slices(index: object) -> object:
    """Convert RefTile objects in index to their corresponding slice objects.

    Args:
        index: Index that may contain RefTile objects or tuples of indices

    Returns:
        Index with RefTile objects replaced by their slice objects
    """
    # Import here to avoid circular dependency
    from .language.ref_tile import RefTile

    def _extract_slice(obj: object) -> object:
        return obj._slice if isinstance(obj, RefTile) else obj

    if isinstance(index, tuple):
        return tuple(_extract_slice(idx) for idx in index)
    return _extract_slice(index)
