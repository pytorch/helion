from __future__ import annotations

import collections
import functools
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Sequence
from typing import TypeVar

if TYPE_CHECKING:
    from types import CellType

T = TypeVar("T")

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


def prev_power_of_2(n: int, max_value: int | None = None) -> int:
    """Floor a budget and optional maximum to a power of two (at least 1)."""
    if max_value is not None:
        n = min(n, max_value)
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def widest_halvable(sizes: Iterable[tuple[T, int, int | None]]) -> T | None:
    """The key of the widest of ``(key, size, minimum)`` whose half stays at
    or above ``minimum`` (the first of equally wide ones); ``None`` when none
    can halve.  A ``None`` minimum never halves."""
    widest: T | None = None
    widest_size = -1
    for key, size, minimum in sizes:
        if minimum is not None and size // 2 >= minimum and size > widest_size:
            widest, widest_size = key, size
    return widest


def indexing_uses_tensor_descriptor(indexing: object) -> bool:
    """Whether a scalar or per-operation indexing config selects descriptors."""
    return indexing == "tensor_descriptor" or (
        isinstance(indexing, (list, tuple)) and "tensor_descriptor" in indexing
    )


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


def is_scalar_index(idx: object) -> bool:
    """Return True if the given index behaves as a single scalar coordinate."""
    import torch

    if isinstance(idx, torch.SymInt):
        # Exclude tiled block ranges.
        from helion._compiler.compile_environment import CompileEnvironment
        from helion._compiler.compile_environment import NoCurrentEnvironment

        try:
            env = CompileEnvironment.current()
            if env.get_block_id(idx) is not None:
                return False
        except NoCurrentEnvironment:
            pass
        return True

    if isinstance(idx, int):
        return True
    if hasattr(idx, "ndim"):
        return idx.ndim == 0
    return False


def _read_cell(cell: CellType) -> tuple[bool, object]:
    """Distinguish an empty closure cell from one containing None."""
    try:
        return True, cell.cell_contents
    except ValueError:
        return False, None
