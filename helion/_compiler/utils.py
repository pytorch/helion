from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Sequence


def _min(a: int | torch.SymInt, b: int | torch.SymInt) -> int | torch.SymInt:
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return cast("int | torch.SymInt", torch.sym_min(a, b))


def _max(a: int | torch.SymInt, b: int | torch.SymInt) -> int | torch.SymInt:
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return cast("int | torch.SymInt", torch.sym_max(a, b))


def _literal_slice_bound(bound: int, size: int | torch.SymInt) -> int | torch.SymInt:
    """A literal ``bound`` clamped to ``[0, size]`` as PyTorch indexing does,
    a negative one counted from the end."""
    if bound < 0:
        return _max(0, bound + size)
    return _min(bound, size)


def _symbolic_slice_bound(
    bound: torch.SymInt, size: int | torch.SymInt
) -> int | torch.SymInt:
    """A symbolic ``bound`` clamped to ``[0, size]`` as PyTorch indexing does,
    counted from the end when negative (selected at runtime: its sign is
    unknown at compile time)."""
    wrapped = cast("torch.SymInt", torch.sym_ite(bound < 0, bound + size, bound))
    return _max(0, _min(wrapped, size))


def normalize_slice(slice_obj: slice, size: int | torch.SymInt) -> slice:
    """``slice_obj`` with the bounds PyTorch indexing uses on a dim of
    ``size``: a negative ``start`` and any ``stop`` are clamped to the dim
    (``_literal_slice_bound``, ``_symbolic_slice_bound``, with ``sym_min`` /
    ``sym_max`` on a dynamic dim, which does not specialize it), and with two
    bounds ``stop`` is raised to at least ``start`` (an empty slice).  A
    non-negative literal ``start`` is kept: past the end it only empties the
    slice (see ``compute_slice_size``), and it keeps the offset static on a
    dynamic dim.  An omitted bound stays ``None``, so an in-range slice is
    unchanged.  Two symbolic bounds (``tile.begin * 2 : tile.begin * 2 + 8``,
    a loop index ``i : i + 1``) are kept as written: clamping would make an
    extent like ``(i + 1) - i`` symbolic, and these index within the tensor
    by construction.  A step must be positive (``InvalidSliceStep``), as in
    PyTorch indexing."""
    if isinstance(slice_obj.step, int) and slice_obj.step <= 0:
        raise exc.InvalidSliceStep(slice_obj)
    start, stop = slice_obj.start, slice_obj.stop
    if isinstance(start, torch.SymInt) and isinstance(stop, torch.SymInt):
        return slice_obj
    both_bounds = start is not None and stop is not None
    if isinstance(start, torch.SymInt):
        start = _symbolic_slice_bound(start, size)
    elif isinstance(start, int) and start < 0:
        start = _literal_slice_bound(start, size)
    if isinstance(stop, torch.SymInt):
        stop = _symbolic_slice_bound(stop, size)
    elif isinstance(stop, int):
        stop = _literal_slice_bound(stop, size)
    if both_bounds and isinstance(start, (int, torch.SymInt)):
        assert isinstance(stop, (int, torch.SymInt))
        stop = _max(start, stop)
    return slice(start, stop, slice_obj.step)


def normalize_index_slices(
    shape: Sequence[int | torch.SymInt], index: Sequence[object]
) -> list[object]:
    """``index`` into a tensor of ``shape`` with each slice normalized
    (``normalize_slice``) against the dim it indexes."""
    consumed = sum(1 for idx in index if idx is not None and idx is not Ellipsis)
    result: list[object] = []
    dim = 0
    for idx in index:
        if idx is Ellipsis:
            dim += len(shape) - consumed
        elif isinstance(idx, slice) and dim < len(shape):
            idx = normalize_slice(idx, shape[dim])
        if idx is not None and idx is not Ellipsis:
            dim += 1
        result.append(idx)
    return result


def compute_slice_size(
    slice_obj: slice, original_size: int | torch.SymInt
) -> int | torch.SymInt:
    """
    Compute the size of a slice operation.

    Args:
        slice_obj: The slice object with start, stop, and step attributes
        original_size: The size of the dimension being sliced

    Returns:
        The size of the resulting sliced dimension, with the bounds clamped to
        the dimension as PyTorch indexing does (``normalize_slice``)
    """
    original_start = slice_obj.start
    slice_obj = normalize_slice(slice_obj, original_size)
    start = slice_obj.start if slice_obj.start is not None else 0
    if slice_obj.stop is not None:
        stop = slice_obj.stop
    elif isinstance(original_start, int):
        # A literal start past the end (kept by normalize_slice) empties it.
        stop = _max(start, original_size)
    else:
        stop = original_size
    if slice_obj.step is not None and slice_obj.step != 1:
        step = slice_obj.step
        return (stop - start + step - 1) // step
    return stop - start
