from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import cast

import torch

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Sequence


def is_full_slice(value: object) -> bool:
    """Whether a subscript is the bare ``:`` without bounds or a step."""
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


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


def indexed_dims(rank: int, index: Sequence[object]) -> list[int | None]:
    """For each element of ``index`` into a tensor of ``rank`` dims, the dim it
    indexes: ``None`` for ``None`` (a new dim) and for ``Ellipsis``, which
    spans the dims the other elements leave."""
    consumed = sum(1 for idx in index if idx is not None and idx is not Ellipsis)
    dims: list[int | None] = []
    dim = 0
    for idx in index:
        if idx is Ellipsis:
            dims.append(None)
            dim += rank - consumed
        elif idx is None:
            dims.append(None)
        else:
            dims.append(dim)
            dim += 1
    return dims


def normalize_index_slices(
    shape: Sequence[int | torch.SymInt], index: Sequence[object]
) -> list[object]:
    """``index`` into a tensor of ``shape`` with each slice normalized
    (``normalize_slice``) against the dim it indexes."""
    return [
        normalize_slice(idx, shape[dim])
        if isinstance(idx, slice) and dim is not None and dim < len(shape)
        else idx
        for idx, dim in zip(index, indexed_dims(len(shape), index), strict=True)
    ]


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


def is_basic_host_subscript(index: object) -> bool:
    """Whether host geometry can replay this index without Python conversions."""
    indices = index if isinstance(index, tuple) else (index,)
    if not all(
        item is None
        or item is Ellipsis
        or type(item) is int
        or isinstance(item, (slice, torch.SymInt))
        for item in indices
    ):
        return False
    saw_ellipsis = False
    for item in indices:
        if item is Ellipsis:
            # Repeated ellipses do not describe one ordinary expansion.
            if saw_ellipsis:
                return False
            saw_ellipsis = True
        if isinstance(item, slice) and any(
            value is not None
            and (
                not isinstance(value, (int, torch.SymInt))
                or isinstance(value, int)
                and not -sys.maxsize <= value <= sys.maxsize
            )
            for value in (item.start, item.stop, item.step)
        ):
            # PySlice_Unpack converts custom __index__ objects and saturates
            # out-of-range Python integers. Keep its existing result there.
            return False
    return True


def host_subscript_geometry(
    tensor: torch.Tensor, index: object
) -> (
    tuple[list[int | torch.SymInt], list[int | torch.SymInt], int | torch.SymInt] | None
):
    """Basic-index view metadata without choosing slice bounds from hints.

    FakeTensor's slice implementation can select a clamp branch with a shape
    guard. Host inference suppresses those guards, so reconstruct the view's
    metadata with symbolic clamps, including offsets and inserted-axis strides.
    Call only after indexing has validated the arguments. Advanced indexing
    keeps FakeTensor's geometry.
    """
    if not is_basic_host_subscript(index):
        return None
    indices = index if isinstance(index, tuple) else (index,)
    shape, input_strides = tensor.size(), tensor.stride()
    remaining = len(shape) - sum(
        item is not None and item is not Ellipsis for item in indices
    )
    sizes: list[int | torch.SymInt] = []
    strides: list[int | torch.SymInt] = []
    offset = tensor.storage_offset()
    dim = 0
    for item in indices:
        if item is None:
            sizes.append(1)
            strides.append(input_strides[dim] * shape[dim] if dim < len(shape) else 1)
        elif item is Ellipsis:
            sizes.extend(shape[dim : dim + remaining])
            strides.extend(input_strides[dim : dim + remaining])
            dim += remaining
        else:
            if isinstance(item, slice):
                size = shape[dim]
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                elif isinstance(start, torch.SymInt):
                    start = _symbolic_slice_bound(start, size)
                else:
                    start = _literal_slice_bound(start, size)
                if stop is None:
                    stop = size
                elif isinstance(stop, torch.SymInt):
                    stop = _symbolic_slice_bound(stop, size)
                else:
                    stop = _literal_slice_bound(stop, size)
                step = 1 if item.step is None else item.step
                sizes.append((_max(0, stop - start) + step - 1) // step)
                strides.append(input_strides[dim] * step)
                offset = offset + start * input_strides[dim]
            else:
                item = cast("int | torch.SymInt", item)
                position = (
                    torch.sym_ite(item < 0, item + shape[dim], item)
                    if isinstance(item, torch.SymInt)
                    else item + shape[dim]
                    if item < 0
                    else item
                )
                offset = offset + position * input_strides[dim]
            dim += 1
    sizes.extend(shape[dim:])
    strides.extend(input_strides[dim:])
    return sizes, strides, offset
