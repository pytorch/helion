"""Register-resident selection networks for CuTe kernels.

Local sorting uses Batcher's odd-even mergesort; merging retains the bitonic
network construction
("Sorting networks and their applications", AFIPS 1968).  Keeping only the
larger half before each merge implements exact top-k selection.
"""

from __future__ import annotations

import functools

import cutlass
from cutlass import Int32
import cutlass.cute as cute


@functools.cache
def _odd_even_sort_network(size: int) -> tuple[tuple[int, int], ...]:
    """Generate Batcher's data-independent network for any power-of-two size."""
    assert size > 0 and (size & (size - 1)) == 0
    comparisons: list[tuple[int, int]] = []

    def merge(begin: int, length: int, stride: int) -> None:
        doubled_stride = stride * 2
        if doubled_stride < length:
            merge(begin, length, doubled_stride)
            merge(begin + stride, length, doubled_stride)
            for left in range(begin + stride, begin + length - stride, doubled_stride):
                comparisons.append((left, left + stride))
        else:
            comparisons.append((begin, begin + stride))

    def sort(begin: int, length: int) -> None:
        if length > 1:
            half = length // 2
            sort(begin, half)
            sort(begin + half, half)
            merge(begin, length, 1)

    sort(0, size)
    return tuple(comparisons)


@cute.jit
def _sort_descending(keys: cute.Tensor) -> None:
    for left, right in _odd_even_sort_network(cute.size(keys.shape)):
        a, b = Int32(keys[left]), Int32(keys[right])
        keys[left], keys[right] = max(a, b), min(a, b)


@cute.jit
def _merge_descending(keys: cute.Tensor) -> None:
    """Sort an already bitonic register sequence in descending order."""
    size = cute.size(keys.shape)
    levels = size.bit_length() - 1
    for step in cutlass.range_constexpr(levels - 1, -1, -1):
        distance = 1 << step
        for group in cutlass.range_constexpr(size // (2 * distance)):
            base = group * 2 * distance
            for offset in cutlass.range(distance, unroll_full=True):
                left = base + offset
                a, b = Int32(keys[left]), Int32(keys[left + distance])
                keys[left], keys[left + distance] = max(a, b), min(a, b)


@cute.jit
def _merge_topk(keys: cute.Tensor, other: cute.Tensor) -> None:
    """Keep the largest half of two descending sequences of equal length."""
    size = cute.size(keys.shape)
    for index in cutlass.range(size, unroll_full=True):
        keys[index] = max(Int32(keys[index]), Int32(other[size - 1 - index]))
    _merge_descending(keys)


@cute.jit
def local_topk(
    keys: cute.Tensor,
    k: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Return the sorted largest ``k`` keys across a contiguous lane subgroup.

    Each lane supplies a one-dimensional Int32 register fragment.  Its size
    and ``k`` must be powers of two, with fragment size at least ``k``.  The
    subgroup width must be a power of two no larger than a warp.  After the
    butterfly merges, every lane holds the same descending top-k sequence.

    Callers encode values and tie-breaking indices in the keys and pad absent
    elements with Int32's minimum value.  The input fragment is not modified.
    """
    size = cute.size(keys.shape)
    assert keys.element_type == Int32
    assert cute.rank(keys.shape) == 1
    assert k > 0 and (k & (k - 1)) == 0
    assert size >= k and (size & (size - 1)) == 0
    assert 0 < lanes_per_row <= 32
    assert (lanes_per_row & (lanes_per_row - 1)) == 0

    selected = cute.make_rmem_tensor(k, Int32)
    for index in cutlass.range(k, unroll_full=True):
        selected[index] = keys[index]
    _sort_descending(selected)

    for chunk in cutlass.range(1, size // k, unroll_full=True):
        other = cute.make_rmem_tensor(k, Int32)
        for index in cutlass.range(k, unroll_full=True):
            other[index] = keys[chunk * k + index]
        _sort_descending(other)
        _merge_topk(selected, other)

    for stage in cutlass.range(lanes_per_row.bit_length() - 1, unroll_full=True):
        other = cute.make_rmem_tensor(k, Int32)
        for index in cutlass.range(k, unroll_full=True):
            other[index] = cute.arch.shuffle_sync_bfly(
                selected[index], offset=1 << stage
            )
        _merge_topk(selected, other)
    return selected
