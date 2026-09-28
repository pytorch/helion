"""Register-resident selection networks for CuTe kernels.

Local sorting uses Batcher's odd-even mergesort; merging retains the bitonic
network construction
("Sorting networks and their applications", AFIPS 1968).  Keeping only the
larger half before each merge implements exact top-k selection.
"""

from __future__ import annotations

import functools

import cutlass
from cutlass import Float32
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
    max_fn = cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
    min_fn = cute.arch.fmin if cutlass.const_expr(keys.element_type == Float32) else min
    for left, right in _odd_even_sort_network(cute.size(keys.shape)):
        a, b = keys.element_type(keys[left]), keys.element_type(keys[right])
        keys[left], keys[right] = max_fn(a, b), min_fn(a, b)


@cute.jit
def _merge_descending(keys: cute.Tensor) -> None:
    """Sort an already bitonic register sequence in descending order."""
    size = cute.size(keys.shape)
    levels = size.bit_length() - 1
    max_fn = cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
    min_fn = cute.arch.fmin if cutlass.const_expr(keys.element_type == Float32) else min
    for step in cutlass.range_constexpr(levels - 1, -1, -1):
        distance = 1 << step
        for group in cutlass.range_constexpr(size // (2 * distance)):
            base = group * 2 * distance
            for offset in cutlass.range(distance, unroll_full=True):
                left = base + offset
                a, b = (
                    keys.element_type(keys[left]),
                    keys.element_type(keys[left + distance]),
                )
                keys[left], keys[left + distance] = max_fn(a, b), min_fn(a, b)


@cute.jit
def _merge_topk(keys: cute.Tensor, other: cute.Tensor) -> None:
    """Keep the largest half of two descending sequences of equal length."""
    size = cute.size(keys.shape)
    max_fn = cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
    for index in cutlass.range(size, unroll_full=True):
        keys[index] = max_fn(
            keys.element_type(keys[index]), keys.element_type(other[size - 1 - index])
        )
    _merge_descending(keys)


@cute.jit
def local_topk(
    keys: cute.Tensor,
    k: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Return the sorted largest ``k`` keys across a contiguous lane subgroup.

    Each lane supplies a one-dimensional Int32 or Float32 register fragment. Its size
    and ``k`` must be powers of two, with fragment size at least ``k``.  The
    subgroup width must be a power of two no larger than a warp.  After the
    butterfly merges, every lane holds the same descending top-k sequence.

    Callers encode values and tie-breaking indices in the keys and pad absent
    elements with a key below every real key. Float32 keys must be finite.
    The input fragment is not modified.
    """
    size = cute.size(keys.shape)
    assert keys.element_type in (Int32, Float32)
    assert cute.rank(keys.shape) == 1
    assert k > 0 and (k & (k - 1)) == 0
    assert size >= k and (size & (size - 1)) == 0
    assert 0 < lanes_per_row <= 32
    assert (lanes_per_row & (lanes_per_row - 1)) == 0

    selected = cute.make_rmem_tensor(k, keys.element_type)
    for index in cutlass.range(k, unroll_full=True):
        selected[index] = keys[index]
    _sort_descending(selected)

    for chunk in cutlass.range(1, size // k, unroll_full=True):
        other = cute.make_rmem_tensor(k, keys.element_type)
        for index in cutlass.range(k, unroll_full=True):
            other[index] = keys[chunk * k + index]
        _sort_descending(other)
        _merge_topk(selected, other)

    for stage in cutlass.range(lanes_per_row.bit_length() - 1, unroll_full=True):
        other = cute.make_rmem_tensor(k, keys.element_type)
        for index in cutlass.range(k, unroll_full=True):
            other[index] = cute.arch.shuffle_sync_bfly(
                selected[index], offset=1 << stage
            )
        _merge_topk(selected, other)
    return selected
