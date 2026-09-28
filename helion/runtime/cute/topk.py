"""Register-resident selection networks for CuTe kernels.

Local sorting chooses Batcher's odd-even mergesort or published compact
networks, optionally pruned to their selected prefix. Merging retains the bitonic
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

from .ordered_key import encode_ordered_key_16 as encode_ordered_topk_key
from .ordered_key import encode_ordinal_key_pair_16 as encode_ordinal_topk_pair
from .register_layout import cyclic_to_vector_narrow as transpose_topk_output
from .register_layout import cyclic_to_vector_wide as transpose_topk_output_wide
from .sorting_networks import COMPACT_SORT_LAYERS

__all__ = [
    "distributed_topk",
    "encode_ordered_topk_key",
    "encode_ordinal_topk_pair",
    "local_topk",
    "transpose_topk_output",
    "transpose_topk_output_wide",
]


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


@functools.cache
def _sort_network(size: int, network: str) -> tuple[tuple[int, int], ...]:
    """Use published compact networks when available, Batcher otherwise."""
    assert network in ("batcher", "compact", "compact_pruned")
    if network != "batcher" and size in COMPACT_SORT_LAYERS:
        return tuple(pair for layer in COMPACT_SORT_LAYERS[size] for pair in layer)
    return _odd_even_sort_network(size)


@functools.cache
def _pruned_sort_network(size: int, k: int) -> tuple[tuple[int, int, bool, bool], ...]:
    """Keep only comparator outputs contributing to the sorted first k wires.

    Backward liveness propagates through both inputs of every needed min/max.
    The remaining program is therefore identical on its first k outputs.
    """
    assert 0 < k <= size and (k & (k - 1)) == 0
    live = set(range(k))
    comparisons: list[tuple[int, int, bool, bool]] = []
    for left, right in reversed(_sort_network(size, "compact")):
        keep_left, keep_right = left in live, right in live
        if keep_left or keep_right:
            comparisons.append((left, right, keep_left, keep_right))
            live.update((left, right))
    return tuple(reversed(comparisons))


def _use_pruned_sort_network(size: int, network: str) -> bool:
    # Bound register/code growth to the published networks. Larger fragments
    # continue using chunked selection and its general Batcher fallback.
    return network == "compact_pruned" and size in COMPACT_SORT_LAYERS


@cute.jit
def _sort_descending(
    keys: cute.Tensor, network: cutlass.Constexpr[str] = "batcher"
) -> None:
    max_fn = cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
    min_fn = cute.arch.fmin if cutlass.const_expr(keys.element_type == Float32) else min
    for left, right in _sort_network(cute.size(keys.shape), network):
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


@functools.cache
def _balanced_chunk_program(chunks: int) -> tuple[tuple[int, int], ...]:
    """Sort leaves and merge equal adjacent groups in depth-first order.

    Equal indices mark a leaf sort. A merge writes the left group's slot and
    releases the right one. Depth-first order avoids keeping every sorted
    chunk live simultaneously while preserving the balanced dependency graph.
    """
    assert chunks > 0 and (chunks & (chunks - 1)) == 0
    program: list[tuple[int, int]] = []

    def visit(begin: int, count: int) -> None:
        if count == 1:
            program.append((begin, begin))
        else:
            half = count // 2
            visit(begin, half)
            visit(begin + half, half)
            program.append((begin, begin + half))

    visit(0, chunks)
    return tuple(program)


@cute.jit
def local_topk(
    keys: cute.Tensor,
    k: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
    sort_network: cutlass.Constexpr[str] = "batcher",
    merge_schedule: cutlass.Constexpr[str] = "sequential",
) -> cute.Tensor:
    """Return the sorted largest ``k`` keys across a contiguous lane subgroup.

    Each lane supplies a one-dimensional Int32 or Float32 register fragment. Its size
    and ``k`` must be powers of two, with fragment size at least ``k``.  The
    subgroup width must be a power of two no larger than a warp.  After the
    butterfly merges, every lane holds the same descending top-k sequence.

    Callers encode values and tie-breaking indices in the keys and pad absent
    elements with a key below every real key. Float32 keys must be finite.
    Negative infinity is also permitted as a padding key.
    The input fragment is not modified.
    """
    size = cute.size(keys.shape)
    assert keys.element_type in (Int32, Float32)
    assert cute.rank(keys.shape) == 1
    assert k > 0 and (k & (k - 1)) == 0
    assert size >= k and (size & (size - 1)) == 0
    assert 0 < lanes_per_row <= 32
    assert (lanes_per_row & (lanes_per_row - 1)) == 0

    assert merge_schedule in ("sequential", "balanced")
    selected = cute.make_rmem_tensor(k, keys.element_type)
    if cutlass.const_expr(_use_pruned_sort_network(size, sort_network)):
        work = cute.make_rmem_tensor(size, keys.element_type)
        for index in cutlass.range_constexpr(size):
            work[index] = keys[index]
        max_fn = (
            cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
        )
        min_fn = (
            cute.arch.fmin if cutlass.const_expr(keys.element_type == Float32) else min
        )
        for left, right, keep_left, keep_right in _pruned_sort_network(size, k):
            a, b = keys.element_type(work[left]), keys.element_type(work[right])
            if cutlass.const_expr(keep_left):
                work[left] = max_fn(a, b)
            if cutlass.const_expr(keep_right):
                work[right] = min_fn(a, b)
        for index in cutlass.range_constexpr(k):
            selected[index] = work[index]
    elif cutlass.const_expr(merge_schedule == "balanced" and size > k):
        chunks = size // k
        partials = cute.make_rmem_tensor((k, chunks), keys.element_type)
        for left, right in _balanced_chunk_program(chunks):
            if cutlass.const_expr(left == right):
                for index in cutlass.range_constexpr(k):
                    partials[index, left] = keys[left * k + index]
                _sort_descending(partials[None, left], sort_network)
            else:
                _merge_topk(partials[None, left], partials[None, right])
        for index in cutlass.range_constexpr(k):
            selected[index] = partials[index, 0]
    else:
        for index in cutlass.range(k, unroll_full=True):
            selected[index] = keys[index]
        _sort_descending(selected, sort_network)

        for chunk in cutlass.range(1, size // k, unroll_full=True):
            other = cute.make_rmem_tensor(k, keys.element_type)
            for index in cutlass.range(k, unroll_full=True):
                other[index] = keys[chunk * k + index]
            _sort_descending(other, sort_network)
            _merge_topk(selected, other)

    for stage in cutlass.range(lanes_per_row.bit_length() - 1, unroll_full=True):
        other = cute.make_rmem_tensor(k, keys.element_type)
        for index in cutlass.range(k, unroll_full=True):
            other[index] = cute.arch.shuffle_sync_bfly(
                selected[index], offset=1 << stage
            )
        _merge_topk(selected, other)
    return selected


@cute.jit
def _merge_cyclic_fragment(keys: cute.Tensor, stage: cutlass.Constexpr[int]) -> None:
    """Merge a bitonic sequence distributed cyclically over 2**stage lanes."""
    _merge_descending(keys)
    max_fn = cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
    min_fn = cute.arch.fmin if cutlass.const_expr(keys.element_type == Float32) else min
    lane = Int32(cute.arch.thread_idx()[0])
    for step in cutlass.range_constexpr(stage - 1, -1, -1):
        distance = 1 << step
        keep_high = (lane & Int32(distance)) == 0
        for index in cutlass.range_constexpr(cute.size(keys.shape)):
            own = keys.element_type(keys[index])
            peer = cute.arch.shuffle_sync_bfly(own, offset=distance)
            value = min_fn(own, peer)
            if keep_high:
                value = max_fn(own, peer)
            keys[index] = value


@cute.jit
def distributed_topk(
    keys: cute.Tensor,
    k: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
    sort_network: cutlass.Constexpr[str] = "batcher",
    merge_schedule: cutlass.Constexpr[str] = "sequential",
) -> cute.Tensor:
    """Return top-k in cyclic rank order, with at least one key per lane.

    Each merge doubles the lane group. Groups with at most k input keys keep
    both halves of the merge; larger groups retain only their largest half.
    This avoids padding every lane to k when its input fragment is smaller.
    Local register merges followed by subgroup butterflies preserve cyclic
    rank ownership. If there are more lanes than k, each k-lane subgroup
    holds an identical result and only the first k lanes should store it.
    """
    assert k > 0 and (k & (k - 1)) == 0
    assert 0 < lanes_per_row <= 32
    assert (lanes_per_row & (lanes_per_row - 1)) == 0
    selected = local_topk(
        keys, min(k, cute.size(keys.shape)), 1, sort_network, merge_schedule
    )
    lane = Int32(cute.arch.thread_idx()[0]) % Int32(lanes_per_row)
    max_fn = cute.arch.fmax if cutlass.const_expr(keys.element_type == Float32) else max
    min_fn = cute.arch.fmin if cutlass.const_expr(keys.element_type == Float32) else min
    for stage in cutlass.range_constexpr(1, lanes_per_row.bit_length()):
        half = 1 << (stage - 1)
        group = 2 * half
        previous_size = cute.size(selected.shape)
        grow = previous_size * group <= k
        upper_half = (lane & Int32(half)) != 0
        if cutlass.const_expr(previous_size == 1 and group > k):
            # Both halves already contain replicated k-lane top-k groups.
            # Opposite ranks form the larger bitonic half; orient every
            # replica identically and merge within k lanes, not group lanes.
            own = keys.element_type(selected[0])
            peer = cute.arch.shuffle_sync_bfly(own, offset=group - 1)
            value = max_fn(own, peer)
            if cutlass.const_expr(k > 1):
                reordered = cute.arch.shuffle_sync_bfly(value, offset=k - 1)
                if upper_half:
                    value = reordered
            merged = cute.make_rmem_tensor(1, keys.element_type)
            merged[0] = value
            _merge_cyclic_fragment(merged, k.bit_length() - 1)
            selected = merged
        elif cutlass.const_expr(grow and previous_size == 1):
            # Concatenated single-register groups become bitonic by reversing
            # the upper group's lanes, then merge entirely through shuffles.
            value = keys.element_type(selected[0])
            if cutlass.const_expr(half > 1):
                reordered = cute.arch.shuffle_sync_bfly(value, offset=half - 1)
                if upper_half:
                    value = reordered
            merged = cute.make_rmem_tensor(1, keys.element_type)
            merged[0] = value
            _merge_cyclic_fragment(merged, stage)
            selected = merged
        else:
            halves = cute.make_rmem_tensor(
                (previous_size // 2, 2 if grow else 1), keys.element_type
            )
            for index in cutlass.range_constexpr(previous_size // 2):
                own = keys.element_type(selected[2 * index])
                peer_source = keys.element_type(selected[2 * index + 1])
                if upper_half:
                    own = keys.element_type(selected[previous_size - 2 - 2 * index])
                    peer_source = keys.element_type(
                        selected[previous_size - 1 - 2 * index]
                    )
                peer = cute.arch.shuffle_sync_bfly(peer_source, offset=group - 1)
                value = max_fn(own, peer)
                if cutlass.const_expr(half > 1):
                    # Correct the reversed upper-half lane positions for
                    # both bitonic halves before sorting them independently.
                    reordered = cute.arch.shuffle_sync_bfly(value, offset=half - 1)
                    if upper_half:
                        value = reordered
                halves[index, 0] = value
                if cutlass.const_expr(grow):
                    low_value = min_fn(own, peer)
                    if cutlass.const_expr(half > 1):
                        low_reordered = cute.arch.shuffle_sync_bfly(
                            low_value, offset=half - 1
                        )
                        if upper_half:
                            low_value = low_reordered
                    halves[index, 1] = low_value
            _merge_cyclic_fragment(halves[None, 0], stage)
            if cutlass.const_expr(grow):
                _merge_cyclic_fragment(halves[None, 1], stage)
                # Every key in the larger half precedes every smaller key.
                # Concatenating per-lane fragments preserves cyclic rank order.
                selected = cute.make_rmem_tensor(previous_size, keys.element_type)
                for index in cutlass.range_constexpr(previous_size // 2):
                    selected[index] = halves[index, 0]
                    selected[index + previous_size // 2] = halves[index, 1]
            else:
                selected = halves[None, 0]
    return selected
