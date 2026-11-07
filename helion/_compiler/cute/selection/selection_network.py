"""Selection expressed as ordinary tensor operations for Helion tracing.

The two axes describe independent candidate groups and positions within each
group. They do not prescribe a CUDA layout: register-program lowering decides
which positions share a thread and which permutations require communication.
Python loops describe compile-time comparator schedules and are unrolled when
Helion traces these helpers, just as when ``make_fx`` traces them directly.

Every stage is functional. In particular, narrowing a selected prefix leaves
ordinary dead tensor elements that a scalarizing backend can eliminate; no
selection-specific operation is needed in the generated register program.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx

from .sorting_networks import COMPACT_SORT_LAYERS

if TYPE_CHECKING:
    from torch.fx import GraphModule


@functools.cache
def _compact_sort_layers(
    size: int, prefix: int
) -> tuple[tuple[tuple[int, int, bool, bool], ...], ...]:
    """Retain live comparator outputs from a published compact network."""
    comparisons = tuple(pair for layer in COMPACT_SORT_LAYERS[size] for pair in layer)

    live = set(range(prefix))
    needed: list[tuple[int, int, bool, bool]] = []
    for left, right in reversed(comparisons):
        high, low = left in live, right in live
        if high or low:
            needed.append((left, right, high, low))
            live.update((left, right))

    # Independent comparisons may share a tensor stage. This preserves all
    # data dependencies while avoiding one full tensor node per comparator.
    layers: list[list[tuple[int, int, bool, bool]]] = []
    previous = [-1] * size
    for left, right, high, low in reversed(needed):
        level = max(previous[left], previous[right]) + 1
        while len(layers) <= level:
            layers.append([])
        layers[level].append((left, right, high, low))
        previous[left] = previous[right] = level
    return tuple(tuple(layer) for layer in layers)


def _columns(keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.gather(keys, 1, positions.expand(keys.shape[0], -1))


def _prefix(keys: torch.Tensor, start: int, size: int) -> torch.Tensor:
    return _columns(keys, torch.arange(start, start + size, device=keys.device))


def _groups(keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.gather(keys, 0, positions.expand(-1, keys.shape[1]))


def _batcher_sort(keys: torch.Tensor) -> torch.Tensor:
    """Express odd-even mergesort layers using affine and bitwise index maps.

    A layer compares alternating adjacent groups of ``distance`` positions.
    Smaller distances omit the two endpoints of each ``2 * span`` merge, which
    is Batcher's odd-even merge rather than a complete bitonic sort. The scalar
    comparator schedule has the same dependency graph; its tensor expression
    needs only logarithmically many layers, independent of comparator count.
    """
    size = keys.shape[1]
    position = torch.arange(size, device=keys.device)
    for power in range(size.bit_length() - 1):
        span = 1 << power
        for step in range(power, -1, -1):
            distance = 1 << step
            if distance == span:
                other = _columns(keys, position ^ distance)
                keys = torch.where(
                    (position & distance) == 0,
                    torch.fmax(keys, other),
                    torch.fmin(keys, other),
                )
            else:
                within = position & (2 * span - 1)
                high = ((position & distance) != 0) & (within < 2 * span - distance)
                low = ((position & distance) == 0) & (within >= distance)
                permutation = torch.where(
                    high,
                    position + distance,
                    torch.where(low, position - distance, position),
                )
                other = _columns(keys, permutation)
                keys = torch.where(
                    high,
                    torch.fmax(keys, other),
                    torch.where(low, torch.fmin(keys, other), keys),
                )
    return keys


def _sort(keys: torch.Tensor, network: str, prefix: int) -> torch.Tensor:
    size = keys.shape[1]
    if network == "batcher" or size not in COMPACT_SORT_LAYERS:
        return _prefix(_batcher_sort(keys), 0, prefix)
    position = torch.arange(size, device=keys.device)
    for layer in _compact_sort_layers(size, prefix):
        permutation = position
        high_mask = position < 0
        low_mask = position < 0
        for left, right, keep_high, keep_low in layer:
            permutation = torch.where(
                position == left,
                right,
                torch.where(position == right, left, permutation),
            )
            if keep_high:
                high_mask = high_mask | (position == left)
            if keep_low:
                low_mask = low_mask | (position == right)
        other = _columns(keys, permutation)
        keys = torch.where(
            high_mask,
            torch.fmax(keys, other),
            torch.where(low_mask, torch.fmin(keys, other), keys),
        )
    return _prefix(keys, 0, prefix)


def _bitonic_merge(keys: torch.Tensor) -> torch.Tensor:
    size = keys.shape[1]
    position = torch.arange(size, device=keys.device)
    for step in range(size.bit_length() - 2, -1, -1):
        distance = 1 << step
        other = _columns(keys, position ^ distance)
        keys = torch.where(
            (position & distance) == 0,
            torch.fmax(keys, other),
            torch.fmin(keys, other),
        )
    return keys


def _merge_topk(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    size = left.shape[1]
    reverse = torch.arange(size - 1, -1, -1, device=left.device)
    return _bitonic_merge(torch.fmax(left, _columns(right, reverse)))


def _local_selection(
    keys: torch.Tensor, k: int, network: str, schedule: str
) -> torch.Tensor:
    size = keys.shape[1]
    if network == "compact_pruned" and size in COMPACT_SORT_LAYERS:
        return _sort(keys, network, k)
    chunks = size // k

    def select(begin: int, count: int) -> torch.Tensor:
        if count == 1:
            return _sort(_prefix(keys, begin * k, k), network, k)
        half = count // 2
        return _merge_topk(select(begin, half), select(begin + half, half))

    if schedule == "balanced" and chunks > 1:
        return select(0, chunks)
    selected = _sort(_prefix(keys, 0, k), network, k)
    for chunk in range(1, chunks):
        other = _sort(_prefix(keys, chunk * k, k), network, k)
        selected = _merge_topk(selected, other)
    return selected


def _merge_cyclic(keys: torch.Tensor, stage: int) -> torch.Tensor:
    keys = _bitonic_merge(keys)
    group = torch.arange(keys.shape[0], device=keys.device)[:, None]
    for step in range(stage - 1, -1, -1):
        distance = 1 << step
        other = _groups(keys, group ^ distance)
        keys = torch.where(
            (group & distance) == 0,
            torch.fmax(keys, other),
            torch.fmin(keys, other),
        )
    return keys


def selection_network(
    keys: torch.Tensor,
    k: int,
    sort_network: str = "batcher",
    merge_schedule: str = "sequential",
    mode: str = "distributed",
    groups_per_result: int | None = None,
) -> torch.Tensor:
    """Select descending keys from a matrix of independent candidate groups.

    The matrix shape, ``k``, and configuration strings are compile-time values.
    Both input dimensions and ``k`` must be powers of two. Values must have an
    ordinary total order, for example packed integer rank/index keys or finite
    floating point ranks with negative infinity used for padding. The explicit
    fmin/fmax comparators use this non-NaN key contract; source values are
    encoded before selection, including NaNs and infinities.

    ``distributed`` returns cyclic rank ownership: output ``[g, r]`` contains
    rank ``r * groups + g``. When ``groups > k``, each consecutive group of k
    rows contains a replica of the result. ``replicated`` instead returns all
    k keys in every group and requires at least k positions per input group.
    ``groups_per_result`` partitions the first axis into independent consecutive
    sets, each with its own selected result; it defaults to the whole axis.

    This function can be called directly inside a Helion kernel: it contains
    only traceable PyTorch operations and compile-time Python control flow.
    Reshape full-slice inputs to the constant logical group shape first, since
    Helion otherwise represents their dimensions as symbolic tile sizes.
    """
    groups, size = keys.shape
    if groups_per_result is None:
        groups_per_result = groups
    assert groups > 0 and groups & (groups - 1) == 0
    assert size > 0 and size & (size - 1) == 0
    assert k > 0 and k & (k - 1) == 0
    assert groups_per_result > 0 and groups_per_result & (groups_per_result - 1) == 0
    assert groups % groups_per_result == 0
    assert k <= groups_per_result * size
    assert sort_network in ("batcher", "compact", "compact_pruned")
    assert merge_schedule in ("sequential", "balanced")
    assert mode in ("distributed", "replicated")
    group = torch.arange(groups, device=keys.device)[:, None]

    if mode == "replicated":
        assert size >= k
        selected = _local_selection(keys, k, sort_network, merge_schedule)
        for stage in range(groups_per_result.bit_length() - 1):
            other = _groups(selected, group ^ (1 << stage))
            selected = _merge_topk(selected, other)
        return selected

    selected = _local_selection(keys, min(k, size), sort_network, merge_schedule)
    for stage in range(1, groups_per_result.bit_length()):
        half = 1 << (stage - 1)
        width = 2 * half
        previous_size = selected.shape[1]
        grow = previous_size * width <= k
        upper = (group & half) != 0
        if previous_size == 1 and width > k:
            other = _groups(selected, group ^ (width - 1))
            merged = torch.fmax(selected, other)
            if k > 1:
                reverse = _groups(merged, group ^ (k - 1))
                merged = torch.where(upper, reverse, merged)
            selected = _merge_cyclic(merged, k.bit_length() - 1)
        elif grow and previous_size == 1:
            merged = selected
            if half > 1:
                reverse = _groups(merged, group ^ (half - 1))
                merged = torch.where(upper, reverse, merged)
            selected = _merge_cyclic(merged, stage)
        else:
            position = torch.arange(previous_size // 2, device=keys.device)
            own_position = torch.where(
                upper, previous_size - 2 - 2 * position, 2 * position
            )
            peer_position = own_position + 1
            own = _columns(selected, own_position)
            other = _groups(_columns(selected, peer_position), group ^ (width - 1))
            high = torch.fmax(own, other)
            if half > 1:
                reverse = _groups(high, group ^ (half - 1))
                high = torch.where(upper, reverse, high)
            high = _merge_cyclic(high, stage)
            if grow:
                low = torch.fmin(own, other)
                if half > 1:
                    reverse = _groups(low, group ^ (half - 1))
                    low = torch.where(upper, reverse, low)
                low = _merge_cyclic(low, stage)
                selected = torch.cat((high, low), dim=1)
            else:
                selected = high
    return selected


@functools.cache
def trace_selection_network(
    groups: int,
    registers: int,
    k: int,
    dtype: torch.dtype,
    sort_network: str = "batcher",
    merge_schedule: str = "sequential",
    mode: str = "distributed",
    groups_per_result: int | None = None,
) -> GraphModule:
    """Trace the same tensor helper used by Helion, without compiling a kernel."""

    def program(keys: torch.Tensor) -> torch.Tensor:
        return selection_network(
            keys, k, sort_network, merge_schedule, mode, groups_per_result
        )

    decompositions = select_decomp_table().copy()
    decompositions.pop(torch.ops.aten.fmin.default, None)
    decompositions.pop(torch.ops.aten.fmax.default, None)
    return make_fx(program, decomposition_table=decompositions)(
        torch.empty((groups, registers), dtype=dtype, device="cpu")
    )
