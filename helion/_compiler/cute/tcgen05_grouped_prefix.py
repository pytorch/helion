"""Typed grouped metadata access and interval normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import cast

import cutlass
import cutlass.cute as cute


@dataclass(frozen=True)
class ProviderPlan:
    """Static metadata domain for the typed Int32 grouped scheduler.

    These checks cover metadata arithmetic; schedule admission is separate. Original
    offset arithmetic remains in its source integer width. Only nonnegative
    scheduler counts use the independently proved Int32 range.
    """

    groups: int
    m_size: int
    n_size: int
    k_size: int
    tile_m: int
    tile_n: int
    offset_bits: Literal[32, 64]
    kind: Literal["offsets", "split_sizes"] = "offsets"
    clipped_negative_start: bool = False

    def __post_init__(self) -> None:
        limit = (1 << 31) - 1
        assert self.groups > 0
        assert 0 < self.tile_m <= limit and 0 < self.tile_n <= limit
        assert self.offset_bits in (32, 64)
        assert self.kind in ("offsets", "split_sizes")
        assert not self.clipped_negative_start or self.kind == "offsets"
        assert 0 <= self.m_size <= limit - self.tile_m + 1
        assert 0 <= self.n_size <= limit - self.tile_n + 1
        assert 0 <= self.k_size <= limit
        assert self.groups <= limit and self.tile_upper_bound <= limit

    @property
    def tile_upper_bound(self) -> int:
        return (
            self.groups
            * ((self.m_size + self.tile_m - 1) // self.tile_m)
            * ((self.n_size + self.tile_n - 1) // self.tile_n)
        )

    @property
    def prefix_bytes(self) -> int:
        return 4 * self.groups


@cute.jit
def normalize_interval(
    raw_start: cutlass.Int32 | cutlass.Int64,
    raw_end_or_size: cutlass.Int32 | cutlass.Int64,
    integer_type: cutlass.Constexpr,
    m_size: cutlass.Constexpr[int],
    is_offsets: cutlass.Constexpr[bool],
    clipped_negative_start: cutlass.Constexpr[bool],
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Keep v30's literal typed subtraction, addition, and clipping order."""
    if cutlass.const_expr(is_offsets):
        raw_extent = max(raw_end_or_size - raw_start, integer_type(0))
        if cutlass.const_expr(clipped_negative_start):
            visible_start = min(max(raw_start, integer_type(0)), integer_type(m_size))
            visible_extent = min(
                max(
                    raw_extent + min(raw_start, integer_type(0)),
                    integer_type(0),
                ),
                integer_type(m_size) - visible_start,
            )
        else:
            source_end = raw_start + min(raw_extent, integer_type(m_size))
            visible_start = min(max(raw_start, integer_type(0)), integer_type(m_size))
            visible_end = min(max(source_end, integer_type(0)), integer_type(m_size))
            visible_extent = (
                max(visible_end - visible_start, integer_type(0))
                if raw_extent > integer_type(0)
                else integer_type(0)
            )
    else:
        source_end = raw_start + min(
            max(raw_end_or_size, integer_type(0)), integer_type(m_size)
        )
        visible_start = min(max(raw_start, integer_type(0)), integer_type(m_size))
        visible_end = min(max(source_end, integer_type(0)), integer_type(m_size))
        visible_extent = (
            max(visible_end - visible_start, integer_type(0))
            if raw_end_or_size > integer_type(0)
            else integer_type(0)
        )
    return cutlass.Int32(visible_start), cutlass.Int32(visible_extent)


@cute.jit
def load_layout_value(
    layout: cute.Tensor, group: cutlass.Int32
) -> cutlass.Int32 | cutlass.Int64:
    # The public wrapper builds an unswizzled rank-one layout for offsets.
    offset_layout = cast("cute.Layout", layout.layout)
    offset_stride = cast("tuple[cutlass.Int64, ...]", offset_layout.stride)[0]
    return (
        layout.iterator + cutlass.Int64(group) * cutlass.Int64(offset_stride)
    ).load()


@cute.jit
def reload_interval(
    layout: cute.Tensor,
    group: cutlass.Int32,
    m_size: cutlass.Constexpr[int],
    is_offsets: cutlass.Constexpr[bool],
    clipped_negative_start: cutlass.Constexpr[bool],
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Reload immutable device values; never retain a host-derived interval."""
    integer_type = layout.element_type
    if cutlass.const_expr(is_offsets):
        raw_start = integer_type(load_layout_value(layout, group))
        raw_end_or_size = integer_type(load_layout_value(layout, group + 1))
    else:
        # Split sizes do not provide a direct start. Reconstruct the exact typed
        # prefix here; checkpoints are a
        # separate optimization. Addition in Z/(2**bits) is associative.
        raw_start = integer_type(0)
        previous = cutlass.Int32(0)
        while previous < group:
            raw_start = raw_start + integer_type(load_layout_value(layout, previous))
            previous = previous + cutlass.Int32(1)
        raw_end_or_size = integer_type(load_layout_value(layout, group))
    return normalize_interval(
        raw_start,
        raw_end_or_size,
        integer_type,
        m_size,
        is_offsets,
        clipped_negative_start,
    )


@cute.jit
def build_tile_prefix(
    layout: cute.Tensor,
    prefix: cute.Tensor,
    groups: cutlass.Constexpr[int],
    m_size: cutlass.Constexpr[int],
    n_size: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    is_offsets: cutlass.Constexpr[bool],
    clipped_negative_start: cutlass.Constexpr[bool],
) -> cutlass.Int32:
    """Called by one physical thread before the original converged CTA barrier."""
    integer_type = layout.element_type
    total = cutlass.Int32(0)
    group = cutlass.Int32(0)
    running_start = integer_type(0)
    while group < groups:
        if cutlass.const_expr(is_offsets):
            raw_start = integer_type(load_layout_value(layout, group))
            raw_end_or_size = integer_type(load_layout_value(layout, group + 1))
        else:
            raw_start = running_start
            raw_end_or_size = integer_type(load_layout_value(layout, group))
            running_start = running_start + raw_end_or_size
        visible_start, visible_extent = normalize_interval(
            raw_start,
            raw_end_or_size,
            integer_type,
            m_size,
            is_offsets,
            clipped_negative_start,
        )
        row_tiles = (visible_extent + cutlass.Int32(tile_m - 1)) // cutlass.Int32(
            tile_m
        )
        column_tiles = cutlass.Int32((n_size + tile_n - 1) // tile_n)
        total = total + row_tiles * column_tiles
        prefix[group] = total
        group = group + cutlass.Int32(1)
    return total


@cute.jit
def resolve_nm_work(
    layout: cute.Tensor,
    prefix: cute.Tensor,
    ordinal: cutlass.Int64,
    groups: cutlass.Constexpr[int],
    m_size: cutlass.Constexpr[int],
    n_size: cutlass.Constexpr[int],
    k_size: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    is_offsets: cutlass.Constexpr[bool],
    clipped_negative_start: cutlass.Constexpr[bool],
) -> tuple[cutlass.Int32, ...]:
    """Return the existing nine-field N,M scheduler mailbox in source coordinates.

    The immutable cumulative prefix supports any ordinal order. Equal adjacent
    prefix entries are empty groups; upper_bound skips them without losing a
    group id. No state from a previously selected group is reused.
    """
    cta_m = cutlass.Int32(0)
    cta_n = cutlass.Int32(0)
    valid = cutlass.Int32(0)
    group = cutlass.Int32(-1)
    visible_start = cutlass.Int32(0)
    visible_extent = cutlass.Int32(0)
    if cutlass.const_expr(m_size > 0 and n_size > 0):
        total = cutlass.Int64(prefix[cutlass.Int32(groups - 1)])
        if ordinal >= cutlass.Int64(0) and ordinal < total:
            # The plan bounds total <= INT32_MAX, so this narrowing is exact.
            index = cutlass.Int32(ordinal)
            low = cutlass.Int32(0)
            high = cutlass.Int32(groups)
            while low < high:
                middle = low + (high - low) // cutlass.Int32(2)
                if prefix[middle] <= index:
                    low = middle + cutlass.Int32(1)
                else:
                    high = middle
            group = low
            previous_end = cutlass.Int32(0)
            if group > cutlass.Int32(0):
                previous_end = prefix[group - cutlass.Int32(1)]
            local = index - previous_end
            visible_start, visible_extent = reload_interval(
                layout,
                group,
                m_size,
                is_offsets,
                clipped_negative_start,
            )
            row_tiles = (visible_extent + cutlass.Int32(tile_m - 1)) // cutlass.Int32(
                tile_m
            )
            column_tiles = cutlass.Int32((n_size + tile_n - 1) // tile_n)
            # SDK AlongM for native (N,M) is source N-fast. Retain the existing
            # Helion source-M-fast permutation under its exact positive guard.
            cta_m = local // column_tiles
            cta_n = local % column_tiles
            if cutlass.Int32(0) < row_tiles <= column_tiles:
                cta_m = local % row_tiles
                cta_n = local // row_tiles
            valid = cutlass.Int32(1)
    return (
        cta_m,
        cta_n,
        valid,
        group,
        group,
        visible_extent,
        cutlass.Int32(n_size),
        cutlass.Int32(k_size),
        visible_start,
    )
