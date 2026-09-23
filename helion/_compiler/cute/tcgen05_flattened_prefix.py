"""Typed element-base proof for a flattened grouped operand/output pair.

This proposal is dormant: it does not install compiler admission. Unsupported
static affine/layout domains must retain existing codegen. The dynamic span
result distinguishes unsupported values from an empty group; it never clips a
nonempty group into a smaller interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cutlass
import cutlass.cute as cute

from .tcgen05_grouped_prefix import load_layout_value


@dataclass(frozen=True)
class FlatGroupedBlockPrefixPlan:
    """One complete warp per 32 groups, within one converged physical CTA."""

    groups: int
    warps: int

    def __post_init__(self) -> None:
        if (
            type(self.groups) is not int
            or type(self.warps) is not int
            or not 0 < self.warps <= 32
            or not 0 < self.groups <= 32 * self.warps
        ):
            raise ValueError("block prefix requires 1..32 complete covering warps")

    @property
    def summary_bytes(self) -> int:
        return self.warps * 2 * 4

    def cache_identity(self) -> tuple[object, ...]:
        return "block_checked_prefix_v1", self.groups, self.warps


@dataclass(frozen=True)
class FlatRowLayout:
    """A unit-inner-stride row recipe over one original flat Float32 buffer.

    ``elements`` counts accessible elements from the tensor's actual data_ptr,
    rather than the backing allocation before its storage offset. Positive row
    strides may include padding. Bounds include the last logical element, not
    the unused padding after it.
    """

    elements: int
    width: int
    row_stride: int
    base_alignment: int

    def __post_init__(self) -> None:
        assert 0 <= self.elements < 1 << 31
        assert 0 < self.width <= self.row_stride < 1 << 31
        assert self.row_stride % 4 == 0
        assert self.base_alignment >= 16
        assert self.base_alignment & (self.base_alignment - 1) == 0

    @property
    def max_rows(self) -> int:
        if self.elements < self.width:
            return 0
        return 1 + (self.elements - self.width) // self.row_stride

    def cache_identity(self) -> tuple[int, ...]:
        return self.elements, self.width, self.row_stride, self.base_alignment


@dataclass(frozen=True)
class FlatGroupedProviderPlan:
    """All static facts needed for the nine-Int32-field mailbox.

    The source row add, row-stride multiply, and final inner-coordinate add must
    have the same proven signed integer width. ``offset_bits`` is that actual
    source width, never the backend's eventual pointer-offset narrowing.
    """

    groups: int
    a: FlatRowLayout
    output: FlatRowLayout
    tile_m: int
    tile_n: int
    offset_bits: Literal[32, 64]

    def __post_init__(self) -> None:
        limit = (1 << 31) - 1
        assert 0 < self.groups <= limit
        assert 0 < self.tile_m <= limit and 0 < self.tile_n <= limit
        assert self.offset_bits in (32, 64)
        assert self.max_rows <= limit - self.tile_m + 1
        assert self.output.width <= limit - self.tile_n + 1
        assert self.tile_upper_bound <= limit

    @property
    def max_rows(self) -> int:
        return min(self.a.max_rows, self.output.max_rows)

    @property
    def tile_upper_bound(self) -> int:
        return (
            self.groups
            * ((self.max_rows + self.tile_m - 1) // self.tile_m)
            * ((self.output.width + self.tile_n - 1) // self.tile_n)
        )

    @property
    def prefix_bytes(self) -> int:
        return 4 * self.groups

    def cache_identity(self) -> tuple[object, ...]:
        return (
            "typed_flat_element_bases_v1",
            self.groups,
            self.a.cache_identity(),
            self.output.cache_identity(),
            self.tile_m,
            self.tile_n,
            self.offset_bits,
            (0, 1, 2, 3, "a_element_base", 5, 6, 7, "output_element_base"),
        )


@cute.jit
def project_flat_interval(
    raw_start: cutlass.Int32 | cutlass.Int64,
    raw_end: cutlass.Int32 | cutlass.Int64,
    integer_type: cutlass.Constexpr,
    a_elements: cutlass.Constexpr[int],
    a_width: cutlass.Constexpr[int],
    a_stride: cutlass.Constexpr[int],
    d_elements: cutlass.Constexpr[int],
    d_width: cutlass.Constexpr[int],
    d_stride: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Boolean]:
    """Retain source-width modular arithmetic before proving any narrowing.

    For a positive extent, the complete row spans must be in bounds. Given the
    plan's positive nonoverlapping row strides and <2**31-element buffers, these
    inequalities are also necessary for every original full-row address to be
    defined. A source-width wrap may yield a valid non-row-aligned element base.
    A wrap introduced only by final Int32 narrowing does not pass this proof.
    """
    raw_extent = integer_type(raw_end - raw_start)
    a_base = cutlass.Int32(0)
    d_base = cutlass.Int32(0)
    extent = cutlass.Int32(0)
    supported = cutlass.Boolean(True)
    if raw_extent > integer_type(0):
        # The multiply occurs in the source width, then widens without loss.
        source_a = cutlass.Int64(integer_type(raw_start * integer_type(a_stride)))
        source_d = cutlass.Int64(integer_type(raw_start * integer_type(d_stride)))
        supported = (source_a >= cutlass.Int64(0)) & (
            source_a <= cutlass.Int64(a_elements - a_width)
        )
        supported = supported & (
            (source_d >= cutlass.Int64(0))
            & (source_d <= cutlass.Int64(d_elements - d_width))
        )
        if supported:
            # Subtractions cannot overflow after the preceding base proof.
            a_rows = cutlass.Int64(1) + (
                cutlass.Int64(a_elements - a_width) - source_a
            ) // cutlass.Int64(a_stride)
            d_rows = cutlass.Int64(1) + (
                cutlass.Int64(d_elements - d_width) - source_d
            ) // cutlass.Int64(d_stride)
            supported = (cutlass.Int64(raw_extent) <= a_rows) & (
                cutlass.Int64(raw_extent) <= d_rows
            )
            if supported:
                a_base = cutlass.Int32(source_a)
                d_base = cutlass.Int32(source_d)
                extent = cutlass.Int32(raw_extent)
    return a_base, d_base, extent, supported


@cute.jit
def reload_flat_interval(
    offsets: cute.Tensor,
    group: cutlass.Int32,
    a_elements: cutlass.Constexpr[int],
    a_width: cutlass.Constexpr[int],
    a_stride: cutlass.Constexpr[int],
    d_elements: cutlass.Constexpr[int],
    d_width: cutlass.Constexpr[int],
    d_stride: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Boolean]:
    integer_type = offsets.element_type
    return project_flat_interval(
        integer_type(load_layout_value(offsets, group)),
        integer_type(load_layout_value(offsets, group + cutlass.Int32(1))),
        integer_type,
        a_elements,
        a_width,
        a_stride,
        d_elements,
        d_width,
        d_stride,
    )


@cute.jit
def build_flat_tile_prefix(
    offsets: cute.Tensor,
    prefix: cute.Tensor,
    groups: cutlass.Constexpr[int],
    a_elements: cutlass.Constexpr[int],
    a_width: cutlass.Constexpr[int],
    a_stride: cutlass.Constexpr[int],
    d_elements: cutlass.Constexpr[int],
    d_width: cutlass.Constexpr[int],
    d_stride: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    """Build an immutable 4G-byte prefix without discarding unsupported groups.

    A negative final cell is an explicit unsupported-domain status, not an empty
    workload. A caller must only use native scheduling for a nonnegative prefix.
    """
    total = cutlass.Int32(0)
    all_supported = cutlass.Boolean(True)
    group = cutlass.Int32(0)
    while group < groups:
        a_base, d_base, extent, supported = reload_flat_interval(
            offsets,
            group,
            a_elements,
            a_width,
            a_stride,
            d_elements,
            d_width,
            d_stride,
        )
        all_supported = all_supported & supported
        if all_supported:
            row_tiles = (extent + cutlass.Int32(tile_m - 1)) // cutlass.Int32(tile_m)
            total = total + row_tiles * cutlass.Int32((d_width + tile_n - 1) // tile_n)
            prefix[group] = total
        else:
            prefix[group] = cutlass.Int32(-1)
        group = group + cutlass.Int32(1)
    return all_supported


@cute.jit
def build_flat_tile_prefix_warp(
    offsets: cute.Tensor,
    prefix: cute.Tensor,
    groups: cutlass.Constexpr[int],
    a_elements: cutlass.Constexpr[int],
    a_width: cutlass.Constexpr[int],
    a_stride: cutlass.Constexpr[int],
    d_elements: cutlass.Constexpr[int],
    d_width: cutlass.Constexpr[int],
    d_stride: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    """Build the same checked prefix with one converged, complete warp.

    Each lane proves one interval before the scan. A second integer scan marks
    the first unsupported interval and every following prefix cell, retaining
    the serial builder's exact -1 sentinel semantics. The caller must provide
    all 32 active lanes and synchronize the block before consuming the prefix.
    """
    lane = cute.arch.lane_idx()
    carry = cutlass.Int32(0)
    invalid_carry = cutlass.Int32(0)
    chunk = cutlass.Int32(0)
    while chunk < (groups + 31) // 32:
        group = chunk * cutlass.Int32(32) + lane
        count = cutlass.Int32(0)
        invalid = cutlass.Int32(0)
        if group < groups:
            a_base, d_base, extent, supported = reload_flat_interval(
                offsets,
                group,
                a_elements,
                a_width,
                a_stride,
                d_elements,
                d_width,
                d_stride,
            )
            row_tiles = (extent + cutlass.Int32(tile_m - 1)) // cutlass.Int32(tile_m)
            count = row_tiles * cutlass.Int32((d_width + tile_n - 1) // tile_n)
            invalid = cutlass.Int32(not supported)
        for distance in cutlass.range_constexpr(5):
            delta = 1 << distance
            previous_count = cute.arch.shuffle_sync_up(count, delta)
            previous_invalid = cute.arch.shuffle_sync_up(invalid, delta)
            if lane >= delta:
                count = count + previous_count
                invalid = invalid + previous_invalid
        count = count + carry
        invalid = invalid + invalid_carry
        if group < groups:
            value = count
            if invalid > cutlass.Int32(0):
                value = cutlass.Int32(-1)
            prefix[group] = value
        carry = cute.arch.shuffle_sync(count, 31)
        invalid_carry = cute.arch.shuffle_sync(invalid, 31)
        chunk = chunk + cutlass.Int32(1)
    return invalid_carry == cutlass.Int32(0)


@cute.jit
def build_flat_tile_prefix_block(
    offsets: cute.Tensor,
    prefix: cute.Tensor,
    summaries: cute.Tensor,
    num_warps: cutlass.Constexpr[int],
    groups: cutlass.Constexpr[int],
    a_elements: cutlass.Constexpr[int],
    a_width: cutlass.Constexpr[int],
    a_stride: cutlass.Constexpr[int],
    d_elements: cutlass.Constexpr[int],
    d_width: cutlass.Constexpr[int],
    d_stride: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
) -> None:
    """Build the same checked prefix with every warp in a converged CTA.

    Each warp scans one chunk and publishes its count and invalid totals.
    All 32 lanes reduce preceding chunks, including lanes without a group.
    The caller publishes the final prefix before any scheduler can read it.
    Counts fit Int32 by FlatGroupedProviderPlan; invalid counts are at most 1024.
    """
    assert 0 < groups <= 32 * num_warps
    assert 0 < num_warps <= 32
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    group = warp * cutlass.Int32(32) + lane
    count = cutlass.Int32(0)
    invalid = cutlass.Int32(0)
    if group < groups:
        a_base, d_base, extent, supported = reload_flat_interval(
            offsets,
            group,
            a_elements,
            a_width,
            a_stride,
            d_elements,
            d_width,
            d_stride,
        )
        row_tiles = (extent + cutlass.Int32(tile_m - 1)) // cutlass.Int32(tile_m)
        count = row_tiles * cutlass.Int32((d_width + tile_n - 1) // tile_n)
        invalid = cutlass.Int32(not supported)
    for distance in cutlass.range_constexpr(5):
        delta = 1 << distance
        previous_count = cute.arch.shuffle_sync_up(count, delta)
        previous_invalid = cute.arch.shuffle_sync_up(invalid, delta)
        if lane >= delta:
            count = count + previous_count
            invalid = invalid + previous_invalid
    if lane == cutlass.Int32(31):
        summaries[warp * cutlass.Int32(2)] = count
        summaries[warp * cutlass.Int32(2) + cutlass.Int32(1)] = invalid
    cute.arch.sync_threads()
    prior_count = cutlass.Int32(0)
    prior_invalid = cutlass.Int32(0)
    if lane < warp:
        prior_count = summaries[lane * cutlass.Int32(2)]
        prior_invalid = summaries[lane * cutlass.Int32(2) + cutlass.Int32(1)]
    prior_count = cute.arch.warp_reduction_sum(prior_count, threads_in_group=32)
    prior_invalid = cute.arch.warp_reduction_sum(prior_invalid, threads_in_group=32)
    if group < groups:
        value = count + prior_count
        if invalid + prior_invalid > cutlass.Int32(0):
            value = cutlass.Int32(-1)
        prefix[group] = value


@cute.jit
def resolve_flat_nm_work(
    offsets: cute.Tensor,
    prefix: cute.Tensor,
    ordinal: cutlass.Int64,
    groups: cutlass.Constexpr[int],
    a_elements: cutlass.Constexpr[int],
    a_width: cutlass.Constexpr[int],
    a_stride: cutlass.Constexpr[int],
    d_elements: cutlass.Constexpr[int],
    d_width: cutlass.Constexpr[int],
    d_stride: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, ...]:
    """Preserve all mailbox consumers and width; repurpose two duplicate fields.

    Field 3 remains the group id. Field 4 carries the A element base and field 8
    the output element base. These are independently proved, not row numbers.
    Unsupported prefixes return valid=-1, distinctly from terminal valid=0.
    """
    cta_m = cutlass.Int32(0)
    cta_n = cutlass.Int32(0)
    valid = cutlass.Int32(0)
    group = cutlass.Int32(-1)
    a_base = cutlass.Int32(0)
    d_base = cutlass.Int32(0)
    extent = cutlass.Int32(0)
    total = cutlass.Int64(prefix[cutlass.Int32(groups - 1)])
    if total < cutlass.Int64(0):
        valid = cutlass.Int32(-1)
    elif ordinal >= cutlass.Int64(0) and ordinal < total:
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
        a_base, d_base, extent, supported = reload_flat_interval(
            offsets,
            group,
            a_elements,
            a_width,
            a_stride,
            d_elements,
            d_width,
            d_stride,
        )
        row_tiles = (extent + cutlass.Int32(tile_m - 1)) // cutlass.Int32(tile_m)
        column_tiles = cutlass.Int32((d_width + tile_n - 1) // tile_n)
        cta_m = local // column_tiles
        cta_n = local % column_tiles
        if cutlass.Int32(0) < row_tiles <= column_tiles:
            cta_m = local % row_tiles
            cta_n = local // row_tiles
        valid = cutlass.Int32(1)
        if not supported:
            valid = cutlass.Int32(-1)
    return (
        cta_m,
        cta_n,
        valid,
        group,
        a_base,
        extent,
        cutlass.Int32(d_width),
        cutlass.Int32(a_width),
        d_base,
    )
