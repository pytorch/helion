"""Host-side plans for a flattened grouped operand/output pair.

These records carry no cutlass dependency: schedule admission and the
autotuner heuristics import them on hosts without the CuTe DSL installed.
The device prefix helpers that consume them live in
``tcgen05_flattened_prefix``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
