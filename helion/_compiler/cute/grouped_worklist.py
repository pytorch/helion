# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES

if TYPE_CHECKING:
    from collections.abc import Sequence


class Tcgen05GroupedWorklistValidationError(ValueError):
    """A value-level violation of the grouped-worklist contract."""


def _analyze_tcgen05_grouped_worklist_rows(
    rows: Sequence[Sequence[int]],
    *,
    group_count: int,
    packed_m: int,
    required_source_m_tile: int | None,
) -> tuple[int, ...]:
    if len(rows) != group_count:
        raise Tcgen05GroupedWorklistValidationError(
            "tcgen05 N,M worklist row count must match B_grouped"
        )
    compatible_source_m_tiles = list(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES)
    expected_store_end = 0
    seen_groups: set[int] = set()
    for row in rows:
        if len(row) != 4:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist rows must have four fields"
            )
        real_group, start, actual_m, aligned_m = (int(value) for value in row)
        if real_group < 0 or real_group >= group_count:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist real group id is outside B_grouped"
            )
        if real_group in seen_groups:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist requires unique real group ids"
            )
        seen_groups.add(real_group)
        if start < 0 or (
            required_source_m_tile is not None and start % required_source_m_tile != 0
        ):
            if required_source_m_tile is None:
                raise Tcgen05GroupedWorklistValidationError(
                    "tcgen05 N,M worklist requires nonnegative group starts"
                )
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist requires group starts aligned to "
                f"{required_source_m_tile} rows"
            )
        compatible_source_m_tiles = [
            source_m_tile
            for source_m_tile in compatible_source_m_tiles
            if start % source_m_tile == 0
        ]
        if actual_m < 0 or actual_m > aligned_m:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist requires 0 <= actual_m <= aligned_m"
            )
        if aligned_m < 0 or (
            required_source_m_tile is not None
            and aligned_m % required_source_m_tile != 0
        ):
            if required_source_m_tile is None:
                raise Tcgen05GroupedWorklistValidationError(
                    "tcgen05 N,M worklist requires nonnegative aligned_m"
                )
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist requires aligned_m to be a nonnegative "
                f"multiple of {required_source_m_tile}"
            )
        compatible_source_m_tiles = [
            source_m_tile
            for source_m_tile in compatible_source_m_tiles
            if aligned_m % source_m_tile == 0
        ]
        if (actual_m == 0) != (aligned_m == 0):
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist requires actual_m and aligned_m to be zero "
                "together"
            )
        if start + aligned_m > packed_m:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist aligned extent exceeds A extent"
            )
        if aligned_m == 0:
            continue
        if start < expected_store_end:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist has overlapping A rows"
            )
        if start > expected_store_end:
            raise Tcgen05GroupedWorklistValidationError(
                "tcgen05 N,M worklist has row holes"
            )
        expected_store_end = start + aligned_m
    if expected_store_end != packed_m:
        raise Tcgen05GroupedWorklistValidationError(
            "tcgen05 N,M worklist aligned extents must cover A rows"
        )
    if seen_groups and seen_groups != set(range(len(seen_groups))):
        raise Tcgen05GroupedWorklistValidationError(
            "tcgen05 N,M worklist requires dense real group ids"
        )
    return tuple(compatible_source_m_tiles)


def tcgen05_grouped_worklist_compatible_source_m_tiles(
    rows: Sequence[Sequence[int]],
    *,
    group_count: int,
    packed_m: int,
) -> tuple[int, ...]:
    """Return source-M tile choices compatible with a valid ``[G, 4]`` worklist.

    Invalid worklist values have no compatible tile choices.
    """
    try:
        return _analyze_tcgen05_grouped_worklist_rows(
            rows,
            group_count=group_count,
            packed_m=packed_m,
            required_source_m_tile=None,
        )
    except (TypeError, ValueError, OverflowError):
        return ()


def validate_tcgen05_grouped_worklist_rows(
    rows: Sequence[Sequence[int]],
    *,
    group_count: int,
    packed_m: int,
    source_m_tile: int,
) -> None:
    """Validate worklist values for one selected source-M tile."""
    _analyze_tcgen05_grouped_worklist_rows(
        rows,
        group_count=group_count,
        packed_m=packed_m,
        required_source_m_tile=source_m_tile,
    )
