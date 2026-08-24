# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from torch._guards import Source

    from ..compile_environment import CompileEnvironment


class Tcgen05GroupedWorklistValidationError(ValueError):
    """A value-level violation of the grouped-worklist contract."""


def register_tcgen05_grouped_worklist_runtime_specialization(
    env: CompileEnvironment,
    fake_tensor: torch.Tensor,
    *,
    grouped_tensor: torch.Tensor,
    packed_tensor: torch.Tensor,
) -> bool:
    """Register the runtime cache-key projection for an external worklist."""
    from ...runtime.cute.launcher import _Tcgen05GroupedWorklistCompatibilityClassifier
    from ..compile_environment import RuntimeInputSpecialization

    source = env.tensor_input_source(fake_tensor)
    if source is None:
        return False

    def dimension_source(tensor: torch.Tensor) -> Source | int | None:
        tensor_source = env.tensor_input_source(tensor)
        if tensor_source is not None:
            return TensorPropertySource(tensor_source, TensorProperty.SIZE, 0)
        if env.settings.static_shapes:
            return env.size_hint(tensor.shape[0])
        return None

    group_count = dimension_source(grouped_tensor)
    packed_m = dimension_source(packed_tensor)
    if group_count is None or packed_m is None:
        return False

    sources = [source]
    if isinstance(group_count, int):
        static_group_count = group_count
    else:
        static_group_count = None
        sources.append(group_count)
    if isinstance(packed_m, int):
        static_packed_m = packed_m
    else:
        static_packed_m = None
        sources.append(packed_m)
    classifier = _Tcgen05GroupedWorklistCompatibilityClassifier(
        static_group_count,
        static_packed_m,
    )
    env.register_runtime_input_specialization(
        f"cute_tcgen05_grouped_worklist:{source!r}",
        RuntimeInputSpecialization(
            sources=tuple(sources),
            classifier_identity=(static_group_count, static_packed_m),
            classifier=classifier,
        ),
    )
    return True


_TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_PREFERENCE = (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
    TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
)


def tcgen05_grouped_worklist_source_m_tiles_by_preference(
    source_m_tiles: Sequence[int],
) -> tuple[int, ...]:
    """Order compatible source-M tiles from established to fallback profiles."""
    return tuple(
        sorted(
            source_m_tiles,
            key=_TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_PREFERENCE.index,
        )
    )


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
