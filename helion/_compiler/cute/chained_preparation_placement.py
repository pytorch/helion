"""Deterministic placement of already-proved preparation storage intervals.

Changing placement never changes an owner, a lifetime, or its tensor layout.
Compare a chronological and a longest-lived-first packing by their complete
declared allocation envelopes; stable ties preserve the chronological choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .warp_specialized_plan import SharedBufferRegion
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from collections.abc import Mapping


def _place_order(
    requests: tuple[SharedBufferRequest, ...],
    order: tuple[SharedBufferRequest, ...],
    minimum_offsets: Mapping[str, int],
    declared_ends: Mapping[str, int],
) -> SharedMemoryLayoutPlan:
    regions: list[SharedBufferRegion] = []
    for request in order:
        alignment = request.alignment
        minimum = minimum_offsets.get(request.name, 0)
        offset = (minimum + alignment - 1) // alignment * alignment
        while True:
            proposal = SharedBufferRegion(
                request.name,
                offset,
                request.byte_size,
                request.live_from,
                request.live_until,
                alignment,
            )
            conflicts = [
                region
                for region in regions
                if proposal.overlaps_lifetime(region)
                and proposal.overlaps_storage(region)
            ]
            if not conflicts:
                break
            end = max(region.byte_end for region in conflicts)
            offset = (end + alignment - 1) // alignment * alignment
        regions.append(proposal)
    by_name = {region.name: region for region in regions}
    extent = max(
        (
            region.byte_offset
            + max(region.byte_size, declared_ends.get(region.name, region.byte_size))
            for region in regions
        ),
        default=0,
    )
    # Preparation frame origins must preserve all original SW128 alignments.
    return SharedMemoryLayoutPlan(
        tuple(by_name[request.name] for request in requests),
        (extent + 127) // 128 * 128,
    )


def place_preparation_storage(
    requests: tuple[SharedBufferRequest, ...],
    minimum_offsets: Mapping[str, int],
    declared_ends: Mapping[str, int],
) -> SharedMemoryLayoutPlan:
    """Select only a smaller validated envelope, not a different live set.

    ``minimum_offsets`` preserves nonnegative origins of shifted full logical
    views. ``declared_ends`` includes their complete view extents, even if only a
    residual subset is accessed. Both apply to both orders before selection.
    """
    chronological = tuple(
        sorted(requests, key=lambda item: (item.live_from, -item.byte_size, item.name))
    )
    duration = tuple(
        sorted(
            requests,
            key=lambda item: (
                item.live_from - item.live_until,
                -item.byte_size,
                item.name,
            ),
        )
    )
    choices = (
        _place_order(requests, chronological, minimum_offsets, declared_ends),
        _place_order(requests, duration, minimum_offsets, declared_ends),
    )
    return min(choices, key=lambda layout: layout.allocated_bytes)
