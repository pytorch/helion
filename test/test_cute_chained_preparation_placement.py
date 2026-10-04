from __future__ import annotations

import random

import pytest

from helion._compiler.cute.chained_preparation_placement import _place_order
from helion._compiler.cute.chained_preparation_placement import (
    place_preparation_storage,
)
from helion._compiler.cute.warp_specialized_plan import SharedBufferRequest


@pytest.mark.parametrize("seed", range(32))
def test_placement_preserves_all_owners_leases_origins_and_declarations(seed):
    rng = random.Random(seed)
    requests = tuple(
        SharedBufferRequest(
            f"value_{index}",
            rng.randrange(1, 17) * 128,
            128,
            start := rng.randrange(16),
            start + rng.randrange(1, 17),
        )
        for index in range(24)
    )
    minimum = {requests[0].name: 2048, requests[1].name: 4096}
    declared = {requests[0].name: 8192, requests[1].name: 16384}
    result = place_preparation_storage(requests, minimum, declared)
    chronological = _place_order(
        requests,
        tuple(
            sorted(
                requests, key=lambda item: (item.live_from, -item.byte_size, item.name)
            )
        ),
        minimum,
        declared,
    )
    assert result.allocated_bytes <= chronological.allocated_bytes
    assert result.allocated_bytes % 128 == 0
    if result.allocated_bytes == chronological.allocated_bytes:
        assert result == chronological
    for request, region in zip(requests, result.regions, strict=True):
        assert (
            region.name,
            region.byte_size,
            region.alignment,
            region.live_from,
            region.live_until,
        ) == (
            request.name,
            request.byte_size,
            request.alignment,
            request.live_from,
            request.live_until,
        )
        assert region.byte_offset >= minimum.get(region.name, 0)
        assert region.byte_offset % region.alignment == 0
        assert (
            region.byte_offset + max(region.byte_size, declared.get(region.name, 0))
            <= result.allocated_bytes
        )
    assert all(
        not (left.overlaps_storage(right) and left.overlaps_lifetime(right))
        for index, left in enumerate(result.regions)
        for right in result.regions[index + 1 :]
    )
    reverse = place_preparation_storage(tuple(reversed(requests)), minimum, declared)
    assert reverse.allocated_bytes == result.allocated_bytes
    assert reverse.regions == tuple(reversed(result.regions))


def test_empty_placement_and_disjoint_half_open_lifetimes():
    assert place_preparation_storage((), {}, {}).allocated_bytes == 0
    requests = (
        SharedBufferRequest("left", 128, 128, 0, 1),
        SharedBufferRequest("right", 128, 128, 1, 2),
    )
    result = place_preparation_storage(requests, {}, {})
    assert result.allocated_bytes == 128
    assert all(region.byte_offset == 0 for region in result.regions)
