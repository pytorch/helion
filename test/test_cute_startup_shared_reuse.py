from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest

from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chunk_prefill_bt32.config import PIPELINE_PLAN
from helion._compiler.cute.startup_shared_reuse import StartupSharedReuse
from helion.exc import BackendUnsupported


def reuse() -> StartupSharedReuse:
    return StartupSharedReuse(
        PIPELINE_PLAN.shared_buffer("factor_stages"),
        PIPELINE_PLAN.role("factor"),
        5,
        4,
        16384,
    )


def test_subdivision_preserves_every_owner_byte_and_other_lifetimes():
    original = reuse()
    revised, scratch = original.resources(PIPELINE_PLAN)
    parts = [p for p in revised.shared_buffers if p.name.startswith("factor_stages_")]
    assert len(parts) == 5
    assert parts[0].byte_offset == original.owner.byte_offset
    assert parts[-1].byte_end == original.owner.byte_end
    assert all(a.byte_end == b.byte_offset for a, b in pairwise(parts))
    assert [p.live_from for p in parts] == [0, 0, 0, 0, 1]
    assert all(p.live_until == original.owner.live_until for p in parts)
    assert scratch.byte_offset == 168960 and scratch.byte_size == 16384
    assert (scratch.live_from, scratch.live_until) == (0, 1)
    assert (original.first_warp, original.last_warp) == (28, 32)
    assert revised.roles == PIPELINE_PLAN.roles
    assert revised.barriers == PIPELINE_PLAN.barriers
    assert revised.shared_bytes == PIPELINE_PLAN.shared_bytes
    for region in PIPELINE_PLAN.shared_buffers:
        if region != original.owner:
            assert revised.shared_buffer(region.name) == region


@pytest.mark.parametrize(
    "field,value",
    (
        ("group", -1),
        ("group", 5),
        ("group", True),
        ("groups", 0),
        ("groups", 4),
        ("groups", True),
        ("byte_size", 0),
        ("byte_size", 41985),
    ),
)
def test_invalid_owner_partition_rejected(field, value):
    with pytest.raises(chain._UnsupportedChain):
        replace(reuse(), **{field: value}).resources(PIPELINE_PLAN)


def test_owner_or_role_cannot_silently_change():
    original = reuse()
    with pytest.raises(chain._UnsupportedChain):
        replace(original, owner=replace(original.owner, byte_offset=2048)).resources(
            PIPELINE_PLAN
        )
    with pytest.raises(chain._UnsupportedChain):
        replace(original, role=replace(original.role, first_warp=8)).resources(
            PIPELINE_PLAN
        )


def test_borrowed_bytes_are_still_checked_against_other_live_regions():
    original = reuse()
    _, scratch = original.resources(PIPELINE_PLAN)
    conflict = replace(scratch, name="foreign_live_buffer")
    pipeline = replace(
        PIPELINE_PLAN, shared_buffers=(*PIPELINE_PLAN.shared_buffers, conflict)
    )
    with pytest.raises(BackendUnsupported):
        original.resources(pipeline)
