from __future__ import annotations

from dataclasses import replace

import pytest

from .test_cute_chained_prepared_operands import _candidate
from helion._compiler.cute.chained_prepared_groups import _valid_frame
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands


def _reserved_image(*, first=0, overlap=False):
    plan, frame, recurrence = _candidate()
    original = plan_prepared_operands(plan, frame, recurrence)
    assert original is not None and len(original) == 1
    image = original[0]
    reserved = replace(
        image.region,
        byte_offset=0 if overlap else frame.layout.allocated_bytes,
        live_from=first,
    )
    frame = replace(
        frame,
        layout=replace(
            frame.layout,
            regions=tuple(
                reserved if item.name == reserved.name else item
                for item in frame.layout.regions
            ),
            allocated_bytes=frame.layout.allocated_bytes + reserved.byte_size,
        ),
    )
    return plan, frame, recurrence, image, reserved


def test_earlier_disjoint_reservation_does_not_change_publication_or_value():
    plan, frame, recurrence, original, reserved = _reserved_image()
    result = plan_prepared_operands(plan, frame, recurrence)
    assert result == (replace(original, region=reserved),)
    publication = next(a for a in frame.actions if reserved.name in a.writes)
    assert publication.event > reserved.live_from
    assert publication.nodes == (original.buffer.node,)
    assert reserved.live_until == frame.actions[-1].publication_event
    assert _valid_frame(frame)


def test_early_reservation_still_protects_every_overlapping_live_region():
    plan, frame, recurrence, _, _ = _reserved_image(overlap=True)
    assert plan_prepared_operands(plan, frame, recurrence) is None


@pytest.mark.parametrize("first", [-1, 100])
def test_invalid_reservation_start_is_not_publication_authority(first):
    plan, frame, recurrence, _, _ = _reserved_image(first=first)
    assert plan_prepared_operands(plan, frame, recurrence) is None


def test_early_byte_reservation_is_not_a_completed_writer():
    _, frame, _, _, reserved = _reserved_image()
    publication = next(a for a in frame.actions if reserved.name in a.writes)
    earlier = frame.actions[publication.event - 1]
    assert reserved.live_from < earlier.event < publication.event
    premature = replace(earlier, reads=(*earlier.reads, reserved.name))
    frame = replace(
        frame,
        actions=tuple(premature if a is earlier else a for a in frame.actions),
    )
    assert not _valid_frame(frame)
