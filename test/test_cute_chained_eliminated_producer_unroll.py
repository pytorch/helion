from __future__ import annotations

import pytest

from helion import exc
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
@pytest.mark.parametrize("trips", [0, 1, 2, 3, 9])
def test_removed_producer_is_distinct_from_actual_unroll(factor, trips):
    tracker = BoundedProducerUnroll(factor)
    tracker.eliminate_loop(trips)
    assert not tracker.activated
    assert tracker.eliminated == (factor > 1 and trips > 1)
    if factor == 1 or trips > 1:
        tracker.validate()
    else:
        with pytest.raises(exc.BackendUnsupported, match="multi-trip"):
            tracker.validate()


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_removal_does_not_change_surviving_factor_or_weaken_single_trip(factor):
    tracker = BoundedProducerUnroll(factor)
    tracker.eliminate_loop(1)
    assert tracker.loop_factor(1) == 1
    with pytest.raises(exc.BackendUnsupported, match="multi-trip"):
        tracker.validate()
    tracker.eliminate_loop(3)
    assert not tracker.activated and tracker.eliminated
    assert tracker.loop_factor(1) == 1
    assert not tracker.activated
    assert tracker.loop_factor(3) == min(factor, 3)
    assert tracker.loop_factor(17) == factor
    assert tracker.activated
    tracker.validate()


def test_elimination_state_is_local_and_original_root_contract_is_unchanged():
    removed = BoundedProducerUnroll(8)
    removed.eliminate_loop(2)
    removed.validate()
    with pytest.raises(exc.BackendUnsupported):
        BoundedProducerUnroll(8).validate()
    root = PointwiseUnroll(8)
    assert root.loop_factor(1) == 1
    with pytest.raises(exc.BackendUnsupported, match="admitted vector"):
        root.validate()
    with pytest.raises(exc.BackendUnsupported, match="whole number"):
        root.loop_factor(3)
    assert not root.activated
    assert root.loop_factor(16) == 8
    root.validate()
