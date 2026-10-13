from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_broadcast_expressions import _broadcast_loop
from .test_cute_chained_preparation_leaves import _source
import helion
from helion import exc
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute.chained_broadcast_retention import bind_broadcast_retention
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_preparation_storage import bind_preparation_storage
from helion._compiler.cute.chained_vector_stage import VectorStaging

KEY = "cute_chained_broadcast_retention"


def _config(enabled=None, *, scan=True, batching=False):
    config = helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_preparation_pipeline=True,
        cute_chained_preparation_cohorts=3,
        cute_chained_compact_preparation=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_vectorize=True,
        cute_chained_vector_group=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=2,
        cute_chained_scan_producer_retention=scan,
        cute_chained_leaf_issue_batching=batching,
    )
    if enabled is not None:
        config.config[KEY] = enabled
    return config


def _args(dtype=torch.bfloat16, steps=1, rounded=False):
    return (
        torch.empty((steps, 32, 64), dtype=dtype),
        torch.empty((steps, 32, 64), dtype=dtype),
        torch.empty((steps, 32, 32), dtype=dtype),
        torch.empty((32, 64)),
        rounded,
        False,
    )


def _capture(
    enabled=True,
    *,
    dtype=torch.bfloat16,
    rounded=False,
    scan=True,
    batching=False,
    check=None,
):
    captured = []
    original = storage.finalize_pipeline_storage

    def finalize(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        captured.append(result)
        if check is not None:
            check(result.preparation)
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        source = _source(
            _broadcast_loop,
            _args(dtype, rounded=rounded),
            _config(enabled, scan=scan, batching=batching),
        )
    assert len(captured) == 1
    return source, captured[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("scan", [False, True])
def test_public_original_typed_scan_subtrees(dtype, rounded, scan):
    source, result = _capture(dtype=dtype, rounded=rounded, scan=scan)
    bound = result.preparation
    assert bound is not None and bound.broadcast_transfers
    assert "chain_broadcast" in source
    assert bound._state.consumed
    assert all(item.receipt.emission.reads for item in bound.broadcast_transfers)
    assert all(
        item.transfer.stop == item.transfer.first + 1
        for item in bound.broadcast_transfers
    )


def test_default_and_false_source_exact():
    assert _capture(None)[0] == _capture(False)[0]
    assert "chain_broadcast" not in _capture(False)[0]


@pytest.mark.parametrize(
    "field", ["receipt", "read", "scope", "lines", "writes", "bound", "role", "dtype"]
)
def test_current_receipt_and_complete_writes_required(field):
    def check(bound):
        physical = bound.physical
        accepted = physical.accepted
        plan, pipeline = accepted.revision.plan, accepted.pipeline
        receipt = accepted.broadcasts[0]
        if field == "bound":
            assert not replace(bound, broadcast_transfers=()).matches(plan, pipeline)
            return
        if field == "receipt":
            target, name, value = accepted, "broadcasts", ()
        elif field == "read":
            target, name, value = receipt.emission.reads[0], "coordinates", ("999",)
        elif field == "scope":
            target, name, value = receipt, "first", receipt.first + 1
        elif field == "lines":
            target, name, value = receipt.emission, "before_steps", ()
        elif field == "role":
            target, name, value = receipt.execution, "thread", "different_thread"
        elif field == "dtype":
            target, name, value = receipt.emission.reads[0], "dtype", torch.float16
        else:
            target = next(action for action in accepted.actions if action.broadcasts)
            name, value = "writes", ()
        previous = getattr(target, name)
        object.__setattr__(target, name, value)
        try:
            assert not bound.matches(plan, pipeline)
            assert bind_preparation_storage(plan, pipeline, physical) is None
            assert bound._state.finalized is not None
            with pytest.raises(ValueError, match="unconsumed final allocation"):
                bound.consume_body(
                    bound._state.finalized,
                    VectorStaging(
                        accepted.vector_state[0], group_enabled=accepted.vector_state[2]
                    ),
                    BoundedProducerUnroll(accepted.unroll_state[0]),
                )
            assert not bound._state.consumed
        finally:
            object.__setattr__(target, name, previous)
        assert bound.matches(plan, pipeline)

    _capture(check=check)


def test_tma_schedule_keeps_broadcast_receipts():
    _, result = _capture(batching=True)
    bound = result.preparation
    assert bound is not None
    assert bound.physical.scheduled_body is not None
    assert bound.broadcast_transfers and bound.leaf_transfers


def test_pending_tma_destination_interference_is_not_a_read_lease():
    def check(bound):
        physical = bound.physical
        source = bound.broadcast_transfers[0].transfer.sources[0]
        body = physical.scheduled_body
        assert body is not None
        leaf = physical.layout.region(body.schedule.leaves[0].owner)
        # A later alias is legal only while inactive at the original read.
        # Making its async destination active over this phase must reject,
        # independently of the table's earlier acceptance.
        phase = bound.broadcast_transfers[0].transfer.first
        changed = replace(
            leaf, byte_offset=source.byte_offset, live_from=phase, live_until=phase + 1
        )
        regions = tuple(
            changed if item.name == leaf.name else item
            for item in physical.layout.regions
        )
        bad = replace(physical, layout=replace(physical.layout, regions=regions))
        assert bind_broadcast_retention(bad) is None
        assert bind_broadcast_retention(physical) == bound.broadcast_transfers

    _capture(batching=True, check=check)


@pytest.mark.parametrize("drop", ["schedule", "receipt"])
def test_scheduled_body_cannot_drop_broadcast_authority_before_consumption(drop):
    def check(bound):
        physical = bound.physical
        accepted = physical.accepted
        target, name = (
            (physical, "scheduled_body")
            if drop == "schedule"
            else (accepted, "broadcasts")
        )
        previous = getattr(target, name)
        object.__setattr__(target, name, None if drop == "schedule" else ())
        try:
            assert not bound.matches(accepted.revision.plan, accepted.pipeline)
            assert (
                bind_preparation_storage(
                    accepted.revision.plan, accepted.pipeline, physical
                )
                is None
            )
        finally:
            object.__setattr__(target, name, previous)

    _capture(batching=True, check=check)


def test_unsupported_noncompact_has_no_fallback():
    config = _config(True)
    config.config["cute_chained_compact_preparation"] = False
    with pytest.raises(exc.InvalidConfig, match=KEY):
        _source(_broadcast_loop, _args(), config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group", [False, True])
def test_original_single_and_group_vector_clients(dtype, group):
    config = _config(True, scan=False)
    config.config.update(
        cute_chained_vector_group=group,
        cute_chained_leaf_pipeline="legacy",
        cute_chained_leaf_count=1,
    )
    args = _args(dtype)
    selected = []
    original = storage.finalize_pipeline_storage

    def finalize(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        selected.extend(result.preparation.physical.accepted.broadcasts)
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        source = _source(_broadcast_loop, args, config)
    assert selected and "chain_broadcast" in source
    assert all(
        item.emission.candidate.ownership.thread_order == "row_major"
        for item in selected
    )
    assert any("_group_step" in "\n".join(item.lines) for item in selected) is group
