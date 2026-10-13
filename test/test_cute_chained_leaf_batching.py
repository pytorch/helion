from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_leaf_sets import _inputs
from .test_cute_chained_leaf_sets import _reused_raw
from .test_cute_chained_preparation_leaves import _leaf_config
from .test_cute_chained_preparation_leaves import _source
from helion import exc
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute.chained_leaf_schedule_emission import seal_leaf_schedule
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_preparation_storage import bind_preparation_storage
from helion._compiler.cute.chained_preparation_storage import (
    plan_scheduled_preparation_storage,
)
from helion._compiler.cute.chained_vector_stage import VectorStaging

KEY = "cute_chained_leaf_issue_batching"


def _config(enabled=None):
    config = _leaf_config()
    config.config.update(
        cute_chained_leaf_count=4,
        cute_chained_preparation_cohorts=3,
        cute_chained_compact_preparation=True,
    )
    if enabled is not None:
        config.config[KEY] = enabled
    return config


def _capture(dtype=torch.bfloat16, steps=3):
    captured = []
    original = storage.finalize_pipeline_storage

    def finalize(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        captured.append(result)
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        source = _source(_reused_raw, _inputs(dtype, steps), _config(True))
    assert len(captured) == 1
    return source, captured[0]


@pytest.fixture(scope="module")
def scheduled_case():
    return _capture()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1, 7])
def test_public_batching_true_compiles_original_masked_values(dtype, steps):
    source, result = _capture(dtype, steps)
    binding = result.preparation
    assert binding is not None
    physical = binding.physical
    body = physical.scheduled_body
    assert body is not None and body.schedule.max_pending == 2
    assert len(binding.leaf_transfers) == 2
    assert tuple(physical.layout.regions) == body.schedule.regions
    assert (
        physical.layout.allocated_bytes == body.schedule.physical.layout.allocated_bytes
    )
    assert binding._state.consumed
    for region, original in zip(
        physical.layout.regions, body.schedule.physical.layout.regions, strict=True
    ):
        assert replace(region, live_from=original.live_from) == original
    assert (
        source.count("mbarrier_arrive_and_expect_tx(chain_slot_bars + 6 + chain_slot")
        == 1
    )


def test_default_false_is_identical_and_single_original_walk(scheduled_case):
    args = _inputs(torch.bfloat16, 3)
    assert _source(_reused_raw, args, _config()) == _source(
        _reused_raw, args, _config(False)
    )
    _, result = scheduled_case
    physical = result.preparation.physical
    body = physical.scheduled_body
    assert body is not None
    assert (
        tuple(line for piece in body.recorded.pieces for line in piece.lines)
        == physical.accepted.lines
    )
    assert body.lines != physical.accepted.lines
    assert sorted(body.lines) == sorted(physical.accepted.lines)


@pytest.mark.parametrize("field", ["lines", "steps", "region", "protocol", "piece"])
def test_scheduled_body_and_extended_table_are_not_interchangeable(
    scheduled_case, field
):
    _, result = scheduled_case
    physical = result.preparation.physical
    body = physical.scheduled_body
    plan, pipeline = physical.accepted.revision.plan, physical.accepted.pipeline
    assert body is not None and body.matches(plan, pipeline)
    if field == "lines":
        changed = replace(body, lines=physical.accepted.lines)
    elif field == "steps":
        changed = replace(
            body, schedule=replace(body.schedule, steps=body.schedule.steps[1:])
        )
    elif field == "region":
        changed = replace(
            body,
            schedule=replace(
                body.schedule, regions=body.schedule.physical.layout.regions
            ),
        )
    elif field == "protocol":
        changed = replace(
            body,
            schedule=replace(
                body.schedule, protocol=replace(body.schedule.protocol, cohorts=False)
            ),
        )
    else:
        piece = replace(body.recorded.pieces[0], lines=())
        changed = replace(
            body,
            recorded=replace(body.recorded, pieces=(piece, *body.recorded.pieces[1:])),
        )
    assert not changed.matches(plan, pipeline)
    assert plan_scheduled_preparation_storage(plan, pipeline, changed) is None
    assert seal_leaf_schedule(plan, pipeline, body.recorded, body.schedule) == body


@pytest.mark.parametrize("field", ["inputs", "outputs", "lines", "protocol"])
def test_same_object_segment_mutations_reject_and_restore(scheduled_case, field):
    _, result = scheduled_case
    physical = result.preparation.physical
    body = physical.scheduled_body
    assert body is not None
    plan, pipeline = physical.accepted.revision.plan, physical.accepted.pipeline
    piece = body.recorded.pieces[0]
    previous = getattr(piece, field)
    altered = {
        "inputs": ((piece.leaf.node, "unpublished"),),
        "outputs": (),
        "lines": (),
        "protocol": ("other_bar", "other_phase"),
    }
    object.__setattr__(piece, field, altered[field])
    try:
        assert not body.matches(plan, pipeline)
        assert not physical.matches(
            plan, pipeline, dict(physical.accepted.revision.shapes)
        )
    finally:
        object.__setattr__(piece, field, previous)
    assert body.matches(plan, pipeline)


def test_late_schedule_rejection_cannot_install_any_body():
    from helion._compiler.cute import chained_matmul
    from helion._compiler.cute import chained_preparation_storage

    with (
        patch.object(
            chained_preparation_storage,
            "plan_scheduled_preparation_storage",
            return_value=None,
        ),
        patch.object(
            chained_matmul,
            "_install_chained_body",
            side_effect=AssertionError("body installed before finalization"),
        ),
        pytest.raises(exc.BackendUnsupported, match="completed schedule"),
    ):
        _source(_reused_raw, _inputs(torch.bfloat16, 1), _config(True))


def test_actual_extended_binding_exact_quota_and_single_consumption():
    original = storage.finalize_pipeline_storage
    results = []

    def finalize(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        binding = result.preparation
        physical = binding.physical
        plan, pipeline = physical.accepted.revision.plan, physical.accepted.pipeline
        fresh = bind_preparation_storage(
            plan, pipeline, physical, cache_layouts=binding.cache_layouts
        )
        assert fresh is not None
        options = kwargs.copy()
        options.update(preparation=fresh, capacity_bytes=result.charged_bytes - 1)
        assert original(*args, **options) is None
        assert fresh._state.finalized is None and not fresh._state.consumed
        options["capacity_bytes"] = result.charged_bytes
        exact = original(*args, **options)
        assert exact is not None and exact.charged_bytes == result.charged_bytes
        accepted = physical.accepted
        body = physical.scheduled_body
        assert body is not None
        vector = VectorStaging(
            accepted.vector_state[0], group_enabled=accepted.vector_state[2]
        )
        unroll = BoundedProducerUnroll(accepted.unroll_state[0])
        assert fresh.consume_body(exact, vector, unroll) == list(body.lines)
        with pytest.raises(ValueError, match="unconsumed"):
            fresh.consume_body(exact, vector, unroll)
        # Same byte charge but original synchronous table is not the bound
        # allocation authority for a scheduled body.
        assert not replace(fresh, physical=body.schedule.physical).matches(
            plan, pipeline
        )
        results.append(exact)
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        _source(_reused_raw, _inputs(torch.bfloat16, 1), _config(True))
    assert len(results) == 1
