from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from itertools import product
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_leaf_sets import _inputs
from .test_cute_chained_leaf_sets import _reused_raw
from .test_cute_chained_preparation_leaves import _leaf_config
from .test_cute_chained_preparation_leaves import _source
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute.chained_leaf_schedule import LeafScheduleAction
from helion._compiler.cute.chained_leaf_schedule import _schedule
from helion._compiler.cute.chained_leaf_schedule import emit_leaf_transfer
from helion._compiler.cute.chained_leaf_schedule import plan_leaf_schedule
from helion._compiler.cute.chained_preparation_leaves import PreparationLeaf
from helion._compiler.cute.chained_rectangular_leaf import RectangularLeafPlan
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion


def _leaf(name):
    node = Graph().placeholder(name)
    node.meta["val"] = torch.empty((16, 64), dtype=torch.bfloat16)
    proof = RectangularLeafPlan(
        indices=("r", "c"),
        mask=None,
        row="r",
        column="c",
        tile_shape=(16, 64),
        shape=(32, 64),
        strides=(64, 1),
        dtype="BFloat16",
        element_bytes=2,
        pitch=64,
        view_shape=(32, 64),
        origin_indices=("0", "0"),
        base="0",
        origin=("0", "0"),
        guard="True",
    )
    return PreparationLeaf(node, name, proof, 0, 10, (10,), {})


def _case():
    leaves = tuple(
        (_leaf(name), index, name) for index, name in enumerate(("x", "y", "z"))
    )
    actions = (
        LeafScheduleAction(0, 0, 0, "leaf", (), ("x",)),
        LeafScheduleAction(1, 1, 2, "leaf", (), ("y",)),
        LeafScheduleAction(1, 2, 4, "action", ("y",), ("reduction_y",)),
        LeafScheduleAction(1, 3, 6, "leaf", (), ("z",)),
        LeafScheduleAction(1, 4, 8, "action", ("z",), ("reduction_z",)),
        LeafScheduleAction(1, 5, 10, "scan_body", ("x", "y", "z"), ("out",)),
        LeafScheduleAction(2, 6, 12, "action", (), ()),
    )
    regions = tuple(
        SharedBufferRegion(name, index * 2048, 2048, start, 12, 128)
        for index, (name, start) in enumerate((("x", 0), ("y", 2), ("z", 6)))
    )
    return actions, leaves, regions


def test_independent_issues_batch_but_each_wait_stays_at_first_actual_reader():
    actions, leaves, regions = _case()
    result = _schedule(actions, leaves, regions)
    assert result is not None
    selected, steps, updated, maximum = result
    assert [(item.issue_before, item.complete_before) for item in selected] == [
        (0, 5),
        (0, 2),
        (0, 4),
    ]
    assert [(step.kind, step.index) for step in steps] == [
        ("issue", 0),
        ("issue", 1),
        ("issue", 2),
        ("complete", 1),
        ("action", 2),
        ("complete", 2),
        ("action", 4),
        ("complete", 0),
        ("action", 5),
        ("action", 6),
    ]
    assert maximum == 3
    assert all(region.live_from == 0 for region in updated)
    assert [
        replace(new, live_from=old.live_from)
        for new, old in zip(updated, regions, strict=True)
    ] == list(regions)
    assert tuple(step.index for step in steps if step.kind == "action") == (2, 4, 5, 6)
    with pytest.raises(FrozenInstanceError):
        selected[0].owner = "other"  # pyrefly: ignore [read-only]


@pytest.mark.parametrize(
    "end,expected", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)]
)
def test_temporal_alias_is_legal_but_limits_the_earlier_async_write(end, expected):
    actions, leaves, regions = _case()
    alias = SharedBufferRegion("prior_scratch", 4096, 2048, 0, end, 128)
    result = _schedule(actions, leaves, (*regions, alias))
    assert result is not None
    selected, _, updated, _ = result
    assert selected[2].issue_before == expected
    assert updated[2].live_from == actions[expected].phase
    assert updated[-1] == alias


def test_full_padded_native_owner_not_just_logical_leaf_bytes_is_checked():
    actions, leaves, regions = _case()
    regions = (*regions[:2], replace(regions[2], byte_size=4096))
    padded_alias = SharedBufferRegion("native_member_padding", 6144, 2048, 0, 4, 128)
    result = _schedule(actions, leaves, (*regions, padded_alias))
    assert result is not None and result[0][2].issue_before == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_leaf",
        "duplicate_owner",
        "duplicate_region",
        "no_reader",
        "early_read",
        "expired",
        "bad_phase",
        "backward_phase",
        "unowned",
        "existing_interference",
        "omitted_leaf",
        "materialized_origin",
    ],
)
def test_missing_stale_or_conflicting_interval_facts_fail_closed(mutation):
    actions, leaves, regions = _case()
    if mutation == "duplicate_leaf":
        leaves = (*leaves, leaves[0])
    elif mutation == "duplicate_owner":
        leaves = (*leaves[:2], (leaves[2][0], 2, "x"))
    elif mutation == "duplicate_region":
        regions = (*regions, regions[0])
    elif mutation == "no_reader":
        actions = tuple(
            replace(action, reads=tuple(name for name in action.reads if name != "z"))
            for action in actions
        )
    elif mutation == "early_read":
        actions = (replace(actions[0], reads=("z",)), *actions[1:])
    elif mutation == "expired":
        regions = (*regions[:2], replace(regions[2], live_until=10))
    elif mutation == "bad_phase":
        actions = (replace(actions[0], phase=True), *actions[1:])
    elif mutation == "backward_phase":
        actions = (*actions[:3], replace(actions[3], phase=3), *actions[4:])
    elif mutation == "unowned":
        leaves = (*leaves[:2], (leaves[2][0], 2, "missing"))
    elif mutation == "omitted_leaf":
        leaves = leaves[:2]
    elif mutation == "materialized_origin":
        actions = (
            *actions[:3],
            replace(actions[3], reads=("reduction_y",)),
            *actions[4:],
        )
    else:
        regions = (*regions, SharedBufferRegion("conflict", 4096, 128, 6, 7, 128))
    assert _schedule(actions, leaves, regions) is None


def test_no_concurrent_pending_input_is_not_a_meaningful_batch():
    actions, leaves, regions = _case()
    actions = tuple(
        action for action in actions if action.kind != "leaf" or action.writes == ("x",)
    )
    assert _schedule(actions, leaves[:1], regions) is None


@pytest.mark.parametrize("full", list(product((False, True), repeat=3)))
def test_pending_state_not_completed_by_issue_or_scalar_fallback(full):
    actions, leaves, regions = _case()
    result = _schedule(actions, leaves, regions)
    assert result is not None
    selected, steps, _, _ = result
    for generation in range(8):
        pending, completed = set(), set()
        arrivals = [0, 0, 0]
        for step in steps:
            if step.kind == "issue":
                assert step.index not in pending | completed
                pending.add(step.index)
                # Full issue expects bytes; fallback arrives synchronously.
                arrivals[step.index] = (
                    leaves[step.index][0].byte_size if full[step.index] else 1
                )
            elif step.kind == "complete":
                assert step.index in pending and arrivals[step.index] > 0
                assert generation & 1 in (0, 1)
                pending.remove(step.index)
                completed.add(step.index)
            else:
                for name in actions[step.index].reads:
                    assert (
                        next(
                            i
                            for i, leaf in enumerate(selected)
                            if leaf.leaf.name == name
                        )
                        in completed
                    )
        assert not pending and len(completed) == 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("cohorts,steps", [(1, 0), (1, 3), (3, 3)])
def test_split_renderer_reassembles_original_source_including_masked_fallback(
    dtype, cohorts, steps
):
    args = _inputs(dtype, steps)
    config = _leaf_config()
    config.config.update(
        cute_chained_leaf_count=4, cute_chained_preparation_cohorts=cohorts
    )
    before = torch.cuda.is_initialized()
    original = _source(_reused_raw, args, config)
    calls = []

    def split(self, cg, plan, boundaries, execution, barrier, phase):
        emission = emit_leaf_transfer(
            self, cg, plan, boundaries, execution, barrier, phase
        )
        calls.append((self, emission, barrier, phase))
        assert emission.completion == (
            f"cute.arch.mbarrier_wait({barrier}, {phase})",
            execution.sync,
        )
        assert "mbarrier_wait" not in "\n".join(emission.issue)
        assert "else:" in emission.issue
        return emission.lines()

    with patch.object(PreparationLeaf, "emit", split):
        actual = _source(_reused_raw, args, config)
    assert calls and actual == original
    assert torch.cuda.is_initialized() == before


@pytest.fixture(scope="module")
def physical_case():
    config = _leaf_config()
    config.config.update(
        cute_chained_leaf_count=4,
        cute_chained_preparation_cohorts=3,
        cute_chained_compact_preparation=True,
    )
    captured = []
    original = storage.finalize_pipeline_storage

    def finalize(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        physical = result.preparation.physical
        plan, pipeline = physical.accepted.revision.plan, physical.accepted.pipeline
        schedule = plan_leaf_schedule(plan, pipeline, physical)
        assert schedule is not None and schedule.matches(plan, pipeline)
        captured.append((plan, pipeline, physical, schedule))
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        _source(_reused_raw, _inputs(torch.bfloat16, 3), config)
    assert len(captured) == 1
    return captured[0]


def test_factory_uses_actual_accepted_table_and_role_local_participants(physical_case):
    plan, pipeline, physical, schedule = physical_case
    assert schedule.matches(plan, pipeline)
    assert physical.accepted.execution.threads == pipeline.cohorts.cohort_threads
    assert len(schedule.leaves) == 2 and schedule.max_pending == 2
    assert schedule.protocol.slots == pipeline.slots == 3
    assert schedule.protocol.leaf_count == 2 and schedule.protocol.cohorts
    assert schedule.protocol.phase == "chain_generation & 1"
    assert {
        schedule.protocol.leaf_index(leaf.ordinal, slot)
        for leaf in schedule.leaves
        for slot in range(3)
    } == set(range(6, 12))
    assert physical.layout.regions != schedule.regions
    for old, new in zip(physical.layout.regions, schedule.regions, strict=True):
        assert new.live_from <= old.live_from
        assert replace(new, live_from=old.live_from) == old


@pytest.mark.parametrize(
    "field", ["steps", "regions", "max_pending", "protocol", "actions"]
)
def test_derived_schedule_fields_are_not_forgeable(physical_case, field):
    plan, pipeline, _, schedule = physical_case
    values = {
        "steps": schedule.steps[:-1],
        "regions": schedule.regions[:-1],
        "max_pending": schedule.max_pending + 1,
        "protocol": replace(schedule.protocol, cohorts=False),
        "actions": schedule.actions[:-1],
    }
    assert not replace(schedule, **{field: values[field]}).matches(plan, pipeline)


@pytest.mark.parametrize("field", ["descriptor", "frame", "physical", "graph"])
def test_stale_original_transfer_graph_and_physical_facts_reject(physical_case, field):
    plan, pipeline, physical, schedule = physical_case
    if field == "descriptor":
        wrapper = pipeline.prepared_leaves[0].wrapper
        with patch.dict(wrapper, {"kernel_args": ["other_atom", "other_tensor"]}):
            assert not schedule.matches(plan, pipeline)
    elif field == "frame":
        changed = replace(
            pipeline, frame=replace(pipeline.frame, actions=pipeline.frame.actions[:-1])
        )
        assert plan_leaf_schedule(plan, changed, physical) is None
    elif field == "physical":
        changed = replace(
            physical,
            layout=replace(physical.layout, regions=physical.layout.regions[:-1]),
        )
        assert plan_leaf_schedule(plan, pipeline, changed) is None
    else:
        node = pipeline.prepared_leaves[0].node
        with patch.dict(node.meta, {"val": node.meta["val"].to(torch.float32)}):
            assert not schedule.matches(plan, pipeline)
    assert schedule.matches(plan, pipeline)
