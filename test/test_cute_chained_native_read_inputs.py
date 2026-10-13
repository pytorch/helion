from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention import _case
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
from helion._compiler.cute import chained_preparation_pipeline as preparation
from helion._compiler.cute.chained_frontier_groups import plan_frontier_group
from helion._compiler.cute.chained_native_read_inputs import NativeReadInput
from helion._compiler.cute.chained_native_read_inputs import bind_retained_native_inputs
from helion._compiler.cute.chained_operand_retention import discover_operand_retention
from helion._compiler.cute.chained_operand_retention import plan_operand_retention_frame
from helion._compiler.cute.warp_specialized_plan import SharedBufferRequest
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan


def _selected(dtype=torch.bfloat16, *, reserve_group=False, **kwargs):
    plan, frame, shapes, values = _case(dtype, **kwargs)
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates
    reservations = (
        tuple(
            SharedBufferRequest(
                name,
                frame.layout.region(name).byte_size,
                alignment=128,
                live_from=3,
                live_until=frame.layout.region(name).live_until,
            )
            for action in frame.actions[3:6]
            for name in action.writes
        )
        if reserve_group
        else ()
    )
    retained = plan_operand_retention_frame(
        plan,
        frame,
        candidates,
        shapes,
        capacity_bytes=frame.layout.allocated_bytes,
        reservations=reservations,
    )
    assert retained is not None
    published = {item.node: f"published_{i}" for i, item in enumerate(candidates)}
    return plan, retained, published, values


def _bind(
    plan, retained, published, first=3, stop=4, *, writes=None, frame=None, group=None
):
    current = retained.frame if frame is None else frame
    if writes is None:
        writes = tuple(
            name for action in current.actions[first:stop] for name in action.writes
        )
    return bind_retained_native_inputs(
        plan, current, retained, published, first, stop, writes, frontier_group=group
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
def test_original_typed_node_full_owner_and_already_offset_alias(dtype, transpose):
    plan, retained, published, values = _selected(dtype, transpose=transpose)
    before = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        for index, node in enumerate(values):
            inputs = _bind(plan, retained, published, 3 + index, 4 + index)
            assert len(inputs) == 1
            source = inputs[0]
            assert source.node is node
            assert source.tensor == published[node]
            assert source.index == index
            assert source.shape == (32, 128)
            assert source.full_shape == ((32, 128) if index == 0 else (64, 128))
            assert source.row_offset == (32 if index == 2 else 0)
            assert source.dtype == dtype
            assert source.matches(plan, published)
            assert not source.matches(plan, {node: "another_alias"})
            assert not source.matches(plan, {})
    assert torch.cuda.is_initialized() == before


def test_group_interval_keeps_full_owner_live_through_last_publication():
    plan, retained, published, values = _selected()
    # A expires after its first frontier. The full grouped B owner survives
    # through both later consumers; each member retains its original ordinal.
    inputs = _bind(plan, retained, published, 3, 6)
    assert tuple(item.node for item in inputs) == values[1:]
    assert tuple(item.index for item in inputs) == (1, 2)
    assert _bind(plan, retained, published, 3, 4)[0].node is values[0]
    assert _bind(plan, retained, published, 3, 7) == ()


def test_published_receipt_and_actual_consumer_are_both_required():
    plan, retained, published, values = _selected()
    assert _bind(plan, retained, {}) == ()
    assert _bind(plan, retained, {values[1]: published[values[1]]}) == ()
    assert _bind(plan, retained, published, 1, 2) == ()
    for invalid in ("", "native + 1", None, 1):
        assert _bind(plan, retained, {values[0]: invalid}) == ()
    # The distinct equal-valued expression is not the retained SSA node.
    other_plan, other, other_published, _ = _selected()
    assert _bind(plan, retained, other_published) == ()
    assert _bind(other_plan, other, published) == ()


@pytest.mark.parametrize(
    "first,stop,writes",
    [
        (True, 4, ("frontier_0",)),
        (3, 4.0, ("frontier_0",)),
        (-1, 4, ("frontier_0",)),
        (4, 3, ("frontier_0",)),
        (3, 3, ()),
        (3, 8, ("frontier_0",)),
        (3, 4, ()),
        (3, 4, ("frontier_1",)),
        (3, 4, ("frontier_0", "first_a")),
        (3, 4, ("frontier_0", "frontier_0")),
        (3, 4, ["frontier_0"]),
        (3, 4, (None,)),
    ],
)
def test_exact_interval_and_complete_write_set(first, stop, writes):
    plan, retained, published, _ = _selected()
    assert _bind(plan, retained, published, first, stop, writes=writes) == ()


@pytest.mark.parametrize("field", ["dtype", "shape", "stride", "args"])
def test_current_graph_metadata_and_expression_revision_are_checked(field):
    plan, retained, published, values = _selected()
    source = _bind(plan, retained, published)[0]
    node = values[0]
    if field == "dtype":
        node.meta["val"] = torch.empty((32, 128), dtype=torch.float32)
    elif field == "shape":
        node.meta["val"] = torch.empty((16, 128), dtype=torch.bfloat16)
    elif field == "stride":
        node.meta["val"] = torch.empty((128, 32), dtype=torch.bfloat16).T
    else:
        node.args = (node.args[0], torch.float16)
    assert not source.matches(plan, published)
    assert _bind(plan, retained, published) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"tensor": "different"},
        {"full_shape": (64, 128)},
        {"shape": (16, 128)},
        {"row_offset": 8},
        {"dtype": torch.float16},
        {"index": 1},
        {"index": False},
        {"row_offset": False},
        {"shape": (32.0, 128)},
        {"full_shape": (32, 128.0)},
    ],
)
def test_public_record_fields_cannot_be_detached_from_selection(change):
    plan, retained, published, _ = _selected()
    source = _bind(plan, retained, published)[0]
    changed = replace(source, **change)
    assert not changed.matches(plan, published | {changed.node: changed.tensor})
    with pytest.raises(FrozenInstanceError):
        cast("Any", source).tensor = "mutable"


def test_plain_constructor_has_no_selection_authority():
    plan, retained, published, values = _selected()
    source = NativeReadInput(
        values[0], published[values[0]], (32, 128), (32, 128), 0, torch.bfloat16, 0
    )
    assert not source.matches(plan, published)


def test_only_original_loop_workspace_remapping_is_allowed():
    plan, retained, published, _ = _selected()
    remapped = replace(plan, loop_workspace=SharedMemoryLayoutPlan((), 128))
    sources = _bind(remapped, retained, published)
    assert len(sources) == 1 and sources[0].matches(remapped, published)
    assert not sources[0].matches(plan, published)
    assert _bind(replace(plan, threads=plan.threads * 2), retained, published) == ()
    assert _bind(plan, retained, published, frame=replace(retained.frame)) == ()


@pytest.mark.parametrize("changed", ["candidate", "frame", "groups", "offsets"])
def test_replaced_derived_retention_records_fail_closed(changed):
    plan, retained, published, _ = _selected()
    if changed == "candidate":
        retained = replace(
            retained,
            candidates=(replace(retained.candidates[0], row_offset=1),),
        )
    elif changed == "frame":
        retained = replace(retained, frame=replace(retained.frame, stages=()))
    elif changed == "groups":
        retained = replace(cast("Any", retained), original_groups=(None,))
    else:
        retained = replace(retained, owner_offsets=())
    assert _bind(plan, retained, published) == ()


def test_only_proved_existing_frontier_group_uses_first_action_read_lease():
    plan, retained, published, values = _selected(reserve_group=True)
    group = plan_frontier_group(retained.frame, 3)
    assert group is not None and group.stop_event == 6
    assert retained.frame.layout.region("first_a").live_until == 4
    assert tuple(item.index for item in _bind(plan, retained, published, 3, 6)) == (
        1,
        2,
    )
    inputs = _bind(plan, retained, published, 3, 6, group=group)
    assert tuple(item.node for item in inputs) == values
    assert all(item.matches(plan, published) for item in inputs)
    assert _bind(plan, retained, published, 3, 5, group=group) == ()
    assert _bind(plan, retained, published, 4, 6, group=group) == ()
    assert (
        _bind(
            plan,
            retained,
            published,
            3,
            6,
            group=replace(group, buffers=group.buffers[:1]),
        )
        == ()
    )
    other_plan, other, _, _ = _selected(reserve_group=True)
    other_group = plan_frontier_group(other.frame, 3)
    assert other_group is not None
    assert _bind(plan, retained, published, 3, 6, group=other_group) == ()
    assert _bind(other_plan, other, published, 3, 6, group=other_group) == ()


def test_group_cannot_bypass_current_read_publication_or_whole_storage_proof():
    plan, retained, published, _ = _selected()
    # A previous-frame group may remain structurally identical after repacking.
    # It is accepted only when the current frame independently proves it again.
    old_group = plan_frontier_group(retained.revision.frame, 3)
    assert old_group is not None
    assert old_group == plan_frontier_group(retained.frame, 3)
    assert len(_bind(plan, retained, published, 3, 6, group=old_group)) == 3
    with patch(
        "helion._compiler.cute.chained_frontier_groups.plan_frontier_group",
        return_value=None,
    ):
        assert _bind(plan, retained, published, 3, 6, group=old_group) == ()
    plan, retained, published, _ = _selected(reserve_group=True)
    group = plan_frontier_group(retained.frame, 3)
    assert group is not None
    assert (
        _bind(plan, retained, published, 3, 6, writes=("frontier_0",), group=group)
        == ()
    )
    assert _bind(plan, retained, {}, 3, 6, group=group) == ()
    # A caller cannot carry a group receipt past a mutation of its read facts.
    changed = replace(
        retained,
        frame=replace(
            retained.frame,
            actions=tuple(
                replace(action, reads=("result_0",)) if action.event == 3 else action
                for action in retained.frame.actions
            ),
        ),
    )
    assert _bind(plan, changed, published, 3, 6, group=group) == ()


@pytest.mark.parametrize("leaves", [1, 4])
def test_actual_retained_frame_preserves_group_layout_and_original_source(leaves):
    kernel, args = _kda_fixture()
    config = _kda_config(leaves=leaves)
    original = preparation._prepare
    captured = []

    def observe(cg, plan, pipeline, *rest):
        retained = pipeline.operand_retention
        assert retained is not None
        frame = pipeline.frame
        seen = set()
        grouped_seen = set()
        singleton_seen = set()
        # Replay the actual original action publication cuts. The production
        # caller supplies the same map only after the corresponding stage runs.
        for action in frame.actions:
            if action.kind != "frontier":
                continue
            published = {
                item.node: f"published_{index}"
                for index, item in enumerate(retained.candidates)
                if item.publication_event <= action.event
            }
            group = plan_frontier_group(frame, action.event)
            stop = action.event + 1 if group is None else group.stop_event
            writes = tuple(
                name
                for part in frame.actions[action.event : stop]
                for name in part.writes
            )
            inputs = bind_retained_native_inputs(
                plan, frame, retained, published, action.event, stop, writes
            )
            if group is not None:
                grouped = bind_retained_native_inputs(
                    plan,
                    frame,
                    retained,
                    published,
                    action.event,
                    stop,
                    writes,
                    frontier_group=group,
                )
                grouped_seen.update(item.index for item in grouped)
                assert all(item.matches(plan, published) for item in grouped)
            singleton_seen.update(
                item.index
                for item in bind_retained_native_inputs(
                    plan,
                    frame,
                    retained,
                    published,
                    action.event,
                    action.event + 1,
                    action.writes,
                )
            )
            for source in inputs:
                assert source.matches(plan, published)
                item = retained.candidates[source.index]
                assert source.node is item.node
                owner = frame.layout.region(item.owner)
                assert owner.live_from <= action.event < stop <= owner.live_until
                for name in writes:
                    spans = [
                        (
                            frame.layout.region(name).byte_offset,
                            frame.layout.region(name).byte_end,
                        )
                    ]
                    for binding in retained.prepared_groups:
                        if name in {
                            member.buffer.name for member in binding.candidate.members
                        }:
                            spans.append(
                                (
                                    binding.byte_offset,
                                    binding.byte_offset + binding.candidate.byte_size,
                                )
                            )
                    assert all(
                        owner.byte_end <= start or end <= owner.byte_offset
                        for start, end in spans
                    )
                seen.add(source.index)
        assert seen == {1, 2}
        assert singleton_seen == {0, 1, 2}
        assert grouped_seen == {0, 1, 2}
        assert retained.candidates[3].full_shape[1] == 32  # Not K_SW128.
        assert retained.prepared_groups
        captured.append(frame.layout.allocated_bytes)
        return original(cg, plan, pipeline, *rest)

    before = torch.cuda.is_initialized()
    expected = _source(kernel, args, config)
    with patch.object(preparation, "_prepare", observe):
        actual = _source(kernel, args, config)
    assert actual == expected
    assert len(captured) == 1
    if leaves == 4:
        assert captured == [65536]
    assert torch.cuda.is_initialized() == before
