from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _input
from .test_cute_chained_preparation_frame import _synthetic
import helion
from helion._compiler.cute import chained_frontier_groups as groups
from helion._compiler.cute import chained_prepared_values
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_frame import PreparationFrame
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_preparation_pipeline import PreparationPipeline
from helion._compiler.cute.chained_preparation_pipeline import _prepare
from helion._compiler.cute.chained_recurrence_workspace import RecurrenceWorkspace
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_vector_stage import VectorStaging
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_preparation_frame import ActionKind


def _frame(
    count: int = 3,
    *,
    first: int = 2,
    shape: tuple[int, ...] = (8, 8),
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> tuple[ChainedMatmulPlan, PreparationFrame]:
    # Reuse a real typed graph/cut/frame contract; only byte actions and images
    # are replaced to isolate this planner from workload discovery/emission.
    plan, cut, shapes = _synthetic()
    original = plan_preparation_frame(plan, cut, shapes)
    assert original is not None
    graph = Graph()
    dtypes = dtypes or (torch.bfloat16,) * count
    assert len(dtypes) == count
    buffers = tuple(
        PreparationBuffer(
            f"image_{index}",
            "frontier",
            _input(graph, f"value_{index}", shape, dtype),
            dtype,
            shape,
        )
        for index, dtype in enumerate(dtypes)
    )
    end = first + count + 1
    regions = [SharedBufferRegion("prefix", 0, 128, 0, end, 128)]
    offset = 128
    for index, buffer in enumerate(buffers):
        size = (math.prod(shape) * buffer.dtype.itemsize + 127) // 128 * 128
        regions.append(
            SharedBufferRegion(buffer.name, offset, size, first + index, end, 128)
        )
        offset += size
    actions = (
        *(
            PreparationAction(
                "frontier", first + index, (), (), None, ("prefix",), (buffer.name,)
            )
            for index, buffer in enumerate(buffers)
        ),
        PreparationAction("ready", end - 1, (), (), None, (), ()),
    )
    return plan, replace(
        original,
        layout=SharedMemoryLayoutPlan(tuple(regions), offset),
        buffers=buffers,
        actions=actions,
        stages=(),
        frontier_order=tuple(
            buffer.node for buffer in buffers if buffer.node is not None
        ),
        peak_live_bytes=offset,
    )


def _region(
    frame: PreparationFrame,
    name: str,
    *,
    byte_offset: int | None = None,
    byte_size: int | None = None,
    live_from: int | None = None,
    live_until: int | None = None,
) -> PreparationFrame:
    return replace(
        frame,
        layout=replace(
            frame.layout,
            regions=tuple(
                replace(
                    region,
                    byte_offset=region.byte_offset
                    if byte_offset is None
                    else byte_offset,
                    byte_size=region.byte_size if byte_size is None else byte_size,
                    live_from=region.live_from if live_from is None else live_from,
                    live_until=region.live_until if live_until is None else live_until,
                )
                if region.name == name
                else region
                for region in frame.layout.regions
            ),
        ),
    )


@pytest.mark.parametrize("count", [2, 3, 5])
def test_maximal_adjacent_prefix_preserves_entire_original_frame(count: int) -> None:
    _, frame = _frame(count)
    layout, buffers, actions = frame.layout, frame.buffers, frame.actions
    intervals = tuple(vars(region).copy() for region in layout.regions)
    graph = frame.frontier_order[0].graph
    graph_before = str(graph)
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None
    assert (result.first_event, result.stop_event) == (2, 2 + count)
    assert result.buffers == buffers
    assert (
        frame.layout is layout and frame.buffers is buffers and frame.actions is actions
    )
    assert tuple(vars(region).copy() for region in layout.regions) == intervals
    assert str(graph) == graph_before
    with pytest.raises(FrozenInstanceError):
        result.stop_event = 99  # type: ignore[misc]


@pytest.mark.parametrize("retained", [2, 3])
def test_later_alias_of_live_prefix_keeps_maximal_safe_earlier_group(
    retained: int,
) -> None:
    _, frame = _frame(retained + 1)
    late_event = 2 + retained
    frame = _region(frame, "prefix", live_until=late_event)
    frame = _region(frame, f"image_{retained}", byte_offset=0)
    frame = replace(
        frame,
        actions=tuple(
            replace(action, reads=()) if action.event == late_event else action
            for action in frame.actions
        ),
    )
    old = frame.layout.region(f"image_{retained}")
    assert not old.overlaps_lifetime(frame.layout.region("prefix"))
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None
    assert result.stop_event == late_event
    assert result.buffers == frame.buffers[:retained]


@pytest.mark.parametrize(
    "live_from,live_until,accepted",
    [
        (0, 3, True),
        (1, 3, True),
        (2, 4, False),
        (3, 5, False),
        (0, 2, False),
        (0, 1, False),
    ],
)
def test_reads_must_be_published_and_live_at_first_event(
    live_from: int, live_until: int, accepted: bool
) -> None:
    _, frame = _frame(2)
    frame = _region(frame, "prefix", live_from=live_from, live_until=live_until)
    assert (groups.plan_frontier_group(frame, 2) is not None) is accepted


def test_later_member_cannot_read_a_value_first_published_by_the_group() -> None:
    _, frame = _frame(3)
    frame = replace(
        frame,
        actions=tuple(
            replace(action, reads=("image_0",)) if action.event == 3 else action
            for action in frame.actions
        ),
    )
    assert groups.plan_frontier_group(frame, 2) is None


@pytest.mark.parametrize("old_until,accepted", [(1, True), (2, True), (3, False)])
def test_reused_destination_lifetime_is_strictly_half_open(
    old_until: int, accepted: bool
) -> None:
    _, frame = _frame(2)
    old = SharedBufferRegion("dead_scratch", 256, 128, 0, old_until, 128)
    frame = replace(
        frame, layout=replace(frame.layout, regions=(*frame.layout.regions, old))
    )
    assert (groups.plan_frontier_group(frame, 2) is not None) is accepted


def test_exact_storage_endpoint_is_not_an_overlap() -> None:
    _, frame = _frame(2)
    old = SharedBufferRegion("neighbor", frame.layout.allocated_bytes, 128, 0, 5, 128)
    frame = replace(
        frame,
        layout=replace(
            frame.layout,
            regions=(*frame.layout.regions, old),
            allocated_bytes=old.byte_end,
        ),
    )
    assert groups.plan_frontier_group(frame, 2) is not None


def test_mixed_float_storage_types_keep_exact_buffers_and_geometry() -> None:
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    _, frame = _frame(dtypes=dtypes)
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None
    assert tuple(buffer.dtype for buffer in result.buffers) == dtypes
    assert result.buffers == frame.buffers


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_small_rank2_payload_preserves_128_byte_padded_region(
    dtype: torch.dtype,
) -> None:
    _, frame = _frame(2, shape=(3, 8), dtypes=(dtype, dtype))
    assert frame.layout.region("image_0").byte_size == 128
    assert 24 * dtype.itemsize < 128
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None and result.buffers == frame.buffers


def test_early_write_checks_allocated_padding_not_only_logical_payload() -> None:
    _, frame = _frame(3, shape=(3, 8))
    # The late destination's48B payload does not reach the old live scratch at
    # 128, but its full256B allocation does. The original event4 reuse is legal.
    frame = _region(frame, "image_2", byte_offset=0, byte_size=256)
    frame = _region(frame, "prefix", byte_offset=128, live_until=4)
    frame = _region(frame, "image_0", byte_offset=256)
    frame = _region(frame, "image_1", byte_offset=384)
    frame = replace(
        frame,
        actions=tuple(
            replace(action, reads=()) if action.event == 4 else action
            for action in frame.actions
        ),
    )
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None and result.stop_event == 4


@pytest.mark.parametrize("shape", [(8,), (8, 8, 1), (4, 16), (0, 8)])
def test_shape_separator_keeps_safe_first_pair(shape: tuple[int, ...]) -> None:
    _, frame = _frame()
    frame = replace(
        frame, buffers=(*frame.buffers[:2], replace(frame.buffers[2], shape=shape))
    )
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None and result.buffers == frame.buffers[:2]


@pytest.mark.parametrize(
    "kind", ["collective", "cache", "fill", "mma", "ready", "leaf"]
)
def test_nonfrontier_action_separates_groups(kind: ActionKind) -> None:
    _, frame = _frame()
    frame = replace(
        frame,
        actions=(
            *frame.actions[:2],
            replace(frame.actions[2], kind=kind),
            *frame.actions[3:],
        ),
    )
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None and result.stop_event == 4


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64, torch.float64])
def test_unsupported_storage_dtype_is_a_prefix_separator(dtype: torch.dtype) -> None:
    _, frame = _frame()
    frame = replace(
        frame, buffers=(*frame.buffers[:2], replace(frame.buffers[2], dtype=dtype))
    )
    result = groups.plan_frontier_group(frame, 2)
    assert result is not None and len(result.buffers) == 2


def test_singleton_missing_event_and_event_gap_do_not_form_a_group() -> None:
    _, singleton = _frame(1)
    assert groups.plan_frontier_group(singleton, 2) is None
    _, frame = _frame()
    assert groups.plan_frontier_group(frame, 1) is None
    assert groups.plan_frontier_group(frame, 99) is None
    gap = replace(
        frame, actions=tuple(action for action in frame.actions if action.event != 3)
    )
    assert groups.plan_frontier_group(gap, 2) is None


@pytest.mark.parametrize(
    "enabled,group_enabled", [(False, False), (False, True), (True, False)]
)
def test_disabled_requests_do_not_probe_or_activate(
    enabled: bool, group_enabled: bool
) -> None:
    plan, frame = _frame()
    vector = VectorStaging(enabled, group_enabled=group_enabled)
    with patch.object(
        groups, "plan_frontier_group", side_effect=AssertionError("must not probe")
    ):
        assert (
            groups.emit_frontier_group(
                Mock(),
                plan,
                frame,
                2,
                {},
                ChainedExecution(128),
                vector,
                BoundedProducerUnroll(1),
                scalar_targets=set(),
            )
            is None
        )
    assert not vector.activated and not vector.group_activated


@pytest.mark.parametrize("threads", [96, 128, 384])
@pytest.mark.parametrize("scalar", [False, True])
def test_group_publication_barriers_are_outside_active_copy_subgroup(
    threads: int, scalar: bool
) -> None:
    plan, frame = _frame(dtypes=(torch.bfloat16, torch.float16, torch.float32))
    execution = ChainedExecution(
        threads,
        thread="prep_thread",
        warp="prep_warp",
        sync="prep_barrier.arrive_and_wait()",
    )
    vector = VectorStaging(True, group_enabled=True)
    unroll = BoundedProducerUnroll(2)
    boundaries = {plan.dots[0]: "existing_c"}
    scalar_targets = (
        {buffer.name for buffer in frame.buffers} if scalar else {"image_2"}
    )
    with patch.object(
        groups, "emit_vector_group", return_value=["publish_original_values()"]
    ) as emit:
        result = groups.emit_frontier_group(
            Mock(),
            plan,
            frame,
            2,
            boundaries,
            execution,
            vector,
            unroll,
            scalar_targets=scalar_targets,
        )
    assert result is not None
    lines, stop = result
    assert stop == 5 and vector.activated and vector.group_activated
    assert lines[-1] == execution.sync and lines.count(execution.sync) == 1
    actual_threads = threads if scalar else 1 << (threads.bit_length() - 1)
    assert emit.call_args.kwargs["execution"] == replace(
        execution, threads=actual_threads
    )
    assert emit.call_args.kwargs["producer_unroll"] is unroll
    assert emit.call_args.args[2] is boundaries
    outputs = emit.call_args.args[3]
    assert [output.node for output in outputs] == [
        buffer.node for buffer in frame.buffers
    ]
    assert [output.vector_store for output in outputs] == [
        buffer.name not in scalar_targets for buffer in frame.buffers
    ]
    assert all(
        output.coordinates("row", "column") == ("row", "column")
        and output.final_value is None
        for output in outputs
    )
    tree = ast.parse("\n".join(lines))
    assert ast.unparse(tree.body[-1]) == execution.sync
    assert isinstance(tree.body[0], ast.If) is (threads != actual_threads)
    assert boundaries == {plan.dots[0]: "existing_c"}


def test_no_shared_expression_benefit_keeps_scalar_fallback_and_activation() -> None:
    plan, frame = _frame()
    vector = VectorStaging(True, group_enabled=True)
    unroll = BoundedProducerUnroll(2)
    with patch.object(groups, "emit_vector_group", return_value=None) as emit:
        assert (
            groups.emit_frontier_group(
                Mock(),
                plan,
                frame,
                2,
                {},
                ChainedExecution(128),
                vector,
                unroll,
                scalar_targets=set(),
            )
            is None
        )
    emit.assert_called_once()
    assert not vector.activated and not vector.group_activated
    assert not unroll.activated and not unroll.eliminated


@pytest.mark.parametrize("grouped", [False, True])
def test_prepare_callsite_skips_only_published_group_and_retains_ordinary_fallback(
    grouped: bool,
) -> None:
    plan, frame = _frame()
    frame = replace(
        frame, buffers=(*frame.buffers[:2], replace(frame.buffers[2], shape=(4, 16)))
    )
    recurrence = RecurrenceWorkspace(
        frame.cut, SharedMemoryLayoutPlan((), 0), (), (), 0, 0, 0, 0
    )
    pipeline = PreparationPipeline(frame, recurrence, None, 2, 128, 128, 128)
    cg = Mock()
    cg.device_function.config = helion.Config()
    vector = VectorStaging(True, group_enabled=grouped)
    execution = ChainedExecution(128, sync="role_barrier.arrive_and_wait()")

    def ordinary(cg, plan, buffer, boundaries, execution, **kwargs):
        return [f"ordinary_{buffer.name}()", execution.sync]

    with (
        patch.object(groups, "emit_vector_group", return_value=["group_publish()"]),
        patch.object(
            chained_prepared_values, "emit_prepared_value", side_effect=ordinary
        ) as emit,
    ):
        lines = _prepare(
            cg,
            plan,
            pipeline,
            execution,
            vector,
            BoundedProducerUnroll(1),
            ScratchLayouts(),
        )
    assert [call.args[2].name for call in emit.call_args_list] == (
        ["image_2"] if grouped else ["image_0", "image_1", "image_2"]
    )
    assert lines.count(execution.sync) == (3 if grouped else 4)
    assert lines[-1] == execution.sync
    assert ("group_publish()" in lines) is grouped
