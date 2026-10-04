from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_preparation_cut import PreparationCut
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_frame import PreparationFrame
from helion._compiler.cute.chained_preparation_frame import PreparationStage
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_row_collectives import plan_row_collective_group
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from helion._compiler.cute.chained_preparation_frame import ActionKind


def _case(rows=16, columns=64, count=2, members=2, dtype=torch.bfloat16, keepdim=False):
    graph = Graph()
    raw = tuple(
        _input(graph, f"input_{i}", (rows, columns), torch.float32)
        for i in range(count)
    )
    sums = []
    operands = []
    for source in raw:
        value = _call(
            graph,
            torch.ops.aten.mul.Tensor,
            (source, source),
            (rows, columns),
            torch.float32,
        )
        reduced = _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (value, [1], keepdim),
            (rows, 1) if keepdim else (rows,),
            torch.float32,
        )
        sums.append(reduced)
        expanded = (
            reduced
            if keepdim
            else _call(
                graph,
                torch.ops.aten.unsqueeze.default,
                (reduced, 1),
                (rows, 1),
                torch.float32,
            )
        )
        combined = _call(
            graph,
            torch.ops.aten.add.Tensor,
            (source, expanded),
            (rows, columns),
            torch.float32,
        )
        operands.append(_convert(graph, combined, dtype))
    dots = tuple(
        _dot(
            graph,
            operands[0],
            _call(
                graph,
                torch.ops.aten.t.default,
                (operands[i % count],),
                (columns, rows),
                dtype,
            ),
        )
        for i in range(members)
    )
    plan = _plan(graph, (*dots, *sums))
    assert plan.region is not None
    group = ContractionGroup(
        tuple(range(members)), (StageGeometry((rows, rows, columns), False),) * members
    )
    plan = replace(
        plan,
        strategy="tcgen05_tmem",
        warp_mma_stages=frozenset({0}),
        contraction_groups=(group,),
    )
    assert plan.region is not None
    cut = PreparationCut(
        plan.region,
        (),
        (),
        raw,
        tuple(node for node in plan.region.nodes if node not in raw),
        (),
        (),
        "synthetic",
    )
    fill_event = count + 1
    stop = fill_event + 1
    end = stop + 2
    regions = []
    buffers = []
    offset = 0

    def allocate(name, kind, node, value_dtype, shape, begin, until):
        nonlocal offset
        size = (
            (torch.empty(shape, dtype=value_dtype).numel() * value_dtype.itemsize + 127)
            // 128
            * 128
        )
        regions.append(SharedBufferRegion(name, offset, size, begin, until, 128))
        buffers.append(PreparationBuffer(name, kind, node, value_dtype, shape))
        offset += size
        return regions[-1]

    for index, source in enumerate(raw):
        allocate(
            f"raw_{index}", "leaf", source, torch.float32, (rows, columns), 0, stop
        )
    for index, node in enumerate(sums):
        allocate(
            f"sum_{index}",
            "collective",
            node,
            torch.float32,
            (rows, 1) if keepdim else (rows,),
            index + 1,
            end,
        )
    a = allocate("operand_a", "a", None, dtype, (rows, columns), fill_event, stop + 1)
    b = allocate(
        "operand_b", "b", None, dtype, (members * rows, columns), fill_event, stop + 1
    )
    for index, node in enumerate(dots):
        allocate(f"result_{index}", "c", node, torch.float32, (rows, rows), stop, end)
    actions = (
        PreparationAction(
            "leaf", 0, raw, (), None, (), tuple(f"raw_{i}" for i in range(count))
        ),
        *(
            PreparationAction(
                "collective", i + 1, (node,), (), 0, (f"raw_{i}",), (f"sum_{i}",)
            )
            for i, node in enumerate(sums)
        ),
        PreparationAction(
            "fill",
            fill_event,
            dots,
            group.stages,
            0,
            tuple(f"sum_{i}" for i in range(count)),
            (a.name, b.name),
        ),
        PreparationAction(
            "mma",
            stop,
            dots,
            group.stages,
            0,
            (a.name, b.name),
            tuple(f"result_{i}" for i in range(members)),
        ),
        PreparationAction(
            "ready",
            stop + 1,
            dots,
            (),
            None,
            tuple(f"result_{i}" for i in range(members)),
            (),
        ),
    )
    frame = PreparationFrame(
        cut,
        SharedMemoryLayoutPlan(tuple(regions), offset),
        tuple(buffers),
        actions,
        (PreparationStage(group, (rows, members * rows, columns), a, b),),
        (),
        offset,
    )
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    return plan, frame, shapes


def _region(frame, name, **changes):
    regions = tuple(
        replace(region, **changes) if region.name == name else region
        for region in frame.layout.regions
    )
    by_name = {region.name: region for region in regions}
    return replace(
        frame,
        layout=replace(frame.layout, regions=regions),
        stages=tuple(
            replace(stage, a=by_name[stage.a.name], b=by_name[stage.b.name])
            for stage in frame.stages
        ),
    )


@pytest.mark.parametrize(
    "rows,columns,count,members",
    [(16, 32, 1, 1), (32, 64, 2, 2), (16, 96, 3, 3), (64, 128, 2, 1)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("keepdim", [False, True])
def test_original_typed_nodes_and_entire_frame_remain_unchanged(
    rows, columns, count, members, dtype, keepdim
):
    plan, frame, shapes = _case(rows, columns, count, members, dtype, keepdim)
    assert plan.region is not None
    before = repr(frame)
    graph = str(plan.region.graph)
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        result = plan_row_collective_group(plan, frame, 1, shapes)
    assert result is not None
    assert (result.first_event, result.stop_event, result.shape) == (
        1,
        count + 2,
        (rows, columns),
    )
    assert result.stage is frame.stages[0]
    assert tuple(item.node for item in result.collectives) == tuple(
        action.nodes[0] for action in frame.actions[1 : count + 1]
    )
    assert tuple(item.name for item in result.buffers) == tuple(
        f"sum_{i}" for i in range(count)
    )
    assert tuple(item.original.name for item in result.affected_regions) == (
        *tuple(f"sum_{i}" for i in range(count)),
        "operand_a",
        "operand_b",
    )
    assert all(
        item.required.live_from == 1
        and item.required.live_until == item.original.live_until
        for item in result.affected_regions
    )
    assert all(
        item.required.live_until >= result.stop_event for item in result.read_regions
    )
    assert repr(frame) == before and str(plan.region.graph) == graph
    with pytest.raises(FrozenInstanceError):
        result.stop_event = 0  # type: ignore[misc]


@pytest.mark.parametrize("event", [-1, 0, 3, 4, 5, 6, True, 1.0, "1"])
def test_only_an_exact_collective_start_is_admitted(event):
    plan, frame, shapes = _case()
    assert plan_row_collective_group(plan, frame, event, shapes) is None


@pytest.mark.parametrize("kind", ["leaf", "cache", "mma", "frontier", "ready"])
def test_intervening_action_is_not_skipped(kind: ActionKind):
    plan, frame, shapes = _case()
    frame = replace(
        frame,
        actions=tuple(
            replace(action, kind=kind) if action.event == 2 else action
            for action in frame.actions
        ),
    )
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


@pytest.mark.parametrize(
    "change",
    [
        "axis",
        "scan",
        "source_dtype",
        "result_dtype",
        "source_shape",
        "result_shape",
        "buffer_dtype",
        "buffer_shape",
    ],
)
def test_collective_semantics_must_match_original_typed_rows(change):
    plan, frame, shapes = _case()
    node = frame.actions[1].nodes[0]
    source = node.args[0]
    assert isinstance(source, Node)
    if change == "axis":
        node.args = (source, [0], False)
    elif change == "scan":
        node.target = torch.ops.aten.cumsum.default
    elif change == "source_dtype":
        source.meta["val"] = torch.empty((16, 64), dtype=torch.bfloat16)
    elif change == "result_dtype":
        node.meta["val"] = torch.empty((16,), dtype=torch.float16)
    elif change == "source_shape":
        shapes[source] = (16, 32)
    elif change == "result_shape":
        shapes[node] = (1, 16)
    else:
        frame = replace(
            frame,
            buffers=tuple(
                (
                    replace(buffer, dtype=torch.float16)
                    if change == "buffer_dtype"
                    else replace(buffer, shape=(1, 16))
                )
                if buffer.node is node
                else buffer
                for buffer in frame.buffers
            ),
        )
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


def test_dependent_reduction_cannot_consume_unpublished_other_rows():
    plan, frame, shapes = _case()
    first, second = (frame.actions[i].nodes[0] for i in (1, 2))
    source = second.args[0]
    assert isinstance(source, Node)
    source.args = (source.args[0], first)
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


@pytest.mark.parametrize(
    "change",
    [
        "no_warp",
        "missing_stage",
        "partial_group",
        "duplicate_member",
        "wrong_shape",
        "short_b",
        "short_lifetime",
        "wrong_mma",
    ],
)
def test_complete_warp_fill_and_original_mma_cut_are_required(change):
    plan, frame, shapes = _case()
    stage = frame.stages[0]
    if change == "no_warp":
        plan = replace(plan, warp_mma_stages=frozenset())
    elif change == "missing_stage":
        frame = replace(frame, stages=())
    elif change == "partial_group":
        frame = replace(
            frame,
            actions=tuple(
                replace(action, nodes=action.nodes[:1])
                if action.kind == "fill"
                else action
                for action in frame.actions
            ),
        )
    elif change == "duplicate_member":
        frame = replace(
            frame, stages=(replace(stage, group=replace(stage.group, stages=(0, 0))),)
        )
    elif change == "wrong_shape":
        frame = replace(frame, stages=(replace(stage, shape=(128, 32, 64)),))
    elif change == "short_b":
        frame = _region(frame, stage.b.name, byte_size=128)
    elif change == "short_lifetime":
        frame = _region(frame, stage.a.name, live_until=4)
    else:
        frame = replace(
            frame,
            actions=tuple(
                replace(action, kind="frontier") if action.kind == "mma" else action
                for action in frame.actions
            ),
        )
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


def _scratch_alias(frame, name, until):
    target = frame.layout.region(name)
    region = SharedBufferRegion(
        "old_scratch", target.byte_offset, target.byte_size, 0, until, 128
    )
    buffer = PreparationBuffer(
        "old_scratch", "cache", None, torch.float32, (target.byte_size // 4,)
    )
    return replace(
        frame,
        layout=replace(frame.layout, regions=(*frame.layout.regions, region)),
        buffers=(*frame.buffers, buffer),
    )


@pytest.mark.parametrize("name", ["sum_1", "operand_a", "operand_b"])
def test_every_advanced_write_rejects_a_previously_legal_alias(name):
    plan, frame, shapes = _case()
    frame = _scratch_alias(frame, name, frame.layout.region(name).live_from)
    from helion._compiler.cute.chained_prepared_groups import _valid_frame

    assert _valid_frame(frame)
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


def test_exact_half_open_release_before_first_event_is_safe():
    plan, frame, shapes = _case()
    frame = _scratch_alias(frame, "sum_1", 1)
    assert plan_row_collective_group(plan, frame, 1, shapes) is not None


def test_read_lifetime_extension_is_explicit_and_can_reject_old_storage_reuse():
    plan, frame, shapes = _case()
    frame = _region(frame, "raw_0", live_until=2)
    result = plan_row_collective_group(plan, frame, 1, shapes)
    assert result is not None
    witness = next(
        item for item in result.read_regions if item.original.name == "raw_0"
    )
    assert (witness.original.live_until, witness.required.live_until) == (2, 4)
    raw = frame.layout.region("raw_0")
    frame = _region(frame, "sum_1", byte_offset=raw.byte_offset)
    from helion._compiler.cute.chained_prepared_groups import _valid_frame

    assert _valid_frame(frame)
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


@pytest.mark.parametrize(
    "change",
    [
        "capacity",
        "unaligned",
        "duplicate_region",
        "events",
        "foreign_graph",
        "changed_graph",
    ],
)
def test_stale_or_invalid_frame_is_not_an_authority(change):
    plan, frame, shapes = _case()
    if change == "capacity":
        frame = replace(frame, layout=replace(frame.layout, allocated_bytes=128))
    elif change == "unaligned":
        frame = _region(frame, "sum_1", byte_offset=3)
    elif change == "duplicate_region":
        frame = replace(
            frame,
            layout=replace(
                frame.layout, regions=(*frame.layout.regions, frame.layout.regions[0])
            ),
        )
    elif change == "events":
        frame = replace(
            frame,
            actions=tuple(
                replace(action, event=17) if action.event == 2 else action
                for action in frame.actions
            ),
        )
    elif change == "foreign_graph":
        other, _, _ = _case()
        plan = replace(plan, region=other.region)
    else:
        assert plan.region is not None
        plan.region.graph.placeholder("late_input")
    assert plan_row_collective_group(plan, frame, 1, shapes) is None


def test_actual_admitted_frame_preserves_all_placements_and_norm_publications():
    previous = sys.modules.get("benchmarks")
    namespace = ModuleType("benchmarks")
    namespace.__path__ = [str(Path(__file__).resolve().parents[1] / "benchmarks")]
    sys.modules["benchmarks"] = namespace
    try:
        kernel, args = _kda_fixture()
    finally:
        if previous is None:
            sys.modules.pop("benchmarks", None)
        else:
            sys.modules["benchmarks"] = previous
    initialized = torch.cuda.is_initialized()
    plan, cut, shapes = _capture(
        kernel,
        args,
        helion.Config(
            block_sizes=[128],
            num_warps=16,
            num_stages=2,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_group_contractions=True,
            cute_chained_scratch_layout="xor",
            cute_chained_pointwise_vectorize=True,
            cute_chained_scan_schedule="warp",
            cute_chained_pointwise_cache_bytes=4096,
            cute_chained_pointwise_unroll=8,
            cute_chained_warp_mma_rows=32,
        ),
    )
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None
    before = repr(frame)
    candidates = tuple(
        result
        for action in frame.actions
        if (result := plan_row_collective_group(plan, frame, action.event, shapes))
        is not None
    )
    assert candidates
    result = next(item for item in candidates if len(item.collectives) == 2)
    assert result.shape == (32, 128)
    assert len(result.affected_regions) == 4
    assert {item.original.name for item in result.affected_regions} == {
        *(buffer.name for buffer in result.buffers),
        result.stage.a.name,
        result.stage.b.name,
    }
    assert result.affected_regions[1].original.live_from == result.first_event + 1
    assert result.affected_regions[1].required.live_from == result.first_event
    assert frame.layout.allocated_bytes == 49920
    assert repr(frame) == before
    assert torch.cuda.is_initialized() == initialized
