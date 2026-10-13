from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_loop import ChainedLoopPlan
from helion._compiler.cute.chained_pointwise_residency import PointwiseCacheEntry
from helion._compiler.cute.chained_pointwise_residency import PointwiseCachePlan
from helion._compiler.cute.chained_preparation_cut import PreparationCut
from helion._compiler.cute.chained_preparation_cut import PreparationImage
from helion._compiler.cute.chained_preparation_cut import plan_preparation_cut
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.contraction_region import ContractionCarry
from helion._compiler.cute.contraction_region import _domain
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import RootGraphInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_preparation_frame import PreparationFrame
    from helion._compiler.device_ir import GraphInfo
    from helion.runtime.kernel import Kernel


def _safe(frame: PreparationFrame) -> None:
    for left, right in combinations(frame.layout.regions, 2):
        assert not (left.overlaps_lifetime(right) and left.overlaps_storage(right))
    for action in frame.actions:
        for name in (*action.reads, *action.writes):
            region = frame.layout.region(name)
            assert region.live_from <= action.event < region.live_until
            assert region.byte_offset % 128 == 0
            assert region.byte_end <= frame.layout.allocated_bytes
        assert all(
            frame.layout.region(name).live_from < action.event for name in action.reads
        )
    end = frame.actions[-1].publication_event
    assert frame.actions[-1].kind == "ready"
    assert all(
        frame.layout.region(buffer.name).live_until == end
        for buffer in frame.buffers
        if buffer.kind == "frontier"
    )
    for stage in frame.stages:
        action = next(
            action
            for action in frame.actions
            if action.kind == "mma" and action.stages == stage.group.stages
        )
        assert stage.a.live_until == stage.b.live_until == action.publication_event


def _capture(
    kernel: Kernel, args: tuple, config: helion.Config
) -> tuple[ChainedMatmulPlan, PreparationCut, dict[Node, tuple[int, ...]]]:
    original = chain.plan_chained_matmul
    results = []

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None and plan.loop is not None:
            cut = plan_preparation_cut(graphs)
            assert cut is not None
            shapes = {
                node: chain._shape(node)
                for node in cut.region.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            }
            results.append((plan, cut, shapes))
        return plan

    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
            patch.object(chain, "plan_chained_matmul", observe),
        ):
            bound.to_code(config)
    assert results
    return results[0]


def test_actual_kda_frame_matches_interval_prototype_without_device_context() -> None:
    kernel, args = _kda_fixture()
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
    # The actual planner runs outside both CompileEnvironment and DeviceFunction.
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None
    assert frame.layout.allocated_bytes == frame.peak_live_bytes == 49920
    assert len(frame.stages) == 7
    assert len(frame.frontier_order) == 11
    assert (
        sum(
            frame.layout.region(buffer.name).byte_size
            for buffer in frame.buffers
            if buffer.kind == "frontier"
        )
        == 46080
    )
    assert {buffer.dtype for buffer in frame.buffers if buffer.kind == "frontier"} >= {
        torch.bfloat16,
        torch.float32,
        torch.int32,
        torch.bool,
    }
    assert (
        frame.frontier_order[0] is cut.images[3].node
    )  # Exact inverse node, no expression rewrite.
    _safe(frame)


@pytest.mark.parametrize("recurrent_first", [False, True])
def test_actual_typed_loop_supports_late_independent_preparation(
    recurrent_first: bool,
) -> None:
    plan, cut, shapes = _capture(
        _typed_sequence,
        _inputs(recurrent_first, typed=True),
        helion.Config(
            block_sizes=[],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_warp_mma_rows=32,
        ),
    )
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None
    expected = ((1,), (2,)) if recurrent_first else ((0,), (1,))
    assert tuple(stage.group.stages for stage in frame.stages) == expected
    assert {buffer.dtype for buffer in frame.buffers if buffer.kind == "frontier"} >= {
        torch.bfloat16,
        torch.float16,
        torch.float32,
    }
    _safe(frame)


def _synthetic(
    kind: str = "cache",
) -> tuple[ChainedMatmulPlan, PreparationCut, dict[Node, tuple[int, ...]]]:
    """Small explicit graph contracts isolate storage planning from admission."""
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    state = _input(graph, "carry", (16, 16), torch.float32)
    first = _dot(graph, left, right)
    cached = _convert(graph, first, torch.bfloat16)
    reduced = _call(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (first, [0], True),
        (1, 16),
        torch.float32,
    )
    _dot(graph, cached, right)
    frontier = {"dot": first, "collective": reduced, "cache": cached}[kind]
    accumulator = _call(
        graph, torch.ops.aten.add.Tensor, (state, frontier), (16, 16), torch.float32
    )
    final = _dot(graph, left, right, accumulator)
    plan = _plan(graph, (final,))
    assert plan.region is not None
    carry = ContractionCarry(2, 0, state, final)
    region = replace(plan.region, carries=(carry,))
    loop = ChainedLoopPlan(
        RootGraphInfo(0, graph),
        ForLoopGraphInfo(1, graph, [left, right, state], [0]),
        region,
        plan.store,
        (left, right, state),
        0,
        3,
        (),
        (),
    )
    entry = PointwiseCacheEntry(
        cached, "cache", (16, 16), torch.bfloat16, 1, 1, 512, 512, (first,), 2, 10
    )
    plan = replace(
        plan,
        region=region,
        loop=loop,
        strategy="tcgen05_tmem",
        warp_mma_stages=frozenset({0, 1}),
        pointwise_cache=PointwiseCachePlan((entry,)),
    )
    recurrent = {state, accumulator, final, plan.store}
    shared = {left, right}
    prep = tuple(node for node in region.nodes if node not in recurrent | shared)
    cut = PreparationCut(
        region,
        (carry,),
        (),
        (left, right),
        prep,
        tuple(node for node in region.nodes if node in recurrent),
        (
            PreparationImage(
                frontier,
                frontier.meta["val"].dtype,
                _domain(frontier.meta["val"]),
                (accumulator,),
            ),
        ),
        loop.storage_key,
    )
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    return plan, cut, shapes


@pytest.mark.parametrize("kind", ["dot", "collective", "cache"])
def test_frontier_already_materialized_reads_own_boundary(kind: str) -> None:
    plan, cut, shapes = _synthetic(kind)
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None
    image = cut.images[0].node
    source = next(
        buffer
        for buffer in frame.buffers
        if buffer.node is image and buffer.kind != "frontier"
    )
    action = next(action for action in frame.actions if action.kind == "frontier")
    assert action.reads == (source.name,)
    assert frame.layout.region(source.name).live_until == action.publication_event
    assert not frame.layout.region(source.name).overlaps_storage(
        frame.layout.region(action.writes[0])
    )
    _safe(frame)


def test_cache_first_fill_shortens_only_materialized_dependencies() -> None:
    plan, cut, shapes = _synthetic()
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None
    cache = next(action for action in frame.actions if action.kind == "cache")
    source = next(
        buffer
        for buffer in frame.buffers
        if buffer.node is plan.dots[0] and buffer.kind == "c"
    )
    assert cache.reads == (source.name,)
    assert frame.layout.region(source.name).live_until == cache.publication_event
    fill = next(
        action
        for action in frame.actions
        if action.kind == "fill" and action.stages == (1,)
    )
    assert fill.reads == cache.writes


def test_action_source_stages_and_original_binding_names() -> None:
    plan, cut, shapes = _synthetic()
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None and plan.pointwise_cache is not None
    collective = next(action for action in frame.actions if action.kind == "collective")
    cache = next(action for action in frame.actions if action.kind == "cache")
    assert collective.source_stage == cache.source_stage == 1
    assert collective.writes == ("chain_collective_0",)
    assert cache.writes == (plan.pointwise_cache.entries[0].name,)
    assert all(
        action.source_stage == action.stages[0]
        for action in frame.actions
        if action.kind in ("fill", "mma")
    )
    assert all(
        action.source_stage is None
        for action in frame.actions
        if action.kind in ("frontier", "ready")
    )
    assert next(
        action for action in frame.actions if action.kind == "frontier"
    ).writes == ("chain_prepared_0",)


@pytest.mark.parametrize(
    "dependency", ["future_cache", "recurrent_mask", "extra_index"]
)
def test_hidden_or_unpublished_dependency_is_not_dropped(dependency: str) -> None:
    plan, cut, shapes = _synthetic()
    assert plan.pointwise_cache is not None
    node = plan.dots[0]
    if dependency == "future_cache":
        node.args = (plan.pointwise_cache.entries[0].node, *node.args[1:])
    else:
        # all_input_nodes includes kwargs and dependency-only index/mask edges.
        node.kwargs = {
            "extra_mask"
            if dependency == "recurrent_mask"
            else "_extra_deps": cut.carries[0].input
        }
    assert plan_preparation_frame(plan, cut, shapes) is None


@pytest.mark.parametrize(
    "bad", ["missing", "rank", "zero", "bool", "static_mismatch", "dtype"]
)
def test_explicit_shape_and_dtype_contract_fails_closed(bad: str) -> None:
    plan, cut, shapes = _synthetic()
    node = cut.images[0].node
    if bad == "missing":
        shapes.pop(node)
    elif bad == "dtype":
        node.meta["val"] = torch.empty((16, 16), dtype=torch.float64)
    else:
        shapes[node] = {
            "rank": (16,),
            "zero": (0, 16),
            "bool": (True, 16),
            "static_mismatch": (32, 16),
        }[bad]
    assert plan_preparation_frame(plan, cut, shapes) is None


@pytest.mark.parametrize(
    "bad",
    [
        "nonloop",
        "strategy",
        "unselected",
        "cross_group",
        "missing_stage",
        "cache_recurrent",
        "collective_recurrent",
        "cache_future",
        "cache_shape",
        "duplicate_frontier",
    ],
)
def test_unsupported_roles_and_schedules_fail_closed(bad: str) -> None:
    plan, cut, shapes = _synthetic()
    assert plan.pointwise_cache is not None
    entry = plan.pointwise_cache.entries[0]
    if bad == "nonloop":
        plan = replace(plan, loop=None)
    elif bad == "strategy":
        plan = replace(plan, strategy="coalesced")
    elif bad == "unselected":
        plan = replace(plan, warp_mma_stages=frozenset({0}))
    elif bad in ("cross_group", "missing_stage"):
        geometry = StageGeometry((16, 16, 16), False)
        groups = (
            ContractionGroup((0,), (geometry,)),
            ContractionGroup((1, 2), (geometry, geometry)),
        )
        plan = replace(
            plan, contraction_groups=groups if bad == "cross_group" else groups[:1]
        )
    elif bad in ("cache_recurrent", "collective_recurrent"):
        node = entry.node if bad == "cache_recurrent" else cut.region.reductions[0]
        cut = replace(
            cut,
            preparation=tuple(n for n in cut.preparation if n is not node),
            recurrence=(*cut.recurrence, node),
        )
    elif bad == "cache_future":
        plan = replace(
            plan, pointwise_cache=PointwiseCachePlan((replace(entry, first_stage=2),))
        )
    elif bad == "cache_shape":
        plan = replace(
            plan, pointwise_cache=PointwiseCachePlan((replace(entry, shape=(32, 16)),))
        )
    else:
        cut = replace(cut, images=(*cut.images, cut.images[0]))
    assert plan_preparation_frame(plan, cut, shapes) is None


def test_graph_revision_identity_and_immutability() -> None:
    plan, cut, shapes = _synthetic()
    _, other, _ = _synthetic()
    assert plan_preparation_frame(plan, other, shapes) is None
    before = tuple((node, node.args, dict(node.kwargs)) for node in cut.region.nodes)
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None
    assert before == tuple(
        (node, node.args, dict(node.kwargs)) for node in cut.region.nodes
    )
    with pytest.raises(FrozenInstanceError):
        frame.peak_live_bytes = 0  # pyrefly: ignore [read-only]
    cut.region.graph.placeholder("new_revision")
    assert plan_preparation_frame(plan, cut, shapes) is None
