from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_loop_tmem_bridges import _coordinate_path
from helion._compiler.cute.chained_loop_tmem_bridges import plan_loop_tmem_bridges
from helion._compiler.cute.chained_pointwise_residency import PointwiseCacheEntry
from helion._compiler.cute.chained_pointwise_residency import PointwiseCachePlan
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion.language import memory_ops


def _candidate(
    width=32,
    *,
    transpose=False,
    dtype=torch.bfloat16,
    mode="cast",
    grouped=False,
    gap=False,
):
    graph = Graph()
    shape = (width, 128) if transpose else (128, width)
    left = _input(graph, "left", (shape[0], 16), dtype)
    right = _input(graph, "right", (16, shape[1]), dtype)
    source = _dot(graph, left, right)
    image = source
    if mode == "arithmetic":
        bias = _input(graph, "bias", shape, torch.float32)
        image = _call(
            graph, torch.ops.aten.sub.Tensor, (image, bias), shape, torch.float32
        )
        image = _call(
            graph, torch.ops.aten.mul.Scalar, (image, 0.5), shape, torch.float32
        )
    if mode == "rounding":
        image = _convert(graph, _convert(graph, image, torch.float16), torch.float32)
    operand = _convert(graph, image, dtype)
    destination_transpose = transpose
    if mode in ("t", "permute", "transpose", "wrong_transpose"):
        target, args = {
            "t": (torch.ops.aten.t.default, (operand,)),
            "permute": (torch.ops.aten.permute.default, (operand, [1, 0])),
            "transpose": (torch.ops.aten.transpose.int, (operand, -1, -2)),
            "wrong_transpose": (torch.ops.aten.t.default, (operand,)),
        }[mode]
        operand = _call(graph, target, args, shape[::-1], dtype)
        if mode != "wrong_transpose":
            destination_transpose = not transpose
    elif mode == "reshape":
        operand = _call(
            graph, torch.ops.aten.view.default, (operand, shape), shape, dtype
        )
    elif mode == "gather":
        index = _input(graph, "index", (shape[1],), torch.int64)
        operand = _call(
            graph,
            torch.ops.aten.index_select.default,
            (operand, 1, index),
            shape,
            dtype,
        )
    elif mode == "random":
        operand = _call(
            graph, torch.ops.aten.dropout.default, (operand, 0.5, True), shape, dtype
        )
    if gap:
        _dot(graph, left, right)
    output_shape = (16, 128) if destination_transpose else (128, 16)
    state = _input(graph, "state", output_shape, torch.float32)
    other_shape = (16, width) if destination_transpose else (width, 16)
    other = _input(graph, "other", other_shape, dtype)
    seed = source if mode == "accumulator" else state
    destination = (
        _dot(graph, other, operand, seed)
        if destination_transpose
        else _dot(graph, operand, other, seed)
    )
    carries, outputs = (state,), (source if mode == "carry" else destination,)
    geometries = [
        StageGeometry((output_shape[0], output_shape[1], width), destination_transpose)
    ]
    if grouped:
        second_operand = _call(
            graph,
            torch.ops.aten.t.default,
            (operand,),
            tuple(operand.meta["val"].shape[::-1]),
            dtype,
        )
        second_shape = (128, 32) if destination_transpose else (32, 128)
        second_state = _input(graph, "second_state", second_shape, torch.float32)
        second_other = _input(
            graph,
            "second_other",
            (width, 32) if destination_transpose else (32, width),
            dtype,
        )
        second = (
            _dot(graph, second_operand, second_other, second_state)
            if destination_transpose
            else _dot(graph, second_other, second_operand, second_state)
        )
        carries += (second_state,)
        outputs += (second,)
        geometries.append(
            StageGeometry(
                (second_shape[0], second_shape[1], width), not destination_transpose
            )
        )
    if mode == "extra_user":
        _call(graph, torch.ops.aten.neg.default, (source,), shape, torch.float32)
    elif mode == "collective":
        _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (source, [1], True),
            (shape[0], 1),
            torch.float32,
        )
    elif mode == "store":
        sink = _input(graph, "sink", shape, torch.float32)
        graph.call_function(
            memory_ops.store, (sink, [slice(None), slice(None)], source, None)
        )
    plan = _loop_plan(graph, carries, outputs)
    source_geometry = StageGeometry((shape[0], shape[1], 16), transpose)
    groups = (ContractionGroup((0,), (source_geometry,)),)
    if gap:
        groups += (ContractionGroup((1,), (source_geometry,)),)
    first = 2 if gap else 1
    groups += (
        ContractionGroup(
            tuple(range(first, first + len(geometries))), tuple(geometries)
        ),
    )
    return replace(plan, contraction_groups=groups), groups


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("mode", ["cast", "arithmetic", "rounding"])
def test_candidates_keep_exact_expression_casts_and_transposed_geometry(
    dtype, transpose, mode
):
    plan, groups = _candidate(dtype=dtype, transpose=transpose, mode=mode)
    assert plan.region is not None
    before = tuple((node, node.args, dict(node.kwargs)) for node in plan.region.nodes)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = plan_loop_tmem_bridges(plan, groups)
    assert result is not None and len(result) == 1
    candidate = result[0]
    assert candidate.region is plan.region and candidate.source is plan.dots[0]
    assert candidate.destination_group is groups[-1]
    assert candidate.source_geometry.transpose is transpose
    assert candidate.physical_shape == (128, 32) and candidate.dtype == dtype
    assert (candidate.publication_event, candidate.issue_event) == (1, 2)
    assert candidate.source in candidate.expression_nodes
    assert all(operand in candidate.expression_nodes for operand in candidate.operands)
    if mode == "rounding":
        assert (
            sum(
                node.target is torch.ops.prims.convert_element_type.default
                for node in candidate.expression_nodes
            )
            == 3
        )
    assert before == tuple(
        (node, node.args, dict(node.kwargs)) for node in plan.region.nodes
    )
    with pytest.raises(FrozenInstanceError):
        candidate.source_stage = 1  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("mode", ["t", "permute", "transpose"])
def test_exact_axis_exchange_is_composed_with_physical_a_coordinates(transpose, mode):
    plan, groups = _candidate(transpose=transpose, mode=mode)
    result = plan_loop_tmem_bridges(plan, groups)
    assert result is not None and len(result) == 1
    assert result[0].source_geometry.transpose != groups[-1].geometries[0].transpose


@pytest.mark.parametrize("transpose", [False, True])
def test_every_group_member_shares_a_with_exact_transpose_and_nonadjacent_lifetime(
    transpose,
):
    plan, groups = _candidate(transpose=transpose, grouped=True, gap=True)
    result = plan_loop_tmem_bridges(plan, groups)
    assert result is not None and len(result) == 1
    candidate = result[0]
    assert candidate.source_stage == 0 and candidate.destination_group.stages == (2, 3)
    assert (
        len(candidate.operands) == 2
        and candidate.operands[0] is not candidate.operands[1]
    )
    assert (candidate.publication_event, candidate.issue_event) == (1, 4)


@pytest.mark.parametrize(
    "mode", ["reshape", "gather", "random", "extra_user", "wrong_transpose"]
)
def test_unknown_or_cross_coordinate_paths_and_extra_users_are_not_candidates(mode):
    plan, groups = _candidate(128 if mode == "wrong_transpose" else 32, mode=mode)
    assert plan_loop_tmem_bridges(plan, groups) == ()


@pytest.mark.parametrize("mode", ["accumulator", "carry", "collective", "store"])
def test_explicit_accumulator_and_carried_result_are_not_operand_only_bridges(mode):
    plan, groups = _candidate(16, mode=mode)
    assert plan_loop_tmem_bridges(plan, groups) == ()


@pytest.mark.parametrize("side", ["source", "destination"])
def test_partial_m128_geometry_is_not_a_complete_operand(side):
    plan, groups = _candidate()
    index = 0 if side == "source" else 1
    group = groups[index]
    changed = replace(group, geometries=(replace(group.geometries[0], transpose=True),))
    groups = tuple(changed if old is group else old for old in groups)
    plan = replace(plan, contraction_groups=groups)
    assert plan_loop_tmem_bridges(plan, groups) == ()


def test_grouped_source_result_is_not_singleton_transport():
    plan, groups = _candidate(gap=True)
    grouped = ContractionGroup(
        (0, 1), (groups[0].geometries[0], groups[1].geometries[0])
    )
    groups = (grouped, groups[-1])
    plan = replace(plan, contraction_groups=groups)
    assert plan_loop_tmem_bridges(plan, groups) == ()


def test_stale_shapes_and_geometries_cannot_override_static_fx_domains():
    plan, groups = _candidate()
    changed = replace(groups[0], geometries=(StageGeometry((128, 64, 16), False),))
    groups = (changed, groups[-1])
    plan = replace(
        plan, shapes=((128, 64, 16), plan.shapes[1]), contraction_groups=groups
    )
    assert plan_loop_tmem_bridges(plan, groups) is None


def test_loop_and_plan_must_retain_the_same_contraction_facts():
    plan, groups = _candidate()
    assert plan.loop is not None
    different = replace(plan.loop.region.contractions[0], result_dtype=torch.float16)
    loop_region = replace(
        plan.loop.region, contractions=(different, *plan.loop.region.contractions[1:])
    )
    plan = replace(plan, loop=replace(plan.loop, region=loop_region))
    assert plan_loop_tmem_bridges(plan, groups) is None


@pytest.mark.parametrize("bad", ["source", "intermediate", "dependency"])
def test_pointwise_cache_boundaries_remain_materialized(bad):
    plan, groups = _candidate(mode="arithmetic")
    result = plan_loop_tmem_bridges(plan, groups)
    assert result is not None and len(result) == 1
    candidate = result[0]
    node = candidate.source if bad == "source" else candidate.expression_nodes[1]
    entry = PointwiseCacheEntry(
        node,
        "cached",
        (128, 32),
        torch.float32,
        1,
        1,
        16384,
        16384,
        (candidate.source,) if bad == "dependency" else (),
        2,
        1,
    )
    plan = replace(plan, pointwise_cache=PointwiseCachePlan((entry,)))
    assert plan_loop_tmem_bridges(plan, groups) == ()


@pytest.mark.parametrize(
    "bad",
    [
        "no_loop",
        "graph_revision",
        "kwargs",
        "dtype",
        "shape",
        "missing_issue",
        "mixed_role",
        "direct",
        "group_geometry",
    ],
)
def test_invalid_retained_graph_or_issue_sequence_fails_closed(bad):
    plan, groups = _candidate(grouped=bad == "mixed_role")
    assert plan.region is not None
    if bad == "no_loop":
        plan = replace(plan, loop=None)
    elif bad == "graph_revision":
        plan.region.graph.placeholder("new_revision")
    elif bad == "kwargs":
        plan.dots[0].kwargs = {"unexpected": 1}
    elif bad == "dtype":
        plan.dots[0].meta["val"] = torch.empty((128, 32), dtype=torch.float16)
    elif bad == "shape":
        plan.dots[0].meta["val"] = torch.empty((128, 64), dtype=torch.float32)
    elif bad == "missing_issue":
        groups = groups[1:]
    elif bad == "mixed_role":
        plan = replace(plan, warp_mma_stages=frozenset({1}))
    elif bad == "direct":
        plan = replace(plan, direct_output=True)
    elif bad == "group_geometry":
        changed = replace(groups[-1], geometries=(StageGeometry((128, 16, 48), False),))
        plan = replace(plan, contraction_groups=(*groups[:-1], changed))
    assert plan_loop_tmem_bridges(plan, groups) is None


def test_source_dependent_broadcast_does_not_pass_coordinate_proof():
    graph = Graph()
    source = _input(graph, "source", (1, 32), torch.float32)
    row = _input(graph, "row", (128, 32), torch.float32)
    value = _call(
        graph, torch.ops.aten.add.Tensor, (source, row), (128, 32), torch.float32
    )
    ancestors = {source: {source}, row: {row}, value: {source, row, value}}
    assert (
        _coordinate_path(value, ("row", "k"), source, ("row", "k"), ancestors) is None
    )


def test_actual_graph_candidates_are_not_explicit_accumulator_residency():
    kernel, args = _kda_fixture()
    plan, _, _ = _capture(
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
    assert plan.contraction_groups is not None
    groups = tuple(
        group
        for group in plan.contraction_groups
        if group.stages[0] not in plan.warp_mma_stages
    )
    result = plan_loop_tmem_bridges(plan, groups)
    assert result is not None
    assert [(item.source_stage, item.destination_group.stages) for item in result] == [
        (10, (11,)),
        (11, (13, 14)),
    ]
    assert all(item.source_geometry.transpose for item in result)
    assert all(item.physical_shape == (128, 32) for item in result)
