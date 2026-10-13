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
from helion._compiler.cute.chained_loop_tmem_carry import plan_loop_tmem_carry
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_tmem_accumulator import (
    plan_tmem_accumulator_residency,
)
from helion.language import memory_ops


def _fixture(width=32, dtype=torch.bfloat16, mode="plain", rows=128):
    graph = Graph()
    state = _input(graph, "state", (rows, width), torch.float32)
    gamma = _input(graph, "gamma", (1, width), torch.float32)
    originals = []
    geometries = []
    for index in range(2):
        image = state
        if mode in ("rounding", "different_casts") and (
            mode == "rounding" or index == 1
        ):
            image = _convert(
                graph, _convert(graph, state, torch.float16), torch.float32
            )
        if mode == "arithmetic_snapshot" and index == 0:
            image = _call(
                graph,
                torch.ops.aten.mul.Scalar,
                (state, 0.5),
                (rows, width),
                torch.float32,
            )
        if mode == "reshape" and index == 0:
            image = _call(
                graph,
                torch.ops.aten.view.default,
                (state, (rows, width)),
                (rows, width),
                torch.float32,
            )
        if mode == "gather" and index == 0:
            indices = _input(graph, "indices", (width,), torch.int64)
            image = _call(
                graph,
                torch.ops.aten.index_select.default,
                (state, 1, indices),
                (rows, width),
                torch.float32,
            )
        operand = _convert(graph, image, dtype)
        if mode == "physical_b" and index == 0:
            transposed = _call(
                graph, torch.ops.aten.t.default, (operand,), (width, rows), dtype
            )
            other = _input(graph, "b_left", (rows, width), dtype)
            originals.append(_dot(graph, other, transposed))
            geometries.append(StageGeometry((rows, rows, width), False))
        elif mode == "duplicate_operand" and index == 0:
            assert rows == width
            originals.append(_dot(graph, operand, operand))
            geometries.append(StageGeometry((rows, rows, width), False))
        else:
            other = _input(graph, f"weight_{index}", (width, 16), dtype)
            originals.append(_dot(graph, operand, other))
            geometries.append(StageGeometry((rows, 16, width), False))
    update = _input(graph, "update", (rows, 16), dtype)
    output_rhs = _input(graph, "output_rhs", (16, 16), dtype)
    carry_rhs = _input(graph, "carry_rhs", (16, width), dtype)
    side = _dot(graph, update, output_rhs, originals[1])
    accumulator = _call(
        graph, torch.ops.aten.mul.Tensor, (state, gamma), (rows, width), torch.float32
    )
    if mode == "masked_accumulator":
        mask = _input(graph, "mask", (rows, width), torch.bool)
        accumulator = _call(
            graph,
            torch.ops.aten.where.self,
            (mask, accumulator, state),
            (rows, width),
            torch.float32,
        )
    carry_update = (
        _input(graph, "other_update", (rows, 16), dtype)
        if mode == "unrelated_group"
        else update
    )
    output = _dot(graph, carry_update, carry_rhs, accumulator)
    sink = _input(graph, "sink", (rows, 16), torch.float32)
    graph.call_function(
        memory_ops.store, (sink, [slice(None), slice(None)], side, None)
    )
    if mode in ("old_store", "new_store"):
        sink_state = _input(graph, "sink_state", (rows, width), torch.float32)
        graph.call_function(
            memory_ops.store,
            (
                sink_state,
                [slice(None), slice(None)],
                state if mode == "old_store" else output,
                None,
            ),
        )
    if mode == "collective":
        _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (state, [1], True),
            (rows, 1),
            torch.float32,
        )
    if mode == "new_consumer":
        _call(
            graph, torch.ops.aten.neg.default, (output,), (rows, width), torch.float32
        )
    if mode in ("shape", "bad_shape"):
        query = graph.call_function(torch.ops.aten.sym_size.int, (state, 0))
        query.meta["val"] = rows + (mode == "bad_shape")
    carries, outputs = (state,), (output,)
    if mode == "cross_carry":
        second_state = _input(graph, "second_state", (rows, 16), torch.float32)
        carries += (second_state,)
        outputs += (originals[1],)
    if mode == "reads_other_carry":
        carries += (gamma,)
        outputs += (gamma,)
    plan = _loop_plan(graph, carries, outputs)
    if mode == "half_carry":
        # The collector already rejects an originally mixed-precision port.
        # A retained region must not permit the same change after collection.
        state.meta["val"] = torch.empty((rows, width), dtype=dtype)
    groups = (
        ContractionGroup((0,), (geometries[0],)),
        ContractionGroup((1,), (geometries[1],)),
        ContractionGroup(
            (2, 3),
            (
                StageGeometry((rows, 16, 16), False),
                StageGeometry((rows, width, 16), False),
            ),
        ),
    )
    return replace(plan, contraction_groups=groups), groups


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [16, 32, 64, 128])
@pytest.mark.parametrize("mode", ["plain", "rounding", "shape"])
def test_exact_fp32_carry_shared_snapshot_and_original_accumulator(dtype, width, mode):
    plan, groups = _fixture(width, dtype, mode)
    assert plan.region is not None
    before = tuple((node, node.args, dict(node.kwargs)) for node in plan.region.nodes)
    residency = plan_tmem_accumulator_residency(plan, groups)
    assert residency is not None
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        candidate = plan_loop_tmem_carry(plan, groups, residency)
    assert candidate is not None
    assert candidate.region is plan.region and candidate.carry is plan.region.carries[0]
    assert candidate.input is candidate.carry.input and candidate.output is plan.dots[3]
    assert candidate.carry_index == candidate.carry.input_index
    assert candidate.final_group is groups[-1] and candidate.member == 3
    assert candidate.member_offset == 16 and candidate.arena_columns == 16 + width
    assert (
        candidate.snapshot_shape == (128, width) and candidate.snapshot_dtype == dtype
    )
    assert [use.group for use in candidate.snapshot_users] == list(groups[:2])
    assert candidate.snapshot_casts[0] == dtype
    if mode == "rounding" and dtype == torch.bfloat16:
        assert candidate.snapshot_casts == (
            torch.bfloat16,
            torch.float32,
            torch.float16,
        )
    assert candidate.accumulator is plan.dots[3].args[2]
    assert candidate.input in candidate.accumulator_nodes
    assert candidate.residency is residency
    assert residency.columns <= candidate.member_offset
    assert before == tuple(
        (node, node.args, dict(node.kwargs)) for node in plan.region.nodes
    )
    with pytest.raises(FrozenInstanceError):
        candidate.member_offset = 0  # pyrefly: ignore [read-only]


@pytest.mark.parametrize(
    "mode",
    [
        "different_casts",
        "arithmetic_snapshot",
        "reshape",
        "gather",
        "physical_b",
        "duplicate_operand",
        "old_store",
        "new_store",
        "collective",
        "new_consumer",
        "cross_carry",
        "reads_other_carry",
        "unrelated_group",
        "masked_accumulator",
        "bad_shape",
        "half_carry",
    ],
)
def test_unclassified_old_or_new_carry_uses_and_precision_changes_reject(mode):
    plan, groups = _fixture(128 if mode == "duplicate_operand" else 32, mode=mode)
    assert plan_loop_tmem_carry(plan, groups) is None


def test_sparse_or_padded_carry_does_not_claim_full_datapath_ownership():
    for rows, width in ((64, 32), (128, 24), (128, 256)):
        plan, groups = _fixture(width, rows=rows)
        assert plan_loop_tmem_carry(plan, groups) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_one_snapshot_serves_every_member_of_an_earlier_group(dtype):
    plan, groups = _fixture(dtype=dtype)
    groups = (
        ContractionGroup((0, 1), (groups[0].geometries[0], groups[1].geometries[0])),
        groups[-1],
    )
    plan = replace(plan, contraction_groups=groups)
    candidate = plan_loop_tmem_carry(plan, groups)
    assert candidate is not None and candidate.residency is None
    assert len(candidate.snapshot_users) == 1
    use = candidate.snapshot_users[0]
    assert use.group is groups[0] and len(use.operands) == 2
    assert use.operands == (plan.dots[0].args[0], plan.dots[1].args[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [16, 32, 64])
def test_transposed_logical_carry_retains_physical_canonical_snapshot(dtype, width):
    graph = Graph()
    state = _input(graph, "state", (width, 128), torch.float32)
    gamma = _input(graph, "gamma", (width, 1), torch.float32)
    projections = []
    for index in range(2):
        snapshot = _convert(graph, state, dtype)
        weights = _input(graph, f"weights_{index}", (16, width), dtype)
        projections.append(_dot(graph, weights, snapshot))
    update = _input(graph, "update", (16, 128), dtype)
    out_weights = _input(graph, "out_weights", (16, 16), dtype)
    state_weights = _input(graph, "state_weights", (width, 16), dtype)
    _dot(graph, out_weights, update, projections[1])
    accumulator = _call(
        graph, torch.ops.aten.mul.Tensor, (state, gamma), (width, 128), torch.float32
    )
    output = _dot(graph, state_weights, update, accumulator)
    groups = (
        ContractionGroup((0,), (StageGeometry((16, 128, width), True),)),
        ContractionGroup((1,), (StageGeometry((16, 128, width), True),)),
        ContractionGroup(
            (2, 3),
            (StageGeometry((16, 128, 16), True), StageGeometry((width, 128, 16), True)),
        ),
    )
    plan = replace(_loop_plan(graph, (state,), (output,)), contraction_groups=groups)
    residency = plan_tmem_accumulator_residency(plan, groups)
    assert residency is not None
    candidate = plan_loop_tmem_carry(plan, groups, residency)
    assert candidate is not None and candidate.geometry.transpose
    assert candidate.snapshot_shape == (128, width)
    assert candidate.geometry.result_coordinates("row", "column") == ("column", "row")
    assert all(
        use.operands == (plan.dots[use.group.stages[0]].args[1],)
        for use in candidate.snapshot_users
    )
    assert candidate.member_offset == 16 and candidate.arena_columns == 16 + width


@pytest.mark.parametrize(
    "mode",
    [
        "missing_group",
        "graph_revision",
        "wrong_geometry",
        "kwargs",
        "no_loop",
        "foreign_residency",
        "resident_overlap",
    ],
)
def test_invalid_retained_proofs_and_overlapping_resident_window_reject(mode):
    plan, groups = _fixture()
    residency = plan_tmem_accumulator_residency(plan, groups)
    assert residency is not None
    if mode == "missing_group":
        groups = groups[1:]
    elif mode == "graph_revision":
        assert plan.region is not None
        plan.region.graph.placeholder("changed_revision")
    elif mode == "wrong_geometry":
        groups = (
            *groups[:-1],
            replace(
                groups[-1],
                geometries=(
                    groups[-1].geometries[0],
                    StageGeometry((128, 64, 16), False),
                ),
            ),
        )
        plan = replace(plan, contraction_groups=groups)
    elif mode == "kwargs":
        plan.dots[0].kwargs = {"unexpected": 1}
    elif mode == "no_loop":
        plan = replace(plan, loop=None)
    elif mode == "foreign_residency":
        foreign, foreign_groups = _fixture()
        residency = plan_tmem_accumulator_residency(foreign, foreign_groups)
    else:
        # An alleged larger predecessor window is stale and cannot authorize
        # overwriting the authoritative carry at final-group offset16.
        residency = replace(residency, columns=32)
    assert plan_loop_tmem_carry(plan, groups, residency) is None


def test_actual_kda_preserves_carry_member_and_relocates_resident_prefix_together():
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
    residency = plan_tmem_accumulator_residency(plan, groups)
    candidate = plan_loop_tmem_carry(plan, groups, residency)
    assert candidate is not None and candidate.residency is residency
    assert candidate.carry_index == 2 and candidate.member == 14
    assert candidate.final_group.stages == (13, 14)
    assert (
        candidate.geometry.logical == (128, 128, 32)
        and not candidate.geometry.transpose
    )
    assert (candidate.member_offset, candidate.arena_columns) == (32, 160)
    assert candidate.snapshot_shape == (128, 128)
    assert candidate.snapshot_casts == (torch.bfloat16,)
    assert [use.group.stages for use in candidate.snapshot_users] == [(10,), (12,)]
    assert all(use.group.geometries[0].transpose for use in candidate.snapshot_users)
    assert candidate.accumulator is plan.dots[14].args[2]
    assert candidate.accumulator.target is torch.ops.aten.mul.Tensor
    assert residency is not None and residency.source_stage == 12
    assert residency.offset == 0 and residency.columns == candidate.member_offset
