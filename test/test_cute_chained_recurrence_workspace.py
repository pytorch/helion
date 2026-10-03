from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_frame import _capture
from .test_cute_chained_preparation_frame import _synthetic
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_preparation_cut import PreparationCut
from helion._compiler.cute.chained_preparation_cut import PreparationImage
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.contraction_region import _domain
from helion.language import memory_ops

if TYPE_CHECKING:
    from torch.fx import Node

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_recurrence_workspace import RecurrenceWorkspace


def _safe(workspace: RecurrenceWorkspace) -> None:
    for left, right in combinations(workspace.layout.regions, 2):
        assert not (left.overlaps_lifetime(right) and left.overlaps_storage(right))
    assert workspace.peak_live_bytes <= workspace.layout.allocated_bytes
    assert (
        workspace.shared_bytes
        == workspace.layout.allocated_bytes + workspace.a_bytes + workspace.b_bytes
    )
    assert workspace.a_bytes % 128 == workspace.b_bytes % 128 == 0
    bindings = dict(workspace.bindings)
    assert all(
        bindings[image.node] == f"chain_prepared_{i}"
        for i, image in enumerate(workspace.cut.images)
    )
    assert not any(
        region.name.startswith("chain_prepared_") for region in workspace.layout.regions
    )


def _cut_for(
    plan: ChainedMatmulPlan, preparation: tuple[Node, ...], shared: tuple[Node, ...]
) -> tuple[PreparationCut, dict[Node, tuple[int, ...]]]:
    assert plan.region is not None and plan.loop is not None
    region = plan.region
    recurrence = tuple(
        node for node in region.nodes if node not in (*preparation, *shared)
    )
    images = tuple(
        PreparationImage(
            node,
            node.meta["val"].dtype,
            _domain(node.meta["val"]),
            tuple(
                user
                for user in region.nodes
                if user in recurrence and node in user.all_input_nodes
            ),
        )
        for node in preparation
        if any(user in recurrence for user in node.users)
    )
    return PreparationCut(
        region,
        region.carries,
        (),
        shared,
        preparation,
        recurrence,
        images,
        plan.loop.storage_key,
    ), {
        node: tuple(node.meta["val"].shape)
        for node in region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }


@pytest.mark.parametrize("kind", ["dot", "collective", "cache"])
def test_prepared_images_stop_traversal_and_do_not_allocate_preparation_c(
    kind: str,
) -> None:
    plan, cut, shapes = _synthetic(kind)
    workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None
    _safe(workspace)
    assert [item.name for item in workspace.layout.regions] == [
        "chain_loop_carry_2",
        "chain_2_c",
    ]
    assert workspace.layout.allocated_bytes == workspace.peak_live_bytes == 1024
    assert [(item.read_event, item.publication_event) for item in workspace.stages] == [
        (4, 5)
    ]
    assert workspace.transition_event == 6
    assert (
        workspace.layout.regions[0].live_until == 5
    )  # Explicit FP32 seed reads state.
    assert workspace.a_bytes == 4096 and workspace.b_bytes == 512


@pytest.mark.parametrize("recurrent_first", [False, True])
def test_actual_unrelated_loop_preserves_original_stage_ids_and_exact_accumulators(
    recurrent_first: bool,
) -> None:
    plan, cut, shapes = _capture(
        _typed_sequence,
        _inputs(recurrent_first, typed=True),
        helion.Config(
            block_sizes=[], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        ),
    )
    workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None
    _safe(workspace)
    expected = ((0,), (3,)) if recurrent_first else ((2,), (3,))
    assert tuple(stage.group.stages for stage in workspace.stages) == expected
    carry = workspace.layout.regions[0]
    first = workspace.layout.region(f"chain_{expected[0][0]}_c")
    last = workspace.layout.region("chain_3_c")
    assert carry.live_until == 7
    assert not carry.overlaps_storage(first)
    assert carry.overlaps_storage(last)
    assert workspace.layout.allocated_bytes == workspace.peak_live_bytes == 2048


def _pair(*, grouped: bool = False, swap: bool = False, dependent: bool = False):
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first_state = _input(graph, "first_state", (16, 16), torch.float32)
    second_state = _input(graph, "second_state", (16, 16), torch.float32)
    prepared = _convert(graph, _dot(graph, left, right), torch.bfloat16)
    first = _dot(graph, prepared, right, first_state)
    second = _dot(graph, prepared, right, first if dependent else second_state)
    plan = _loop_plan(
        graph, (first_state, second_state), (second, first) if swap else (first, second)
    )
    if grouped:
        geometry = StageGeometry((16, 16, 16), False)
        plan = replace(
            plan,
            contraction_groups=(
                ContractionGroup((0,), (geometry,)),
                ContractionGroup((1, 2), (geometry, geometry)),
            ),
        )
    cut, shapes = _cut_for(plan, (plan.dots[0], prepared), (left, right))
    return plan, cut, shapes


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_two_carry_transition_and_group_publication_preserve_zero_trip_and_swaps(
    grouped: bool, steps: int
) -> None:
    plan, cut, shapes = _pair(grouped=grouped, swap=True)
    workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None
    _safe(workspace)
    assert workspace.layout.allocated_bytes == workspace.peak_live_bytes == 2048
    regions = workspace.layout.regions
    assert regions[0].live_from == regions[1].live_from == -1
    assert not regions[0].overlaps_storage(regions[1])
    assert regions[2].live_from == 3
    assert regions[3].live_from == (3 if grouped else 5)
    arena = torch.empty(workspace.layout.allocated_bytes // 4)
    views = {
        item.name: arena[item.byte_offset // 4 : item.byte_end // 4].reshape(16, 16)
        for item in regions
    }
    states = [views[regions[i].name] for i in range(2)]
    for index, state in enumerate(states):
        state.fill_(index + 1)
    for _ in range(steps):
        # Zero-product dots retain their explicit accumulator. All grouped
        # operands/seeds finish reading before any member publication.
        for stage in workspace.stages:
            results = [
                (index, states[index - 1].clone()) for index in stage.group.stages
            ]
            for index, value in results:
                views[f"chain_{index}_c"].copy_(value)
        next_values = [views["chain_2_c"].clone(), views["chain_1_c"].clone()]
        for state, value in zip(states, next_values, strict=True):
            state.copy_(value)
    assert torch.equal(states[0], torch.full((16, 16), 2.0 if steps % 2 else 1.0))
    assert torch.equal(states[1], torch.full((16, 16), 1.0 if steps % 2 else 2.0))


@pytest.mark.parametrize("use", ["value", "index", "mask", "kwargs", "carry_output"])
def test_all_final_read_edges_extend_current_carry_lifetime(use: str) -> None:
    graph = Graph()
    state = _input(graph, "state", (16, 16), torch.float32)
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    destination = _input(graph, "destination", (16, 16), torch.float32)
    prepared = _convert(graph, _dot(graph, operand, operand), torch.bfloat16)
    result = _dot(graph, prepared, operand, state)
    indices = [_convert(graph, state, torch.int32)] if use == "index" else [slice(None)]
    mask = (
        _call(graph, torch.ops.aten.gt.Scalar, (state, 0), (16, 16), torch.bool)
        if use == "mask"
        else None
    )
    store = graph.call_function(
        memory_ops.store,
        (destination, indices, state if use == "value" else result, mask),
    )
    if use == "kwargs":
        store.kwargs = {"_extra_deps": state}
    final = (
        _call(
            graph, torch.ops.aten.add.Tensor, (state, result), (16, 16), torch.float32
        )
        if use == "carry_output"
        else result
    )
    plan = _loop_plan(graph, (state,), (final,))
    cut, shapes = _cut_for(plan, (plan.dots[0], prepared), (operand, destination))
    workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None
    _safe(workspace)
    assert workspace.layout.regions[0].live_until == workspace.transition_event + 1
    assert workspace.layout.allocated_bytes == 2048


@pytest.mark.parametrize(
    "bad",
    [
        "root",
        "strategy",
        "direct_output",
        "recurrence_warp",
        "cross_role_group",
        "dependent_group",
        "missing_frontier",
        "duplicate_frontier",
        "foreign_cut",
        "storage_key",
        "stale_graph",
        "recurrence_collective",
        "recurrence_cache",
        "half_product",
        "missing_shape",
        "shape_mismatch",
        "carry_dtype",
    ],
)
def test_unsupported_contracts_fail_closed(bad: str) -> None:
    plan, cut, shapes = _synthetic()
    if bad == "root":
        plan = replace(plan, loop=None)
    elif bad == "strategy":
        plan = replace(plan, strategy="warp")
    elif bad == "direct_output":
        plan = replace(plan, direct_output=True)
    elif bad == "recurrence_warp":
        plan = replace(plan, warp_mma_stages=frozenset({0, 1, 2}))
    elif bad in ("cross_role_group", "dependent_group"):
        if bad == "dependent_group":
            plan, cut, shapes = _pair(grouped=True, dependent=True)
        else:
            geometry = StageGeometry((16, 16, 16), False)
            plan = replace(
                plan,
                contraction_groups=(
                    ContractionGroup((0,), (geometry,)),
                    ContractionGroup((1, 2), (geometry, geometry)),
                ),
            )
    elif bad == "missing_frontier":
        cut = replace(cut, images=())
    elif bad == "duplicate_frontier":
        cut = replace(cut, images=cut.images * 2)
    elif bad == "foreign_cut":
        _, cut, _ = _synthetic()
    elif bad == "storage_key":
        cut = replace(cut, storage_proof_key="different")
    elif bad == "stale_graph":
        cut.region.graph.placeholder("changed")
    elif bad in ("recurrence_collective", "recurrence_cache"):
        assert plan.pointwise_cache is not None
        node = (
            cut.region.reductions[0]
            if bad == "recurrence_collective"
            else plan.pointwise_cache.entries[0].node
        )
        cut = replace(
            cut,
            preparation=tuple(item for item in cut.preparation if item is not node),
            recurrence=(*cut.recurrence, node),
        )
    elif bad == "half_product":
        node = plan.dots[-1]
        node.args = (*node.args[:3], torch.float16)
    elif bad == "missing_shape":
        shapes.pop(cut.carries[0].input)
    elif bad == "shape_mismatch":
        shapes[cut.carries[0].input] = (32, 16)
    else:
        cut.carries[0].input.meta["val"] = torch.empty((16, 16), dtype=torch.float16)
    assert plan_recurrence_workspace(plan, cut, shapes) is None


def test_plan_is_immutable_and_deterministic_without_modifying_graph() -> None:
    plan, cut, shapes = _pair()
    before = tuple(
        (node, node.args, dict(node.kwargs), dict(node.meta))
        for node in cut.region.nodes
    )
    workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None and workspace == plan_recurrence_workspace(
        plan, cut, shapes
    )
    assert before == tuple(
        (node, node.args, dict(node.kwargs), dict(node.meta))
        for node in cut.region.nodes
    )
    with pytest.raises(FrozenInstanceError):
        workspace.a_bytes = 0  # pyrefly: ignore [read-only]


def test_actual_kda_recurrence_and_preparation_accounting_without_device_context() -> (
    None
):
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
    workspace = plan_recurrence_workspace(plan, cut, shapes)
    frame = plan_preparation_frame(plan, cut, shapes)
    assert workspace is not None and frame is not None
    _safe(workspace)
    assert tuple(
        index for stage in workspace.stages for index in stage.group.stages
    ) == (10, 11, 12, 13, 14)
    assert workspace.a_bytes == 32768 and workspace.b_bytes == 10240
    assert frame.layout.allocated_bytes == 49920
    assert all(
        dict(workspace.bindings)[node] == f"chain_{index}_c"
        for index, node in enumerate(plan.dots)
        if node in cut.recurrence
    )
    assert all(
        f"chain_{index}_c" not in {region.name for region in workspace.layout.regions}
        for index in range(10)
    )
    assert workspace.layout.allocated_bytes == workspace.peak_live_bytes
    assert workspace.layout.allocated_bytes == 98304
    assert plan.dots[13].args[2] is plan.dots[12]
    carry = workspace.layout.region("chain_loop_carry_2")
    assert carry.live_until == 27  # Stage14 reads the original FP32 gamma seed.
    assert workspace.layout.region("chain_12_c").live_until == 27
    assert workspace.layout.region("chain_11_c").live_until == 27
    # Before protocol storage, this exact schedule already exceeds the stated
    # 227 KiB pipeline budget. Accounting does not silently change the schedule.
    assert 2 * frame.layout.allocated_bytes + workspace.shared_bytes == 241152
    assert 241152 - 232448 == 8704
