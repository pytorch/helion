from __future__ import annotations

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
from .test_cute_chained_group_guards import _group_config_candidate
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
import helion
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_contraction_groups import contraction_groups
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_tcgen_stage import stage_geometry
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
from helion._compiler.cute.cute_reshape import _get_tile_shape
from helion.language import memory_ops
from helion.language import scan_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.device_ir import GraphInfo

if TYPE_CHECKING:
    from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan


def _assert_safe(layout: SharedMemoryLayoutPlan) -> None:
    for region in layout.regions:
        assert region.byte_offset % 128 == 0
        assert region.byte_end <= layout.allocated_bytes
        assert region.live_from < region.live_until
    for left, right in combinations(layout.regions, 2):
        assert not (left.overlaps_lifetime(right) and left.overlaps_storage(right))


@pytest.mark.parametrize("explicit_accumulator", [False, True])
def test_consumed_inputs_recycle_after_reads_before_publication(
    explicit_accumulator: bool,
) -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    if explicit_accumulator:
        second = _dot(graph, left, right, first)
    else:
        squared = _call(
            graph, torch.ops.aten.mul.Tensor, (first, first), (16, 16), torch.float32
        )
        second = _dot(graph, _convert(graph, squared, torch.bfloat16), right)
    third = _dot(graph, _convert(graph, second, torch.bfloat16), right)
    layout = plan_contraction_workspace(_plan(graph, (third,)))
    assert layout.allocated_bytes == 1024
    assert [region.byte_offset for region in layout.regions] == [0, 0, 0]
    assert [(region.live_from, region.live_until) for region in layout.regions] == [
        (1, 3),
        (3, 5),
        (5, 7),
    ]
    _assert_safe(layout)


def test_root_live_out_keeps_an_earlier_result_alive() -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    second = _dot(graph, _convert(graph, first, torch.bfloat16), right)
    third = _dot(graph, _convert(graph, second, torch.bfloat16), right)
    layout = plan_contraction_workspace(_plan(graph, (first, third)))
    assert layout.allocated_bytes == 2048
    assert [region.byte_offset for region in layout.regions] == [0, 1024, 1024]
    assert layout.regions[0].live_until == 7
    _assert_safe(layout)


def test_grouped_outputs_coexist_including_dead_members() -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    _dot(graph, left, right)
    third = _dot(graph, _convert(graph, first, torch.bfloat16), right)
    plan = _plan(graph, (third,))
    geometry = StageGeometry((16, 16, 16), False)
    plan = replace(
        plan,
        contraction_groups=(
            ContractionGroup((0, 1), (geometry, geometry)),
            ContractionGroup((2,), (geometry,)),
        ),
    )
    layout = plan_contraction_workspace(plan)
    assert layout.allocated_bytes == 2048
    assert [region.live_from for region in layout.regions] == [1, 1, 5]
    assert not layout.regions[0].overlaps_storage(layout.regions[1])
    assert layout.regions[1].live_until == 2
    _assert_safe(layout)


@pytest.mark.parametrize("kind", ["scan", "sum"])
def test_collective_materialization_cuts_transitive_dot_lifetime(kind: str) -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    if kind == "scan":
        collective = _call(
            graph,
            scan_ops._associative_scan,
            (0, first, 0, False, False),
            (16, 16),
            torch.float32,
        )
    else:
        collective = _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (first, [0], True),
            (1, 16),
            torch.float32,
        )
    second = _dot(graph, left, right)
    epilogue = _call(
        graph,
        torch.ops.aten.add.Tensor,
        (second, collective),
        (16, 16),
        torch.float32,
    )
    layout = plan_contraction_workspace(_plan(graph, (epilogue,)))
    assert layout.allocated_bytes == 1024
    assert layout.regions[0].live_until == 3
    assert layout.regions[1].byte_offset == 0
    _assert_safe(layout)


def test_terminal_collective_reads_after_the_last_dot() -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    second = _dot(graph, left, right)
    reduction = _call(
        graph, torch.ops.aten.sum.dim_IntList, (first, [0]), (16,), torch.float32
    )
    layout = plan_contraction_workspace(_plan(graph, (second, reduction)))
    assert layout.allocated_bytes == 2048
    assert layout.regions[0].live_until == 5
    _assert_safe(layout)


@pytest.mark.parametrize("use", ["value", "index", "mask"])
def test_stores_read_at_epilogue_even_if_the_fx_store_appears_early(use: str) -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    output = _input(
        graph, "output", (256,) if use == "index" else (16, 16), torch.float32
    )
    first = _dot(graph, left, right)
    value = first if use == "value" else left
    indices = (
        [_convert(graph, first, torch.int32)]
        if use == "index"
        else [slice(None), slice(None)]
    )
    mask = (
        _call(graph, torch.ops.aten.gt.Scalar, (first, 0.0), (16, 16), torch.bool)
        if use == "mask"
        else None
    )
    graph.call_function(memory_ops.store, (output, indices, value, mask))
    second = _dot(graph, left, right)
    layout = plan_contraction_workspace(_plan(graph, (second,)))
    assert layout.allocated_bytes == 2048
    assert layout.regions[0].live_until == 5
    _assert_safe(layout)


def test_carried_output_stays_live_until_advance_carries() -> None:
    original = chained_matmul.plan_chained_matmul
    layouts: list[SharedMemoryLayoutPlan] = []

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        assert plan is not None and plan.loop is not None and plan.region is not None
        assert plan.region.carries[0].output is plan.dots[1]
        layouts.append(plan_contraction_workspace(plan))
        return plan

    with (
        _cpu_codegen(),
        patch.object(chained_matmul, "plan_chained_matmul", side_effect=observe),
    ):
        bound = _group_config_candidate._bind_isolated(
            (
                torch.empty((3, 64, 16), dtype=torch.bfloat16),
                torch.empty((3, 16, 32), dtype=torch.bfloat16),
                torch.empty((64, 32), dtype=torch.float32),
            )
        )
        bound.to_code(
            helion.Config(
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_group_contractions=True,
            )
        )
    (layout,) = layouts
    assert layout.allocated_bytes == 16384
    assert [region.live_until for region in layout.regions] == [5, 5]
    _assert_safe(layout)


def test_largest_simultaneous_result_is_placed_first() -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    small_rhs = _input(graph, "small_rhs", (16, 8), torch.bfloat16)
    large_rhs = _input(graph, "large_rhs", (16, 32), torch.bfloat16)
    small, large = _dot(graph, left, small_rhs), _dot(graph, left, large_rhs)
    plan = _plan(graph, (small, large))
    plan = replace(
        plan,
        contraction_groups=(
            ContractionGroup(
                (0, 1),
                (StageGeometry((16, 8, 16), False), StageGeometry((16, 32, 16), False)),
            ),
        ),
    )
    layout = plan_contraction_workspace(plan)
    assert layout.allocated_bytes == 2560
    assert [region.byte_offset for region in layout.regions] == [2048, 0]
    _assert_safe(layout)


def test_workspace_counts_logical_elements_not_mma_padding() -> None:
    graph = Graph()
    left = _input(graph, "left", (13, 17), torch.bfloat16)
    right = _input(graph, "right", (17, 11), torch.bfloat16)
    result = _dot(graph, left, right)
    layout = plan_contraction_workspace(_plan(graph, (result,)))
    assert layout.regions[0].byte_size == 13 * 11 * 4
    assert layout.allocated_bytes == 640
    _assert_safe(layout)


def test_later_collective_does_not_reopen_an_earlier_dot_lifetime() -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    scan = _call(
        graph,
        scan_ops._associative_scan,
        (0, first, 0, False, False),
        (16, 16),
        torch.float32,
    )
    second = _dot(graph, left, right)
    reduction = _call(
        graph, torch.ops.aten.sum.dim_IntList, (scan, [0]), (16,), torch.float32
    )
    layout = plan_contraction_workspace(_plan(graph, (second, reduction)))
    assert layout.allocated_bytes == 1024
    assert layout.regions[0].live_until == 3
    _assert_safe(layout)


def test_bt32_dv128_trace_fits_with_only_contraction_result_reuse() -> None:
    from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32

    from .test_cute_chunk_prefill import _inputs

    kernel = helion.kernel(
        kda_prefill_native_math_bt32.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides={
            "cute_chained_group_contractions": True,
            "cute_chained_mma_schedule": "tcgen05_tmem",
        },
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[128],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_group_contractions=True,
            )
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graph = chained_matmul._classify_chained_graph(
                bound.host_function.device_ir.graphs
            )
            assert graph is not None and graph.loop is not None
            shapes = []
            for spec in graph.region.contractions:
                lhs = _get_tile_shape(spec.lhs.meta["val"], bound.env, config)
                rhs = _get_tile_shape(spec.rhs.meta["val"], bound.env, config)
                shapes.append((lhs[0], rhs[1], lhs[1]))
            # Build the resource input from the real admitted graph. Launch
            # axes are irrelevant to this planner; no device context is needed.
            plan = ChainedMatmulPlan(
                graph.root.graph_id,
                graph.dots,
                graph.store,
                (),
                tuple(shapes),
                torch.bfloat16,
                128,
                graph.scans,
                strategy="tcgen05_tmem",
                region=graph.region,
                loop=graph.loop,
            )
            geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
            assert all(geometry is not None for geometry in geometries)
            admitted = tuple(item for item in geometries if item is not None)
            groups = contraction_groups(plan, admitted)
            plan = replace(plan, contraction_groups=groups)
            layout = plan_contraction_workspace(plan)
            collective_bytes = sum(
                (
                    4
                    * torch.Size(
                        _get_tile_shape(node.meta["val"], bound.env, config)
                    ).numel()
                    + 127
                )
                // 128
                * 128
                for node in (*graph.region.scans, *graph.region.reductions)
            )
            carry_bytes = sum(
                4
                * torch.Size(
                    _get_tile_shape(carry.input.meta["val"], bound.env, config)
                ).numel()
                for carry in graph.region.carries
            )
        # Keep this regression scoped to C-only allocation; loop carry/C
        # packing has its own source and execution tests.
        with patch(
            "helion._compiler.cute.chained_loop_workspace.plan_loop_workspace",
            return_value=None,
        ):
            source = bound.to_code(config)
    assert len(plan.dots) == 15
    assert groups[-1].stages == (13, 14)
    assert groups[-1].physical == (128, 160, 32)
    assert sum(4 * m * n for m, n, _ in shapes) == 172032
    assert layout.allocated_bytes == 81920
    assert layout.region("chain_14_c").byte_offset == 0
    assert layout.region("chain_13_c").byte_offset == 65536
    physical = tuple(group.physical for group in groups)
    operand_bytes = 2 * (
        max(m * k for m, _, k in physical) + max(n * k for _, n, k in physical)
    )
    # Two separately aligned allocations: the 15 barriers and TMEM holder.
    total = (
        operand_bytes + layout.allocated_bytes + 256 + carry_bytes + collective_bytes
    )
    assert total == 214016
    assert total < 232448
    assert "chain_c_workspace = cute.arch.alloc_smem(cutlass.Float32, 20480," in source
    assert "cute.make_tensor(chain_c_workspace + 16384," in source
    assert "(128, 160)" in source
    assert "chain_13_mma" in source and "chain_14_mma" not in source
    _assert_safe(layout)
