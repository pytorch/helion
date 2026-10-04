from __future__ import annotations

import ast
from dataclasses import replace
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
from .test_cute_chained_loop_workspace_integration import _args as _paired_args
from .test_cute_chained_loop_workspace_integration import _config as _paired_config
from .test_cute_chained_loop_workspace_integration import _paired_carries
from .test_cute_chained_workspace import _assert_safe
import helion
from helion._compiler.cute import chained_loop
from helion._compiler.cute import chained_loop_workspace
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_loop import ChainedLoopPlan
from helion._compiler.cute.chained_loop_workspace import plan_loop_workspace
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import LoopCarry
from helion._compiler.device_ir import LoopInterface
from helion._compiler.device_ir import RootGraphInfo
from helion.language import memory_ops
from helion.language import scan_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from helion._compiler.device_ir import GraphInfo


def _loop_plan(
    graph: Graph, carries: tuple[Node, ...], outputs: tuple[Node, ...]
) -> ChainedMatmulPlan:
    """Build typed loop ports without a device or a fabricated loop namespace."""
    output = graph.output(outputs)
    placeholders = tuple(graph.find_nodes(op="placeholder"))
    interface = LoopInterface(
        len(placeholders),
        tuple(LoopCarry(placeholders.index(node), i) for i, node in enumerate(carries)),
    )
    body = ForLoopGraphInfo(1, graph, [], [0], loop_interface=interface)
    graph.lint()
    region = collect_contraction_region(body)
    assert region is not None
    root = Graph()
    initial = tuple(
        _input(
            root, f"initial_{i}", tuple(node.meta["val"].shape), node.meta["val"].dtype
        )
        for i, node in enumerate(placeholders)
    )
    call = root.call_function(tuple, (initial,))
    root.output(call)
    loop = ChainedLoopPlan(
        RootGraphInfo(0, root), body, region, call, initial, 0, 3, (), ()
    )
    return ChainedMatmulPlan(
        0,
        tuple(spec.node for spec in region.contractions),
        output,
        (),
        tuple(
            (
                spec.lhs.meta["val"].shape[0],
                spec.rhs.meta["val"].shape[1],
                spec.lhs.meta["val"].shape[1],
            )
            for spec in region.contractions
        ),
        torch.bfloat16,
        128,
        region.scans,
        strategy="tcgen05_tmem",
        region=region,
        loop=loop,
    )


def _layout(plan: ChainedMatmulPlan) -> SharedMemoryLayoutPlan:
    assert plan.region is not None
    shapes = tuple(
        tuple(carry.input.meta["val"].shape) for carry in plan.region.carries
    )
    result = plan_loop_workspace(plan, shapes)
    assert result is not None
    _assert_safe(result)
    original = plan_contraction_workspace(plan)
    assert result.allocated_bytes <= original.allocated_bytes + sum(
        (4 * rows * cols + 127) // 128 * 128 for rows, cols in shapes
    )
    for old in original.regions:
        new = result.region(old.name)
        assert (new.byte_size, new.live_from, new.live_until, new.alignment) == (
            old.byte_size,
            old.live_from,
            old.live_until,
            old.alignment,
        )
    return result


@pytest.mark.parametrize("use", ["operand", "accumulator", "kwargs"])
def test_current_carry_recycles_only_after_all_stage_reads(use: str) -> None:
    graph = Graph()
    carry = _input(graph, "state", (16, 16), torch.float32)
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    if use == "kwargs":
        # Preserve dependencies that expression lowering might omit.
        left = _convert(graph, left, torch.bfloat16)
        left.kwargs = {"_extra_deps": carry}
    first = _dot(
        graph,
        _convert(graph, carry, torch.bfloat16) if use == "operand" else left,
        right,
        carry if use == "accumulator" else None,
    )
    second = _dot(graph, _convert(graph, first, torch.bfloat16), right)
    plan = _loop_plan(graph, (carry,), (second,))
    layout = _layout(plan)
    assert layout.allocated_bytes == 1024
    assert [(item.live_from, item.live_until) for item in layout.regions] == [
        (-1, 1),
        (1, 3),
        (3, 5),
    ]
    assert {item.byte_offset for item in layout.regions} == {0}


def test_late_explicit_accumulator_prevents_earlier_publication_from_clobbering_carry() -> (
    None
):
    graph = Graph()
    carry = _input(graph, "state", (16, 16), torch.float32)
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    first = _dot(graph, operand, operand)
    second = _dot(graph, _convert(graph, first, torch.bfloat16), operand, carry)
    layout = _layout(_loop_plan(graph, (carry,), (second,)))
    current = layout.regions[0]
    assert current.live_until == 3
    assert not current.overlaps_storage(layout.region("chain_0_c"))
    assert current.overlaps_storage(layout.region("chain_1_c"))


@pytest.mark.parametrize("use", ["value", "index", "mask", "kwargs"])
def test_early_source_store_keeps_current_carry_to_the_epilogue(use: str) -> None:
    graph = Graph()
    carry = _input(graph, "state", (16, 16), torch.float32)
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    output = _input(graph, "output", (16, 16), torch.float32)
    indices = [_convert(graph, carry, torch.int32)] if use == "index" else [slice(None)]
    mask = (
        _call(graph, torch.ops.aten.gt.Scalar, (carry, 0.0), (16, 16), torch.bool)
        if use == "mask"
        else None
    )
    store = graph.call_function(
        memory_ops.store, (output, indices, carry if use == "value" else left, mask)
    )
    if use == "kwargs":
        store.kwargs = {"_extra_deps": carry}
    first = _dot(graph, left, left)
    second = _dot(graph, _convert(graph, first, torch.bfloat16), left)
    layout = _layout(_loop_plan(graph, (carry,), (second,)))
    assert layout.regions[0].live_until == 5
    assert layout.allocated_bytes == 2048
    assert not layout.regions[0].overlaps_storage(layout.region("chain_1_c"))


@pytest.mark.parametrize("kind", ["scan", "sum"])
@pytest.mark.parametrize("terminal", [False, True])
def test_collectives_read_current_carry_at_their_source_order_cut(
    kind: str, terminal: bool
) -> None:
    graph = Graph()
    carry = _input(graph, "state", (16, 16), torch.float32)
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    first = _dot(graph, left, left)
    second = _dot(graph, left, left) if terminal else None
    collective = (
        _call(
            graph,
            scan_ops._associative_scan,
            (0, carry, 0, False, False),
            (16, 16),
            torch.float32,
        )
        if kind == "scan"
        else _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (carry, [0], True),
            (1, 16),
            torch.float32,
        )
    )
    if second is None:
        second = _dot(graph, left, left)
    combined = _call(
        graph, torch.ops.aten.add.Tensor, (first, second), (16, 16), torch.float32
    )
    result = _call(
        graph,
        torch.ops.aten.add.Tensor,
        (combined, collective),
        (16, 16),
        torch.float32,
    )
    layout = _layout(_loop_plan(graph, (carry,), (result,)))
    assert layout.regions[0].live_until == (5 if terminal else 3)
    assert (
        layout.regions[0].overlaps_storage(layout.region("chain_1_c")) is not terminal
    )


def _two_carries(*, swap: bool = False) -> ChainedMatmulPlan:
    graph = Graph()
    left = _input(graph, "left_state", (16, 16), torch.float32)
    right = _input(graph, "right_state", (16, 16), torch.float32)
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    first = _dot(graph, _convert(graph, left, torch.bfloat16), operand)
    second = _dot(graph, _convert(graph, right, torch.bfloat16), operand)
    return _loop_plan(
        graph, (left, right), (second, first) if swap else (first, second)
    )


def test_multiple_carries_are_distinct_at_initialization_but_reusable_later() -> None:
    layout = _layout(_two_carries())
    left, right = layout.regions[:2]
    assert left.live_from == right.live_from == -1
    assert not left.overlaps_storage(right)
    assert left.live_until == 1 and right.live_until == 3
    assert left.overlaps_storage(layout.region("chain_0_c"))
    assert right.overlaps_storage(layout.region("chain_1_c"))
    assert layout.allocated_bytes == 2048


def test_cross_carry_outputs_read_all_old_states_before_writeback() -> None:
    graph = Graph()
    left = _input(graph, "left_state", (16, 16), torch.float32)
    right = _input(graph, "right_state", (16, 16), torch.float32)
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    dot = _dot(graph, operand, operand)
    next_left = _call(
        graph, torch.ops.aten.add.Tensor, (dot, right), (16, 16), torch.float32
    )
    next_right = _call(
        graph, torch.ops.aten.add.Tensor, (dot, left), (16, 16), torch.float32
    )
    layout = _layout(_loop_plan(graph, (left, right), (next_left, next_right)))
    assert [item.live_until for item in layout.regions] == [3, 3, 3]
    assert layout.allocated_bytes == 3072


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_periodic_transition_and_zero_trip_exports_require_two_phase_writeback(
    steps: int,
) -> None:
    plan = _two_carries(swap=True)
    assert plan.loop is not None
    plan = replace(plan, loop=replace(plan.loop, end=steps))
    layout = _layout(plan)
    arena = torch.empty(layout.allocated_bytes // 4, dtype=torch.float32)
    views = {
        item.name: arena[item.byte_offset // 4 : item.byte_end // 4].reshape(16, 16)
        for item in layout.regions
    }
    carries = [views[item.name] for item in layout.regions[:2]]
    initial = [torch.full((16, 16), float(i + 1)) for i in range(2)]
    for destination, value in zip(carries, initial, strict=True):
        destination.copy_(value)
    for _ in range(steps):
        # Each stage finishes its old-carry reads before publishing C. An
        # identity RHS makes each mathematical contraction preserve its input.
        for i in range(2):
            registers = carries[i].clone()
            views[f"chain_{i}_c"].copy_(registers)
        # All next values are captured before ANY current/C slot is overwritten.
        next_values = [views["chain_1_c"].clone(), views["chain_0_c"].clone()]
        for destination, value in zip(carries, next_values, strict=True):
            destination.copy_(value)
    for actual, expected in zip(
        carries, initial[::-1] if steps % 2 else initial, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if steps:
        # A per-carry read/write transition would silently destroy the second
        # next value, despite neither output expression reading an old carry.
        views["chain_0_c"].fill_(1)
        views["chain_1_c"].fill_(2)
        carries[0].copy_(views["chain_1_c"])
        carries[1].copy_(views["chain_0_c"])
        assert torch.equal(carries[0], carries[1])


def test_logical_lengths_ignore_source_strides_and_mma_padding() -> None:
    graph = Graph()
    small = _input(graph, "small", (13, 11), torch.float32)
    large = _input(graph, "large", (13, 17), torch.float32)
    large.meta["val"] = torch.empty((17, 13), dtype=torch.float32).t()
    right_small = _input(graph, "right_small", (11, 11), torch.bfloat16)
    right_large = _input(graph, "right_large", (17, 17), torch.bfloat16)
    first = _dot(graph, _convert(graph, small, torch.bfloat16), right_small)
    second = _dot(graph, _convert(graph, large, torch.bfloat16), right_large)
    layout = _layout(_loop_plan(graph, (small, large), (first, second)))
    assert [item.byte_size for item in layout.regions] == [572, 884, 572, 884]
    assert [item.byte_offset for item in layout.regions] == [896, 0, 896, 0]
    assert layout.allocated_bytes == 1536


def test_grouped_results_publish_together_after_all_carry_readers() -> None:
    plan = _two_carries()
    geometry = StageGeometry((16, 16, 16), False)
    plan = replace(
        plan, contraction_groups=(ContractionGroup((0, 1), (geometry, geometry)),)
    )
    layout = _layout(plan)
    assert [item.live_until for item in layout.regions[:2]] == [1, 1]
    assert [item.live_from for item in layout.regions[2:]] == [1, 1]
    assert layout.allocated_bytes == 2048


@pytest.mark.parametrize("intervening_collective", [False, True])
def test_invalid_group_dependencies_fail_closed(intervening_collective: bool) -> None:
    graph = Graph()
    carry = _input(graph, "state", (16, 16), torch.float32)
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    first = _dot(graph, _convert(graph, carry, torch.bfloat16), operand)
    if intervening_collective:
        source = _call(
            graph,
            scan_ops._associative_scan,
            (0, first, 0, False, False),
            (16, 16),
            torch.float32,
        )
    else:
        source = first
    second = _dot(graph, _convert(graph, source, torch.bfloat16), operand)
    plan = _loop_plan(graph, (carry,), (second,))
    geometry = StageGeometry((16, 16, 16), False)
    plan = replace(
        plan, contraction_groups=(ContractionGroup((0, 1), (geometry, geometry)),)
    )
    assert plan_loop_workspace(plan, ((16, 16),)) is None


@pytest.mark.parametrize("valid", [True, False])
def test_only_proven_shape_metadata_stops_carry_read_traversal(valid: bool) -> None:
    graph = Graph()
    carry = _input(graph, "state", (16, 16), torch.float32)
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    first = _dot(graph, _convert(graph, carry, torch.bfloat16), operand)
    shape = graph.call_function(torch.ops.aten.sym_size.int, (carry, 0))
    shape.meta["val"] = 16 if valid else 17
    result = _call(
        graph, torch.ops.aten.mul.Scalar, (first, shape), (16, 16), torch.float32
    )
    plan = _loop_plan(graph, (carry,), (result,))
    if valid:
        layout = _layout(plan)
        assert layout.allocated_bytes == 1024
        assert layout.regions[0].live_until == 1
    else:
        assert plan_loop_workspace(plan, ((16, 16),)) is None


@pytest.mark.parametrize(
    "shapes",
    [
        (),
        ((16, 16),),
        ((16, 16), (16, 17)),
        ((16, 16), (16, 0)),
        ((16, 16), (True, 16)),
        ((16, 16), (16, -1)),
        ((16, 16), (256,)),
        ((16, 16), (16, 16, 1)),
    ],
)
def test_invalid_resolved_carry_shapes_fail_closed(shapes) -> None:
    assert plan_loop_workspace(_two_carries(), shapes) is None


@pytest.mark.parametrize(
    "change",
    [
        "root",
        "warp",
        "direct_output",
        "dtype",
        "stale_graph",
        "region",
        "dot_count",
        "group",
    ],
)
def test_unsupported_or_stale_plan_fails_closed(change: str) -> None:
    plan = _two_carries()
    assert plan.region is not None
    if change == "root":
        plan = replace(plan, loop=None)
    elif change == "warp":
        plan = replace(plan, strategy="warp")
    elif change == "direct_output":
        plan = replace(plan, direct_output=True)
    elif change == "dtype":
        plan.region.carries[0].input.meta["val"] = torch.empty(
            (16, 16), dtype=torch.float16
        )
    elif change == "stale_graph":
        plan.region.graph.placeholder("new_revision")
    elif change == "region":
        plan = replace(plan, region=replace(plan.region))
    elif change == "dot_count":
        plan = replace(plan, shapes=plan.shapes[:1])
    else:
        plan = replace(plan, contraction_groups=())
    assert plan_loop_workspace(plan, ((16, 16), (16, 16))) is None


def test_larger_greedy_layout_falls_back_to_separate_storage() -> None:
    plan = _two_carries()
    with patch.object(
        chained_loop_workspace,
        "_reuse_requests",
        return_value=SharedMemoryLayoutPlan((), 8192),
    ):
        assert plan_loop_workspace(plan, ((16, 16), (16, 16))) is None


def test_planning_is_deterministic_and_preserves_node_revision() -> None:
    plan = _two_carries()
    assert plan.region is not None
    before = tuple(
        (node, node.args, dict(node.kwargs), dict(node.meta))
        for node in plan.region.nodes
    )
    assert _layout(plan) == _layout(plan)
    after = tuple(
        (node, node.args, dict(node.kwargs), dict(node.meta))
        for node in plan.region.nodes
    )
    assert before == after


def test_real_common_loop_plan_uses_emitter_resolved_carry_shapes() -> None:
    original = chained_matmul.plan_chained_matmul
    observed = []

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        assert plan is not None and plan.region is not None and plan.loop is not None
        shapes = tuple(
            chained_matmul._shape(carry.input) for carry in plan.region.carries
        )
        layout = plan_loop_workspace(plan, shapes)
        assert layout is not None
        _assert_safe(layout)
        observed.append(layout)
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
    assert len(observed) == 1


@pytest.mark.parametrize("resident_index", [0, 1])
def test_mixed_resident_carries_keep_read_and_writeback_joins(resident_index):
    original = chained_loop.advance_carries
    observed = []

    def advance(cg, plan, boundaries, scratch):
        assert plan.loop is not None and len(plan.loop.region.carries) == 2
        resident = plan.loop.region.carries[resident_index]
        other = plan.loop.region.carries[1 - resident_index]
        execution = ChainedExecution(
            plan.threads, sync="transition_barrier.arrive_and_wait()"
        )
        lines = original(
            cg,
            plan,
            boundaries,
            scratch,
            execution=execution,
            resident_carries=frozenset((resident.input_index,)),
        )
        tree = ast.parse("\n".join(lines))
        joins = [
            index
            for index, node in enumerate(tree.body)
            if ast.unparse(node) == execution.sync
        ]
        assert len(joins) == 2 and joins[1] == len(tree.body) - 1
        prefix = plan.loop.carry_name(other.input_index)
        for nodes, target in (
            (tree.body[: joins[0]], f"{prefix}_next"),
            (tree.body[joins[0] + 1 : joins[1]], prefix),
        ):
            writes = {
                ast.unparse(node.value)
                for statement in nodes
                for node in ast.walk(statement)
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
            }
            assert writes == {target}
        assert plan.loop.carry_name(resident.input_index) not in {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        observed.append(lines)
        return original(cg, plan, boundaries, scratch)

    with _cpu_codegen(), patch.object(chained_loop, "advance_carries", advance):
        _paired_carries._bind_isolated(_paired_args("cpu", swap=True)).to_code(
            _paired_config()
        )
    assert len(observed) == 1
