from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
import importlib
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
from .test_cute_chained_recurrence_workspace import _cut_for
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.chained_tcgen05 import _layout
from helion._compiler.cute.chained_tcgen_stage import StageGeometry


def _candidate(rows=16, k=16, dtype=torch.bfloat16, *, mode="direct"):
    graph = Graph()
    left = _input(graph, "left", (rows, 16), dtype)
    right = _input(graph, "right", (16, k), dtype)
    rhs = _input(graph, "rhs", (k, 128), dtype)
    state = _input(graph, "state", (rows, 128), torch.float32)
    product = _dot(graph, left, right)
    image = _convert(graph, product, dtype)
    source = _convert(graph, image, dtype) if mode == "cast" else image
    first = _dot(graph, source, rhs, state)
    second = (
        _dot(graph, image, rhs, state if mode == "grouped" else first)
        if mode in ("repeat", "incompatible", "grouped")
        else None
    )
    plan = _loop_plan(graph, (state,), (first if second is None else second,))
    prep = StageGeometry((rows, k, 16), False)
    native = StageGeometry((rows, 128, k), True)
    groups = (ContractionGroup((0,), (prep,)),)
    if mode == "grouped":
        groups += (ContractionGroup((1, 2), (native, native)),)
    else:
        groups += (ContractionGroup((1,), (native,)),)
        if second is not None:
            other = (
                replace(native, transpose=False) if mode == "incompatible" else native
            )
            groups += (ContractionGroup((2,), (other,)),)
    plan = replace(plan, contraction_groups=groups, warp_mma_stages=frozenset({0}))
    cut, shapes = _cut_for(plan, (product, image), (left, right, rhs))
    frame = plan_preparation_frame(plan, cut, shapes)
    recurrence = plan_recurrence_workspace(plan, cut, shapes)
    assert frame is not None and recurrence is not None
    return plan, frame, recurrence


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,k", [(16, 16), (32, 32), (32, 48), (32, 128), (64, 64)])
def test_complete_frontier_preserves_identity_storage_and_common_layout(rows, k, dtype):
    plan, frame, recurrence = _candidate(rows, k, dtype)
    before = (frame.layout, frame.actions, recurrence.layout, recurrence.stages)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = plan_prepared_operands(plan, frame, recurrence)
    assert result is not None and len(result) == 1
    operand = result[0]
    assert operand.stage == 1 and operand.role == "b" and operand.operand_index == 0
    assert operand.buffer.node is plan.dots[1].args[0]
    assert operand.buffer.dtype == dtype
    assert operand.physical_shape == (rows, k)
    assert operand.region is frame.layout.region(operand.buffer.name)
    assert operand.region.byte_size == rows * k * 2
    name = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    assert operand.native_layout == tuple(
        _layout(operand.buffer.name, (rows, k), 1, name)
    )
    assert before == (frame.layout, frame.actions, recurrence.layout, recurrence.stages)
    with pytest.raises(FrozenInstanceError):
        operand.stage = 0  # pyrefly: ignore [read-only]


def test_multiple_compatible_consumers_share_one_unchanged_image():
    plan, frame, recurrence = _candidate(mode="repeat")
    result = plan_prepared_operands(plan, frame, recurrence)
    assert result is not None and [item.stage for item in result] == [1, 2]
    assert result[0].buffer is result[1].buffer
    assert result[0].region is result[1].region
    assert result[0].native_layout == result[1].native_layout


def test_recollected_region_wrappers_preserve_exact_graph_revision():
    plan, frame, recurrence = _candidate()
    assert plan.region is not None and plan.loop is not None
    copied_cut = replace(frame.cut, region=replace(frame.cut.region))
    copied_plan = replace(
        plan,
        region=replace(plan.region),
        loop=replace(plan.loop, region=replace(plan.loop.region)),
    )
    assert plan_prepared_operands(
        copied_plan,
        replace(frame, cut=copied_cut),
        replace(recurrence, cut=copied_cut),
    ) == plan_prepared_operands(plan, frame, recurrence)


@pytest.mark.parametrize("mode", ["cast", "grouped", "incompatible"])
def test_no_cast_bypass_partial_group_or_incompatible_view(mode):
    plan, frame, recurrence = _candidate(mode=mode)
    assert plan_prepared_operands(plan, frame, recurrence) == ()


@pytest.mark.parametrize(
    "bad",
    [
        "foreign_cut",
        "graph_revision",
        "dot_kwargs",
        "dot_target",
        "dot_operand_dtype",
        "dot_operand_domain",
        "shape",
        "dtype",
        "alignment",
        "offset",
        "small_allocation",
        "short_lifetime",
        "wrong_publication",
        "binding",
        "missing_stage",
        "stage_geometry",
        "stage_event",
        "empty_group",
    ],
)
def test_inconsistent_plan_or_typed_lifetime_fails_closed(bad):
    plan, frame, recurrence = _candidate()
    assert plan.region is not None
    image = next(buffer for buffer in frame.buffers if buffer.kind == "frontier")
    assert image.node is not None
    if bad == "foreign_cut":
        recurrence = replace(recurrence, cut=_candidate()[1].cut)
    elif bad == "graph_revision":
        plan.region.graph.placeholder("late")
    elif bad == "dot_kwargs":
        plan.dots[1].kwargs = {"unexpected": image.node}
    elif bad == "dot_target":
        plan.dots[1].target = torch.ops.aten.add.Tensor
    elif bad in ("dot_operand_dtype", "dot_operand_domain"):
        value = image.node.meta["val"]
        image.node.meta["val"] = torch.empty(
            value.shape if bad == "dot_operand_dtype" else (32, 16),
            dtype=torch.float32 if bad == "dot_operand_dtype" else value.dtype,
            device="meta",
        )
    elif bad in ("shape", "dtype"):
        changed = (
            replace(image, shape=(16, 32))
            if bad == "shape"
            else replace(image, dtype=torch.float32)
        )
        frame = replace(
            frame,
            buffers=tuple(changed if item is image else item for item in frame.buffers),
        )
    elif bad in ("alignment", "offset", "small_allocation", "short_lifetime"):
        allocation = frame.layout.region(image.name)
        if bad == "alignment":
            changed = replace(allocation, alignment=16)
        elif bad == "offset":
            changed = replace(allocation, byte_offset=allocation.byte_offset + 2)
        elif bad == "small_allocation":
            changed = replace(allocation, byte_size=128)
        else:
            changed = replace(allocation, live_until=allocation.live_until - 1)
        frame = replace(
            frame,
            layout=replace(
                frame.layout,
                regions=tuple(
                    changed if item is allocation else item
                    for item in frame.layout.regions
                ),
            ),
        )
    elif bad == "wrong_publication":
        frame = replace(
            frame,
            actions=tuple(
                replace(action, nodes=()) if action.kind == "frontier" else action
                for action in frame.actions
            ),
        )
    elif bad == "binding":
        recurrence = replace(
            recurrence,
            bindings=tuple(
                (node, "different") if node is image.node else (node, name)
                for node, name in recurrence.bindings
            ),
        )
    elif bad == "missing_stage":
        recurrence = replace(recurrence, stages=())
    elif bad == "stage_geometry":
        stage = recurrence.stages[0]
        recurrence = replace(
            recurrence,
            stages=(
                replace(
                    stage,
                    group=replace(
                        stage.group,
                        geometries=(
                            replace(stage.group.geometries[0], transpose=False),
                        ),
                    ),
                ),
            ),
        )
    elif bad == "stage_event":
        recurrence = replace(
            recurrence, stages=(replace(recurrence.stages[0], read_event=0),)
        )
    elif bad == "empty_group":
        assert plan.contraction_groups is not None
        plan = replace(
            plan,
            contraction_groups=(ContractionGroup((), ()), *plan.contraction_groups),
        )
    assert plan_prepared_operands(plan, frame, recurrence) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,k", [(16, 16), (32, 32), (32, 48), (32, 128)])
def test_actual_cute_native_layout_is_bijective_and_copy_legal(rows, k, dtype):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    plan, frame, recurrence = _candidate(rows, k, dtype)
    result = plan_prepared_operands(plan, frame, recurrence)
    assert result is not None
    operand = result[0]
    statement = ast.parse(operand.native_layout[0]).body[0]
    assert isinstance(statement, ast.Assign)
    expression = compile(ast.Expression(statement.value), "<native-layout>", "eval")
    ir = importlib.import_module("cutlass._mlir.ir")
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layout = eval(
                expression, {"cute": cute, "cutlass": cutlass, "tcgen05": tcgen05}
            )
            assert {
                int(layout((row, col))) for row in range(rows) for col in range(k)
            } == set(range(rows * k))
            # Every allowed frame base phase preserves the same allocation and
            # aligned eight-element copies, including a non-power-of-two K.
            for base_bytes in range(0, 1024, 128):
                base = base_bytes // 2
                shifted = cute.make_composed_layout(layout.inner, base, layout.outer)
                assert {
                    int(shifted((row, col))) for row in range(rows) for col in range(k)
                } == set(range(base, base + rows * k))
                for row in range(rows):
                    for col in range(0, k, 8):
                        offsets = [int(shifted((row, col + j))) for j in range(8)]
                        assert offsets[0] % 8 == 0
                        assert offsets == list(range(offsets[0], offsets[0] + 8))
            element_type = (
                cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
            )
            pointer = cute.make_ptr(
                element_type, 0, cute.AddressSpace.smem, assumed_align=128
            )
            image = cute.make_tensor(
                cute.recast_ptr(pointer, layout.inner, dtype=element_type), layout.outer
            )
            consumer = cute.make_tensor(
                cute.recast_ptr(pointer, layout.inner, dtype=element_type), layout.outer
            )
            assert str(image) == str(consumer)
            source = cute.local_tile(image, (1, 8), (rows - 1, k // 8 - 1))
            values = cute.make_rmem_tensor(source.shape, element_type)
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), element_type, num_bits_per_copy=128
                ),
                source,
                values,
            )
        assert module.operation.verify()


def test_actual_admitted_graph_selects_only_complete_singleton_b_images():
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
    frame = plan_preparation_frame(plan, cut, shapes)
    recurrence = plan_recurrence_workspace(plan, cut, shapes)
    assert frame is not None and recurrence is not None
    result = plan_prepared_operands(plan, frame, recurrence)
    assert result is not None
    assert [(item.stage, item.buffer.name, item.physical_shape) for item in result] == [
        (10, "chain_prepared_2", (32, 128)),
        (11, "chain_prepared_3", (32, 32)),
        (12, "chain_prepared_6", (32, 128)),
    ]
    assert frame.layout.allocated_bytes == 49920
