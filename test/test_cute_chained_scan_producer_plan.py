from __future__ import annotations

from dataclasses import replace
import importlib

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_preparation_cut import PreparationCut
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_frame import PreparationFrame
from helion._compiler.cute.chained_preparation_frame import PreparationStage
from helion._compiler.cute.chained_preparation_pipeline import PreparationPipeline
from helion._compiler.cute.chained_recurrence_workspace import RecurrenceWorkspace
from helion._compiler.cute.chained_scan_producer import plan_scan_producer
from helion._compiler.cute.chained_scratch_layout import xor_swizzle
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan
from helion.language import scan_ops
from helion.language import view_ops


def _case(
    width=64, position=17, dtype=torch.bfloat16, *, dependent=False, escape=False
):
    graph = Graph()
    shape = (32, width)
    raw = _input(graph, "polynomial_input", shape, torch.float32)
    weights = _input(graph, "independent_weights", shape, torch.float32)
    source = _call(graph, torch.ops.aten.mul.Tensor, (raw, raw), shape, torch.float32)
    scan = _call(
        graph,
        scan_ops._associative_scan,
        (1, source, 0, False, False),
        shape,
        torch.float32,
    )
    iota = _call(graph, torch.ops.prims.iota.default, (32,), (32,), torch.int32)
    iota.kwargs = {"start": 0, "step": 1, "dtype": torch.int32}
    equals = _call(graph, torch.ops.aten.eq.Scalar, (iota, position), (32,), torch.bool)
    mask = _call(
        graph, view_ops.subscript, (equals, [slice(None), None]), (32, 1), torch.bool
    )
    zero = _call(graph, torch.ops.aten.scalar_tensor.default, (0.0,), (), torch.float32)
    selected = _call(
        graph, torch.ops.aten.where.self, (mask, scan, zero), shape, torch.float32
    )
    endpoint = _call(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (selected, [0], False),
        (width,),
        torch.float32,
    )
    reduced = _call(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (scan if dependent else weights, [1], True),
        (32, 1),
        torch.float32,
    )
    combined = _call(
        graph, torch.ops.aten.add.Tensor, (scan, reduced), shape, torch.float32
    )
    nonlinear = _call(
        graph, torch.ops.aten.sigmoid.default, (combined,), shape, torch.float32
    )
    left = _convert(graph, nonlinear, dtype)
    right = _convert(graph, weights, dtype)
    transpose = _call(graph, torch.ops.aten.t.default, (right,), (width, 32), dtype)
    dot = _dot(graph, left, transpose)
    plan = _plan(graph, (dot, endpoint))
    assert plan.region is not None
    group = ContractionGroup((0,), (StageGeometry((32, 32, width), False),))
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
        (raw, weights),
        tuple(node for node in plan.region.nodes if node not in (raw, weights)),
        (),
        (),
        "polynomial",
    )
    buffers, regions = [], []
    offset = 0

    def allocate(name, kind, node, value_dtype, logical_shape, begin, end):
        nonlocal offset
        size = (
            (
                torch.empty(logical_shape, dtype=value_dtype).numel()
                * value_dtype.itemsize
                + 127
            )
            // 128
            * 128
        )
        region = SharedBufferRegion(name, offset, size, begin, end, 128)
        buffers.append(PreparationBuffer(name, kind, node, value_dtype, logical_shape))
        regions.append(region)
        offset += size
        return region

    allocate("raw", "leaf", raw, torch.float32, shape, 0, 2)
    allocate("weights", "leaf", weights, torch.float32, shape, 0, 5)
    allocate("prefix", "collective", scan, torch.float32, shape, 1, 7 if escape else 5)
    allocate("endpoint", "collective", endpoint, torch.float32, (width,), 2, 7)
    allocate("row_sum", "collective", reduced, torch.float32, (32, 1), 3, 5)
    a = allocate("a", "a", None, dtype, shape, 4, 6)
    b = allocate("b", "b", None, dtype, shape, 4, 6)
    allocate("c", "c", dot, torch.float32, (32, 32), 5, 7)
    actions = (
        PreparationAction("leaf", 0, (raw, weights), (), None, (), ("raw", "weights")),
        PreparationAction("collective", 1, (scan,), (), 0, ("raw",), ("prefix",)),
        PreparationAction(
            "collective", 2, (endpoint,), (), 0, ("prefix",), ("endpoint",)
        ),
        PreparationAction(
            "collective",
            3,
            (reduced,),
            (),
            0,
            ("prefix",) if dependent else ("weights",),
            ("row_sum",),
        ),
        PreparationAction(
            "fill", 4, (dot,), (0,), 0, ("prefix", "row_sum", "weights"), ("a", "b")
        ),
        PreparationAction("mma", 5, (dot,), (0,), 0, ("a", "b"), ("c",)),
        PreparationAction(
            "ready",
            6,
            (),
            (),
            None,
            ("c", "endpoint", "prefix") if escape else ("c", "endpoint"),
            (),
        ),
    )
    layout = SharedMemoryLayoutPlan(tuple(regions), offset)
    frame = PreparationFrame(
        cut,
        layout,
        tuple(buffers),
        actions,
        (PreparationStage(group, (32, 32, width), a, b),),
        (),
        offset,
    )
    recurrence = RecurrenceWorkspace(
        cut, SharedMemoryLayoutPlan((), 0), (), (), 0, 0, 0, 0
    )
    pipeline = PreparationPipeline(frame, recurrence, None, 2, 128, 128, 384)
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    return plan, pipeline, shapes


@pytest.mark.parametrize("width", (32, 64, 128, 256))
@pytest.mark.parametrize("position", (0, 7, 17, 31))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_unrelated_polynomial_graph_preserves_exact_nodes_and_typed_views(
    width, position, dtype
):
    plan, pipeline, shapes = _case(width, position, dtype)
    selected = plan_scan_producer(plan, pipeline, shapes, scratch_mode="xor")
    assert selected is not None
    assert selected.revision.frame is pipeline.frame
    assert selected.scan.node is pipeline.frame.actions[1].nodes[0]
    assert selected.residual_rows == (position,)
    assert selected.matches(plan, pipeline, shapes)
    assert selected.stage.a == pipeline.frame.stages[0].a
    residual = next(region for region in selected.regions if ":row:" in region.name)
    full = pipeline.frame.layout.region("prefix")
    assert (residual.byte_offset, residual.byte_size) == (
        full.byte_offset + position * width * 4,
        width * 4,
    )
    assert (
        selected.phases[3]
        < selected.phases[1]
        == selected.phases[4]
        < selected.phases[2]
    )


@pytest.mark.parametrize("dependent,escape", ((True, False), (False, True)))
def test_dependent_prelude_or_later_prefix_reader_declines(dependent, escape):
    plan, pipeline, shapes = _case(dependent=dependent, escape=escape)
    assert plan_scan_producer(plan, pipeline, shapes, scratch_mode="xor") is None


def test_late_alias_reservation_and_mutable_graph_cannot_retain_authority():
    plan, pipeline, shapes = _case()
    candidate = plan_scan_producer(plan, pipeline, shapes, scratch_mode="xor")
    assert candidate is not None
    altered = dict(shapes)
    altered[candidate.scan.node] = (32, 32)
    assert not candidate.matches(plan, pipeline, altered)
    candidate.scan.source.args = (candidate.scan.source.args[0], 0.5)
    assert not candidate.matches(plan, pipeline, shapes)


@pytest.mark.parametrize("threads", (32, 64, 256, 384))
def test_unproved_copy_participant_geometry_declines(threads):
    plan, pipeline, shapes = _case()
    assert (
        plan_scan_producer(
            plan,
            replace(pipeline, preparation_threads=threads),
            shapes,
            scratch_mode="xor",
        )
        is None
    )


@pytest.mark.parametrize("width", (32, 64, 128, 256))
def test_actual_cute_residual_rows_keep_full_view_mapping(width):
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    parameters = xor_swizzle((32, width))
    assert parameters is not None
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            full = cute.make_composed_layout(
                cute.make_swizzle(*parameters),
                0,
                cute.make_layout((32, width), stride=(width, 1)),
            )
            for row in (0, 7, 17, 31):
                addresses = [int(full((row, column))) for column in range(width)]
                assert set(addresses) == set(range(row * width, (row + 1) * width))
                # The residual record describes bytes of the ORIGINAL view;
                # using a new row-zero swizzle would change these addresses.
                if row:
                    dense_rebase = [
                        row * width + int(full((0, column))) for column in range(width)
                    ]
                    assert addresses != dense_rebase
