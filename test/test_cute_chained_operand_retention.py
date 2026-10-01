from __future__ import annotations

from dataclasses import replace
import importlib
from typing import Any
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
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_operand_retention import discover_operand_retention
from helion._compiler.cute.chained_operand_retention import plan_operand_retention_frame
from helion._compiler.cute.chained_preparation_cut import PreparationCut
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_frame import PreparationFrame
from helion._compiler.cute.chained_preparation_frame import PreparationStage
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedBufferRequest
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan


def _case(dtype=torch.bfloat16, *, transpose=False, duplicate=False, escaping=False):
    graph = Graph()
    rows, columns = 32, 128
    raw = tuple(
        _input(graph, f"raw_{i}", (rows, columns), torch.float32) for i in range(3)
    )
    products = tuple(
        _call(
            graph,
            torch.ops.aten.mul.Tensor,
            (node, 2.0),
            (rows, columns),
            torch.float32,
        )
        for node in raw
    )
    values = tuple(_convert(graph, node, dtype) for node in products)
    if transpose:
        common = _call(
            graph, torch.ops.aten.t.default, (values[0],), (columns, rows), dtype
        )
        dots = tuple(_dot(graph, node, common) for node in values[1:])
    else:
        rhs = tuple(
            _call(graph, torch.ops.aten.t.default, (node,), (columns, rows), dtype)
            for node in values[1:]
        )
        dots = tuple(_dot(graph, values[0], node) for node in rhs)
    last = _convert(graph, values[2], torch.float32)
    if escaping:
        last = _call(
            graph,
            torch.ops.aten.add.Tensor,
            (last, raw[2]),
            (rows, columns),
            torch.float32,
        )
    frontier = (
        _convert(graph, values[0], torch.float32),
        _convert(graph, products[1], dtype) if duplicate else values[1],
        last,
    )
    plan = _plan(graph, (*dots, *frontier))
    group = ContractionGroup(
        (0, 1), (StageGeometry((rows, rows, columns), transpose),) * 2
    )
    plan = replace(
        plan,
        strategy="tcgen05_tmem",
        contraction_groups=(group,),
        warp_mma_stages=frozenset({0}),
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
    regions, buffers = [], []
    offset = 0

    def alloc(name, kind, node, value_dtype, shape, first, last):
        nonlocal offset
        size = (
            (torch.empty(shape, dtype=value_dtype).numel() * value_dtype.itemsize + 127)
            // 128
            * 128
        )
        region = SharedBufferRegion(name, offset, size, first, last, 128)
        offset += size
        regions.append(region)
        buffers.append(PreparationBuffer(name, kind, node, value_dtype, shape))
        return region

    for i, node in enumerate(raw):
        alloc(f"raw_{i}", "leaf", node, torch.float32, (rows, columns), 0, 6)
    a = alloc("first_a", "a", None, dtype, (rows, columns), 1, 3)
    b = alloc("first_b", "b", None, dtype, (2 * rows, columns), 1, 3)
    for i, node in enumerate(dots):
        alloc(f"result_{i}", "c", node, torch.float32, (rows, rows), 2, 3)
    for i, node in enumerate(frontier):
        alloc(
            f"frontier_{i}",
            "frontier",
            node,
            node.meta["val"].dtype,
            (rows, columns),
            3 + i,
            7,
        )
    actions = (
        PreparationAction(
            "leaf", 0, raw, (), None, (), tuple(f"raw_{i}" for i in range(3))
        ),
        PreparationAction(
            "fill",
            1,
            dots,
            group.stages,
            0,
            tuple(f"raw_{i}" for i in range(3)),
            (a.name, b.name),
        ),
        PreparationAction(
            "mma", 2, dots, group.stages, 0, (a.name, b.name), ("result_0", "result_1")
        ),
        *(
            PreparationAction(
                "frontier", 3 + i, (node,), (), None, (f"raw_{i}",), (f"frontier_{i}",)
            )
            for i, node in enumerate(frontier)
        ),
        PreparationAction(
            "ready", 6, frontier, (), None, tuple(f"frontier_{i}" for i in range(3)), ()
        ),
    )
    frame = PreparationFrame(
        cut,
        SharedMemoryLayoutPlan(tuple(regions), offset),
        tuple(buffers),
        actions,
        (PreparationStage(group, (rows, 2 * rows, columns), a, b),),
        frontier,
        offset,
    )
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    return plan, frame, shapes, values


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
def test_exact_typed_nodes_and_full_group_aliases(dtype, transpose):
    plan, frame, shapes, values = _case(dtype, transpose=transpose)
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    assert tuple(item.node for item in candidates) == values
    assert tuple(item.row_offset for item in candidates) == (0, 0, 32)
    assert tuple(item.source_stage for item in candidates) == (0, 0, 1)
    assert tuple(item.consumer_events for item in candidates) == ((3,), (4,), (5,))
    assert all(
        item.publication_event == 3 and item.major_mode == "k" for item in candidates
    )
    assert candidates[2].full_shape == (64, 128)
    assert candidates[2].alias_lines("full_b", "retained") == (
        "retained = cute.domain_offset((32, 0), full_b)",
    )
    retained = plan_operand_retention_frame(
        plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
    )
    assert retained is not None and retained.matches(plan, frame, shapes)
    assert tuple(retained.frame.actions[i].reads for i in (3, 4, 5)) == (
        ("first_a",),
        ("first_b",),
        ("first_b",),
    )
    assert retained.frame.layout.region("first_a").live_until == 4
    assert retained.frame.layout.region("first_b").live_until == 6
    assert all(
        retained.frame.layout.region(f"raw_{i}").live_until == 2 for i in range(3)
    )
    assert all(
        retained.frame.layout.region(region.name).live_from == region.live_from
        for region in frame.layout.regions
    )
    assert retained.leaf_reads == tuple((frame.buffers[i].node, (1,)) for i in range(3))
    for stage in retained.frame.stages:
        assert stage.a == retained.frame.layout.region(stage.a.name)
        assert stage.b == retained.frame.layout.region(stage.b.name)


def test_equal_expression_is_not_the_same_ssa_value():
    plan, frame, shapes, values = _case(duplicate=True)
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    assert tuple(item.node for item in candidates) == (values[0], values[2])
    retained = plan_operand_retention_frame(
        plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
    )
    assert retained is not None
    assert retained.frame.actions[4].reads == ("raw_1",)
    assert retained.frame.layout.region("raw_1").live_until == 5


def test_other_users_keep_the_original_leaf_alive():
    plan, frame, shapes, _ = _case(escaping=True)
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    retained = plan_operand_retention_frame(
        plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
    )
    assert retained is not None
    assert retained.frame.actions[5].reads == ("first_b", "raw_2")
    assert retained.frame.layout.region("raw_2").live_until == 6


def test_reservations_extend_but_never_shorten_original_starts():
    plan, frame, shapes, _ = _case()
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    original = frame.layout.region("frontier_2")
    reservation = SharedBufferRequest(
        original.name, original.byte_size, original.alignment, 1, 7
    )
    retained = plan_operand_retention_frame(
        plan,
        frame,
        candidates,
        shapes,
        reservations=(reservation,),
        capacity_bytes=frame.layout.allocated_bytes,
    )
    assert retained is not None
    assert retained.frame.layout.region(original.name).live_from == 1
    assert retained.matches(plan, frame, shapes, reservations=(reservation,))
    assert not retained.matches(plan, frame, shapes)
    assert (
        plan_operand_retention_frame(
            plan,
            frame,
            candidates,
            shapes,
            reservations=(replace(reservation, byte_size=128),),
            capacity_bytes=frame.layout.allocated_bytes,
        )
        is None
    )


def test_stale_graph_frame_shapes_and_candidate_fail_closed():
    plan, frame, shapes, values = _case()
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    result = plan_operand_retention_frame(
        plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
    )
    assert result is not None
    assert not result.matches(plan, replace(frame), shapes)
    wrong_shapes = {**shapes, values[0]: (16, 128)}
    assert not result.matches(plan, frame, wrong_shapes)
    assert (
        plan_operand_retention_frame(
            plan,
            frame,
            (replace(candidates[0], row_offset=1),),
            shapes,
            capacity_bytes=frame.layout.allocated_bytes,
        )
        is None
    )
    values[0].name = "not_a_matching_rule"
    assert result.matches(plan, frame, shapes)
    old = values[0].args
    values[0].args = (values[1], torch.bfloat16)
    assert not result.matches(plan, frame, shapes)
    assert (
        plan_operand_retention_frame(
            plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
        )
        is None
    )
    values[0].args = old


def test_derived_output_witness_rejects_replaced_candidates_frame_and_reads():
    plan, frame, shapes, _ = _case()
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    result = plan_operand_retention_frame(
        plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
    )
    assert result is not None and result.matches(plan, frame, shapes)
    changed_frames = (
        frame,
        replace(result.frame, actions=result.frame.actions[:-1]),
        replace(result.frame, stages=()),
        replace(result.frame, buffers=()),
        replace(
            result.frame,
            layout=replace(result.frame.layout, allocated_bytes=128),
        ),
    )
    for changed in changed_frames:
        assert not replace(result, frame=changed).matches(plan, frame, shapes)
    for changed in (
        replace(result, candidates=()),
        replace(result, candidates=(replace(candidates[0], row_offset=1),)),
        replace(result, leaf_reads=()),
        replace(result, owner_offsets=()),
    ):
        assert not changed.matches(plan, frame, shapes)


@pytest.mark.parametrize("capacity", [True, 0, -1, 127])
def test_failed_capacity_has_no_discounted_fallback(capacity):
    plan, frame, shapes, _ = _case()
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    assert (
        plan_operand_retention_frame(
            plan, frame, candidates, shapes, capacity_bytes=capacity
        )
        is None
    )
    assert frame.layout.region("raw_0").live_until == 6


def test_foreign_region_and_padded_or_underallocated_owners_reject():
    plan, frame, shapes, _ = _case()
    foreign_plan, _, _, _ = _case()
    assert discover_operand_retention(foreign_plan, frame, shapes) is None
    stage = frame.stages[0]
    assert (
        discover_operand_retention(
            plan, replace(frame, stages=(replace(stage, shape=(16, 64, 128)),)), shapes
        )
        is None
    )
    smaller = replace(stage.b, byte_size=stage.b.byte_size - 128)
    layout = replace(
        frame.layout,
        regions=tuple(
            smaller if region.name == smaller.name else region
            for region in frame.layout.regions
        ),
    )
    changed = replace(frame, layout=layout, stages=(replace(stage, b=smaller),))
    assert discover_operand_retention(plan, changed, shapes) is None


def test_original_shape_dtype_masks_remain_recorded_not_inferred_finite():
    plan, frame, shapes, values = _case()
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    assert all(
        item.logical_shape == shapes[item.node]
        and item.dtype == item.node.meta["val"].dtype
        for item in candidates
    )
    # This pure component makes no finite-value or domain-empty assertion.
    # Original pointwise operators and casts remain the exact graph objects.
    original = values[0].args[0]
    assert isinstance(original, Node)
    original.target = torch.ops.aten.div.Tensor
    renewed = discover_operand_retention(plan, frame, shapes)
    assert renewed is not None
    assert renewed[0].node is values[0]
    assert (
        plan_operand_retention_frame(
            plan, frame, candidates, shapes, capacity_bytes=frame.layout.allocated_bytes
        )
        is None
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("exchange", [False, True])
def test_actual_cute_scalar_reads_preserve_full_layout_and_both_k_panels(
    dtype, exchange
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    passmanager = importlib.import_module("cutlass._mlir.passmanager")

    plan, frame, shapes, _ = _case(dtype)
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    # Alias construction is mechanical, not a proof token. Exercise both
    # logical orientations independently of later expression admission.
    candidate = replace(candidates[2], logical_modes=(1, 0) if exchange else (0, 1))
    scalar = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            output = cute.make_tensor(
                cute.make_ptr(
                    scalar, 1 << 24, cute.AddressSpace.smem, assumed_align=128
                ),
                cute.make_layout(128),
            )
            index = 0
            for base in (0, 128, 65536 + 896):
                layout = cute.tile_to_shape(
                    tcgen05.make_smem_layout_atom(
                        tcgen05.SmemLayoutAtomKind.K_SW128, scalar
                    ),
                    (64, 128),
                    order=(0, 1),
                )
                full = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            scalar, base, cute.AddressSpace.smem, assumed_align=128
                        ),
                        layout.inner,
                        dtype=scalar,
                    ),
                    layout.outer,
                )
                namespace: dict[str, Any] = {"cute": cute, "full": full}
                exec("\n".join(candidate.alias_lines("full", "alias")), namespace)
                alias = namespace["alias"]
                for row in (0, 31):
                    for column in (0, 63, 64, 127):
                        output[index] = (
                            alias[column, row] if exchange else alias[row, column]
                        )
                        output[index + 1] = full[row + 32, column]
                        index += 2
        assert module.operation.verify()
        passmanager.PassManager.parse(
            "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        values, stores = {}, []
        for view in module.body.operations:
            op = view.operation
            if op.name == "arith.constant":
                result = op.attributes["value"].value
            elif op.name in ("llvm.inttoptr", "llvm.ptrtoint"):
                result = values[op.operands[0]]
            elif op.name == "llvm.getelementptr":
                indices = list(op.attributes["rawConstantIndices"])
                assert len(indices) == 1
                result = values[op.operands[0]] + 2 * indices[0]
            elif op.name in ("arith.andi", "arith.shrui", "arith.xori"):
                left, right = (values[operand] for operand in op.operands)
                result = (
                    left & right
                    if op.name == "arith.andi"
                    else left >> right
                    if op.name == "arith.shrui"
                    else left ^ right
                )
            elif op.name == "llvm.load":
                result = ("loaded", values[op.operands[0]])
            elif op.name == "llvm.store":
                stores.append((values[op.operands[1]], values[op.operands[0]]))
                continue
            else:
                assert op.name == "llvm.intr.assume", op.name
                continue
            values[op.results[0]] = result
        assert len(stores) == index
        assert all(stores[i][1] == stores[i + 1][1] for i in range(0, index, 2))
        assert len({value for _, value in stores}) == index // 2


def test_actual_kda_captures_shared_ssa_images_and_rebuilds_reads_cpu():
    from .test_cute_chained_loop_tmem_transport import _source
    from .test_cute_chained_preparation_cut import _kda_fixture
    from .test_cute_chained_preparation_pipeline import _config
    from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
    from helion._compiler.cute.chained_frontier_groups import plan_frontier_group
    from helion._compiler.cute.chained_matmul import _shape
    from helion._compiler.cute.chained_register_islands import plan_register_islands

    kernel, args = _kda_fixture()
    config = _config(16, pipeline=True)
    config.config.update(
        block_sizes=[128],
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=4,
        cute_chained_preparation_cohorts=3,
        cute_chained_register_islands=True,
        cute_chained_vector_group=True,
    )
    original = pipeline_module._prepare
    observed = []

    def capture(cg, plan, pipeline, *extra):
        assert plan.region is not None and plan.contraction_groups is not None
        frame = pipeline.frame
        shapes = {
            node: _shape(node)
            for node in plan.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        candidates = discover_operand_retention(plan, frame, shapes)
        assert candidates is not None
        selected = tuple(
            item for item in candidates if item.group == frame.stages[0].group
        )
        assert len(selected) == 3
        assert all(
            item.node.target is torch.ops.prims.convert_element_type.default
            and item.dtype == torch.bfloat16
            for item in selected
        )
        assert tuple(item.row_offset for item in selected) == (0, 0, 32)
        reservations = []
        entries = (
            frozenset(entry.node for entry in plan.pointwise_cache.entries)
            if plan.pointwise_cache is not None
            else frozenset()
        )
        islands = plan_register_islands(
            plan.region,
            plan.contraction_groups,
            shapes,
            fast_math=True,
            entry_boundaries=entries,
        )
        for island in islands:
            first = next(
                action.event
                for action in frame.actions
                if action.kind == "fill" and action.stages == island.groups[0].stages
            )
            for node in island.exports:
                buffer = next(
                    buffer
                    for buffer in frame.buffers
                    if buffer.kind == "c" and buffer.node is node
                )
                region = frame.layout.region(buffer.name)
                reservations.append(
                    SharedBufferRequest(
                        region.name,
                        region.byte_size,
                        region.alignment,
                        first,
                        region.live_until,
                    )
                )
        # Reserve every adjacent same-shaped frontier prefix even if a later
        # expression proof chooses ordinary publication instead of grouping.
        stop = 0
        for action in frame.actions:
            if action.event < stop or action.kind != "frontier":
                continue
            group = plan_frontier_group(frame, action.event)
            if group is None:
                continue
            stop = group.stop_event
            for buffer in group.buffers:
                region = frame.layout.region(buffer.name)
                reservations.append(
                    SharedBufferRequest(
                        region.name,
                        region.byte_size,
                        region.alignment,
                        group.first_event,
                        region.live_until,
                    )
                )
        retained = plan_operand_retention_frame(
            plan,
            frame,
            selected,
            shapes,
            prepared_groups=pipeline.prepared_groups,
            reservations=tuple(reservations),
            capacity_bytes=232448 // pipeline.slots // 128 * 128,
        )
        assert retained is not None
        assert retained.frame.layout.allocated_bytes == 65536
        assert len(retained.frame.layout.regions) == len(frame.layout.regions)
        assert retained.matches(
            plan,
            frame,
            shapes,
            prepared_groups=pipeline.prepared_groups,
            reservations=tuple(reservations),
        )
        assert all(item.consumer_events for item in selected)
        assert tuple(
            retained.frame.layout.region(item.owner).live_until for item in selected
        ) == (30, 32, 32)
        assert pipeline.prepared_groups
        damaged = replace(
            pipeline.prepared_groups[0],
            byte_offset=pipeline.prepared_groups[0].byte_offset + 128,
        )
        assert (
            plan_operand_retention_frame(
                plan,
                frame,
                selected,
                shapes,
                prepared_groups=(damaged, *pipeline.prepared_groups[1:]),
                reservations=tuple(reservations),
                capacity_bytes=232448,
            )
            is None
        )
        assert not retained.matches(
            plan,
            frame,
            shapes,
            prepared_groups=(damaged, *pipeline.prepared_groups[1:]),
            reservations=tuple(reservations),
        )
        assert all(
            retained.frame.layout.region(region.name).live_from <= region.live_from
            for region in frame.layout.regions
        )
        observed.append(retained)
        return original(cg, plan, pipeline, *extra)

    with patch.object(pipeline_module, "_prepare", capture):
        source = _source(kernel, args, config)
    assert len(observed) == 1 and "chain_leaf_2" in source
