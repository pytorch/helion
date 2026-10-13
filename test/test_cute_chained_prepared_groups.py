from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
import importlib
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
from .test_cute_chained_preparation_frame import _safe
from .test_cute_chained_recurrence_workspace import _cut_for
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_prepared_groups import _native_row_subview
from helion._compiler.cute.chained_prepared_groups import coallocate_prepared_groups
from helion._compiler.cute.chained_prepared_groups import prepared_group_candidates
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.chained_tcgen_stage import StageGeometry


def _candidate(
    widths=(32, 128),
    k=32,
    dtype=torch.bfloat16,
    *,
    transpose=(True, False),
    mode="direct",
):
    graph = Graph()
    common = _input(graph, "common", (128, k), dtype)
    common_t = _call(graph, torch.ops.aten.t.default, (common,), (k, 128), dtype)
    shared = [common, common_t]
    preparation = []
    images = []
    prep_geometries = []
    for index, (width, flipped) in enumerate(zip(widths, transpose, strict=True)):
        shape = (width, k) if flipped else (k, width)
        left = _input(graph, f"left_{index}", (shape[0], 16), dtype)
        right = _input(graph, f"right_{index}", (16, shape[1]), dtype)
        shared.extend((left, right))
        product = _dot(graph, left, right)
        image = _convert(graph, product, dtype)
        preparation.extend((product, image))
        images.append(image)
        prep_geometries.append(StageGeometry((shape[0], shape[1], 16), False))
    carries, outputs, rec_geometries = [], [], []
    for index, (width, flipped) in enumerate(zip(widths, transpose, strict=True)):
        image = images[0] if mode == "duplicate" else images[index]
        if index == 0 and mode == "cast":
            image = _convert(graph, image, dtype)
        elif index == 0 and mode == "incomplete":
            image = _input(graph, "unprepared", tuple(image.meta["val"].shape), dtype)
            shared.append(image)
        shape = (width, 128) if flipped else (128, width)
        carry = _input(graph, f"carry_{index}", shape, torch.float32)
        carries.append(carry)
        outputs.append(
            _dot(graph, image, common_t, carry)
            if flipped
            else _dot(graph, common, image, carry)
        )
        rec_geometries.append(StageGeometry((shape[0], shape[1], k), flipped))
    if mode == "extra_user":
        _call(
            graph,
            torch.ops.aten.neg.default,
            (images[0],),
            tuple(images[0].meta["val"].shape),
            dtype,
        )
    plan = _loop_plan(graph, tuple(carries), tuple(outputs))
    count = len(widths)
    groups = tuple(
        ContractionGroup((index,), (geometry,))
        for index, geometry in enumerate(prep_geometries)
    )
    groups += (ContractionGroup(tuple(range(count, count * 2)), tuple(rec_geometries)),)
    plan = replace(
        plan, contraction_groups=groups, warp_mma_stages=frozenset(range(count))
    )
    cut, shapes = _cut_for(plan, tuple(preparation), tuple(shared))
    frame = plan_preparation_frame(plan, cut, shapes)
    recurrence = plan_recurrence_workspace(plan, cut, shapes)
    assert frame is not None and recurrence is not None
    return plan, frame, recurrence


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "widths,k,transpose",
    [
        ((16, 16), 16, (True, True)),
        ((32, 128), 32, (True, False)),
        ((128, 32), 32, (False, True)),
        ((16, 48, 32), 32, (True, False, True)),
        ((64, 64), 64, (False, False)),
    ],
)
def test_complete_ordered_members_keep_original_nodes_views_and_frame_actions(
    dtype, widths, k, transpose
):
    plan, frame, recurrence = _candidate(widths, k, dtype, transpose=transpose)
    before = (
        frame.layout,
        frame.actions,
        frame.buffers,
        frame.stages,
        frame.frontier_order,
    )
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        candidates = prepared_group_candidates(plan, frame, recurrence)
        assert candidates is not None and len(candidates) == 1
        placement = coallocate_prepared_groups(
            plan, frame, recurrence, candidates, frame_capacity_bytes=1 << 20
        )
    assert placement is not None
    candidate = candidates[0]
    assert candidate.group is recurrence.stages[0].group
    assert candidate.physical_shape == (sum(widths), k)
    assert candidate.dtype == dtype and candidate.byte_size == 2 * sum(widths) * k
    assert placement.frame.layout.allocated_bytes % 128 == 0
    assert placement.groups[0].byte_offset % 128 == 0
    assert (
        tuple(member.row_offset for member in candidate.members)
        == candidate.group.offsets
    )
    for member, width, flipped in zip(
        candidate.members, widths, transpose, strict=True
    ):
        assert member.buffer.node is plan.dots[member.stage].args[member.operand_index]
        assert member.physical_shape == (width, k)
        assert member.logical_modes == ((0, 1) if flipped else (1, 0))
        assert member.byte_offset == member.row_offset * k * 2
        original = frame.layout.region(member.buffer.name)
        current = placement.frame.layout.region(member.buffer.name)
        assert (
            current.byte_offset == placement.groups[0].byte_offset + member.byte_offset
        )
        assert replace(current, byte_offset=original.byte_offset) == original
    assert placement.frame.actions is frame.actions
    assert placement.frame.buffers is frame.buffers
    assert placement.frame.frontier_order is frame.frontier_order
    assert placement.frame.cut is frame.cut
    assert before == (
        frame.layout,
        frame.actions,
        frame.buffers,
        frame.stages,
        frame.frontier_order,
    )
    _safe(placement.frame)
    assert prepared_group_candidates(plan, placement.frame, recurrence) == candidates
    assert plan_prepared_operands(plan, placement.frame, recurrence) is not None
    for stage in placement.frame.stages:
        assert stage.a is placement.frame.layout.region(stage.a.name)
        assert stage.b is placement.frame.layout.region(stage.b.name)
    with pytest.raises(FrozenInstanceError):
        candidate.byte_size = 0  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("mode", ["cast", "incomplete", "extra_user", "duplicate"])
def test_partial_cast_duplicate_or_extra_consumer_retains_ordinary_path(mode):
    plan, frame, recurrence = _candidate((32, 32), transpose=(True, True), mode=mode)
    assert prepared_group_candidates(plan, frame, recurrence) == ()
    placement = coallocate_prepared_groups(
        plan, frame, recurrence, (), frame_capacity_bytes=frame.layout.allocated_bytes
    )
    assert placement is not None and placement.frame is frame


def test_padded_member_is_not_a_complete_native_image():
    plan, frame, recurrence = _candidate((8, 32), transpose=(False, True))
    assert prepared_group_candidates(plan, frame, recurrence) == ()


@pytest.mark.parametrize("k", [48, 80, 96, 128])
def test_multiple_k_panels_cannot_coallocate_dense_member_images(k):
    plan, frame, recurrence = _candidate((32, 32), k, transpose=(True, True))
    assert prepared_group_candidates(plan, frame, recurrence) == ()


@pytest.mark.parametrize(
    "bad",
    [
        "capacity",
        "bool_capacity",
        "negative_capacity",
        "stale_member",
        "stale_offset",
        "duplicate_candidate",
        "foreign_candidate",
        "graph_revision",
        "frame_overlap",
        "frame_publication",
        "stale_ab_handle",
    ],
)
def test_invalid_candidate_placement_or_capacity_fails_closed(bad):
    plan, frame, recurrence = _candidate()
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    capacity = 1 << 20
    if bad == "capacity":
        capacity = 1
    elif bad == "bool_capacity":
        capacity = True
    elif bad == "negative_capacity":
        capacity = -1
    elif bad in ("stale_member", "stale_offset"):
        member = candidates[0].members[0]
        changed = (
            replace(member, stage=99)
            if bad == "stale_member"
            else replace(member, byte_offset=128)
        )
        candidates = (
            replace(candidates[0], members=(changed, *candidates[0].members[1:])),
        )
    elif bad == "duplicate_candidate":
        candidates = (candidates[0], candidates[0])
    elif bad == "foreign_candidate":
        other = _candidate()
        foreign = prepared_group_candidates(*other)
        assert foreign is not None
        candidates = foreign
    elif bad == "graph_revision":
        assert plan.region is not None
        plan.region.graph.placeholder("late")
    elif bad == "frame_overlap":
        first, second = frame.layout.regions[:2]
        frame = replace(
            frame,
            layout=replace(
                frame.layout,
                regions=(
                    first,
                    replace(
                        second, byte_offset=first.byte_offset, live_from=first.live_from
                    ),
                    *frame.layout.regions[2:],
                ),
            ),
        )
    elif bad == "frame_publication":
        frame = replace(
            frame, actions=(replace(frame.actions[0], event=9), *frame.actions[1:])
        )
    elif bad == "stale_ab_handle":
        stage = frame.stages[0]
        frame = replace(
            frame,
            stages=(
                replace(
                    stage, a=replace(stage.a, byte_offset=stage.a.byte_offset + 128)
                ),
                *frame.stages[1:],
            ),
        )
    assert (
        coallocate_prepared_groups(
            plan, frame, recurrence, candidates, frame_capacity_bytes=capacity
        )
        is None
    )


def test_exact_capacity_admission_and_singleton_proofs_need_rebinding():
    plan, frame, recurrence = _candidate()
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None
    first = coallocate_prepared_groups(
        plan, frame, recurrence, candidates, frame_capacity_bytes=1 << 20
    )
    assert first is not None
    exact = coallocate_prepared_groups(
        plan,
        frame,
        recurrence,
        candidates,
        frame_capacity_bytes=first.frame.layout.allocated_bytes,
    )
    assert exact == first
    assert (
        coallocate_prepared_groups(
            plan,
            frame,
            recurrence,
            candidates,
            frame_capacity_bytes=first.frame.layout.allocated_bytes - 1,
        )
        is None
    )


def test_multiple_groups_allow_only_original_ordered_subsets():
    plan, frame, _ = _candidate(
        (16, 16, 16, 16), 16, transpose=(True, True, True, True)
    )
    assert plan.contraction_groups is not None
    grouped = plan.contraction_groups[-1]
    plan = replace(
        plan,
        contraction_groups=(
            *plan.contraction_groups[:-1],
            ContractionGroup(grouped.stages[:2], grouped.geometries[:2]),
            ContractionGroup(grouped.stages[2:], grouped.geometries[2:]),
        ),
    )
    assert plan.region is not None
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    recurrence = plan_recurrence_workspace(plan, frame.cut, shapes)
    assert recurrence is not None
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 2
    for selected in (candidates, candidates[:1], candidates[1:]):
        placement = coallocate_prepared_groups(
            plan, frame, recurrence, selected, frame_capacity_bytes=1 << 20
        )
        assert placement is not None
        assert tuple(item.candidate for item in placement.groups) == selected
        _safe(placement.frame)
    assert (
        coallocate_prepared_groups(
            plan, frame, recurrence, candidates[::-1], frame_capacity_bytes=1 << 20
        )
        is None
    )


def test_frame_stride_retains_required_alignment_on_every_slot():
    plan, frame, recurrence = _candidate((32, 32), 64, transpose=(True, True))
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None
    placement = coallocate_prepared_groups(
        plan, frame, recurrence, candidates, frame_capacity_bytes=1 << 20
    )
    assert placement is not None
    for base in range(0, 1024, 128):
        for slot in range(4):
            absolute = (
                base
                + slot * placement.frame.layout.allocated_bytes
                + placement.groups[0].byte_offset
            )
            assert absolute % 128 == 0


def _layout_expression(statement):
    parsed = ast.parse(statement).body[0]
    assert isinstance(parsed, ast.Assign)
    return compile(ast.Expression(parsed.value), "<native-layout>", "eval")


def _lowered_store_addresses(module):
    """Evaluate the actual static LLVM pointers, not a layout-only model."""
    manager = importlib.import_module("cutlass._mlir.passmanager")
    manager.PassManager.parse(
        "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,"
        "convert-cute-to-core,canonicalize)"
    ).run(module.operation)
    assert module.operation.verify()
    values, stores = {}, []
    for view in module.body.operations:
        op = view.operation
        if op.name == "arith.constant":
            value = op.attributes["value"].value
        elif op.name in ("llvm.inttoptr", "llvm.ptrtoint"):
            value = values[op.operands[0]]
        elif op.name == "llvm.getelementptr":
            index = list(op.attributes["rawConstantIndices"])
            assert len(index) == 1
            value = values[op.operands[0]] + 2 * index[0]
        elif op.name in ("arith.andi", "arith.shrui", "arith.xori"):
            left, right = (values[arg] for arg in op.operands)
            if op.name == "arith.andi":
                value = left & right
            elif op.name == "arith.shrui":
                value = left >> right
            else:
                value = left ^ right
        elif op.name == "llvm.store":
            stores.append(values[op.operands[1]])
            continue
        else:
            assert op.name == "llvm.intr.assume"
            continue
        values[op.results[0]] = value
    return stores


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [16, 32, 64])
def test_actual_tensor_store_lowering_proves_byte_address_subviews(dtype, k):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    plan, frame, recurrence = _candidate((16, 48), k, dtype, transpose=(False, True))
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    candidate = candidates[0]
    ir = importlib.import_module("cutlass._mlir.ir")
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    env = {"cutlass": cutlass, "cute": cute, "tcgen05": tcgen05}
    expected = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            full = eval(_layout_expression(candidate.native_layout[0]), env)
            bits = (k // 8).bit_length() - 1
            for base in range(1024, 2048, 128):
                pointer = cute.make_ptr(
                    element_type, base, cute.AddressSpace.smem, assumed_align=128
                )
                whole = cute.make_tensor(
                    cute.recast_ptr(pointer, full.inner, dtype=element_type), full.outer
                )
                for member in candidate.members:
                    native = eval(_layout_expression(member.native_layout[0]), env)
                    logical = (
                        native.outer
                        if member.logical_modes == (0, 1)
                        else cute.select(native.outer, mode=[1, 0])
                    )
                    member_pointer = cute.make_ptr(
                        element_type,
                        base + member.byte_offset,
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    image = cute.make_tensor(
                        cute.recast_ptr(
                            member_pointer, native.inner, dtype=element_type
                        ),
                        logical,
                    )
                    for row, column in (
                        (0, 0),
                        (1, 0),
                        (7, k - 1),
                        (member.physical_shape[0] - 1, k - 1),
                    ):
                        coords = (
                            (row, column)
                            if member.logical_modes == (0, 1)
                            else (column, row)
                        )
                        whole[member.row_offset + row, column] = element_type(1)
                        image[coords] = element_type(1)
                        address = base + 2 * ((member.row_offset + row) * k + column)
                        swizzled = address ^ ((address & (((1 << bits) - 1) << 7)) >> 3)
                        expected.extend((swizzled, swizzled))
                        assert (
                            base + member.byte_offset
                            <= swizzled
                            < base
                            + member.byte_offset
                            + 2 * member.physical_shape[0] * k
                        )
        assert _lowered_store_addresses(module) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_admitted_geometry_has_exact_outer_map_and_swizzle_period(dtype):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    from helion._compiler.cute.chained_tcgen05 import _layout

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype_name = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    env = {"cutlass": cutlass, "cute": cute, "tcgen05": tcgen05}
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            for k in (16, 32, 64):
                period = 16 * k
                bits = (k // 8).bit_length() - 1
                native = eval(
                    _layout_expression(_layout("value", (16, k), 1, dtype_name)[0]),
                    env,
                )
                for rows in range(16, 257, 16):
                    native = eval(
                        _layout_expression(
                            _layout("value", (rows, k), 1, dtype_name)[0]
                        ),
                        env,
                    )
                    outer = cute.coalesce(native.outer, target_profile=(1, 1))
                    assert outer.shape == (rows, k)
                    assert outer.stride == (k, 1)
                    for offset in range(0, rows, 16):
                        for count in range(16, rows - offset + 1, 16):
                            assert _native_row_subview((rows, k), (count, k), offset)
                            assert 2 * offset * k % 128 == 2 * count * k % 128 == 0
                # Exhaust the complete absolute-address phase period, rather
                # than checking only selected row/column examples. The outer
                # map above lifts this to every row partition and arbitrarily
                # high common pointer addresses. These are BYTE addresses,
                # as independently checked against actual tensor lowering.
                swizzled = cute.make_composed_layout(
                    native.inner, 0, cute.make_layout(2 * period, stride=1)
                )
                for address in range(2 * period):
                    assert int(swizzled(address)) == (
                        address ^ (((address >> 7) & ((1 << bits) - 1)) << 4)
                    )
                for base in range(0, period, 128):
                    assert {
                        int(swizzled(base + index)) for index in range(period)
                    } == set(range(base, base + period))
                assert not _native_row_subview((32, k), (8, k), 8)
                assert not _native_row_subview((32, k), (16, k), 8)
                assert not _native_row_subview((32, k), (16, k), -16)
                assert not _native_row_subview((32, k), (16, k), 32)
            for k in (48, 80, 96, 128):
                native = eval(
                    _layout_expression(_layout("value", (32, k), 1, dtype_name)[0]),
                    env,
                )
                member = eval(
                    _layout_expression(_layout("value", (16, k), 1, dtype_name)[0]),
                    env,
                )
                atom_k = min(128, (2 * k) & -(2 * k)) // 2
                assert int(native.outer((0, atom_k))) != int(member.outer((0, atom_k)))
                assert not _native_row_subview((32, k), (16, k), 0)
        assert module.operation.verify()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "widths,k,transpose",
    [
        ((32, 128), 32, (True, False)),
        ((16, 48, 32), 16, (False, True, False)),
        ((64, 64), 64, (False, True)),
    ],
)
def test_actual_cute_member_views_cover_exact_group_and_verify_copy_mlir(
    dtype, widths, k, transpose
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    plan, frame, recurrence = _candidate(widths, k, dtype, transpose=transpose)
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    candidate = candidates[0]
    ir = importlib.import_module("cutlass._mlir.ir")
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    env = {"cutlass": cutlass, "cute": cute, "tcgen05": tcgen05}
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            full = eval(_layout_expression(candidate.native_layout[0]), env)
            swizzled = cute.make_composed_layout(
                full.inner, 0, cute.make_layout(32 * k + candidate.byte_size, stride=1)
            )
            for base_bytes in range(0, 16 * k, 128):
                seen = set()
                for member in candidate.members:
                    native = eval(_layout_expression(member.native_layout[0]), env)
                    logical = (
                        native.outer
                        if member.logical_modes == (0, 1)
                        else cute.select(native.outer, mode=[1, 0])
                    )
                    rows, reduction = member.physical_shape
                    for row in range(rows):
                        for column in range(reduction):
                            coords = (
                                (row, column)
                                if member.logical_modes == (0, 1)
                                else (column, row)
                            )
                            address = int(
                                swizzled(
                                    base_bytes
                                    + member.byte_offset
                                    + 2 * int(logical(coords))
                                )
                            )
                            assert address == int(
                                swizzled(
                                    base_bytes
                                    + 2
                                    * int(full.outer((member.row_offset + row, column)))
                                )
                            )
                            assert address not in seen
                            seen.add(address)
                    pointer = cute.make_ptr(
                        element_type,
                        base_bytes + member.byte_offset,
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    tensor = cute.make_tensor(
                        cute.recast_ptr(pointer, native.inner, dtype=element_type),
                        native.outer,
                    )
                    source = cute.local_tile(
                        tensor, (1, 8), (rows - 1, reduction // 8 - 1)
                    )
                    values = cute.make_rmem_tensor(source.shape, element_type)
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            element_type,
                            num_bits_per_copy=128,
                        ),
                        source,
                        values,
                    )
                assert seen == set(
                    range(base_bytes, base_bytes + candidate.byte_size, 2)
                )
        assert module.operation.verify()


def test_actual_kda_group_preserves_49920_frame_and_all_original_lifetimes():
    kernel, _ = _kda_fixture()
    shape = (1, 16384, 32, 128)
    q, k, v, gate = (torch.empty(shape, dtype=torch.bfloat16) for _ in range(4))
    state = torch.empty((16, 32, 128, 128), dtype=torch.float32)
    lengths = torch.tensor([525, 1523, 781, 1267] * 4)
    args = (
        q,
        k,
        v,
        gate,
        torch.empty(shape[:-1], dtype=torch.bfloat16),
        torch.empty((32,), dtype=torch.float32),
        torch.empty((32, 128), dtype=torch.float32),
        state,
        torch.empty_like(v),
        torch.empty_like(state),
        torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))),
        128**-0.5,
        -5 * 1.4426950408889634,
    )
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
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    candidate = candidates[0]
    assert candidate.group.stages == (13, 14)
    assert [
        (member.buffer.name, member.byte_offset, member.logical_modes)
        for member in candidate.members
    ] == [("chain_prepared_4", 0, (0, 1)), ("chain_prepared_5", 2048, (1, 0))]
    assert (candidate.byte_size, candidate.live_from, candidate.live_until) == (
        10240,
        20,
        31,
    )
    placement = coallocate_prepared_groups(
        plan, frame, recurrence, candidates, frame_capacity_bytes=49920
    )
    assert placement is not None
    assert (
        placement.frame.layout.allocated_bytes
        == placement.reservation_peak_bytes
        == 49920
    )
    assert placement.groups[0].byte_offset == 21248
    assert (
        placement.frame.actions is frame.actions
        and placement.frame.frontier_order is frame.frontier_order
    )
    for before, after in zip(
        frame.layout.regions, placement.frame.layout.regions, strict=True
    ):
        assert replace(after, byte_offset=before.byte_offset) == before
    old_singletons = plan_prepared_operands(plan, frame, recurrence)
    new_singletons = plan_prepared_operands(plan, placement.frame, recurrence)
    assert old_singletons is not None and new_singletons is not None
    assert (
        [item.stage for item in old_singletons]
        == [item.stage for item in new_singletons]
        == [10, 11, 12]
    )
    assert any(
        old.region != new.region
        for old, new in zip(old_singletons, new_singletons, strict=True)
    )
    _safe(placement.frame)
