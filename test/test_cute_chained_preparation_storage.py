from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from .test_cute_chained_prepared_groups import _candidate
from .test_cute_chained_prepared_image_transfers import _raw_case
from helion._compiler.cute.chained_preparation_cut import PreparationImage
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_storage import plan_preparation_storage
from helion._compiler.cute.chained_prepared_groups import coallocate_prepared_groups
from helion._compiler.cute.chained_prepared_groups import prepared_group_candidates
from helion._compiler.cute.chained_prepared_image_emission import (
    bind_raw_widening_transfers,
)
from helion._compiler.cute.chained_prepared_image_transfers import (
    discover_raw_widening_transfers,
)
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan
from helion._compiler.device_ir import RootGraphInfo
from helion.language import memory_ops


def _bind(plan, frame, shapes):
    transfers = discover_raw_widening_transfers(plan, frame, shapes)
    assert transfers is not None and len(transfers) == 1
    bound = bind_raw_widening_transfers(plan, frame, shapes, transfers)
    assert bound is not None
    return bound


def _safe(result, frame):
    assert len(result.views) == len(frame.buffers)
    assert len({region.name for region in result.layout.regions}) == len(
        result.layout.regions
    )
    for index, left in enumerate(result.layout.regions):
        assert left.byte_offset % 128 == 0
        assert left.byte_end <= result.layout.allocated_bytes
        for right in result.layout.regions[index + 1 :]:
            assert not (left.overlaps_lifetime(right) and left.overlaps_storage(right))
    for view, buffer in zip(result.views, frame.buffers, strict=True):
        assert view.semantic is buffer
        original = frame.layout.region(buffer.name)
        assert (view.live_from, view.live_until) == (
            original.live_from,
            original.live_until,
        )
        owner = result.layout.region(view.owner)
        assert owner.byte_offset <= view.byte_offset
        assert view.byte_offset + view.byte_size <= owner.byte_end


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("masked", [False, True])
def test_raw_physical_image_keeps_original_semantic_frame_and_all_leases(dtype, masked):
    plan, frame, shapes, raw, widening, _ = _raw_case(dtype, masked=masked)
    bound = _bind(plan, frame, shapes)
    before = (frame.layout, frame.actions, frame.buffers, frame.stages, raw.args)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = plan_preparation_storage(
            plan, frame, shapes, raw=bound, capacity_bytes=1 << 20
        )
    assert result is not None and result.matches(plan, frame, shapes)
    _safe(result, frame)
    assert result.revision.frame is frame
    assert before == (
        frame.layout,
        frame.actions,
        frame.buffers,
        frame.stages,
        raw.args,
    )
    image = next(view for view in result.views if view.semantic.node is widening)
    assert image.semantic.dtype == widening.meta["val"].dtype == torch.float32
    assert image.stored_node is raw and image.dtype == dtype
    assert image.shape == (32, 128)
    assert image.byte_size == 8192
    assert frame.layout.region(image.semantic.name).byte_size == 16384
    for view in result.views:
        if view is not image:
            assert view.byte_size == frame.layout.region(view.semantic.name).byte_size
            assert view.stored_node is view.semantic.node
            assert view.dtype == view.semantic.dtype
    assert plan.prepared_widenings is None


def test_default_preserves_exact_layout_and_capacity_including_unused_slack():
    plan, frame, shapes, _, _, _ = _raw_case()
    capacity = frame.layout.allocated_bytes + 128
    frame = replace(frame, layout=replace(frame.layout, allocated_bytes=capacity))
    result = plan_preparation_storage(plan, frame, shapes, capacity_bytes=capacity)
    assert result is not None and result.layout is frame.layout
    assert result.matches(plan, frame, shapes)
    for view, original in zip(result.views, frame.layout.regions, strict=True):
        assert view.owner == original.name
        assert view.byte_offset == original.byte_offset
        assert view.byte_size == original.byte_size
    assert (
        plan_preparation_storage(plan, frame, shapes, capacity_bytes=capacity - 1)
        is None
    )


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.float32])
def test_scalar_frontier_keeps_original_type_and_full_aligned_allocation(dtype):
    plan, frame, shapes, _, _, _ = _raw_case()
    assert plan.region is not None
    graph = plan.region.graph
    with graph.inserting_before(plan.store):
        scalar = _call(graph, torch.ops.aten.scalar_tensor.default, (1,), (), dtype)
    region = collect_contraction_region(RootGraphInfo(0, graph))
    assert region is not None
    plan = replace(plan, region=region)
    buffer = PreparationBuffer("scalar_value", "frontier", scalar, dtype, ())
    allocation = SharedBufferRegion(
        buffer.name,
        frame.layout.allocated_bytes,
        128,
        len(frame.actions) - 2,
        len(frame.actions),
        128,
    )
    action = frame.actions[-2]
    frame = replace(
        frame,
        cut=replace(frame.cut, region=region),
        buffers=(*frame.buffers, buffer),
        layout=SharedMemoryLayoutPlan(
            (*frame.layout.regions, allocation), allocation.byte_end
        ),
        actions=(
            *frame.actions[:-2],
            replace(
                action,
                nodes=(*action.nodes, scalar),
                writes=(*action.writes, buffer.name),
            ),
            frame.actions[-1],
        ),
    )
    shapes[scalar] = ()
    result = plan_preparation_storage(
        plan, frame, shapes, capacity_bytes=frame.layout.allocated_bytes
    )
    assert result is not None and result.matches(plan, frame, shapes)
    view = result.views[-1]
    assert view.stored_node is scalar and view.shape == () and view.dtype == dtype
    assert view.byte_size == 128 and view.byte_offset == allocation.byte_offset


@pytest.mark.parametrize("capacity", [True, -1, 1.0, "65536", 0])
def test_invalid_or_insufficient_capacity_fails_closed(capacity):
    plan, frame, shapes, _, _, _ = _raw_case()
    assert (
        plan_preparation_storage(plan, frame, shapes, capacity_bytes=capacity) is None
    )


def test_compact_capacity_exact_boundary_and_deterministic_result():
    plan, frame, shapes, _, _, _ = _raw_case()
    raw = _bind(plan, frame, shapes)
    result = plan_preparation_storage(
        plan, frame, shapes, raw=raw, capacity_bytes=1 << 20
    )
    assert result is not None
    required = result.layout.allocated_bytes
    assert required < frame.layout.allocated_bytes
    exact = plan_preparation_storage(
        plan, frame, shapes, raw=raw, capacity_bytes=required
    )
    assert exact is not None and exact.layout == result.layout
    assert exact.views == result.views
    assert (
        plan_preparation_storage(
            plan, frame, shapes, raw=raw, capacity_bytes=required - 1
        )
        is None
    )


@pytest.mark.parametrize("change", ["shape", "args", "dtype", "frame", "raw"])
def test_stale_graph_frame_or_transfer_fails_closed(change):
    plan, frame, shapes, node, _, _ = _raw_case()
    raw = _bind(plan, frame, shapes)
    result = plan_preparation_storage(
        plan, frame, shapes, raw=raw, capacity_bytes=1 << 20
    )
    assert result is not None
    if change == "shape":
        shapes = {**shapes, node: (16, 128)}
    elif change == "args":
        node.args = (*node.args[:2], None, node.args[3])
    elif change == "dtype":
        node.meta["val"] = torch.empty((32, 128), dtype=torch.float32)
    elif change == "frame":
        frame = replace(frame, frontier_order=(*frame.frontier_order, node))
    else:
        raw = replace(raw, bindings=())
        assert not replace(result, raw=raw).matches(plan, frame, shapes)
    if change != "raw":
        assert not result.matches(plan, frame, shapes)
    assert (
        plan_preparation_storage(plan, frame, shapes, raw=raw, capacity_bytes=1 << 20)
        is None
    )


@pytest.mark.parametrize("change", ["views", "layout", "capacity", "groups"])
def test_derived_output_witness_rejects_tampering(change):
    plan, frame, shapes, _, _, _ = _raw_case()
    result = plan_preparation_storage(
        plan, frame, shapes, raw=_bind(plan, frame, shapes), capacity_bytes=1 << 20
    )
    assert result is not None
    if change == "views":
        result = replace(result, views=())
    elif change == "layout":
        result = replace(result, layout=replace(result.layout, allocated_bytes=0))
    elif change == "capacity":
        result = replace(result, capacity_bytes=0)
    else:
        _, _, _, groups = _grouped_case(torch.bfloat16, raw_image=False)
        result = replace(result, prepared_groups=groups)
    assert not result.matches(plan, frame, shapes)


def _grouped_case(dtype, *, raw_image):
    plan, frame, recurrence = _candidate(dtype=dtype)
    if raw_image:
        assert plan.region is not None
        graph = plan.region.graph
        output = next(node for node in graph.nodes if node.op == "output")
        old_outputs = output.args[0]
        graph.erase_node(output)
        host = _input(graph, "unrelated_host", (32, 128), dtype)
        rows = _input(graph, "row_indices", (32, 1), torch.int32)
        columns = _input(graph, "column_indices", (1, 128), torch.int32)
        source = _call(
            graph,
            memory_ops.load,
            (host, [rows, columns], None, None),
            (32, 128),
            dtype,
        )
        widening = _convert(graph, source, torch.float32)
        use = _call(
            graph,
            torch.ops.aten.add.Tensor,
            (old_outputs[0], widening),
            (32, 128),
            torch.float32,
        )
        fresh = _loop_plan(
            graph,
            tuple(carry.input for carry in plan.region.carries),
            (use, *old_outputs[1:]),
        )
        plan = replace(plan, region=fresh.region, loop=fresh.loop, store=fresh.store)
        assert plan.region is not None and plan.loop is not None
        event = len(frame.actions) - 1
        buffer = PreparationBuffer(
            f"chain_prepared_{len(frame.cut.images)}",
            "frontier",
            widening,
            torch.float32,
            (32, 128),
        )
        region = SharedBufferRegion(
            buffer.name, frame.layout.allocated_bytes, 16384, event, event + 2, 128
        )
        frame = replace(
            frame,
            cut=replace(
                frame.cut,
                region=plan.region,
                carries=plan.region.carries,
                storage_proof_key=plan.loop.storage_key,
                shared_inputs=(*frame.cut.shared_inputs, host, rows, columns),
                preparation=(*frame.cut.preparation, source, widening),
                recurrence=(
                    *(node for node in frame.cut.recurrence if node is not output),
                    use,
                    fresh.store,
                ),
                images=(
                    *frame.cut.images,
                    PreparationImage(widening, torch.float32, (32, 128), (use,)),
                ),
            ),
            buffers=(*frame.buffers, buffer),
            layout=SharedMemoryLayoutPlan(
                (
                    *tuple(
                        replace(item, live_until=event + 2)
                        if item.live_until == event + 1
                        else item
                        for item in frame.layout.regions
                    ),
                    region,
                ),
                region.byte_end,
            ),
            actions=(
                *frame.actions[:-1],
                PreparationAction(
                    "frontier", event, (widening,), (), None, (), (buffer.name,)
                ),
                replace(frame.actions[-1], event=event + 1),
            ),
        )
    assert plan.region is not None
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    if raw_image:
        recurrence = plan_recurrence_workspace(plan, frame.cut, shapes)
        assert recurrence is not None
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    placed = coallocate_prepared_groups(
        plan, frame, recurrence, candidates, frame_capacity_bytes=1 << 20
    )
    assert placed is not None
    return plan, placed.frame, shapes, placed.groups


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("raw_image", [False, True])
def test_native_group_is_indivisible_with_original_member_offsets(dtype, raw_image):
    plan, frame, shapes, groups = _grouped_case(dtype, raw_image=raw_image)
    result = plan_preparation_storage(
        plan,
        frame,
        shapes,
        raw=_bind(plan, frame, shapes) if raw_image else None,
        prepared_groups=groups,
        capacity_bytes=1 << 20,
    )
    assert result is not None and result.matches(plan, frame, shapes)
    _safe(result, frame)
    group = groups[0]
    owner = result.layout.region(group.candidate.name)
    assert owner.byte_size == group.candidate.byte_size == 10240
    assert (owner.live_from, owner.live_until) == (
        group.candidate.live_from,
        group.candidate.live_until,
    )
    for member in group.candidate.members:
        view = next(
            view for view in result.views if view.semantic.name == member.buffer.name
        )
        assert view.native_group is group
        assert view.owner == owner.name
        assert view.member_byte_offset == member.byte_offset
        assert view.byte_offset == owner.byte_offset + member.byte_offset
    if not raw_image:
        assert owner.byte_offset == group.byte_offset
        assert result.layout.allocated_bytes == frame.layout.allocated_bytes


@pytest.mark.parametrize(
    "change", ["offset", "size", "layout", "duplicate", "lease", "plan"]
)
def test_changed_or_duplicate_native_union_rejects(change):
    plan, frame, shapes, groups = _grouped_case(torch.bfloat16, raw_image=False)
    group = groups[0]
    if change == "plan":
        plan = replace(plan, contraction_groups=())
    elif change == "duplicate":
        groups = (*groups, group)
    elif change == "offset":
        groups = (replace(group, byte_offset=group.byte_offset + 128),)
    else:
        if change == "size":
            candidate = replace(
                group.candidate, byte_size=group.candidate.byte_size - 128
            )
        elif change == "layout":
            candidate = replace(group.candidate, native_layout=("dense",))
        else:
            candidate = replace(
                group.candidate, live_until=group.candidate.live_until - 1
            )
        groups = (replace(group, candidate=candidate),)
    assert (
        plan_preparation_storage(
            plan, frame, shapes, prepared_groups=groups, capacity_bytes=1 << 20
        )
        is None
    )


@pytest.mark.parametrize("kind", ["frontier", "a", "b", "c"])
def test_no_ordinary_allocation_or_early_reservation_is_discounted(kind):
    plan, frame, shapes, _, _, _ = _raw_case()
    buffer = next((item for item in frame.buffers if item.kind == kind), None)
    # The fixture has all four physical allocation categories; keep the
    # inventory explicit so a changed fixture cannot silently narrow coverage.
    assert buffer is not None
    original = frame.layout.region(buffer.name)
    if kind == "frontier":
        changed = replace(original, live_from=0)
        frame = replace(
            frame,
            layout=replace(
                frame.layout,
                regions=tuple(
                    changed if region is original else region
                    for region in frame.layout.regions
                ),
            ),
        )
    raw = _bind(plan, frame, shapes)
    result = plan_preparation_storage(
        plan, frame, shapes, raw=raw, capacity_bytes=1 << 20
    )
    assert result is not None
    view = next(item for item in result.views if item.semantic is buffer)
    assert view.byte_size == original.byte_size
    assert view.live_from == (0 if kind == "frontier" else original.live_from)
    assert view.live_until == original.live_until
