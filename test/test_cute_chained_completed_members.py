from __future__ import annotations

from dataclasses import replace
import inspect
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_completed_members import plan_completed_member_map
from helion._compiler.cute.chained_completed_members import plan_completed_member_store
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_store_expression import lower_store_point
from helion._compiler.cute.chained_store_expression import materialized_store_read
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
import helion.language as hl
from helion.language import memory_ops


def _case(dtype=torch.bfloat16, output_dtype=torch.bfloat16, *, offset=0, width=32):
    graph = Graph()
    a = _input(graph, "a", (width, 16), dtype)
    b = _input(graph, "b", (16, 128), dtype)
    c = _input(graph, "c", (128, 16), dtype)
    value = _dot(graph, a, b)
    other = _dot(graph, c, b)
    converted = _convert(graph, value, output_dtype)
    output = _input(graph, "destination", (width, 128), output_dtype)
    row = _input(graph, "indirect_rows", (width,), torch.int32)
    column = _input(graph, "columns", (128,), torch.int32)
    mask = _input(graph, "row_mask", (width, 1), torch.bool)
    store = graph.call_function(
        memory_ops.store, (output, [row, column], converted, mask)
    )
    plan = _plan(graph, (other,))
    stages = (0, 1) if offset == 0 else (1, 0)
    geometries = (
        (StageGeometry((width, 128, 16), True), StageGeometry((128, 128, 16), False))
        if offset == 0
        else (
            StageGeometry((128, 128, 16), False),
            StageGeometry((width, 128, 16), True),
        )
    )
    group = ContractionGroup(stages, geometries)
    plan = replace(
        plan, store=store, contraction_groups=(group,), strategy="tcgen05_tmem"
    )
    shapes = {
        n: tuple(n.meta["val"].shape)
        for n in graph.nodes
        if isinstance(n.meta.get("val"), torch.Tensor)
    }
    return plan, group, store, shapes, value


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_unrelated_candidate_preserves_original_nodes_and_typed_snapshots(
    dtype, output_dtype
):
    plan, group, store, shapes, value = _case(dtype, output_dtype)
    result = plan_completed_member_store(plan, group, 0, store, shapes)
    assert result is not None and result.matches(plan, shapes)
    assert result.node is value and result.store is store
    assert result.descendants[-1] is store.args[2]
    assert result.native.same_store_order
    assert result.native.logical_shape == (32, 128)


@pytest.mark.parametrize(
    "full,offset,width",
    [(144, 0, 16), (160, 0, 32), (160, 16, 48), (256, 128, 64), (256, 128, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_actual_full_native_partition_preserves_member_owner_order(
    full, offset, width, dtype
):
    before = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = plan_completed_member_map(
            (128, full), (width, 128), offset, width, True, dtype
        )
    assert result is not None
    assert result.same_store_order
    for thread, slots in enumerate(result.selected_slots):
        assert [result.coordinates[thread][i] for i in slots] == [
            (thread, offset + j) for j in range(width)
        ]
        assert [result.tmem_offsets[thread][i] for i in slots] == [
            (thread << 16) + offset + j for j in range(width)
        ]
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize(
    "shape,logical,offset,width,transpose,dtype",
    [
        ((128, 160), (128, 32), 0, 32, False, torch.bfloat16),
        ((64, 160), (32, 64), 0, 32, True, torch.bfloat16),
        ((128, 160), (32, 128), True, 32, True, torch.bfloat16),
        ((128, 160), (32, 128), 0, 32, 1, torch.bfloat16),
        ((128, 160), (32, 128), 0, 32, True, torch.float32),
        ((128, 160), (33, 128), 0, 32, True, torch.bfloat16),
        ((128, 160), (32, 127), 0, 32, True, torch.bfloat16),
        ((128, 160), (0, 128), 0, 32, True, torch.bfloat16),
    ],
)
def test_wrong_owner_order_or_geometry_declines(
    shape, logical, offset, width, transpose, dtype
):
    assert (
        plan_completed_member_map(shape, logical, offset, width, transpose, dtype)
        is None
    )


@pytest.mark.parametrize(
    "change", ["extra_user", "gather", "index", "mask", "carry", "shape", "order"]
)
def test_exclusive_value_use_required(change):
    plan, group, store, shapes, value = _case()
    assert plan.region is not None
    selectors = cast("list[Node]", store.args[1])
    graph = value.graph
    if change in ("extra_user", "gather"):
        with graph.inserting_before(store):
            extra = graph.call_function(
                torch.ops.aten.gather.default
                if change == "gather"
                else torch.ops.aten.neg.default,
                (value, 0, selectors[0]) if change == "gather" else (value,),
            )
            extra.meta["val"] = torch.empty_like(value.meta["val"])
            shapes[extra] = shapes[value]
        plan = replace(plan, region=replace(plan.region, nodes=tuple(graph.nodes)))
    elif change == "index":
        store.args = (
            store.args[0],
            [value, selectors[1]],
            store.args[2],
            store.args[3],
        )
    elif change == "mask":
        store.args = (*store.args[:3], value)
    elif change == "carry":
        from helion._compiler.cute.contraction_region import ContractionCarry

        plan = replace(
            plan,
            region=replace(
                plan.region, carries=(ContractionCarry(0, 0, value, value),)
            ),
        )
    elif change == "shape":
        shapes[value] = (31, 128)
    else:
        group = ContractionGroup(
            (0, 1), (StageGeometry((32, 128, 16), False), group.geometries[1])
        )
        plan = replace(plan, contraction_groups=(group,))
    assert plan_completed_member_store(plan, group, 0, store, shapes) is None


@pytest.mark.parametrize(
    "change", ["kwargs", "dtype", "group", "native", "record", "shapes", "target"]
)
def test_deep_same_object_mutations_reject(change):
    plan, group, store, shapes, value = _case()
    selected = plan_completed_member_store(plan, group, 0, store, shapes)
    assert selected is not None
    if change == "kwargs":
        value.kwargs = {"_extra_deps": [cast("list[Node]", store.args[1])[0]]}
    elif change == "dtype":
        value.meta["val"] = value.meta["val"].to(torch.float16)
    elif change == "group":
        object.__setattr__(group.geometries[0], "transpose", False)
    elif change == "native":
        # Replace the record, not the memoized native geometry shared by tests.
        selected = replace(selected, native=replace(selected.native, selected_slots=()))
    elif change == "record":
        selected = replace(selected, descendants=())
    elif change == "shapes":
        shapes[value] = (16, 128)
    else:
        store.target = torch.ops.aten.clone.default
    assert not selected.matches(plan, shapes)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_sequence(
    a,
    b,
    c,
    d,
    initial,
    indices,
    mask,
    output_dtype: hl.constexpr,
    masked: hl.constexpr,
):
    steps = a.shape[0]
    dtype: torch.dtype = output_dtype  # pyrefly: ignore [bad-assignment]
    history = torch.empty((steps, 32, 128), dtype=dtype, device=a.device)
    final = torch.empty_like(initial)
    for _task in hl.tile(1, block_size=1):
        initial_rows, initial_columns = hl.arange(128), hl.arange(128)
        state = initial[initial_rows, initial_columns]
        for step in hl.tile(steps, block_size=1):
            rows, inner, columns = hl.arange(32), hl.arange(32), hl.arange(128)
            prepared = hl.dot(
                (a[step.id, rows, inner] + c[step.id, rows, inner]).to(a.dtype),
                b[step.id, inner, columns],
                out_dtype=torch.float32,
            ).to(a.dtype)
            snapshot = state.to(a.dtype)
            result = hl.dot(prepared, snapshot, out_dtype=torch.float32)
            carry_rows = hl.arange(128)
            state = hl.dot(
                snapshot.T,
                d[step.id, carry_rows, columns],
                acc=state,
                out_dtype=torch.float32,
            )
            target_rows = indices[rows]
            valid = mask[rows]
            if masked:
                hl.store(
                    history,
                    [step.id, target_rows, columns],
                    torch.tanh(result + 0.125).to(dtype),
                    extra_mask=valid[:, None],
                )
            else:
                history[step.id, target_rows, columns] = torch.tanh(result + 0.125).to(
                    dtype
                )
        final[initial_rows, initial_columns] = state
    return history, final


def _source(dtype, output_dtype, *, delegate=False, check=None, steps=1, masked=True):
    args = (
        torch.empty((steps, 32, 32), dtype=dtype),
        torch.empty((steps, 32, 128), dtype=dtype),
        torch.empty((steps, 32, 32), dtype=dtype),
        torch.empty((steps, 128, 128), dtype=dtype),
        torch.empty((128, 128)),
        torch.arange(32, dtype=torch.int32),
        torch.ones((32,), dtype=torch.bool),
        output_dtype,
        masked,
    )
    config = helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
    )
    original = chain._emit_store
    captured = []

    def replacement(
        cg,
        plan,
        store,
        boundaries,
        staged,
        scans,
        prefix="chain_store",
        *,
        execution=None,
    ):
        shape = chain._shape(store.args[2])
        coords = (f"{prefix} // {shape[1]}", f"{prefix} % {shape[1]}")
        if not delegate:
            return original(
                cg, plan, store, boundaries, staged, scans, prefix, execution=execution
            )
        point = lower_store_point(
            cg,
            plan,
            store,
            boundaries,
            staged,
            scans,
            coordinates=coords,
            coordinate_names=(prefix,),
        )
        if check is not None:
            check(cg, plan, store, boundaries, staged, scans, prefix, point)
        captured.append(point)
        execution = execution or ChainedExecution(plan.threads)
        m, n = point.shape
        return [
            f"for {prefix}_step in cutlass.range_constexpr({(m * n + execution.threads - 1) // execution.threads}):",
            f"    {prefix} = {execution.thread} + {prefix}_step * {execution.threads}",
            f"    if {prefix} < {m * n}:",
            chain._indent(point.lines, 8),
            f"        if {' and '.join(point.bounds)}:",
            f"            {point.target}[{', '.join(point.indices)}] = {point.value}",
        ]

    with _cpu_codegen(), patch.object(chain, "_emit_store", replacement):
        bound = _store_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            dict(inspect.signature(_store_sequence.fn).bind(*args).arguments)
        ):
            source = bound.to_code(config)
    return source, captured


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_delegation_keeps_original_full_source_exact(dtype, output_dtype):
    before, _ = _source(dtype, output_dtype)
    after, points = _source(dtype, output_dtype, delegate=True)
    assert before == after and len(points) == 2
    assert "cutlass.Boolean" in " ".join((*points[0].lines, *points[0].bounds))
    assert any("Int32(0)" in line for line in points[0].lines)


@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_zero_trip_and_unmasked_store_delegation(steps, masked):
    before, _ = _source(torch.float16, torch.float32, steps=steps, masked=masked)
    after, points = _source(
        torch.float16, torch.float32, delegate=True, steps=steps, masked=masked
    )
    assert before == after and len(points) == 2


def test_native_padding_and_no_reusable_mutable_cache_record():
    first = plan_completed_member_map(
        (128, 160), (31, 128), 16, 32, True, torch.bfloat16
    )
    assert first is not None and all(len(slots) == 31 for slots in first.selected_slots)
    object.__setattr__(first, "selected_slots", ())
    second = plan_completed_member_map(
        (128, 160), (31, 128), 16, 32, True, torch.bfloat16
    )
    assert second is not None and second is not first
    assert len(second.selected_slots) == 128 and second.same_store_order


def test_original_guarded_read_and_cross_coordinate_rejection():
    checked = []

    def check(cg, plan, store, boundaries, staged, scans, prefix, point):
        if store not in plan.region.stores:
            return
        shapes = {
            n: chain._shape(n)
            for n in plan.region.nodes
            if isinstance(n.meta.get("val"), torch.Tensor)
        }
        group = plan.contraction_groups[-1]
        selected = plan_completed_member_store(
            plan, group, group.stages[0], store, shapes
        )
        assert selected is not None and selected.matches(plan, shapes)
        replacement = materialized_store_read(
            plan,
            boundaries,
            selected.node,
            point.coordinates,
            "original_register[index]",
        )
        result = lower_store_point(
            cg,
            plan,
            store,
            boundaries,
            staged,
            scans,
            coordinates=point.coordinates,
            coordinate_names=(prefix,),
            completed_read=replacement,
        )
        text = "\n".join(result.lines)
        assert "original_register[index]" in text and "cutlass.Float32(0)" in text
        assert boundaries[selected.node] not in text
        bad = materialized_store_read(
            plan,
            boundaries,
            selected.node,
            point.coordinates[::-1],
            "original_register[index]",
        )
        with pytest.raises(chain._UnsupportedChain, match="cross-coordinate"):
            lower_store_point(
                cg,
                plan,
                store,
                boundaries,
                staged,
                scans,
                coordinates=point.coordinates,
                coordinate_names=(prefix,),
                completed_read=bad,
            )
        with pytest.raises(chain._UnsupportedChain, match="changed materialized"):
            lower_store_point(
                cg,
                plan,
                store,
                {**boundaries, selected.node: "other"},
                staged,
                scans,
                coordinates=point.coordinates,
                coordinate_names=(prefix,),
                completed_read=replacement,
            )
        checked.append(selected)

    _source(torch.bfloat16, torch.float16, delegate=True, check=check)
    assert len(checked) == 1
