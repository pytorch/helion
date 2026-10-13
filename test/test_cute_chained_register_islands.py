from __future__ import annotations

from dataclasses import replace
from typing import Any
from typing import cast

import pytest
import torch
from torch.fx import Graph
from torch.fx import GraphModule
from torch.fx import Interpreter

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_register_islands import plan_register_islands
from helion._compiler.cute.chained_register_islands import register_island_matches
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.device_ir import RootGraphInfo
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language import tile_ops
from helion.language.matmul_ops import dot


def _fixture(dtype=torch.float16, *, nonlinear=None, escaping=False, permuted=False):
    graph = Graph()
    left = _input(graph, "image", (32, 32), dtype)
    right = _input(graph, "weights", (32, 32), dtype)
    iota = graph.call_function(
        torch.ops.prims.iota.default,
        (32,),
        {
            "start": 0,
            "step": 1,
            "dtype": torch.int64,
            "device": torch.device("cpu"),
            "requires_grad": False,
        },
    )
    iota.meta["val"] = torch.empty((32,), dtype=torch.int64)
    block = graph.call_function(
        torch.ops.aten.div.Tensor_mode, (iota, 16), {"rounding_mode": "floor"}
    )
    block.meta["val"] = torch.empty((32,), dtype=torch.int64)
    rows = _call(
        graph, torch.ops.aten.unsqueeze.default, (block, 1), (32, 1), torch.int64
    )
    columns = _call(
        graph, torch.ops.aten.unsqueeze.default, (block, 0), (1, 32), torch.int64
    )
    mask = _call(
        graph,
        torch.ops.aten.ne.Tensor if permuted else torch.ops.aten.eq.Tensor,
        (rows, columns),
        (32, 32),
        torch.bool,
    )
    zero = graph.call_function(
        torch.ops.aten.scalar_tensor.default, (0,), {"dtype": dtype}
    )
    zero.meta["val"] = torch.empty((), dtype=dtype)
    a = _call(graph, torch.ops.aten.where.self, (mask, left, zero), (32, 32), dtype)
    b = _call(graph, torch.ops.aten.where.self, (mask, right, zero), (32, 32), dtype)
    first = _dot(graph, a, b)
    value = first
    if nonlinear is not None:
        value = _call(graph, nonlinear, (value,), (32, 32), torch.float32)
    snapshot = _convert(graph, value, dtype)
    if permuted:
        # A distinct original typed image; unlike an entry buffer, one cached
        # SSA value is initially limited to one tile per owning component.
        b = _call(
            graph, torch.ops.aten.where.self, (mask, right, zero), (32, 32), dtype
        )
    second = _dot(graph, snapshot, b)
    snapshot2 = _convert(graph, second, dtype)
    third = _dot(graph, snapshot, snapshot2)
    plan = _plan(graph, (third, snapshot) if escaping else (third,))
    groups = tuple(
        ContractionGroup((stage,), (StageGeometry((32, 32, 32), stage == 1),))
        for stage in range(3)
    )
    shapes = {
        node: tuple(node.meta["val"].shape)
        for node in graph.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    return (
        plan,
        groups,
        shapes,
        frozenset((left, right)),
        (first, snapshot, second, snapshot2, third),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_generic_block_dag_retains_original_versions_and_all_use_exports(dtype):
    plan, groups, shapes, entries, values = _fixture(dtype, escaping=True)
    assert plan.region is not None
    (island,) = plan_register_islands(
        plan.region, groups, shapes, fast_math=True, entry_boundaries=entries
    )
    assert island.groups == groups
    assert len(island.components) == 2
    assert [
        tuple(issue.stage for issue in component.issues)
        for component in island.components
    ] == [(0, 1, 2)] * 2
    assert [component.issues[0].origins for component in island.components] == [
        (0, 0, 0),
        (16, 16, 16),
    ]
    assert island.exports == (values[0], values[-1])
    assert set(island.entries) == entries
    records = {record.node: record for record in island.values}
    assert all(node in records for node in values)
    assert records[values[0]].dtype is torch.float32
    assert records[values[1]].dtype is dtype
    assert values[1] in records[values[0]].consumers
    assert values[2] in records[values[1]].consumers
    assert values[-1] in records[values[1]].consumers
    assert records[values[1]].last_use > records[values[-1]].defined_at
    assert all(record.node in island.domain_nodes for record in island.values)
    assert register_island_matches(
        island, plan.region, groups, shapes, entry_boundaries=entries
    )


def test_false_is_immediate_noop_and_policy_is_strict_bool():
    opaque = cast("Any", None)
    assert plan_register_islands(opaque, opaque, opaque, fast_math=False) == ()
    for policy in (None, 0, 1, "true", [], object()):
        with pytest.raises(TypeError, match="explicit bool"):
            plan_register_islands(opaque, opaque, opaque, fast_math=cast("Any", policy))


@pytest.mark.parametrize(
    "target",
    [
        torch.ops.aten.exp.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.reciprocal.default,
    ],
)
def test_nonlinear_zero_images_do_not_inherit_support(target):
    plan, groups, shapes, entries, _ = _fixture(nonlinear=target)
    assert plan.region is not None
    assert (
        plan_register_islands(
            plan.region, groups, shapes, fast_math=True, entry_boundaries=entries
        )
        == ()
    )


def test_names_have_no_semantic_role_and_revision_tracks_original_edges():
    plan, groups, shapes, entries, values = _fixture()
    assert plan.region is not None
    (island,) = plan_register_islands(
        plan.region, groups, shapes, fast_math=True, entry_boundaries=entries
    )
    for index, node in enumerate(plan.region.nodes):
        node.name = f"unrelated_{index}"
    assert register_island_matches(
        island, plan.region, groups, shapes, entry_boundaries=entries
    )
    values[1].args = (values[2], torch.float16)
    assert not register_island_matches(
        island, plan.region, groups, shapes, entry_boundaries=entries
    )


def test_complete_ordered_groups_and_matching_geometry_are_required():
    plan, groups, shapes, entries, _ = _fixture()
    assert plan.region is not None
    for invalid in (
        groups[1:],
        tuple(reversed(groups)),
        (replace(groups[0], geometries=()), *groups[1:]),
    ):
        assert (
            plan_register_islands(
                plan.region, invalid, shapes, fast_math=True, entry_boundaries=entries
            )
            == ()
        )
    mismatched = (
        replace(groups[0], geometries=(StageGeometry((16, 32, 32), False),)),
        *groups[1:],
    )
    result = plan_register_islands(
        plan.region, mismatched, shapes, fast_math=True, entry_boundaries=entries
    )
    # A later independent span may still be considered, but never the malformed group.
    assert all(groups[0] not in candidate.groups for candidate in result)


def test_unrelated_permutation_dag_has_distinct_mode_origins():
    plan, groups, shapes, entries, _ = _fixture(permuted=True)
    assert plan.region is not None
    (island,) = plan_register_islands(
        plan.region, groups, shapes, fast_math=True, entry_boundaries=entries
    )
    assert [
        tuple(issue.origins for issue in component.issues)
        for component in island.components
    ] == [
        ((0, 0, 16), (0, 16, 0), (0, 16, 0)),
        ((16, 16, 0), (16, 0, 16), (16, 0, 16)),
    ]


def test_actual_kda_graph_discovers_reviewed_span_without_stage_selection():
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
    groups = plan.contraction_groups
    assert groups is not None
    assert plan.pointwise_cache is not None
    entries = frozenset(entry.node for entry in plan.pointwise_cache.entries)
    islands = plan_register_islands(
        cut.region, groups, shapes, fast_math=True, entry_boundaries=entries
    )
    assert len(islands) == 1
    island = islands[0]
    assert tuple(stage for group in island.groups for stage in group.stages) == (
        2,
        3,
        4,
        5,
        6,
        7,
    )
    assert island.exports == tuple(plan.dots[stage] for stage in (4, 5, 7))
    assert len(island.components) == 2
    # Six C plus the complete sixteen-node typed closure, including nodes the
    # artifact inlined instead of caching. Candidates do not allocate registers.
    assert len(island.values) == 22
    assert {value.dtype for value in island.values} == {torch.float16, torch.float32}
    assert register_island_matches(
        island, cut.region, groups, shapes, entry_boundaries=entries
    )


def _region(plan):
    assert plan.region is not None
    region = collect_contraction_region(RootGraphInfo(0, plan.region.graph))
    assert region is not None
    return region


@pytest.mark.parametrize("fill", [1.0, float("inf"), float("nan")])
def test_nonzero_or_nonfinite_mask_fills_cannot_prove_sparse_support(fill):
    plan, groups, shapes, entries, _ = _fixture()
    assert plan.region is not None
    zero = next(
        node
        for node in plan.region.nodes
        if node.target is torch.ops.aten.scalar_tensor.default
    )
    zero.args = (fill,)
    assert (
        plan_register_islands(
            _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
        )
        == ()
    )


@pytest.mark.parametrize("origin", ["dynamic_iota_start", "tile_index"])
def test_runtime_origin_is_never_a_local_static_coordinate(origin):
    plan, groups, shapes, entries, _ = _fixture()
    assert plan.region is not None
    graph = plan.region.graph
    iota = next(
        node for node in graph.nodes if node.target is torch.ops.prims.iota.default
    )
    with graph.inserting_before(iota):
        runtime = graph.placeholder("runtime_origin")
        runtime.meta["val"] = 0
    if origin == "dynamic_iota_start":
        iota.kwargs = {**iota.kwargs, "start": runtime}
    else:
        iota.target = tile_ops.tile_index
        iota.args = (runtime,)
        iota.kwargs = {}
    assert (
        plan_register_islands(
            _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
        )
        == ()
    )


@pytest.mark.parametrize("step", [0, True, 1.0])
def test_unproved_iota_step_is_unknown_not_a_static_mask(step):
    plan, groups, shapes, entries, _ = _fixture()
    assert plan.region is not None
    iota = next(
        node
        for node in plan.region.nodes
        if node.target is torch.ops.prims.iota.default
    )
    iota.kwargs = {**iota.kwargs, "step": step}
    assert (
        plan_register_islands(
            _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
        )
        == ()
    )


@pytest.mark.parametrize("fill", [0.0, -0.0, 1.0])
def test_original_mask_to_is_retained_for_late_domain_validation(fill):
    plan, groups, shapes, entries, values = _fixture()
    assert plan.region is not None
    snapshot = values[1]
    snapshot.target = _tracing_ops._mask_to
    snapshot.args = (values[0], fill)
    # Restore the required input format with a distinct original cast.
    snapshot.meta["val"] = torch.empty((32, 32), dtype=torch.float32)
    with plan.region.graph.inserting_after(snapshot):
        narrowed = _convert(plan.region.graph, snapshot, torch.float16)
    values[2].replace_input_with(snapshot, narrowed)
    values[-1].replace_input_with(snapshot, narrowed)
    shapes[narrowed] = (32, 32)
    candidates = plan_register_islands(
        _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
    )
    if fill != 0:
        assert candidates == ()
    else:
        (island,) = candidates
        assert snapshot in island.domain_nodes
        assert snapshot in {value.node for value in island.values}
        assert snapshot.args == (values[0], fill)


def test_gather_and_explicit_seed_do_not_enter_an_island():
    for change in ("gather", "seed"):
        plan, groups, shapes, entries, values = _fixture()
        assert plan.region is not None
        if change == "seed":
            node = values[2]
            node.args = (*node.args[:2], values[0], torch.float32)
        else:
            snapshot = values[1]
            iota = next(
                node
                for node in plan.region.nodes
                if node.target is torch.ops.prims.iota.default
            )
            snapshot.target = torch.ops.aten.index_select.default
            snapshot.args = (values[0], 0, iota)
            # Keep a real FP32 gather and its distinct required FP16 snapshot.
            snapshot.meta["val"] = torch.empty((32, 32), dtype=torch.float32)
            with plan.region.graph.inserting_after(snapshot):
                narrowed = _convert(plan.region.graph, snapshot, torch.float16)
            values[2].replace_input_with(snapshot, narrowed)
            values[-1].replace_input_with(snapshot, narrowed)
            shapes[narrowed] = (32, 32)
        assert (
            plan_register_islands(
                _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
            )
            == ()
        )


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_support_contains_every_original_finite_interpreter_value(permuted, dtype):
    plan, groups, shapes, entries, _ = _fixture(dtype, permuted=permuted)
    assert plan.region is not None
    (island,) = plan_register_islands(
        plan.region, groups, shapes, fast_math=True, entry_boundaries=entries
    )
    observed = {}

    class Original(Interpreter):
        def call_function(self, target, args, kwargs):
            if target is dot:
                return args[0].float() @ args[1].float()
            return super().call_function(target, args, kwargs)

        def run_node(self, n):
            value = super().run_node(n)
            observed[n] = value
            return value

    generator = torch.Generator().manual_seed(214)
    args = [torch.randn((32, 32), generator=generator).to(dtype) for _ in range(2)]
    Original(GraphModule({}, plan.region.graph)).run(*args)
    for value in island.values:
        tensor = observed[value.node]
        assert tensor.dtype is value.dtype
        for row, column in (tensor != 0).nonzero().tolist():
            assert value.support_rows[row] & (1 << column)


def test_inconsistent_resolved_shapes_and_cyclic_rewrites_fail_closed():
    plan, groups, shapes, entries, values = _fixture()
    assert plan.region is not None
    invalid = {
        node: (64, 64) if shape == (32, 32) else shape for node, shape in shapes.items()
    }
    assert (
        plan_register_islands(
            plan.region, groups, invalid, fast_math=True, entry_boundaries=entries
        )
        == ()
    )
    values[1].args = (values[2], torch.float16)
    assert (
        plan_register_islands(
            plan.region, groups, shapes, fast_math=True, entry_boundaries=entries
        )
        == ()
    )


@pytest.mark.parametrize("use", ["value", "mask", "index"])
def test_every_escaping_store_use_retains_original_c_boundary(use):
    plan, groups, shapes, entries, values = _fixture()
    assert plan.region is not None
    graph = plan.region.graph
    output = next(node for node in graph.nodes if node.op == "output")
    rows = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.aten.unsqueeze.default and shapes[node] == (32, 1)
    )
    columns = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.aten.unsqueeze.default and shapes[node] == (1, 32)
    )
    source = values[0]
    with graph.inserting_before(output):
        mask = (
            _call(graph, torch.ops.aten.ge.Scalar, (source, 0), (32, 32), torch.bool)
            if use == "mask"
            else None
        )
        index = _convert(graph, source, torch.int64) if use == "index" else rows
        graph.call_function(
            memory_ops.store,
            (
                next(iter(entries)),
                (index, columns),
                source if use == "value" else values[-1],
                mask,
            ),
        )
    for node in graph.nodes:
        if isinstance(value := node.meta.get("val"), torch.Tensor):
            shapes[node] = tuple(value.shape)
    (island,) = plan_register_islands(
        _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
    )
    assert source in island.exports
    assert values[-1] in island.exports


def test_intervening_store_prevents_fusing_across_the_effect():
    plan, groups, shapes, entries, values = _fixture()
    assert plan.region is not None
    graph = plan.region.graph
    with graph.inserting_before(values[2]):
        graph.call_function(
            memory_ops.store,
            (next(iter(entries)), (slice(None), slice(None)), values[0], None),
        )
    candidates = plan_register_islands(
        _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
    )
    assert all(
        values[0]
        not in {
            issue.node
            for component in candidate.components
            for issue in component.issues
        }
        for candidate in candidates
    )


def test_extra_dependency_kwargs_are_not_silently_ignored():
    plan, groups, shapes, entries, values = _fixture()
    assert plan.region is not None
    values[1].kwargs = {"_extra_deps": [next(iter(entries))]}
    assert (
        plan_register_islands(
            _region(plan), groups, shapes, fast_math=True, entry_boundaries=entries
        )
        == ()
    )
