from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_producer_teams import _args as cooperative_args
from .test_cute_chained_producer_teams import _cooperative_loop
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_pointwise_residency import PointwiseCachePlan
from helion._compiler.cute.chained_pointwise_residency import allocate_pointwise_cache
from helion._compiler.cute.chained_pointwise_residency import (
    emit_pointwise_cache_before,
)
from helion._compiler.cute.chained_pointwise_residency import (
    has_pointwise_residency_candidate,
)
from helion._compiler.cute.chained_pointwise_residency import plan_pointwise_cache
from helion._compiler.cute.chained_pointwise_residency import validate_pointwise_cache
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.contraction_region import ContractionCarry
import helion.language as hl
from helion.language import memory_ops
from helion.language import scan_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST


def _unary(graph, target, source):
    return _call(
        graph,
        target,
        (source,),
        tuple(source.meta["val"].shape),
        source.meta["val"].dtype,
    )


def _repeated(dtype=torch.float32, *, rank=2, extent=16, narrow=True):
    graph = Graph()
    source = _input(
        graph, "unrelated_source", (extent,) if rank == 1 else (extent, extent), dtype
    )
    computed = _unary(graph, torch.ops.aten.exp.default, source)
    operand = computed
    if rank == 1:
        operand = _call(
            graph,
            torch.ops.aten.expand.default,
            (computed, [extent, extent]),
            (extent, extent),
            dtype,
        )
    if narrow:
        operand = _convert(graph, operand, torch.bfloat16)
    rhs = _input(graph, "right", (extent, extent), torch.bfloat16 if narrow else dtype)
    first, second = _dot(graph, operand, rhs), _dot(graph, operand, rhs)
    return _plan(graph, (first, second)), computed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [1, 2])
def test_typed_rank_one_and_two_candidates_keep_declared_dtype(dtype, rank):
    plan, computed = _repeated(dtype, rank=rank, narrow=False)
    cache = plan_pointwise_cache(plan, 4096)
    assert len(cache.entries) == 1
    entry = cache.entries[0]
    assert entry.node is computed
    assert entry.dtype == dtype
    assert entry.first_stage == 0 and entry.last_stage == 1
    assert entry.producer_uses == 2
    assert cache.shared_bytes == (entry.byte_size + 127) // 128 * 128
    assert cache == plan_pointwise_cache(plan, 4096)
    with pytest.raises(FrozenInstanceError):
        entry.first_stage = 99  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("budget", [0, 1, 127, 128, 255, 256, 512])
def test_budget_includes_alignment_and_never_rounds_down(budget):
    plan, _ = _repeated(rank=1, extent=33)
    cache = plan_pointwise_cache(plan, budget)
    assert cache.shared_bytes <= budget
    assert bool(cache.entries) == (budget >= 256)


def test_grouped_common_left_is_one_physical_producer_not_two_reuses():
    plan, _ = _repeated()
    geometry = StageGeometry((16, 16, 16), False)
    grouped = replace(
        plan, contraction_groups=(ContractionGroup((0, 1), (geometry, geometry)),)
    )
    assert plan_pointwise_cache(plan, 4096).entries
    assert not plan_pointwise_cache(grouped, 4096).entries


def test_grouped_multiple_operand_reads_count_but_need_distinct_stages():
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.float32)
    computed = _convert(
        graph, _unary(graph, torch.ops.aten.exp.default, source), torch.bfloat16
    )
    dots = tuple(_dot(graph, computed, computed) for _ in range(3))
    plan = _plan(graph, dots)
    geometry = StageGeometry((16, 16, 16), False)
    grouped = replace(
        plan,
        contraction_groups=(
            ContractionGroup((0, 1), (geometry, geometry)),
            ContractionGroup((2,), (geometry,)),
        ),
    )
    cache = plan_pointwise_cache(grouped, 4096)
    assert cache.entries[0].producer_uses == 5  # common A + two B; then A + B
    assert (cache.entries[0].first_stage, cache.entries[0].last_stage) == (0, 2)


def test_nested_candidates_select_one_boundary_without_double_counting():
    plan, _ = _repeated()
    cache = plan_pointwise_cache(plan, 1 << 20)
    assert len(cache.entries) == 1
    entry = cache.entries[0]
    assert entry.node.target is torch.ops.prims.convert_element_type.default
    assert entry.dtype == torch.bfloat16
    assert entry.estimated_saved_operations == 16 * 16 * 9


@pytest.mark.parametrize("kind", ["dot", "scan", "reduction", "carry"])
def test_materialized_dependencies_cut_ancestry_and_define_ready_stage(kind):
    graph = Graph()
    lhs = _input(graph, "left", (16, 16), torch.bfloat16)
    rhs = _input(graph, "right", (16, 16), torch.bfloat16)
    boundary = _dot(graph, lhs, rhs)
    if kind == "scan":
        boundary = _call(
            graph,
            scan_ops._associative_scan,
            (0, boundary, 0, False, False),
            (16, 16),
            torch.float32,
        )
    elif kind == "reduction":
        boundary = _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (boundary, [0], True),
            (1, 16),
            torch.float32,
        )
    elif kind == "carry":
        boundary = _input(graph, "state", (16, 16), torch.float32)
    computed = _unary(graph, torch.ops.aten.exp.default, boundary)
    operand = _convert(graph, computed, torch.bfloat16)
    if kind == "reduction":
        operand = _call(
            graph,
            torch.ops.aten.expand.default,
            (operand, [16, 16]),
            (16, 16),
            torch.bfloat16,
        )
    second, third = _dot(graph, operand, rhs), _dot(graph, operand, rhs)
    plan = _plan(graph, (second, third))
    if kind == "carry":
        assert plan.region is not None
        plan = replace(
            plan,
            region=replace(
                plan.region, carries=(ContractionCarry(0, 0, boundary, third),)
            ),
        )
    cache = plan_pointwise_cache(plan, 4096)
    assert cache.entries[0].dependencies == (boundary,)
    assert cache.entries[0].first_stage == 1


@pytest.mark.parametrize("kind", ["creation", "raw", "cast", "view", "bitcast"])
def test_no_arithmetic_and_unproven_bitcasts_are_not_candidates(kind):
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.float32)
    if kind == "creation":
        source = _call(
            graph, torch.ops.aten.full.default, ([16, 16], 1.0), (16, 16), torch.float32
        )
        source = _unary(graph, torch.ops.aten.exp.default, source)
    elif kind == "view":
        source = _call(
            graph,
            torch.ops.aten.view.default,
            (source, [16, 16]),
            (16, 16),
            torch.float32,
        )
    elif kind == "bitcast":
        source = _call(
            graph,
            torch.ops.aten.view.dtype,
            (source, torch.float32),
            (16, 16),
            torch.float32,
        )
        source = _unary(graph, torch.ops.aten.exp.default, source)
    source = _convert(graph, source, torch.bfloat16)
    rhs = _input(graph, "rhs", (16, 16), torch.bfloat16)
    dots = (_dot(graph, source, rhs), _dot(graph, source, rhs))
    assert not plan_pointwise_cache(_plan(graph, dots), 4096).entries


@pytest.mark.parametrize("kind", ["internal_load", "indexed_view"])
def test_unproven_internal_coordinate_maps_are_not_cached(kind):
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.float32)
    computed = _unary(graph, torch.ops.aten.exp.default, source)
    if kind == "internal_load":
        mapped = _call(
            graph,
            memory_ops.load,
            (computed, [slice(None), slice(None)], None, None),
            (16, 16),
            torch.float32,
        )
    else:
        mapped = _call(
            graph,
            view_ops.subscript,
            (computed, [slice(0, 16), slice(None)]),
            (16, 16),
            torch.float32,
        )
    operand = _convert(graph, mapped, torch.bfloat16)
    rhs = _input(graph, "rhs", (16, 16), torch.bfloat16)
    dots = (_dot(graph, operand, rhs), _dot(graph, operand, rhs))
    cache = plan_pointwise_cache(_plan(graph, dots), 4096)
    # A post-gather computed result may still be materialized, but the earlier
    # expression may not be rebound as an out-of-range-zero cache boundary.
    assert all(entry.node is not computed for entry in cache.entries)


def test_same_host_read_write_source_is_not_cached():
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.float32)
    loaded = _call(
        graph,
        memory_ops.load,
        (source, [slice(None), slice(None)], None, None),
        (16, 16),
        torch.float32,
    )
    computed = _convert(
        graph, _unary(graph, torch.ops.aten.exp.default, loaded), torch.bfloat16
    )
    rhs = _input(graph, "rhs", (16, 16), torch.bfloat16)
    first, second = _dot(graph, computed, rhs), _dot(graph, computed, rhs)
    graph.call_function(memory_ops.store, (source, [slice(None), slice(None)], second))
    assert not plan_pointwise_cache(_plan(graph, (first, second)), 4096).entries


def test_missing_dependencies_fail_before_binding_or_emitting():
    graph = Graph()
    lhs = _input(graph, "left", (16, 16), torch.bfloat16)
    rhs = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, lhs, rhs)
    computed = _convert(
        graph, _unary(graph, torch.ops.aten.exp.default, first), torch.bfloat16
    )
    _dot(graph, computed, rhs)
    third = _dot(graph, computed, rhs)
    plan = _plan(graph, (third,))
    cache = plan_pointwise_cache(plan, 4096)
    boundaries = {}
    cg = cast("GenerateAST", None)
    assert emit_pointwise_cache_before(cg, plan, boundaries, cache, 0) == []
    with pytest.raises(chain._UnsupportedChain, match="published source"):
        emit_pointwise_cache_before(cg, plan, boundaries, cache, 1)
    assert not boundaries


def test_independent_allocation_and_layout_activation_are_codegen_local():
    plan, _ = _repeated(narrow=False)
    cache = plan_pointwise_cache(plan, 4096)
    first = ScratchLayouts("xor", read_buffers=frozenset())
    second = ScratchLayouts("xor", read_buffers=frozenset())
    source = "\n".join(allocate_pointwise_cache(cache, first))
    assert "alignment=128" in source
    assert "chain_c_workspace" not in source
    assert first.read_buffers == frozenset((cache.entries[0].name,))
    assert first.xor_buffers == {cache.entries[0].name}
    first.validate()
    assert second.read_buffers == frozenset()
    assert cache == plan_pointwise_cache(plan, 4096)
    assert allocate_pointwise_cache(PointwiseCachePlan(), second) == []


@pytest.mark.parametrize("budget", [True, False, 0.0, -1, "4096", None])
def test_noninteger_or_negative_budgets_fail_closed(budget):
    with pytest.raises(exc.BackendUnsupported, match="positive integer budget"):
        validate_pointwise_cache(None, budget)


def test_validator_requires_an_effective_selected_plan_except_zero():
    plan, _ = _repeated()
    validate_pointwise_cache(None, 0)
    validate_pointwise_cache(plan, 0)
    for ineffective in (
        None,
        plan,
        replace(plan, pointwise_cache=PointwiseCachePlan()),
    ):
        with pytest.raises(exc.BackendUnsupported, match="effective typed residency"):
            validate_pointwise_cache(ineffective, 4096)
    selected = replace(plan, pointwise_cache=plan_pointwise_cache(plan, 4096))
    validate_pointwise_cache(selected, 4096)


def test_epilogue_use_extends_reported_lifetime_without_recycling_storage():
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.bfloat16)
    computed = _unary(graph, torch.ops.aten.exp.default, source)
    rhs = _input(graph, "rhs", (16, 16), torch.bfloat16)
    first, second = _dot(graph, computed, rhs), _dot(graph, computed, rhs)
    cache = plan_pointwise_cache(_plan(graph, (first, second, computed)), 4096)
    assert cache.entries[0].last_stage == 2
    assert cache.shared_bytes == 512


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_reuse_loop(a, rhs, initial, rank_one: hl.constexpr):
    steps, m, k = a.shape
    history = torch.empty((steps, m, k), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, k], block_size=[16, 16]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            if rank_one:
                leaf = hl.load(a, [step.id, 0, kk], extra_mask=(kk % 2 == 0))
                repeated = (
                    torch.exp(leaf.float())[None, :] * a[step.id, rows, kk].float()
                )
            else:
                leaf = hl.load(
                    a, [step.id, rows, kk], extra_mask=(rows.index % 2 == 0)[:, None]
                )
                repeated = torch.exp(leaf.float())
            operand = repeated.to(torch.bfloat16).float().to(torch.float16)
            first = hl.dot(operand, rhs[step.id, kk, cols], out_dtype=torch.float32)
            state = hl.dot(
                operand,
                rhs[step.id, kk, cols],
                acc=state + first,
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        final[rows, cols] = state
    return history, final


@pytest.mark.parametrize("reused_arithmetic", [False, True])
def test_structural_discovery_needs_reused_arithmetic_not_just_shared_loads(
    reused_arithmetic,
):
    with _cpu_codegen():
        if reused_arithmetic:
            bound = _masked_reuse_loop._bind_isolated(
                (
                    torch.empty((2, 19, 16), dtype=torch.bfloat16),
                    torch.empty((2, 16, 16), dtype=torch.float16),
                    torch.empty((19, 16), dtype=torch.float32),
                    True,
                )
            )
        else:
            bound = _cooperative_loop._bind_isolated(cooperative_args("cpu", 2, True))
        facts = []
        original = chain.plan_chained_matmul

        def observe(graphs):
            facts.append(has_pointwise_residency_candidate(graphs))
            return original(graphs)

        with patch.object(chain, "plan_chained_matmul", observe):
            bound.to_code(
                helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")
            )
        assert facts and all(fact is reused_arithmetic for fact in facts)


@pytest.mark.parametrize("rank_one", [False, True])
def test_real_expression_emission_preserves_masks_casts_and_loop_scope(rank_one):
    args = (
        torch.empty((2, 19, 16), dtype=torch.bfloat16),
        torch.empty((2, 16, 16), dtype=torch.float16),
        torch.empty((19, 16), dtype=torch.float32),
        rank_one,
    )
    original_plan = chain.plan_chained_matmul
    original_allocate = chained_tcgen_stage.allocate_stages
    original_emit = chained_tcgen_stage.emit_stage
    selected = []
    emitted = []

    def observe_plan(graphs):
        plan = original_plan(graphs)
        if plan is not None:
            assert has_pointwise_residency_candidate(graphs)
            cache = plan_pointwise_cache(plan, 128 if rank_one else 4096)
            assert cache.entries
            selected.append(cache)
        return plan

    def allocate(*args, **kwargs):
        return [
            *original_allocate(*args, **kwargs),
            *allocate_pointwise_cache(selected[-1], ScratchLayouts()),
        ]

    def emit(cg, plan, boundaries, stage, *args, **kwargs):
        before = emit_pointwise_cache_before(cg, plan, boundaries, selected[-1], stage)
        emitted.extend(before)
        return [*before, *original_emit(cg, plan, boundaries, stage, *args, **kwargs)]

    with (
        _cpu_codegen(),
        patch.object(chain, "plan_chained_matmul", observe_plan),
        patch.object(chained_tcgen_stage, "allocate_stages", allocate),
        patch.object(chained_tcgen_stage, "emit_stage", emit),
    ):
        source = _masked_reuse_loop._bind_isolated(args).to_code(
            helion.Config(num_warps=16, cute_chained_mma_schedule="tcgen05_tmem")
        )
    assert source.index("chain_pointwise_cache_0 = cute.make_tensor") < source.index(
        "for chain_loop_index"
    )
    assert source.index("for chain_loop_index") < source.index(
        "for chain_pointwise_cache_0_step"
    )
    fill = "\n".join(emitted)
    assert any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitAnd)
        and isinstance(node.right, ast.Constant)
        and node.right.value == 1
        for node in ast.walk(ast.parse(fill))
    )
    assert "cutlass.Int32(2)" in fill and ".load() if" in fill
    assert "cute.math.exp2" in fill
    assert "cutlass.BFloat16" in source and "cutlass.Float16" in source
    assert fill.endswith("cute.arch.sync_threads()")
    assert fill.count("cute.arch.sync_threads()") == 1
    assert "if chain_pointwise_cache_0_index <" in fill
    assert ast.parse(source)
    assert "chain_pointwise_cache_0[chain_1_a_1_index" in source
