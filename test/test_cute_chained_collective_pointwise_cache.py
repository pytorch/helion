from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_residency import (
    emit_pointwise_cache_before_collective,
)
from helion._compiler.cute.chained_pointwise_residency import plan_pointwise_cache
import helion.language as hl
from helion.language import memory_ops
from helion.language import scan_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.generate_ast import GenerateAST


def _event_graph(
    *, prior_scan: bool = False, unsafe_view: str | None = None
) -> tuple[ChainedMatmulPlan, Node, tuple[Node, Node]]:
    graph = Graph()
    lhs = _input(graph, "left", (16, 16), torch.bfloat16)
    rhs = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, lhs, rhs)
    if prior_scan:
        first = _call(
            graph,
            scan_ops._associative_scan,
            (0, first, 0, False, False),
            (16, 16),
            torch.float32,
        )
    computed = _call(
        graph, torch.ops.aten.exp.default, (first,), (16, 16), torch.float32
    )
    source = computed
    if unsafe_view == "internal_load":
        source = _call(
            graph,
            memory_ops.load,
            (computed, [slice(None), slice(None)], None, None),
            (16, 16),
            torch.float32,
        )
    elif unsafe_view == "indexed_view":
        source = _call(
            graph,
            view_ops.subscript,
            (computed, [slice(0, 16), slice(None)]),
            (16, 16),
            torch.float32,
        )
    reductions = tuple(
        _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (source, [axis], True),
            (1, 16) if axis == 0 else (16, 1),
            torch.float32,
        )
        for axis in (0, 1)
    )
    joined = _call(
        graph, torch.ops.aten.add.Tensor, reductions, (16, 16), torch.float32
    )
    final = _dot(graph, _convert(graph, joined, torch.bfloat16), rhs)
    return _plan(graph, (final,)), computed, (reductions[0], reductions[1])


def test_two_collectives_in_one_dot_stage_have_an_exact_publication_event() -> None:
    plan, computed, reductions = _event_graph()
    assert not plan_pointwise_cache(plan, 4096).entries
    cache = plan_pointwise_cache(plan, 4096, collective_consumers=True)
    assert len(cache.entries) == 1
    entry = cache.entries[0]
    assert entry.node is computed
    assert entry.first_collective is reductions[0]
    assert entry.first_stage == entry.last_stage == 1
    assert entry.dependencies == (plan.dots[0],)
    assert entry.producer_uses == 2
    assert entry.dtype == torch.float32 and cache.shared_bytes == 1024


@pytest.mark.parametrize("budget", [0, 127, 512, 1023, 1024])
def test_collective_budget_accounts_for_the_whole_typed_image(budget: int) -> None:
    plan, _, _ = _event_graph()
    cache = plan_pointwise_cache(plan, budget, collective_consumers=True)
    assert cache.shared_bytes <= budget
    assert bool(cache.entries) == (budget >= 1024)


@pytest.mark.parametrize("strategy", ["tcgen05_tmem", "k_major"])
def test_unimplemented_emission_routes_reject(strategy: str) -> None:
    plan, _, _ = _event_graph()
    assert not plan_pointwise_cache(
        replace(plan, strategy=strategy), 4096, collective_consumers=True
    ).entries


def test_nested_event_publications_are_not_silently_admitted() -> None:
    plan, _, _ = _event_graph()
    with pytest.raises(ValueError, match="independent typed"):
        plan_pointwise_cache(
            plan, 4096, max_entries=2, nested=True, collective_consumers=True
        )


def test_prior_collective_dependency_must_be_published_before_the_cache() -> None:
    plan, computed, reductions = _event_graph(prior_scan=True)
    cache = plan_pointwise_cache(plan, 4096, collective_consumers=True)
    entry = cache.entries[0]
    assert entry.node is computed
    assert entry.first_collective is reductions[0]
    assert plan.region is not None
    assert entry.dependencies == plan.region.scans
    selected = replace(plan, pointwise_cache=cache)
    boundaries = {plan.dots[0]: "first_dot"}
    with pytest.raises(chain._UnsupportedChain, match="published source"):
        emit_pointwise_cache_before_collective(
            cast("GenerateAST", None),
            selected,
            boundaries,
            reductions[0],
            execution=ChainedExecution(plan.threads),
        )
    assert computed not in boundaries


@pytest.mark.parametrize("kind", ["internal_load", "indexed_view"])
def test_event_cache_preserves_unproven_coordinate_rejection(kind: str) -> None:
    plan, computed, _ = _event_graph(unsafe_view=kind)
    cache = plan_pointwise_cache(plan, 4096, collective_consumers=True)
    assert all(entry.node is not computed for entry in cache.entries)


def test_event_cache_rejects_arithmetic_reading_written_host_storage() -> None:
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.float32)
    loaded = _call(
        graph,
        memory_ops.load,
        (source, [slice(None), slice(None)], None, None),
        (16, 16),
        torch.float32,
    )
    computed = _call(
        graph, torch.ops.aten.exp.default, (loaded,), (16, 16), torch.float32
    )
    reductions = tuple(
        _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (computed, [axis], True),
            (1, 16) if axis == 0 else (16, 1),
            torch.float32,
        )
        for axis in (0, 1)
    )
    operand = _call(
        graph, torch.ops.aten.add.Tensor, reductions, (16, 16), torch.float32
    )
    rhs = _input(graph, "right", (16, 16), torch.bfloat16)
    output = _dot(graph, _convert(graph, operand, torch.bfloat16), rhs)
    graph.call_function(memory_ops.store, (source, [slice(None), slice(None)], output))
    assert not plan_pointwise_cache(
        _plan(graph, (output,)), 4096, collective_consumers=True
    ).entries


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _selected_products(
    left: torch.Tensor, right: torch.Tensor, masked: hl.constexpr
) -> torch.Tensor:
    result = torch.empty_like(left)
    for batch in hl.tile(left.size(0), block_size=1):
        matrix = hl.dot(left[batch.id, :, :], right[batch.id, :, :])
        index = hl.arange(16)
        value = matrix * 0.125
        if masked:
            value = (
                value
                + hl.load(
                    left,
                    [batch.id, index[:, None], index[None, :]],
                    extra_mask=(index[:, None] % 3 != 1),
                ).float()
            )
        computed = torch.exp(value).to(torch.float16).float()
        row = torch.where(index[:, None] == 3, computed, 0.0).sum(0)
        column = torch.where(index[None, :] == 7, computed, 0.0).sum(1)
        operand = (row[:, None] + column[None, :]).to(left.dtype)
        result[batch.id, :, :] = hl.dot(operand, right[batch.id, :, :]).to(left.dtype)
    return result


def _config(enabled: bool, **extra: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            "cute_chained_mma_schedule": "coalesced",
            "cute_chained_pointwise_cache_bytes": 4096 if enabled else 0,
            "cute_chained_pointwise_cache_collectives": enabled,
        }
        | extra
    )


def _args(
    device: str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((2, 16, 16), dtype=dtype, device=device),
        torch.empty((2, 16, 16), dtype=dtype, device=device),
    )


def _source(enabled: bool, *, masked: bool = False) -> str:
    args = (*_args(), masked)
    with _cpu_codegen():
        bound = _selected_products._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_selected_products, args)
        ):
            return bound.to_code(_config(enabled))


def _zero_adds(source: str) -> list[str]:
    return [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.endswith("_acc")
        and node.targets[0].id.startswith("chain_collective_")
        and isinstance(node.value, ast.BinOp)
        and any(
            isinstance(child, ast.Name) and child.id == node.targets[0].id
            for child in ast.walk(node.value)
        )
    ]


@pytest.mark.parametrize("masked", [False, True])
def test_source_publishes_before_first_collective_and_keeps_zero_add_order(
    masked: bool,
) -> None:
    original, cached = _source(False, masked=masked), _source(True, masked=masked)
    assert "chain_pointwise_cache_0" not in original
    assert cached.index("for chain_pointwise_cache_0_step") < cached.index(
        "for chain_collective_0_vector_step"
    )
    assert _zero_adds(cached) == _zero_adds(original)
    assert len(_zero_adds(cached)) == 10
    assert "cutlass.Float16(" in cached


@pytest.mark.parametrize(
    "extra",
    [
        {"cute_chained_pointwise_cache_collectives": 1},
        {"cute_chained_pointwise_cache_bytes": 0},
        {"cute_chained_mma_schedule": "tcgen05_tmem"},
        {
            "cute_chained_pointwise_cache_nested": True,
            "cute_chained_pointwise_cache_entries": 2,
        },
    ],
)
def test_explicit_unsupported_options_reject(extra: dict[str, object]) -> None:
    with _cpu_codegen():
        bound = _selected_products._bind_isolated((*_args(), False))
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalized_config(dict(_config(True, **extra)))


@pytest.mark.parametrize("disabled", [False, True])
def test_collective_seed_is_one_bounded_sibling_and_honors_overrides(
    disabled: bool,
) -> None:
    with _cpu_codegen():
        bound = _selected_products._bind_isolated((*_args(), False))
        assert bound.config_spec.cute_chained_pointwise_collective_search_enabled
        assert not bound.config_spec.cute_chained_pointwise_residency_search_enabled
        discovered = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_chained_pointwise_cache_collectives")
        ]
        assert len(discovered) == 1
        bound.config_spec.normalized_config(discovered[0].config)
        parent = _config(False)
        original = [parent, helion.Config(num_warps=8)]
        overrides = (
            {"cute_chained_pointwise_cache_collectives": False} if disabled else {}
        )
        with patch.dict(bound.env.settings.autotune_config_overrides, overrides):
            seeds = CuteChainedMatmulHeuristic._with_collective_cache_seed(
                bound.env, original
            )
        assert seeds[: len(original)] == original
        assert len(seeds) == len(original) + (not disabled)
        if not disabled:
            assert seeds[-1].config["cute_chained_pointwise_cache_bytes"] == 16384
            bound.config_spec.normalized_config(seeds[-1].config)
            assert (
                CuteChainedMatmulHeuristic._with_collective_cache_seed(bound.env, seeds)
                == seeds
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_collective_cache_preserves_typed_values_gpu(
    dtype: torch.dtype, transposed: bool, masked: bool
) -> None:
    torch.manual_seed(91021)
    args = tuple(
        (torch.randint(-2, 3, (2, 16, 16), device="cuda") * 0.015625).to(dtype)
        for _ in range(2)
    )
    if transposed:
        args = tuple(value.transpose(-1, -2) for value in args)
    original_inputs = tuple(value.clone() for value in args)
    matrix = args[0].double() @ args[1].double()
    value = matrix * 0.125
    if masked:
        mask = torch.arange(16, device="cuda")[:, None] % 3 != 1
        value = value + torch.where(mask, args[0].double(), 0.0)
    computed = torch.exp(value).to(torch.float16).double()
    operand = (
        (computed[:, 3, :, None] + computed[:, :, 7].unsqueeze(-2)).to(dtype).double()
    )
    expected = (operand @ args[1].double()).to(dtype)
    results = []
    for enabled in (False, True):
        results.append(
            _selected_products._bind_isolated((*args, masked)).compile_config(
                _config(enabled)
            )(*args, masked)
        )
        torch.testing.assert_close(results[-1], expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    for before, after in zip(original_inputs, args, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
