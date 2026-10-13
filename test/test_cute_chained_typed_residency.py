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
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_root_typed_operands import _config
from .test_cute_chained_workspace import _assert_safe
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute import chained_pointwise_residency
from helion._compiler.cute.chained_pointwise_residency import plan_pointwise_cache
from helion._compiler.cute.chained_root_typed_operands import _cast_path
from helion._compiler.cute.chained_tcgen05 import _shared_memory_bytes
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_pointwise_residency import PointwiseCachePlan
    from helion._compiler.device_ir import GraphInfo


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cached_chain(
    left: torch.Tensor,
    right: torch.Tensor,
    gate: torch.Tensor,
    dependent: hl.constexpr,
    seeded: hl.constexpr,
    half: hl.constexpr,
) -> torch.Tensor:
    if half:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    out = torch.empty(left.shape, dtype=torch.float32, device=left.device)
    for batch in hl.tile(left.size(0), block_size=1):
        idx = batch.id
        first = hl.dot(left[idx, :, :], right[idx, :, :])
        if dependent:
            factor = torch.exp(first * 0.015625)
        else:
            factor = torch.exp(gate[idx, :])[:, None]
        rhs = torch.where(
            hl.arange(64)[None, :] < 59,
            right[idx, :, :].to(torch.float32) * factor,
            0,
        ).to(dtype)
        second = hl.dot(first.to(dtype), rhs)
        if seeded:
            third = hl.dot(second.to(dtype), rhs, acc=first)
        else:
            third = hl.dot(second.to(dtype), rhs)
        out[idx, :, :] = third
    return out


def _args(
    dependent: bool, seeded: bool, dtype: torch.dtype, device: str = "cpu"
) -> tuple[object, ...]:
    return (
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, 64), device=device),
        dependent,
        seeded,
        dtype == torch.float16,
    )


def _code(args: tuple[object, ...], budget: int) -> tuple[str, ChainedMatmulPlan, int]:
    plans: list[ChainedMatmulPlan] = []
    footprints: list[int] = []
    original = chained_matmul.plan_chained_matmul

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            plans.append(plan)
            footprints.append(_shared_memory_bytes(plan))
        return plan

    with _cpu_codegen(), patch.object(chained_matmul, "plan_chained_matmul", observe):
        source = _cached_chain._bind_isolated(args).to_code(
            _config(cute_chained_pointwise_cache_bytes=budget)
        )
    assert len(plans) == len(footprints) == 1
    return source, plans[0], footprints[0]


@pytest.mark.parametrize("dependent,budget", [(False, 4096), (True, 16384)])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_codegen_composes_exact_cache_with_typed_images_and_fp32_seeds(
    dependent: bool, budget: int, seeded: bool, dtype: torch.dtype
) -> None:
    source, plan, footprint = _code(_args(dependent, seeded, dtype), budget)
    retained, cache = plan.root_typed_operands, plan.pointwise_cache
    assert retained is not None and cache is not None and len(cache.entries) == 1
    assert len(retained.images) == 2
    _assert_safe(retained.operands)
    _assert_safe(retained.fp32)
    assert footprint <= 232448
    entry = cache.entries[0]
    assert entry.first_stage == 1
    assert (plan.dots[0] in entry.dependencies) == dependent
    assert {region.name for region in retained.fp32.regions} == (
        {"chain_0_c"} if dependent or seeded else set()
    )
    assert "chain_1_a_step" not in source and "chain_2_a_step" not in source
    assert source.count("cute.autovec_copy(chain_typed_") == 2
    assert source.count("chain_pointwise_cache_0 = cute.make_tensor(") == 1
    assert source.index("for chain_pointwise_cache_0_step") < source.index(
        "chain_1_mma ="
    )
    kernel = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion__cached_chain"
    )
    publication = next(
        index
        for index, node in enumerate(kernel.body)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "chain_pointwise_cache_0_step"
    )
    assert ast.unparse(kernel.body[publication + 1]) == "cute.arch.sync_threads()"
    assert all(
        entry.node not in use.path for image in retained.images for use in image.uses
    )


def test_zero_budget_keeps_original_typed_source() -> None:
    args = _args(False, False, torch.float16)
    with _cpu_codegen():
        bound = _cached_chain._bind_isolated(args)
        source = bound.to_code(_config())
        assert bound.to_code(_config(cute_chained_pointwise_cache_bytes=0)) == source


def test_cache_bytes_cannot_bypass_final_shared_capacity_check() -> None:
    args = _args(False, False, torch.float16)
    _, _, without = _code(args, 0)
    _, plan, with_cache = _code(args, 4096)
    assert plan.pointwise_cache is not None
    assert with_cache == without + plan.pointwise_cache.shared_bytes
    assert with_cache > without
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config,
            "per_cta_smem_capacity_bytes",
            return_value=without,
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        _cached_chain._bind_isolated(args).to_code(
            _config(cute_chained_pointwise_cache_bytes=4096)
        )


@pytest.mark.parametrize("keep", ["cache", "seed", "export"])
def test_cache_publication_reads_survive_elided_operand_reads(keep: str) -> None:
    graph = Graph()
    left = _input(graph, "left", (64, 64), torch.bfloat16)
    right = _input(graph, "right", (64, 64), torch.bfloat16)
    first = _dot(graph, left, right)
    computed = _call(
        graph, torch.ops.aten.exp.default, (first,), (64, 64), torch.float32
    )
    rhs = _convert(graph, computed, torch.bfloat16)
    second = _dot(graph, _convert(graph, first, torch.bfloat16), rhs)
    third = _dot(
        graph,
        _convert(graph, second, torch.bfloat16),
        rhs,
        first if keep == "seed" else None,
    )
    plan = _plan(graph, (first, third) if keep == "export" else (third,))
    cache = plan_pointwise_cache(plan, 16384)
    assert len(cache.entries) == 1 and cache.entries[0].dependencies == (first,)
    # Model independently supplied operand images. The cache still evaluates
    # its original FP32 recipe before stage1 even if no MMA reads it directly.
    options = {
        "include_final": False,
        "skip_operand_reads": frozenset({(1, 0), (1, 1), (2, 0), (2, 1)}),
        "omit_unused": True,
    }
    without = plan_contraction_workspace(plan, **options)
    workspace = plan_contraction_workspace(
        replace(plan, pointwise_cache=cache), **options
    )
    assert {region.name for region in workspace.regions} == {"chain_0_c"}
    assert workspace.regions[0].live_until == {"cache": 3, "seed": 5, "export": 7}[keep]
    assert bool(without.regions) == (keep != "cache")
    _assert_safe(workspace)


def test_cached_binding_on_a_retained_cast_path_rejects_before_emission() -> None:
    original = chained_pointwise_residency.plan_pointwise_cache
    attempts = []

    def overlap(
        plan: ChainedMatmulPlan,
        budget_bytes: int,
        *,
        max_entries: int = 1,
        nested: bool = False,
        collective_consumers: bool = False,
    ) -> PointwiseCachePlan:
        cache = original(
            plan,
            budget_bytes,
            max_entries=max_entries,
            nested=nested,
            collective_consumers=collective_consumers,
        )
        assert len(cache.entries) == 1
        use = _cast_path(plan.dots, 1, 0)
        assert use is not None
        attempts.append(use)
        entry = cache.entries[0]
        return replace(
            cache,
            entries=(
                replace(
                    entry,
                    node=use.operand,
                    shape=(64, 64),
                    dtype=use.dtype,
                    byte_size=8192,
                    allocated_bytes=8192,
                ),
            ),
        )

    with (
        _cpu_codegen(),
        patch.object(chained_pointwise_residency, "plan_pointwise_cache", overlap),
        pytest.raises(exc.BackendUnsupported),
    ):
        _cached_chain._bind_isolated(_args(False, False, torch.float16)).to_code(
            _config(cute_chained_pointwise_cache_bytes=16384)
        )
    assert attempts


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_typed_cache_gpu_reference_seed_masks_replay_and_inputs(
    dependent: bool, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = _args(dependent, True, dtype, "cuda")
    left, right, gate = args[:3]
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
    assert isinstance(gate, torch.Tensor)
    generator = torch.Generator(device="cuda").manual_seed(71903)
    for value in (left, right):
        value.copy_(
            torch.randint(-2, 3, value.shape, generator=generator, device="cuda") / 16
        )
    gate.copy_(torch.linspace(-0.25, 0.25, 64, device="cuda"))
    before = tuple(value.clone() for value in (left, right, gate))
    first = (left.double() @ right.double()).float()
    factor = torch.exp(first * 0.015625) if dependent else torch.exp(gate)[:, :, None]
    rhs = (right.float() * factor).to(dtype)
    rhs[:, :, 59:] = 0
    second = (first.to(dtype).double() @ rhs.double()).float()
    expected = (second.to(dtype).double() @ rhs.double() + first.double()).float()
    implementation = _cached_chain._bind_isolated(args).compile_config(
        _config(cute_chained_pointwise_cache_bytes=16384 if dependent else 4096)
    )
    actual = implementation(*args)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = implementation(*args)
    for _ in range(3):
        replayed.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(replayed, actual, atol=0, rtol=0)
    for value, saved in zip((left, right, gate), before, strict=True):
        assert torch.equal(value, saved)
