from __future__ import annotations

import ast
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
from .test_cute_chained_workspace import _assert_safe
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
import helion.language as hl
from helion.language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _long_chain(
    left: torch.Tensor,
    right: torch.Tensor,
    initial: torch.Tensor,
    stages: int,
    mixed: hl.constexpr,
    fanout: hl.constexpr,
) -> torch.Tensor:
    stages = hl.specialize(stages)
    out = torch.empty_like(initial)
    for tile in hl.tile(left.size(0), block_size=1):
        batch = tile.id
        rhs = right[batch, :, :]
        state = hl.dot(left[batch, :, :], rhs, acc=initial[batch, :, :])
        first = state
        for _step in hl.static_range(stages - 1):
            # The seed reads permuted C coordinates, crossing warp ownership.
            # Reusing its arena slot must wait for every warp's seed reads.
            seed = state.T * 0.125
            if mixed:
                state = hl.dot(state.to(torch.float16), rhs.to(torch.float16), acc=seed)
            else:
                state = hl.dot(state.to(left.dtype), rhs, acc=seed)
        if fanout:
            state = state + first * 0.0625
        out[batch, :, :] = state
    return out


def _args(
    *,
    stages: int = 6,
    mixed: bool = False,
    fanout: bool = True,
    device: str = "cpu",
    size: int = 128,
) -> tuple[object, ...]:
    return (
        torch.empty((2, size, size), dtype=torch.bfloat16, device=device),
        torch.empty((2, size, size), dtype=torch.bfloat16, device=device),
        torch.empty((2, size, size), dtype=torch.float32, device=device),
        stages,
        mixed,
        fanout,
    )


def _code(
    *,
    stages: int = 6,
    mixed: bool = False,
    fanout: bool = True,
    schedule: str = "coalesced",
    capacity: int = 232448,
    size: int = 128,
) -> tuple[str, ChainedMatmulPlan]:
    plans: list[ChainedMatmulPlan] = []
    original = chained_matmul.plan_chained_matmul

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            plans.append(plan)
        return plan

    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
        patch.object(chained_matmul, "plan_chained_matmul", observe),
    ):
        bound = _long_chain._bind_isolated(
            _args(stages=stages, mixed=mixed, fanout=fanout, size=size)
        )
        source = bound.to_code(
            helion.Config(num_warps=4, cute_chained_mma_schedule=schedule)
        )
    assert len(plans) == 1
    return source, plans[0]


@pytest.mark.parametrize("padding", [0, 4])
@pytest.mark.parametrize("explicit_accumulator", [False, True])
def test_padded_workspace_recycles_after_operand_and_seed_reads(
    padding: int, explicit_accumulator: bool
) -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    second = (
        _dot(graph, left, right, first)
        if explicit_accumulator
        else _dot(
            graph,
            _convert(graph, first, torch.float16),
            _convert(graph, right, torch.float16),
        )
    )
    third = _dot(graph, _convert(graph, second, torch.bfloat16), right)
    layout = plan_contraction_workspace(_plan(graph, (third,)), row_padding=padding)
    assert layout.allocated_bytes == 4 * 16 * (16 + padding)
    assert [region.byte_offset for region in layout.regions] == [0, 0, 0]
    assert [(region.live_from, region.live_until) for region in layout.regions] == [
        (1, 3),
        (3, 5),
        (5, 7),
    ]
    _assert_safe(layout)


@pytest.mark.parametrize("use", ["fanout", "output", "store", "mask"])
def test_padded_workspace_keeps_exported_and_later_consumed_values(use: str) -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    first = _dot(graph, left, right)
    second = _dot(graph, _convert(graph, first, torch.bfloat16), right)
    if use == "fanout":
        third = _dot(graph, _convert(graph, second, torch.bfloat16), right, first)
        outputs = (third,)
    else:
        third = _dot(graph, _convert(graph, second, torch.bfloat16), right)
        outputs = (first, third) if use == "output" else (third,)
        if use in ("store", "mask"):
            output = _input(graph, "stored", (16, 16), torch.float32)
            mask = (
                _call(
                    graph, torch.ops.aten.gt.Scalar, (first, 0.0), (16, 16), torch.bool
                )
                if use == "mask"
                else None
            )
            graph.call_function(
                memory_ops.store,
                (
                    output,
                    [slice(None), slice(None)],
                    first if use == "store" else third,
                    mask,
                ),
            )
    layout = plan_contraction_workspace(_plan(graph, outputs), row_padding=4)
    assert layout.allocated_bytes == 2560
    assert layout.regions[0].byte_offset != layout.regions[1].byte_offset
    assert layout.regions[0].live_until == (5 if use == "fanout" else 7)
    _assert_safe(layout)


@pytest.mark.parametrize("padding", [-1, 1.0, True])
def test_padded_workspace_rejects_invalid_padding(padding: object) -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    plan = _plan(graph, (_dot(graph, left, left),))
    with pytest.raises(ValueError, match="row padding"):
        plan_contraction_workspace(plan, row_padding=padding)  # type: ignore[arg-type]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "schedule", ["coalesced", "coalesced_unrolled", "cp_async", "k_major"]
)
def test_long_root_uses_padded_result_arena_only_for_capacity(
    mixed: bool, schedule: str
) -> None:
    source, plan = _code(mixed=mixed, schedule=schedule)
    layout = plan.warp_workspace
    assert layout is not None
    assert layout.allocated_bytes == (131072 if schedule == "k_major" else 135168)
    _assert_safe(layout)
    assert "chain_c_workspace = cute.arch.alloc_smem(cutlass.Float32," in source
    assert "chain_0_c_ptr = chain_c_workspace" in source
    assert source.count("MmaF16BF16Op") == 6
    assert "ptp_thread" not in source
    if mixed:
        assert "MmaF16BF16Op(cutlass.BFloat16," in source
        assert "MmaF16BF16Op(cutlass.Float16," in source
        assert "cute.recast_ptr(chain_a_workspace, dtype=cutlass.Float16)" in source
    # Every stage's final shared reads precede a CTA join before publication.
    tree = ast.parse(source)
    for stage in range(6):
        publication = f"cute.autovec_copy(chain_{stage}_acc, chain_{stage}_sc)"
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(ast.unparse(statement) == publication for statement in node.body)
        )
        position = next(
            index
            for index, statement in enumerate(function.body)
            if ast.unparse(statement) == publication
        )
        # partition_C follows a CTA-wide join; publication is its next effect.
        assert ast.unparse(function.body[position - 2]) == "cute.arch.sync_threads()"
        marker = source.index(publication)
        compute = source.rfind(f"cute.gemm(chain_{stage}_mma", 0, marker)
        assert compute >= 0
        assert "cute.arch.sync_threads()" in source[compute:marker]


def test_fitting_root_retains_separate_results_and_original_barrier_count() -> None:
    source, plan = _code(stages=2)
    assert plan.warp_workspace is None
    assert "chain_c_workspace" not in source
    assert source.count("cute.arch.sync_threads()") == 4
    assert "chain_0_c_ptr = cute.arch.alloc_smem(cutlass.Float32, 16896," in source


def test_rescue_still_checks_complete_aligned_shared_capacity() -> None:
    _, plan = _code(capacity=204800)
    assert plan.warp_workspace is not None
    with pytest.raises(exc.BackendUnsupported, match="shared bytes exceed 204799"):
        _code(capacity=204799)


def test_rescue_declines_branch_local_fragments_from_narrow_active_warps() -> None:
    with pytest.raises(exc.BackendUnsupported, match="shared bytes exceed 4096"):
        _code(size=16, capacity=4096)
    # The existing narrow-stage path still works when separate results fit.
    _, plan = _code(size=16)
    assert plan.warp_workspace is None


@pytest.mark.parametrize(
    "schedule",
    ["cp_async_register", "cp_async_register_reuse", "cp_async_register_reuse_scan"],
)
def test_rescue_does_not_claim_register_bridge_publication(schedule: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="shared bytes exceed"):
        _code(schedule=schedule)


@requires_cuda
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fanout", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "coalesced_unrolled"])
def test_reused_result_arena_preserves_mixed_operands_and_permuted_seeds_gpu(
    mixed: bool, fanout: bool, schedule: str
) -> None:
    required_bytes = 204800 if fanout else 137216
    if (
        CuteTcgen05Config.per_cta_smem_capacity_bytes(torch.device("cuda"))
        < required_bytes
    ):
        pytest.skip("device cannot hold this rescued result arena and operands")
    torch.manual_seed(1011)
    # Three M128 results already exceed the separate-allocation limit; more
    # stages add slow DSL compilation without exercising another lifetime.
    stages = 3
    args = _args(stages=stages, mixed=mixed, fanout=fanout, device="cuda")
    left, right, initial = args[:3]
    assert isinstance(left, torch.Tensor)
    assert isinstance(right, torch.Tensor)
    assert isinstance(initial, torch.Tensor)
    for tensor in (left, right, initial):
        tensor.normal_(std=0.0625)
    originals = tuple(tensor.clone() for tensor in (left, right, initial))
    first = (left.double() @ right.double() + initial.double()).float()
    expected = first
    dtype = torch.float16 if mixed else left.dtype
    for _step in range(stages - 1):
        expected = (
            expected.to(dtype).double() @ right.to(dtype).double()
            + expected.transpose(-1, -2).double() * 0.125
        ).float()
    if fanout:
        expected = expected + first * 0.0625
    bound = _long_chain._bind_isolated(args)
    actual = bound.compile_config(
        helion.Config(num_warps=4, cute_chained_mma_schedule=schedule)
    )(*args)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=2e-2)
    for tensor, original in zip((left, right, initial), originals, strict=True):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)
