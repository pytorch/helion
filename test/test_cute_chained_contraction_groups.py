from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from benchmarks.cute.kda_prefill_kernels import kda_chunk_recurrence_fp32
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop import _kda_args
from .test_cute_chained_loop import _kda_config
import helion
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_contraction_groups import contraction_groups
from helion._compiler.cute.chained_tcgen_stage import stage_geometry
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo


def test_kda_physical_common_operand_groups() -> None:
    original = chained_matmul.plan_chained_matmul
    decisions: list[tuple[tuple[int, ...], ...]] = []

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
            assert all(geometry is not None for geometry in geometries)
            admitted = tuple(
                geometry for geometry in geometries if geometry is not None
            )
            decisions.append(
                tuple(group.stages for group in contraction_groups(plan, admitted))
            )
        return plan

    with (
        _cpu_codegen(),
        patch.object(chained_matmul, "plan_chained_matmul", side_effect=observe),
    ):
        bound = kda_chunk_recurrence_fp32._bind_isolated(_kda_args())
        bound.to_code(_kda_config("tcgen05_tmem"))
    assert decisions == [((0, 1), (2, 3))]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fanout_loop(left, first_rhs, second_rhs, initial, steps: int):
    steps = hl.specialize(steps)
    _, m, k = left.shape
    n = second_rhs.shape[-1]
    history = torch.empty(
        (max(steps, 1), m, first_rhs.shape[-1]), dtype=torch.float32, device=left.device
    )
    final = torch.empty_like(initial)
    for rows, columns in hl.tile([m, n], block_size=[64, n]):
        state = initial[rows, columns]
        for step in hl.tile(steps, block_size=1):
            kk, short_columns = hl.arange(k), hl.arange(first_rhs.shape[-1])
            common = left[step.id, rows, kk]
            first = hl.dot(
                common, first_rhs[step.id, kk, short_columns], out_dtype=torch.float32
            )
            state = hl.dot(
                common,
                second_rhs[step.id, kk, columns],
                acc=state,
                out_dtype=torch.float32,
            )
            history[step.id, rows, short_columns] = first
        final[rows, columns] = state
    return history, final


def _fanout_args(
    device: str | torch.device, n: int, steps: int, dtype: torch.dtype
) -> tuple:
    return (
        torch.randn((max(steps, 1), 64, 16), device=device, dtype=dtype) * 0.1,
        torch.randn((max(steps, 1), 16, 16), device=device, dtype=dtype) * 0.1,
        torch.randn((max(steps, 1), 16, n), device=device, dtype=dtype) * 0.1,
        torch.randn((64, n), device=device, dtype=torch.float32) * 0.1,
        steps,
    )


def _group_config() -> helion.Config:
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
    )


@pytest.mark.parametrize("n", [32, 128])
def test_grouped_fanout_cpu_source(n: int) -> None:
    with _cpu_codegen():
        bound = _fanout_loop._bind_isolated(_fanout_args("cpu", n, 3, torch.bfloat16))
        source = bound.to_code(_group_config())
    assert "chain_0_mma" in source and "chain_1_mma" not in source
    assert f"(128, {16 + n})" in source
    assert "chain_0_seed" in source
    assert "chain_1_c" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("n", [32, 128])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_fanout_gpu(n: int, steps: int, dtype: torch.dtype) -> None:
    torch.manual_seed(815)
    args = _fanout_args(DEVICE, n, steps, dtype)
    left, first_rhs, second_rhs, initial, _ = args
    frozen = tuple(value.clone() for value in args[:-1])
    bound = _fanout_loop._bind_isolated(args)
    source = bound.to_code(_group_config())
    assert "chain_0_mma" in source and "chain_1_mma" not in source
    compiled = bound.compile_config(_group_config())
    history, final = compiled(*args)
    expected = initial.clone()
    for step in range(steps):
        expected += left[step].float() @ second_rhs[step].float()
        torch.testing.assert_close(
            history[step],
            left[step].float() @ first_rhs[step].float(),
            rtol=5e-4,
            atol=5e-4,
        )
    torch.testing.assert_close(final, expected, rtol=5e-4, atol=5e-4)
    history_again, final_again = compiled(*args)
    torch.testing.assert_close(final_again, final, rtol=0, atol=0)
    if steps:
        torch.testing.assert_close(
            history_again[:steps], history[:steps], rtol=0, atol=0
        )
    for actual, original in zip(args[:-1], frozen, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
