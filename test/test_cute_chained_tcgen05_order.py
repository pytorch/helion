from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _past_first(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    batch, chunks, k, n = state.shape
    length = v.shape[1] // chunks
    out = torch.empty_like(v)
    for task, row, col in hl.tile(
        [batch * chunks, length, n], block_size=[1, None, None]
    ):
        bi, ci = task.begin // chunks, task.begin % chunks
        kk, qq = hl.arange(k), hl.arange(length)
        r, q = ci * length + row.index, ci * length + qq
        left = a[bi, r, kk]
        past = hl.dot(left, state[bi, ci, kk, col])
        cb = hl.dot(left, b[bi, q, kk].T)
        local = hl.dot(cb.to(a.dtype), v[bi, q, col])
        out[bi, r, col] = (local + past + v[bi, r, col].float()).to(a.dtype)
    return out


def _args(device: Any, chunks: int, batch: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(74)
    args = tuple(
        torch.randint(-3, 4, shape, device=device, generator=generator).to(
            torch.bfloat16
        )
        / 16
        for shape in (
            (batch, chunks * 128, 128),
            (batch, chunks * 128, 128),
            (batch, chunks * 128, 64),
            (batch, chunks, 128, 64),
        )
    )
    args[3][:, 0].zero_()
    return args


def test_past_first_codegen() -> None:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        code = _past_first._bind_isolated(_args("cpu", 2)).to_code(
            helion.Config(
                block_sizes=[128, 64],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
            )
        )
    assert "chain_0_c =" in code
    assert "chain_2_bridge" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch,chunks", [(2, 1), (2, 2), (256, 2)])
def test_past_first_exact(batch: int, chunks: int) -> None:
    args = _args(DEVICE, chunks, batch)
    a, b, v, state = args
    left = a.float().view(batch, chunks, 128, 128)
    right = b.float().view(batch, chunks, 128, 128)
    value = v.float().view(batch, chunks, 128, 64)
    cb = (left @ right.transpose(-1, -2)).to(a.dtype).float()
    expected = (cb @ value + left @ state.float() + value).to(a.dtype).reshape_as(v)
    before = tuple(arg.clone() for arg in args)
    fn = _past_first._bind_isolated(args).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    for _ in range(5):
        actual = fn(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for current, saved in zip(args, before, strict=True):
        torch.testing.assert_close(current, saved, atol=0, rtol=0)
