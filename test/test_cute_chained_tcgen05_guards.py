from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_memory import _both_operands_chain
from .test_cute_chained_memory import _live_producer_chain
from .test_cute_chained_memory import _transpose_bridge_chain
import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tcgen_scan_lifetime(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    q, columns = b.shape[1], v.shape[2]
    out = torch.empty((batch, rows, columns), dtype=v.dtype, device=v.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk, qq = hl.arange(reduction), hl.arange(q)
        # The first MMA uses a raw scan leaf, not the scan result itself.
        left = (a[bi, row, kk].float() * delta[bi, row][:, None].float()).to(a.dtype)
        first = hl.dot(left, b[bi, qq, kk].T)
        decay = hl.cumsum(delta[bi, qq].float(), dim=0)
        weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
        weights *= delta[bi, qq][None, :].float()
        result = hl.dot(weights.to(v.dtype), v[bi, qq, col])
        out[bi, row, col] = result.to(v.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tcgen_domain_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, mode: hl.constexpr
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    q, columns = b.shape[1], v.shape[2]
    out = torch.empty((batch, rows, columns), dtype=v.dtype, device=v.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk, qq = hl.arange(reduction), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        if mode == "transpose":
            weights = first.T
        elif mode == "negative_permute":
            weights = first.permute(-2, -1)
        else:
            weights = first
        result = hl.dot(weights.to(v.dtype), v[bi, qq, col])
        result += v[bi, row, col].float()
        out[bi, row, col] = result.to(v.dtype)
    return out


def _args(device: Any, logical_q: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(716)
    # Zero contractions isolate the residual exactly. V has128 physically
    # valid rows even when only100 are staged for the second MMA reduction.
    return (
        torch.zeros((2, 128, 64), device=device, dtype=torch.bfloat16),
        torch.zeros((2, logical_q, 64), device=device, dtype=torch.bfloat16),
        torch.randn(
            (2, 128, 64), device=device, dtype=torch.bfloat16, generator=generator
        ),
    )


def _code(
    args: tuple[Any, ...],
    mode: str | None = "plain",
    kernel: Any = _tcgen_domain_chain,
    blocks: tuple[int, ...] = (128, 64),
) -> str:
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
        return kernel._bind_isolated(
            (*args, *((mode,) if mode is not None else ()))
        ).to_code(
            helion.Config(
                block_sizes=list(blocks),
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
            )
        )


@pytest.mark.parametrize("logical_q", [100, 128])
def test_tcgen_shared_leaf_requires_complete_domain(logical_q: int) -> None:
    code = _code(_args("cpu", logical_q))
    assert "chain_1_mma" in code
    assert ("chain_epi_input_0_values" in code) is (logical_q == 128)
    if logical_q == 100:
        assert "< 100" in code


def test_tcgen_bridge_negative_identity_permute() -> None:
    code = _code(_args("cpu", 128), "negative_permute")
    assert "OperandSource.TMEM" in code


def test_tcgen_early_staging_does_not_read_uninitialized_scan_cache() -> None:
    a, b, v = _args("cpu", 128)
    delta = torch.empty((2, 128), dtype=torch.bfloat16)
    code = _code((a, b, v, delta), None, _tcgen_scan_lifetime)
    allocation = code.index("chain_scan_0_input_0 =")
    assert code.index("chain_scan_0_input_0[") > allocation
    assert code.index("chain_0_a_load") < allocation


@pytest.mark.parametrize("case", ["transpose", "both_operands", "live_producer"])
def test_tcgen_bridge_requires_exclusive_identity_region(case: str) -> None:
    a, b, v = _args("cpu", 128)
    if case == "transpose":
        kernel, args = _transpose_bridge_chain, (a, b, v)
    elif case == "both_operands":
        kernel, args = _both_operands_chain, (a, b)
    else:
        square = torch.empty((2, 128, 128), dtype=torch.bfloat16)
        kernel, args = _live_producer_chain, (a, b, square)
    code = _code(args, None, kernel, ())
    assert "chain_0_c =" in code
    assert "OperandSource.TMEM" not in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("logical_q", [100, 128])
def test_tcgen_shared_leaf_residual_runtime(logical_q: int) -> None:
    args = _args(DEVICE, logical_q)
    before = tuple(value.clone() for value in args)
    fn = _tcgen_domain_chain._bind_isolated((*args, "plain")).compile_config(
        helion.Config(
            block_sizes=[128, 64],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
        )
    )
    actual = fn(*args, "plain")
    torch.testing.assert_close(actual, args[2], atol=0, rtol=0)
    repeated = fn(*args, "plain")
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert repeated.data_ptr() != actual.data_ptr()
    for value, saved in zip(args, before, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)
