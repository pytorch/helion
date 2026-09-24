from __future__ import annotations

import os
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_tcgen05 import _compile
import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    transpose: hl.constexpr,
) -> torch.Tensor:
    batch, rows, physical_reduction = a.shape
    reduction = scale.shape[1]
    columns = b.shape[1] if transpose else b.shape[2]
    out = torch.empty((batch, rows, columns), device=a.device, dtype=a.dtype)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(reduction)
        if transpose:
            right = b[bi, col, kk].T
        else:
            right = b[bi, kk, col]
        factor = torch.exp(scale[bi, kk].float()) * bias[bi].float()
        right = (right.float() * factor[:, None] + 1.0).to(a.dtype)
        left = (a[bi, row, kk].float() + 1.0).to(a.dtype)
        out[bi, row, col] = hl.dot(left, right).to(a.dtype)
    return out


def _args(
    device: Any, kind: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[Any, ...]:
    reduction = 49 if kind in ("tail", "padded") else 128
    physical = 64 if kind == "padded" else reduction
    generator = torch.Generator(device=device).manual_seed(936)
    a = (
        torch.randn((2, 128, physical), generator=generator, device=device, dtype=dtype)
        * 0.1
    )
    transpose = kind == "transpose"
    shape = (2, 64, physical) if transpose else (2, physical, 64)
    if kind == "stride":
        b = torch.randn(
            (*shape[:-1], shape[-1] * 2),
            generator=generator,
            device=device,
            dtype=dtype,
        )[..., ::2]
    elif kind == "offset":
        storage = torch.randn(
            (2 * reduction * 64 + 1,), generator=generator, device=device, dtype=dtype
        )
        b = storage[1:].view(shape)
    else:
        b = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    scale = (
        torch.randn((2, reduction), generator=generator, device=device, dtype=dtype)
        * 0.05
    )
    bias = torch.randn((2,), generator=generator, device=device, dtype=dtype) * 0.1
    return a, b, scale, bias, transpose


def _config(enabled: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=enabled,
    )


def _code(
    args: tuple[Any, ...], enabled: bool = True, kernel: Any = _pointwise_dot
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
        return kernel._bind_isolated(args).to_code(_config(enabled))


@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded"]
)
def test_pointwise_codegen(kind: str) -> None:
    code = _code(_args("cpu", kind))
    assert ("chain_0_b_pointwise_copy =" in code) == (kind != "stride")
    if kind != "stride":
        assert "_pointwise_leaf_0_pointer.toint() % 16 == 0" in code
        assert ".layout.stride[" in code
        assert "_pointwise_element in cutlass.range_constexpr(8)" in code
        assert "for chain_0_b_step in cutlass.range" in code  # Masked fallback.
    if kind in ("tail", "padded"):
        assert "< 49" in code


def test_pointwise_default_retains_scalar_staging() -> None:
    assert "_pointwise_copy =" not in _code(_args("cpu", "dense"), False)


def test_pointwise_broadcast_factor_is_outside_vector_loop() -> None:
    code = _code(_args("cpu", "dense"))
    body = code.split("for chain_0_b_pointwise_step", 1)[1].split("else:", 1)[0]
    outside, inside = body.split("for chain_0_b_pointwise_element", 1)
    assert "cute.math.exp2" in outside
    assert "cute.math.exp2" not in inside


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "transpose", "padded"])
def test_pointwise_real_cpu_compile(dtype: torch.dtype, kind: str) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, kind, str(dtype).split(".")[-1]],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pointwise_runtime(kind: str, dtype: torch.dtype) -> None:
    args = _args(DEVICE, kind, dtype)
    a, b, scale, bias, transpose = args
    frozen = tuple(value.clone() for value in args[:-1])
    reduction = scale.shape[1]
    right = (b.transpose(-2, -1) if transpose else b)[:, :reduction]
    right = (
        right.float() * (scale.float().exp() * bias.float()[:, None])[:, :, None] + 1
    ).to(dtype)
    left = (a[:, :, :reduction].float() + 1.0).to(dtype)
    expected = (left.float() @ right.float()).to(dtype)
    run = _pointwise_dot._bind_isolated(args).compile_config(_config())
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
    repeated = run(*args)
    assert actual.data_ptr() != repeated.data_ptr()
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    for before, value in zip(frozen, args[:-1], strict=True):
        torch.testing.assert_close(before, value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pointwise_reused_callable_checks_layout_and_alignment() -> None:
    args = _args(DEVICE, "dense")
    run = _pointwise_dot._bind_isolated(args).compile_config(_config())
    for kind in ("dense", "stride", "offset"):
        current = _args(DEVICE, kind)
        a, b, scale, bias, _ = current
        right = (
            b.float() * (scale.float().exp() * bias.float()[:, None])[:, :, None] + 1
        ).to(a.dtype)
        expected = ((a.float() + 1).to(a.dtype).float() @ right.float()).to(a.dtype)
        torch.testing.assert_close(run(*current), expected, atol=0.015, rtol=0.015)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[2]]
    args = _args("cpu", sys.argv[1], dtype)
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        ptx = _compile(_code(args), args, None, entry="_pointwise_dot")
    assert "ld.global.v4.b32" in ptx
    assert "st.shared.v4.b32" in ptx
