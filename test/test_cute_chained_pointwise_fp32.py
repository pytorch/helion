from __future__ import annotations

import os
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pointwise import _args
from .test_cute_chained_pointwise import _code
from .test_cute_chained_pointwise import _config
from .test_cute_chained_pointwise import _pointwise_dot
from .test_cute_chained_tcgen05 import _compile
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _mixed_args(
    device: Any, kind: str, dtype: torch.dtype = torch.bfloat16, seed: int = 0
) -> tuple[Any, ...]:
    a, b, scale, bias, transpose = _args(device, kind, dtype)
    # Keep fractional values that would be lost by an early BF16 conversion.
    b = b.float() + 0.0038 + seed * 0.00013
    if kind == "offset":
        storage = torch.empty(b.numel() + 1, device=device, dtype=torch.float32)
        offset = storage[1:].view(b.shape)
        offset.copy_(b)
        b = offset
    elif kind == "stride":
        storage = torch.full(
            (*b.shape[:-1], b.shape[-1] * 2),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        strided = storage[..., ::2]
        strided.copy_(b)
        b = strided
    elif kind == "broadcast":
        b = b[:, :1].expand_as(b)
    return a, b, scale.float(), bias.float(), transpose


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded", "broadcast"]
)
def test_fp32_pointwise_codegen(dtype: torch.dtype, kind: str) -> None:
    code = _code(_mixed_args("cpu", kind, dtype))
    prefix = "chain_0_b_pointwise_leaf_0"
    assert (f"{prefix}_copy =" in code) == (kind != "stride")
    if kind != "stride":
        assert (
            f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom("
            "cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)"
        ) in code
        assert f"{prefix}_thread.partition_S({prefix}_source)" in code
        assert (
            f"{prefix}_values = cute.make_rmem_tensor("
            f"{prefix}_partition[None, 0, 0].shape, cutlass.Float32)"
        ) in code
        assert f"cute.copy({prefix}_copy, {prefix}_partition" in code
        assert f"{prefix}_pointer.toint() % 16 == 0" in code
        assert "for chain_0_b_step in cutlass.range" in code
    if kind in ("tail", "padded"):
        assert "< 49" in code
    if kind == "broadcast":
        assert "stride=(0, 1)" in code


def test_fp32_pointwise_disabled_preserves_scalar_loads() -> None:
    assert "_pointwise_copy =" not in _code(_mixed_args("cpu", "dense"), False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "transpose", "padded", "broadcast"])
def test_fp32_pointwise_cpu_compile(dtype: torch.dtype, kind: str) -> None:
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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "padded", "broadcast"]
)
def test_fp32_pointwise_runtime(dtype: torch.dtype, kind: str) -> None:
    run = None
    for seed in range(5):
        args = _mixed_args(DEVICE, kind, dtype, seed)
        a, b, scale, bias, transpose = args
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _pointwise_dot._bind_isolated(args).compile_config(_config())
        reduction = scale.shape[1]
        raw = (b.transpose(-2, -1) if transpose else b)[:, :reduction]
        factor = (scale.exp() * bias[:, None])[:, :, None]
        right = (raw * factor + 1.0).to(dtype)
        left = (a[:, :, :reduction].float() + 1.0).to(dtype)
        expected = (left.double() @ right.double()).to(dtype)
        actual = run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        repeated = run(*args)
        assert repeated.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        torch.testing.assert_close(args[:-1], before, atol=0, rtol=0)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[2]]
    args = _mixed_args("cpu", sys.argv[1], dtype)
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        ptx = _compile(_code(args), args, None, entry="_pointwise_dot")
    assert "ld.global.v4.b32" in ptx
    assert "st.shared.v4.b32" in ptx
