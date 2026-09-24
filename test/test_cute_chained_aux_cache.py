from __future__ import annotations

import os
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pointwise import _code as _pointwise_code
from .test_cute_chained_tcgen05 import _compile
from .test_cute_chained_tcgen05 import _inputs as _chain_inputs
from .test_cute_chained_tcgen05 import _tcgen_chain
import helion
from helion._compiler.cute.chained_aux_cache import make_late_auxiliary_cache
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _auxiliary_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    q, columns = b.shape[1], v.shape[2]
    out = torch.empty((batch, rows, columns), dtype=a.dtype, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk, qq = hl.arange(reduction), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        if mode == "reverse":
            factor = delta[bi, q - 1 - qq].float()
        else:
            factor = delta[bi, qq].float()
        weights = (
            first * torch.exp(factor)[None, :] * weight[bi, qq][None, :].float()
        ).to(a.dtype)
        result = hl.dot(weights, v[bi, qq, col])
        if mode == "not_last":
            result += hl.dot(a[bi, row, kk], v[bi, kk, col])
        result += delta[bi, delta.shape[1] - 1].float()
        out[bi, row, col] = result.to(a.dtype)
    return out


def _args(
    device: Any, mode: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[Any, ...]:
    q = 100 if mode == "tail" else 128
    generator = torch.Generator(device=device).manual_seed(967)
    return (
        torch.randn((2, 128, 128), device=device, dtype=dtype, generator=generator)
        * 0.03,
        torch.randn((2, q, 128), device=device, dtype=dtype, generator=generator)
        * 0.03,
        torch.randn((2, 128, 64), device=device, dtype=dtype, generator=generator)
        * 0.03,
        torch.randn((2, 128), device=device, dtype=torch.float32, generator=generator)
        * 0.1,
        torch.randn((2, 128), device=device, dtype=dtype, generator=generator) * 0.1,
        mode,
    )


def _config(enabled: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_auxiliary_cache=enabled,
    )


def _code(
    args: tuple[Any, ...], enabled: bool = True, kernel: Any = _auxiliary_chain
) -> str:
    # Reuse the central CPU hardware guard; only replace the selected config.
    with patch(
        "test.test_cute_chained_pointwise._config", return_value=_config(enabled)
    ):
        return _pointwise_code(args, kernel=kernel)


@pytest.mark.parametrize("mode", ["plain", "reverse", "tail", "not_last"])
def test_auxiliary_cache_codegen(mode: str) -> None:
    source = _code(_args("cpu", mode))
    assert ("chain_late_aux_0 =" in source) == (mode != "not_last")
    if mode != "not_last":
        assert "cute.recast_ptr(chain_a_workspace + 0, dtype=cutlass.Float32)" in source
        assert (
            "cute.recast_ptr(chain_a_workspace + 256, dtype=cutlass.BFloat16)" in source
        )
        fill = source.index("chain_late_aux_0 =")
        bridge = source.index("chain_1_bridge_layout =")
        assert source.rfind("cute.arch.sync_threads()", 0, fill) > 0
        assert source.index("cute.arch.sync_threads()", fill) < bridge
        assert "cutlass.Float32(chain_late_aux_0[" in source[bridge:]
    if mode == "tail":
        # Scalar index127 must not use a vector whose logical extent is100.
        assert "127" in source and "< 100" in source


def test_auxiliary_cache_disabled_is_unchanged() -> None:
    assert "chain_late_aux_0 =" not in _code(_args("cpu", "plain"), False)


def test_auxiliary_cache_has_no_new_shared_allocation() -> None:
    args = _args("cpu", "plain")
    allocations = [
        [line for line in _code(args, enabled).splitlines() if "alloc_smem(" in line]
        for enabled in (False, True)
    ]
    assert allocations[0] == allocations[1]


def test_auxiliary_cache_respects_borrowed_capacity() -> None:
    def bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs["arena_bytes"] = 128
        return make_late_auxiliary_cache(*args, **kwargs)

    with patch(
        "helion._compiler.cute.chained_tcgen05.make_late_auxiliary_cache",
        side_effect=bounded,
    ):
        assert "chain_late_aux_0 =" not in _code(_args("cpu", "plain"))


def test_auxiliary_cache_reuses_existing_scan_leaf() -> None:
    args = (*_chain_inputs("cpu"), "scan")
    source = _code(args, kernel=_tcgen_chain)
    assert "chain_scan_0_input_0" in source
    assert "chain_late_aux_0 =" not in source


@pytest.mark.parametrize("mode", ["plain", "reverse", "tail"])
def test_auxiliary_cache_real_cpu_compile(mode: str) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, mode],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["plain", "reverse", "tail", "not_last"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_auxiliary_cache_runtime(mode: str, dtype: torch.dtype) -> None:
    args = _args(DEVICE, mode, dtype)
    a, b, v, delta, weight, _ = args
    frozen = tuple(value.clone() for value in args[:-1])
    q = b.shape[1]
    factor = delta[:, :q].flip(-1) if mode == "reverse" else delta[:, :q]
    weights = (
        (a.float() @ b.float().transpose(-2, -1))
        * factor.exp()[:, None, :]
        * weight[:, None, :q].float()
    ).to(dtype)
    expected = weights.float() @ v[:, :q].float()
    if mode == "not_last":
        expected += a.float() @ v.float()
    expected = (expected + delta[:, -1, None, None]).to(dtype)
    run = _auxiliary_chain._bind_isolated(args).compile_config(_config())
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.015)
    repeated = run(*args)
    assert actual.data_ptr() != repeated.data_ptr()
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    for before, value in zip(frozen, args[:-1], strict=True):
        torch.testing.assert_close(before, value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auxiliary_cache_runtime_strides_offsets_and_readonly_aliases() -> None:
    args = _args(DEVICE, "plain")
    run = _auxiliary_chain._bind_isolated(args).compile_config(_config())
    a, b, v, delta, weight, mode = args
    for offset in (0, 1):
        storage = torch.randn((2, 257), device=DEVICE, dtype=delta.dtype) * 0.1
        changed_delta = storage[:, offset : offset + 256 : 2]
        weight_storage = torch.randn((257,), device=DEVICE, dtype=weight.dtype) * 0.1
        changed_weight = weight_storage[offset : offset + 256].view(2, 128)
        changed = (a, b, v, changed_delta, changed_weight, mode)
        weights = (
            (a.float() @ b.float().transpose(-2, -1))
            * changed_delta.exp()[:, None, :]
            * changed_weight[:, None, :].float()
        ).to(a.dtype)
        expected = (weights.float() @ v.float() + changed_delta[:, -1, None, None]).to(
            a.dtype
        )
        torch.testing.assert_close(run(*changed), expected, atol=0.003, rtol=0.015)
    aliased = (*args[:3], weight, weight, mode)
    alias_run = _auxiliary_chain._bind_isolated(aliased).compile_config(_config())
    before = weight.clone()
    weights = (
        (a.float() @ b.float().transpose(-2, -1))
        * weight.float().exp()[:, None, :]
        * weight[:, None, :].float()
    ).to(a.dtype)
    expected = (weights.float() @ v.float() + weight[:, -1, None, None].float()).to(
        a.dtype
    )
    torch.testing.assert_close(alias_run(*aliased), expected, atol=0.003, rtol=0.015)
    torch.testing.assert_close(weight, before, atol=0, rtol=0)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    args = _args("cpu", sys.argv[1])
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        ptx = _compile(_code(args), args, None, entry="_auxiliary_chain")
    assert "tcgen05.mma" in ptx
