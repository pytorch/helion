from __future__ import annotations

import inspect
import linecache
import os
import sys
import tempfile
import types
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
def _tcgen_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    scale: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batch, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    for batch_tile, row, col in hl.tile([batch, m, n], block_size=[1, None, None]):
        bi = batch_tile.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        if mode != "plain":
            decay = hl.cumsum(delta[bi, qq].float(), dim=0)
            weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
            weights *= delta[bi, qq][None, :].float()
            weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        else:
            weights = first
        result = hl.dot(weights.to(a.dtype), v[bi, qq, col])
        if mode == "three":
            independent = hl.dot(a[bi, row, kk], v[bi, kk, col])
            result += independent * torch.exp(delta[bi, row].float())[:, None]
        if mode != "plain":
            result += v[bi, row, col].float() * scale[bi].float()
        out[bi, row, col] = result.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tcgen_single(
    a: torch.Tensor,
    b: torch.Tensor,
    delta: torch.Tensor,
    use_scan: hl.constexpr,
    fp32_output: hl.constexpr,
) -> torch.Tensor:
    batch, k, m = a.shape
    n = b.shape[2]
    out = torch.empty(
        (batch, m, n), device=a.device, dtype=torch.float32 if fp32_output else a.dtype
    )
    for bt, row, col in hl.tile([batch, m, n], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(k)
        if use_scan:
            scale = hl.cumsum(delta[bi, kk].float(), dim=0)
        else:
            scale = delta[bi, kk].float()
        right = (b[bi, kk, col].float() * torch.exp(scale)[:, None]).to(a.dtype)
        result = hl.dot(a[bi, kk, row].T, right)
        out[bi, row, col] = result.to(out.dtype)
    return out


def _inputs(
    device: Any,
    dtype: torch.dtype = torch.bfloat16,
    q: int = 128,
    k: int = 128,
    n: int = 64,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(78)
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, generator=generator) * 0.05
        for shape in ((2, 128, k), (2, q, k), (2, q, n), (2, q), (2,))
    )


def _code(args: tuple[torch.Tensor, ...], mode: str, n: int = 64) -> str:
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
        return _tcgen_chain._bind_isolated((*args, mode)).to_code(
            helion.Config(
                block_sizes=[128, n],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
            )
        )


def _compile(
    code: str, args: tuple[Any, ...], mode: str | None, entry: str = "_tcgen_chain"
) -> str:
    """Compile the generated kernel with fake CuTe tensors; never launch CUDA."""
    import cutlass
    import cutlass.cute as cute

    name = "_helion_tcgen_chain_cpu_test"
    path = f"<{name}>"
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    linecache.cache[path] = (len(code), None, code.splitlines(keepends=True), path)
    exec(compile(code, path, "exec"), module.__dict__)
    compiled: list[str] = []

    def capture(kernel: Any, grid: Any, *tensors: torch.Tensor, **kwargs: Any) -> None:
        parameters = list(inspect.signature(kernel.__wrapped__).parameters)
        signature = ", ".join(parameters)
        wrapper_source = (
            "@cute.jit\n"
            f"def cpu_compile({signature}):\n"
            f"    device_kernel({signature}).launch(grid=(2, 1, 1), block=(128, 1, 1))\n"
        )
        wrapper_name = name + "_wrapper"
        wrapper_path = f"<{wrapper_name}>"
        wrapper = types.ModuleType(wrapper_name)
        wrapper.__file__ = wrapper_path
        wrapper.__dict__.update(cute=cute, device_kernel=kernel)
        sys.modules[wrapper_name] = wrapper
        linecache.cache[wrapper_path] = (
            len(wrapper_source),
            None,
            wrapper_source.splitlines(keepends=True),
            wrapper_path,
        )
        exec(compile(wrapper_source, wrapper_path, "exec"), wrapper.__dict__)
        dtypes = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
        }
        fake = tuple(
            cute.runtime.make_fake_tensor(
                dtypes[tensor.dtype],
                tuple(tensor.shape),
                tuple(tensor.stride()),
                assumed_align=2,
            )
            for tensor in tensors
        )
        with tempfile.TemporaryDirectory(prefix="helion_tcgen_chain_") as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                compiled.append(
                    cute.compile(
                        wrapper.cpu_compile, *fake, options=f"--dump-dir {directory}"
                    ).__ptx__
                )
            finally:
                os.chdir(previous_directory)

    module.__dict__[entry](
        *args, *((mode,) if mode is not None else ()), _launcher=capture
    )
    assert not torch.cuda.is_initialized()
    return compiled[0]


@pytest.mark.parametrize("mode", ["plain", "scan", "three"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tcgen_chain_codegen(mode: str, dtype: torch.dtype) -> None:
    code = _code(_inputs("cpu", dtype), mode)
    assert "OperandSource.TMEM" in code
    assert "tcgen05.St32x32bOp" in code
    assert "chain_tptr.toint()" in code
    assert "chain_store_copy" in code
    assert "chain_allocator.free" in code
    if mode == "scan":
        assert "chain_epi_input_0_values" in code
        assert "chain_scan_0_input_0" in code
    if mode == "three":
        assert "chain_1_c =" in code


@pytest.mark.parametrize(
    "q,k,n", [(32, 16, 32), (64, 64, 64), (128, 128, 128), (256, 16, 32)]
)
def test_tcgen_chain_widths_codegen(q: int, k: int, n: int) -> None:
    assert "OperandSource.TMEM" in _code(_inputs("cpu", q=q, k=k, n=n), "plain", n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["plain", "scan", "three"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tcgen_chain_correctness(mode: str, dtype: torch.dtype) -> None:
    args = _inputs(DEVICE, dtype)
    a, b, v, delta, scale = args
    saved = tuple(arg.clone() for arg in args)
    first = a.float() @ b.float().transpose(-1, -2)
    if mode in ("scan", "three"):
        decay = delta.float().cumsum(-1)
        first = (
            first
            * torch.exp(decay[:, :, None] - decay[:, None, :])
            * delta[:, None, :].float()
        )
        first = torch.where(
            torch.ones((128, 128), device=DEVICE, dtype=torch.bool).tril(), first, 0.0
        )
    expected = first.to(dtype).float() @ v.float()
    if mode == "three":
        expected += (a.float() @ v.float()) * torch.exp(delta.float())[:, :, None]
    if mode in ("scan", "three"):
        expected += v.float() * scale[:, None, None].float()
    bound = _tcgen_chain._bind_isolated((*args, mode))
    compiled = bound.compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    actual = compiled(*args, mode)
    repeated = compiled(*args, mode)
    torch.testing.assert_close(actual, expected.to(dtype), atol=0.005, rtol=0.02)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    assert actual.data_ptr() != repeated.data_ptr()
    for before, after in zip(saved, args, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def _single_code(args: tuple[Any, ...]) -> str:
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
        return _tcgen_single._bind_isolated(args).to_code(
            helion.Config(
                block_sizes=[128, 64],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
            )
        )


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_tcgen_single_codegen(scan: bool, fp32: bool) -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((2, 64, 128), (2, 64, 64), (2, 64))
    )
    code = _single_code((*args, scan, fp32))
    assert "OperandSource.SMEM" in code
    assert "OperandMajorMode.MN" in code
    assert "St32x32bOp" not in code
    if scan:
        assert code.index("chain_scan_0_values =") < code.index("chain_0_b_load")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "q,k,n", [(32, 16, 32), (64, 64, 64), (128, 128, 128), (256, 16, 32)]
)
def test_tcgen_widths_correctness(q: int, k: int, n: int) -> None:
    args = _inputs(DEVICE, q=q, k=k, n=n)
    a, b, v, _, _ = args
    expected = (
        (a.float() @ b.float().transpose(-1, -2)).to(a.dtype).float() @ v.float()
    ).to(a.dtype)
    compiled = _tcgen_chain._bind_isolated((*args, "plain")).compile_config(
        helion.Config(
            block_sizes=[128, n], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    torch.testing.assert_close(
        compiled(*args, "plain"), expected, atol=0.005, rtol=0.02
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_tcgen_single_correctness(scan: bool, fp32: bool) -> None:
    torch.manual_seed(38)
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.02
        for shape in ((2, 64, 128), (2, 64, 64), (2, 64))
    )
    a, b, delta = args
    scale = delta.float().cumsum(-1) if scan else delta.float()
    expected = (
        a.float().transpose(-1, -2)
        @ (b.float() * scale.exp()[:, :, None]).to(a.dtype).float()
    )
    if not fp32:
        expected = expected.to(a.dtype)
    compiled = _tcgen_single._bind_isolated((*args, scan, fp32)).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    saved = tuple(arg.clone() for arg in args)
    actual = compiled(*args, scan, fp32)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
    torch.testing.assert_close(actual, compiled(*args, scan, fp32), atol=0, rtol=0)
    for before, after in zip(saved, args, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tcgen_chain_many_ctas() -> None:
    # Enough independent CTAs to receive nonzero TMEM allocation bases.
    torch.manual_seed(927)
    args = tuple(
        torch.randint(-3, 4, shape, device=DEVICE).to(torch.bfloat16) / 16
        for shape in (
            (512, 128, 128),
            (512, 128, 128),
            (512, 128, 64),
            (512, 128),
            (512,),
        )
    )
    a, b, v, _, _ = args
    expected = (
        (a.float() @ b.float().transpose(-1, -2)).to(a.dtype).float() @ v.float()
    ).to(a.dtype)
    fn = _tcgen_chain._bind_isolated((*args, "plain")).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    for _ in range(5):
        torch.testing.assert_close(fn(*args, "plain"), expected, atol=0, rtol=0)
