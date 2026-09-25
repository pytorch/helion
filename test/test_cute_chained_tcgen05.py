from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ).start()

import ast
import inspect
import linecache
import math
import operator
import re
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace
from typing import Any

import pytest
import sympy
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_matmul import _copy_code
from helion._compiler.cute.chained_matmul import _copy_index
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_matmul import _signed_floor_adjustment
from helion._compiler.cute.chained_matmul import _signed_remainder_adjustment
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = skipUnlessBackends(["cute"])


# Tcgen05.


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


def _tcgen_inputs(
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


def _tcgen_code(args: tuple[torch.Tensor, ...], mode: str, n: int = 64) -> str:
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


def _tcgen_compile(
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
    code = _tcgen_code(_tcgen_inputs("cpu", dtype), mode)
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
    assert "OperandSource.TMEM" in _tcgen_code(
        _tcgen_inputs("cpu", q=q, k=k, n=n), "plain", n
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["plain", "scan", "three"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tcgen_chain_correctness(mode: str, dtype: torch.dtype) -> None:
    args = _tcgen_inputs(DEVICE, dtype)
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
    args = _tcgen_inputs(DEVICE, q=q, k=k, n=n)
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


def _tcgen_args(device: Any, chunks: int, batch: int = 2) -> tuple[torch.Tensor, ...]:
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
        code = _past_first._bind_isolated(_tcgen_args("cpu", 2)).to_code(
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
    args = _tcgen_args(DEVICE, chunks, batch)
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


# Memory.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _memory_chain(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _computed_memory_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(
            (a[bi, row, kk].float() * 0.5).to(a.dtype),
            (b[bi, qq, kk].float() * 0.75).to(a.dtype).T,
        )
        out[bi, row, col] = hl.dot(
            first.to(a.dtype), (v[bi, qq, col].float() * 0.25).to(a.dtype)
        ).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gathered_memory_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    permutation_k: torch.Tensor,
    permutation_q: torch.Tensor,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk = permutation_k[hl.arange(k)]
        qq = hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, permutation_q[qq], col]).to(
            a.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _both_operands_chain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    out = torch.empty((batches, m, m), device=a.device, dtype=a.dtype)
    for batch in hl.tile(batches, block_size=1):
        bi = batch.begin
        qq = hl.arange(m)
        kk = hl.arange(k)
        first = hl.dot(a[bi, qq, kk], b[bi, qq, kk].T)
        converted = first.to(a.dtype)
        out[bi, :, :] = hl.dot(converted, converted).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_bridge_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    n = v.shape[2]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch in hl.tile(batches, block_size=1):
        bi = batch.begin
        qq = hl.arange(m)
        kk = hl.arange(k)
        nn = hl.arange(n)
        first = hl.dot(a[bi, qq, kk], b[bi, qq, kk].T)
        out[bi, :, :] = hl.dot(first.T.to(a.dtype), v[bi, qq, nn]).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _live_producer_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch in hl.tile(batches, block_size=1):
        bi = batch.begin
        mm = hl.arange(m)
        kk = hl.arange(k)
        qq = hl.arange(q)
        nn = hl.arange(n)
        first = hl.dot(a[bi, mm, kk], b[bi, qq, kk].T)
        result = hl.dot(first.to(a.dtype), v[bi, qq, nn])
        out[bi, :, :] = (result + first).to(a.dtype)
    return out


def _view(shape: tuple[int, ...], layout: str, device: Any) -> torch.Tensor:
    batches, rows, cols = shape
    if layout.startswith("offset"):
        offset = int(layout.removeprefix("offset"))
        storage = torch.randn(
            math.prod(shape) + offset, device=device, dtype=torch.bfloat16
        )
        value = storage[offset:].view(shape)
    elif layout.startswith("padded"):
        padding = int(layout.removeprefix("padded"))
        value = torch.randn(
            (batches, rows, cols + padding), device=device, dtype=torch.bfloat16
        )[..., :cols]
    elif layout == "inner_stride2":
        value = torch.randn(
            (batches, rows, cols * 2), device=device, dtype=torch.bfloat16
        )[..., ::2]
    elif layout == "transpose":
        value = torch.randn(
            (batches, cols, rows), device=device, dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        assert layout == "contiguous"
        value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    return value.mul_(0.1)


def _memory_inputs(
    layout: str, device: Any, *, tails: bool = False
) -> tuple[torch.Tensor, ...]:
    m, k, q, n = (35, 64, 49, 24) if tails else (32, 64, 64, 32)
    return tuple(
        _view(shape, layout, device) for shape in ((2, m, k), (2, q, k), (2, q, n))
    )


def _memory_config(
    warps: int, schedule: str = "cp_async", *, full: bool = False
) -> helion.Config:
    return helion.Config(
        block_sizes=[] if full else [16, 32],
        num_warps=warps,
        cute_chained_mma_schedule=schedule,
    )


def _memory_reference(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    first = (a.double() @ b.double().transpose(-1, -2)).to(a.dtype)
    return (first.double() @ v.double()).to(a.dtype)


def _memory_check(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
) -> None:
    frozen = tuple(arg.clone() for arg in args)
    first = fn(*args)
    saved = first.clone()
    second = fn(*args)
    torch.testing.assert_close(first, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(second, first, atol=0, rtol=0)
    torch.testing.assert_close(first, saved, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()
    for value, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


def _memory_cpu_code(
    kernel: Any,
    args: tuple[torch.Tensor, ...],
    warps: int,
    schedule: str = "cp_async",
    *,
    full: bool = False,
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
        kernel.reset()
        return kernel.bind(args).to_code(_memory_config(warps, schedule, full=full))


def test_async_memory_codegen_patches_already_imported_support() -> None:
    with patch.object(
        chained_matmul,
        "get_cute_mma_support",
        side_effect=AssertionError("unpatched previously imported binding"),
    ) as original:
        code = _memory_cpu_code(_memory_chain, _memory_inputs("contiguous", "cpu"), 4)
        assert "chain_0_mma" in code
        original.assert_not_called()
        assert chained_matmul.get_cute_mma_support is original


@pytest.mark.parametrize("warps", [1, 2, 4])
@pytest.mark.parametrize(
    "layout,expected_async",
    [
        ("contiguous", True),
        ("offset1", True),
        ("offset8", True),
        ("padded1", False),
        ("padded8", True),
        ("inner_stride2", False),
        ("transpose", True),
    ],
)
def test_async_memory_codegen_guards(
    layout: str, expected_async: bool, warps: int
) -> None:
    code = _memory_cpu_code(_memory_chain, _memory_inputs(layout, "cpu"), warps)
    assert "_helion_cute_pointer_alignment = 1" in code
    assert ("CopyG2SOp" in code) is expected_async
    guards = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If) and ".toint() % 16" in ast.unparse(node.test)
    ]
    assert bool(guards) is expected_async
    for guard in guards:
        condition = ast.unparse(guard.test)
        assert "chain_thread" not in condition
        assert ".load()" not in condition
        assert "0 <=" in condition
        assert "CopyG2SOp" in ast.unparse(guard)
        assert ".load()" in "\n".join(ast.unparse(node) for node in guard.orelse)
    if expected_async:
        assert "cp_async_wait_group(0)" in code


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_codegen_tail_bounds(warps: int) -> None:
    code = _memory_cpu_code(
        _memory_chain, _memory_inputs("contiguous", "cpu", tails=True), warps
    )
    assert "CopyG2SOp" in code
    guards = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If) and ".toint() % 16" in ast.unparse(node.test)
    ]
    conditions = "\n".join(ast.unparse(node.test) for node in guards)
    assert "< 35" in conditions
    assert "< 49" in conditions
    assert ".load()" in code


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_codegen_computed_and_gathered_fallback(warps: int) -> None:
    args = _memory_inputs("contiguous", "cpu")
    code = _memory_cpu_code(_computed_memory_chain, args, warps)
    assert "chain_0_mma" in code
    assert "CopyG2SOp" not in code
    gathered = (*args, torch.randperm(64), torch.randperm(64))
    code = _memory_cpu_code(_gathered_memory_chain, gathered, warps)
    assert "chain_0_mma" in code
    assert "CopyG2SOp" not in code


@pytest.mark.parametrize("warps", [1, 2, 4])
@pytest.mark.parametrize(
    "layout", ["offset1", "offset8", "padded1", "padded8", "inner_stride2", "transpose"]
)
def test_async_memory_views(layout: str, warps: int) -> None:
    args = _memory_inputs(layout, DEVICE)
    _memory_chain.reset()
    fn = _memory_chain.bind(args).compile_config(_memory_config(warps))
    _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_alignment_is_checked_on_every_call(warps: int) -> None:
    aligned = _memory_inputs("contiguous", DEVICE)
    _memory_chain.reset()
    fn = _memory_chain.bind(aligned).compile_config(_memory_config(warps))
    _memory_check(fn, aligned, _memory_reference(*aligned))
    for offset in (1, 8):
        args = _memory_inputs(f"offset{offset}", DEVICE)
        assert all(
            value.stride() == old.stride()
            for value, old in zip(args, aligned, strict=True)
        )
        _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_strides_are_checked_on_every_call(warps: int) -> None:
    aligned = _memory_inputs("contiguous", DEVICE)
    _memory_chain.reset()
    fn = _memory_chain.bind(aligned).compile_config(_memory_config(warps))
    _memory_check(fn, aligned, _memory_reference(*aligned))
    for layout in ("padded8", "transpose", "inner_stride2"):
        args = _memory_inputs(layout, DEVICE)
        assert all(
            value.shape == old.shape for value, old in zip(args, aligned, strict=True)
        )
        _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_codegen_exclusive_region(warps: int) -> None:
    args = tuple(_view((2, 32, 64), "contiguous", "cpu") for _ in range(2))
    square = _view((2, 32, 32), "contiguous", "cpu")
    for kernel, values in (
        (_both_operands_chain, args),
        (_transpose_bridge_chain, (*args, square)),
        (_live_producer_chain, (*args, square)),
    ):
        code = _memory_cpu_code(kernel, values, warps, "cp_async_register", full=True)
        assert "chain_0_c_ptr" in code
        assert "_bridge_values" not in code


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_codegen_narrow_producer(warps: int) -> None:
    args = tuple(
        _view(shape, "contiguous", "cpu")
        for shape in ((2, 32, 64), (2, 16, 64), (2, 16, 32))
    )
    code = _memory_cpu_code(_memory_chain, args, warps, "cp_async_register")
    assert ("chain_1_a_bridge_values" in code) is (warps != 4)
    assert ("chain_0_c_ptr" in code) is (warps == 4)


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_exclusive_region(warps: int) -> None:
    a, b = (_view((2, 32, 64), "contiguous", DEVICE) for _ in range(2))
    v = _view((2, 32, 32), "contiguous", DEVICE)
    first = a.double() @ b.double().transpose(-1, -2)
    converted = first.to(a.dtype).double()
    cases = (
        (_both_operands_chain, (a, b), (converted @ converted).to(a.dtype)),
        (
            _transpose_bridge_chain,
            (a, b, v),
            (converted.transpose(-1, -2) @ v.double()).to(a.dtype),
        ),
        (_live_producer_chain, (a, b, v), (converted @ v.double() + first).to(a.dtype)),
    )
    for kernel, args, expected in cases:
        kernel.reset()
        fn = kernel.bind(args).compile_config(
            _memory_config(warps, "cp_async_register", full=True)
        )
        _memory_check(fn, args, expected)


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_tail_tiles(warps: int) -> None:
    args = _memory_inputs("contiguous", DEVICE, tails=True)
    _memory_chain.reset()
    fn = _memory_chain.bind(args).compile_config(
        _memory_config(warps, "cp_async_register")
    )
    _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_tail_tiles(warps: int) -> None:
    args = _memory_inputs("contiguous", DEVICE, tails=True)
    _memory_chain.reset()
    fn = _memory_chain.bind(args).compile_config(_memory_config(warps))
    _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_computed_and_gathered_fallback(warps: int) -> None:
    args = _memory_inputs("contiguous", DEVICE)
    _computed_memory_chain.reset()
    fn = _computed_memory_chain.bind(args).compile_config(_memory_config(warps))
    transformed = tuple(
        (value.float() * scale).to(value.dtype)
        for value, scale in zip(args, (0.5, 0.75, 0.25), strict=True)
    )
    _memory_check(fn, args, _memory_reference(*transformed))
    pk, pq = (torch.randperm(64, device=DEVICE) for _ in range(2))
    gathered = (*args, pk, pq)
    _gathered_memory_chain.reset()
    fn = _gathered_memory_chain.bind(gathered).compile_config(_memory_config(warps))
    a, b, v = args
    _memory_check(fn, gathered, _memory_reference(a[..., pk], b[..., pk], v[:, pq, :]))


# Tcgen05 guards.


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


def _tcgen_guard_args(device: Any, logical_q: int) -> tuple[torch.Tensor, ...]:
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


def _tcgen_guard_code(
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
    code = _tcgen_guard_code(_tcgen_guard_args("cpu", logical_q))
    assert "chain_1_mma" in code
    assert ("chain_epi_input_0_values" in code) is (logical_q == 128)
    if logical_q == 100:
        assert "< 100" in code


def test_tcgen_bridge_negative_identity_permute() -> None:
    code = _tcgen_guard_code(_tcgen_guard_args("cpu", 128), "negative_permute")
    assert "OperandSource.TMEM" in code


def test_tcgen_early_staging_does_not_read_uninitialized_scan_cache() -> None:
    a, b, v = _tcgen_guard_args("cpu", 128)
    delta = torch.empty((2, 128), dtype=torch.bfloat16)
    code = _tcgen_guard_code((a, b, v, delta), None, _tcgen_scan_lifetime)
    allocation = code.index("chain_scan_0_input_0 =")
    assert code.index("chain_scan_0_input_0[") > allocation
    assert code.index("chain_0_a_load") < allocation


@pytest.mark.parametrize("case", ["transpose", "both_operands", "live_producer"])
def test_tcgen_bridge_requires_exclusive_identity_region(case: str) -> None:
    a, b, v = _tcgen_guard_args("cpu", 128)
    if case == "transpose":
        kernel, args = _transpose_bridge_chain, (a, b, v)
    elif case == "both_operands":
        kernel, args = _both_operands_chain, (a, b)
    else:
        square = torch.empty((2, 128, 128), dtype=torch.bfloat16)
        kernel, args = _live_producer_chain, (a, b, square)
    code = _tcgen_guard_code(args, None, kernel, ())
    assert "chain_0_c =" in code
    assert "OperandSource.TMEM" not in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("logical_q", [100, 128])
def test_tcgen_shared_leaf_residual_runtime(logical_q: int) -> None:
    args = _tcgen_guard_args(DEVICE, logical_q)
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


# Copy cast.


@pytest.mark.parametrize("dtype", ["Int32", "Int64"])
def test_copy_index_preserves_uniform_integer_cast(dtype: str) -> None:
    row, origin = sympy.symbols("row origin", integer=True)
    value = _copy_index(
        f"cutlass.{dtype}(origin * 128)", {}, {"row": row, "origin": origin}
    )
    assert sympy.diff(value, row) == 0
    env = SimpleNamespace(
        index_dtype=torch.int64,
        backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int64"),
    )
    with patch(
        "helion._compiler.cute.chained_matmul.CompileEnvironment.current",
        return_value=env,
    ):
        expected = f"cutlass.{dtype}((128 * cutlass.Int64(origin)))"
        if dtype == "Int32":
            expected = f"cutlass.Int64({expected})"
        assert _copy_code(value) == expected


@pytest.mark.parametrize("dtype", ["Int32", "Int64"])
@pytest.mark.parametrize(
    "expression",
    ["row + index", "index - 1", "index * 0", "index // 2", "index % 2", "-index"],
)
def test_copy_index_rejects_post_cast_arithmetic(dtype: str, expression: str) -> None:
    with pytest.raises(_UnsupportedChain, match="arithmetic after fixed-width"):
        _copy_index(
            expression,
            {"index": f"cutlass.{dtype}(origin)"},
            {name: sympy.Symbol(name, integer=True) for name in ("row", "origin")},
        )


@pytest.mark.parametrize("dtype", ["Int32", "Int64"])
def test_copy_index_varying_cast_is_not_proven_affine(dtype: str) -> None:
    row = sympy.Symbol("row", integer=True)
    value = _copy_index(f"cutlass.{dtype}(row)", {}, {"row": row})
    assert not isinstance(sympy.diff(value, row), sympy.Integer)


@pytest.mark.parametrize(
    "source", ["cutlass.Float32(row)", "other.Int32(row)", "cutlass.Int32(row, 2)"]
)
def test_copy_index_rejects_non_integer_or_unknown_calls(source: str) -> None:
    with pytest.raises(_UnsupportedChain):
        _copy_index(source, {}, {"row": sympy.Symbol("row", integer=True)})


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunked_chain(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, length, state = a.shape
    dim = v.shape[-1]
    chunks = length // 128
    state = hl.specialize(state)
    out = torch.empty_like(v)
    for block, row, col in hl.tile(
        [batch * chunks, 128, dim], block_size=[1, None, None]
    ):
        bi = block.begin // chunks
        ci = block.begin % chunks
        q = hl.arange(128)
        qr = ci * 128 + row.index
        qk = ci * 128 + q
        first = hl.dot(a[bi, qr, :], b[bi, qk, :].T)
        result = hl.dot(first.to(a.dtype), v[bi, qk, col])
        out[bi, qr, col] = (result + v[bi, qr, col].float()).to(v.dtype)
    return out


def _copy_cast_config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )


def test_chunked_chain_codegen_keeps_scalar_post_cast_indexing() -> None:
    values = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((2, 384, 128), (2, 384, 128), (2, 384, 64))
    )
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
        code = _chunked_chain._bind_isolated(values).to_code(_copy_cast_config())
    assert "OperandSource.TMEM" in code
    assert "_async_pointer" not in code
    assert "chain_store_copy" not in code
    assert "chain_epi_input_0_values" not in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_chain_correctness() -> None:
    torch.manual_seed(73)
    values = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.1
        for shape in ((2, 384, 128), (2, 384, 128), (2, 384, 64))
    )
    saved = tuple(value.clone() for value in values)
    a, b, v = (value.reshape(2, 3, 128, -1) for value in values)
    expected = (
        (
            (a.float() @ b.float().transpose(-1, -2)).to(a.dtype).float() @ v.float()
            + v.float()
        )
        .to(v.dtype)
        .reshape(values[-1].shape)
    )
    function = _chunked_chain._bind_isolated(values).compile_config(_copy_cast_config())
    actual = function(*values)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
    torch.testing.assert_close(actual, function(*values), atol=0, rtol=0)
    for value, before in zip(values, saved, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrow_batch_offset(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, overflow: hl.constexpr
) -> torch.Tensor:
    batch, rows, state = a.shape
    columns = v.shape[-1]
    state = hl.specialize(state)
    out = torch.empty_like(v)
    for block, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        q = hl.arange(128)
        # This narrowing is explicit user semantics, not a pointer-width cast.
        bi = torch.scalar_tensor(block.begin, dtype=torch.int32)
        if overflow:
            bi = ((bi * 1073741824) // 1073741824) % 3
        first = hl.dot(a[bi, row, :], b[bi, q, :].T)
        result = hl.dot(first.to(a.dtype), v[bi, q, col])
        out[block.begin, row, col] = result.to(v.dtype)
    return out


def test_explicit_narrow_index_promotes_before_wide_stride_product() -> None:
    # No physical allocation: each index fits Int32, but index * stride does
    # not. At bi=131072, the A/B offset is 2**31 elements, not -2**31.
    with FakeTensorMode():
        values = tuple(
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in (
                (1048577, 128, 128),
                (1048577, 128, 128),
                (1048577, 128, 64),
            )
        )
        code = _tcgen_guard_code((*values, False), None, _narrow_batch_offset)
    pointers = [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.endswith("_async_pointer")
            for target in node.targets
        )
    ]
    assert len(pointers) == 3
    for pointer in pointers:
        text = ast.unparse(pointer)
        assert "cutlass.Int64(cutlass.Int32(" in text
        for product in ast.walk(pointer):
            if isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult):
                for operand in (product.left, product.right):
                    assert not (
                        isinstance(operand, ast.Call)
                        and ast.unparse(operand.func) == "cutlass.Int32"
                    ), "Physical stride multiplication must not execute in Int32"


def test_explicit_wrapping_index_arithmetic_uses_scalar_fallback() -> None:
    values = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((4, 128, 128), (4, 128, 128), (4, 128, 64))
    )
    code = _tcgen_guard_code((*values, True), None, _narrow_batch_offset)
    # All selectors are valid, but cancellation in unbounded integer algebra
    # would incorrectly use [0, 1, 2, 0] instead of the wrapped [0, 1, 1, 2].
    selector = ((torch.arange(4, dtype=torch.int32) * 1073741824) // 1073741824) % 3
    assert selector.tolist() == [0, 1, 1, 2]
    assert "OperandSource.TMEM" in code
    assert "_async_pointer" not in code
    assert "1073741824" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wrapping_batch_selector_runtime_exact() -> None:
    function = None
    selected = torch.tensor([0, 1, 1, 2], device=DEVICE)
    for seed in range(5):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        values = tuple(
            torch.randint(-2, 3, shape, device=DEVICE, generator=generator).to(
                torch.bfloat16
            )
            / 16
            for shape in ((4, 128, 128), (4, 128, 128), (4, 128, 64))
        )
        saved = tuple(value.clone() for value in values)
        if function is None:
            function = _narrow_batch_offset._bind_isolated(
                (*values, True)
            ).compile_config(
                helion.Config(
                    block_sizes=[128, 64],
                    num_warps=4,
                    cute_chained_mma_schedule="tcgen05_tmem",
                )
            )
        a, b, v = (value.index_select(0, selected).double() for value in values)
        first = (a @ b.transpose(-1, -2)).to(torch.bfloat16).double()
        expected = (first @ v).to(torch.bfloat16)
        actual = function(*values, True)
        repeated = function(*values, True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        assert actual.data_ptr() != repeated.data_ptr()
        for value, before in zip(values, saved, strict=True):
            torch.testing.assert_close(value, before, atol=0, rtol=0)


# Tcgen05 config.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _config_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    row_block: hl.constexpr,
    scan: hl.constexpr,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), dtype=a.dtype, device=a.device)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, row_block, None]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        if scan:
            decay = hl.cumsum(v[bi, qq, 0].float(), dim=0)
            first *= torch.exp(decay[row][:, None] - decay[qq][None, :])
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _permuted_config_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), dtype=a.dtype, device=a.device)
    for col, batch, row in hl.tile([n, batches, m], block_size=[None, 1, None]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


@pytest.fixture
def _cpu_support():
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
        yield


def _tcgen_config_inputs(
    *, m: int = 128, n: int = 256, k: int = 128, q: int = 128
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((2, m, k), (2, q, k), (2, q, n))
    )


def _tcgen_config_bound(
    *, row_block: int | None = None, scan: bool = False, **shape: int
):
    _config_chain.reset()
    return _config_chain.bind((*_tcgen_config_inputs(**shape), row_block, scan))


def _tcgen_seeds(bound: Any) -> list[helion.Config]:
    return [
        seed
        for seed in _without_early_release_seed(
            bound.config_spec.compiler_seed_configs, bound.config_spec
        )
        if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
        and not seed.config.get("cute_chained_pointwise_vectorize", False)
        and not seed.config.get("cute_chained_auxiliary_cache", False)
    ]


def _without_early_release_seed(
    seeds: list[helion.Config], spec: Any | None = None
) -> list[helion.Config]:
    # Release siblings are introduced with the TMEM-lifetime feature.
    return seeds


@pytest.mark.usefixtures("_cpu_support")
def test_tcgen05_chain_seeds_are_additive_and_semantic() -> None:
    bound = _tcgen_config_bound()
    spec = bound.config_spec
    seeds = _tcgen_seeds(bound)
    assert [seed.block_sizes for seed in seeds] == [
        [128, n] for n in (32, 64, 128, 256)
    ]
    assert all(seed.num_warps == 4 and seed.pid_type == "flat" for seed in seeds)
    assert (
        spec.compiler_seed_configs[0].config["cute_chained_mma_schedule"] == "coalesced"
    )
    assert spec.compiler_seed_configs[0].block_sizes == [16, 16]
    assert any(seed.num_warps == 8 for seed in spec.compiler_seed_configs)
    assert "tcgen05_tmem" in spec._cute_chained_mma_schedules()
    for seed in seeds:
        normalized = spec.normalized_config(seed)
        assert normalized.block_sizes == seed.block_sizes
        assert normalized.num_warps == 4 and normalized.pid_type == "flat"


@pytest.mark.usefixtures("_cpu_support")
def test_tcgen05_chain_seeds_follow_permuted_root_axes() -> None:
    _permuted_config_chain.reset()
    bound = _permuted_config_chain.bind(_tcgen_config_inputs())
    assert [seed.block_sizes for seed in _tcgen_seeds(bound)] == [
        [n, 128] for n in (32, 64, 128, 256)
    ]


@pytest.mark.usefixtures("_cpu_support")
def test_tcgen05_chain_scan_persistent_reduction_axes() -> None:
    assert [
        seed.block_sizes for seed in _tcgen_seeds(_tcgen_config_bound(scan=True))
    ] == [[128, n] for n in (32, 64, 128, 256)]


@pytest.mark.usefixtures("_cpu_support")
@pytest.mark.parametrize("shape", [{"m": 127}, {"m": 64}, {"n": 95}, {"q": 512}])
def test_tcgen05_chain_ineligible_shapes_keep_warp_schedules(
    shape: dict[str, Any],
) -> None:
    bound = _tcgen_config_bound(**shape)
    assert not _tcgen_seeds(bound)
    assert "tcgen05_tmem" not in bound.config_spec._cute_chained_mma_schedules()
    assert (
        "cp_async_register_reuse_scan"
        in bound.config_spec._cute_chained_mma_schedules()
    )


@pytest.mark.usefixtures("_cpu_support")
def test_tcgen05_chain_honors_fixed_block_sizes() -> None:
    assert not _tcgen_seeds(_tcgen_config_bound(row_block=16))
    assert [
        seed.block_sizes for seed in _tcgen_seeds(_tcgen_config_bound(row_block=128))
    ] == [[n] for n in (32, 64, 128, 256)]


@pytest.mark.usefixtures("_cpu_support")
def test_tcgen05_chain_hardware_gate() -> None:
    support = default_cute_mma_support()
    support.tcgen05_f16bf16 = False
    with patch(
        "helion._compiler.cute.mma_support.get_cute_mma_support", return_value=support
    ):
        bound = _tcgen_config_bound()
    assert not _tcgen_seeds(bound)
    with pytest.raises(exc.InvalidConfig, match="schedule"):
        bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[128, 64], cute_chained_mma_schedule="tcgen05_tmem"
            )
        )


@pytest.mark.usefixtures("_cpu_support")
@pytest.mark.parametrize("warps", [1, 2, 8])
def test_tcgen05_chain_requires_four_warps(warps: int) -> None:
    spec = _tcgen_config_bound().config_spec
    config = helion.Config(
        block_sizes=[128, 64], num_warps=warps, cute_chained_mma_schedule="tcgen05_tmem"
    )
    with pytest.raises(exc.InvalidConfig, match="num_warps=4"):
        spec.normalized_config(config)
    spec.normalize(config, _fix_invalid=True)
    assert config.num_warps == 4
    assert config.config["cute_chained_mma_schedule"] == "tcgen05_tmem"


@pytest.mark.usefixtures("_cpu_support")
def test_tcgen05_chain_explicit_unsupported_plan_cannot_fall_back() -> None:
    bound = _tcgen_config_bound()
    config = helion.Config(
        block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )
    with (
        patch(
            "helion._compiler.cute.chained_matmul.plan_chained_matmul",
            return_value=None,
        ),
        pytest.raises(exc.BackendUnsupported, match="tcgen05_tmem"),
    ):
        bound.to_code(config)


# Tcgen05 address.


def _assert_tmem_allocation_base(ptx: str) -> None:
    """TMEM A and C must retain the same runtime allocation-base ancestor."""
    definitions = {
        match[2]: (match[1], re.findall(r"%r\d+", match[3]))
        for match in re.finditer(r"(?m)^\s*([a-z0-9_.]+)\s+(%r\d+),\s*([^;]+);", ptx)
    }

    def shared_loads(register: str) -> set[str]:
        pending = [register]
        seen: set[str] = set()
        result: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current not in definitions:
                continue
            operation, operands = definitions[current]
            if operation.startswith("ld.shared."):
                result.add(current)
            else:
                pending.extend(operands)
        return result

    addresses = re.findall(r"tcgen05\.mma[^\n]*?\[(%r\d+)\],\s*\[(%r\d+)\]", ptx)
    assert addresses, "Expected at least one TMEM-source MMA"
    for destination, operand_a in addresses:
        assert shared_loads(destination) & shared_loads(operand_a), (
            f"TMEM A {operand_a} lost the allocation base used by C {destination}"
        )


@pytest.mark.parametrize("dynamic", [False, True])
def test_tmem_address_check_detects_absolute_zero(dynamic: bool) -> None:
    address = "add.s32 %r3, %r1, 8;" if dynamic else "mov.b32 %r3, 8;"
    ptx = (
        "ld.shared.b32 %r1, [allocation];\n"
        "add.s32 %r2, %r1, 128;\n"
        f"{address}\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%r2], [%r3], desc;\n"
    )
    if dynamic:
        _assert_tmem_allocation_base(ptx)
    else:
        with pytest.raises(AssertionError, match="lost the allocation base"):
            _assert_tmem_allocation_base(ptx)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_tcgen_bridge_retains_allocated_base_in_ptx(dtype: str) -> None:
    # Isolate fake compilation from CUDA initialization by earlier GPU tests.
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP_PTX": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, "tcgen05_address", dtype],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Signed arithmetic.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _signed_arithmetic(
    a: torch.Tensor,
    b: torch.Tensor,
    integer_dtype: torch.dtype,
    divisor: int,
    mode: hl.constexpr,
) -> torch.Tensor:
    divisor = hl.specialize(divisor)
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=a.dtype, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        if mode == "scalar_index":
            bi = (bt.begin - 2) % 3
        elif mode == "scalar_floor_index":
            bi = (bt.begin - 2) // divisor + 1
        elif mode == "scalar_positive_floor_index":
            bi = bt.begin // divisor
        else:
            bi = bt.begin
        kk = hl.arange(reduction)
        first = hl.dot((a[bi, row, kk].float() * 0.5).to(a.dtype), b[bi, kk, col])
        numerator = row.index.to(integer_dtype) - 64
        if mode in (
            "scalar_index",
            "scalar_floor_index",
            "scalar_positive_floor_index",
        ):
            value = numerator[:, None] * 0
        elif mode == "floor":
            value = torch.div(numerator, divisor, rounding_mode="floor")[:, None]
        elif mode == "floor_op":
            value = torch.floor_divide(numerator, divisor)[:, None]
        elif mode == "remainder":
            value = torch.remainder(numerator, divisor)[:, None]
        else:
            denominator = (col.index.to(integer_dtype) % 2) * 6 - 3
            if mode == "scalar_left":
                value = torch.remainder(-7, denominator)[None, :]
            elif mode == "floor_tensor":
                value = torch.div(
                    numerator[:, None], denominator[None, :], rounding_mode="floor"
                )
            elif mode == "floor_scalar_left":
                value = torch.div(-7, denominator, rounding_mode="floor")[None, :]
            else:
                value = torch.remainder(numerator[:, None], denominator[None, :])
        out[bt.begin, row, col] = (first + value.float()).to(a.dtype)
    return out


def _signed_values(device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((2, 128, 64), dtype=torch.bfloat16, device=device),
        torch.zeros((2, 64, 64), dtype=torch.bfloat16, device=device),
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_signed_remainder_correction_matches_torch(dtype: torch.dtype) -> None:
    limits = torch.iinfo(dtype)
    dividends = [limits.min, limits.min + 1, -11, -6, -1, 0, 1, 6, 11, limits.max]
    divisors = [limits.min, -7, -3, -1, 1, 3, 7, limits.max]
    expression = _signed_remainder_adjustment("remainder", "divisor")
    for dividend in dividends:
        for divisor in divisors:
            if dividend == limits.min and divisor == -1:
                continue  # The native signed division overflow is unchanged.
            quotient = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                quotient = -quotient
            remainder = dividend - quotient * divisor
            actual = eval(expression, {}, {"remainder": remainder, "divisor": divisor})
            expected = torch.remainder(
                torch.tensor(dividend, dtype=dtype), torch.tensor(divisor, dtype=dtype)
            ).item()
            assert actual == expected


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_signed_floor_exact_quotient_correction_matches_torch(
    dtype: torch.dtype,
) -> None:
    limits = torch.iinfo(dtype)
    dividends = [limits.min, limits.min + 1, -11, -6, -1, 0, 1, 6, 11, limits.max]
    divisors = [limits.min, -7, -3, -1, 1, 3, 7, limits.max]
    expression = _signed_floor_adjustment("quotient", "remainder", "divisor")
    for dividend in dividends:
        for divisor in divisors:
            if dividend == limits.min and divisor == -1:
                continue
            quotient = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                quotient = -quotient
            remainder = dividend - quotient * divisor
            exact_dividend = dividend - remainder
            assert limits.min <= exact_dividend <= limits.max
            assert exact_dividend % divisor == 0
            actual = eval(
                expression,
                {},
                {
                    "quotient": exact_dividend // divisor,
                    "remainder": remainder,
                    "divisor": divisor,
                },
            )
            expected = torch.div(
                torch.tensor(dividend, dtype=dtype),
                torch.tensor(divisor, dtype=dtype),
                rounding_mode="floor",
            ).item()
            assert actual == expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "mode",
    [
        "floor",
        "floor_op",
        "floor_tensor",
        "floor_scalar_left",
        "remainder",
        "tensor",
        "scalar_left",
    ],
)
@pytest.mark.parametrize("divisor", [-3, 3])
def test_signed_arithmetic_codegen(dtype: torch.dtype, mode: str, divisor: int) -> None:
    code = _tcgen_guard_code(
        (*_signed_values("cpu"), dtype, divisor, mode), None, _signed_arithmetic
    )
    assert "chain_0_mma" in code
    assert "!= 0" in code
    if mode.startswith("floor"):
        assert " // " in code and " % " in code and " - 1 if " in code


def test_negative_symint_remainder_codegen() -> None:
    values = (
        torch.empty((4, 128, 64), dtype=torch.bfloat16),
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
    )
    code = _tcgen_guard_code(
        (*values, torch.int32, 3, "scalar_index"), None, _signed_arithmetic
    )
    assert "!= 0" in code
    assert "_async_pointer" not in code


@pytest.mark.parametrize("divisor", [-3, 3])
def test_negative_symint_floor_codegen(divisor: int) -> None:
    values = (
        torch.empty((4, 128, 64), dtype=torch.bfloat16),
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
    )
    code = _tcgen_guard_code(
        (*values, torch.int32, divisor, "scalar_floor_index"), None, _signed_arithmetic
    )
    assert " - 1 if " in code and " % " in code
    assert "_async_pointer" not in code


def test_nonnegative_symint_floor_preserves_index_fast_path() -> None:
    code = _tcgen_guard_code(
        (*_signed_values("cpu"), torch.int32, 3, "scalar_positive_floor_index"),
        None,
        _signed_arithmetic,
    )
    assert "_async_pointer" in code
    assert " - 1 if " not in code


def test_scalar_float_floor_keeps_existing_lowering() -> None:
    graph = torch.fx.Graph()
    node = graph.call_function(operator.floordiv, (1.5, -3.0))
    node.meta["val"] = -1.0
    expression = _Expression.__new__(_Expression)
    assert expression.scalar(node) == "(1.5 // -3.0)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["scalar_index", "scalar_floor_index"])
@pytest.mark.parametrize("divisor", [-3, 3])
def test_negative_symint_remainder_runtime_exact(mode: str, divisor: int) -> None:
    values = (
        torch.arange(1, 5, device=DEVICE, dtype=torch.bfloat16)[:, None, None]
        .expand(4, 128, 64)
        .contiguous(),
        torch.eye(64, device=DEVICE, dtype=torch.bfloat16)[None, :, :]
        .expand(4, 64, 64)
        .contiguous(),
    )
    saved = tuple(value.clone() for value in values)
    function = _signed_arithmetic._bind_isolated(
        (*values, torch.int32, divisor, mode)
    ).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    selected = (
        torch.tensor([1, 2, 0, 1], device=DEVICE)
        if mode == "scalar_index"
        else torch.div(
            torch.arange(4, device=DEVICE) - 2, divisor, rounding_mode="floor"
        )
        + 1
    )
    expected = values[0].index_select(0, selected) * 0.5
    actual = function(*values, torch.int32, divisor, mode)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    repeated = function(*values, torch.int32, divisor, mode)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert repeated.data_ptr() != actual.data_ptr()
    for value, before in zip(values, saved, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "mode", ["floor", "floor_tensor", "scalar_floor_index", "remainder"]
)
def test_signed_arithmetic_real_cpu_compile(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", __name__, "signed", mode],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_103a",
            "CUTE_DSL_KEEP_PTX": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "mode",
    [
        "floor",
        "floor_op",
        "floor_tensor",
        "floor_scalar_left",
        "remainder",
        "tensor",
        "scalar_left",
    ],
)
@pytest.mark.parametrize("divisor", [-3, 3])
def test_signed_arithmetic_runtime_exact(
    dtype: torch.dtype, mode: str, divisor: int
) -> None:
    values = _signed_values(DEVICE)
    function = _signed_arithmetic._bind_isolated(
        (*values, dtype, divisor, mode)
    ).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    numerator = torch.arange(128, dtype=dtype) - 64
    denominator = (torch.arange(64, dtype=dtype) % 2) * 6 - 3
    if mode in ("floor", "floor_op"):
        expected = torch.div(numerator, divisor, rounding_mode="floor")[:, None]
    elif mode == "floor_tensor":
        expected = torch.div(
            numerator[:, None], denominator[None, :], rounding_mode="floor"
        )
    elif mode == "floor_scalar_left":
        expected = torch.div(-7, denominator, rounding_mode="floor")[None, :]
    elif mode == "remainder":
        expected = torch.remainder(numerator, divisor)[:, None]
    elif mode == "scalar_left":
        expected = torch.remainder(-7, denominator)[None, :]
    else:
        expected = torch.remainder(numerator[:, None], denominator[None, :])
    expected = expected.expand(2, 128, 64).to(device=DEVICE, dtype=torch.bfloat16)
    actual = function(*values, dtype, divisor, mode)
    repeated = function(*values, dtype, divisor, mode)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert actual.data_ptr() != repeated.data_ptr()
    for value in values:
        assert torch.count_nonzero(value) == 0


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "tcgen05_address":
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[1]]
        values = _tcgen_inputs("cpu", dtype)
        ptx = _tcgen_compile(_tcgen_code(values, "plain"), values, "plain")
        _assert_tmem_allocation_base(ptx)
        assert not torch.cuda.is_initialized()
    elif command == "signed":
        args = (*_signed_values("cpu"), torch.int64, -3, sys.argv[1])
        code = _tcgen_guard_code(args, None, _signed_arithmetic)
        ptx = _tcgen_compile(code, args, None, "_signed_arithmetic")
        assert "tcgen05.mma" in ptx
        assert not torch.cuda.is_initialized()
    else:
        raise AssertionError(f"Unknown test driver: {command}")
