from __future__ import annotations

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _launch_block
import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, rows, reduction = a.shape
    columns = b.shape[-1]
    out = torch.empty((batch, rows, columns), device=a.device, dtype=a.dtype)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        acc = hl.zeros([bt, row, col], dtype=torch.float32)
        for kk in hl.tile(reduction):
            acc = torch.baddbmm(acc, a[bt, row, kk], b[bt, kk, col])
        out[bt, row, col] = acc.to(out.dtype)
    return out


def _config(pid: str, columns: int) -> helion.Config:
    return helion.Config(
        block_sizes=[128, columns, 128],
        loop_orders=[[1, 0, 2]],
        pid_type=pid,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_consumer_regs=256,
        tcgen05_l2_swizzle_size=1,
        tcgen05_strategy="role_local_monolithic",
    )


@pytest.mark.parametrize(
    "pid", ["flat", "persistent_blocked", "persistent_interleaved"]
)
@pytest.mark.parametrize("columns", [16, 32, 64, 128])
@pytest.mark.parametrize("explicit_auto", [False, True])
def test_auto_tcgen05_threads_keep_hardware_warp_width(
    pid: str, columns: int, explicit_auto: bool
) -> None:
    args = (
        torch.empty((3, 128, 128), dtype=torch.bfloat16),
        torch.empty((3, 128, 256), dtype=torch.bfloat16),
    )
    config = _config(pid, columns)
    if explicit_auto:
        config.config["num_threads"] = [0, 0, 0]
    code = _code(_batched_product, args, config)
    # The six-role plan is four epilogue warps, one MMA warp and one TMA
    # warp. A generic SIMT budget must not turn an auto axis into 64 lanes
    # and double the number of physical warps after specialization.
    assert _launch_block(code) == (32, 6, 1)
    assert "setmaxregister_increase(256)" in code
    assert "setmaxregister_decrease(120)" in code


def test_explicit_tcgen05_thread_width_remains_explicit() -> None:
    args = (
        torch.empty((3, 128, 128), dtype=torch.bfloat16),
        torch.empty((3, 128, 256), dtype=torch.bfloat16),
    )
    config = _config("persistent_blocked", 16)
    config.config["num_threads"] = [32, 8, 0]
    code = _code(_batched_product, args, config)
    assert _launch_block(code) == (32, 6, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "pid", ["flat", "persistent_blocked", "persistent_interleaved"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_narrow_tcgen05_role_launch_runtime(pid: str, dtype: torch.dtype) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05 FP16/BF16 support")
    generator = torch.Generator(device=DEVICE).manual_seed(9641)
    a = torch.randn((3, 128, 128), generator=generator, device=DEVICE, dtype=dtype)
    b = torch.randn((3, 128, 256), generator=generator, device=DEVICE, dtype=dtype)
    saved = (a.clone(), b.clone())
    expected = torch.bmm(a.double(), b.double()).to(dtype)
    run = _batched_product._bind_isolated((a, b)).compile_config(_config(pid, 16))
    first = run(a, b)
    torch.testing.assert_close(first, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(run(a, b), first, atol=0, rtol=0)
    torch.testing.assert_close((a, b), saved, atol=0, rtol=0)
