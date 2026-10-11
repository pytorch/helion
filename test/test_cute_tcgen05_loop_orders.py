"""tcgen05 matmuls under a ``loop_orders`` permutation of the root tile.

``loop_orders=[[1, 0]]`` swaps the thread axes the root tile strategy gives the
M and N blocks, but the tcgen05 role launch is ``(physical_m_threads, warps,
1)`` either way.  The role coordinates read the launch axes, not the blocks'
(``mma_tidx = thread_idx[0] + thread_idx[1] * 32``), so the edge-tile operand
copies fill shared memory from the right threads and partial M or N tiles
match eager (full tiles go through TMA).
"""

from __future__ import annotations

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc + bias[tile_n]
    return out


_MMA_TIDX = (
    "mma_tidx = cutlass.Int32(cute.arch.thread_idx()[0]) + "
    "cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(32)"
)


@pytest.mark.parametrize("loop_order", [[0, 1], [1, 0]])
@skipUnlessBackends(["cute"])
def test_role_coordinates_follow_the_launch_axes(loop_order: list[int]) -> None:
    args = (
        torch.empty((200, 96), dtype=torch.float16),
        torch.empty((96, 136), dtype=torch.float16),
        torch.empty((136,), dtype=torch.float32),
    )
    with _mock_cuda_unavailable(), _cpu_target():
        bound = _matmul_bias._bind_isolated(args)
        with bound.env:
            config = dict(bound.config_spec.default_config().config)
        config.update(block_sizes=[64, 32, 16], loop_orders=[loop_order])
        code = bound.to_code(helion.Config(**config))
    assert "block=(32, 6, 1)" in code
    assert _MMA_TIDX in code
    assert "mma_active = cutlass.Int32(cute.arch.thread_idx()[1]) < " in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("collective", [None, "warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_permuted_loop_order_matches_on_partial_tiles(collective: str | None) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(0)
    for (m, n, k), block_sizes in (
        ((200, 136, 96), [64, 32, 16]),
        ((200, 256, 96), [128, 16, 32]),
        ((256, 136, 96), [128, 64, 32]),
    ):
        a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE)
        bound = _matmul_bias.bind((a, b, bias))
        with bound.env:
            config = dict(bound.config_spec.default_config().config)
        config.update(block_sizes=block_sizes, loop_orders=[[1, 0]])
        if collective is not None:
            config.update(
                cute_collective_mma=True,
                cute_collective_compute=collective,
                cute_collective_copy="async_cached",
                cute_collective_stages=2,
            )
        for pid_type in ("flat", "xyz"):
            config["pid_type"] = pid_type
            bound.set_config(helion.Config(**config))
            torch.testing.assert_close(
                bound(a, b, bias),
                a.float() @ b.float() + bias,
                atol=5e-2,
                rtol=1e-2,
            )
