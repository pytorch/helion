"""A block's thread queries answer from the loop codegen runs it on.

Loops of several roots can tile one block: a root's grid over a registered
block size and another root's device loop over it.  The thread-extent and
thread-axis queries answered from the first strategy listed for the block,
the root's grid, even inside the other root's loop, where codegen lays the
block out on the loop's own strategy.  Demoted there to a lane loop past the
third thread axis, a sum over the block still combined each thread's lanes
across 32 threads of axis 0: other rows' and surplus threads'.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _root_ahead_of_shared_block_sums(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.size(0)
    doubled = torch.empty_like(x)
    sums = torch.empty(y.shape[:3], dtype=torch.float32, device=y.device)
    block_n = hl.register_block_size(n)
    for tile_a in hl.tile(n, block_size=block_n):
        doubled[tile_a] = x[tile_a] * 2
    for tile_b, tile_h, tile_r in hl.tile([y.size(0), y.size(1), y.size(2)]):
        acc = hl.zeros([tile_b, tile_h, tile_r], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + y[tile_b, tile_h, tile_r, tile_n].sum(-1)
        sums[tile_b, tile_h, tile_r] = acc
    return doubled, sums


_kernel = helion.kernel(
    _root_ahead_of_shared_block_sums,
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
)
# The second root's three grid blocks take the launch axes, so its loop over
# block_n walks the block's 32 threads in a lane loop.
_CONFIG = helion.Config(block_sizes=[32, 2, 2, 2], num_threads=[32, 2, 2, 2])


def test_a_demoted_loop_reduces_within_each_thread() -> None:
    with mock_cuda_unavailable():
        bound = cpu_bind(_kernel, (torch.empty(64), torch.empty(2, 2, 2, 64)))
        code = bound.to_code(_CONFIG)
    assert "for lane_0 in range(32)" in code, code
    assert "warp_reduction" not in code, code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_a_demoted_loop_sums_each_row_once() -> None:
    x = torch.randn(64, device=DEVICE)
    y = torch.randn(2, 2, 2, 64, device=DEVICE)
    doubled, sums = _kernel.bind((x, y)).compile_config(_CONFIG)(x, y)
    torch.testing.assert_close(doubled, x * 2)
    torch.testing.assert_close(sums, y.sum(-1), rtol=1e-4, atol=1e-4)
