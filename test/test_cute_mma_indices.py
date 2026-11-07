"""Native masked MMA reductions with both supported index widths."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_mma_indices import causal_dot
from helion._testing.cute_mma_indices import masked_dot

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@skipIfRefEager("requires compiled configurations")
def test_mma_index_width_replays(mode, dtype):
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 requires an SM100-family GPU")
    x = (torch.arange(128 * 16, device=DEVICE).reshape(128, 16) % 5 - 2).bfloat16()
    y = (torch.arange(16 * 16, device=DEVICE).reshape(16, 16) % 3 - 1).bfloat16()
    kernel = helion.kernel(
        masked_dot.fn, backend="cute", static_shapes=True, index_dtype=dtype
    )
    compiled = kernel.bind((x, y)).compile_config(
        helion.Config(
            block_sizes=[16, 64],
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
        )
    )
    rows = torch.arange(128, device=DEVICE)[:, None]
    cols = torch.arange(16, device=DEVICE)[None, :]
    visible = (rows >= cols) & (rows < 64)
    for delta in (0, 1):
        changed = x + delta
        expected = torch.where(visible, changed.float() @ y.float(), 0).sum(dim=0)
        torch.testing.assert_close(compiled(changed, y), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@skipIfRefEager("requires compiled configurations")
def test_varying_loop_bound_replays(mode, dtype):
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 requires an SM100-family GPU")
    x = (torch.arange(128 * 128, device=DEVICE).reshape(128, 128) % 5 - 2).bfloat16()
    y = (torch.arange(128 * 16, device=DEVICE).reshape(128, 16) % 3 - 1).bfloat16()
    kernel = helion.kernel(
        causal_dot.fn, backend="cute", static_shapes=True, index_dtype=dtype
    )
    compiled = kernel.bind((x, y)).compile_config(
        helion.Config(
            block_sizes=[64, 16, 64],
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
        )
    )
    rows = torch.arange(128, device=DEVICE)[:, None]
    inner = torch.arange(128, device=DEVICE)[None, :]
    visible = inner < (rows // 64 + 1) * 64
    for delta in (0, 1):
        changed = x + delta
        expected = torch.where(visible, changed.float(), 0) @ y.float()
        torch.testing.assert_close(compiled(changed, y), expected, rtol=0, atol=0)
