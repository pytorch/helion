"""Changed-input replay of independent phases with a tunable workspace shape."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_ordered_mma import partitioned_matmul_sum

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("splits", [1, 2, 4, 8])
@skipIfRefEager("requires compiled configurations")
def test_partitioned_mma_phases_replay_with_new_inputs(mode, splits):
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 requires SM100-family")
    x = (torch.arange(2 * 8 * 64 * 16, device=DEVICE).reshape(2, 8, 64, 16) % 7 - 3).to(
        torch.bfloat16
    )
    y = (torch.arange(2 * 8 * 16 * 16, device=DEVICE).reshape(2, 8, 16, 16) % 5 - 2).to(
        torch.bfloat16
    )
    bound = partitioned_matmul_sum.bind((x, y))
    config = helion.Config(
        block_sizes=[splits, 128, 64],
        cute_mma_dag=mode,
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        load_eviction_policies=["", "", "", ""],
    )
    compiled = bound.compile_config(config)
    for shift in (0, 1):
        changed = x + shift
        expected = torch.matmul(changed.float() * 0.5, y.float()).sum(dim=1)
        torch.testing.assert_close(compiled(changed, y), expected, rtol=0, atol=0)
