"""Selected configuration scalars keep native MMA loop semantics."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_config_scalar import repeated_dot

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("repeats", [1, 8])
@skipIfRefEager("requires compiled configurations")
def test_configuration_scalar_loop_bound_replays(mode, repeats):
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 requires an SM100-family GPU")
    x = (torch.arange(64 * 16, device=DEVICE).reshape(64, 16) % 5 - 2).bfloat16()
    y = (torch.arange(16 * 16, device=DEVICE).reshape(16, 16) % 3 - 1).bfloat16()
    compiled = repeated_dot.bind((x, y)).compile_config(
        helion.Config(
            block_sizes=[repeats, 64, 16],
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
        )
    )
    for delta in (0, 1):
        changed = x + delta
        torch.testing.assert_close(
            compiled(changed, y), changed.float() @ y.float(), rtol=0, atol=0
        )
