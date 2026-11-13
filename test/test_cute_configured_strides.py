"""Configured single-root MMA pitches preserve distinct output groups."""

from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_configured_strides import pitched_config
from helion._testing.cute_configured_strides import pitched_matmul


@skipUnlessCuteAvailable("requires CuTe SDK")
@skipUnlessBackends(["cute"])
@skipIfRefEager("tests configured MMA storage")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("pitch", [64, 256])
def test_single_root_mma_output_pitch_matches_allocation(mode, pitch):
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 requires an SM100-family GPU")
    x = (torch.arange(2 * 64 * 16, device=DEVICE) % 7 - 3).reshape(2, 64, 16)
    y = (torch.arange(2 * 16 * 16, device=DEVICE) % 5 - 2).reshape(2, 16, 16)
    x, y = x.to(torch.bfloat16), y.to(torch.bfloat16)
    config = pitched_config(mode, pitch)
    compiled = pitched_matmul.bind((x, y)).compile_config(config)
    for offset in (0, 1):
        changed_x = x + offset
        result = compiled(changed_x, y)
        assert result.stride() == (pitch * 16, 16, 1)
        torch.testing.assert_close(
            result, changed_x.float() @ y.float(), atol=0, rtol=0
        )
