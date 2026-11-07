"""Native replay coverage for separately launched computed and SIMT roots."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_ordered_metadata import computed_metadata_phases


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shift", [None, 2.0])
def test_ordered_computed_storage_view_replay(shift):
    owner = torch.randn(2 * 128 + 3, device=DEVICE, dtype=torch.bfloat16)
    x = owner[3:].as_strided((2, 7, 8), (128, 12, 1))
    y = torch.randn(2, 8, 9, device=DEVICE, dtype=torch.bfloat16)
    bound = computed_metadata_phases.bind((x, y, shift))
    config = dict(bound.config_spec.default_config()) | {
        "pid_type": "flat",
        "cute_fragment_half_mma": True,
    }
    compiled = bound.compile_config(helion.Config.from_dict(config))
    original = owner.clone()
    for scale in (1.0, -0.5):
        x.mul_(scale)
        rounded = (x.float() * (8**-0.5)).to(x.dtype).float()
        value = rounded @ y.float()
        expected = (
            value + value.amax(dim=1, keepdim=True) + (1.0 if shift is None else shift)
        )
        output = compiled(x, y, shift)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    assert not torch._C._is_alias_of(output, x)
    # Only the explicit host updates above write the input's storage.
    expected_owner = original.clone()
    expected_owner[3:].as_strided((2, 7, 8), (128, 12, 1)).mul_(-0.5)
    torch.testing.assert_close(owner, expected_owner, rtol=0, atol=0)
