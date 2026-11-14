"""Native replay checks for full-slice consumer axis liveness."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cute_phase_liveness import slice_sum_phases


@pytest.mark.parametrize("tensor_indices", [False, True])
def test_projected_static_memory_consumer_matches_reference(tensor_indices):
    x = (torch.arange(2 * 3 * 7 * 8, device=DEVICE).reshape(2, 3, 7, 8) % 7 - 3).to(
        torch.bfloat16
    )
    y = (torch.arange(2 * 3 * 8 * 11, device=DEVICE).reshape(2, 3, 8, 11) % 5 - 2).to(
        torch.bfloat16
    )
    bound = slice_sum_phases.bind((x, y, tensor_indices))
    config = dict(bound.config_spec.default_config()) | {
        "pid_type": "flat",
    }
    compiled = bound.compile_config(helion.Config.from_dict(config))
    for multiplier in (1.0, -1.0):
        x.mul_(multiplier)
        value = x.float() @ y.float()
        expected = (
            value + value.sum(dim=2, keepdim=True) + value.sum(dim=3, keepdim=True)
        ).sum(dim=1)
        assert not torch.equal(expected[0], expected[1])
        assert not torch.equal(expected[:, 0], expected[:, 1])
        assert not torch.equal(expected[:, :, 0], expected[:, :, 1])
        torch.testing.assert_close(
            compiled(x, y, tensor_indices), expected, rtol=0, atol=0
        )
