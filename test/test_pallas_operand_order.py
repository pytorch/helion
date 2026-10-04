"""Tests for the operand order of commutative broadcasts on Pallas."""

from __future__ import annotations

import re

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="pallas", static_shapes=True)
def scaled_rows(state: torch.Tensor, k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    hv, dv, _ = state.shape
    out = torch.empty([hv, dv], dtype=torch.float32, device=state.device)
    for _ in hl.grid(1):
        s = state[:, :, :]
        k_b = k[:, :][:, None, :]
        out[:, :] = g[:][:, None] * torch.sum(k_b * s, -1)
    return out


@onlyBackends(["pallas"])
class TestOperandOrder(TestCase):
    def test_broadcast_first_for_lane_reduced_operand(self) -> None:
        hv, dv, dk = 2, 128, 128
        state = torch.randn(hv, dv, dk, device=DEVICE)
        k = torch.randn(hv, dk, device=DEVICE)
        g = torch.randn(hv, device=DEVICE)
        code, out = code_and_output(scaled_rows, (state, k, g))
        torch.testing.assert_close(
            out, g[:, None] * (state * k[:, None, :]).sum(-1), atol=1e-4, rtol=1e-4
        )
        # The full-size state goes first in the broadcast product ...
        self.assertIn("= s * k_b\n", code)
        # ... but the [2, 128] lane reduction goes second: its layout runs
        # down the sublanes, while the broadcast of g has the compact layout.
        reduced = re.search(r"(\w+) = lax\.convert_element_type\(jnp\.sum\(", code)
        assert reduced is not None
        self.assertRegex(code, rf"= \w+ \* {reduced.group(1)}\n")
