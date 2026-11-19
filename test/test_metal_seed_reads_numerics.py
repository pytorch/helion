"""The kernel of ``test_metal_seed_reads.py`` on an Apple GPU: the fallback
computes, where MPP read the addmm's result for the second loop's start."""

from __future__ import annotations

import sys
import unittest

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.metal_seed_reads import SEED_READ_CONFIG
from helion._testing.metal_seed_reads import carry_read_after_its_update
from helion._testing.metal_seed_reads import carry_read_after_its_update_op
from helion._testing.metal_seed_reads import seed_read_after_its_loop

# Metal runs on macOS only: CI's Linux jobs skip the whole file.
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]


@unittest.skipUnless(torch.backends.mps.is_available(), "requires a Metal GPU")
class TestMetalSeedReadNumerics(unittest.TestCase):
    def test_seed_read_after_its_loop(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        y = torch.randn(64, 32, device=DEVICE)
        b = torch.randn(32, 32, device=DEVICE)
        bound = seed_read_after_its_loop.bind((x, y, b))
        out = bound.compile_config(SEED_READ_CONFIG)(x, y, b)
        expected = x.cpu().double() @ y.cpu().double() + 4 * b.cpu().double()
        torch.testing.assert_close(out.cpu().double(), expected, rtol=1e-4, atol=1e-4)

    def test_carry_read_after_its_update(self) -> None:
        """``prev`` is the accumulator before the last K tile's addmm."""
        x = torch.randn(32, 64, device=DEVICE)
        y = torch.randn(64, 32, device=DEVICE)
        bound = carry_read_after_its_update.bind((x, y))
        out, prev = bound.compile_config(SEED_READ_CONFIG)(x, y)
        xd, yd = x.cpu().double(), y.cpu().double()
        torch.testing.assert_close(out.cpu().double(), xd @ yd, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            prev.cpu().double(), xd[:, :48] @ yd[:48], rtol=1e-4, atol=1e-4
        )

    def test_carry_read_after_its_update_op(self) -> None:
        """``prev`` is the carry before the last K tile's update."""
        x = torch.randn(32, 64, device=DEVICE)
        y = torch.randn(64, 32, device=DEVICE)
        bound = carry_read_after_its_update_op.bind((x, y))
        out, prev = bound.compile_config(SEED_READ_CONFIG)(x, y)
        xd, yd = x.cpu().double(), y.cpu().double()
        acc = torch.zeros(32, 32, dtype=torch.float64)
        before = acc
        for k0 in range(0, 64, 16):
            before = acc
            acc = acc * 0.5 + xd[:, k0 : k0 + 16] @ yd[k0 : k0 + 16]
        torch.testing.assert_close(out.cpu().double(), acc, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(prev.cpu().double(), before, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
