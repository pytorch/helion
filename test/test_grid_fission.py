from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(autotune_effort="none")
def add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        result[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return result


@helion.kernel(autotune_effort="none")
def add_3d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile_a, tile_b, tile_c in hl.tile([x.size(0), x.size(1), x.size(2)]):
        result[tile_a, tile_b, tile_c] = (
            x[tile_a, tile_b, tile_c] + y[tile_a, tile_b, tile_c]
        )
    return result


@onlyBackends(["triton"])
class TestGridFission(TestCase):
    def test_2d_fission_1(self):
        """Fission last dim of 2D tile into inner loop."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[1]
        )
        torch.testing.assert_close(result, x + y)
        # Fissioned dim should use tl.range, not tl.program_id for that dim
        self.assertIn("tl.range(", code)

    def test_2d_no_fission(self):
        """Baseline: no fission uses only program_id."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[0]
        )
        torch.testing.assert_close(result, x + y)
        # No fission means no tl.range for grid dims
        self.assertNotIn("tl.range(", code)

    def test_3d_fission_1(self):
        """Fission last dim of 3D tile into inner loop."""
        x = torch.randn(8, 16, 32, device=DEVICE)
        y = torch.randn(8, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel, (x, y), block_sizes=[8, 16, 32], grid_fissions=[1]
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_3d_fission_2(self):
        """Fission last 2 dims of 3D tile into inner loops."""
        x = torch.randn(8, 16, 32, device=DEVICE)
        y = torch.randn(8, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel, (x, y), block_sizes=[8, 16, 32], grid_fissions=[2]
        )
        torch.testing.assert_close(result, x + y)
        # Two fissioned dims means tl.range should appear
        self.assertIn("tl.range(", code)

    def test_fission_correctness_non_divisible(self):
        """Fission with non-block-divisible sizes."""
        x = torch.randn(50, 70, device=DEVICE)
        y = torch.randn(50, 70, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[1]
        )
        torch.testing.assert_close(result, x + y)

    def test_singular_alias(self):
        """grid_fission (singular) should work as alias for grid_fissions."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fission=[1]
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_persistent_rejects_fission(self):
        """Persistent kernel should reject non-zero grid_fissions."""
        from helion.exc import InvalidConfig

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        with self.assertRaises(InvalidConfig):
            code_and_output(
                add_2d_kernel,
                (x, y),
                block_sizes=[32, 32],
                grid_fissions=[1],
                pid_type="persistent_blocked",
            )


if __name__ == "__main__":
    unittest.main()
