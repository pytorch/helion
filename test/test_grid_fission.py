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
    def test_2d_full_fission(self):
        """Full fission (factor=-1) on last dim of 2D tile into inner loop."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[[0, -1]]
        )
        torch.testing.assert_close(result, x + y)
        # Fissioned dim should use tl.range, not tl.program_id for that dim
        self.assertIn("tl.range(", code)

    def test_2d_no_fission(self):
        """Baseline: no fission uses only program_id."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[[0, 0]]
        )
        torch.testing.assert_close(result, x + y)
        # No fission means no tl.range for grid dims
        self.assertNotIn("tl.range(", code)

    def test_3d_full_fission_1(self):
        """Full fission on last dim of 3D tile into inner loop."""
        x = torch.randn(8, 16, 32, device=DEVICE)
        y = torch.randn(8, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel, (x, y), block_sizes=[8, 16, 32], grid_fissions=[[0, 0, -1]]
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_3d_full_fission_2(self):
        """Full fission on last 2 dims of 3D tile into inner loops."""
        x = torch.randn(8, 16, 32, device=DEVICE)
        y = torch.randn(8, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_fissions=[[0, -1, -1]],
        )
        torch.testing.assert_close(result, x + y)
        # Two fissioned dims means tl.range should appear
        self.assertIn("tl.range(", code)

    def test_full_fission_correctness_non_divisible(self):
        """Full fission with non-block-divisible sizes."""
        x = torch.randn(50, 70, device=DEVICE)
        y = torch.randn(50, 70, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[[0, -1]]
        )
        torch.testing.assert_close(result, x + y)

    def test_singular_alias(self):
        """grid_fission (singular) should work as alias for grid_fissions."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fission=[0, -1]
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
                grid_fissions=[[0, -1]],
                pid_type="persistent_blocked",
            )

    def test_2d_partial_fission(self):
        """Partial fission (factor=2) on last dim: grid shrinks by 2, loop of 2."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[[0, 2]]
        )
        torch.testing.assert_close(result, x + y)
        # Partial fission should use both program_id and tl.range
        self.assertIn("tl.range(", code)

    def test_3d_partial_fission_last_dim(self):
        """Partial fission (factor=4) on last dim of 3D tile."""
        x = torch.randn(8, 16, 128, device=DEVICE)
        y = torch.randn(8, 16, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_fissions=[[0, 0, 4]],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_partial_fission_non_divisible(self):
        """Partial fission with non-divisible sizes."""
        x = torch.randn(50, 70, device=DEVICE)
        y = torch.randn(50, 70, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_fissions=[[0, 2]]
        )
        torch.testing.assert_close(result, x + y)

    def test_mixed_partial_and_full_fission(self):
        """Mix of partial fission on one dim and full fission on another."""
        x = torch.randn(8, 64, 128, device=DEVICE)
        y = torch.randn(8, 64, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_fissions=[[0, 2, -1]],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_nested_partial_fission(self):
        """Partial fission on multiple dims simultaneously.

        This is a regression test: previously fission loop variables for
        outer dims were never used, causing stale offsets in the kernel body.
        """
        x = torch.randn(8, 64, 128, device=DEVICE)
        y = torch.randn(8, 64, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[2, 16, 32],
            grid_fissions=[[2, 2, 0]],
        )
        torch.testing.assert_close(result, x + y)
        # Both fission loops should appear
        self.assertGreaterEqual(code.count("tl.range("), 2)

    def test_partial_fission_non_divisible_dim(self):
        """Partial fission where dimension size is not divisible by factor.

        Regression test: block_size=1 dims skip mask generation
        (known_multiple(1) is always true), but partial fission iterates
        ceil(numel/factor)*factor times which can exceed numel, causing
        illegal memory access without a bounds mask.
        """
        # dim 0 size (5) is NOT divisible by fission factor (4)
        x = torch.randn(5, 64, 128, device=DEVICE)
        y = torch.randn(5, 64, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[1, 16, 32],
            grid_fissions=[[4, 0, 0]],
        )
        torch.testing.assert_close(result, x + y)

    def test_partial_fission_large_factor_oob(self):
        """Large fission factor with small dim → most iterations out-of-bounds.

        Regression test for the paged-attention crash: when a fission factor
        (e.g. 64) far exceeds ceil(dim_size / block_size) (e.g. 4), the
        out-of-bounds offset inflated inner-loop trip counts, causing illegal
        memory access.  The if-guard must skip the entire body.
        """
        # dim 2 size=32, block_size=8, factor=64 → only 4 of 64 iters valid
        x = torch.randn(4, 16, 32, device=DEVICE)
        y = torch.randn(4, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[4, 16, 8],
            grid_fissions=[[0, 0, 64]],
        )
        torch.testing.assert_close(result, x + y)


if __name__ == "__main__":
    unittest.main()
