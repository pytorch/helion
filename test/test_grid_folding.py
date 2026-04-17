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
class TestGridFolding(TestCase):
    def test_2d_full_folding(self):
        """Full folding (factor=-1) on last dim of 2D tile into inner loop."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, -1]]
        )
        torch.testing.assert_close(result, x + y)
        # Foldinged dim should use tl.range, not tl.program_id for that dim
        self.assertIn("tl.range(", code)

    def test_2d_no_folding(self):
        """Baseline: no folding uses only program_id."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 0]]
        )
        torch.testing.assert_close(result, x + y)
        # No folding means no tl.range for grid dims
        self.assertNotIn("tl.range(", code)

    def test_3d_full_folding_1(self):
        """Full folding on last dim of 3D tile into inner loop."""
        x = torch.randn(8, 16, 32, device=DEVICE)
        y = torch.randn(8, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel, (x, y), block_sizes=[8, 16, 32], grid_foldings=[[0, 0, -1]]
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_3d_full_folding_2(self):
        """Full folding on last 2 dims of 3D tile into inner loops."""
        x = torch.randn(8, 16, 32, device=DEVICE)
        y = torch.randn(8, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, -1, -1]],
        )
        torch.testing.assert_close(result, x + y)
        # Two folded dims means tl.range should appear
        self.assertIn("tl.range(", code)

    def test_full_folding_correctness_non_divisible(self):
        """Full folding with non-block-divisible sizes."""
        x = torch.randn(50, 70, device=DEVICE)
        y = torch.randn(50, 70, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, -1]]
        )
        torch.testing.assert_close(result, x + y)

    def test_singular_alias(self):
        """grid_folding (singular) should work as alias for grid_foldings."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_folding=[0, -1]
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_persistent_rejects_folding(self):
        """Persistent kernel should reject non-zero grid_foldings."""
        from helion.exc import InvalidConfig

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        with self.assertRaises(InvalidConfig):
            code_and_output(
                add_2d_kernel,
                (x, y),
                block_sizes=[32, 32],
                grid_foldings=[[0, -1]],
                pid_type="persistent_blocked",
            )

    def test_2d_partial_folding(self):
        """Partial folding (factor=2) on last dim: grid shrinks by 2, loop of 2."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 2]]
        )
        torch.testing.assert_close(result, x + y)
        # Partial folding should use both program_id and tl.range
        self.assertIn("tl.range(", code)

    def test_3d_partial_folding_last_dim(self):
        """Partial folding (factor=4) on last dim of 3D tile."""
        x = torch.randn(8, 16, 128, device=DEVICE)
        y = torch.randn(8, 16, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 0, 4]],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_partial_folding_non_divisible(self):
        """Partial folding with non-divisible sizes."""
        x = torch.randn(50, 70, device=DEVICE)
        y = torch.randn(50, 70, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 2]]
        )
        torch.testing.assert_close(result, x + y)

    def test_mixed_partial_and_full_folding(self):
        """Mix of partial folding on one dim and full folding on another."""
        x = torch.randn(8, 64, 128, device=DEVICE)
        y = torch.randn(8, 64, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 2, -1]],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_nested_partial_folding(self):
        """Partial folding on multiple dims simultaneously.

        This is a regression test: previously folding loop variables for
        outer dims were never used, causing stale offsets in the kernel body.
        """
        x = torch.randn(8, 64, 128, device=DEVICE)
        y = torch.randn(8, 64, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[2, 16, 32],
            grid_foldings=[[2, 2, 0]],
        )
        torch.testing.assert_close(result, x + y)
        # Both folding loops should appear
        self.assertGreaterEqual(code.count("tl.range("), 2)

    def test_partial_folding_non_divisible_dim(self):
        """Partial folding where dimension size is not divisible by factor.

        Regression test: block_size=1 dims skip mask generation
        (known_multiple(1) is always true), but partial folding iterates
        ceil(numel/factor)*factor times which can exceed numel, causing
        illegal memory access without a bounds mask.
        """
        # dim 0 size (5) is NOT divisible by folding factor (4)
        x = torch.randn(5, 64, 128, device=DEVICE)
        y = torch.randn(5, 64, 128, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[1, 16, 32],
            grid_foldings=[[4, 0, 0]],
        )
        torch.testing.assert_close(result, x + y)

    def test_partial_folding_large_factor_oob(self):
        """Large folding factor with small dim → most iterations out-of-bounds.

        Regression test for the paged-attention crash: when a folding factor
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
            grid_foldings=[[0, 0, 64]],
        )
        torch.testing.assert_close(result, x + y)

    def test_all_dims_folded_rejects(self):
        """Foldinging ALL dims collapses the grid to 1 → InvalidConfig."""
        from helion.exc import InvalidConfig

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        with self.assertRaises(InvalidConfig):
            code_and_output(
                add_2d_kernel,
                (x, y),
                block_sizes=[32, 32],
                grid_foldings=[[-1, -1]],
            )


class TestGridFoldingGridSize(TestCase):
    """Verify that grid folding actually reduces the GPU launch grid."""

    @staticmethod
    def _extract_grid_size(code: str) -> int:
        """Evaluate the grid tuple from the generated launcher call.

        Parses ``_launcher(_kernel, (expr,), ...)`` and evaluates the
        grid expression using the ``_BLOCK_SIZE_*`` constants defined
        in the host wrapper.
        """
        import re

        # Collect _BLOCK_SIZE_* = <int> definitions from the host wrapper
        block_sizes: dict[str, int] = {}
        for m in re.finditer(r"(_BLOCK_SIZE_\d+)\s*=\s*(\d+)", code):
            block_sizes[m.group(1)] = int(m.group(2))

        # Find the grid tuple by matching balanced parentheses after _launcher
        launcher_idx = code.find("_launcher(")
        assert launcher_idx >= 0, "Could not find _launcher call"
        # Skip to the grid tuple: _launcher(_kernel, (<grid>,), ...)
        # Find the second '(' which starts the grid tuple
        first_paren = code.index("(", launcher_idx)
        comma = code.index(",", first_paren)
        grid_start = code.index("(", comma)
        # Match balanced parens to find the grid tuple end
        depth = 0
        for i in range(grid_start, len(code)):
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
                if depth == 0:
                    grid_tuple_str = code[grid_start : i + 1]
                    break
        else:
            raise AssertionError("Unbalanced parentheses in grid tuple")

        # Evaluate the tuple to get the grid size (product of all elements)
        import math

        import triton

        grid_tuple = eval(grid_tuple_str, {"triton": triton, **block_sizes})
        return math.prod(grid_tuple)

    def test_full_folding_reduces_grid(self):
        """Full folding [0,-1] should halve the grid vs [0,0] for a 2D tile."""
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        code_nf, _ = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 0]]
        )
        code_ff, _ = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, -1]]
        )
        grid_nf = self._extract_grid_size(code_nf)
        grid_ff = self._extract_grid_size(code_ff)
        # 128/32=4 blocks per dim. No folding: 4*4=16. Full folding on N: 4.
        self.assertEqual(grid_nf, 16)
        self.assertEqual(grid_ff, 4)

    def test_partial_folding_reduces_grid(self):
        """Partial folding [2,0] should shrink M grid by factor 2."""
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        code_nf, _ = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 0]]
        )
        code_pf, _ = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[2, 0]]
        )
        grid_nf = self._extract_grid_size(code_nf)
        grid_pf = self._extract_grid_size(code_pf)
        # No folding: 4*4=16. Partial M=2: (4/2)*4=8.
        self.assertEqual(grid_nf, 16)
        self.assertEqual(grid_pf, 8)

    def test_3d_full_folding_reduces_grid(self):
        """Full folding on last dim of 3D tile reduces grid proportionally."""
        x = torch.randn(8, 16, 64, device=DEVICE)
        y = torch.randn(8, 16, 64, device=DEVICE)
        code_nf, _ = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 0, 0]],
        )
        code_ff, _ = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 0, -1]],
        )
        grid_nf = self._extract_grid_size(code_nf)
        grid_ff = self._extract_grid_size(code_ff)
        # 8/8=1, 16/16=1, 64/32=2. No folding: 1*1*2=2. Full on C: 1*1=1.
        self.assertEqual(grid_nf, 2)
        self.assertEqual(grid_ff, 1)

    def test_mixed_folding_reduces_grid(self):
        """Mixed partial+full folding compounds the grid reduction."""
        x = torch.randn(8, 64, 128, device=DEVICE)
        y = torch.randn(8, 64, 128, device=DEVICE)
        code_nf, _ = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 0, 0]],
        )
        code_mf, _ = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 2, -1]],
        )
        grid_nf = self._extract_grid_size(code_nf)
        grid_mf = self._extract_grid_size(code_mf)
        # 8/8=1, 64/16=4, 128/32=4. No folding: 1*4*4=16.
        # Partial B=2, full C: 1*(4/2)=2.
        self.assertEqual(grid_nf, 16)
        self.assertEqual(grid_mf, 2)


if __name__ == "__main__":
    unittest.main()
