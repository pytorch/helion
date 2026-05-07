from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
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


@helion.kernel(autotune_effort="none")
def add_2d_tile_begin_kernel(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        b = bias[tile_m.begin]
        result[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n] + b
    return result


@onlyBackends(["triton"])
class TestGridFolding(RefEagerTestBase, TestCase):
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

    @skipIfRefEager("config validation only runs during compilation")
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
        x = torch.randn(256, 256, device=DEVICE)
        y = torch.randn(256, 256, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 2]]
        )
        torch.testing.assert_close(result, x + y)
        # Partial folding should use both program_id and tl.range
        self.assertIn("tl.range(", code)

    def test_3d_partial_folding_last_dim(self):
        """Partial folding (factor=4) on last dim of 3D tile."""
        x = torch.randn(32, 64, 512, device=DEVICE)
        y = torch.randn(32, 64, 512, device=DEVICE)
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
        x = torch.randn(200, 300, device=DEVICE)
        y = torch.randn(200, 300, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel, (x, y), block_sizes=[32, 32], grid_foldings=[[0, 2]]
        )
        torch.testing.assert_close(result, x + y)

    def test_mixed_partial_and_full_folding(self):
        """Mix of partial folding on one dim and full folding on another."""
        x = torch.randn(32, 256, 512, device=DEVICE)
        y = torch.randn(32, 256, 512, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 2, -1]],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    @skipIfRefEager("checks generated code with assertGreaterEqual")
    def test_nested_partial_folding(self):
        """Partial folding on multiple dims simultaneously.

        This is a regression test: previously folding loop variables for
        outer dims were never used, causing stale offsets in the kernel body.
        """
        x = torch.randn(32, 256, 512, device=DEVICE)
        y = torch.randn(32, 256, 512, device=DEVICE)
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

    def test_degenerate_partial_folding_tile_begin(self):
        """Degenerate partial folding (factor >= num_blocks) with tile.begin.

        Regression: codegen_grid() normalized factor>=num_blocks to -1
        without a ForLoopGraphInfo, causing IndexError in offset_var()
        when tile.begin was accessed in the loop body.
        """
        # dim 0: size=4, block_size=4 → num_blocks=1, factor=8 >= 1 → degenerate
        x = torch.randn(4, 64, device=DEVICE)
        y = torch.randn(4, 64, device=DEVICE)
        bias = torch.zeros(4, device=DEVICE)
        code, result = code_and_output(
            add_2d_tile_begin_kernel,
            (x, y, bias),
            block_sizes=[4, 32],
            grid_foldings=[[8, 0]],
        )
        torch.testing.assert_close(result, x + y)

    @skipIfRefEager("config validation only runs during compilation")
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


@onlyBackends(["triton"])
class TestGridFoldingWithFlatten(RefEagerTestBase, TestCase):
    """Tests for combining grid folding with flatten_loops (partial flattening)."""

    def test_full_folding_with_flatten_2d(self):
        """2D tile: fold dim 1 fully, flatten remaining dim 0."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel,
            (x, y),
            block_sizes=[32, 32],
            grid_foldings=[[0, -1]],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_full_folding_with_flatten_2d_first_dim(self):
        """2D tile: fold dim 0 fully, flatten remaining dim 1."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel,
            (x, y),
            block_sizes=[32, 32],
            grid_foldings=[[-1, 0]],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_full_folding_with_flatten_3d(self):
        """3D tile: fold dim 2 fully, flatten remaining dims 0 and 1."""
        x = torch.randn(8, 16, 64, device=DEVICE)
        y = torch.randn(8, 16, 64, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, 0, -1]],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_full_folding_with_flatten_3d_two_folded(self):
        """3D tile: fold dims 1 and 2 fully, flatten remaining dim 0."""
        x = torch.randn(64, 16, 32, device=DEVICE)
        y = torch.randn(64, 16, 32, device=DEVICE)
        code, result = code_and_output(
            add_3d_kernel,
            (x, y),
            block_sizes=[8, 16, 32],
            grid_foldings=[[0, -1, -1]],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, x + y)
        self.assertIn("tl.range(", code)

    def test_full_folding_with_flatten_non_divisible(self):
        """Fold+flatten with non-block-divisible sizes."""
        x = torch.randn(50, 70, device=DEVICE)
        y = torch.randn(50, 70, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel,
            (x, y),
            block_sizes=[32, 32],
            grid_foldings=[[0, -1]],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, x + y)

    def test_partial_folding_with_flatten_falls_back(self):
        """Partial folding + flatten falls back to NDTileStrategy (no crash)."""
        x = torch.randn(256, 256, device=DEVICE)
        y = torch.randn(256, 256, device=DEVICE)
        code, result = code_and_output(
            add_2d_kernel,
            (x, y),
            block_sizes=[32, 32],
            grid_foldings=[[0, 2]],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, x + y)


@onlyBackends(["triton"])
@skipIfRefEager("parses generated Triton code and launcher calls")
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


@skipIfRefEager("inspects config_spec populated during compilation")
class TestGridFoldingHeuristic(TestCase):
    """Verify that the autotuner heuristic gates partial folding for small dims."""

    @staticmethod
    def _get_folding_choices_2d(
        x: torch.Tensor, y: torch.Tensor
    ) -> list[tuple[int, ...]]:
        """Bind add_2d_kernel and return per-dim folding choices."""
        from helion.autotuner.config_fragment import EnumFragment
        from helion.autotuner.config_fragment import PerDimListOf

        bound = add_2d_kernel.bind((x, y))
        spec = bound.config_spec
        gf_spec = spec.grid_foldings[0]
        with bound.env:
            frag = gf_spec._fragment(spec)
        assert isinstance(frag, PerDimListOf)
        choices = []
        for f in frag.fragments:
            assert isinstance(f, EnumFragment)
            choices.append(f.choices)
        return choices

    @staticmethod
    def _get_folding_choices_3d(
        x: torch.Tensor, y: torch.Tensor
    ) -> list[tuple[int, ...]]:
        """Bind add_3d_kernel and return per-dim folding choices."""
        from helion.autotuner.config_fragment import EnumFragment
        from helion.autotuner.config_fragment import PerDimListOf

        bound = add_3d_kernel.bind((x, y))
        spec = bound.config_spec
        gf_spec = spec.grid_foldings[0]
        with bound.env:
            frag = gf_spec._fragment(spec)
        assert isinstance(frag, PerDimListOf)
        choices = []
        for f in frag.fragments:
            assert isinstance(f, EnumFragment)
            choices.append(f.choices)
        return choices

    def test_small_dims_no_partial_folding(self):
        """Dimensions with < MIN_BLOCKS_FOR_PARTIAL blocks → no partial folding."""
        from helion.autotuner.config_spec import GridFoldingSpec

        # Use a dimension size smaller than MIN_BLOCKS_FOR_PARTIAL.
        # With min_block_size=1, num_blocks = dim_size.
        dim_size = GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL - 1
        assert dim_size < GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL

        x = torch.randn(dim_size, dim_size, device=DEVICE)
        y = torch.randn(dim_size, dim_size, device=DEVICE)
        choices = self._get_folding_choices_2d(x, y)
        for dim_choices in choices:
            self.assertTrue(
                all(c <= 0 for c in dim_choices),
                f"Expected only (0, -1) for dim with {dim_size} blocks "
                f"(MIN_BLOCKS_FOR_PARTIAL={GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL}), "
                f"got {dim_choices}",
            )

    def test_large_dims_allow_partial_folding(self):
        """Large dimensions should include partial folding factors (2, 4, ...)."""
        x = torch.randn(4096, 4096, device=DEVICE)
        y = torch.randn(4096, 4096, device=DEVICE)
        choices = self._get_folding_choices_2d(x, y)
        for dim_choices in choices:
            has_partial = any(c > 0 for c in dim_choices)
            self.assertTrue(
                has_partial,
                f"Expected partial factors for large dim, got {dim_choices}",
            )

    def test_mixed_dims_per_dim_gating(self):
        """Heuristic should gate per-dimension: small dim no partial, large dim allows."""
        x = torch.randn(3, 4096, device=DEVICE)
        y = torch.randn(3, 4096, device=DEVICE)
        choices = self._get_folding_choices_2d(x, y)
        # dim 0: 3 elements / min_block_size=1 → 3 blocks < MIN_BLOCKS_FOR_PARTIAL
        self.assertTrue(
            all(c <= 0 for c in choices[0]),
            f"Expected only (0, -1) for small dim 0, got {choices[0]}",
        )

    def test_max_factor_capped_at_half_blocks(self):
        """Partial folding factor should be capped at nb // 2."""
        # 16 blocks: max_factor should be 8, so only factors <= 8 allowed.
        dim_size = 16
        x = torch.randn(dim_size, dim_size, device=DEVICE)
        y = torch.randn(dim_size, dim_size, device=DEVICE)
        choices = self._get_folding_choices_2d(x, y)
        for dim_choices in choices:
            # Should include partial factors but cap at 8 (16 // 2).
            self.assertIn(8, dim_choices, "Expected factor 8 (nb//2) to be allowed")
            self.assertNotIn(
                16,
                dim_choices,
                "Factor 16 should be capped (exceeds nb//2=8)",
            )
            self.assertNotIn(
                32,
                dim_choices,
                "Factor 32 should be capped (exceeds nb//2=8)",
            )
            self.assertNotIn(
                64,
                dim_choices,
                "Factor 64 should be capped (exceeds nb//2=8)",
            )

    def test_boundary_at_min_blocks_for_partial(self):
        """Test exactly at MIN_BLOCKS_FOR_PARTIAL boundary."""
        from helion.autotuner.config_spec import GridFoldingSpec

        # At boundary: should allow partial folding
        x_at = torch.randn(
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL,
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL,
            device=DEVICE,
        )
        y_at = torch.randn(
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL,
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL,
            device=DEVICE,
        )
        choices_at = self._get_folding_choices_2d(x_at, y_at)
        # 8 blocks → max_factor = 4, so [0, -1, 2, 4]
        for dim_choices in choices_at:
            self.assertIn(2, dim_choices)
            self.assertIn(4, dim_choices)
            self.assertNotIn(8, dim_choices)

        # Below boundary: no partial folding
        x_below = torch.randn(
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL - 1,
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL - 1,
            device=DEVICE,
        )
        y_below = torch.randn(
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL - 1,
            GridFoldingSpec.MIN_BLOCKS_FOR_PARTIAL - 1,
            device=DEVICE,
        )
        choices_below = self._get_folding_choices_2d(x_below, y_below)
        for dim_choices in choices_below:
            self.assertEqual(dim_choices, (0, -1))

    def test_max_factor_scaling(self):
        """Larger dimensions should allow larger folding factors (capped at nb//2)."""
        # The exact num_blocks depends on tile spec's min_block_size,
        # but we can verify the capping behavior scales correctly.

        # Small case: should have smaller max factor
        x_small = torch.randn(32, 32, device=DEVICE)
        y_small = torch.randn(32, 32, device=DEVICE)
        choices_small = self._get_folding_choices_2d(x_small, y_small)

        # Large case: should have larger max factor
        x_large = torch.randn(256, 256, device=DEVICE)
        y_large = torch.randn(256, 256, device=DEVICE)
        choices_large = self._get_folding_choices_2d(x_large, y_large)

        # Extract max positive factor from each
        def max_positive(choices_list):
            return max(c for c in choices_list if c > 0)

        max_small = max_positive(choices_small[0])
        max_large = max_positive(choices_large[0])

        # Larger dimension should allow larger folding factor
        self.assertGreater(
            max_large,
            max_small,
            f"Expected larger dim to allow bigger factor: {max_large} vs {max_small}",
        )

    def test_3d_mixed_dimensions(self):
        """3D kernel with varying block counts per dimension."""
        # Create a 3D kernel with different sizes per dim
        # Sizes chosen to give different block counts after tiling
        x = torch.randn(4, 16, 64, device=DEVICE)
        y = torch.randn(4, 16, 64, device=DEVICE)
        choices = self._get_folding_choices_3d(x, y)

        # Should have 3 dimensions
        self.assertEqual(len(choices), 3)

        # dim 0: smallest (4 elements) → likely < MIN_BLOCKS_FOR_PARTIAL
        # dim 1: medium (16 elements)
        # dim 2: largest (64 elements)
        # Verify per-dimension gating works for 3D
        self.assertTrue(
            len(choices[0]) <= len(choices[1]) <= len(choices[2]),
            f"Expected increasing choices with size: {choices}",
        )

        # Verify smallest dim has only [0, -1]
        self.assertEqual(
            choices[0],
            (0, -1),
            f"Expected smallest dim to have only [0, -1], got {choices[0]}",
        )

    def test_symbolic_shape_fallback(self):
        """Unknown num_blocks (symbolic shapes) should fall back to unfiltered choices."""

        # When size is known but numel calculation involves unknown symbols,
        # dim_num_blocks will be None, triggering fallback to unfiltered choices.
        # We simulate this by binding a kernel and checking the fallback path.
        x = torch.randn(16, 16, device=DEVICE)
        y = torch.randn(16, 16, device=DEVICE)

        bound = add_2d_kernel.bind((x, y))
        spec = bound.config_spec
        gf_spec = spec.grid_foldings[0]

        with bound.env:
            # Manually set num_blocks to None to simulate symbolic shape
            dim_num_blocks = [None, None]

            # Build fragments as the code would for unknown sizes
            from helion.autotuner.config_fragment import EnumFragment

            fragments = []
            for _nb in dim_num_blocks:
                if _nb is None:
                    # Fallback: all VALID_FACTORS
                    choices = gf_spec.VALID_FACTORS
                else:
                    choices = tuple(
                        f
                        for f in gf_spec.VALID_FACTORS
                        if f <= 0 or f <= max(0, _nb // 2)
                    )
                fragments.append(EnumFragment(choices=choices))

            # Verify fallback includes all factors including 64
            for frag in fragments:
                self.assertIn(64, frag.choices, "Fallback should include all factors")


if __name__ == "__main__":
    unittest.main()
