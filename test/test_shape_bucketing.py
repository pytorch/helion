from __future__ import annotations

import unittest

import torch

from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
import helion.language as hl
from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


def _dummy(x: torch.Tensor) -> torch.Tensor:
    return x


class TestShapeBucketing(TestCase):
    def test_min2_bucketing_default(self) -> None:
        k = kernel(_dummy, settings=Settings(static_shapes=False))

        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t7 = torch.empty(7, 3)

        key_0 = k.specialization_key([t0])
        key_1 = k.specialization_key([t1])
        key_2 = k.specialization_key([t2])
        key_7 = k.specialization_key([t7])

        # min2: 0,1,>=2 (as 2)
        self.assertNotEqual(key_0, key_2)
        self.assertNotEqual(key_1, key_2)
        self.assertEqual(key_2, key_7)

    def test_zero_nonzero_bucketing(self) -> None:
        k = kernel(
            _dummy,
            settings=Settings(static_shapes=False, shape_bucketing="zero_nonzero"),
        )

        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)

        key_0 = k.specialization_key([t0])
        key_1 = k.specialization_key([t1])
        key_2 = k.specialization_key([t2])

        # zero_nonzero: keep 0 distinct; unify 1 with >=2
        self.assertNotEqual(key_0, key_2)
        self.assertEqual(key_1, key_2)

    @skipIfNotCUDA()
    def test_zero_nonzero_runtime_correctness_smaller_first(self) -> None:
        """Test compiling with size==1 first, then running on size==2."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16

        # Compile with M=1 first (smaller), then reuse for M=2 (larger)
        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        x2 = torch.randn(2, K, device=device, dtype=torch.float32)

        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)
        pw_add(x1, y1)  # compile with size==1
        pw_add(x2, y2)  # reuse for size==2

        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

    @skipIfNotCUDA()
    def test_zero_nonzero_runtime_correctness_larger_first(self) -> None:
        """Test compiling with size==2 first, then running on size==1."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16

        # Compile with M=2 first (larger), then reuse for M=1 (smaller)
        x2 = torch.randn(2, K, device=device, dtype=torch.float32)
        x1 = torch.randn(1, K, device=device, dtype=torch.float32)

        y2 = torch.empty_like(x2)
        y1 = torch.empty_like(x1)
        pw_add(x2, y2)  # compile with size==2
        pw_add(x1, y1)  # reuse for size==1

        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

    @skipIfNotCUDA()
    def test_codegen_differs_for_singleton(self) -> None:
        """Test that min2 bucketing produces different code for M=1 vs M=2."""
        def pw_add_fn(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16

        # Use min2 to force distinct specialization keys per shape
        settings = Settings(
            static_shapes=False, autotune_effort="none", shape_bucketing="min2"
        )

        k1 = kernel(pw_add_fn, settings=settings)
        k2 = kernel(pw_add_fn, settings=settings)

        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        b1 = k1.bind((x1, y1))
        code1 = b1.to_triton_code()

        x2 = torch.randn(2, K, device=device, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        b2 = k2.bind((x2, y2))
        code2 = b2.to_triton_code()

        # With min2 bucketing, M=1 and M=2 should produce different code
        self.assertNotEqual(code1, code2)

    @skipIfNotCUDA()
    def test_zero_nonzero_general_only_single_compile_smaller_first(self) -> None:
        """Compile first with M=1, then call with M=2 under zero_nonzero; ensure a single compiled callable is reused."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16

        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        x2 = torch.randn(2, K, device=device, dtype=torch.float32)
        y2 = torch.empty_like(x2)

        # Bind on M=1 to capture the bound kernel instance
        b = pw_add.bind((x1, y1))

        # First call (M=1) → compile once
        pw_add(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Second call (M=2) → should not compile again
        pw_add(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Subsequent calls should reuse without increasing cache entries
        num_entries = len(b._compile_cache)
        pw_add(x2, y2)
        pw_add(x1, y1)
        self.assertEqual(len(b._compile_cache), num_entries)

    @skipIfNotCUDA()
    def test_zero_nonzero_runtime_correctness_varying_singleton_dim_row_to_col(
        self,
    ) -> None:
        """Compile at (1, K) then run at (K, 1) under zero_nonzero; must be correct and reuse single compiled callable."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16

        x_row = torch.randn(1, K, device=device, dtype=torch.float32)
        y_row = torch.empty_like(x_row)
        x_col = torch.randn(K, 1, device=device, dtype=torch.float32)
        y_col = torch.empty_like(x_col)

        # Bind on (1, K) to capture the bound kernel instance
        b = pw_add.bind((x_row, y_row))

        # First call compiles once
        pw_add(x_row, y_row)
        torch.testing.assert_close(y_row, x_row + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Flip which dim is 1; still correct; cache unchanged
        pw_add(x_col, y_col)
        torch.testing.assert_close(y_col, x_col + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

    @skipIfNotCUDA()
    def test_zero_nonzero_runtime_correctness_varying_singleton_dim_col_to_row(
        self,
    ) -> None:
        """Compile at (K, 1) then run at (1, K) under zero_nonzero; must be correct and reuse single compiled callable."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16

        x_col = torch.randn(K, 1, device=device, dtype=torch.float32)
        y_col = torch.empty_like(x_col)
        x_row = torch.randn(1, K, device=device, dtype=torch.float32)
        y_row = torch.empty_like(x_row)

        b = pw_add.bind((x_col, y_col))

        # First call compiles once
        pw_add(x_col, y_col)
        torch.testing.assert_close(y_col, x_col + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Flip which dim is 1; still correct; cache unchanged
        pw_add(x_row, y_row)
        torch.testing.assert_close(y_row, x_row + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

    @skipIfNotCUDA()
    def test_zero_nonzero_codegen_identical_m1_vs_m2(self) -> None:
        """Under zero_nonzero, M=1 vs M=2 should produce identical codegen."""
        def pw_add_fn(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 16
        settings = Settings(
            static_shapes=False, autotune_effort="none", shape_bucketing="zero_nonzero"
        )

        k1 = kernel(pw_add_fn, settings=settings)
        k2 = kernel(pw_add_fn, settings=settings)

        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        b1 = k1.bind((x1, y1))
        code1 = b1.to_triton_code()

        x2 = torch.randn(2, K, device=device, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        b2 = k2.bind((x2, y2))
        code2 = b2.to_triton_code()

        # Under zero_nonzero, code should be identical
        self.assertEqual(code1, code2)
        # Only journal once since they're the same
        self.assertExpectedJournal(code1)

    @skipIfNotCUDA()
    def test_zero_nonzero_runtime_correctness_varying_singleton_dim_3d(self) -> None:
        """Compile at (1, K, K) then run across different 3D 1-ness patterns; must be correct and reuse a single compiled callable."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def pw_add3d(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        device = torch.device("cuda", 0)
        K = 8

        x100 = torch.randn(1, K, K, device=device, dtype=torch.float32)
        y100 = torch.empty_like(x100)

        # Bind and compile once with (1, K, K)
        b = pw_add3d.bind((x100, y100))
        pw_add3d(x100, y100)
        torch.testing.assert_close(y100, x100 + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Now flip which dimension is 1 across various patterns, correctness should hold and cache size remain 1
        for shape in [(K, 1, K), (K, K, 1), (1, 1, K), (1, K, 1), (K, 1, 1)]:
            x = torch.randn(*shape, device=device, dtype=torch.float32)
            y = torch.empty_like(x)
            pw_add3d(x, y)
            torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)
            self.assertEqual(len(b._compile_cache), 1)

    @skipIfNotCUDA()
    def test_zero_nonzero_reduction_smaller_first(self) -> None:
        """Test reduction kernel with zero_nonzero, compiling with size==1 first."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        device = torch.device("cuda", 0)
        K = 64

        # Compile with M=1 first (smaller), then reuse for M=2 and M=3
        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        x2 = torch.randn(2, K, device=device, dtype=torch.float32)
        x3 = torch.randn(3, K, device=device, dtype=torch.float32)

        result1 = row_sum(x1)
        result2 = row_sum(x2)
        result3 = row_sum(x3)

        torch.testing.assert_close(result1, x1.sum(-1))
        torch.testing.assert_close(result2, x2.sum(-1))
        torch.testing.assert_close(result3, x3.sum(-1))

    @skipIfNotCUDA()
    def test_zero_nonzero_reduction_larger_first(self) -> None:
        """Test reduction kernel with zero_nonzero, compiling with size==2 first."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        device = torch.device("cuda", 0)
        K = 64

        # Compile with M=2 first (larger), then reuse for M=1
        x2 = torch.randn(2, K, device=device, dtype=torch.float32)
        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        x3 = torch.randn(3, K, device=device, dtype=torch.float32)

        result2 = row_sum(x2)
        result1 = row_sum(x1)
        result3 = row_sum(x3)

        torch.testing.assert_close(result2, x2.sum(-1))
        torch.testing.assert_close(result1, x1.sum(-1))
        torch.testing.assert_close(result3, x3.sum(-1))

    @skipIfNotCUDA()
    def test_zero_nonzero_reduction_cache_key(self) -> None:
        """Test that reduction kernels share the same bound kernel under zero_nonzero."""
        @kernel(
            settings=Settings(
                static_shapes=False,
                shape_bucketing="zero_nonzero",
                autotune_effort="none",
            )
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        device = torch.device("cuda", 0)
        K = 64

        x1 = torch.randn(1, K, device=device, dtype=torch.float32)
        x2 = torch.randn(2, K, device=device, dtype=torch.float32)

        # Verify specialization keys are the same
        key1 = row_sum.specialization_key((x1,))
        key2 = row_sum.specialization_key((x2,))
        self.assertEqual(key1, key2)

        # Bind smaller shape first to compile the kernel
        bound1 = row_sum.bind((x1,))
        bound2 = row_sum.bind((x2,))

        # Should share the same bound kernel
        self.assertIs(bound1, bound2)


if __name__ == "__main__":
    unittest.main()
