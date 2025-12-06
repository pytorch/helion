from __future__ import annotations

import unittest

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfCpu
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


def _dummy(x: torch.Tensor) -> torch.Tensor:
    return x


@skipIfCpu("needs to be debugged")
class TestShapeBucketing(RefEagerTestBase, TestCase):
    maxDiff = 16384

    # =========================================================================
    # Specialization Key Tests
    # =========================================================================

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_zeros_mode_specialization_keys(self) -> None:
        """Test zeros mode: 0 is distinct, 1 and >=2 are unified."""
        k = kernel(_dummy, settings=Settings(static_shapes="zeros"))

        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t7 = torch.empty(7, 3)

        key_0 = k.specialization_key([t0])
        key_1 = k.specialization_key([t1])
        key_2 = k.specialization_key([t2])
        key_7 = k.specialization_key([t7])

        # zeros mode: 0 is distinct; 1 and >=2 are unified
        self.assertNotEqual(key_0, key_1)
        self.assertNotEqual(key_0, key_2)
        self.assertEqual(key_1, key_2)
        self.assertEqual(key_2, key_7)

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_zeros_ones_mode_specialization_keys(self) -> None:
        """Test zeros_ones mode: 0, 1, and >=2 are all distinct buckets."""
        k = kernel(_dummy, settings=Settings(static_shapes="zeros_ones"))

        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)
        t7 = torch.empty(7, 3)

        key_0 = k.specialization_key([t0])
        key_1 = k.specialization_key([t1])
        key_2 = k.specialization_key([t2])
        key_3 = k.specialization_key([t3])
        key_7 = k.specialization_key([t7])

        # zeros_ones: 0, 1, >=2 are distinct buckets
        self.assertNotEqual(key_0, key_1)
        self.assertNotEqual(key_0, key_2)
        self.assertNotEqual(key_1, key_2)
        # 2, 3, 7 all bucket to >=2
        self.assertEqual(key_2, key_3)
        self.assertEqual(key_2, key_7)

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_zeros_multidim_cache_key(self) -> None:
        """Test zeros mode cache key behavior with multiple dimensions."""
        k = kernel(_dummy, settings=Settings(static_shapes="zeros"))

        # Create tensors with different shapes: (1, 1) vs (2, 2) vs (1, 2)
        t_11 = torch.empty(1, 1)
        t_22 = torch.empty(2, 2)
        t_12 = torch.empty(1, 2)

        # In zeros mode, 1 and 2 are unified, so all keys should be the same
        key_11 = k.specialization_key([t_11])
        key_22 = k.specialization_key([t_22])
        key_12 = k.specialization_key([t_12])

        self.assertEqual(key_11, key_22)
        self.assertEqual(key_22, key_12)

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_zeros_ones_multidim_different_keys(self) -> None:
        """Test that zeros_ones mode creates different keys for different 0/1 patterns."""
        k = kernel(_dummy, settings=Settings(static_shapes="zeros_ones"))

        # Create tensors with different shapes
        t_11 = torch.empty(1, 1)
        t_22 = torch.empty(2, 2)
        t_12 = torch.empty(1, 2)

        # With zeros_ones mode, different 0/1 patterns produce different keys
        key_11 = k.specialization_key([t_11])
        key_22 = k.specialization_key([t_22])
        key_12 = k.specialization_key([t_12])

        self.assertNotEqual(key_11, key_22)
        self.assertNotEqual(key_22, key_12)
        self.assertNotEqual(key_11, key_12)

    # =========================================================================
    # Runtime Correctness Tests
    # =========================================================================

    @skipIfNotCUDA()
    def test_zeros_runtime_correctness(self) -> None:
        """Test zeros mode runtime correctness, compiling with size==1 first then size==2."""
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 16

        # Test size==1 first (per jansel's review comment - this is the harder case)
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        x3 = torch.randn(3, K, device=DEVICE, dtype=torch.float32)

        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)
        y3 = torch.empty_like(x3)

        # Compile with M=1 first, then reuse for M=2 and M=3
        pw_add(x1, y1)
        pw_add(x2, y2)
        pw_add(x3, y3)

        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y3, x3 + 1.0, rtol=1e-4, atol=1e-4)

        # Also test larger-first order to ensure bidirectional reuse
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def pw_add2(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 2.0

        # Compile with M=2 first, then reuse for M=1
        pw_add2(x2, y2)
        pw_add2(x1, y1)

        torch.testing.assert_close(y2, x2 + 2.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y1, x1 + 2.0, rtol=1e-4, atol=1e-4)

    @skipIfRefEager("compile cache not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_compile_cache_reuse(self) -> None:
        """Test that zeros mode reuses a single compiled callable across sizes."""
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 16

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)

        # Bind on M=1 to capture the bound kernel instance
        b = pw_add.bind((x1, y1))

        # First call (M=1) compiles once
        pw_add(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Second call (M=2) should not compile again
        pw_add(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Subsequent calls should reuse without increasing cache entries
        num_entries = len(b._compile_cache)
        pw_add(x2, y2)
        pw_add(x1, y1)
        self.assertEqual(len(b._compile_cache), num_entries)

    @skipIfRefEager("compile cache not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_varying_singleton_dim(self) -> None:
        """Test zeros mode with varying singleton dimensions (row vs col)."""
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 16

        x_row = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y_row = torch.empty_like(x_row)
        x_col = torch.randn(K, 1, device=DEVICE, dtype=torch.float32)
        y_col = torch.empty_like(x_col)

        # Bind on (1, K) to capture the bound kernel instance
        b = pw_add.bind((x_row, y_row))

        # Test row first (1, K), then column (K, 1)
        pw_add(x_row, y_row)
        torch.testing.assert_close(y_row, x_row + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        pw_add(x_col, y_col)
        torch.testing.assert_close(y_col, x_col + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Test the reverse order with a new kernel
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def pw_add2(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 2.0

        b2 = pw_add2.bind((x_col, y_col))

        # Column first (K, 1), then row (1, K)
        pw_add2(x_col, y_col)
        torch.testing.assert_close(y_col, x_col + 2.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b2._compile_cache), 1)

        pw_add2(x_row, y_row)
        torch.testing.assert_close(y_row, x_row + 2.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b2._compile_cache), 1)

    @skipIfRefEager("compile cache not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_varying_singleton_dim_3d(self) -> None:
        """Test zeros mode with 3D tensors and varying singleton patterns."""
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def pw_add3d(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 8

        x100 = torch.randn(1, K, K, device=DEVICE, dtype=torch.float32)
        y100 = torch.empty_like(x100)

        # Bind and compile once with (1, K, K)
        b = pw_add3d.bind((x100, y100))
        pw_add3d(x100, y100)
        torch.testing.assert_close(y100, x100 + 1.0, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(b._compile_cache), 1)

        # Test various 3D patterns, all should reuse the same compiled kernel
        for shape in [(K, 1, K), (K, K, 1), (1, 1, K), (1, K, 1), (K, 1, 1)]:
            x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
            y = torch.empty_like(x)
            pw_add3d(x, y)
            torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)
            self.assertEqual(len(b._compile_cache), 1)

    @skipIfRefEager("bound kernels not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_multidim_runtime_correctness(self) -> None:
        """Test zeros mode runtime correctness with multiple dimensions."""

        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 3
            return out

        # Create tensors with different shapes
        x_11 = torch.randn([1, 1], device=DEVICE)
        x_22 = torch.randn([2, 2], device=DEVICE)
        x_12 = torch.randn([1, 2], device=DEVICE)

        # With zeros mode, all should share the same bound kernel
        bound_11 = fn.bind((x_11,))
        bound_22 = fn.bind((x_22,))
        bound_12 = fn.bind((x_12,))

        self.assertTrueIfInNormalMode(bound_11 is bound_22)
        self.assertTrueIfInNormalMode(bound_22 is bound_12)

        # Verify correctness for all shapes
        result_11 = fn(x_11)
        result_22 = fn(x_22)
        result_12 = fn(x_12)
        torch.testing.assert_close(result_11, x_11 * 3)
        torch.testing.assert_close(result_22, x_22 * 3)
        torch.testing.assert_close(result_12, x_12 * 3)

    # =========================================================================
    # Reduction Kernel Tests
    # =========================================================================

    @skipIfNotCUDA()
    def test_zeros_reduction(self) -> None:
        """Test reduction kernel with zeros mode, testing both compile orders."""
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        K = 64

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        x3 = torch.randn(3, K, device=DEVICE, dtype=torch.float32)

        # Compile with M=1 first (per jansel's review), then reuse for M=2 and M=3
        result1 = row_sum(x1)
        result2 = row_sum(x2)
        result3 = row_sum(x3)

        torch.testing.assert_close(result1, x1.sum(-1))
        torch.testing.assert_close(result2, x2.sum(-1))
        torch.testing.assert_close(result3, x3.sum(-1))

        # Also test larger-first order with a new kernel
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def row_sum2(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        # Compile with M=2 first, then reuse for M=1
        result2b = row_sum2(x2)
        result1b = row_sum2(x1)
        result3b = row_sum2(x3)

        torch.testing.assert_close(result2b, x2.sum(-1))
        torch.testing.assert_close(result1b, x1.sum(-1))
        torch.testing.assert_close(result3b, x3.sum(-1))

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_reduction_cache_key(self) -> None:
        """Test that reduction kernels share the same bound kernel under zeros mode."""
        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        K = 64

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)

        # Verify specialization keys are the same
        key1 = row_sum.specialization_key((x1,))
        key2 = row_sum.specialization_key((x2,))
        self.assertEqual(key1, key2)

        # Bind smaller shape first to compile the kernel
        bound1 = row_sum.bind((x1,))
        bound2 = row_sum.bind((x2,))

        # Should share the same bound kernel
        self.assertIs(bound1, bound2)

    # =========================================================================
    # Code Generation Tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_codegen_differs_for_zeros_ones(self) -> None:
        """Test that zeros_ones mode produces different code for M=1 vs M=2."""
        def pw_add_fn(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 16

        # Use zeros_ones to force distinct specialization keys per shape
        settings = Settings(static_shapes="zeros_ones", autotune_effort="none")

        k1 = kernel(pw_add_fn, settings=settings)
        k2 = kernel(pw_add_fn, settings=settings)

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        b1 = k1.bind((x1, y1))
        code1 = b1.to_triton_code()

        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        b2 = k2.bind((x2, y2))
        code2 = b2.to_triton_code()

        # With zeros_ones mode, M=1 and M=2 should produce different code
        self.assertNotEqual(code1, code2)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_codegen_identical_m1_vs_m2(self) -> None:
        """Under zeros mode, M=1 vs M=2 should produce identical codegen."""
        def pw_add_fn(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 16
        settings = Settings(static_shapes="zeros", autotune_effort="none")

        k1 = kernel(pw_add_fn, settings=settings)
        k2 = kernel(pw_add_fn, settings=settings)

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        b1 = k1.bind((x1, y1))
        code1 = b1.to_triton_code()

        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        b2 = k2.bind((x2, y2))
        code2 = b2.to_triton_code()

        # Under zeros mode, code should be identical
        self.assertEqual(code1, code2)
        # Only journal once since they're the same
        self.assertExpectedJournal(code1)

    @skipIfRefEager("Code generation comparison not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zeros_code_generation(self) -> None:
        """Test that zeros mode still generates correct code."""

        @kernel(settings=Settings(static_shapes="zeros", autotune_effort="none"))
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([1, 64], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 32])
        torch.testing.assert_close(result, x + 1)
        # Verify dynamic size handling is present in generated code
        self.assertIn("x.size(0), ", code)
        self.assertIn("x_size_0, ", code)

    # =========================================================================
    # Bound Kernel Tests
    # =========================================================================

    @skipIfRefEager("bound kernels not relevant in ref eager mode")
    def test_zeros_ones_different_bound_kernels(self) -> None:
        """Test that zeros_ones mode produces different bound kernels for dim=1 vs dim>=2."""

        @kernel(settings=Settings(static_shapes="zeros_ones", autotune_effort="none"))
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # Create tensors with different first dimensions: 1 vs 2
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        # With zeros_ones mode:
        # - x_dim1 (shape 1,64) should have different bound kernel than x_dim2 (shape 2,64)
        # - x_dim2 and x_dim3 should share the same bound kernel (both >= 2)
        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))
        bound3 = fn.bind((x_dim3,))

        # Different bound kernels for dim=1 vs dim>=2
        self.assertTrueIfInNormalMode(bound1 is not bound2)
        # Same bound kernel for dim=2 vs dim=3 (both bucketed to 2)
        self.assertTrueIfInNormalMode(bound2 is bound3)

        # Verify correctness
        result1 = fn(x_dim1)
        result2 = fn(x_dim2)
        result3 = fn(x_dim3)
        torch.testing.assert_close(result1, x_dim1 * 2)
        torch.testing.assert_close(result2, x_dim2 * 2)
        torch.testing.assert_close(result3, x_dim3 * 2)

    # =========================================================================
    # Backward Compatibility Tests
    # =========================================================================

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat(self) -> None:
        """Test backward compatibility: True maps to 'all', False maps to 'zeros_ones'."""
        # Test True -> 'all' mode (exact sizes are part of the key)
        k_true = kernel(_dummy, settings=Settings(static_shapes=True))

        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        key2_true = k_true.specialization_key([t2])
        key3_true = k_true.specialization_key([t3])

        # With static_shapes='all', exact sizes are part of the key
        self.assertNotEqual(key2_true, key3_true)

        # Test False -> 'zeros_ones' mode
        k_false = kernel(_dummy, settings=Settings(static_shapes=False))

        t1 = torch.empty(1, 3)

        key1_false = k_false.specialization_key([t1])
        key2_false = k_false.specialization_key([t2])
        key3_false = k_false.specialization_key([t3])

        # zeros_ones: 1 is distinct from 2, but 2 and 3 are the same
        self.assertNotEqual(key1_false, key2_false)
        self.assertEqual(key2_false, key3_false)


if __name__ == "__main__":
    unittest.main()
