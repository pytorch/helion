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
    def test_specialization_keys_by_mode(self) -> None:
        """Test specialization key behavior for none, ones, and all modes."""
        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)
        t7 = torch.empty(7, 3)

        # Test none mode: 0 is distinct, 1 and >=2 are unified
        with self.subTest(mode="none"):
            k = kernel(_dummy, settings=Settings(static_shapes="none"))
            key_0 = k.specialization_key([t0])
            key_1 = k.specialization_key([t1])
            key_2 = k.specialization_key([t2])
            key_7 = k.specialization_key([t7])

            self.assertNotEqual(key_0, key_1)
            self.assertNotEqual(key_0, key_2)
            self.assertEqual(key_1, key_2)
            self.assertEqual(key_2, key_7)

        # Test ones mode: 0, 1, and >=2 are all distinct buckets
        with self.subTest(mode="ones"):
            k = kernel(_dummy, settings=Settings(static_shapes="ones"))
            key_0 = k.specialization_key([t0])
            key_1 = k.specialization_key([t1])
            key_2 = k.specialization_key([t2])
            key_3 = k.specialization_key([t3])
            key_7 = k.specialization_key([t7])

            self.assertNotEqual(key_0, key_1)
            self.assertNotEqual(key_0, key_2)
            self.assertNotEqual(key_1, key_2)
            self.assertEqual(key_2, key_3)
            self.assertEqual(key_2, key_7)

        # Test all mode (static_shapes=True): each size is distinct
        with self.subTest(mode="all"):
            k = kernel(_dummy, settings=Settings(static_shapes="all"))
            key_2 = k.specialization_key([t2])
            key_3 = k.specialization_key([t3])
            self.assertNotEqual(key_2, key_3)

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_multidim_specialization_keys(self) -> None:
        """Test specialization key behavior with multiple dimensions for none and ones modes."""
        t_11 = torch.empty(1, 1)
        t_22 = torch.empty(2, 2)
        t_12 = torch.empty(1, 2)

        # In none mode, 1 and 2 are unified, so all keys should be the same
        with self.subTest(mode="none"):
            k = kernel(_dummy, settings=Settings(static_shapes="none"))
            key_11 = k.specialization_key([t_11])
            key_22 = k.specialization_key([t_22])
            key_12 = k.specialization_key([t_12])

            self.assertEqual(key_11, key_22)
            self.assertEqual(key_22, key_12)

        # With ones mode, different 0/1 patterns produce different keys
        with self.subTest(mode="ones"):
            k = kernel(_dummy, settings=Settings(static_shapes="ones"))
            key_11 = k.specialization_key([t_11])
            key_22 = k.specialization_key([t_22])
            key_12 = k.specialization_key([t_12])

            self.assertNotEqual(key_11, key_22)
            self.assertNotEqual(key_22, key_12)
            self.assertNotEqual(key_11, key_12)

    # =========================================================================
    # Runtime Correctness and Cache Reuse Tests
    # =========================================================================

    @skipIfNotCUDA()
    def test_none_runtime_correctness_and_cache(self) -> None:
        """Test none mode runtime correctness and cache reuse with various compile orders."""
        K = 16

        # Test compiling with M=1 first (the harder case per jansel's review)
        @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        x3 = torch.randn(3, K, device=DEVICE, dtype=torch.float32)

        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)
        y3 = torch.empty_like(x3)

        # Bind to capture the bound kernel instance for cache inspection
        b = pw_add.bind((x1, y1))

        # Compile with M=1 first, then reuse for M=2 and M=3
        pw_add(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        pw_add(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        pw_add(x3, y3)
        torch.testing.assert_close(y3, x3 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify cache reuse (should be single entry for all sizes)
        self.assertTrueIfInNormalMode(len(b._compile_cache) == 1)

        # Test larger-first order with a new kernel
        @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
        def pw_add2(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 2.0

        # Compile with M=2 first, then reuse for M=1
        pw_add2(x2, y2)
        pw_add2(x1, y1)

        torch.testing.assert_close(y2, x2 + 2.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y1, x1 + 2.0, rtol=1e-4, atol=1e-4)

    @skipIfNotCUDA()
    def test_none_varying_singleton_dims(self) -> None:
        """Test none mode with varying singleton dimensions in 2D and 3D tensors."""
        @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
        def pw_add(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 8

        # Test 2D: row (1, K) vs column (K, 1)
        with self.subTest(dims="2D"):
            x_row = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
            y_row = torch.empty_like(x_row)
            x_col = torch.randn(K, 1, device=DEVICE, dtype=torch.float32)
            y_col = torch.empty_like(x_col)

            b = pw_add.bind((x_row, y_row))

            pw_add(x_row, y_row)
            torch.testing.assert_close(y_row, x_row + 1.0, rtol=1e-4, atol=1e-4)
            self.assertTrueIfInNormalMode(len(b._compile_cache) == 1)

            pw_add(x_col, y_col)
            torch.testing.assert_close(y_col, x_col + 1.0, rtol=1e-4, atol=1e-4)
            self.assertTrueIfInNormalMode(len(b._compile_cache) == 1)

        # Test 3D with various singleton patterns
        with self.subTest(dims="3D"):
            @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
            def pw_add3d(x: torch.Tensor, out: torch.Tensor) -> None:
                for tile in hl.tile(x.size()):
                    out[tile] = x[tile] + 1.0

            x100 = torch.randn(1, K, K, device=DEVICE, dtype=torch.float32)
            y100 = torch.empty_like(x100)

            b = pw_add3d.bind((x100, y100))
            pw_add3d(x100, y100)
            torch.testing.assert_close(y100, x100 + 1.0, rtol=1e-4, atol=1e-4)
            self.assertTrueIfInNormalMode(len(b._compile_cache) == 1)

            # Test various 3D patterns, all should reuse the same compiled kernel
            for shape in [(K, 1, K), (K, K, 1), (1, 1, K), (1, K, 1), (K, 1, 1)]:
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                y = torch.empty_like(x)
                pw_add3d(x, y)
                torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)
                self.assertTrueIfInNormalMode(len(b._compile_cache) == 1)

    @skipIfRefEager("bound kernels not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_multidim_runtime_and_bound_kernel(self) -> None:
        """Test none mode runtime correctness and bound kernel sharing with multiple dimensions."""

        @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 3
            return out

        x_11 = torch.randn([1, 1], device=DEVICE)
        x_22 = torch.randn([2, 2], device=DEVICE)
        x_12 = torch.randn([1, 2], device=DEVICE)

        # With none mode, all should share the same bound kernel
        bound_11 = fn.bind((x_11,))
        bound_22 = fn.bind((x_22,))
        bound_12 = fn.bind((x_12,))

        self.assertTrueIfInNormalMode(bound_11 is bound_22)
        self.assertTrueIfInNormalMode(bound_22 is bound_12)

        # Verify correctness for all shapes
        torch.testing.assert_close(fn(x_11), x_11 * 3)
        torch.testing.assert_close(fn(x_22), x_22 * 3)
        torch.testing.assert_close(fn(x_12), x_12 * 3)

    # =========================================================================
    # Reduction Kernel Tests
    # =========================================================================

    @skipIfNotCUDA()
    def test_none_reduction(self) -> None:
        """Test reduction kernel with none mode, testing both compile orders and cache key sharing."""
        K = 64

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        x3 = torch.randn(3, K, device=DEVICE, dtype=torch.float32)

        # Test M=1 first (per jansel's review)
        with self.subTest(order="small_first"):
            @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
            def row_sum(x: torch.Tensor) -> torch.Tensor:
                out = x.new_empty([x.size(0)])
                for tile in hl.tile(x.size(0)):
                    out[tile] = x[tile, :].sum(-1)
                return out

            torch.testing.assert_close(row_sum(x1), x1.sum(-1))
            torch.testing.assert_close(row_sum(x2), x2.sum(-1))
            torch.testing.assert_close(row_sum(x3), x3.sum(-1))

        # Test larger-first order
        with self.subTest(order="large_first"):
            @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
            def row_sum2(x: torch.Tensor) -> torch.Tensor:
                out = x.new_empty([x.size(0)])
                for tile in hl.tile(x.size(0)):
                    out[tile] = x[tile, :].sum(-1)
                return out

            torch.testing.assert_close(row_sum2(x2), x2.sum(-1))
            torch.testing.assert_close(row_sum2(x1), x1.sum(-1))
            torch.testing.assert_close(row_sum2(x3), x3.sum(-1))

        # Verify specialization keys and bound kernel sharing
        with self.subTest(check="cache_key"):
            @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
            def row_sum3(x: torch.Tensor) -> torch.Tensor:
                out = x.new_empty([x.size(0)])
                for tile in hl.tile(x.size(0)):
                    out[tile] = x[tile, :].sum(-1)
                return out

            key1 = row_sum3.specialization_key((x1,))
            key2 = row_sum3.specialization_key((x2,))
            self.assertTrueIfInNormalMode(key1 == key2)

            bound1 = row_sum3.bind((x1,))
            bound2 = row_sum3.bind((x2,))
            self.assertTrueIfInNormalMode(bound1 is bound2)

    # =========================================================================
    # Code Generation Tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_codegen_by_mode(self) -> None:
        """Test code generation differences between ones and none modes for M=1 vs M=2."""
        def pw_add_fn(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        K = 16

        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        x2 = torch.randn(2, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)

        # Test ones mode: M=1 and M=2 should produce different code
        with self.subTest(mode="ones"):
            settings = Settings(static_shapes="ones", autotune_effort="none")
            k1 = kernel(pw_add_fn, settings=settings)
            k2 = kernel(pw_add_fn, settings=settings)

            b1 = k1.bind((x1, y1))
            code1 = b1.to_triton_code()

            b2 = k2.bind((x2, y2))
            code2 = b2.to_triton_code()

            self.assertNotEqual(code1, code2)

        # Test none mode: M=1 and M=2 should produce identical code
        with self.subTest(mode="none"):
            settings = Settings(static_shapes="none", autotune_effort="none")
            k1 = kernel(pw_add_fn, settings=settings)
            k2 = kernel(pw_add_fn, settings=settings)

            b1 = k1.bind((x1, y1))
            code1 = b1.to_triton_code()

            b2 = k2.bind((x2, y2))
            code2 = b2.to_triton_code()

            self.assertEqual(code1, code2)
            self.assertExpectedJournal(code1)

    @skipIfRefEager("Code generation comparison not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_code_generation(self) -> None:
        """Test that none mode generates correct code with dynamic size handling."""

        @kernel(settings=Settings(static_shapes="none", autotune_effort="none"))
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
    def test_ones_different_bound_kernels(self) -> None:
        """Test that ones mode produces different bound kernels for dim=1 vs dim>=2."""

        @kernel(settings=Settings(static_shapes="ones", autotune_effort="none"))
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))
        bound3 = fn.bind((x_dim3,))

        # Different bound kernels for dim=1 vs dim>=2
        self.assertTrueIfInNormalMode(bound1 is not bound2)
        # Same bound kernel for dim=2 vs dim=3 (both bucketed to 2)
        self.assertTrueIfInNormalMode(bound2 is bound3)

        # Verify correctness
        torch.testing.assert_close(fn(x_dim1), x_dim1 * 2)
        torch.testing.assert_close(fn(x_dim2), x_dim2 * 2)
        torch.testing.assert_close(fn(x_dim3), x_dim3 * 2)

    # =========================================================================
    # Backward Compatibility Tests
    # =========================================================================

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat(self) -> None:
        """Test backward compatibility: True maps to 'all', False maps to 'ones'."""
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        # Test True -> 'all' mode (exact sizes are part of the key)
        k_true = kernel(_dummy, settings=Settings(static_shapes=True))
        key2_true = k_true.specialization_key([t2])
        key3_true = k_true.specialization_key([t3])
        self.assertNotEqual(key2_true, key3_true)

        # Test False -> 'ones' mode
        k_false = kernel(_dummy, settings=Settings(static_shapes=False))
        key1_false = k_false.specialization_key([t1])
        key2_false = k_false.specialization_key([t2])
        key3_false = k_false.specialization_key([t3])

        # ones: 1 is distinct from 2, but 2 and 3 are the same
        self.assertNotEqual(key1_false, key2_false)
        self.assertEqual(key2_false, key3_false)


if __name__ == "__main__":
    unittest.main()
