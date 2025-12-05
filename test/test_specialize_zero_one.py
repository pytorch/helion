from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfCpu
from helion._testing import skipIfRefEager
import helion.language as hl


@skipIfCpu("needs to be debugged")
class TestSpecializeZeroOne(RefEagerTestBase, TestCase):
    """Tests for the specialize_zero_one setting that controls 0/1 specialization behavior."""

    maxDiff = 16384

    def test_default_specialize_zero_one_enabled(self):
        """Test that by default, specialize_zero_one=True and different bound kernels
        are created for tensors with dimension 1 vs dimension 2."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # Create tensors with different first dimensions: 1 vs 2
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        # With default specialize_zero_one=True:
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

    def test_specialize_zero_one_disabled_cache_key(self):
        """Test that with specialize_zero_one=False, tensors with dimension 1 and
        dimension 2 have the same specialization key and share the same bound kernel."""

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # Create tensors with different first dimensions: 1 vs 2 vs 3
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        # Verify specialization keys are the same when specialize_zero_one=False
        key1 = fn.specialization_key((x_dim1,))
        key2 = fn.specialization_key((x_dim2,))
        key3 = fn.specialization_key((x_dim3,))

        self.assertTrueIfInNormalMode(key1 == key2)
        self.assertTrueIfInNormalMode(key2 == key3)

        # With specialize_zero_one=False:
        # All shapes should share the same bound kernel
        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))
        bound3 = fn.bind((x_dim3,))

        # Same bound kernel for all dimensions when specialize_zero_one=False
        self.assertTrueIfInNormalMode(bound1 is bound2)
        self.assertTrueIfInNormalMode(bound2 is bound3)

        # Verify each shape produces correct results independently
        # (compile with x_dim1 shape, run on same shape)
        result1 = fn(x_dim1)
        torch.testing.assert_close(result1, x_dim1 * 2)

    def test_specialize_zero_one_disabled_smaller_first(self):
        """Test specialize_zero_one=False when binding with smaller shape (size==1) first.
        This is the critical test: kernel compiled for size==1 must work on size==2.
        """

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # Create tensors - smaller shape first
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        # Bind smaller shape first to compile the kernel
        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))
        bound3 = fn.bind((x_dim3,))

        # All should share the same bound kernel
        self.assertTrueIfInNormalMode(bound1 is bound2)
        self.assertTrueIfInNormalMode(bound2 is bound3)

        # Verify correctness - run with size==1 first, then size==2 and size==3
        # This tests that the kernel compiled for size==1 works on larger sizes
        result1 = fn(x_dim1)
        result2 = fn(x_dim2)
        result3 = fn(x_dim3)
        torch.testing.assert_close(result1, x_dim1 * 2)
        torch.testing.assert_close(result2, x_dim2 * 2)
        torch.testing.assert_close(result3, x_dim3 * 2)

    def test_specialize_zero_one_disabled_larger_first(self):
        """Test specialize_zero_one=False when binding with larger shape first.
        This ensures the kernel compiled for larger shapes works for smaller ones."""

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # Create tensors - bind with larger shape first
        x_dim3 = torch.randn([3, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim1 = torch.randn([1, 64], device=DEVICE)

        # Bind larger shape first to compile the kernel
        bound3 = fn.bind((x_dim3,))
        bound2 = fn.bind((x_dim2,))
        bound1 = fn.bind((x_dim1,))

        # All should share the same bound kernel
        self.assertTrueIfInNormalMode(bound1 is bound2)
        self.assertTrueIfInNormalMode(bound2 is bound3)

        # Verify correctness - all shapes should work
        result3 = fn(x_dim3)
        result2 = fn(x_dim2)
        result1 = fn(x_dim1)
        torch.testing.assert_close(result3, x_dim3 * 2)
        torch.testing.assert_close(result2, x_dim2 * 2)
        torch.testing.assert_close(result1, x_dim1 * 2)

    def test_specialize_zero_one_disabled_with_zero_dim(self):
        """Test that with specialize_zero_one=False, tensors with dimension 0, 1, and 2
        all share the same specialization key."""

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # Create tensors with different first dimensions: 0 vs 1 vs 2
        x_dim0 = torch.randn([0, 64], device=DEVICE)
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)

        # Verify specialization keys are the same when specialize_zero_one=False
        key0 = fn.specialization_key((x_dim0,))
        key1 = fn.specialization_key((x_dim1,))
        key2 = fn.specialization_key((x_dim2,))

        self.assertTrueIfInNormalMode(key0 == key1)
        self.assertTrueIfInNormalMode(key1 == key2)

        # With specialize_zero_one=False:
        # All shapes should share the same bound kernel
        bound0 = fn.bind((x_dim0,))
        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))

        # Same bound kernel for all dimensions when specialize_zero_one=False
        self.assertTrueIfInNormalMode(bound0 is bound1)
        self.assertTrueIfInNormalMode(bound1 is bound2)

        # Verify correctness for empty tensor
        result0 = fn(x_dim0)
        torch.testing.assert_close(result0, x_dim0 * 2)

    @skipIfRefEager("Code generation comparison not relevant in ref eager mode")
    def test_specialize_zero_one_code_generation(self):
        """Test that specialize_zero_one=False still generates correct code."""

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([1, 64], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 32])
        torch.testing.assert_close(result, x + 1)
        self.assertIn("x.size(0), ", code)
        self.assertIn("x_size_0, ", code)
        self.assertExpectedJournal(code)

    def test_specialize_zero_one_with_reduction_cache_key(self):
        """Test specialize_zero_one=False cache key behavior with a reduction operation."""

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        # Create tensors with different first dimensions
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)

        # Verify specialization keys are the same
        key1 = fn.specialization_key((x_dim1,))
        key2 = fn.specialization_key((x_dim2,))
        self.assertTrueIfInNormalMode(key1 == key2)

        # With specialize_zero_one=False, these should share the same bound kernel
        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))

        self.assertTrueIfInNormalMode(bound1 is bound2)

        # Verify correctness for each shape independently
        fn.reset()  # Clear cache to test each shape separately
        result1 = fn(x_dim1)
        torch.testing.assert_close(result1, x_dim1.sum(-1))

        fn.reset()
        result2 = fn(x_dim2)
        torch.testing.assert_close(result2, x_dim2.sum(-1))

    def test_specialize_zero_one_reduction_smaller_first(self):
        """Test reduction kernel with specialize_zero_one=False, compiling with size==1 first.
        This is the critical test for reductions: kernel compiled for size==1 must work on size==2.
        """

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        # Create tensors - smaller shape first
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        # Bind smaller shape first to compile the kernel
        bound1 = fn.bind((x_dim1,))
        bound2 = fn.bind((x_dim2,))
        bound3 = fn.bind((x_dim3,))

        # All should share the same bound kernel
        self.assertTrueIfInNormalMode(bound1 is bound2)
        self.assertTrueIfInNormalMode(bound2 is bound3)

        # Run with size==1 first, then size==2 and size==3 using the same cached kernel
        # This tests that the reduction kernel compiled for size==1 works on larger sizes
        result1 = fn(x_dim1)
        result2 = fn(x_dim2)
        result3 = fn(x_dim3)
        torch.testing.assert_close(result1, x_dim1.sum(-1))
        torch.testing.assert_close(result2, x_dim2.sum(-1))
        torch.testing.assert_close(result3, x_dim3.sum(-1))

    def test_specialize_zero_one_multidim_cache_key(self):
        """Test specialize_zero_one=False cache key behavior with multiple dimensions."""

        @helion.kernel(static_shapes=False, specialize_zero_one=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 3
            return out

        # Create tensors with different shapes: (1, 1) vs (2, 2) vs (1, 2)
        x_11 = torch.randn([1, 1], device=DEVICE)
        x_22 = torch.randn([2, 2], device=DEVICE)
        x_12 = torch.randn([1, 2], device=DEVICE)

        # Verify specialization keys are all the same
        key_11 = fn.specialization_key((x_11,))
        key_22 = fn.specialization_key((x_22,))
        key_12 = fn.specialization_key((x_12,))

        self.assertTrueIfInNormalMode(key_11 == key_22)
        self.assertTrueIfInNormalMode(key_22 == key_12)

        # With specialize_zero_one=False, all should share the same bound kernel
        bound_11 = fn.bind((x_11,))
        bound_22 = fn.bind((x_22,))
        bound_12 = fn.bind((x_12,))

        self.assertTrueIfInNormalMode(bound_11 is bound_22)
        self.assertTrueIfInNormalMode(bound_22 is bound_12)

        # Verify correctness when binding with larger shape first
        fn.reset()  # Clear cache
        # Bind larger shape first, then smaller shapes
        _ = fn.bind((x_22,))
        result_22 = fn(x_22)
        result_11 = fn(x_11)
        result_12 = fn(x_12)
        torch.testing.assert_close(result_22, x_22 * 3)
        torch.testing.assert_close(result_11, x_11 * 3)
        torch.testing.assert_close(result_12, x_12 * 3)

    def test_specialize_zero_one_enabled_multidim(self):
        """Test that specialize_zero_one=True (default) creates different kernels for different 0/1 patterns."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 3
            return out

        # Create tensors with different shapes
        x_11 = torch.randn([1, 1], device=DEVICE)
        x_22 = torch.randn([2, 2], device=DEVICE)
        x_12 = torch.randn([1, 2], device=DEVICE)

        # With default specialize_zero_one=True:
        # - (1, 1) -> bucketed to (1, 1)
        # - (2, 2) -> bucketed to (2, 2)
        # - (1, 2) -> bucketed to (1, 2)
        # All three should have different bound kernels
        bound_11 = fn.bind((x_11,))
        bound_22 = fn.bind((x_22,))
        bound_12 = fn.bind((x_12,))

        self.assertTrueIfInNormalMode(bound_11 is not bound_22)
        self.assertTrueIfInNormalMode(bound_22 is not bound_12)
        self.assertTrueIfInNormalMode(bound_11 is not bound_12)

        # Verify correctness
        result_11 = fn(x_11)
        result_22 = fn(x_22)
        result_12 = fn(x_12)
        torch.testing.assert_close(result_11, x_11 * 3)
        torch.testing.assert_close(result_22, x_22 * 3)
        torch.testing.assert_close(result_12, x_12 * 3)

    def test_specialize_zero_one_keys_differ_when_enabled(self):
        """Test that specialization keys differ for 0, 1, 2 dimensions when specialize_zero_one=True."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        x_dim0 = torch.randn([0, 64], device=DEVICE)
        x_dim1 = torch.randn([1, 64], device=DEVICE)
        x_dim2 = torch.randn([2, 64], device=DEVICE)
        x_dim3 = torch.randn([3, 64], device=DEVICE)

        key0 = fn.specialization_key((x_dim0,))
        key1 = fn.specialization_key((x_dim1,))
        key2 = fn.specialization_key((x_dim2,))
        key3 = fn.specialization_key((x_dim3,))

        # With specialize_zero_one=True:
        # - 0 is bucketed to 0
        # - 1 is bucketed to 1
        # - 2, 3, etc. are all bucketed to 2
        self.assertTrueIfInNormalMode(key0 != key1)
        self.assertTrueIfInNormalMode(key1 != key2)
        self.assertTrueIfInNormalMode(key2 == key3)  # Both bucketed to 2

        # Verify correctness
        torch.testing.assert_close(fn(x_dim1), x_dim1 * 2)
        torch.testing.assert_close(fn(x_dim2), x_dim2 * 2)
        torch.testing.assert_close(fn(x_dim3), x_dim3 * 2)


if __name__ == "__main__":
    unittest.main()
