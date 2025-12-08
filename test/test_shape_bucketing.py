from __future__ import annotations

import unittest

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import skipIfCpu
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


# =============================================================================
# Kernel definitions
# =============================================================================


def pointwise_add_kernel(x: torch.Tensor, out: torch.Tensor) -> None:
    """Simple pointwise kernel: out = x + 1.0"""
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1.0


def reduction_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Reduction kernel: sum along last dimension."""
    out = x.new_empty([x.size(0)])
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


# =============================================================================
# Test class
# =============================================================================


@skipIfCpu("needs to be debugged")
class TestShapeBucketing(RefEagerTestBase, TestCase):
    maxDiff = 16384

    # =========================================================================
    # static_shapes="none" + pointwise kernel tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_dim0(self) -> None:
        """none mode, pointwise, 2D: vary dim0 (1->M)."""
        K = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used (cache has single entry)
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_dim1(self) -> None:
        """none mode, pointwise, 2D: vary dim1 (1->K)."""
        M = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(M, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(M, 8, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_both_dims(self) -> None:
        """none mode, pointwise, 2D: both dims start at 1."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with (1, 1) first
        x1 = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with larger sizes
        x2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    # =========================================================================
    # static_shapes="none" + reduction kernel tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_reduction_2d_vary_dim0(self) -> None:
        """none mode, reduction, 2D: vary dim0 (1->M)."""
        K = 64

        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, K, device=DEVICE, dtype=torch.float32)
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1,))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    # =========================================================================
    # static_shapes="ones" + pointwise kernel tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim0_eq_1(self) -> None:
        """ones mode, pointwise, 2D: dim0=1 (specialized code)."""
        K = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        # Compile and run with size=1
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound = k.bind((x1, y1))
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim0_gt_1(self) -> None:
        """ones mode, pointwise, 2D: dim0>1 (general code)."""
        K = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        # Compile with size=1 first (as required by test pattern)
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used (cache has 2 entries)
        bound1 = k.bind((x1, y1))
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        code = bound2.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim1_eq_1(self) -> None:
        """ones mode, pointwise, 2D: dim1=1 (specialized code)."""
        M = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        # Compile and run with size=1
        x1 = torch.randn(M, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound = k.bind((x1, y1))
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim1_gt_1(self) -> None:
        """ones mode, pointwise, 2D: dim1>1 (general code)."""
        M = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(M, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(M, 8, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound1 = k.bind((x1, y1))
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        code = bound2.to_triton_code()
        self.assertExpectedJournal(code)

    # =========================================================================
    # static_shapes="ones" + reduction kernel tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_reduction_2d_dim0_eq_1(self) -> None:
        """ones mode, reduction, 2D: dim0=1 (specialized code)."""
        K = 64

        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        # Compile and run with size=1
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound = k.bind((x1,))
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_reduction_2d_dim0_gt_1(self) -> None:
        """ones mode, reduction, 2D: dim0>1 (general code)."""
        K = 64

        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, K, device=DEVICE, dtype=torch.float32)
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound1 = k.bind((x1,))
        bound2 = k.bind((x2,))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        code = bound2.to_triton_code()
        self.assertExpectedJournal(code)

    # =========================================================================
    # static_shapes="all" + pointwise kernel tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_pointwise_2d_dim0_eq_1(self) -> None:
        """all mode, pointwise, 2D: dim0=1 (exact size specialized)."""
        K = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        # Compile and run with size=1
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the exact-size specialized code
        bound = k.bind((x1, y1))
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_pointwise_2d_dim0_gt_1(self) -> None:
        """all mode, pointwise, 2D: dim0>1 (each size gets unique code)."""
        K = 16

        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, K, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used (each exact size is distinct)
        bound1 = k.bind((x1, y1))
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size=4 code
        code = bound2.to_triton_code()
        self.assertExpectedJournal(code)

    # =========================================================================
    # static_shapes="all" + reduction kernel tests
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_reduction_2d_dim0_eq_1(self) -> None:
        """all mode, reduction, 2D: dim0=1 (exact size specialized)."""
        K = 64

        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        # Compile and run with size=1
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Journal the exact-size specialized code
        bound = k.bind((x1,))
        code = bound.to_triton_code()
        self.assertExpectedJournal(code)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_reduction_2d_dim0_gt_1(self) -> None:
        """all mode, reduction, 2D: dim0>1 (each size gets unique code)."""
        K = 64

        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, K, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, K, device=DEVICE, dtype=torch.float32)
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound1 = k.bind((x1,))
        bound2 = k.bind((x2,))
        self.assertIsNot(bound1, bound2)

        # Journal the size=4 code
        code = bound2.to_triton_code()
        self.assertExpectedJournal(code)

    # =========================================================================
    # Backward compatibility tests (True/False -> all/ones)
    # =========================================================================

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat_true_maps_to_all(self) -> None:
        """Test backward compatibility: True maps to 'all' mode."""
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        def dummy(x: torch.Tensor) -> torch.Tensor:
            return x

        k = kernel(dummy, settings=Settings(static_shapes=True))
        key2 = k.specialization_key([t2])
        key3 = k.specialization_key([t3])

        # In 'all' mode, each exact size is distinct
        self.assertNotEqual(key2, key3)

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat_false_maps_to_ones(self) -> None:
        """Test backward compatibility: False maps to 'ones' mode."""
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        def dummy(x: torch.Tensor) -> torch.Tensor:
            return x

        k = kernel(dummy, settings=Settings(static_shapes=False))
        key1 = k.specialization_key([t1])
        key2 = k.specialization_key([t2])
        key3 = k.specialization_key([t3])

        # In 'ones' mode: 1 is distinct from >=2, but 2 and 3 are same
        self.assertNotEqual(key1, key2)
        self.assertEqual(key2, key3)


if __name__ == "__main__":
    unittest.main()
