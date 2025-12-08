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
    # static_shapes="none" tests - same code for size=1 and size>1
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_dim0(self) -> None:
        """none mode, pointwise, 2D: vary dim0 (1->M). Same code for both."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, 16, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used (cache has single entry)
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code (one entry - same code for both sizes)
        self.assertExpectedJournal(bound.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_dim1(self) -> None:
        """none mode, pointwise, 2D: vary dim1 (1->K). Same code for both."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(16, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        self.assertExpectedJournal(bound.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_both_dims(self) -> None:
        """none mode, pointwise, 2D: both dims start at 1. Same code for both."""
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
        self.assertExpectedJournal(bound.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_reduction_2d(self) -> None:
        """none mode, reduction, 2D: vary dim0 (1->M). Same code for both."""
        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1,))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        self.assertExpectedJournal(bound.to_triton_code())

    # =========================================================================
    # static_shapes="ones" tests - different code for size=1 vs size>1
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim0(self) -> None:
        """ones mode, pointwise, 2D: dim0 varies. Different code for size=1 vs >1."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        x1 = torch.randn(1, 16, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        # Compile with size=1 first
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound1 = k.bind((x1, y1))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim1(self) -> None:
        """ones mode, pointwise, 2D: dim1 varies. Different code for size=1 vs >1."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        x1 = torch.randn(16, 1, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        # Compile with size=1 first
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound1 = k.bind((x1, y1))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_reduction_2d(self) -> None:
        """ones mode, reduction, 2D: dim0 varies. Different code for size=1 vs >1."""
        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)

        # Compile with size=1 first
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound1 = k.bind((x1,))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2,))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        self.assertExpectedJournal(bound2.to_triton_code())

    # =========================================================================
    # static_shapes="all" tests - unique code for each exact size
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_pointwise_2d(self) -> None:
        """all mode, pointwise, 2D: unique code for each exact size."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        x1 = torch.randn(1, 16, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        # Compile with size=1 first
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 code
        bound1 = k.bind((x1, y1))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size=4 code
        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_reduction_2d(self) -> None:
        """all mode, reduction, 2D: unique code for each exact size."""
        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)

        # Compile with size=1 first
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Journal the size=1 code
        bound1 = k.bind((x1,))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2,))
        self.assertIsNot(bound1, bound2)

        # Journal the size=4 code
        self.assertExpectedJournal(bound2.to_triton_code())

    # =========================================================================
    # Backward compatibility tests (True/False -> all/ones)
    # =========================================================================

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat_bool_to_string_modes(self) -> None:
        """Test backward compatibility: True maps to 'all', False maps to 'ones'."""
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        def dummy(x: torch.Tensor) -> torch.Tensor:
            return x

        # True -> 'all' mode: verify normalization and behavior
        settings_all = Settings(static_shapes=True)
        self.assertEqual(settings_all.static_shapes, "all")  # normalized to string
        k_all = kernel(dummy, settings=settings_all)
        key_all_2 = k_all.specialization_key([t2])
        key_all_3 = k_all.specialization_key([t3])
        self.assertNotEqual(key_all_2, key_all_3)  # each exact size is distinct

        # False -> 'ones' mode: verify normalization and behavior
        settings_ones = Settings(static_shapes=False)
        self.assertEqual(settings_ones.static_shapes, "ones")  # normalized to string
        k_ones = kernel(dummy, settings=settings_ones)
        key_ones_1 = k_ones.specialization_key([t1])
        key_ones_2 = k_ones.specialization_key([t2])
        key_ones_3 = k_ones.specialization_key([t3])
        self.assertNotEqual(key_ones_1, key_ones_2)  # 1 is distinct from >=2
        self.assertEqual(key_ones_2, key_ones_3)  # but 2 and 3 are same


if __name__ == "__main__":
    unittest.main()
