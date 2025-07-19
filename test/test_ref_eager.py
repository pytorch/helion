from __future__ import annotations

import contextlib
import io
import math
import unittest
from unittest.mock import patch

import pytest
import torch

from . import test_examples
from .ref_utils import clear_kernel_caches_and_set_ref_mode
import helion
from helion._testing import TestCase
import helion.language as hl


class TestExamplesRefEager(test_examples.TestExamples):
    """Run all TestExamples tests in reference eager mode."""

    # NOTE: All tests in TestExamples are run in ref eager mode by default in this test file.

    def assertExpectedJournal(self, value: str) -> None:
        """Skip journal assertions in ref mode since we don't generate Triton code."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Clear kernel caches and set ref mode to EAGER
        clear_kernel_caches_and_set_ref_mode(helion.RefMode.EAGER)

    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Clear kernel caches and reset to OFF mode
        clear_kernel_caches_and_set_ref_mode(helion.RefMode.OFF)

    def test_add(self):
        # Mock the critical functions that should NOT be called in ref eager mode
        # These functions are only used in normal Helion mode to generate and compile Triton code
        with (
            patch("helion.runtime.kernel.BoundKernel.to_triton_code") as mock_to_triton,
            patch("helion.runtime.kernel.BoundKernel.compile_config") as mock_compile,
        ):
            # Set up the mocks to fail if called
            mock_to_triton.side_effect = AssertionError(
                "to_triton_code should NOT be called in reference eager mode!"
            )
            mock_compile.side_effect = AssertionError(
                "compile_config should NOT be called in reference eager mode!"
            )

            # Run the original test
            super().test_add()

            # Assert that neither function was called
            mock_to_triton.assert_not_called()
            mock_compile.assert_not_called()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_concat(self):
        super().test_concat()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_concat_block_ptr(self):
        super().test_concat_block_ptr()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_cross_entropy(self):
        super().test_cross_entropy()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_jagged_dense_add(self):
        super().test_jagged_dense_add()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_jagged_mean(self):
        super().test_jagged_mean()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_matmul_split_k(self):
        super().test_matmul_split_k()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_moe_matmul_ogs(self):
        super().test_moe_matmul_ogs()

    @pytest.mark.skip(reason="tile.* API is not supported yet")
    def test_segment_reduction(self):
        super().test_segment_reduction()


class TestRefEagerMisc(TestCase):
    def test_print_intermediate_tensor(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def print_intermediate_tensor_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                x_val = x[tile_m, tile_n]
                y_val = y[tile_m, tile_n]
                sum_val = x_val + y_val
                print("x: ", x_val)
                print("y: ", y_val)
                print("sum: ", sum_val)
                out[tile_m, tile_n] = sum_val
            return out

        x = torch.ones([2, 2], device="cuda", dtype=torch.float32) * 10.0
        y = torch.ones([2, 2], device="cuda", dtype=torch.float32) * 5.0
        expected = x + y

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            result = print_intermediate_tensor_kernel(x, y)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

        # Check that the print statements produced output
        output = captured_output.getvalue()
        self.assertIn("x: ", output)
        self.assertIn("y: ", output)
        self.assertIn("sum: ", output)
        self.assertIn("[[10., 10.]", output)  # x values
        self.assertIn("[[5., 5.]", output)  # y values
        self.assertIn("[[15., 15.]", output)  # sum values

    def test_print_in_invalid_helion_kernel(self):
        """Test that print works even in invalid Helion kernels in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def incorrect_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("processing tile: ", val)
                # `pass` below causes this kernel to be invalid.
                # But we show that in ref-eager mode, the `print` statement above still works,
                # which is useful for debugging.
                pass  # noqa: PIE790
            return x

        x = torch.ones([2, 2], device="cuda", dtype=torch.float32) * math.pi

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            _ = incorrect_kernel(x)

        # Check that the print statement produced output
        output = captured_output.getvalue()
        self.assertIn("processing tile: ", output)
        self.assertIn("[[3.14", output)  # The value printed

    def test_ref_eager_kernel_config(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            return x + x * 2.0

        x = torch.randn(128, device="cuda")
        result = kernel(x)
        expected = x + x * 2.0
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
