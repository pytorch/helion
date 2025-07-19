from __future__ import annotations

import contextlib
import io
import math
import os
import unittest

# Import test utilities for the print tests
import torch

# Import the module to get test methods without importing the class
from . import test_examples
import helion
from helion._testing import TestCase
import helion.language as hl


class TestExamplesRefEager(test_examples.TestExamples):
    """Run all TestExamples tests in reference eager mode via HELION_REF_EAGER=1."""

    # NOTE: All tests in TestExamples are run in ref eager mode by default in this test file.
    # Below, we override specific tests with smaller input sizes to make ref eager mode tests run faster.

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var
        self._original_env_var_value = os.environ.get("HELION_REF_EAGER")
        # Set ref eager mode
        os.environ["HELION_REF_EAGER"] = "1"

    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_env_var_value is not None:
            os.environ["HELION_REF_EAGER"] = self._original_env_var_value
        elif "HELION_REF_EAGER" in os.environ:
            del os.environ["HELION_REF_EAGER"]

    def test_add(self):
        """Override test_add to verify ref eager mode execution."""
        from torch._dynamo.utils import counters

        # Clear counters before running the test
        counters.clear()

        # Run the original test
        super().test_add()

        # In ref eager mode, torch.compile should NOT be called
        # So there should be no torch.compile-related counters
        self.assertEqual(
            counters.get("frames", {}).get("total", 0),
            0,
            "torch.compile should not be invoked in ref eager mode",
        )
        self.assertEqual(
            counters.get("aot_autograd", {}).get("total", 0),
            0,
            "AOT autograd should not be invoked in ref eager mode",
        )

    def test_print_intermediate_tensor(self):
        @helion.kernel
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
        """Test that print works even in invalid Helion kernels in reference eager mode."""

        @helion.kernel
        def incorrect_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("processing tile: ", val)
                # `pass` below causes this kernel to be invalid.
                # But we show that in ref-eager mode, the `print` statement above still works,
                # which is useful for debugging.
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


class TestRefEagerKernelConfig(TestCase):
    """Test @helion.kernel(ref_eager=True) parameter functionality."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var to ensure clean state
        self._original_env_var_value = os.environ.get("HELION_REF_EAGER")
        # Remove env var to test parameter-only behavior
        if "HELION_REF_EAGER" in os.environ:
            del os.environ["HELION_REF_EAGER"]

    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_env_var_value is not None:
            os.environ["HELION_REF_EAGER"] = self._original_env_var_value

    def test_ref_eager_kernel_config(self):
        @helion.kernel(ref_eager=True)
        def compile_test(x: torch.Tensor) -> torch.Tensor:
            return x + x * 2.0

        x = torch.randn(128, device="cuda")
        result = compile_test(x)
        expected = x + x * 2.0
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
