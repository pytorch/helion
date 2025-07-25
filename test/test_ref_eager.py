from __future__ import annotations

import contextlib
import io
import math
import unittest

import pytest
import torch

from . import test_examples
import helion
from helion._testing import TestCase
import helion.language as hl


from pathlib import Path
from typing import TYPE_CHECKING

import helion
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path

if TYPE_CHECKING:
    from helion.runtime.settings import RefMode


def clear_kernel_caches_and_set_ref_mode(ref_mode: RefMode) -> None:
    """Clear kernel caches and set ref_mode on all kernels in examples."""
    # Get all Python files in the examples directory
    example_files = Path(EXAMPLES_DIR).glob("*.py")

    for example_file in example_files:
        try:
            # Import the module
            mod = import_path(example_file)

            # Find all Helion kernels in the module and update their settings
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, helion.Kernel):
                    # Reset the kernel to clear any cached bound kernels
                    attr.reset()
                    # Update the kernel's ref_mode setting
                    attr.settings.ref_mode = ref_mode
        except Exception:
            # Skip files that can't be imported or have issues
            pass


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
        """Test that print works even in invalid Helion kernels in reference eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
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
    """Test @helion.kernel(ref_mode=RefMode.EAGER) parameter functionality."""

    def test_ref_eager_kernel_config(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def compile_test(x: torch.Tensor) -> torch.Tensor:
            return x + x * 2.0

        x = torch.randn(128, device="cuda")
        result = compile_test(x)
        expected = x + x * 2.0
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
