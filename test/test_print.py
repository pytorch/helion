from __future__ import annotations

import math
import os
from pathlib import Path
import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRocm
import helion.language as hl


@pytest.fixture(autouse=True)
def _store_capfd_on_class(request, capfd):
    """
    Expose pytest's capfd fixture as `self._capfd` inside the TestPrint class
    (works for unittest.TestCase-style tests).
    """
    if request.cls is not None:
        request.cls._capfd = capfd


class TestPrint(RefEagerTestDisabled, TestCase):
    def run_kernel_and_capture_output(self, kernel_fn, args):
        """Helper to run kernel and capture output"""
        if hasattr(self, "_capfd"):
            # Using pytest - use capfd for cleaner output capture

            # Reset kernel to ensure compilation happens
            kernel_fn.reset()
            code, result = code_and_output(kernel_fn, args)

            # Wait for any device prints to reach the host
            if hasattr(result, "device") and result.device.type == DEVICE.type:
                torch.accelerator.synchronize()

            # Grab what pytest captured:  stdout + stderr
            out, err = self._capfd.readouterr()
            return code, result, out + err
        # Running with unittest directly - use file descriptor redirection
        import sys
        import tempfile

        # Reset kernel to ensure compilation happens
        kernel_fn.reset()

        # Create a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_filename = temp_file.name

        # Save original stdout file descriptor
        stdout_fd = sys.stdout.fileno()
        stdout_copy = os.dup(stdout_fd)

        try:
            # Open temp file and redirect stdout to it at the file descriptor level
            with open(temp_filename, "w+") as f:
                os.dup2(f.fileno(), stdout_fd)
                sys.stdout.flush()

                # Get the generated code and result
                code, result = code_and_output(kernel_fn, args)

                # Force GPU synchronization to ensure all device prints complete
                if hasattr(result, "device") and result.device.type == DEVICE.type:
                    torch.accelerator.synchronize()

                # Ensure all output is flushed
                sys.stdout.flush()

            # Read captured output
            output_str = Path(temp_filename).read_text()

        finally:
            # Restore original stdout
            os.dup2(stdout_copy, stdout_fd)
            os.close(stdout_copy)
            # Clean up temp file
            os.unlink(temp_filename)

        return code, result, output_str

    def run_test_with_and_without_triton_interpret_envvar(self, test_func):
        """Helper to run a test function with and without TRITON_INTERPRET=1"""
        original_env = os.environ.get("TRITON_INTERPRET")

        try:
            # First run without TRITON_INTERPRET
            if original_env:
                os.environ.pop("TRITON_INTERPRET", None)
            test_func(interpret_mode=False)

            # Then run with TRITON_INTERPRET=1
            os.environ["TRITON_INTERPRET"] = "1"
            test_func(interpret_mode=True)
        finally:
            # Restore original env
            if original_env is None:
                os.environ.pop("TRITON_INTERPRET", None)
            else:
                os.environ["TRITON_INTERPRET"] = original_env

    @skipIfRocm("failure on rocm")
    def test_basic_print(self):
        """Test basic print with prefix and tensor values"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    val = x[tile_m, tile_n]
                    print("tensor value: ", val)
                    out[tile_m, tile_n] = val * 2
                return out

            x = torch.ones([2, 2], device=DEVICE) * 42.0  # Use predictable values

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_kernel, (x,)
            )
            torch.testing.assert_close(result, x * 2)

            # Check that print is generated in the code
            self.assertIn("'tensor value: '", code)
            self.assertIn("tl.device_print('tensor value: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )
            for line in output_lines:
                self.assertIn("tensor value: 42", line)
                self.assertTrue("pid" in line and "idx" in line)

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_multiple_tensors(self):
        """Test print with multiple tensor arguments"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_multi_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    x_val = x[tile_m, tile_n]
                    y_val = y[tile_m, tile_n]
                    print("x and y: ", x_val, y_val)
                    out[tile_m, tile_n] = x_val + y_val
                return out

            x = torch.ones([2, 2], device=DEVICE) * 10.0
            y = torch.ones([2, 2], device=DEVICE) * 20.0

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_multi_kernel, (x, y)
            )
            torch.testing.assert_close(result, x + y)

            # Check that print is generated with multiple format specifiers
            self.assertIn("'x and y: '", code)
            self.assertIn("tl.device_print('x and y: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )
            # NOTE: tl.device_print prints each operand on a separate line
            for line in output_lines:
                self.assertIn("x and y:", line)
                # Each line will have either operand 0 (value 10) or operand 1 (value 20)
                self.assertTrue("10" in line or "20" in line)

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    def test_print_no_prefix_error(self):
        """Test that print without arguments raises an error"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_no_args_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print()  # This should fail
                    out[tile_m, tile_n] = x[tile_m, tile_n]
                return out

            x = torch.randn([32, 32], device=DEVICE)
            with pytest.raises(
                helion.exc.InternalError,
                match="print\\(\\) requires at least one argument",
            ):
                code_and_output(print_no_args_kernel, (x,))

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    def test_print_non_string_prefix_error(self):
        """Test that print with non-string prefix raises an error"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_bad_prefix_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print(123, x[tile_m, tile_n])  # Non-string prefix
                    out[tile_m, tile_n] = x[tile_m, tile_n]
                return out

            x = torch.randn([32, 32], device=DEVICE)
            with pytest.raises(
                helion.exc.InternalError,
                match="First argument to print\\(\\) must be a string prefix",
            ):
                code_and_output(print_bad_prefix_kernel, (x,))

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    def test_print_compile_time_value_error(self):
        """Test that printing compile-time values works (now supported)"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_shape_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print("shape: ", m)  # Compile-time value inside loop
                    out[tile_m, tile_n] = x[tile_m, tile_n]
                return out

            x = torch.randn([32, 32], device=DEVICE)
            with pytest.raises(
                helion.exc.InternalError,
                match="print\\(\\) only supports runtime tensor values",
            ):
                code_and_output(print_shape_kernel, (x,))

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_prefix_only(self):
        def run_test(interpret_mode):
            @helion.kernel
            def print_message_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print("processing tile")
                    out[tile_m, tile_n] = x[tile_m, tile_n] * 2
                return out

            x = torch.ones([2, 2], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_message_kernel, (x,)
            )
            torch.testing.assert_close(result, x * 2)

            # Check that print is generated
            self.assertIn("'processing tile'", code)
            self.assertIn("tl.device_print('processing tile'", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )
            for line in output_lines:
                self.assertIn("processing tile", line)

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_in_nested_loops(self):
        def run_test(interpret_mode):
            @helion.kernel
            def print_nested_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                m, k = x.shape
                k2, n = y.shape
                assert k == k2
                out = torch.zeros([m, n], device=x.device, dtype=x.dtype)

                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=x.dtype)
                    for tile_k in hl.tile(k):
                        x_val = x[tile_m, tile_k]
                        y_val = y[tile_k, tile_n]
                        print("inner loop x: ", x_val)
                        print("inner loop y: ", y_val)
                        acc = torch.addmm(acc, x_val, y_val)
                    print("accumulator: ", acc)
                    out[tile_m, tile_n] = acc
                return out

            x = torch.ones([16, 16], device=DEVICE) * 2.0
            y = torch.ones([16, 16], device=DEVICE) * 3.0

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_nested_kernel, (x, y)
            )
            # This is a matrix multiplication: result = x @ y
            expected = x @ y
            torch.testing.assert_close(result, expected)

            # Check that print is generated in the code
            self.assertIn("'inner loop x: '", code)
            self.assertIn("'inner loop y: '", code)
            self.assertIn("'accumulator: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )
            for line in output_lines:
                # Check that each line has one of the expected patterns
                if "inner loop x:" in line:
                    self.assertIn(
                        "2.0", line, f"Expected x value of 2.0 in line: {line}"
                    )
                elif "inner loop y:" in line:
                    self.assertIn(
                        "3.0", line, f"Expected y value of 3.0 in line: {line}"
                    )
                elif "accumulator:" in line:
                    # For a 16x16 matmul with all 2s and 3s, each accumulator element is 2*3*16 = 96
                    self.assertIn(
                        "96.0",
                        line,
                        f"Expected accumulator value of 96.0 in line: {line}",
                    )
                else:
                    self.fail(f"Unexpected output line: {line}")

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    def test_print_outside_tile_loops(self):
        """Test print statements outside tile loops"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_outside_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape

                # Print outside tile loops
                print("starting kernel")

                for tile_m, tile_n in hl.tile([m, n]):
                    out[tile_m, tile_n] = x[tile_m, tile_n] * 2
                return out

            x = torch.ones([2, 2], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_outside_kernel, (x,)
            )
            torch.testing.assert_close(result, x * 2)

            self.assertIn("print('starting kernel')", code)
            self.assertIn("starting kernel", output)

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_with_conditional(self):
        """Test print with conditional statements"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_conditional_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    val = x[tile_m, tile_n]
                    # Print the actual value with a label indicating if it's positive or negative
                    mask = val > 0
                    # Always print the value, but with different prefixes based on condition
                    print("value is positive: ", val)
                    print("value sign: ", torch.where(mask, 1.0, -1.0))
                    out[tile_m, tile_n] = torch.where(mask, val * 2, val * 3)
                return out

            x = torch.tensor([[1.0, 2.0], [3.0, -4.0]], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_conditional_kernel, (x,)
            )
            expected = torch.where(x > 0, x * 2, x * 3)
            torch.testing.assert_close(result, expected)

            # Check that print is generated
            self.assertIn("'value is positive: '", code)
            self.assertIn("'value sign: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )

            # Check each line for expected values
            # For input [[1.0, 2.0], [3.0, -4.0]]
            for line in output_lines:
                if "value is positive:" in line:
                    # Should contain one of the actual values from the input tensor
                    self.assertTrue(
                        "1.0" in line
                        or "2.0" in line
                        or "3.0" in line
                        or "-4.0" in line,
                        f"Expected one of the input values in line: {line}",
                    )
                elif "value sign:" in line:
                    # Should contain either 1.0 (for positive) or -1.0 (for negative)
                    self.assertTrue(
                        "1.0" in line or "-1.0" in line,
                        f"Expected sign value (1.0 or -1.0) in line: {line}",
                    )
                else:
                    self.fail(f"Unexpected output line: {line}")

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_computed_values(self):
        """Test print with computed/derived values"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_computed_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    x_val = x[tile_m, tile_n]
                    y_val = y[tile_m, tile_n]
                    sum_val = x_val + y_val
                    prod_val = x_val * y_val
                    print("sum: ", sum_val)
                    print("product: ", prod_val)
                    print("x/y ratio: ", x_val / y_val)
                    out[tile_m, tile_n] = sum_val + prod_val
                return out

            x = torch.tensor([[6.0, 8.0]], device=DEVICE)
            y = torch.tensor([[2.0, 4.0]], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_computed_kernel, (x, y)
            )
            torch.testing.assert_close(result, x + y + x * y)

            # Check that prints are generated
            self.assertIn("'sum: '", code)
            self.assertIn("'product: '", code)
            self.assertIn("'x/y ratio: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )

            # For x=6, y=2: sum=8, product=12, ratio=3
            # For x=8, y=4: sum=12, product=32, ratio=2
            for line in output_lines:
                self.assertTrue(
                    "sum: 8" in line
                    or "sum: 12" in line
                    or "product: 12" in line
                    or "product: 32" in line
                    or "x/y ratio: 3" in line
                    or "x/y ratio: 2" in line
                )

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @unittest.skip("TODO(yf225): make printing reduction output work")
    def test_print_reduction(self):
        """Test print reduction output"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_reduction_kernel(x: torch.Tensor) -> torch.Tensor:
                m, n = x.shape
                out = torch.zeros([m], device=x.device, dtype=x.dtype)

                # Simple reduction using built-in sum
                for tile_m in hl.tile(m):
                    row_data = x[tile_m, :]
                    # Do the reduction
                    row_sum = row_data.sum(-1)
                    print("row sum: ", row_sum)
                    out[tile_m] = row_sum
                return out

            # Use smaller tensor for testing
            x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_reduction_kernel, (x,)
            )
            torch.testing.assert_close(result, x.sum(dim=1))

            # Check that prints are generated
            self.assertIn("'row sum: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )
            for line in output_lines:
                self.assertTrue("row sum: 6" in line or "row sum: 15" in line)

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_multiple_data_types(self):
        """Test print with different tensor data types"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_dtypes_kernel(
                x_float: torch.Tensor, x_int: torch.Tensor
            ) -> torch.Tensor:
                out = torch.empty_like(x_float)
                m, n = x_float.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    f_val = x_float[tile_m, tile_n]
                    i_val = x_int[tile_m, tile_n]
                    print("float val: ", f_val)
                    print("int val: ", i_val)
                    # Convert int to float for output
                    out[tile_m, tile_n] = f_val + i_val.to(x_float.dtype)
                return out

            x_float = torch.tensor(
                [[math.pi, math.e]], device=DEVICE, dtype=torch.float32
            )
            x_int = torch.tensor([[42, 100]], device=DEVICE, dtype=torch.int32)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_dtypes_kernel, (x_float, x_int)
            )
            torch.testing.assert_close(result, x_float + x_int.float())

            # Check that prints are generated
            self.assertIn("'float val: '", code)
            self.assertIn("'int val: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )
            for line in output_lines:
                self.assertTrue(
                    "float val:" in line or "int val:" in line,
                    f"Expected print prefix in line: {line}",
                )
                # Check for expected values based on the prefix
                if "float val:" in line:
                    self.assertTrue(
                        "3.14" in line or "2.71" in line,
                        f"Expected float value (3.14 or 2.71) in line: {line}",
                    )
                elif "int val:" in line:
                    self.assertTrue(
                        "42" in line or "100" in line,
                        f"Expected int value (42 or 100) in line: {line}",
                    )

        self.run_test_with_and_without_triton_interpret_envvar(run_test)

    @skipIfRocm("failure on rocm")
    def test_print_with_starred_args(self):
        """Test print with starred/unpacked arguments"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_starred_kernel(
                x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    x_val = x[tile_m, tile_n]
                    y_val = y[tile_m, tile_n]
                    z_val = z[tile_m, tile_n]

                    # Create a list of values and unpack with *
                    values = [x_val, y_val, z_val]
                    print("unpacked values: ", *values)

                    # Also test with tuple unpacking
                    tuple_values = (x_val + 1, y_val + 1, z_val + 1)
                    print("unpacked tuple: ", *tuple_values)

                    out[tile_m, tile_n] = x_val + y_val + z_val
                return out

            x = torch.tensor([[1.0, 2.0]], device=DEVICE)
            y = torch.tensor([[3.0, 4.0]], device=DEVICE)
            z = torch.tensor([[5.0, 6.0]], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_and_capture_output(
                print_starred_kernel, (x, y, z)
            )
            torch.testing.assert_close(result, x + y + z)

            # Check that print is generated with multiple arguments
            self.assertIn("'unpacked values: '", code)
            self.assertIn("'unpacked tuple: '", code)
            # Should have multiple tl.device_print calls with the unpacked args
            self.assertIn("tl.device_print('unpacked values: '", code)
            self.assertIn("tl.device_print('unpacked tuple: '", code)

            output_lines = [line for line in output.strip().split("\n") if line]
            self.assertGreater(
                len(output_lines), 0, "Expected print output to be captured"
            )

            # Check that each line contains expected output
            # tl.device_print prints each operand on a separate line, so we check individually
            for line in output_lines:
                self.assertTrue(
                    "unpacked values:" in line or "unpacked tuple:" in line,
                    f"Expected print prefix in line: {line}",
                )
                # Each line should contain one of the expected values
                self.assertTrue(
                    any(val in line for val in ["1", "2", "3", "4", "5", "6", "7"]),
                    f"Expected to find a numeric value in line: {line}",
                )

        self.run_test_with_and_without_triton_interpret_envvar(run_test)


if __name__ == "__main__":
    unittest.main()
