from __future__ import annotations

import ast
import importlib
import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl
from helion.runtime.settings import _get_backend


def add_combine_fn(x, y):
    """Simple addition combine function for prefix sum."""
    return x + y


def max_combine_fn(x, y):
    """Maximum combine function for prefix maximum."""
    return torch.maximum(x, y)


def mul_combine_fn(x, y):
    """Multiplication combine function for prefix product."""
    return x * y


def min_combine_fn(x, y):
    """Minimum combine function for prefix minimum."""
    return torch.minimum(x, y)


def helion_combine_fn(left_values, left_indices, right_values, right_indices):
    """Tuple combine function with unpacked arguments (matching GitHub issue example)."""
    # Segmented scan: if indices are the same, add values; otherwise, take right values (reset)
    same_segment = left_indices == right_indices
    combined_values = torch.where(
        same_segment, left_values + right_values, right_values
    )
    combined_indices = right_indices  # Always propagate the right index
    return combined_values, combined_indices


def segmented_combine_fn(left_values, left_indices, right_values, right_indices):
    """Segmented scan: reset accumulation when segment changes."""
    same_segment = left_indices == right_indices
    combined_values = torch.where(
        same_segment, left_values + right_values, right_values
    )
    combined_indices = right_indices  # Always propagate the right index
    return combined_values, combined_indices


def argmax_combine_fn(left_values, left_indices, right_values, right_indices):
    """Cumulative argmax: keep the value and index of the maximum element seen so far."""
    # If right value is greater, take right value and index; otherwise keep left
    take_right = right_values > left_values
    combined_values = torch.where(take_right, right_values, left_values)
    combined_indices = torch.where(take_right, right_indices, left_indices)
    return combined_values, combined_indices


def helion_combine_tuple_fn(left_tuple, right_tuple):
    """Tuple combine function with tuple arguments (matching reduce format)."""
    left_values, left_indices = left_tuple
    right_values, right_indices = right_tuple
    # Segmented scan: if indices are the same, add values; otherwise, take right values (reset)
    same_segment = left_indices == right_indices
    combined_values = torch.where(
        same_segment, left_values + right_values, right_values
    )
    combined_indices = right_indices  # Always propagate the right index
    return combined_values, combined_indices


def argmax_combine_tuple_fn(left_tuple, right_tuple):
    """Cumulative argmax using tuple format."""
    left_values, left_indices = left_tuple
    right_values, right_indices = right_tuple
    # If right value is greater, take right value and index; otherwise keep left
    take_right = right_values > left_values
    combined_values = torch.where(take_right, right_values, left_values)
    combined_indices = torch.where(take_right, right_indices, left_indices)
    return combined_values, combined_indices


def cumsum_helper(x: torch.Tensor) -> torch.Tensor:
    """Helper function that performs cumulative sum using hl.associative_scan."""
    return hl.associative_scan(add_combine_fn, x, dim=0)


@helion.kernel
def jit_add_combine_fn(x, y):
    """Addition combine function with @helion.kernel decorator (should be ignored)."""
    return x + y


@onlyBackends(["triton", "cute"])
class TestAssociativeScan(RefEagerTestBase, TestCase):
    def test_computed_cumsum_numeric_dtypes(self) -> None:
        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor, reverse: hl.constexpr) -> torch.Tensor:
            output = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                columns = hl.arange(x.size(1))
                values = torch.where(columns[None, :] < x.size(1), x[row, :] + 1, 0)
                output[row, :] = hl.cumsum(values, dim=-1, reverse=reverse)
            return output

        for dtype in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            x = (torch.arange(195, device=DEVICE).reshape(3, 65) % 3).to(dtype)
            for reverse in (False, True):
                with self.subTest(dtype=dtype, reverse=reverse):
                    _, actual = code_and_output(kernel, (x, reverse), block_sizes=[1])
                    values = x + 1
                    if reverse:
                        values = values.flip([-1])
                    expected = values.cumsum(-1, dtype=dtype)
                    if reverse:
                        expected = expected.flip([-1])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @onlyBackends(["cute"])
    def test_computed_cumsum_unmasked_tail(self) -> None:
        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor, reverse: hl.constexpr) -> torch.Tensor:
            output = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                values = x[row, :] + 1
                output[row, :] = hl.cumsum(values, dim=-1, reverse=reverse)
            return output

        for dtype in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            x = (torch.arange(195, device=DEVICE).reshape(3, 65) % 3).to(dtype)
            for reverse in (False, True):
                with self.subTest(dtype=dtype, reverse=reverse):
                    _, actual = code_and_output(kernel, (x, reverse), block_sizes=[1])
                    values = x + 1
                    if reverse:
                        values = values.flip([-1])
                    expected = values.cumsum(-1, dtype=dtype)
                    if reverse:
                        expected = expected.flip([-1])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_computed_cumsum_axes_and_source_reuse(self) -> None:
        @helion.kernel(static_shapes=True)
        def kernel(
            x: torch.Tensor, axis: hl.constexpr, reverse: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = torch.empty_like(x)
            reused = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                values = x[row, :, :] * 2
                output[row, :, :] = hl.cumsum(values, dim=axis, reverse=reverse)
                reused[row, :, :] = values + 3
            return output, reused

        x = torch.arange(3 * 5 * 65, device=DEVICE, dtype=torch.int32).reshape(3, 5, 65)
        for axis in (1, 2):
            for reverse in (False, True):
                with self.subTest(axis=axis, reverse=reverse):
                    _, (actual, reused) = code_and_output(
                        kernel, (x, axis, reverse), block_sizes=[1]
                    )
                    values = x * 2
                    expected = values.flip([axis]) if reverse else values
                    expected = expected.cumsum(axis, dtype=x.dtype)
                    if reverse:
                        expected = expected.flip([axis])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(reused, values + 3, rtol=0, atol=0)

    def test_computed_cumsum_bitcast_partial_tiles(self) -> None:
        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor, reverse: hl.constexpr) -> torch.Tensor:
            output = torch.empty(x.shape, dtype=torch.int32, device=x.device)
            for row, col in hl.tile(x.shape, block_size=[2, 32]):
                flags = (x[row, col].view(torch.int32) < 0).to(torch.int32)
                output[row, col] = hl.cumsum(flags, dim=-1, reverse=reverse)
            return output

        x = torch.randn(5, 65, device=DEVICE)
        x[:, ::4] = -0.0
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                _, actual = code_and_output(kernel, (x, reverse))
                flags = (x.view(torch.int32) < 0).to(torch.int32)
                chunks = []
                for values in flags.split(32, dim=-1):
                    if reverse:
                        values = values.flip([-1])
                    values = values.cumsum(-1, dtype=torch.int32)
                    chunks.append(values.flip([-1]) if reverse else values)
                torch.testing.assert_close(
                    actual, torch.cat(chunks, -1), rtol=0, atol=0
                )

    def test_computed_fragment_indexed_scalar_fill(self) -> None:
        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
            scans = torch.empty_like(x)
            for row, col in hl.tile(x.shape, block_size=[2, 32]):
                destination[row, col] = -7
                values = x[row, col] + 1
                scans[row, col] = hl.cumsum(values, dim=-1)
                hl.store(destination, [row, col], 3, extra_mask=values > 0)
            return scans

        for dtype in (torch.int32, torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = (torch.arange(325, device=DEVICE).reshape(5, 65) % 5 - 2).to(dtype)
                backing = torch.full((7, 132), -99, dtype=dtype, device=DEVICE)
                destination = backing[1:6, 1:131:2]
                _, actual = code_and_output(kernel, (x, destination))
                expected_backing = torch.full_like(backing, -99)
                expected_backing[1:6, 1:131:2] = torch.where(x + 1 > 0, 3, -7).to(dtype)
                expected_scans = torch.cat(
                    [
                        values.cumsum(-1, dtype=dtype)
                        for values in (x + 1).split(32, -1)
                    ],
                    -1,
                )
                torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected_scans, rtol=0, atol=0)

    def test_associative_scan_basic_addition(self):
        """Test basic associative_scan functionality with prefix sum."""

        @helion.kernel(autotune_effort="none")
        def test_scan_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_scan_kernel, (x,))

        # Test the actual scan operation
        expected = torch.tensor(
            [[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0], [9.0, 19.0, 30.0, 42.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains the correct helper function
        if _get_backend() == "triton":
            self.assertIn("def add_combine_fn_", code)
            self.assertIn("param_0 + param_1", code)
            self.assertIn("tl.associative_scan", code)

    def test_associative_scan_maximum(self):
        """Test associative_scan with maximum combine function."""

        @helion.kernel(autotune_effort="none")
        def test_max_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(max_combine_fn, row_data, dim=1)
            return result

        # Test input with decreasing and increasing values
        x = torch.tensor(
            [[1.0, 5.0, 2.0, 8.0, 3.0], [7.0, 1.0, 9.0, 2.0, 4.0]],
            device=DEVICE,
        )

        code, result = code_and_output(test_max_kernel, (x,))

        # Expected prefix maximum
        expected = torch.tensor(
            [[1.0, 5.0, 5.0, 8.0, 8.0], [7.0, 7.0, 9.0, 9.0, 9.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains maximum operation (either tl.maximum or triton_helpers.maximum)
        if _get_backend() == "triton":
            self.assertTrueIfInNormalMode(
                "tl.maximum" in code or "triton_helpers.maximum" in code
            )

    def test_associative_scan_multiplication(self):
        """Test associative_scan with multiplication combine function."""

        @helion.kernel(autotune_effort="none")
        def test_mul_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(mul_combine_fn, row_data, dim=1)
            return result

        # Test input for prefix product
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 0.5, 3.0, 2.0]],
            device=DEVICE,
        )

        code, result = code_and_output(test_mul_kernel, (x,))

        # Expected prefix product
        expected = torch.tensor(
            [[1.0, 2.0, 6.0, 24.0], [2.0, 1.0, 3.0, 6.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains multiplication
        if _get_backend() == "triton":
            self.assertIn("param_0 * param_1", code)

    def test_associative_scan_minimum(self):
        """Test associative_scan with minimum combine function."""

        @helion.kernel(autotune_effort="none")
        def test_min_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(min_combine_fn, row_data, dim=1)
            return result

        # Test input with various values
        x = torch.tensor(
            [[5.0, 2.0, 8.0, 1.0, 6.0], [3.0, 7.0, 1.0, 9.0, 2.0]],
            device=DEVICE,
        )

        code, result = code_and_output(test_min_kernel, (x,))

        # Expected prefix minimum
        expected = torch.tensor(
            [[5.0, 2.0, 2.0, 1.0, 1.0], [3.0, 3.0, 1.0, 1.0, 1.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains minimum operation (either tl.minimum or triton_helpers.minimum)
        if _get_backend() == "triton":
            self.assertTrueIfInNormalMode(
                "tl.minimum" in code or "triton_helpers.minimum" in code
            )

    def test_associative_scan_multiple_functions(self):
        """Test using multiple different combine functions in one kernel."""

        @helion.kernel(autotune_effort="none")
        def test_multi_kernel(x: torch.Tensor) -> torch.Tensor:
            sum_result = torch.empty_like(x)
            max_result = torch.empty_like(x)

            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                # Prefix sum
                sum_result[i, :] = hl.associative_scan(add_combine_fn, row_data, dim=1)
                # Prefix maximum
                max_result[i, :] = hl.associative_scan(max_combine_fn, row_data, dim=1)

            # Return sum for testing
            return sum_result

        x = torch.tensor([[1.0, 3.0, 2.0, 4.0]], device=DEVICE)

        code, result = code_and_output(test_multi_kernel, (x,))

        # Test the sum result
        expected_sum = torch.tensor([[1.0, 4.0, 6.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected_sum)

        # Verify multiple helper functions are generated
        if _get_backend() == "triton":
            self.assertIn("add_combine_fn_", code)
            self.assertIn("max_combine_fn_", code)
            self.assertIn("param_0 + param_1", code)
            # Check for maximum operation (either format)
            self.assertTrueIfInNormalMode(
                "tl.maximum" in code or "triton_helpers.maximum" in code
            )

    def test_associative_scan_type_propagation(self):
        """Test that associative_scan type propagation works correctly."""

        @helion.kernel(autotune_effort="none")
        def test_type_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(add_combine_fn, row_data, dim=1)
            return result

        x = torch.randn(16, 1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(test_type_kernel, (x,))

        # Verify the output has the same type and shape as input
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device, x.device)
        # Verify it produces the correct cumulative sum
        expected = torch.cumsum(x, dim=1)
        # Use relaxed tolerance for large tensors due to accumulated floating-point errors
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_associative_scan_different_dtypes(self):
        """Test associative_scan with different data types."""

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            with self.subTest(dtype=str(dtype)):

                @helion.kernel(autotune_effort="none")
                def test_dtype_kernel(x: torch.Tensor) -> torch.Tensor:
                    result = torch.empty_like(x)
                    for i in hl.tile(x.size(0)):
                        row_data = x[i, :]
                        result[i, :] = hl.associative_scan(
                            add_combine_fn, row_data, dim=1
                        )
                    return result

                # Use integer values for all dtypes to avoid precision issues
                x_vals = [[1, 2, 3, 4], [5, 6, 7, 8]]
                x = torch.tensor(x_vals, device=DEVICE, dtype=dtype)

                code, result = code_and_output(test_dtype_kernel, (x,))

                # Verify output dtype matches input
                self.assertEqual(result.dtype, x.dtype)

                # Check correctness for numeric types
                if dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                    expected = torch.cumsum(x, dim=1)
                    # Convert expected to match result dtype if needed
                    if expected.dtype != result.dtype:
                        expected = expected.to(result.dtype)
                    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_associative_scan_different_sizes(self):
        """Test associative_scan with different tensor sizes."""

        test_shapes = [
            (1, 4),  # Single row
            (3, 8),  # Multiple rows
            (5, 16),  # Medium size
            (2, 1),  # Single column
            (4, 1024),  # Large size
            (8, 1024),  # Multiple large rows
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):

                @helion.kernel(autotune_effort="none")
                def test_size_kernel(x: torch.Tensor) -> torch.Tensor:
                    result = torch.empty_like(x)
                    for i in hl.tile(x.size(0)):
                        row_data = x[i, :]
                        result[i, :] = hl.associative_scan(
                            add_combine_fn, row_data, dim=1
                        )
                    return result

                x = torch.randn(shape, device=DEVICE)
                code, result = code_and_output(test_size_kernel, (x,))

                # Verify output shape matches input
                self.assertEqual(result.shape, x.shape)

                # Verify correctness
                expected = torch.cumsum(x, dim=1)
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_associative_scan_reverse(self):
        """Test associative_scan with reverse=True parameter."""

        @helion.kernel(autotune_effort="none")
        def test_reverse_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(
                    add_combine_fn, row_data, dim=1, reverse=True
                )
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)

        code, result = code_and_output(test_reverse_kernel, (x,))

        # For reverse prefix sum: [10, 9, 7, 4] (sum from right to left)
        expected = torch.tensor([[10.0, 9.0, 7.0, 4.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify reverse parameter is in generated code
        self.assertIn("reverse=True", code)

    def test_associative_scan_edge_cases(self):
        """Test associative_scan edge cases."""

        # Single element
        @helion.kernel(autotune_effort="none")
        def test_single_element(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(add_combine_fn, row_data, dim=1)
            return result

        x_single = torch.tensor([[5.0]], device=DEVICE)
        code, result = code_and_output(test_single_element, (x_single,))
        expected = torch.tensor([[5.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Two elements
        x_two = torch.tensor([[3.0, 7.0]], device=DEVICE)
        code, result = code_and_output(test_single_element, (x_two,))
        expected = torch.tensor([[3.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_associative_scan_large_scale(self):
        """Test associative_scan with large tensors for performance validation."""

        @helion.kernel(autotune_effort="none")
        def test_large_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(add_combine_fn, row_data, dim=1)
            return result

        # Test with large tensor
        x = torch.randn(32, 1024, device=DEVICE)
        code, result = code_and_output(test_large_kernel, (x,))

        # Verify correctness on large scale
        expected = torch.cumsum(x, dim=1)
        # Use relaxed tolerance for large tensors due to accumulated floating-point errors
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify output properties
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan maps to hl.associative_scan only during tracing; "
        "ref eager mode runs the raw torch HOP, which is unsupported"
    )
    def test_associative_scan_torch_hops_mapping(self):
        """Test that torch._higher_order_ops.associative_scan automatically maps to hl.associative_scan."""

        @helion.kernel(autotune_effort="none")
        def test_torch_hops_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                # Use torch._higher_order_ops.associative_scan directly
                result[i, :] = torch._higher_order_ops.associative_scan(
                    add_combine_fn, row_data, dim=1
                )
            return result

        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs correctly
        code, result = code_and_output(test_torch_hops_kernel, (x,))

        # Expected prefix sum results
        expected = torch.tensor(
            [[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code contains the proper combine function and associative scan
        if _get_backend() == "triton":
            self.assertIn("def add_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)
            self.assertIn("param_0 + param_1", code)

    def test_associative_scan_code_generation(self):
        """Test that the generated code structure is correct."""

        @helion.kernel(autotune_effort="none")
        def test_codegen_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(add_combine_fn, row_data, dim=1)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)
        code, result = code_and_output(test_codegen_kernel, (x,))

        # Verify the result is correct
        expected = torch.tensor([[1.0, 3.0, 6.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Check essential code structure
        if _get_backend() == "triton":
            self.assertIn("@triton.jit", code)
            self.assertIn("def add_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)
        self.assertIn("return", code)

        # Verify no placeholders remain
        self.assertNotIn("TODO", code)
        self.assertNotIn("placeholder", code)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with nested @helion.kernel is not supported by ref eager mode yet"
    )
    def test_associative_scan_jit_decorator_ignored(self):
        """Test that @helion.kernel decorator on combine functions is ignored."""

        @helion.kernel(autotune_effort="none")
        def test_jit_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(jit_add_combine_fn, row_data, dim=1)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)
        code, result = code_and_output(test_jit_kernel, (x,))

        # Expected prefix sum results
        expected = torch.tensor([[1.0, 3.0, 6.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code contains the proper combine function and associative scan
        if _get_backend() == "triton":
            self.assertIn("def jit_add_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)
            self.assertIn("param_0 + param_1", code)
        # Verify @helion.kernel decorator doesn't appear in generated code
        self.assertNotIn("@helion.kernel", code)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with tuple arg is not supported by ref eager mode yet"
    )
    def test_associative_scan_tuple_args(self):
        """Test associative_scan with tuple arguments (matching GitHub issue #237 pattern)."""

        @helion.kernel(autotune_effort="none")
        def test_segmented_kernel(
            indices: torch.Tensor, input_data: torch.Tensor
        ) -> torch.Tensor:
            E, C = input_data.shape
            output = torch.zeros(
                (E, C), dtype=input_data.dtype, device=input_data.device
            )

            for tile_e, tile_f in hl.tile([E, C]):
                vals = input_data[tile_e, tile_f]
                # Broadcast indices to match vals shape for the scan
                idxs = indices[tile_e].unsqueeze(1).expand_as(vals)

                # Create tuple inside the device loop (as per GitHub issue example)
                input_tuple = (vals, idxs)

                # Use torch._higher_order_ops.associative_scan as in the example
                out_vals, out_idxs = torch._higher_order_ops.associative_scan(
                    # pyrefly: ignore [bad-argument-type]
                    helion_combine_fn,
                    input_tuple,
                    0,
                )

                output[tile_e, tile_f] = out_vals

            return output

        # Create test data
        E, C = 4, 2
        indices = torch.tensor(
            [0.0, 0.0, 1.0, 1.0], device=DEVICE
        )  # Use float to match input_data
        input_data = torch.ones((E, C), device=DEVICE)

        code, result = code_and_output(test_segmented_kernel, (indices, input_data))

        # Expected: cumulative sum for each position
        expected = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code structure
        if _get_backend() == "triton":
            self.assertIn("def helion_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with tuple arg is not supported by ref eager mode yet"
    )
    def test_associative_scan_segmented_reduction(self):
        """Test associative_scan for segmented reduction use case."""

        @helion.kernel(autotune_effort="none")
        def segmented_scan_kernel(
            indices: torch.Tensor, input_data: torch.Tensor
        ) -> torch.Tensor:
            E, C = input_data.shape
            output = torch.zeros(
                (E, C), dtype=input_data.dtype, device=input_data.device
            )

            for tile_e, tile_f in hl.tile([E, C]):
                vals = input_data[tile_e, tile_f]
                # Convert indices to float to match vals dtype and broadcast to match shape
                idxs = indices[tile_e].float().unsqueeze(1).expand_as(vals)

                # Use tuple argument functionality for segmented scan
                out_vals, _ = torch._higher_order_ops.associative_scan(
                    # pyrefly: ignore [bad-argument-type]
                    segmented_combine_fn,
                    (vals, idxs),
                    0,
                )

                output[tile_e, tile_f] = out_vals

            return output

        # Test segmented reduction
        E, C = 6, 3
        # Segments: [0,0], [1,1,1], [2] - three segments of sizes 2, 3, 1
        indices = torch.tensor([0, 0, 1, 1, 1, 2], device=DEVICE)
        input_data = torch.ones((E, C), device=DEVICE)

        code, result = code_and_output(segmented_scan_kernel, (indices, input_data))

        # Expected: cumulative sum within each segment
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # segment 0, position 0
                [2.0, 2.0, 2.0],  # segment 0, position 1
                [1.0, 1.0, 1.0],  # segment 1, position 0
                [2.0, 2.0, 2.0],  # segment 1, position 1
                [3.0, 3.0, 3.0],  # segment 1, position 2
                [1.0, 1.0, 1.0],  # segment 2, position 0
            ],
            device=DEVICE,
        )

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code structure
        if _get_backend() == "triton":
            self.assertIn("def segmented_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with tuple arg is not supported by ref eager mode yet"
    )
    def test_associative_scan_cumulative_argmax(self):
        """Test cumulative argmax using tuple args with (float, int) types."""

        @helion.kernel(autotune_effort="none")
        def cumulative_argmax_kernel(
            input_data: torch.Tensor, positions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            max_values = torch.zeros_like(input_data)
            max_indices = torch.zeros_like(input_data, dtype=torch.int32)
            for tile_e in hl.tile(input_data.size(0)):
                vals = input_data[tile_e, :]
                # Convert positions to float to match vals dtype, then broadcast to match vals shape
                indices = positions[:].to(torch.float32).unsqueeze(0).expand_as(vals)

                # Use hl.associative_scan directly with tuple args - return both values and indices
                out_vals, out_indices = hl.associative_scan(
                    argmax_combine_fn, (vals, indices), dim=1
                )

                max_values[tile_e, :] = out_vals
                max_indices[tile_e, :] = out_indices.to(torch.int32)

            return max_values, max_indices

        input_data = torch.tensor(
            [
                [1.0, 5.5, 2.0],
                [3.0, 2.0, 4.0],
                [2.0, 7.0, 1.0],
                [4.1, 1.0, 3.0],
            ],
            device=DEVICE,
        )
        positions = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int32)
        code, (result_values, result_indices) = code_and_output(
            cumulative_argmax_kernel, (input_data, positions)
        )

        # Expected cumulative maximum values
        expected_values = torch.tensor(
            [
                [1.0, 5.5, 5.5],
                [3.0, 3.0, 4.0],
                [2.0, 7.0, 7.0],
                [4.1, 4.1, 4.1],
            ],
            device=DEVICE,
        )

        # Expected indices of the maximum values (which row they came from)
        expected_indices = torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 2],
                [0, 1, 1],
                [0, 0, 0],
            ],
            device=DEVICE,
            dtype=torch.int32,
        )

        torch.testing.assert_close(result_values, expected_values)
        torch.testing.assert_close(result_indices, expected_indices)

        # Verify the generated code structure
        if _get_backend() == "triton":
            self.assertIn("def argmax_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)

    def test_associative_scan_in_helper_function(self):
        """Test calling a function that internally uses hl.associative_scan."""

        @helion.kernel(autotune_effort="none")
        def test_helper_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                # Use the cumsum_helper function which internally calls hl.associative_scan
                result[i, :] = cumsum_helper(x[i, :])
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_helper_kernel, (x,))

        # Verify the result is correct (cumsum along dim=0)
        expected = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [6.0, 8.0, 10.0, 12.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains the helper function and associative scan
        if _get_backend() == "triton":
            self.assertIn("def add_combine_fn_", code)
            self.assertIn("tl.associative_scan", code)
            self.assertIn("param_0 + param_1", code)

    def test_cumsum_basic(self):
        """Test basic cumsum functionality."""

        @helion.kernel(autotune_effort="none")
        def test_cumsum_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = torch.cumsum(row_data, dim=1)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=DEVICE)

        code, result = code_and_output(test_cumsum_kernel, (x,))

        # Expected cumulative sum
        expected = torch.tensor(
            [[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains cumsum implementation
        if _get_backend() == "triton":
            self.assertIn("def add_", code)
            self.assertIn("param_0 + param_1", code)
            self.assertIn("tl.associative_scan", code)

    def test_cumsum_reverse(self):
        """Test cumsum with reverse=True."""

        @helion.kernel(autotune_effort="none")
        def test_cumsum_reverse_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.cumsum(row_data, dim=1, reverse=True)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)

        code, result = code_and_output(test_cumsum_reverse_kernel, (x,))

        # For reverse cumsum: [10, 9, 7, 4] (sum from right to left)
        expected = torch.tensor([[10.0, 9.0, 7.0, 4.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Verify reverse parameter is used
        self.assertIn("reverse=True", code)

    def test_cumsum_different_dtypes(self):
        """Test cumsum with different data types."""

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            with self.subTest(dtype=str(dtype)):

                @helion.kernel(autotune_effort="none")
                def test_cumsum_dtype_kernel(x: torch.Tensor) -> torch.Tensor:
                    result = torch.empty_like(x)
                    for i in hl.tile(x.size(0)):
                        row_data = x[i, :]
                        result[i, :] = torch.cumsum(row_data, dim=1)
                    return result

                x = torch.tensor(
                    [[1, 2, 3, 4], [5, 6, 7, 8]], device=DEVICE, dtype=dtype
                )

                code, result = code_and_output(test_cumsum_dtype_kernel, (x,))

                # Verify output dtype matches input
                self.assertEqual(result.dtype, x.dtype)

                # Check correctness
                expected = torch.cumsum(x, dim=1)
                # Convert expected to match result dtype if needed
                if expected.dtype != result.dtype:
                    expected = expected.to(result.dtype)
                torch.testing.assert_close(result, expected)

    def test_cumprod_basic(self):
        """Test basic cumprod functionality."""

        @helion.kernel(autotune_effort="none")
        def test_cumprod_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = torch.cumprod(row_data, dim=1)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 0.5, 3.0, 2.0]], device=DEVICE)

        code, result = code_and_output(test_cumprod_kernel, (x,))

        # Expected cumulative product
        expected = torch.tensor(
            [[1.0, 2.0, 6.0, 24.0], [2.0, 1.0, 3.0, 6.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains cumprod implementation
        if _get_backend() == "triton":
            self.assertIn("def mul_", code)
            self.assertIn("param_0 * param_1", code)
            self.assertIn("tl.associative_scan", code)

    def test_cumprod_reverse(self):
        """Test cumprod with reverse=True."""

        @helion.kernel(autotune_effort="none")
        def test_cumprod_reverse_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.cumprod(row_data, dim=1, reverse=True)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)

        code, result = code_and_output(test_cumprod_reverse_kernel, (x,))

        # For reverse cumprod: [24, 24, 12, 4] (product from right to left)
        expected = torch.tensor([[24.0, 24.0, 12.0, 4.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Verify reverse parameter is used
        self.assertIn("reverse=True", code)

    def test_cumprod_different_dtypes(self):
        """Test cumprod with different data types."""

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            with self.subTest(dtype=str(dtype)):

                @helion.kernel(autotune_effort="none")
                def test_cumprod_dtype_kernel(x: torch.Tensor) -> torch.Tensor:
                    result = torch.empty_like(x)
                    for i in hl.tile(x.size(0)):
                        row_data = x[i, :]
                        result[i, :] = hl.cumprod(row_data, dim=1)
                    return result

                x = torch.tensor(
                    [[1, 2, 3, 2], [2, 1, 2, 2]], device=DEVICE, dtype=dtype
                )

                code, result = code_and_output(test_cumprod_dtype_kernel, (x,))

                # Verify output dtype matches input
                self.assertEqual(result.dtype, x.dtype)

                # Check correctness
                expected = torch.cumprod(x, dim=1)
                # Convert expected to match result dtype if needed
                if expected.dtype != result.dtype:
                    expected = expected.to(result.dtype)
                torch.testing.assert_close(result, expected)

    def test_cumsum_cumprod_mixed(self):
        """Test using both cumsum and cumprod in the same kernel."""

        @helion.kernel(autotune_effort="none")
        def test_mixed_kernel(x: torch.Tensor) -> torch.Tensor:
            sum_result = torch.empty_like(x)
            prod_result = torch.empty_like(x)

            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                # Cumulative sum
                sum_result[i, :] = torch.cumsum(row_data, dim=1)
                # Cumulative product
                prod_result[i, :] = torch.cumprod(row_data, dim=1)

            # Return sum for testing
            return sum_result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)

        code, result = code_and_output(test_mixed_kernel, (x,))

        # Test the sum result
        expected_sum = torch.tensor([[1.0, 3.0, 6.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected_sum)

        # Verify both helper functions are generated
        if _get_backend() == "triton":
            self.assertIn("add_", code)
            self.assertIn("mul_", code)
            self.assertIn("param_0 + param_1", code)
            self.assertIn("param_0 * param_1", code)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with tuple arg is not supported by ref eager mode yet"
    )
    def test_associative_scan_tuple_format(self):
        """Test associative_scan with tuple format combine function (like reduce format)."""

        @helion.kernel(autotune_effort="none")
        def test_segmented_tuple_kernel(
            indices: torch.Tensor, input_data: torch.Tensor
        ) -> torch.Tensor:
            E, C = input_data.shape
            output = torch.zeros(
                (E, C), dtype=input_data.dtype, device=input_data.device
            )

            for tile_e, tile_f in hl.tile([E, C]):
                vals = input_data[tile_e, tile_f]
                # Broadcast indices to match vals shape for the scan
                idxs = indices[tile_e].unsqueeze(1).expand_as(vals)

                # Create tuple inside the device loop (as per GitHub issue example)
                input_tuple = (vals, idxs)

                # Use the tuple format combine function
                out_vals, out_idxs = torch._higher_order_ops.associative_scan(
                    helion_combine_tuple_fn, input_tuple, 0
                )

                output[tile_e, tile_f] = out_vals

            return output

        # Create test data
        E, C = 4, 2
        indices = torch.tensor(
            [0.0, 0.0, 1.0, 1.0], device=DEVICE
        )  # Use float to match input_data
        input_data = torch.ones((E, C), device=DEVICE)

        code, result = code_and_output(
            test_segmented_tuple_kernel, (indices, input_data)
        )

        # Expected: cumulative sum for each position
        expected = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code structure
        if _get_backend() == "triton":
            self.assertIn("def helion_combine_tuple_fn_", code)
            self.assertIn("tl.associative_scan", code)

    def test_associative_scan_argmax_tuple_format(self):
        """Test cumulative argmax using tuple format combine function."""

        @helion.kernel(autotune_effort="none")
        def cumulative_argmax_tuple_kernel(
            input_data: torch.Tensor, positions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            max_values = torch.zeros_like(input_data)
            max_indices = torch.zeros_like(input_data, dtype=torch.int32)
            for tile_e in hl.tile(input_data.size(0)):
                vals = input_data[tile_e, :]
                # Convert positions to float to match vals dtype, then broadcast to match vals shape
                indices = positions[:].to(torch.float32).unsqueeze(0).expand_as(vals)

                # Use hl.associative_scan directly with tuple format - return both values and indices
                out_vals, out_indices = hl.associative_scan(
                    argmax_combine_tuple_fn, (vals, indices), dim=1
                )

                max_values[tile_e, :] = out_vals
                max_indices[tile_e, :] = out_indices.to(torch.int32)

            return max_values, max_indices

        input_data = torch.tensor(
            [
                [1.0, 5.5, 2.0],
                [3.0, 2.0, 4.0],
                [2.0, 7.0, 1.0],
                [4.1, 1.0, 3.0],
            ],
            device=DEVICE,
        )
        positions = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int32)
        code, (result_values, result_indices) = code_and_output(
            cumulative_argmax_tuple_kernel, (input_data, positions)
        )

        # Expected cumulative maximum values
        expected_values = torch.tensor(
            [
                [1.0, 5.5, 5.5],
                [3.0, 3.0, 4.0],
                [2.0, 7.0, 7.0],
                [4.1, 4.1, 4.1],
            ],
            device=DEVICE,
        )

        # Expected indices of the maximum values (which row they came from)
        expected_indices = torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 2],
                [0, 1, 1],
                [0, 0, 0],
            ],
            device=DEVICE,
            dtype=torch.int32,
        )

        torch.testing.assert_close(result_values, expected_values)
        torch.testing.assert_close(result_indices, expected_indices)

        # Verify the generated code structure
        if _get_backend() == "triton":
            self.assertIn("def argmax_combine_tuple_fn_", code)
            self.assertIn("tl.associative_scan", code)

    @skipIfNotCUDA()
    @skipIfRefEager(
        "promoted-seed reduction_loops is only materialized in compiled mode"
    )
    @skipIfTileIR("TileIR reduction tiling differs")
    @skipIfCute("reduction seed is Triton-only; CuTe uses its own reduction tiling")
    def test_scan_in_reduction_default_config_not_looped(self) -> None:
        """Regression: a scan (cumsum) inside a reduction over an axis co-resident with
        a wide feature must not be emitted as a LOOPED reduction. The looped path
        re-runs the scan per chunk with no cross-chunk prefix carry (silently wrong);
        the reduction roller now refuses to roll a scan-containing reduction, keeping
        it persistent. Run with the promoted default (no explicit config) and match
        torch.cumsum; before the fix the seed looped this and was ~55% wrong. Shapes
        are kept small so the persistent tile fits a small GPU."""

        @helion.kernel(autotune_effort="none")
        def cumsum_mid_reduce(x: torch.Tensor) -> torch.Tensor:
            m, _r, n = x.shape
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                c = torch.cumsum(x[tile_m, :, :].to(torch.float32), dim=1)
                out[tile_m, :] = c.amax(1)
            return out

        x = torch.randn([8, 4, 2048], device=DEVICE, dtype=torch.float32)
        expected = torch.cumsum(x.double(), dim=1).amax(1).float()
        _code, out = code_and_output(cumsum_mid_reduce, (x,))
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_tiled_scan(x: torch.Tensor, kind: hl.constexpr, reverse: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        for column in hl.tile(x.size(1), block_size=8):
            value = x[row, column]
            if kind == "sum":
                result = hl.cumsum(value, dim=-1, reverse=reverse)
            elif kind == "prod":
                result = hl.cumprod(value, dim=-1, reverse=reverse)
            elif kind == "min":
                result = hl.associative_scan(
                    torch.minimum, value, dim=-1, reverse=reverse
                )
            else:
                result = hl.associative_scan(
                    torch.maximum, value, dim=-1, reverse=reverse
                )
            out[row, column] = result
    return out


def _execute_direct_scan_program(source, inputs):
    """Interpret generated tensor subscripts using their actual storage strides."""
    from test.test_indexing import _execute_pointwise_thread_program

    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    tensor_args = {arg.arg for arg in kernel.args.args if arg.annotation is None}
    modules = {
        alias.asname: importlib.import_module(alias.name)
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.asname == "_source_module"
    }

    class TensorIndex(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in modules:
                value = vars(modules[node.value.id])[node.attr]
                assert isinstance(value, int)
                return ast.Constant(value=value)
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in tensor_args:
                indices = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                tensor = node.value.id
                offset = " + ".join(
                    f"({ast.unparse(index)}) * {tensor}.layout.stride[{axis}]"
                    for axis, index in enumerate(indices)
                )
                return ast.parse(
                    f"({tensor}.iterator + ({offset})).load()", mode="eval"
                ).body
            return node

    # The scalar interpreter replaces module imports with CPU stand-ins. Keep
    # the real pure host launch-dimension guard for symbolic tile extents.
    wrapper = next(
        node for node in reversed(tree.body) if isinstance(node, ast.FunctionDef)
    )
    wrapper.body[:0] = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and any(alias.asname == "_cute_checked_block_dims" for alias in node.names)
    ]
    result, _ = _execute_pointwise_thread_program(
        ast.unparse(ast.fix_missing_locations(TensorIndex().visit(tree))), inputs
    )
    return result


def _ordinary_scan_expected(x, kind, reverse, chunk_size=8):
    results = []
    for chunk in x.split(chunk_size, dim=-1):
        value = chunk.flip([-1]) if reverse else chunk
        if kind == "sum":
            value = value.cumsum(-1, dtype=x.dtype)
        elif kind == "prod":
            value = value.cumprod(-1, dtype=x.dtype)
        elif kind == "min":
            value = value.cummin(-1).values
        else:
            value = value.cummax(-1).values
        results.append(value.flip([-1]) if reverse else value)
    return torch.cat(results, -1)


@pytest.fixture
def _serial_scan_fallback(monkeypatch):
    # These models execute scalar threads independently. Exercise the serial
    # fallback explicitly; native tests retain the default parallel dispatcher.
    from helion._compiler.backend_registry import repair_backend_codegen
    from helion._compiler.cute import scan_ops

    # Complete the upstream lazy reload before patching its actual emitter.
    repair_backend_codegen("cute")
    monkeypatch.setattr(scan_ops, "_cute_try_parallel_scan", lambda *args: None)


@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.usefixtures("_serial_scan_fallback")
def test_ordinary_tiled_scan_generated_values(kind, reverse, dtype):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    backing = ((torch.arange(3 * 34).reshape(3, 34) % 3) + 1).to(dtype)
    x = backing[:, ::2]
    snapshot = backing.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_ordinary_tiled_scan, (x, kind, reverse))
        source = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" not in source
    actual = _execute_direct_scan_program(source, (x, kind, reverse))
    torch.testing.assert_close(
        actual, _ordinary_scan_expected(x, kind, reverse), rtol=0, atol=0
    )
    torch.testing.assert_close(backing, snapshot, rtol=0, atol=0)


@pytest.mark.usefixtures("_serial_scan_fallback")
def test_ordinary_scan_host_fixed_chunk_generated_values():
    from unittest.mock import patch

    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target
    import test.test_cute_computed_fragment as fragment_tests
    from test.test_cute_computed_fragment import _fragment_host_fixed_chunk_scan

    x = (torch.arange(3 * 17).reshape(3, 17) % 7).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_host_fixed_chunk_scan, (x,))
        source = bound.to_code(bound.config_spec.default_config())
    for chunk_size in (4, 8, 16):
        with patch.object(fragment_tests, "_FRAGMENT_FIXED_CHUNK", chunk_size):
            actual = _execute_direct_scan_program(source, (x,))
        torch.testing.assert_close(
            actual, _ordinary_scan_expected(x, "sum", False, chunk_size), rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize("reverse", [False, True])
@onlyBackends(["cute"])
def test_ordinary_tiled_scan_native(kind, reverse):
    x = ((torch.arange(3 * 34, device=DEVICE).reshape(3, 34) % 3) + 1).float()[:, ::2]
    before = x.clone()
    torch.testing.assert_close(
        _ordinary_tiled_scan(x, kind, reverse),
        _ordinary_scan_expected(x, kind, reverse),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def _scan_product_and_sum(left_product, left_sum, right_product, right_sum):
    return left_product * right_product, left_sum + right_sum


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_tiled_tuple_scan(x: torch.Tensor, reverse: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        for column in hl.tile(x.size(1), block_size=8):
            values = x[row, column]
            product, total = hl.associative_scan(
                _scan_product_and_sum, (values, values), dim=-1, reverse=reverse
            )
            out[row, column] = product + total
    return out


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.usefixtures("_serial_scan_fallback")
def test_ordinary_tiled_tuple_scan_generated_values(reverse):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    x = ((torch.arange(3 * 17).reshape(3, 17) % 3) + 1).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_ordinary_tiled_tuple_scan, (x, reverse))
        source = bound.to_code(bound.config_spec.default_config())
    actual = _execute_direct_scan_program(source, (x, reverse))
    expected = _ordinary_scan_expected(x, "prod", reverse) + _ordinary_scan_expected(
        x, "sum", reverse
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_bounded_scan(
    x: torch.Tensor, begin: hl.constexpr, end: hl.constexpr, reverse: hl.constexpr
):
    out = torch.empty((x.size(0), end - begin), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        for col in hl.tile(begin, end, block_size=8):
            out[row, col.index - begin] = hl.cumprod(
                x[row, col], dim=-1, reverse=reverse
            )
    return out


@pytest.mark.parametrize(
    "begin,end,width", [(3, 20, 25), (3, 19, 25), (0, 17, 25), (3, 20, 20)]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.usefixtures("_serial_scan_fallback")
def test_ordinary_bounded_scan_generated_values(begin, end, width, reverse):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    x = ((torch.arange(3 * width).reshape(3, width) % 3) + 1).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_ordinary_bounded_scan, (x, begin, end, reverse))
        source = bound.to_code(bound.config_spec.default_config())
    actual = _execute_direct_scan_program(source, (x, begin, end, reverse))
    expected = _ordinary_scan_expected(x[:, begin:end], "prod", reverse)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
@onlyBackends(["cute"])
def test_ordinary_bounded_scan_native(reverse):
    x = ((torch.arange(3 * 25, device=DEVICE).reshape(3, 25) % 3) + 1).float()
    before = x.clone()
    torch.testing.assert_close(
        _ordinary_bounded_scan(x, 3, 20, reverse),
        _ordinary_scan_expected(x[:, 3:20], "prod", reverse),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_grid_bounded_scan(
    x: torch.Tensor,
    begin: int,
    end: int,
    reverse: hl.constexpr,
):
    out = torch.empty((x.size(0), end - begin), dtype=x.dtype, device=x.device)
    for row, col in hl.tile([0, begin], [x.size(0), end], block_size=[2, 8]):
        out[row, col.index - begin] = hl.cumprod(x[row, col], dim=-1, reverse=reverse)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_first_axis_bounded_scan(
    x: torch.Tensor,
    begin: int,
    end: int,
    reverse: hl.constexpr,
):
    out = torch.empty((end - begin, x.size(1)), dtype=x.dtype, device=x.device)
    for row in hl.tile(begin, end, block_size=8):
        for col in hl.tile(x.size(1), block_size=2):
            out[row.index - begin, col] = hl.cumprod(
                x[row, col], dim=0, reverse=reverse
            )
    return out


@pytest.mark.parametrize("first_axis", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.usefixtures("_serial_scan_fallback")
def test_ordinary_grid_scan_generated_values(first_axis, reverse):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    x = ((torch.arange(3 * 25).reshape(3, 25) % 3) + 1).float()
    if first_axis:
        x = x.t().contiguous()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        kernel = (
            _ordinary_first_axis_bounded_scan
            if first_axis
            else _ordinary_grid_bounded_scan
        )
        bound = _cpu_bind(kernel, (x, 3, 20, reverse))
        source = bound.to_code(bound.config_spec.default_config())
    # Reuse the exact generated host wrapper with different runtime bounds.
    for begin, end in [(3, 20), (1, 18), (2, 21)]:
        actual = _execute_direct_scan_program(source, (x, begin, end, reverse))
        selected = x[begin:end].t() if first_axis else x[:, begin:end]
        expected = _ordinary_scan_expected(selected, "prod", reverse)
        if first_axis:
            expected = expected.t()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("first_axis", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@onlyBackends(["cute"])
def test_ordinary_grid_scan_native(first_axis, reverse):
    x = ((torch.arange(3 * 25, device=DEVICE).reshape(3, 25) % 3) + 1).float()
    if first_axis:
        x = x.t().contiguous()
    before = x.clone()
    for begin, end in [(3, 20), (1, 18), (2, 21)]:
        kernel = (
            _ordinary_first_axis_bounded_scan
            if first_axis
            else _ordinary_grid_bounded_scan
        )
        actual = kernel(x, begin, end, reverse)
        selected = x[begin:end].t() if first_axis else x[:, begin:end]
        expected = _ordinary_scan_expected(selected, "prod", reverse)
        if first_axis:
            expected = expected.t()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["owner", "end"])
@pytest.mark.usefixtures("_serial_scan_fallback")
def test_ordinary_grid_scan_requires_logical_owner(monkeypatch, missing):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    from helion._compiler.cute.cute_reshape import _resolve_dim_block_id
    import helion._compiler.cute.scan_ops as scan_ops
    from helion.language.memory_ops import _cute_remap_block_id

    original = scan_ops._cute_codegen_serial_scan

    def without_bound(state, helper, inputs, dim, reverse):
        val = inputs[0].meta["val"]
        block = _resolve_dim_block_id(state.codegen, val, dim % val.ndim)
        assert block is not None
        active = _cute_remap_block_id(state, block)
        loops = state.codegen.active_device_loops.get(active)
        owner = loops[-1] if loops else state.codegen.current_grid_state
        assert owner is not None and active in owner.block_id_to_info
        if missing == "owner":
            owner.block_id_to_info.pop(active)
        else:
            owner.block_id_to_info[active].grid_end_expr = None
        return original(state, helper, inputs, dim, reverse)

    x = torch.ones(3, 25)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_ordinary_grid_bounded_scan, (x, 3, 20, True))
        # Initialize lazy backend registration before replacing its emitter.
        bound.to_code(bound.config_spec.default_config())
        original = scan_ops._cute_codegen_serial_scan
        monkeypatch.setattr(scan_ops, "_cute_codegen_serial_scan", without_bound)
        with pytest.raises(
            helion.exc.BackendUnsupported, match="logical tile " + missing
        ):
            bound.to_code(bound.config_spec.default_config())


def test_ordinary_default_scan_keeps_parallel_dispatch(monkeypatch):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    from helion._compiler.backend_registry import repair_backend_codegen
    from helion._compiler.cute import scan_ops

    repair_backend_codegen("cute")
    original = scan_ops._cute_try_parallel_scan
    results = []

    def observe(*args):
        result = original(*args)
        results.append(result is not None)
        return result

    monkeypatch.setattr(scan_ops, "_cute_try_parallel_scan", observe)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_ordinary_tiled_scan, (torch.ones(3, 17), "sum", False))
        bound.to_code(bound.config_spec.default_config())
    assert results and all(results)


@pytest.mark.parametrize("extent", [17, "chunk_size"])
@pytest.mark.parametrize("reverse", [False, True])
def test_serial_scan_position_handles_symbolic_extent(extent, reverse):
    from types import SimpleNamespace

    from helion._compiler.cute.scan_ops import _cute_serial_scan_position

    state = SimpleNamespace(device_function=SimpleNamespace(new_var=lambda name: name))
    position, lines = _cute_serial_scan_position(state, "step", extent, reverse)
    namespace = {"cutlass": SimpleNamespace(Int32=int), "chunk_size": 17}
    actual = []
    for step in range(17):
        namespace["step"] = step
        exec("\n".join(line.strip() for line in lines), namespace)
        actual.append(namespace[position])
    assert actual == list(range(16, -1, -1) if reverse else range(17))


@helion.kernel(backend="cute", autotune_effort="none")
def _singleton_inclusive_scan(
    x,
    y,
    dim: hl.constexpr,
    reverse: hl.constexpr,
    tuple_input: hl.constexpr,
    masked: hl.constexpr,
    scan: hl.constexpr,
):
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    axis = 1 if dim == 0 else 0
    for tile in hl.tile(x.size(axis), block_size=8):
        if dim == 0:
            if masked:
                first = hl.load(x, [slice(None), tile], extra_mask=y[:, tile] % 4 == 0)
                second = hl.load(y, [slice(None), tile], extra_mask=y[:, tile] % 4 == 0)
            else:
                first = x[:, tile]
                second = y[:, tile]
        else:
            if masked:
                first = hl.load(x, [tile, slice(None)], extra_mask=y[tile, :] % 4 == 0)
                second = hl.load(y, [tile, slice(None)], extra_mask=y[tile, :] % 4 == 0)
            else:
                first = x[tile, :]
                second = y[tile, :]
        if scan:
            if tuple_input:
                first, second = hl.associative_scan(
                    _scan_product_and_sum, (first, second), dim=dim, reverse=reverse
                )
            else:
                first = hl.associative_scan(
                    torch.minimum, first, dim=dim, reverse=reverse
                )
        if dim == 0:
            out_x[:, tile] = first
            out_y[:, tile] = second
        else:
            out_x[tile, :] = first
            out_y[tile, :] = second
    return out_x, out_y


def _execute_singleton_scan_program(source, inputs):
    # Reuse the existing generated-thread address/unique-write model for a
    # multi-output wrapper, retaining the original tensor storage and dtypes.
    tree = ast.parse(source)
    wrapper = next(
        node for node in reversed(tree.body) if isinstance(node, ast.FunctionDef)
    )
    result = next(node for node in wrapper.body if isinstance(node, ast.Return))
    result.value = ast.Call(
        func=ast.Name(id="_ScanOutputs", ctx=ast.Load()),
        args=[result.value],
        keywords=[],
    )
    tree.body.insert(
        0,
        ast.parse(
            "class _ScanOutputs(tuple):\n    def numel(self):\n        return sum(value.numel() for value in self)\n"
        ).body[0],
    )
    return _execute_direct_scan_program(
        ast.unparse(ast.fix_missing_locations(tree)), inputs
    )


def _singleton_scan_data(dim, device="cpu"):
    # Strides, signed zero, quiet NaN payloads and exact wide integers survive
    # the singleton identity; no arithmetic on these values is required.
    bits = torch.tensor(
        [
            -2147483648,
            0,
            1,
            0x7FC12345,
            0x7F800000,
            -8388608,
            0x3F800000,
            0x00800000,
            0x3EAAAAAB,
            0x40000000,
            0x40400000,
            0x40800000,
            0x40A00000,
            0x40C00000,
            0x40E00000,
            0x41000000,
            0x41100000,
        ],
        dtype=torch.int32,
        device=device,
    )
    backing = torch.empty((17, 2), dtype=torch.float32, device=device)
    backing[:, 0] = bits.view(torch.float32)
    backing[:, 1] = 123
    x = backing[:, :1]
    y = (torch.arange(34, device=device, dtype=torch.int64) + (1 << 60)).reshape(17, 2)[
        :, :1
    ]
    if dim == 0:
        x, y = x.t(), y.t()
    return x, y


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tuple_input", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_singleton_scan_generated_identity(dim, reverse, tuple_input, masked):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    x, y = _singleton_scan_data(dim)
    before = (x.contiguous().view(torch.int32).clone(), y.clone())
    args = (x, y, dim, reverse, tuple_input, masked, True)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_singleton_inclusive_scan, args)
        source = bound.to_code(bound.config_spec.default_config())
    actual = _execute_singleton_scan_program(source, args)
    expected_x, expected_y = x.clone(), y.clone()
    if masked:
        expected_x.reshape(-1)[1::2] = 0
        expected_y.reshape(-1)[1::2] = 0
    torch.testing.assert_close(
        actual[0].contiguous().view(torch.int32),
        expected_x.contiguous().view(torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual[1], expected_y, rtol=0, atol=0)
    torch.testing.assert_close(
        x.contiguous().view(torch.int32), before[0], rtol=0, atol=0
    )
    torch.testing.assert_close(y, before[1], rtol=0, atol=0)
    assert "scan_acc" not in source
    assert "scan_initialized" not in source


@pytest.mark.parametrize(
    "dim,reverse,tuple_input,masked",
    [(0, True, True, False), (-1, False, False, True), (-1, True, True, False)],
)
@onlyBackends(["cute"])
def test_singleton_scan_native_identity(dim, reverse, tuple_input, masked):
    x, y = _singleton_scan_data(dim, DEVICE)
    x_before, y_before = x.contiguous().view(torch.int32).clone(), y.clone()
    _code, actual = code_and_output(
        _singleton_inclusive_scan, (x, y, dim, reverse, tuple_input, masked, True)
    )
    expected_x, expected_y = x.clone(), y.clone()
    if masked:
        expected_x.reshape(-1)[1::2] = 0
        expected_y.reshape(-1)[1::2] = 0
    torch.testing.assert_close(
        actual[0].contiguous().view(torch.int32),
        expected_x.contiguous().view(torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual[1], expected_y, rtol=0, atol=0)
    torch.testing.assert_close(
        x.contiguous().view(torch.int32), x_before, rtol=0, atol=0
    )
    torch.testing.assert_close(y, y_before, rtol=0, atol=0)


@pytest.mark.usefixtures("_serial_scan_fallback")
def test_singleton_scan_rebind_does_not_specialize_shape_hints():
    from unittest.mock import patch

    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    kernel = helion.kernel(
        _singleton_inclusive_scan.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    storage = torch.arange(17 * 16).reshape(17, 16).float()
    integer = (torch.arange(17 * 16, dtype=torch.int64) + (1 << 60)).reshape(17, 16)
    bindings = {}
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        for width in (1, 4, 1):
            x, y = storage[:, :width], integer[:, :width]
            args = (x, y, -1, False, False, False, True)
            bound = kernel.bind(args)
            if width in bindings:
                assert bound is bindings[width]
            bindings[width] = bound
            source = bound.to_code(bound.config_spec.default_config())
            actual = _execute_singleton_scan_program(source, args)
            torch.testing.assert_close(actual[0], x.cummin(-1).values, rtol=0, atol=0)
            torch.testing.assert_close(actual[1], y, rtol=0, atol=0)
            assert ("scan_acc" in source) == (width != 1)
    assert bindings[1] is not bindings[4]


@onlyBackends(["cute"])
def test_singleton_scan_native_dtype_payloads():
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        x = torch.tensor(
            [-0.0, float("inf"), float("-inf"), float("nan"), 1 + 2**-40],
            dtype=dtype,
            device=DEVICE,
        )[:, None]
        y = (torch.arange(5, device=DEVICE, dtype=torch.int64) + (1 << 60))[:, None]
        bits_dtype = torch.int64 if dtype == torch.float64 else torch.int16
        before = x.contiguous().view(bits_dtype).clone()
        _code, actual = code_and_output(
            _singleton_inclusive_scan, (x, y, -1, True, True, False, True)
        )
        assert actual[0].dtype == dtype
        torch.testing.assert_close(
            actual[0].contiguous().view(bits_dtype), before, rtol=0, atol=0
        )
        torch.testing.assert_close(actual[1], y, rtol=0, atol=0)
        torch.testing.assert_close(
            x.contiguous().view(bits_dtype), before, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
