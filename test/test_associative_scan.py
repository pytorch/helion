from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


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


@helion.jit
def jit_add_combine_fn(x, y):
    """Addition combine function with @helion.jit decorator (should be ignored)."""
    return x + y


class TestAssociativeScan(RefEagerTestBase, TestCase):
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
        self.assertExpectedJournal(code)

        # Test the actual scan operation
        expected = torch.tensor(
            [[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0], [9.0, 19.0, 30.0, 42.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains the correct helper function
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
        self.assertExpectedJournal(code)

        # Expected prefix maximum
        expected = torch.tensor(
            [[1.0, 5.0, 5.0, 8.0, 8.0], [7.0, 7.0, 9.0, 9.0, 9.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains maximum operation (either tl.maximum or triton_helpers.maximum)
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
        self.assertExpectedJournal(code)

        # Expected prefix product
        expected = torch.tensor(
            [[1.0, 2.0, 6.0, 24.0], [2.0, 1.0, 3.0, 6.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains multiplication
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
        self.assertExpectedJournal(code)

        # Expected prefix minimum
        expected = torch.tensor(
            [[5.0, 2.0, 2.0, 1.0, 1.0], [3.0, 3.0, 1.0, 1.0, 1.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains minimum operation (either tl.minimum or triton_helpers.minimum)
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
        self.assertExpectedJournal(code)

        # Test the sum result
        expected_sum = torch.tensor([[1.0, 4.0, 6.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected_sum)

        # Verify multiple helper functions are generated
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

        self.assertExpectedJournal(code)
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
            with self.subTest(dtype=dtype):

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

                self.assertExpectedJournal(code)
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

                self.assertExpectedJournal(code)
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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)
        expected = torch.tensor([[5.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Two elements
        x_two = torch.tensor([[3.0, 7.0]], device=DEVICE)
        code, result = code_and_output(test_single_element, (x_two,))
        self.assertExpectedJournal(code)
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

        self.assertExpectedJournal(code)
        # Verify correctness on large scale
        expected = torch.cumsum(x, dim=1)
        # Use relaxed tolerance for large tensors due to accumulated floating-point errors
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify output properties
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)

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
        self.assertExpectedJournal(code)

        # Expected prefix sum results
        expected = torch.tensor(
            [[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code contains the proper combine function and associative scan
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
        self.assertExpectedJournal(code)

        # Verify the result is correct
        expected = torch.tensor([[1.0, 3.0, 6.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Check essential code structure
        self.assertIn("@triton.jit", code)
        self.assertIn("def add_combine_fn_", code)
        self.assertIn("tl.associative_scan", code)
        self.assertIn("return", code)

        # Verify no placeholders remain
        self.assertNotIn("TODO", code)
        self.assertNotIn("placeholder", code)

    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with nested @helion.jit is not supported by ref eager mode yet"
    )
    def test_associative_scan_jit_decorator_ignored(self):
        """Test that @helion.jit decorator on combine functions is ignored."""

        @helion.kernel(autotune_effort="none")
        def test_jit_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.associative_scan(jit_add_combine_fn, row_data, dim=1)
            return result

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)
        code, result = code_and_output(test_jit_kernel, (x,))
        self.assertExpectedJournal(code)

        # Expected prefix sum results
        expected = torch.tensor([[1.0, 3.0, 6.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code contains the proper combine function and associative scan
        self.assertIn("def jit_add_combine_fn_", code)
        self.assertIn("tl.associative_scan", code)
        self.assertIn("param_0 + param_1", code)
        # Verify @helion.jit decorator doesn't appear in generated code
        self.assertNotIn("@helion.jit", code)

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
                    helion_combine_fn, input_tuple, 0
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
        self.assertExpectedJournal(code)

        # Expected: cumulative sum for each position
        expected = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify the generated code structure
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
                    segmented_combine_fn, (vals, idxs), 0
                )

                output[tile_e, tile_f] = out_vals

            return output

        # Test segmented reduction
        E, C = 6, 3
        # Segments: [0,0], [1,1,1], [2] - three segments of sizes 2, 3, 1
        indices = torch.tensor([0, 0, 1, 1, 1, 2], device=DEVICE)
        input_data = torch.ones((E, C), device=DEVICE)

        code, result = code_and_output(segmented_scan_kernel, (indices, input_data))
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

        # Verify the result is correct (cumsum along dim=0)
        expected = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [6.0, 8.0, 10.0, 12.0]],
            device=DEVICE,
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains the helper function and associative scan
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
        self.assertExpectedJournal(code)

        # Expected cumulative sum
        expected = torch.tensor(
            [[1.0, 3.0, 6.0, 10.0], [5.0, 11.0, 18.0, 26.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains cumsum implementation
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
        self.assertExpectedJournal(code)

        # For reverse cumsum: [10, 9, 7, 4] (sum from right to left)
        expected = torch.tensor([[10.0, 9.0, 7.0, 4.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Verify reverse parameter is used
        self.assertIn("reverse=True", code)

    def test_cumsum_different_dtypes(self):
        """Test cumsum with different data types."""

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):

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

                self.assertExpectedJournal(code)
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
        self.assertExpectedJournal(code)

        # Expected cumulative product
        expected = torch.tensor(
            [[1.0, 2.0, 6.0, 24.0], [2.0, 1.0, 3.0, 6.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code contains cumprod implementation
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
        self.assertExpectedJournal(code)

        # For reverse cumprod: [24, 24, 12, 4] (product from right to left)
        expected = torch.tensor([[24.0, 24.0, 12.0, 4.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Verify reverse parameter is used
        self.assertIn("reverse=True", code)

    def test_cumprod_different_dtypes(self):
        """Test cumprod with different data types."""

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):

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

                self.assertExpectedJournal(code)
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
        self.assertExpectedJournal(code)

        # Test the sum result
        expected_sum = torch.tensor([[1.0, 3.0, 6.0, 10.0]], device=DEVICE)
        torch.testing.assert_close(result, expected_sum)

        # Verify both helper functions are generated
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
        self.assertExpectedJournal(code)

        # Expected: cumulative sum for each position
        expected = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]], device=DEVICE
        )
        torch.testing.assert_close(result, expected)

        # Verify the generated code structure
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
        self.assertExpectedJournal(code)

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
        self.assertIn("def argmax_combine_tuple_fn_", code)
        self.assertIn("tl.associative_scan", code)


if __name__ == "__main__":
    unittest.main()
