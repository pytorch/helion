from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


def add_combine_fn(x, y):
    """Simple addition combine function for sum reduction."""
    return x + y


def max_combine_fn(x, y):
    """Maximum combine function for max reduction."""
    return torch.maximum(x, y)


def mul_combine_fn(x, y):
    """Multiplication combine function for product reduction."""
    return x * y


def min_combine_fn(x, y):
    """Minimum combine function for min reduction."""
    return torch.minimum(x, y)


def tuple_add_combine_fn(left_tuple, right_tuple):
    """Tuple combine function for tuple reduction."""
    left_values, left_indices = left_tuple
    right_values, right_indices = right_tuple
    combined_values = left_values + right_values
    combined_indices = left_indices + right_indices
    return combined_values, combined_indices


def argmax_combine_fn(left_tuple, right_tuple):
    """Combine function for argmax: returns (value, index) of maximum element."""
    left_value, left_index = left_tuple
    right_value, right_index = right_tuple

    # If right value is greater, take right; otherwise keep left
    take_right = right_value > left_value
    max_value = torch.where(take_right, right_value, left_value)
    max_index = torch.where(take_right, right_index, left_index)

    return max_value, max_index


def tuple_add_combine_unpacked_fn(
    left_values, left_indices, right_values, right_indices
):
    """Tuple combine function for tuple reduction (unpacked format)."""
    combined_values = left_values + right_values
    combined_indices = left_indices + right_indices
    return combined_values, combined_indices


def argmax_combine_unpacked_fn(left_value, left_index, right_value, right_index):
    """Combine function for argmax (unpacked format): returns (value, index) of maximum element."""
    # If right value is greater, take right; otherwise keep left
    take_right = right_value > left_value
    max_value = torch.where(take_right, right_value, left_value)
    max_index = torch.where(take_right, right_index, left_index)

    return max_value, max_index


@helion.jit
def jit_add_combine_fn(x, y):
    """Addition combine function with @helion.jit decorator (should be ignored)."""
    return x + y


class TestReduce(RefEagerTestBase, TestCase):
    def test_reduce_basic_sum(self):
        """Test basic reduce functionality with sum reduction along a dimension."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]  # Shape: [TILE_SIZE, seq_len]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_kernel, (x,))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected = torch.tensor([10.0, 26.0, 42.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains triton reduce calls
        self.assertIn("tl.reduce", code)
        self.assertIn("add_combine_fn_", code)

    def test_reduce_max(self):
        """Test reduce with maximum operation."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_max_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(max_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0], [9.0, 12.0, 10.0, 11.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_max_kernel, (x,))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected = torch.tensor([4.0, 8.0, 12.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_with_keep_dims(self):
        """Test reduce with keep_dims=True."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_keep_dims_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0), 1], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.reduce(
                    add_combine_fn, row_data, dim=1, keep_dims=True
                )
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_keep_dims_kernel, (x,))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected = torch.tensor([[10.0], [26.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        # Check that keep_dims=True is in the generated code
        self.assertIn("keep_dims=True", code)

    def test_reduce_all_dims(self):
        """Test reduce with dim=None (reduce all dimensions) - SKIP for now."""
        # Skip this test for now - dim=None has complex implementation issues
        # with symbolic shapes that require more work to fix properly
        self.skipTest("dim=None reduction requires more complex implementation")

    def test_reduce_min(self):
        """Test reduce with minimum operation."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_min_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(min_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[4.0, 1.0, 3.0, 2.0], [8.0, 5.0, 7.0, 6.0], [12.0, 9.0, 11.0, 10.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_min_kernel, (x,))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected = torch.tensor([1.0, 5.0, 9.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_product(self):
        """Test reduce with multiplication operation using other=1."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_product_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(mul_combine_fn, row_data, dim=1, other=1.0)
            return result

        # Create test input with non-power-2 size (3 elements)
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 1.0, 5.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_product_kernel, (x,))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected = torch.tensor([6.0, 24.0, 5.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_jit_combine_fn(self):
        """Test reduce with @helion.jit decorated combine function."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_jit_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(jit_add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_jit_kernel, (x,))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected = torch.tensor([10.0, 26.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_tuple_input(self):
        """Test reduce with tuple input."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_tuple_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_x = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            result_y = torch.empty([y.size(0)], dtype=y.dtype, device=y.device)

            for i in hl.tile(x.size(0)):
                row_x = x[i, :]
                row_y = y[i, :]
                input_tuple = (row_x, row_y)
                reduced_tuple = hl.reduce(tuple_add_combine_fn, input_tuple, dim=1)
                result_x[i], result_y[i] = reduced_tuple

            return result_x, result_y

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, (result_x, result_y) = code_and_output(test_reduce_tuple_kernel, (x, y))
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected_x = torch.tensor([6.0, 15.0], device=DEVICE)
        expected_y = torch.tensor([3.0, 6.0], device=DEVICE)
        torch.testing.assert_close(result_x, expected_x)
        torch.testing.assert_close(result_y, expected_y)

    def test_reduce_different_dtypes(self):
        """Test reduce with different tensor dtypes."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_int_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        # Create test input with integer dtype
        x = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_int_kernel, (x,))

        self.assertExpectedJournal(code)
        # Test the actual reduce operation
        expected = torch.tensor([10, 26], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

    def test_reduce_tuple_unpacking_oneline(self):
        """Test tuple unpacking in one line: a, b = hl.reduce(...)"""

        @helion.kernel(autotune_effort="none")
        def test_tuple_oneline_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]  # Shape: [TILE_SIZE, seq_len]
                row_indices = indices[i, :]  # Shape: [TILE_SIZE, seq_len]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test one-line tuple unpacking - reduce 2D tensors on dim=1 to get 1D results
                max_value, max_index = hl.reduce(
                    argmax_combine_fn, value_index_pairs, dim=1
                )

                # max_index is now 1D tensor that can be assigned directly
                result[i] = max_index

            return result

        # Create test input with known argmax positions
        values = torch.tensor(
            [
                [1.0, 5.0, 3.0, 2.0],  # max=5.0 at index 1
                [9.0, 2.0, 7.0, 8.0],  # max=9.0 at index 0
                [4.0, 6.0, 8.0, 3.0],  # max=8.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_tuple_oneline_kernel, (values, indices))
        self.assertExpectedJournal(code)

        # Test the actual argmax operation
        expected = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)

        # Check that the generated code contains the expected elements
        self.assertIn("tl.reduce", code)
        self.assertIn("argmax_combine_fn_", code)

    def test_reduce_tuple_unpacking_twoline(self):
        """Test tuple unpacking in two lines: result = hl.reduce(...); a, b = result"""

        @helion.kernel(autotune_effort="none")
        def test_tuple_twoline_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]  # Shape: [TILE_SIZE, seq_len]
                row_indices = indices[i, :]  # Shape: [TILE_SIZE, seq_len]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test two-line tuple unpacking - reduce 2D tensors on dim=1 to get 1D results
                reduction_result = hl.reduce(
                    argmax_combine_fn, value_index_pairs, dim=1
                )
                max_value, max_index = reduction_result

                # max_index is now 1D tensor that can be assigned directly
                result[i] = max_index

            return result

        # Create test input with known argmax positions
        values = torch.tensor(
            [
                [1.0, 5.0, 3.0, 2.0],  # max=5.0 at index 1
                [9.0, 2.0, 7.0, 8.0],  # max=9.0 at index 0
                [4.0, 6.0, 8.0, 3.0],  # max=8.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_tuple_twoline_kernel, (values, indices))
        self.assertExpectedJournal(code)

        # Test the actual argmax operation
        expected = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)

        # Check that the generated code contains the expected elements
        self.assertIn("tl.reduce", code)
        self.assertIn("argmax_combine_fn_", code)

    def test_reduce_argmax_negative_values(self):
        """Test argmax with all negative values using other=(-inf, 0)."""

        @helion.kernel(autotune_effort="none")
        def test_argmax_negative_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]  # Shape: [TILE_SIZE, seq_len]
                row_indices = indices[i, :]  # Shape: [TILE_SIZE, seq_len]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test argmax with negative values - use -inf for values, 0 for indices
                max_value, max_index = hl.reduce(
                    argmax_combine_fn,
                    value_index_pairs,
                    dim=1,
                    other=(float("-inf"), 0),
                )

                # max_index is now 1D tensor that can be assigned directly
                result[i] = max_index

            return result

        # Create test input with all negative values
        values = torch.tensor(
            [
                [-5.0, -1.0, -3.0, -2.0],  # max=-1.0 at index 1
                [-9.0, -8.0, -7.0, -10.0],  # max=-7.0 at index 2
                [-4.0, -6.0, -2.0, -3.0],  # max=-2.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_argmax_negative_kernel, (values, indices))
        self.assertExpectedJournal(code)

        # Test the actual argmax operation
        expected = torch.tensor([1, 2, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)

        # Check that the generated code contains the expected elements
        self.assertIn("tl.reduce", code)
        self.assertIn("argmax_combine_fn_", code)

    def test_reduce_code_generation(self):
        """Test that reduce generates correct Triton code."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_codegen_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)

        # Test that the kernel compiles and generates expected code
        code, result = code_and_output(test_reduce_codegen_kernel, (x,))
        self.assertExpectedJournal(code)

        # Check that the generated code contains the expected elements
        self.assertIn("tl.reduce", code)
        self.assertIn("add_combine_fn_", code)
        self.assertIn("@triton.jit", code)

        # Verify correctness
        expected = torch.tensor([6.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_tuple_unpacked_format(self):
        """Test reduce with tuple input using unpacked format combine function."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_tuple_unpacked_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_x = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            result_y = torch.empty([y.size(0)], dtype=y.dtype, device=y.device)

            for i in hl.tile(x.size(0)):
                row_x = x[i, :]
                row_y = y[i, :]
                input_tuple = (row_x, row_y)
                reduced_tuple = hl.reduce(
                    tuple_add_combine_unpacked_fn, input_tuple, dim=1
                )
                result_x[i], result_y[i] = reduced_tuple

            return result_x, result_y

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, (result_x, result_y) = code_and_output(
            test_reduce_tuple_unpacked_kernel, (x, y)
        )
        self.assertExpectedJournal(code)

        # Test the actual reduce operation
        expected_x = torch.tensor([6.0, 15.0], device=DEVICE)
        expected_y = torch.tensor([3.0, 6.0], device=DEVICE)
        torch.testing.assert_close(result_x, expected_x)
        torch.testing.assert_close(result_y, expected_y)

    def test_reduce_argmax_unpacked_format(self):
        """Test argmax with unpacked format combine function."""

        @helion.kernel(autotune_effort="none")
        def test_argmax_unpacked_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]
                row_indices = indices[i, :]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test unpacked format combine function
                max_value, max_index = hl.reduce(
                    argmax_combine_unpacked_fn, value_index_pairs, dim=1
                )

                result[i] = max_index

            return result

        # Create test input with known argmax positions
        values = torch.tensor(
            [
                [1.0, 5.0, 3.0, 2.0],  # max=5.0 at index 1
                [9.0, 2.0, 7.0, 8.0],  # max=9.0 at index 0
                [4.0, 6.0, 8.0, 3.0],  # max=8.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_argmax_unpacked_kernel, (values, indices))
        self.assertExpectedJournal(code)

        # Test the actual argmax operation
        expected = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)


if __name__ == "__main__":
    unittest.main()
