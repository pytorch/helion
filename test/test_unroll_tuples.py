from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel(autotune_effort="none")
def kernel_tuple_addition(
    a_shared_tuple: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Basic test: iterate over a tuple of tensors and sum them."""
    out = torch.empty_like(a_shared_tuple[0])
    for tile_n in hl.tile(out.size(0)):
        acc = torch.zeros([tile_n], dtype=torch.float32, device=out.device)
        for a_tensor in a_shared_tuple:
            acc += a_tensor[tile_n]
        out[tile_n] = acc
    return out


@helion.kernel(autotune_effort="none")
def kernel_tuple_with_scaling(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tensor3: torch.Tensor,
    scale1: float,
    scale2: float,
    scale3: float,
) -> torch.Tensor:
    """Test iteration over tensors with corresponding scalar multipliers."""
    tensors = (tensor1, tensor2, tensor3)
    scales = (scale1, scale2, scale3)
    output = torch.zeros_like(tensor1)
    for tile_idx in hl.tile(output.size(0)):
        temp = torch.zeros([tile_idx], dtype=torch.float32, device=output.device)
        for tensor, scale in zip(tensors, scales, strict=True):
            temp += tensor[tile_idx] * scale
        output[tile_idx] = temp
    return output


@helion.kernel(autotune_effort="none")
def kernel_nested_tuple_iteration(
    a_tuple: tuple[torch.Tensor, torch.Tensor],
    b_tuple: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test nested iteration over multiple tuples."""
    result = torch.zeros_like(a_tuple[0])
    for tile_idx in hl.tile(result.size(0)):
        temp = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)

        for a_tensor in a_tuple:
            temp += a_tensor[tile_idx]

        for b_tensor in b_tuple:
            temp *= b_tensor[tile_idx]

        result[tile_idx] = temp
    return result


@helion.kernel(autotune_effort="none")
def kernel_constants_iteration(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test iteration over a tuple/list of constants."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over constants
        for multiplier in (1, 2, 3):
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_constants_iteration(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test iteration over a list of constants."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over constants in a list
        for multiplier in [0.5, 1.5, 2.5]:
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_zip_iteration(
    tensors_a: tuple[torch.Tensor, torch.Tensor],
    tensors_b: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test iteration over zip of tuples."""
    result = torch.zeros_like(tensors_a[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over zip of tensors
        for a_tensor, b_tensor in zip(tensors_a, tensors_b, strict=False):
            acc += a_tensor[tile_idx] * b_tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_static_range_iteration(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test iteration using hl.static_range."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Use static_range for unrolled loop
        for i in hl.static_range(4):
            acc += x[tile_idx] * (i + 1)
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_static_range_with_start(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test static_range with start parameter."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Use static_range(start, end)
        for i in hl.static_range(2, 5):
            acc += x[tile_idx] * i
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_mixed_constants_and_tensors(
    tensors: tuple[torch.Tensor, torch.Tensor],
    constants: tuple[int, int],
) -> torch.Tensor:
    """Test mixed iteration over both tensors and constants."""
    result = torch.zeros_like(tensors[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)

        # First, iterate over tensors
        for tensor in tensors:
            acc += tensor[tile_idx]

        # Then, iterate over constants and multiply
        for constant in constants:
            acc *= constant

        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_enumerate_iteration(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test iteration using enumerate over tensors."""
    result = torch.zeros_like(tensors[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate with enumerate to get index and tensor
        for i, tensor in enumerate(tensors):
            acc += tensor[tile_idx] * (i + 1)  # Weight by index + 1
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_enumerate_with_start(
    tensors: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test enumerate with custom start value."""
    result = torch.zeros_like(tensors[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Enumerate starting from 5
        for i, tensor in enumerate(tensors, start=5):
            acc += tensor[tile_idx] * i
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_enumerate_constants(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test enumerate over constants."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Enumerate over constant values
        for i, multiplier in enumerate((2, 3, 4)):
            acc += x[tile_idx] * multiplier * i
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_simple_list_comprehension(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test simple list comprehension with constants."""
    result = torch.zeros_like(x)
    multipliers = [m * 2 for m in (1, 2, 3)]
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for multiplier in multipliers:
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_with_function(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test list comprehension with expressions."""
    result = torch.zeros_like(x)

    # Test list comprehension with more complex expressions
    squared_values = [i * i for i in (1, 2, 3)]
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for value in squared_values:
            acc += x[tile_idx] * value
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_with_tensors(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test list comprehension that produces a list of tensors."""
    result = torch.zeros_like(tensors[0])

    # This should work - creating a list of tensor references
    tensor_list = list(tensors)

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for tensor in tensor_list:
            acc += tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_nested_list_comprehension(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test nested list comprehension (flattened)."""
    result = torch.zeros_like(x)

    # Create pairs manually first since nested comprehensions with multiple generators
    # are not supported yet
    base_pairs = ((1, 3), (1, 4), (2, 3), (2, 4))
    pairs = list(base_pairs)

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for i, j in pairs:
            acc += x[tile_idx] * (i + j)
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_with_tuple_unrolling(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test interaction between list comprehension and tuple unrolling."""
    result = torch.zeros_like(tensors[0])

    # Create scaled versions of tensors using list comprehension
    scales = [0.5, 1.0, 1.5]
    scaled_tensors = [t * scale for t, scale in zip(tensors, scales, strict=False)]

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over the scaled tensors (both list comp and tuple unrolling)
        for scaled_tensor in scaled_tensors:
            acc += scaled_tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_host_and_device(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test list comprehension that works in both host and device code."""
    result = torch.zeros_like(x)

    # Host code - create list comprehension
    host_multipliers = [i * 2 for i in (1, 2, 3, 4)]

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)

        # Device code - create list comprehension (should be unrolled)
        device_multipliers = [i + 1 for i in (0, 1, 2)]

        # Use both host and device comprehensions
        for host_mult in host_multipliers:
            for device_mult in device_multipliers:
                acc += x[tile_idx] * host_mult * device_mult

        result[tile_idx] = acc
    return result


class TestUnrollTuples(RefEagerTestBase, TestCase):
    def test_basic_tuple_addition(self):
        """Test basic iteration over tuple of tensors with addition."""
        size = (32,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tuple_arg = (tensor1, tensor2, tensor3)

        code, result = code_and_output(kernel_tuple_addition, (tuple_arg,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness
        expected = tensor1 + tensor2 + tensor3
        torch.testing.assert_close(result, expected)

    def test_tuple_with_scaling_factors(self):
        """Test iteration with corresponding scalar values."""
        size = (48,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        scale1, scale2, scale3 = 2.0, 0.5, 1.5

        code, result = code_and_output(
            kernel_tuple_with_scaling,
            (tensor1, tensor2, tensor3, scale1, scale2, scale3),
        )

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness
        expected = tensor1 * scale1 + tensor2 * scale2 + tensor3 * scale3
        torch.testing.assert_close(result, expected)

    def test_nested_tuple_iteration(self):
        """Test nested loops over multiple tuples."""
        size = (40,)
        a1 = torch.randn(size, device=DEVICE)
        a2 = torch.randn(size, device=DEVICE)
        b1 = torch.randn(size, device=DEVICE) + 1.0  # Avoid zeros for multiplication
        b2 = torch.randn(size, device=DEVICE) + 1.0

        a_tuple = (a1, a2)
        b_tuple = (b1, b2)

        code, result = code_and_output(
            kernel_nested_tuple_iteration, (a_tuple, b_tuple)
        )

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness
        temp = a1 + a2
        expected = temp * b1 * b2
        torch.testing.assert_close(result, expected)

    def test_single_element_tuple(self):
        """Test with single-element tuple."""
        size = (16,)
        tensor = torch.randn(size, device=DEVICE)

        tuple_arg = (tensor,)

        code, result = code_and_output(kernel_tuple_addition, (tuple_arg,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should just copy the tensor
        torch.testing.assert_close(result, tensor)

    def test_constants_iteration(self):
        """Test iteration over tuple of constants."""
        size = (24,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_constants_iteration, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x * (1 + 2 + 3) = x * 6
        expected = x * 6
        torch.testing.assert_close(result, expected)

    def test_list_constants_iteration(self):
        """Test iteration over list of constants."""
        size = (20,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_list_constants_iteration, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x * (0.5 + 1.5 + 2.5) = x * 4.5
        expected = x * 4.5
        torch.testing.assert_close(result, expected)

    def test_zip_iteration(self):
        """Test iteration over zip of tuples."""
        # Create one reference tensor and use it to create others with same size
        reference = torch.randn((36,), device=DEVICE)
        a1 = torch.randn_like(reference)
        a2 = torch.randn_like(reference)
        b1 = torch.randn_like(reference)
        b2 = torch.randn_like(reference)

        tensors_a = (a1, a2)
        tensors_b = (b1, b2)

        code, result = code_and_output(kernel_zip_iteration, (tensors_a, tensors_b))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be a1*b1 + a2*b2
        expected = a1 * b1 + a2 * b2
        torch.testing.assert_close(result, expected)

    def test_static_range_iteration(self):
        """Test iteration using hl.static_range."""
        size = (28,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_static_range_iteration, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x * (1 + 2 + 3 + 4) = x * 10
        expected = x * 10
        torch.testing.assert_close(result, expected)

    def test_static_range_with_start(self):
        """Test static_range with start parameter."""
        size = (18,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_static_range_with_start, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x * (2 + 3 + 4) = x * 9
        expected = x * 9
        torch.testing.assert_close(result, expected)

    def test_mixed_constants_and_tensors(self):
        """Test mixed iteration over both tensors and constants."""
        size = (22,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2)
        constants = (2, 3)

        code, result = code_and_output(
            kernel_mixed_constants_and_tensors, (tensors, constants)
        )

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be (tensor1 + tensor2) * 2 * 3
        expected = (tensor1 + tensor2) * 2 * 3
        torch.testing.assert_close(result, expected)

    def test_enumerate_iteration(self):
        """Test iteration using enumerate over tensors."""
        size = (24,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(kernel_enumerate_iteration, (tensors,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be tensor1*1 + tensor2*2 + tensor3*3
        expected = tensor1 * 1 + tensor2 * 2 + tensor3 * 3
        torch.testing.assert_close(result, expected)

    def test_enumerate_with_start(self):
        """Test enumerate with custom start value."""
        size = (18,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2)

        code, result = code_and_output(kernel_enumerate_with_start, (tensors,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be tensor1*5 + tensor2*6 (start=5)
        expected = tensor1 * 5 + tensor2 * 6
        torch.testing.assert_close(result, expected)

    def test_enumerate_constants(self):
        """Test enumerate over constants."""
        size = (20,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_enumerate_constants, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x*(2*0 + 3*1 + 4*2) = x*(0 + 3 + 8) = x*11
        expected = x * 11
        torch.testing.assert_close(result, expected)

    def test_simple_list_comprehension(self):
        """Test simple list comprehension with constants."""
        size = (16,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_simple_list_comprehension, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x * (2 + 4 + 6) = x * 12
        expected = x * 12
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_with_function(self):
        """Test list comprehension with expressions."""
        size = (14,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_list_comprehension_with_function, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be x * (1 + 4 + 9) = x * 14
        expected = x * 14
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_with_tensors(self):
        """Test list comprehension that produces a list of tensors."""
        size = (18,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(
            kernel_list_comprehension_with_tensors, (tensors,)
        )

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be sum of all tensors
        expected = tensor1 + tensor2 + tensor3
        torch.testing.assert_close(result, expected)

    def test_nested_list_comprehension(self):
        """Test nested list comprehension."""
        size = (12,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_nested_list_comprehension, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - pairs are (1,3), (1,4), (2,3), (2,4)
        # should be x * (4 + 5 + 5 + 6) = x * 20
        expected = x * 20
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_with_tuple_unrolling(self):
        """Test interaction between list comprehension and tuple unrolling."""
        size = (22,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(
            kernel_list_comprehension_with_tuple_unrolling, (tensors,)
        )

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness - should be tensor1*0.5 + tensor2*1.0 + tensor3*1.5
        expected = tensor1 * 0.5 + tensor2 * 1.0 + tensor3 * 1.5
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_host_and_device(self):
        """Test list comprehension that works in both host and device code."""
        size = (26,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_list_comprehension_host_and_device, (x,))

        # Validate generated code
        self.assertExpectedJournal(code)

        # Test correctness
        # host_multipliers = [2, 4, 6, 8]
        # device_multipliers = [1, 2, 3]
        # Total should be x * (2*1 + 2*2 + 2*3 + 4*1 + 4*2 + 4*3 + 6*1 + 6*2 + 6*3 + 8*1 + 8*2 + 8*3)
        # = x * (2 + 4 + 6 + 4 + 8 + 12 + 6 + 12 + 18 + 8 + 16 + 24) = x * 120
        expected = x * 120
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
