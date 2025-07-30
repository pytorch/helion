from __future__ import annotations

import itertools
import unittest

import torch
import triton

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel(config=helion.Config(block_sizes=[32, 32, 32]), dot_precision="tf32")
def dot_kernel_acc_arg(
    x: torch.Tensor, y: torch.Tensor, acc_dtype: torch.dtype
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=acc_dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=acc_dtype)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(config=helion.Config(block_sizes=[32, 32, 32]), dot_precision="tf32")
def dot_kernel_no_acc_arg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    if x.dtype == torch.int8:
        acc_dtype = torch.int32
    else:
        acc_dtype = torch.float32
    out = torch.empty([m, n], dtype=acc_dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=acc_dtype)
        for tile_k in hl.tile(k):
            acc += hl.dot(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


# Define test parameters
INPUT_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
ACC_DTYPES = [None, torch.float16, torch.float32, torch.int32]
STATIC_SHAPES_OPTIONS = [True, False]

# Define expected failures
EXPECTED_FAILURES = {
    # int8 requires int32 accumulator
    (torch.int8, torch.int8, torch.float16),
    (torch.int8, torch.int8, torch.float32),
    # float16 accumulator only supported with float16 or fp8 inputs (Triton constraint)
    (torch.float32, torch.float32, torch.float16),
    (torch.bfloat16, torch.bfloat16, torch.float16),
    # int32 accumulator only supported for int8 inputs
    (torch.float16, torch.float16, torch.int32),
    (torch.float32, torch.float32, torch.int32),
    (torch.bfloat16, torch.bfloat16, torch.int32),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.int32),
    (torch.float8_e5m2, torch.float8_e5m2, torch.int32),
}


def make_test_function(input_dtype, acc_dtype, static_shapes_option):
    """Create a test function for a specific combination of parameters."""
    combo = (input_dtype, input_dtype, acc_dtype)

    def test_impl(self):
        # Skip FP8 tests if GPU doesn't support it
        if (
            input_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and torch.cuda.get_device_capability(0)[0] < 9
        ):
            self.skipTest(f"FP8 dtype {input_dtype} not supported on this GPU")

        # Create test tensors
        if input_dtype == torch.int8:
            x = torch.randint(-10, 10, (64, 64), device=DEVICE, dtype=input_dtype)
            y = torch.randint(-10, 10, (64, 64), device=DEVICE, dtype=input_dtype)
        elif input_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) * 0.5
            y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) * 0.5
            x = x.to(input_dtype)
            y = y.to(input_dtype)
        else:
            x = torch.randn(64, 64, device=DEVICE, dtype=input_dtype)
            y = torch.randn(64, 64, device=DEVICE, dtype=input_dtype)

        def run_kernel():
            if acc_dtype is None:
                dot_kernel_no_acc_arg._static_shapes = static_shapes_option
                return code_and_output(dot_kernel_no_acc_arg, (x, y))
            dot_kernel_acc_arg._static_shapes = static_shapes_option
            return code_and_output(dot_kernel_acc_arg, (x, y, acc_dtype))

        # Check if this combination should fail
        if combo in EXPECTED_FAILURES:
            # Use assertRaises for expected failures
            with self.assertRaises(
                (
                    triton.compiler.errors.CompilationError,
                    RuntimeError,
                    helion.exc.InternalError,
                    ValueError,
                    OSError,
                )
            ):
                code, result = run_kernel()
            return

        # Normal test execution for non-failing cases
        code, result = run_kernel()

        # Compute expected result based on accumulator dtype
        if input_dtype == torch.int8:
            expected = (x.cpu().to(torch.int32) @ y.cpu().to(torch.int32)).to(DEVICE)
        else:
            # For floating point, compute in float32 for accuracy
            x_f32 = x.to(torch.float32)
            y_f32 = y.to(torch.float32)
            expected = x_f32 @ y_f32

            # Convert expected to match kernel output dtype
            if acc_dtype == torch.float16:
                expected = expected.to(torch.float16)
            elif acc_dtype == torch.int32:
                expected = expected.to(torch.int32)
            # else: already float32 for acc_f32 or implicit float32 acc

        # Check result with appropriate tolerance
        if input_dtype == torch.int8:
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
        elif input_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # FP8 has lower precision, use higher tolerance
            torch.testing.assert_close(result, expected, atol=5e-3, rtol=0.5)
        elif input_dtype == torch.float16 and acc_dtype == torch.float16:
            # Use higher tolerance when accumulator is float16 due to precision limits
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=0.5)
        elif input_dtype == torch.float32:
            # Use higher tolerance for TF32 mode
            torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)
        else:
            torch.testing.assert_close(result, expected)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

    return test_impl


class TestDot(TestCase):
    pass


# Dynamically generate test methods
for input_dtype, acc_dtype, static_shapes_option in itertools.product(
    INPUT_DTYPES, ACC_DTYPES, STATIC_SHAPES_OPTIONS
):
    # Create test method name
    input_dtype_name = str(input_dtype).split(".")[-1]
    acc_dtype_name = "None" if acc_dtype is None else str(acc_dtype).split(".")[-1]
    static_shapes_name = "static_shape" if static_shapes_option else "dynamic_shape"
    test_name = (
        f"test_input_{input_dtype_name}_acc_{acc_dtype_name}_{static_shapes_name}"
    )

    # Create and add the test method
    _test_func = make_test_function(input_dtype, acc_dtype, static_shapes_option)
    _test_func.__name__ = test_name
    setattr(TestDot, test_name, _test_func)


if __name__ == "__main__":
    unittest.main()
