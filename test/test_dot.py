from __future__ import annotations

import itertools
import unittest

import torch
import triton

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
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
# With revised codegen (no fused acc when dtypes differ), many combinations are supported via
# separate addition. We keep only truly unsupported cases here.
EXPECTED_FAILURES = {
    # int8 requires int32 accumulator
    (torch.int8, torch.int8, torch.float16),
    (torch.int8, torch.int8, torch.float32),
    # int32 accumulation for floating inputs is not supported yet in our numeric checks
    (torch.float16, torch.float16, torch.int32),
    (torch.float32, torch.float32, torch.int32),
    (torch.bfloat16, torch.bfloat16, torch.int32),
}


def make_test_function(input_dtype, acc_dtype, static_shapes_option):
    """Create a test function for a specific combination of parameters."""
    combo = (input_dtype, input_dtype, acc_dtype)

    @skipIfRocm("Core dumps with rocm -- https://github.com/pytorch/helion/issues/445")
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
        elif input_dtype == torch.bfloat16 and acc_dtype == torch.float16:
            # bfloat16 inputs with float16 accumulation can be noisier
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=0.5)
        elif input_dtype == torch.float32:
            # Use higher tolerance for TF32 mode
            torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)
        else:
            torch.testing.assert_close(result, expected)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

    return test_impl


class TestDot(RefEagerTestBase, TestCase):
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_hl_dot_codegen_acc_differs_uses_addition(self):
        # Reuse existing kernel that calls hl.dot(..., acc=acc)
        input_dtype = torch.bfloat16
        acc_dtype = torch.float32
        x = torch.randn(64, 64, device=DEVICE, dtype=input_dtype)
        y = torch.randn(64, 64, device=DEVICE, dtype=input_dtype)
        code, out = code_and_output(dot_kernel_acc_arg, (x, y, acc_dtype))
        # Validate we use tl.dot and casting around accumulation (fused or separate-add)
        self.assertIn("tl.dot(", code)
        self.assertIn("tl.cast", code)

    # Note: numerical behavior for differing acc dtype is covered by existing dot tests; here we focus on codegen shape

    # torch.baddbmm codegen shape is covered indirectly by broader matmul tests; skipping a brittle code-inspection here

    @skipIfRefEager("Debug dtype codegen checks rely on compiled code")
    def test_baddbmm_pipeline_debug_dtype_asserts(self):
        # Reproduces scripts/repro512.py within the test suite and asserts
        # the kernel compiles and runs with debug dtype asserts enabled.
        @helion.kernel(
            use_default_config=True,
            static_shapes=True,
            dot_precision="tf32",
            debug_dtype_asserts=True,
        )
        def repro_baddbmm_kernel(
            q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
        ) -> torch.Tensor:
            # This kernel mirrors the pipeline from scripts/repro512.py and will
            # trigger dtype checks when HELION_DEBUG_DTYPE_ASSERTS is enabled.
            b_dim = hl.specialize(q_in.size(0))
            m_dim = hl.specialize(q_in.size(1))  # noqa: F841
            n_dim = hl.specialize(k_in.size(1))
            head_dim = hl.specialize(q_in.size(2))
            assert n_dim == v_in.size(1)
            assert head_dim == k_in.size(2) == v_in.size(2)

            q = q_in  # [B, M, H]
            k = k_in.transpose(1, 2)  # [B, H, N]
            v = v_in  # [B, N, H]

            out = torch.empty_like(q)
            # Single tile over full batch to avoid symbolic broadcasting in baddbmm
            for tile_b in hl.tile(b_dim, block_size=b_dim):
                qb = q[tile_b, :, :]
                kb = k[tile_b, :, :]
                vb = v[tile_b, :, :]
                qk = torch.bmm(qb, kb)  # [tile_b, M, N]
                p = torch.sigmoid(qk)
                p = qk * p
                acc0 = torch.zeros_like(qb, dtype=torch.float32)
                out[tile_b, :, :] = torch.baddbmm(acc0, p.to(vb.dtype), vb)
            return out

        B, M, N, H = 1, 64, 64, 64
        x_dtype = torch.bfloat16
        q = torch.randn(B, M, H, device=DEVICE, dtype=x_dtype)
        k = torch.randn(B, N, H, device=DEVICE, dtype=x_dtype)
        v = torch.randn(B, N, H, device=DEVICE, dtype=x_dtype)
        code, out = code_and_output(repro_baddbmm_kernel, (q, k, v))
        self.assertEqual(out.dtype, x_dtype)
        self.assertEqual(list(out.shape), [B, M, H])
        # Ensure debug assertions and safe sigmoid casting are present in codegen
        self.assertIn("tl.static_assert", code)
        self.assertIn("tl.sigmoid(tl.cast", code)


# Define ref mode test failures
REF_EAGER_TEST_FAILURES = {
    "test_input_float8_e5m2_acc_None_dynamic_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_None_static_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_float16_dynamic_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_float16_static_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_float32_dynamic_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_float32_static_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_int32_dynamic_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_float8_e5m2_acc_int32_static_shape": "Matmul with float8_e5m2 dtype not supported in ref eager mode",
    "test_input_int8_acc_None_dynamic_shape": "int8 @ int8 -> int32 is not supported in ref eager mode",
    "test_input_int8_acc_None_static_shape": "int8 @ int8 -> int32 is not supported in ref eager mode",
    "test_input_int8_acc_int32_dynamic_shape": "int8 @ int8 -> int32 is not supported in ref eager mode",
    "test_input_int8_acc_int32_static_shape": "int8 @ int8 -> int32 is not supported in ref eager mode",
}

# Define ref mode test failures for FP8 e4m3fn on GPUs with low compute capability (< 9.0)
REF_EAGER_TEST_FAILURES_FP8_E4M3FN_LOW_COMPUTE_CAP = {
    "test_input_float8_e4m3fn_acc_None_dynamic_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_None_static_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_float16_dynamic_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_float16_static_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_float32_dynamic_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_float32_static_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_int32_dynamic_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
    "test_input_float8_e4m3fn_acc_int32_static_shape": "Matmul with float8_e4m3fn dtype not supported on this GPU in ref eager mode",
}

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

    # Skip int accumulator with floating-point inputs — not a meaningful configuration
    if acc_dtype is torch.int32 and input_dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        _test_func = unittest.skip(
            "skip: int accumulator with float matmul is not supported"
        )(_test_func)

    # Apply skipIfRefEager decorator if needed
    if test_name in REF_EAGER_TEST_FAILURES:
        _test_func = skipIfRefEager(REF_EAGER_TEST_FAILURES[test_name])(_test_func)
    elif test_name in REF_EAGER_TEST_FAILURES_FP8_E4M3FN_LOW_COMPUTE_CAP:
        # For e4m3fn tests, only skip if GPU capability < 9
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9:
            _test_func = skipIfRefEager(
                REF_EAGER_TEST_FAILURES_FP8_E4M3FN_LOW_COMPUTE_CAP[test_name]
            )(_test_func)

    # Additional ref eager skips for unsupported accumulator/input combos
    if acc_dtype is torch.float16 and input_dtype in (
        torch.bfloat16,
        torch.float32,
    ):
        _test_func = skipIfRefEager(
            "float16 accumulator not supported for bf16/f32 in ref eager mode"
        )(_test_func)

    setattr(TestDot, test_name, _test_func)


if __name__ == "__main__":
    unittest.main()
