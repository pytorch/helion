from __future__ import annotations

import itertools
from typing import Callable
import unittest

import torch
import triton

import helion
from helion._compat import min_dot_size
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import is_cuda
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
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
        def _is_cuda_fp8_supported():
            if not is_cuda():
                return False
            return torch.cuda.get_device_capability(0)[0] >= 9

        def _is_xpu_fp8_supported():
            if not torch.xpu.is_available():
                return False

            from packaging import version

            return version.parse(triton.__version__) >= version.parse("3.5")

        is_fp8_supported = _is_cuda_fp8_supported() or _is_xpu_fp8_supported()
        if (
            input_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not is_fp8_supported
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
        # Test case 1: fused accumulation (acc_dtype = float32, common dtype = bfloat16)
        input_dtype = torch.bfloat16
        acc_dtype = torch.float32
        x = torch.randn(64, 64, device=DEVICE, dtype=input_dtype)
        y = torch.randn(64, 64, device=DEVICE, dtype=input_dtype)
        code, out = code_and_output(dot_kernel_acc_arg, (x, y, acc_dtype))
        # Validate we use tl.dot with out_dtype parameter (fused accumulation)
        self.assertIn("tl.dot(", code)
        self.assertIn("out_dtype=tl.float32", code)

        # Test case 2: separate addition (acc_dtype = float16, common dtype = float32)
        # TODO(Eikan): Support this case on XPU
        if not torch.xpu.is_available():
            input_dtype_2 = torch.float32
            acc_dtype_2 = torch.float16
            x2 = torch.randn(64, 64, device=DEVICE, dtype=input_dtype_2)
            y2 = torch.randn(64, 64, device=DEVICE, dtype=input_dtype_2)
            code2, out2 = code_and_output(dot_kernel_acc_arg, (x2, y2, acc_dtype_2))
            # Validate we use separate addition pattern with cast
            self.assertIn("tl.dot(", code2)
            # Check for the addition pattern: acc + result
            self.assertIn(" + ", code2)
            # Check that we cast the result to acc_dtype
            self.assertIn("tl.cast", code2)

        # Test case 3: separate addition (acc_dtype = int32, common dtype = int8)
        input_dtype_3 = torch.int8
        acc_dtype_3 = torch.int32
        x3 = torch.randint(-10, 10, (64, 64), device=DEVICE, dtype=input_dtype_3)
        y3 = torch.randint(-10, 10, (64, 64), device=DEVICE, dtype=input_dtype_3)
        code3, out3 = code_and_output(dot_kernel_acc_arg, (x3, y3, acc_dtype_3))
        # Validate we use separate addition pattern
        self.assertIn("tl.dot(", code3)
        # Check for the addition pattern: acc + result
        self.assertIn(" + ", code3)
        # Check that we cast the result to acc_dtype
        self.assertIn("tl.cast", code3)

    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_hl_dot_out_dtype_argument(self):
        @helion.kernel(
            config=helion.Config(block_sizes=[32, 32, 32]), dot_precision="tf32"
        )
        def dot_kernel_out_dtype(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float16, device=x.device)

            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
                for tile_k in hl.tile(k):
                    acc = hl.dot(
                        x[tile_m, tile_k],
                        y[tile_k, tile_n],
                        acc=acc,
                        out_dtype=torch.float16,
                    )
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(32, 48, device=DEVICE, dtype=torch.float32)
        y = torch.randn(48, 16, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(dot_kernel_out_dtype, (x, y))

        self.assertEqual(result.dtype, torch.float16)
        expected = (x @ y).to(torch.float16)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
        self.assertIn("out_dtype=tl.float16", code)

    def test_torch_matmul_3d(self):
        @helion.kernel(static_shapes=True)
        def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            b, m, k = A.size()
            _, _, n = B.size()
            out = torch.empty(
                [b, m, n],
                device=A.device,
                dtype=torch.promote_types(A.dtype, B.dtype),
            )
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(
                        A[tile_b, tile_m, tile_k],
                        B[tile_b, tile_k, tile_n],
                    )
                    acc += A[tile_b, tile_m, tile_k] @ B[tile_b, tile_k, tile_n]
                out[tile_b, tile_m, tile_n] = acc
            return out

        batch, m, k, n = 4, 32, 16, 24
        A = torch.randn([batch, m, k], device=DEVICE, dtype=torch.float16)
        B = torch.randn([batch, k, n], device=DEVICE, dtype=torch.float16)

        _, result = code_and_output(bmm, (A, B))
        expected = torch.bmm(A, B).to(result.dtype) * 2
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    # Note: numerical behavior for differing acc dtype is covered by existing dot tests; here we focus on codegen shape

    # torch.baddbmm codegen shape is covered indirectly by broader matmul tests; skipping a brittle code-inspection here

    @skipIfRefEager("Debug dtype codegen checks rely on compiled code")
    @skipIfXPU("Failed on XPU - https://github.com/pytorch/helion/issues/772")
    def test_baddbmm_pipeline_debug_dtype_asserts(self):
        # Reproduces scripts/repro512.py within the test suite and asserts
        # the kernel compiles and runs with debug dtype asserts enabled.
        @helion.kernel(
            autotune_effort="none",
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

    def _test_small_dims(
        self,
        m_dim,
        k_dim,
        n_dim,
        mm_func,
        check_code=False,
        check_matmul_cast_pattern=False,
        *,
        rtol: float = 1e-2,
        atol: float = 1e-3,
    ):
        @helion.kernel(config=helion.Config(block_sizes=[16, 16, 16]))
        def mm_small_dims(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.zeros(m, n, dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = mm_func(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(m_dim, k_dim, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k_dim, n_dim, device=DEVICE, dtype=torch.bfloat16)

        if check_code:
            code, result = code_and_output(mm_small_dims, (x, y, mm_func))
            self.assertExpectedJournal(code)
            if check_matmul_cast_pattern:
                self.assertIn(
                    "mm = tl.cast(tl.dot(tl.cast(load, tl.bfloat16), tl.cast(load_1, tl.bfloat16), input_precision='tf32', out_dtype=tl.float32), tl.bfloat16)",
                    code,
                )
        else:
            result = mm_small_dims(x, y, mm_func)

        expected = torch.matmul(x, y).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def _test_reshape_m_1(
        self, mm_func, check_code=False, *, rtol: float = 1e-2, atol: float = 1e-3
    ):
        """Test matrix multiplication with M=1 created through reshape."""

        @helion.kernel(config=helion.Config(block_sizes=[16, 16]))
        def mm_reshape_m_1(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            # x is a vector that we reshape to have M=1
            k = x.size(0)
            x_reshaped = x.view(1, k)  # M=1, K=k

            k2, n = y.size()
            assert k == k2

            out = torch.zeros(1, n, dtype=torch.float32, device=x.device)

            # Only tile over the N dimension; use ':' for the size-1 M dim
            for tile_n in hl.tile(n):
                acc = hl.zeros([1, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = mm_func(acc, x_reshaped[:, tile_k], y[tile_k, tile_n])
                out[:, tile_n] = acc

            return out.view(n)  # Reshape back to vector

        k, n = 32, 64
        x = torch.randn(k, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)

        # Use dynamic shapes for this test to stabilize codegen
        mm_reshape_m_1.settings.static_shapes = False
        mm_reshape_m_1.reset()

        check_journal = (
            check_code
            and is_cuda()
            and min_dot_size(DEVICE, x.dtype, y.dtype) == (1, 1, 16)
        )

        if check_journal:
            code, result = code_and_output(mm_reshape_m_1, (x, y, mm_func))
            self.assertExpectedJournal(code)
        else:
            result = mm_reshape_m_1(x, y, mm_func)

        expected = (x.view(1, k) @ y).view(n).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def _test_reshape_n_1(self, mm_func, *, rtol: float = 1e-2, atol: float = 1e-3):
        """Test matrix multiplication with N=1 created through reshape."""

        @helion.kernel(config=helion.Config(block_sizes=[16, 16]))
        def mm_reshape_n_1(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            m, k = x.size()

            # y is a vector that we reshape to have N=1
            k2 = y.size(0)
            assert k == k2
            y_reshaped = y.view(k, 1)  # K=k, N=1

            out = torch.zeros(m, 1, dtype=torch.float32, device=x.device)

            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, 1], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = mm_func(acc, x[tile_m, tile_k], y_reshaped[tile_k, :])
                out[tile_m, :] = acc

            return out.view(m)  # Reshape back to vector

        m, k = 64, 32
        x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k, device=DEVICE, dtype=torch.bfloat16)

        result = mm_reshape_n_1(x, y, mm_func)
        expected = (x @ y.view(k, 1)).view(m).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def _test_reshape_k_1(self, mm_func):
        """Test matrix multiplication with K=1 created through reshape."""

        @helion.kernel(config=helion.Config(block_sizes=[16, 16]))
        def mm_reshape_k_1(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            # x is a vector reshaped to have K=1
            m = x.size(0)
            x_reshaped = x.view(m, 1)  # M=m, K=1

            # y is a vector reshaped to have K=1
            n = y.size(0)
            y_reshaped = y.view(1, n)  # K=1, N=n

            out = torch.zeros(m, n, dtype=torch.float32, device=x.device)

            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                # K is 1; don't tile it — slice with ':'
                acc = mm_func(acc, x_reshaped[tile_m, :], y_reshaped[:, tile_n])
                out[tile_m, tile_n] = acc

            return out

        m, n = 64, 32
        x = torch.randn(m, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)

        result = mm_reshape_k_1(x, y, mm_func)
        expected = (x.view(m, 1) @ y.view(1, n)).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-3)

    def _test_reshape_k_2(
        self, mm_func, check_code=False, *, rtol: float = 1e-2, atol: float = 1e-3
    ):
        """Test matrix multiplication with K=2 created through reshape."""

        @helion.kernel(config=helion.Config(block_sizes=[16, 16]))
        def mm_reshape_k_2(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            # x is a 2*m vector reshaped to have K=2
            m_total = x.size(0)
            m = m_total // 2
            x_reshaped = x.view(m, 2)  # M=m, K=2

            # y is a 2*n vector reshaped to have K=2
            n_total = y.size(0)
            n = n_total // 2
            y_reshaped = y.view(2, n)  # K=2, N=n

            out = torch.zeros(m, n, dtype=torch.float32, device=x.device)

            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                # K is 2; don't tile it — slice with ':'
                acc = mm_func(acc, x_reshaped[tile_m, :], y_reshaped[:, tile_n])
                out[tile_m, tile_n] = acc

            return out

        m, n = 64, 32
        x = torch.randn(2 * m, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(2 * n, device=DEVICE, dtype=torch.bfloat16)

        # Use dynamic shapes for this test to stabilize codegen
        mm_reshape_k_2.settings.static_shapes = False
        mm_reshape_k_2.reset()

        check_journal = (
            check_code
            and is_cuda()
            and min_dot_size(DEVICE, x.dtype, y.dtype) == (1, 1, 16)
        )

        if check_journal:
            code, result = code_and_output(mm_reshape_k_2, (x, y, mm_func))
            self.assertExpectedJournal(code)
        else:
            result = mm_reshape_k_2(x, y, mm_func)

        expected = (x.view(m, 2) @ y.view(2, n)).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_hl_dot_small_m_dim(self):
        """Test hl.dot with M=2 which is smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=2,
            k_dim=32,
            n_dim=64,
            mm_func=lambda acc, a, b: hl.dot(a, b, acc=acc),
        )

    def test_hl_dot_small_n_dim(self):
        """Test hl.dot with N=3 which is smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=32,
            k_dim=64,
            n_dim=3,
            mm_func=lambda acc, a, b: hl.dot(a, b, acc=acc),
        )

    def test_hl_dot_small_k_dim(self):
        """Test hl.dot with K=4 which is smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=32,
            k_dim=4,
            n_dim=64,
            mm_func=lambda acc, a, b: hl.dot(a, b, acc=acc),
        )

    def test_hl_dot_multiple_small_dims(self):
        """Test hl.dot with multiple dims smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=5,
            k_dim=6,
            n_dim=7,
            mm_func=lambda acc, a, b: hl.dot(a, b, acc=acc),
            check_code=True,
        )

    def test_addmm_small_m_dim(self):
        """Test torch.addmm with M=2 smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(m_dim=2, k_dim=32, n_dim=64, mm_func=torch.addmm)

    def test_addmm_small_n_dim(self):
        """Test torch.addmm with N=3 smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(m_dim=32, k_dim=64, n_dim=3, mm_func=torch.addmm)

    def test_addmm_small_k_dim(self):
        """Test torch.addmm with K=4 smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(m_dim=32, k_dim=4, n_dim=64, mm_func=torch.addmm)

    def test_addmm_multiple_small_dims(self):
        """Test torch.addmm with multiple dims smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=5, k_dim=6, n_dim=7, mm_func=torch.addmm, check_code=True
        )

    def test_addmm_reshape_m_1(self):
        """Test torch.addmm with M=1 created through reshape."""
        self._test_reshape_m_1(torch.addmm, check_code=True)

    def test_addmm_reshape_n_1(self):
        """Test torch.addmm with N=1 created through reshape."""
        self._test_reshape_n_1(torch.addmm)

    def test_addmm_reshape_k_1(self):
        """Test torch.addmm with K=1 created through reshape."""
        self._test_reshape_k_1(torch.addmm)

    def test_addmm_reshape_k_2(self):
        """Test torch.addmm with K=2 created through reshape."""
        self._test_reshape_k_2(torch.addmm, check_code=True)

    def _test_reshape_m_2(self, mm_func, *, rtol: float = 1e-2, atol: float = 1e-3):
        """Test matrix multiplication with M=2 created through reshape."""

        @helion.kernel(config=helion.Config(block_sizes=[16, 16]))
        def mm_reshape_m_2(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            # x is a 2*k vector that we reshape to have M=2
            k_total = x.size(0)
            k = k_total // 2
            x_reshaped = x.view(2, k)  # M=2, K=k

            k2, n = y.size()
            assert k == k2

            out = torch.zeros(2, n, dtype=torch.float32, device=x.device)

            # M is 2; don't tile it — slice with ':'
            for tile_n in hl.tile(n):
                acc = hl.zeros([2, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = mm_func(acc, x_reshaped[:, tile_k], y[tile_k, tile_n])
                out[:, tile_n] = acc

            return out.view(2 * n)  # Reshape back to vector

        k, n = 32, 64
        torch.manual_seed(0)
        x = torch.randn(2 * k, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)

        result = mm_reshape_m_2(x, y, mm_func)
        expected = (x.view(2, k) @ y).view(2 * n).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def _test_reshape_n_2(self, mm_func, *, rtol: float = 1e-2, atol: float = 1e-3):
        """Test matrix multiplication with N=2 created through reshape."""

        @helion.kernel(config=helion.Config(block_sizes=[16, 16]))
        def mm_reshape_n_2(
            x: torch.Tensor,
            y: torch.Tensor,
            mm_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            m, k = x.size()

            # y is a 2*k vector that we reshape to have N=2
            k_total = y.size(0)
            k2 = k_total // 2
            assert k == k2
            y_reshaped = y.view(k, 2)  # K=k, N=2

            out = torch.zeros(m, 2, dtype=torch.float32, device=x.device)

            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, 2], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = mm_func(acc, x[tile_m, tile_k], y_reshaped[tile_k, :])
                out[tile_m, :] = acc

            return out.view(m * 2)  # Reshape back to vector

        m, k = 64, 32
        x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k * 2, device=DEVICE, dtype=torch.bfloat16)

        result = mm_reshape_n_2(x, y, mm_func)
        expected = (x @ y.view(k, 2)).view(m * 2).to(torch.float32)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_addmm_reshape_m_2(self):
        """Test torch.addmm with M=2 created through reshape."""
        self._test_reshape_m_2(torch.addmm)

    def test_addmm_reshape_n_2(self):
        """Test torch.addmm with N=2 created through reshape."""
        self._test_reshape_n_2(torch.addmm)

    def test_hl_dot_reshape_m_1(self):
        """Test hl.dot with M=1 created through reshape."""
        self._test_reshape_m_1(lambda acc, a, b: hl.dot(a, b, acc=acc))

    def test_hl_dot_reshape_n_1(self):
        """Test hl.dot with N=1 created through reshape."""
        self._test_reshape_n_1(lambda acc, a, b: hl.dot(a, b, acc=acc))

    def test_hl_dot_reshape_k_1(self):
        """Test hl.dot with K=1 created through reshape."""
        self._test_reshape_k_1(lambda acc, a, b: hl.dot(a, b, acc=acc))

    def test_hl_dot_reshape_k_2(self):
        """Test hl.dot with K=2 created through reshape."""
        self._test_reshape_k_2(lambda acc, a, b: hl.dot(a, b, acc=acc))

    def test_hl_dot_reshape_m_2(self):
        """Test hl.dot with M=2 created through reshape."""
        self._test_reshape_m_2(lambda acc, a, b: hl.dot(a, b, acc=acc))

    def test_hl_dot_reshape_n_2(self):
        """Test hl.dot with N=2 created through reshape."""
        self._test_reshape_n_2(lambda acc, a, b: hl.dot(a, b, acc=acc))

    def test_mm_small_m_dim(self):
        """Test torch.mm with M=2 smaller than the minimum of 16 for tl.dot."""
        # Allow slightly larger absolute error for torch.mm small-dim tiles
        self._test_small_dims(
            m_dim=2,
            k_dim=32,
            n_dim=64,
            mm_func=lambda acc, a, b: acc + torch.mm(a, b),
            atol=6e-2,
            rtol=1e-2,
        )

    def test_mm_small_n_dim(self):
        """Test torch.mm with N=3 smaller than the minimum of 16 for tl.dot."""
        # Allow slightly larger absolute error for torch.mm small-dim tiles
        self._test_small_dims(
            m_dim=32,
            k_dim=64,
            n_dim=3,
            mm_func=lambda acc, a, b: acc + torch.mm(a, b),
            atol=6e-2,
            rtol=1e-2,
        )

    def test_mm_small_k_dim(self):
        """Test torch.mm with K=4 smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=32,
            k_dim=4,
            n_dim=64,
            mm_func=lambda acc, a, b: acc + torch.mm(a, b),
        )

    def test_mm_multiple_small_dims(self):
        """Test torch.mm with multiple dims smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=5,
            k_dim=6,
            n_dim=7,
            mm_func=lambda acc, a, b: acc + torch.mm(a, b),
            check_code=True,
            check_matmul_cast_pattern=True,
        )

    def test_mm_reshape_m_1(self):
        """Test torch.mm with M=1 created through reshape."""
        self._test_reshape_m_1(
            lambda acc, a, b: acc + torch.mm(a, b), rtol=1e-2, atol=5e-2
        )

    def test_mm_reshape_n_1(self):
        """Test torch.mm with N=1 created through reshape."""
        self._test_reshape_n_1(
            lambda acc, a, b: acc + torch.mm(a, b), rtol=1e-2, atol=5e-2
        )

    def test_mm_reshape_k_1(self):
        """Test torch.mm with K=1 created through reshape."""
        self._test_reshape_k_1(lambda acc, a, b: acc + torch.mm(a, b))

    def test_mm_reshape_k_2(self):
        """Test torch.mm with K=2 created through reshape."""
        self._test_reshape_k_2(
            lambda acc, a, b: acc + torch.mm(a, b), rtol=1e-2, atol=5e-2
        )

    def test_mm_reshape_m_2(self):
        """Test torch.mm with M=2 created through reshape."""
        self._test_reshape_m_2(
            lambda acc, a, b: acc + torch.mm(a, b), rtol=1e-2, atol=5e-2
        )

    def test_mm_reshape_n_2(self):
        """Test torch.mm with N=2 created through reshape."""
        self._test_reshape_n_2(
            lambda acc, a, b: acc + torch.mm(a, b), rtol=1e-2, atol=5e-2
        )

    def test_matmul_small_m_dim(self):
        """Test torch.matmul with M=2 smaller than the minimum of 16 for tl.dot."""
        # Allow slightly larger absolute error for small-dim tiles
        self._test_small_dims(
            m_dim=2,
            k_dim=32,
            n_dim=64,
            mm_func=lambda acc, a, b: acc + torch.matmul(a, b),
            atol=6e-2,
            rtol=1e-2,
        )

    def test_matmul_small_n_dim(self):
        """Test torch.matmul with N=3 smaller than the minimum of 16 for tl.dot."""
        # Allow slightly larger absolute error for small-dim tiles
        self._test_small_dims(
            m_dim=32,
            k_dim=64,
            n_dim=3,
            mm_func=lambda acc, a, b: acc + torch.matmul(a, b),
            atol=6e-2,
            rtol=1e-2,
        )

    def test_matmul_small_k_dim(self):
        """Test torch.matmul with K=4 smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=32,
            k_dim=4,
            n_dim=64,
            mm_func=lambda acc, a, b: acc + torch.matmul(a, b),
        )

    def test_matmul_multiple_small_dims(self):
        """Test torch.matmul with multiple dims smaller than the minimum of 16 for tl.dot."""
        self._test_small_dims(
            m_dim=5,
            k_dim=6,
            n_dim=7,
            mm_func=lambda acc, a, b: acc + torch.matmul(a, b),
            check_code=True,
            check_matmul_cast_pattern=True,
        )

    def test_matmul_reshape_m_1(self):
        """Test torch.matmul with M=1 created through reshape."""
        self._test_reshape_m_1(
            lambda acc, a, b: acc + torch.matmul(a, b), rtol=1e-2, atol=5e-2
        )

    def test_matmul_reshape_n_1(self):
        """Test torch.matmul with N=1 created through reshape."""
        self._test_reshape_n_1(
            lambda acc, a, b: acc + torch.matmul(a, b), rtol=1e-2, atol=5e-2
        )

    def test_matmul_reshape_k_1(self):
        """Test torch.matmul with K=1 created through reshape."""
        self._test_reshape_k_1(lambda acc, a, b: acc + torch.matmul(a, b))

    def test_matmul_reshape_k_2(self):
        """Test torch.matmul with K=2 created through reshape."""
        self._test_reshape_k_2(
            lambda acc, a, b: acc + torch.matmul(a, b), rtol=1e-2, atol=5e-2
        )

    def test_matmul_reshape_m_2(self):
        """Test torch.matmul with M=2 created through reshape."""
        self._test_reshape_m_2(
            lambda acc, a, b: acc + torch.matmul(a, b), rtol=1e-2, atol=6.3e-2
        )

    def test_matmul_reshape_n_2(self):
        """Test torch.matmul with N=2 created through reshape."""
        self._test_reshape_n_2(
            lambda acc, a, b: acc + torch.matmul(a, b), rtol=1e-2, atol=5e-2
        )


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

    # Apply skipIfXPU decorator if needed
    if acc_dtype is torch.float16 and input_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bfloat16,
        torch.float32,
    ):
        _test_func = skipIfXPU("skip: float6 accmulator for non-fp16 input data types")(
            _test_func
        )

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
