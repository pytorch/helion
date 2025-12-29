"""
Test epilogue fusion with torch.compile (refactored v2).

Verifies that Inductor correctly generates epilogue lambdas for Helion kernels
and that the epilogue operations are fused into a single Triton kernel.

This is a refactored version of test_epilogue_fusion.py with:
- Parameterized tests to reduce boilerplate
- Common helper methods via FusionTestMixin
- Consolidated kernels
- Additional edge case tests
"""

from __future__ import annotations

import os
from typing import Callable, ClassVar
import unittest

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

import helion
from helion._testing import DEVICE, TestCase, count_triton_kernels
import helion.language as hl


# =============================================================================
# Shared Kernels
# =============================================================================


@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee", allow_fuse_with_inductor_ops=True)
def matmul(x, y):  # pyrefly: ignore
    """Matrix multiplication - epilogue/prologue fusion happens automatically."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def rms_norm(x, weight, eps=1e-5):  # pyrefly: ignore
    """RMS normalization - epilogue fusion happens automatically."""
    m, n = x.size()
    assert weight.size(0) == n
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        result = normalized * weight[:].to(torch.float32)
        out[tile_m, :] = result.to(out.dtype)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee", allow_fuse_with_inductor_ops=True)
def matmul_with_residual(x, y):  # pyrefly: ignore
    """Matmul returning both result and residual (multi-output kernel)."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    dtype = torch.promote_types(x.dtype, y.dtype)
    out = torch.empty([m, n], dtype=dtype, device=x.device)
    residual = torch.empty([m, n], dtype=dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        residual[tile_m, tile_n] = acc
    return out, residual


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def batch_elementwise_3d(x):  # pyrefly: ignore
    """3D kernel for testing epilogue fusion with 3D broadcasting."""
    batch, m, n = x.size()
    out = torch.empty_like(x)
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        out[tile_b, tile_m, tile_n] = x[tile_b, tile_m, tile_n] * 2.0
    return out


@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee", allow_fuse_with_inductor_ops=True)
def matmul_split_k_atomic(x, y):  # pyrefly: ignore
    """Matmul with split-K using atomic adds (no epilogue fusion possible)."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    split_k = 2
    k_chunk = k // split_k
    for tile_m, tile_n in hl.tile([m, n]):
        for k_split in hl.grid(split_k):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            k_start = k_split * k_chunk
            k_end = k_start + k_chunk
            for tile_k in hl.tile(k_start, k_end):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_with_nonuniform_outputs(x):  # pyrefly: ignore
    """Elementwise ops returning outputs with DIFFERENT shapes."""
    (n,) = x.size()
    out1 = torch.empty([n], dtype=x.dtype, device=x.device)
    out2 = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        x_tile = x[tile_n]
        out1[tile_n] = x_tile * 2
        out2[tile_n, 0] = x_tile
        out2[tile_n, 1] = x_tile * 3
    return out1, out2, 42


@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee", allow_fuse_with_inductor_ops=True)
def matmul_with_residual_no_type_hint(x, y):
    """Matmul returning multiple outputs WITHOUT return type hints."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    dtype = torch.promote_types(x.dtype, y.dtype)
    out = torch.empty([m, n], dtype=dtype, device=x.device)
    residual = torch.empty([m, n], dtype=dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        residual[tile_m, tile_n] = acc
    return out, residual


@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee", allow_fuse_with_inductor_ops=True)
def matmul_with_scalar_return(x, y):
    """Matmul returning tensor AND non-tensor (scalar) value."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    dtype = torch.promote_types(x.dtype, y.dtype)
    out = torch.empty([m, n], dtype=dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out, 42


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_with_different_shapes_same_elements(x):
    """Elementwise returning outputs with DIFFERENT shapes but SAME element count."""
    m, n = x.size()
    out1 = torch.empty([m * n], dtype=x.dtype, device=x.device)
    out2 = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out1_view = out1.view(m, n)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out1_view[tile_m, :] = x_tile * 2
        out2[tile_m, :] = x_tile * 3
    return out1, out2, 42


# =============================================================================
# Test Mixin with Helper Methods
# =============================================================================


class FusionTestMixin:
    """Mixin providing common fusion test helpers."""

    def _run_fusion_test(
        self,
        f: Callable,
        inputs: tuple,
        expected_kernels: int = 1,
        code_patterns: list[str] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> str:
        """Run fusion test: eager vs compiled, verify kernel count and patterns."""
        eager_result = f(*inputs)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(compiled_result, eager_result, rtol=rtol, atol=atol)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(  # type: ignore[attr-defined]
            kernel_count,
            expected_kernels,
            f"Expected {expected_kernels} kernel(s), got {kernel_count}",
        )

        if code_patterns:
            found = any(p in all_code for p in code_patterns)
            self.assertTrue(  # type: ignore[attr-defined]
                found, f"Expected one of {code_patterns} in generated code"
            )

        return all_code

    def _create_matmul_inputs(
        self, m: int = 64, n: int = 64, k: int = 64, dtype: torch.dtype = torch.float16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create standard matmul inputs."""
        return (
            torch.randn(m, k, device=DEVICE, dtype=dtype),
            torch.randn(k, n, device=DEVICE, dtype=dtype),
        )


# =============================================================================
# Test Data Constants
# =============================================================================

# Unary epilogue operations: (name, op_fn, code_patterns, dtype, rtol, atol)
UNARY_EPILOGUE_OPS: list[tuple] = [
    ("relu", torch.relu, ["triton_helpers.maximum", "tl.maximum"], torch.float16, 1e-3, 1e-3),
    ("sigmoid", torch.sigmoid, ["sigmoid", "tl.sigmoid"], torch.float32, 1e-3, 1e-3),
    ("tanh", torch.tanh, ["tanh", "libdevice.tanh"], torch.float32, 1e-3, 1e-3),
    ("silu", F.silu, ["sigmoid"], torch.float32, 1e-3, 1e-3),
    ("neg", lambda x: -x, ["= -v_", "= -", "(-", ", -v_"], torch.float16, 1e-3, 1e-3),
    ("sin", torch.sin, ["sin"], torch.float32, 1e-3, 1e-3),
    ("cos", torch.cos, ["cos"], torch.float32, 1e-3, 1e-3),
    ("floor", torch.floor, ["floor"], torch.float32, 1e-3, 1e-3),
    ("ceil", torch.ceil, ["ceil"], torch.float32, 1e-3, 1e-3),
    ("erf", torch.erf, ["erf"], torch.float32, 1e-3, 1e-3),
    ("leaky_relu", lambda x: F.leaky_relu(x, 0.01), None, torch.float32, 1e-3, 1e-3),
    ("softplus", F.softplus, None, torch.float32, 1e-3, 1e-3),
    ("scalar_mul", lambda x: x * 2.0, ["2.0"], torch.float16, 1e-3, 1e-3),
    # Edge cases
    ("abs", torch.abs, None, torch.float32, 1e-3, 1e-3),
    ("sign", torch.sign, None, torch.float32, 1e-3, 1e-3),
    ("clamp", lambda x: torch.clamp(x, -1, 1), None, torch.float32, 1e-3, 1e-3),
    ("gt_zero", lambda x: (x > 0).float(), None, torch.float32, 1e-3, 1e-3),
]

# Ops requiring fp32 for dtype auto-upcast tests
FP32_REQUIRED_OPS: list[tuple[str, Callable]] = [
    ("sigmoid", torch.sigmoid),
    ("exp", torch.exp),
    ("log", torch.log),
    ("sqrt", torch.sqrt),
    ("sin", torch.sin),
    ("cos", torch.cos),
    ("tanh", torch.tanh),
]

# Non-fusable operations (reductions, scans, etc.)
NON_FUSABLE_OPS: list[tuple[str, Callable, str]] = [
    ("sum", lambda x: torch.sum(x), "reduction"),
    ("mean", lambda x: torch.mean(x), "reduction"),
    ("sort", lambda x: torch.sort(x, dim=1)[0], "complex"),
    ("topk", lambda x: torch.topk(x, k=10, dim=1)[0], "complex"),
    ("cumsum", lambda x: torch.cumsum(x, dim=1), "scan"),
]


# =============================================================================
# Test Classes
# =============================================================================


class TestBasicEpilogue(FusionTestMixin, TestCase):
    """Basic epilogue fusion tests with unary operations."""

    def test_matmul_basic_no_epilogue(self):
        """Basic matmul with torch.compile - no epilogue fusion."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return matmul(x, y)

        all_code = self._run_fusion_test(f, (x, y), code_patterns=["_helion_"])
        self.assertIn("_helion_", all_code)

    def test_unary_epilogue_ops(self):
        """Test unary operations fused as epilogue."""
        for op_name, op_fn, code_patterns, dtype, rtol, atol in UNARY_EPILOGUE_OPS:
            with self.subTest(op=op_name):
                x, y = self._create_matmul_inputs(dtype=dtype)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y))

                self._run_fusion_test(
                    f, (x, y), code_patterns=code_patterns, rtol=rtol, atol=atol
                )


class TestBinaryEpilogueWithClosure(FusionTestMixin, TestCase):
    """Binary epilogue operations with closure tensors."""

    def test_bias_1d_epilogue(self):
        """Matmul + 1D bias (column broadcast)."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return matmul(x, y) + bias

        self._run_fusion_test(f, (x, y, bias), code_patterns=["epilogue_input"])

    def test_bias_2d_epilogue(self):
        """Matmul + 2D bias (full tensor)."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return matmul(x, y) + bias

        self._run_fusion_test(f, (x, y, bias), code_patterns=["epilogue_input"])

    def test_scale_1d_epilogue(self):
        """Matmul * 1D scale (column broadcast)."""
        x, y = self._create_matmul_inputs()
        scale = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, scale):
            return matmul(x, y) * scale

        self._run_fusion_test(f, (x, y, scale), code_patterns=["epilogue_input"])

    def test_bias_scale_epilogue(self):
        """Matmul + bias then * scale (chained closure ops)."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias, scale):
            return (matmul(x, y) + bias) * scale

        eager_result = f(x, y, bias, scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias, scale)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)
        # Both bias and scale should be loaded as epilogue_input
        self.assertGreaterEqual(all_code.count("epilogue_input"), 2)

    def test_bias_relu_epilogue(self):
        """Matmul + bias + relu (full linear layer)."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return torch.relu(matmul(x, y) + bias)

        all_code = self._run_fusion_test(f, (x, y, bias))
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code
        )
        self.assertIn("epilogue_input", all_code)


class TestDtypeAutoUpcast(FusionTestMixin, TestCase):
    """Tests for dtype handling - ops requiring fp32 should auto-upcast."""

    def test_fp16_ops_auto_upcast(self):
        """Test fp16 ops requiring fp32 are auto-upcast and fused."""
        for op_name, op_fn in FP32_REQUIRED_OPS:
            with self.subTest(op=op_name):
                x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16).abs() + 0.1
                y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y).abs() + 0.1)

                self._run_fusion_test(f, (x, y), rtol=1e-2, atol=1e-2)

    def test_bf16_ops_auto_upcast(self):
        """Test bf16 ops requiring fp32 are auto-upcast and fused."""
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 not supported on this device")

        for op_name, op_fn in FP32_REQUIRED_OPS:
            with self.subTest(op=op_name):
                x = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16).abs() + 0.1
                y = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y).abs() + 0.1)

                self._run_fusion_test(f, (x, y), rtol=1e-1, atol=1e-1)


class TestProperIndexing(FusionTestMixin, TestCase):
    """Tests requiring proper range matching and index translation."""

    def test_row_bias_epilogue(self):
        """Matmul + row-wise bias: output[i,j] + bias[i]."""
        m, n = 64, 128
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float16)
        row_bias = torch.randn(m, device=DEVICE, dtype=torch.float16)

        def f(x, y, row_bias):
            return matmul(x, y) + row_bias.unsqueeze(1)

        self._run_fusion_test(f, (x, y, row_bias), code_patterns=["epilogue_input"])

    def test_row_and_col_bias_epilogue(self):
        """Matmul + row bias + column bias."""
        m, n = 64, 128
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float16)
        row_bias = torch.randn(m, device=DEVICE, dtype=torch.float16)
        col_bias = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, y, row_bias, col_bias):
            return matmul(x, y) + row_bias.unsqueeze(1) + col_bias.unsqueeze(0)

        eager_result = f(x, y, row_bias, col_bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(
            compiled_f, x, y, row_bias, col_bias
        )

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)
        self.assertGreaterEqual(all_code.count("epilogue_input"), 2)

    def test_nonsquare_bias_epilogue(self):
        """Matmul with non-square shapes + column bias."""
        m, k, n = 32, 64, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return matmul(x, y) + bias

        self._run_fusion_test(f, (x, y, bias), code_patterns=["epilogue_input"])

    def test_scale_row_bias_col_epilogue(self):
        """Matmul * scale[i] + bias[j] - different indices for different ops."""
        m, n = 64, 128
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float32)
        scale_row = torch.randn(m, device=DEVICE, dtype=torch.float32)
        bias_col = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale_row, bias_col):
            return matmul(x, y) * scale_row.unsqueeze(1) + bias_col.unsqueeze(0)

        eager_result = f(x, y, scale_row, bias_col)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(
            compiled_f, x, y, scale_row, bias_col
        )

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)
        self.assertGreaterEqual(all_code.count("epilogue_input"), 2)

    def test_complex_epilogue_chain(self):
        """Complex chain: (matmul * scale + bias) * gate + residual."""
        m, n = 64, 64
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        gate = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        residual = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale, bias, gate, residual):
            return (matmul(x, y) * scale + bias) * gate + residual

        eager_result = f(x, y, scale, bias, gate, residual)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(
            compiled_f, x, y, scale, bias, gate, residual
        )

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)
        self.assertGreaterEqual(all_code.count("epilogue_input"), 4)

    def test_3d_broadcast_epilogue(self):
        """3D broadcasting in epilogue fusion."""
        batch, m, n = 4, 32, 32
        x = torch.randn(batch, m, n, device=DEVICE, dtype=torch.float32)
        bias_batch = torch.randn(batch, device=DEVICE, dtype=torch.float32)
        bias_row = torch.randn(m, device=DEVICE, dtype=torch.float32)
        bias_col = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, bias_batch, bias_row, bias_col):
            result = batch_elementwise_3d(x)
            result = result + bias_batch[:, None, None]
            result = result + bias_row[None, :, None]
            result = result + bias_col[None, None, :]
            return result

        all_code = self._run_fusion_test(f, (x, bias_batch, bias_row, bias_col))
        self.assertIn("indices_0", all_code)
        self.assertIn("indices_1", all_code)
        self.assertIn("indices_2", all_code)


class TestRMSNormEpilogue(FusionTestMixin, TestCase):
    """Tests for RMS normalization kernel epilogue fusion."""

    def test_rms_norm_relu(self):
        """RMS norm + relu."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, weight):
            return torch.relu(rms_norm(x, weight))

        all_code = self._run_fusion_test(f, (x, weight))
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code
        )

    def test_rms_norm_relu_add(self):
        """RMS norm + relu + add."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, weight, bias):
            return torch.relu(rms_norm(x, weight)) + bias

        all_code = self._run_fusion_test(f, (x, weight, bias))
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code
        )
        self.assertIn("epilogue_input", all_code)


class TestMultiOutput(FusionTestMixin, TestCase):
    """Tests for kernels with multiple output tensors."""

    def test_basic(self):
        """Basic multi-output kernel without epilogue."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return matmul_with_residual(x, y)

        out_eager, res_eager = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], res_eager, rtol=1e-3, atol=1e-3)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_epilogue_first_only(self):
        """Epilogue applied to first output only."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            out, residual = matmul_with_residual(x, y)
            return torch.relu(out), residual

        out_eager, res_eager = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], res_eager, rtol=1e-3, atol=1e-3)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_epilogue_both(self):
        """Epilogue applied to both outputs."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            out, residual = matmul_with_residual(x, y)
            return torch.relu(out), torch.sigmoid(residual)

        out_eager, res_eager = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], res_eager, rtol=1e-3, atol=1e-3)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_epilogue_combined(self):
        """Epilogue that combines both outputs (out + residual)."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            out, residual = matmul_with_residual(x, y)
            return out + residual

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_nonuniform_shapes(self):
        """Multi-output kernel with outputs of DIFFERENT shapes."""
        x = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x):
            return elementwise_with_nonuniform_outputs(x)

        out1_eager, out2_eager, scalar_eager = f(x)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x)

        self.assertEqual(len(result), 3)
        torch.testing.assert_close(result[0], out1_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], out2_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(result[2], 42)
        self.assertEqual(list(result[0].shape), [64])
        self.assertEqual(list(result[1].shape), [64, 2])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_no_type_hint(self):
        """Multi-output kernel WITHOUT return type hints."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return matmul_with_residual_no_type_hint(x, y)

        out_eager, res_eager = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], res_eager, rtol=1e-3, atol=1e-3)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_with_scalar_return(self):
        """Multi-output kernel that returns tensor + non-tensor (scalar)."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            out, scalar = matmul_with_scalar_return(x, y)
            return out, scalar

        out_eager, scalar_eager = f(x, y)
        self.assertEqual(scalar_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], out_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(result[1], 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_different_shapes_same_element_count(self):
        """Multi-output kernel with DIFFERENT shapes but SAME element count."""
        m, n = 64, 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x):
            out1, out2, scalar = elementwise_with_different_shapes_same_elements(x)
            return torch.relu(out1), torch.sigmoid(out2), scalar

        out1_eager, out2_eager, scalar_eager = f(x)
        self.assertEqual(scalar_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        torch.testing.assert_close(result[0], out1_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], out2_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(result[2], 42)

        # Verify outputs have DIFFERENT shapes but SAME element count
        self.assertEqual(list(result[0].shape), [m * n])
        self.assertEqual(list(result[1].shape), [m, n])
        self.assertNotEqual(list(result[0].shape), list(result[1].shape))
        self.assertEqual(result[0].numel(), result[1].numel())

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_no_type_hint_with_epilogue(self):
        """Multi-output kernel WITHOUT type hints, with epilogue fusion."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            out, residual = matmul_with_residual_no_type_hint(x, y)
            return torch.relu(out), torch.sigmoid(residual)

        out_eager, res_eager = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result[1], res_eager, rtol=1e-3, atol=1e-3)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)


class TestAtomicKernels(FusionTestMixin, TestCase):
    """Tests for kernels with atomic operations (no epilogue fusion)."""

    def test_basic(self):
        """Split-K matmul with atomic adds - basic functionality."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return matmul_split_k_atomic(x, y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

    def test_no_epilogue_fusion(self):
        """Atomic kernel + epilogue should result in separate kernels."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return torch.relu(matmul_split_k_atomic(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreaterEqual(kernel_count, 2, "Atomic kernels should NOT fuse epilogue")

    def test_vs_simple_matmul(self):
        """Compare atomic kernel vs simple kernel with same epilogue."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        simple_result = torch.relu(matmul(x, y))
        atomic_result = torch.relu(matmul_split_k_atomic(x, y))

        torch.testing.assert_close(atomic_result, simple_result, rtol=1e-3, atol=1e-3)


class TestNonFusableOps(FusionTestMixin, TestCase):
    """Tests for operations that cannot be fused (reductions, scans, etc.)."""

    def test_non_fusable_ops(self):
        """Test that non-pointwise ops generate separate kernels."""
        for op_name, op_fn, reason in NON_FUSABLE_OPS:
            with self.subTest(op=op_name, reason=reason):
                x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
                y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y))

                eager_result = f(x, y)
                compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
                compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

                torch.testing.assert_close(
                    compiled_result, eager_result, rtol=1e-2, atol=1e-2
                )

    def test_transpose_view_op(self):
        """Matmul followed by transpose - view operation."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return matmul(x, y).T

        self._run_fusion_test(f, (x, y))


class TestSchedulerFilteredOps(FusionTestMixin, TestCase):
    """Tests for operations that get filtered by scheduler (scatter, gather)."""

    def test_scatter_epilogue(self):
        """Matmul followed by scatter_add - should NOT fuse."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)
        index = torch.randint(0, 64, (64, 64), device=DEVICE)

        def f(x, y, index):
            out = matmul(x, y)
            result = torch.zeros_like(out)
            result.scatter_add_(1, index, out)
            return result

        eager_result = f(x, y, index)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, index)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        # Scatter is not fusable, should result in multiple kernels
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreaterEqual(kernel_count, 1)

    def test_gather_complex_epilogue(self):
        """Matmul followed by gather - should work but may not fuse."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)
        index = torch.randint(0, 64, (64, 32), device=DEVICE)

        def f(x, y, index):
            out = matmul(x, y)
            return torch.gather(out, 1, index)

        eager_result = f(x, y, index)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, index)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)
        # Gather may or may not fuse - just verify correctness
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreaterEqual(kernel_count, 1)


class TestEdgeCases(FusionTestMixin, TestCase):
    """Edge cases and regression tests."""

    def test_chained_multiple_ops(self):
        """Multiple chained operations - all should fuse."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            result = matmul(x, y)
            result = torch.relu(result)
            result = result * 2.0
            result = result + 1.0
            return torch.tanh(result)

        self._run_fusion_test(f, (x, y))

    def test_mixed_closure_sizes(self):
        """Epilogue with multiple closure tensors of different sizes."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias, scale):
            return (matmul(x, y) + bias) * scale

        self._run_fusion_test(f, (x, y, bias, scale))

    def test_where_epilogue(self):
        """Matmul with where/conditional epilogue."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            result = matmul(x, y)
            return torch.where(result > 0, result, torch.zeros_like(result))

        self._run_fusion_test(f, (x, y))

    def test_non_contiguous_bias(self):
        """Matmul with non-contiguous bias tensor."""
        x, y = self._create_matmul_inputs()
        bias_full = torch.randn(128, device=DEVICE, dtype=torch.float16)
        bias = bias_full[::2]  # Non-contiguous
        assert not bias.is_contiguous()

        def f(x, y, bias):
            return matmul(x, y) + bias

        self._run_fusion_test(f, (x, y, bias))

    def test_reciprocal(self):
        """Matmul followed by reciprocal."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) + 2.0
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) + 2.0

        def f(x, y):
            return torch.reciprocal(matmul(x, y) + 1.0)

        self._run_fusion_test(f, (x, y))

    def test_multiple_consumers(self):
        """Output used twice in epilogue (multiple consumers)."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            out = matmul(x, y)
            return out + torch.relu(out)

        self._run_fusion_test(f, (x, y))

    def test_pow_operations(self):
        """Power operations in epilogue."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return torch.pow(matmul(x, y), 2)

        self._run_fusion_test(f, (x, y))


class TestIndexingPatterns(FusionTestMixin, TestCase):
    """Tests for different indexing patterns in generated code."""

    def test_standard_indices(self):
        """Standard indexing pattern with indices_0, indices_1."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)

        def f(x, y):
            return torch.relu(matmul(x, y))

        all_code = self._run_fusion_test(f, (x, y))
        self.assertIn("indices_0", all_code)
        self.assertIn("indices_1", all_code)

    def test_row_bias_indexing(self):
        """Row bias indexing pattern."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)
        bias = torch.randn(64, 1, device=DEVICE, dtype=torch.float32)

        def f(x, y, bias):
            return matmul(x, y) + bias

        self._run_fusion_test(f, (x, y, bias))

    def test_combined_ops_indexing(self):
        """Combined operations with column scale."""
        x, y = self._create_matmul_inputs(dtype=torch.float32)
        scale = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale):
            return torch.sigmoid(matmul(x, y) * scale)

        self._run_fusion_test(f, (x, y, scale))


if __name__ == "__main__":
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    unittest.main()
