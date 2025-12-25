"""
Test prologue fusion with torch.compile.

Verifies that Inductor correctly generates prologue lambdas for Helion kernels
and that the prologue operations are fused into a single Triton kernel.

Prologue fusion is the complement of epilogue fusion:
- Epilogue fusion: kernel_output -> pointwise_ops (ops fused AFTER kernel)
- Prologue fusion: pointwise_ops -> kernel_input (ops fused INTO kernel, before compute)
"""
from __future__ import annotations

import os
import unittest

import pytest

import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, TestCase, count_triton_kernels
from torch._inductor.utils import run_and_get_code


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def matmul_with_prologue(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Matrix multiplication - prologue/epilogue fusion happens automatically."""
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


class TestPrologueFusionBasic(TestCase):
    """Basic prologue fusion tests - single input transformations."""

    def test_matmul_relu_x_prologue(self):
        """Matmul with relu on first input - should ideally fuse into prologue.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            # relu(x) @ y - the relu should ideally be fused as prologue
            return matmul_with_prologue(torch.relu(x), y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Correctness check
        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        # Analyze generated code
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with relu fused as prologue, got {kernel_count}")
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu (maximum) fused into Triton kernel"
        )

    def test_matmul_sigmoid_x_prologue(self):
        """Matmul with sigmoid on first input.

        Note: Uses fp32 because Triton's sigmoid requires fp32 or fp64.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul_with_prologue(torch.sigmoid(x), y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with sigmoid fused as prologue, got {kernel_count}")
        self.assertTrue(
            "sigmoid" in all_code or "tl.sigmoid" in all_code,
            "Expected sigmoid fused into Triton kernel"
        )

    def test_matmul_scalar_mul_x_prologue(self):
        """Matmul with scalar multiplication on first input.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul_with_prologue(x * 2.0, y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = (x * 2.0) @ y
        # Verify fusion preserves semantics: compiled should match eager exactly
        torch.testing.assert_close(compiled_result, eager_result, rtol=0, atol=0)
        # Sanity check vs baseline (relaxed due to tiled vs non-tiled accumulation order)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-2, atol=0.1)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with scalar multiply fused as prologue, got {kernel_count}")
        self.assertIn("2.0", all_code, "Expected scalar 2.0 fused into Triton kernel")


class TestPrologueFusionWithClosure(TestCase):
    """Prologue fusion with external tensors (closures)."""

    def test_matmul_add_bias_x_prologue(self):
        """Matmul with bias addition on first input.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)  # [K] broadcast

        def f(x, y, bias):
            # (x + bias) @ y - bias add fused as prologue
            return matmul_with_prologue(x + bias, y)

        eager_result = f(x, y, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias)

        baseline = (x + bias) @ y
        # Verify fusion preserves semantics: compiled should match eager exactly
        torch.testing.assert_close(compiled_result, eager_result, rtol=0, atol=0)
        # Sanity check vs baseline (relaxed due to tiled vs non-tiled accumulation order)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-2, atol=0.1)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with bias add fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg1_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg1_1) in Triton kernel for prologue")

    def test_matmul_mul_scale_x_prologue(self):
        """Matmul with scale multiplication on first input.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(64, device=DEVICE, dtype=torch.float32)  # [K] broadcast

        def f(x, y, scale):
            return matmul_with_prologue(x * scale, y)

        eager_result = f(x, y, scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, scale)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with scale multiply fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg1_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg1_1) in Triton kernel for prologue")


class TestPrologueFusionBothInputs(TestCase):
    """Prologue fusion on multiple inputs."""

    def test_matmul_relu_both_inputs(self):
        """Matmul with relu on both inputs.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul_with_prologue(torch.relu(x), torch.relu(y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with relu on both inputs fused as prologue, got {kernel_count}")
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu (maximum) fused into Triton kernel"
        )


class TestPrologueEpilogueCombined(TestCase):
    """Tests combining prologue and epilogue fusion."""

    def test_matmul_relu_prologue_sigmoid_epilogue(self):
        """relu(x) @ y -> sigmoid - prologue and epilogue combined.

        Note: Uses fp32 because Triton's sigmoid requires fp32 or fp64.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            # relu as prologue, sigmoid as epilogue
            return torch.sigmoid(matmul_with_prologue(torch.relu(x), y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with relu prologue and sigmoid epilogue fused, got {kernel_count}")
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu (maximum) fused into Triton kernel"
        )
        self.assertTrue(
            "sigmoid" in all_code or "tl.sigmoid" in all_code,
            "Expected sigmoid fused into Triton kernel"
        )

    def test_matmul_scale_prologue_bias_epilogue(self):
        """(x * scale) @ y + bias - scale as prologue, bias as epilogue.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(64, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale, bias):
            return matmul_with_prologue(x * scale, y) + bias

        eager_result = f(x, y, scale, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, scale, bias)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with scale prologue and bias epilogue fused, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg1_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg1_1) in Triton kernel for prologue")
        # The epilogue external buffer is also named _ext_..., there should be at least 2 (one for prologue, one for epilogue)
        self.assertGreaterEqual(all_code.count("_ext_"), 2, "Expected at least 2 external buffers (prologue + epilogue) in Triton kernel")


class TestPrologueFusionRequiresProperIndexing(TestCase):
    """Tests that REQUIRE proper range matching and index translation for prologue.

    These tests will FAIL unless:
    1. Helion exposes input ranges to Inductor for proper fusion decisions
    2. Codegen happens inside Helion with access to tile indices
    3. Index translation correctly maps Inductor dimensions to Helion tile indices

    For prologues, the indexing is even more complex because:
    - First input x[M, K] uses indices (tile_m, tile_k)
    - Second input y[K, N] uses indices (tile_k, tile_n)
    - These share the K dimension (tile_k) but have different other dimensions
    """

    def test_matmul_row_scale_x_prologue(self):
        """Matmul with row-wise scaling on first input: (x * scale[i]) @ y.

        This requires:
        - scale has shape [M] (first dimension of x)
        - Must load scale using indices corresponding to tile_m
        - Requires proper dimension mapping for x's first dimension

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        row_scale = torch.randn(m, device=DEVICE, dtype=torch.float32)

        def f(x, y, row_scale):
            # row_scale broadcasts as [M, 1] to multiply x[M, K]
            return matmul_with_prologue(x * row_scale.unsqueeze(1), y)

        eager_result = f(x, y, row_scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, row_scale)

        baseline = (x * row_scale.unsqueeze(1)) @ y
        # Verify fusion preserves semantics: compiled should match eager exactly
        torch.testing.assert_close(compiled_result, eager_result, rtol=0, atol=0)
        # Sanity check vs baseline (relaxed due to tiled vs non-tiled accumulation order)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-2, atol=0.1)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with row-wise scale fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg0_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg0_1) in Triton kernel for prologue")

    def test_matmul_col_scale_x_prologue(self):
        """Matmul with column-wise scaling on first input: (x * scale[k]) @ y.

        This requires:
        - scale has shape [K] (second dimension of x, shared with y)
        - Must load scale using indices corresponding to tile_k
        - Requires proper dimension mapping for the reduction dimension

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        col_scale = torch.randn(k, device=DEVICE, dtype=torch.float32)

        def f(x, y, col_scale):
            # col_scale broadcasts as [1, K] to multiply x[M, K]
            return matmul_with_prologue(x * col_scale, y)

        eager_result = f(x, y, col_scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, col_scale)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=0.02)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with column-wise scale fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg0_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg0_1) in Triton kernel for prologue")

    def test_matmul_row_scale_y_prologue(self):
        """Matmul with row-wise scaling on second input: x @ (y * scale[k]).

        This requires:
        - scale has shape [K] (first dimension of y, shared with x)
        - Must load scale using indices corresponding to tile_k
        - Tests prologue on the second input with proper indexing

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        row_scale = torch.randn(k, device=DEVICE, dtype=torch.float32)

        def f(x, y, row_scale):
            # row_scale broadcasts as [K, 1] to multiply y[K, N]
            return matmul_with_prologue(x, y * row_scale.unsqueeze(1))

        eager_result = f(x, y, row_scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, row_scale)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with row-wise scale on y fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg0_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg0_1) in Triton kernel for prologue")

    def test_matmul_col_scale_y_prologue(self):
        """Matmul with column-wise scaling on second input: x @ (y * scale[n]).

        This requires:
        - scale has shape [N] (second dimension of y)
        - Must load scale using indices corresponding to tile_n
        - Tests prologue on the second input with proper indexing

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        col_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, col_scale):
            # col_scale broadcasts as [1, N] to multiply y[K, N]
            return matmul_with_prologue(x, y * col_scale)

        eager_result = f(x, y, col_scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, col_scale)

        baseline = x @ (y * col_scale)
        # Verify fusion preserves semantics: compiled should match eager exactly
        torch.testing.assert_close(compiled_result, eager_result, rtol=0, atol=0)
        # Sanity check vs baseline (relaxed due to tiled vs non-tiled accumulation order)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-2, atol=0.1)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with column-wise scale on y fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg0_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg0_1) in Triton kernel for prologue")

    def test_matmul_different_scales_both_inputs(self):
        """Different scaling on both inputs: (x * scale_x) @ (y * scale_y).

        This is the hardest prologue test: two different external tensors,
        each using different tile indices for different inputs.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        scale_x = torch.randn(k, device=DEVICE, dtype=torch.float32)  # [K] for x
        scale_y = torch.randn(n, device=DEVICE, dtype=torch.float32)  # [N] for y

        def f(x, y, scale_x, scale_y):
            return matmul_with_prologue(x * scale_x, y * scale_y)

        eager_result = f(x, y, scale_x, scale_y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, scale_x, scale_y)

        baseline = (x * scale_x) @ (y * scale_y)
        # Verify fusion preserves semantics: compiled should match eager exactly
        torch.testing.assert_close(compiled_result, eager_result, rtol=0, atol=0)
        # Sanity check vs baseline (relaxed due to tiled vs non-tiled accumulation order)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-2, atol=0.1)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with scales on both inputs fused as prologue, got {kernel_count}")
        # Check for external buffer parameter in kernel (e.g., _ext_arg0_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg0_1) in Triton kernel for prologue")

    def test_matmul_2d_transform_x_prologue(self):
        """Matmul with 2D element-wise transform on first input.

        tanh(x[i, k]) @ y - full 2D indexing for prologue using tanh.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.

        Note: We use tanh instead of (x + transform) because Inductor's prologue
        fusion heuristic rejects fusions where read_bytes > write_bytes * 1.1.
        Adding a full 2D tensor doubles the read bytes, which fails this heuristic.
        tanh is an in-place element-wise op that doesn't add extra memory reads.
        """
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul_with_prologue(torch.tanh(x), y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare fused vs eager (not against torch baseline to avoid flakiness)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with 2D tanh fused as prologue, got {kernel_count}")
        self.assertIn("libdevice", all_code.lower(), "Expected tanh (libdevice) fused into Triton kernel")


class TestPrologueFusionQuantization(TestCase):
    """Tests for quantization-style prologue patterns (dequantization)."""

    def test_matmul_dequant_x_prologue(self):
        """Simulated dequantization: (x_int - zero_point) * scale @ y.

        This is a common pattern in quantized inference where int8 inputs
        are dequantized before matmul.

        Note: Uses fp32 to bypass PyTorch's low-precision prologue heuristic.
        """
        # Simulate quantized input (int8 -> float32)
        x_int = torch.randint(-128, 127, (64, 64), device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        scale = torch.tensor(0.1, device=DEVICE, dtype=torch.float32)
        zero_point = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

        def f(x_int, y, scale, zero_point):
            # Dequantization as prologue
            x_dequant = (x_int - zero_point) * scale
            return matmul_with_prologue(x_dequant, y)

        eager_result = f(x_int, y, scale, zero_point)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x_int, y, scale, zero_point)

        baseline = ((x_int - zero_point) * scale) @ y
        # Verify fusion preserves semantics: compiled should match eager exactly
        torch.testing.assert_close(compiled_result, eager_result, rtol=0, atol=0)
        # Sanity check vs baseline (relaxed due to tiled vs non-tiled accumulation order)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-2, atol=0.2)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with dequantization fused as prologue, got {kernel_count}")
        # Scale is loaded from external buffer tensor (e.g., _ext_arg1_1), not embedded as constant
        # Check for external buffer parameter in kernel (e.g., _ext_arg1_1)
        self.assertIn("_ext_", all_code, "Expected external buffer (e.g., _ext_arg1_1) in Triton kernel for prologue")


if __name__ == "__main__":
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    unittest.main()
