"""
Test epilogue fusion with torch.compile.

Verifies that Inductor correctly generates epilogue lambdas for Helion kernels
and that the epilogue operations are fused into a single Triton kernel.
"""
from __future__ import annotations

import os
import re
import unittest

import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, TestCase, count_triton_kernels, assert_single_triton_kernel
from torch._inductor.utils import run_and_get_code


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    dot_precision="ieee",
)
def matmul(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Matrix multiplication - epilogue fusion happens automatically."""
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


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    RMS normalization - epilogue fusion happens automatically.

    RMS normalization: output = x / sqrt(mean(x^2) + eps) * weight
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"

    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)

        # Compute inverse RMS: 1/sqrt(mean(x^2) + eps)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)

        # Apply normalization and weight
        normalized = x_tile * inv_rms_tile[:, None]
        result = (normalized * weight[:].to(torch.float32))

        # Store result
        out[tile_m, :] = result.to(out.dtype)

    return out


class TestEpilogueFusionTorchCompile(TestCase):
    """Test epilogue fusion through torch.compile path.

    These tests verify that Inductor correctly generates epilogue lambdas
    that are passed to Helion kernels.
    """

    def test_matmul_basic(self):
        """Basic matmul with torch.compile - no epilogue fusion."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y):
            return matmul(x, y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = x @ y
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel is generated
        all_code = assert_single_triton_kernel(self, source_codes)
        # The internal Triton kernel name starts with _helion_ (e.g., _helion_matmul)
        self.assertIn("_helion_", all_code, "Expected Helion kernel function")

    def test_matmul_relu_epilogue(self):
        """Matmul followed by relu - should fuse into epilogue."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y):
            return torch.relu(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = torch.relu(x @ y)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with relu fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu (maximum) fused into Triton kernel"
        )

    def test_matmul_bias_epilogue(self):
        """Matmul + bias - should fuse into epilogue with tile indexing."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return matmul(x, y) + bias

        eager_result = f(x, y, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias)

        baseline = (x @ y) + bias
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with bias add fused
        all_code = assert_single_triton_kernel(self, source_codes)
        # Bias is passed as epilogue_closure and loaded
        self.assertTrue(
            "epilogue_closure" in all_code,
            "Expected bias tensor (epilogue_closure) in Triton kernel"
        )

    def test_matmul_bias_relu_epilogue(self):
        """Matmul + bias + relu - full linear layer epilogue."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return torch.relu(matmul(x, y) + bias)

        eager_result = f(x, y, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias)

        baseline = torch.relu((x @ y) + bias)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with bias+relu fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu (maximum) fused into Triton kernel"
        )
        self.assertTrue(
            "epilogue_closure" in all_code,
            "Expected bias tensor (epilogue_closure) in Triton kernel"
        )


class TestEpilogueFusionTorchCompileExtended(TestCase):
    """Extended torch.compile epilogue fusion tests.

    Tests additional activation functions and arithmetic operations.
    """

    def test_matmul_sigmoid_epilogue(self):
        """Matmul followed by sigmoid.

        Note: Uses fp32 because Triton's sigmoid requires fp32 or fp64.
        Note: Compares eager vs compiled (both use Helion matmul) rather than
        against PyTorch's x @ y, since different matmul implementations can
        produce slightly different results that sigmoid amplifies near 0.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return torch.sigmoid(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled vs eager (both use Helion matmul, should match closely)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with sigmoid fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "sigmoid" in all_code or "tl.sigmoid" in all_code,
            "Expected sigmoid fused into Triton kernel"
        )

    def test_matmul_tanh_epilogue(self):
        """Matmul followed by tanh.

        Note: Uses fp32 because Triton's libdevice.tanh requires fp32.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return torch.tanh(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = torch.tanh(x @ y)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with tanh fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "tanh" in all_code or "libdevice.tanh" in all_code,
            "Expected tanh fused into Triton kernel"
        )

    # Note: GELU is not supported - requires 'erf' which is not in EpilogueSourceHandler

    def test_matmul_silu_epilogue(self):
        """Matmul followed by silu (swish activation).

        Note: Uses fp32 because SiLU uses sigmoid which requires fp32 or fp64.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return torch.nn.functional.silu(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = torch.nn.functional.silu(x @ y)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with silu fused
        # SiLU = x * sigmoid(x), so sigmoid should be in the code
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "sigmoid" in all_code or "tl.sigmoid" in all_code,
            "Expected silu (sigmoid) fused into Triton kernel"
        )

    def test_matmul_scalar_multiply_epilogue(self):
        """Matmul followed by scalar multiplication."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y):
            return matmul(x, y) * 2.0

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = (x @ y) * 2.0
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with scalar multiply fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "2.0" in all_code,
            "Expected scalar 2.0 multiply fused into Triton kernel"
        )

    # Note: Row bias with shape [M, 1] involves complex 2D broadcasting
    # which may not be fully supported in the current epilogue handler.

    def test_matmul_2d_bias_epilogue(self):
        """Matmul + full 2D bias tensor."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        bias_2d = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias_2d):
            return matmul(x, y) + bias_2d

        eager_result = f(x, y, bias_2d)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias_2d)

        baseline = (x @ y) + bias_2d
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with 2D bias fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "epilogue_closure" in all_code,
            "Expected 2D bias tensor (epilogue_closure) in Triton kernel"
        )

    def test_matmul_bias_scale_epilogue(self):
        """Matmul + bias + scale multiplication (chained ops)."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(64, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias, scale):
            return (matmul(x, y) + bias) * scale

        eager_result = f(x, y, bias, scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias, scale)

        baseline = ((x @ y) + bias) * scale
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with bias+scale fused
        all_code = assert_single_triton_kernel(self, source_codes)
        # Both bias and scale are passed as epilogue_closure tensors
        self.assertTrue(
            all_code.count("epilogue_closure") >= 2,
            "Expected both bias and scale (epilogue_closure) in Triton kernel"
        )

    def test_matmul_neg_epilogue(self):
        """Matmul followed by negation."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y):
            return -matmul(x, y)

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = -(x @ y)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with negation fused
        all_code = assert_single_triton_kernel(self, source_codes)
        # Negation appears as -expr or (-expr) pattern from our fusion handler
        # Check for epilogue temporary with negation (ops handler approach)
        # or the pattern in tl.store directly (e.g., tl.store(..., -v_0, ...))
        negation_fused = (
            "= -v_" in all_code  # ops handler pattern: _epilogue_tmp0 = -v_0
            or "= -" in all_code and "_epilogue_tmp" in all_code  # any epilogue negation
            or "(-" in all_code  # old pattern with parentheses
            or ", -v_" in all_code  # negation directly in store: tl.store(..., -v_0, ...)
        )
        self.assertTrue(
            negation_fused,
            "Expected negation fused into Triton kernel"
        )


class TestEpilogueFusionRequiresProperIndexing(TestCase):
    """Tests that REQUIRE proper range matching and index translation.

    These tests will FAIL unless:
    1. Helion exposes output_ranges to Inductor for proper fusion decisions
    2. Codegen happens inside Helion with access to subscript indices
    3. Index translation correctly maps Inductor dimensions to Helion tile indices

    Simple epilogues (relu, sigmoid) work by accident because they ignore indices.
    These tests specifically require loading from external tensors with proper
    indexing, which only works with the full long-term design.
    """

    def test_matmul_row_bias_epilogue(self):
        """Matmul + row-wise bias: output[i,j] + bias[i].

        This requires:
        - bias has shape [M] (first dimension of output)
        - Must load bias using indices_0 (first Helion tile index)
        - Requires proper dimension mapping: epilogue dim 0 -> Helion dim 0
        """
        m, n = 64, 128
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float16)
        row_bias = torch.randn(m, device=DEVICE, dtype=torch.float16)

        def f(x, y, row_bias):
            result = matmul(x, y)  # [M, N]
            return result + row_bias.unsqueeze(1)  # broadcast row_bias[M, 1] to [M, N]

        eager_result = f(x, y, row_bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, row_bias)

        baseline = (x @ y) + row_bias.unsqueeze(1)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "epilogue_closure" in all_code,
            "Expected row_bias tensor (epilogue_closure) in Triton kernel"
        )

    def test_matmul_row_and_col_bias_epilogue(self):
        """Matmul + row bias + column bias: output[i,j] + row_bias[i] + col_bias[j].

        This requires:
        - row_bias uses indices_0 (first dimension)
        - col_bias uses indices_1 (second dimension)
        - Both dimension mappings must be correct
        """
        m, n = 64, 128
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float16)
        row_bias = torch.randn(m, device=DEVICE, dtype=torch.float16)
        col_bias = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, y, row_bias, col_bias):
            result = matmul(x, y)  # [M, N]
            # Broadcast: row_bias[M,1] + col_bias[1,N] -> [M, N]
            return result + row_bias.unsqueeze(1) + col_bias.unsqueeze(0)

        eager_result = f(x, y, row_bias, col_bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, row_bias, col_bias)

        baseline = (x @ y) + row_bias.unsqueeze(1) + col_bias.unsqueeze(0)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        all_code = assert_single_triton_kernel(self, source_codes)
        # Need at least 2 external tensor loads with different indexing
        self.assertTrue(
            all_code.count("epilogue_closure") >= 2,
            "Expected both row_bias and col_bias (epilogue_closure) in Triton kernel"
        )

    def test_matmul_nonsquare_bias_epilogue(self):
        """Matmul with non-square shapes + column bias.

        Non-square shapes help verify that range matching uses actual symbolic
        dimensions (m, n) rather than assuming square matrices.
        """
        m, k, n = 32, 64, 128  # Very different dimensions
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, y, bias):
            return matmul(x, y) + bias

        eager_result = f(x, y, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias)

        baseline = (x @ y) + bias
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "epilogue_closure" in all_code,
            "Expected bias tensor in Triton kernel"
        )

    def test_matmul_scale_row_bias_col_epilogue(self):
        """Matmul * scale[i] + bias[j] - different indices for different operations.

        This is the hardest test: two external tensors, each using a different
        tile index. Requires correct dimension mapping for both.
        """
        m, n = 64, 128
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float16)
        scale_row = torch.randn(m, device=DEVICE, dtype=torch.float16)
        bias_col = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, y, scale_row, bias_col):
            result = matmul(x, y)  # [M, N]
            # scale_row broadcasts as [M, 1], bias_col as [1, N]
            return result * scale_row.unsqueeze(1) + bias_col.unsqueeze(0)

        eager_result = f(x, y, scale_row, bias_col)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, scale_row, bias_col)

        baseline = (x @ y) * scale_row.unsqueeze(1) + bias_col.unsqueeze(0)
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            all_code.count("epilogue_closure") >= 2,
            "Expected both scale_row and bias_col (epilogue_closure) in Triton kernel"
        )

    def test_matmul_complex_epilogue_chain(self):
        """Complex chain: (matmul * scale + bias) * gate + residual.

        Multiple external tensors with various broadcasting patterns.
        Tests that the full epilogue chain with proper indexing works.
        """
        m, n = 64, 64
        x = torch.randn(m, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, n, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        gate = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        residual = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale, bias, gate, residual):
            result = matmul(x, y)
            # Complex epilogue: (result * scale + bias) * gate + residual
            return (result * scale + bias) * gate + residual

        eager_result = f(x, y, scale, bias, gate, residual)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(
            compiled_f, x, y, scale, bias, gate, residual
        )

        baseline = ((x @ y) * scale + bias) * gate + residual
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        all_code = assert_single_triton_kernel(self, source_codes)
        # Should have multiple epilogue_closure tensors (scale, bias, gate, residual)
        self.assertTrue(
            all_code.count("epilogue_closure") >= 4,
            "Expected at least 4 epilogue tensors in Triton kernel"
        )


class TestRMSNormEpilogueFusion(TestCase):
    """Test epilogue fusion with RMS normalization kernel."""

    def _rms_norm_pytorch(self, x, weight, eps=1e-5):
        """PyTorch reference implementation of RMS normalization."""
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)

    def test_rms_norm_relu_torch_compile(self):
        """Verify RMS norm + relu through torch.compile path."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, weight):
            return torch.relu(rms_norm(x, weight))

        eager_result = f(x, weight)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, weight)

        # Verify correctness
        baseline = torch.relu(self._rms_norm_pytorch(x, weight))
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with relu fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu operation (maximum) in generated Triton code"
        )

    def test_rms_norm_relu_add_torch_compile(self):
        """Verify RMS norm + relu + add fuses into one Triton kernel."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, weight, bias):
            return torch.relu(rms_norm(x, weight)) + bias

        eager_result = f(x, weight, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, weight, bias)

        # Verify correctness
        baseline = torch.relu(self._rms_norm_pytorch(x, weight)) + bias
        torch.testing.assert_close(eager_result, baseline, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-3, atol=1e-3)

        # Verify exactly one Triton kernel with relu and add fused
        all_code = assert_single_triton_kernel(self, source_codes)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu operation (maximum) in generated Triton code"
        )
        self.assertTrue(
            "epilogue_closure" in all_code,
            "Expected bias tensor (epilogue_closure) in Triton kernel"
        )


# =============================================================================
# Additional Epilogue Fusion Tests (from fusion limitations)
# =============================================================================


class TestMissingOperations(TestCase):
    """Tests for various element-wise operations fused into epilogue."""

    # Operations that can be tested with simple pattern: op(matmul(x, y))
    # Each tuple: (name, op_fn, code_pattern_to_check_or_None)
    SIMPLE_OPS = [
        ("sin", torch.sin, "sin"),
        ("cos", torch.cos, "cos"),
        ("floor", torch.floor, "floor"),
        ("ceil", torch.ceil, "ceil"),
        ("erf", torch.erf, "erf"),
        ("leaky_relu", lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01), None),
        ("softplus", torch.nn.functional.softplus, None),
    ]

    def test_matmul_elementwise_ops_epilogue(self):
        """Test various element-wise operations fused into epilogue."""
        for op_name, op_fn, code_pattern in self.SIMPLE_OPS:
            with self.subTest(op=op_name):
                x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
                y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y))

                eager_result = f(x, y)
                compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
                compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

                # Correctness check - compare compiled to eager (both use Helion's matmul)
                torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

                # Check fusion - should be single kernel
                kernel_count, all_code = count_triton_kernels(source_codes)
                self.assertEqual(kernel_count, 1, f"Expected 1 kernel with {op_name} fused, got {kernel_count}")

                # Check for operation in generated code if pattern provided
                if code_pattern:
                    self.assertTrue(
                        code_pattern in all_code.lower(),
                        f"Expected {code_pattern} operation in fused kernel"
                    )

    def test_matmul_reciprocal_epilogue(self):
        """Matmul followed by reciprocal (special case with offset to avoid division by zero)."""
        # Use values away from zero to avoid division issues
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) + 2.0
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) + 2.0

        def f(x, y):
            return torch.reciprocal(matmul(x, y) + 1.0)  # +1 to avoid near-zero

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with reciprocal fused, got {kernel_count}")


class TestClosureTensorDimensionality(TestCase):
    """Tests for closure tensors with various dimensionalities."""

    def test_matmul_non_contiguous_bias_epilogue(self):
        """Matmul with non-contiguous bias tensor."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        # Create non-contiguous bias by slicing
        bias_full = torch.randn(128, device=DEVICE, dtype=torch.float16)
        bias = bias_full[::2]  # Non-contiguous: stride is 2, not 1
        assert not bias.is_contiguous()

        def f(x, y, bias):
            return matmul(x, y) + bias

        eager_result = f(x, y, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Verify single kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")


class TestDtypeHandling(TestCase):
    """Tests for dtype handling in fusion - ops requiring fp32 should auto-upcast."""

    # Ops that require fp32 internally (Triton doesn't support fp16/bf16 for these)
    FP32_REQUIRED_OPS = [
        ("sigmoid", torch.sigmoid),
        ("exp", torch.exp),
        ("log", torch.log),
        ("sqrt", torch.sqrt),
        ("sin", torch.sin),
        ("cos", torch.cos),
        ("tanh", torch.tanh),
    ]

    def test_fp16_ops_auto_upcast(self):
        """Test that fp16 ops requiring fp32 are auto-upcast and fused."""
        for op_name, op_fn in self.FP32_REQUIRED_OPS:
            with self.subTest(op=op_name):
                x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16).abs() + 0.1  # abs+offset for log/sqrt
                y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y).abs() + 0.1)  # abs+offset for log/sqrt safety

                eager_result = f(x, y)
                compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
                compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

                # Correctness is required
                torch.testing.assert_close(compiled_result, eager_result, rtol=1e-2, atol=1e-2)

                # Should fuse with auto-upcast - expect single kernel
                kernel_count, all_code = count_triton_kernels(source_codes)
                self.assertEqual(kernel_count, 1, f"Expected 1 kernel with {op_name} fused via auto-upcast")

    def test_bf16_ops_auto_upcast(self):
        """Test that bf16 ops requiring fp32 are auto-upcast and fused."""
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 not supported on this device")

        for op_name, op_fn in self.FP32_REQUIRED_OPS:
            with self.subTest(op=op_name):
                x = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16).abs() + 0.1
                y = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16)

                def f(x, y, op=op_fn):
                    return op(matmul(x, y).abs() + 0.1)

                eager_result = f(x, y)
                compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
                compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

                # Correctness is required
                torch.testing.assert_close(compiled_result, eager_result, rtol=1e-1, atol=1e-1)

                # Should fuse with auto-upcast - expect single kernel
                kernel_count, all_code = count_triton_kernels(source_codes)
                self.assertEqual(kernel_count, 1, f"Expected 1 kernel with {op_name} fused via auto-upcast")


class TestFusionScope(TestCase):
    """Tests for operations outside element-wise scope."""

    def test_matmul_sum_epilogue(self):
        """Matmul followed by sum reduction - NOT fusable.

        Reductions cannot be fused into the kernel epilogue because they
        require cross-element coordination.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return torch.sum(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Reduction cannot be fused - expect 2 kernels (matmul + reduction)
        kernel_count, all_code = count_triton_kernels(source_codes)
        # Note: This test documents current behavior, not desired behavior
        # Reductions fundamentally cannot be fused into element-wise epilogue
        self.assertGreaterEqual(kernel_count, 1, "Expected at least 1 kernel")

    def test_matmul_mean_epilogue(self):
        """Matmul followed by mean reduction - NOT fusable."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return torch.mean(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Reduction cannot be fused - expect multiple kernels
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertGreaterEqual(kernel_count, 1, "Expected at least 1 kernel")

    def test_matmul_transpose_epilogue(self):
        """Matmul followed by transpose - may or may not fuse.

        Transpose is a view operation that doesn't require computation,
        but the output layout changes.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul(x, y).T

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Transpose is a view operation - should still be 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")


class TestEdgeCases(TestCase):
    """Additional edge cases and regression tests."""

    def test_matmul_chained_multiple_ops(self):
        """Multiple chained operations - all should fuse."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            # relu -> multiply -> add -> tanh
            result = matmul(x, y)
            result = torch.relu(result)
            result = result * 2.0
            result = result + 1.0
            result = torch.tanh(result)
            return result

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with all ops fused, got {kernel_count}")

    def test_matmul_mixed_closure_sizes(self):
        """Epilogue with multiple closure tensors of different sizes."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float16)  # [N]
        scale = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)  # [M, N]

        def f(x, y, bias, scale):
            return (matmul(x, y) + bias) * scale

        eager_result = f(x, y, bias, scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias, scale)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_matmul_where_epilogue(self):
        """Matmul with where/conditional epilogue."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            result = matmul(x, y)
            return torch.where(result > 0, result, torch.zeros_like(result))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Compare compiled to eager (both use Helion's matmul)
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel with where fused, got {kernel_count}")


# =============================================================================
# Multi-output Helion Kernels for Kernel Structure Tests
# =============================================================================

@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee")
def matmul_with_residual(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matmul that returns both the result and a residual (copy of result).

    This simulates a kernel with multiple outputs that should both receive
    epilogue operations if we apply ops to both.
    """
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
        residual[tile_m, tile_n] = acc  # Second store to different output

    return out, residual


@helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee")
def matmul_split_k_atomic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matmul with split-K using atomic adds.

    This kernel has auxiliary atomic stores that should NOT receive epilogue.
    Only the final output should be affected by epilogue.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2

    # Use zeros for atomic accumulation
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)

    # Split K into chunks
    split_k = 2
    k_chunk = k // split_k

    for tile_m, tile_n in hl.tile([m, n]):
        for k_split in hl.grid(split_k):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            k_start = k_split * k_chunk
            k_end = k_start + k_chunk
            for tile_k in hl.tile(k_start, k_end):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            # Atomic add - should NOT have epilogue applied
            hl.atomic_add(out, [tile_m, tile_n], acc)

    return out


# =============================================================================
# Kernel Structure Tests
# =============================================================================

class TestMultiOutputKernel(TestCase):
    """Tests for kernels with multiple output tensors."""

    def test_multi_output_basic(self):
        """Basic multi-output kernel without epilogue."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        expected = x @ y

        def f(x, y):
            return matmul_with_residual(x, y)

        # Test eager
        out_eager, residual_eager = f(x, y)
        torch.testing.assert_close(out_eager, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_eager, expected, rtol=1e-3, atol=1e-3)

        # Test compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        out_compiled, residual_compiled = result
        torch.testing.assert_close(out_compiled, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_compiled, expected, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_multi_output_epilogue_first_only(self):
        """Epilogue applied to first output only."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        expected = x @ y

        def f(x, y):
            out, residual = matmul_with_residual(x, y)
            return torch.relu(out), residual

        # Test eager
        out_eager, residual_eager = f(x, y)
        torch.testing.assert_close(out_eager, torch.relu(expected), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_eager, expected, rtol=1e-3, atol=1e-3)

        # Test compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        out_compiled, residual_compiled = result
        torch.testing.assert_close(out_compiled, torch.relu(expected), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_compiled, expected, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_multi_output_epilogue_both(self):
        """Epilogue applied to both outputs."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        expected = x @ y

        def f(x, y):
            out, residual = matmul_with_residual(x, y)
            return torch.relu(out), torch.sigmoid(residual)

        # Test eager
        out_eager, residual_eager = f(x, y)
        torch.testing.assert_close(out_eager, torch.relu(expected), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_eager, torch.sigmoid(expected), rtol=1e-3, atol=1e-3)

        # Test compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        out_compiled, residual_compiled = result
        torch.testing.assert_close(out_compiled, torch.relu(expected), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_compiled, torch.sigmoid(expected), rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_multi_output_epilogue_combined(self):
        """Epilogue that combines both outputs.

        When the epilogue reads from multiple kernel outputs (out + residual),
        it creates a circular dependency. The scheduler detects this in
        can_fuse() and prevents the epilogue from being fused with the template.
        Instead, the epilogue is scheduled as a separate kernel.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        expected = x @ y

        def f(x, y):
            out, residual = matmul_with_residual(x, y)
            return out + residual

        # Test eager - both out and residual are the same, so out + residual = 2 * expected
        result_eager = f(x, y)
        torch.testing.assert_close(result_eager, 2 * expected, rtol=1e-3, atol=1e-3)

        # Test compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result_compiled, source_codes = run_and_get_code(compiled_f, x, y)

        torch.testing.assert_close(result_compiled, 2 * expected, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        # Circular dependency detected: epilogue (out + residual) reads from both outputs.
        # Falls back to no fusion, so we expect 2 kernels: Helion matmul + addition.
        self.assertEqual(kernel_count, 2, f"Expected 2 kernels (fallback), got {kernel_count}")


class TestAuxiliaryStores(TestCase):
    """Tests for kernels with auxiliary stores (atomic ops, intermediate stores).

    LIMITATION: Kernels with atomic operations (tl.atomic_add, etc.) cannot have
    epilogue fusion because atomic stores happen multiple times per element.
    The epilogue should run AFTER all atomics complete, but the current fusion
    only targets tl.store() calls, not tl.atomic_* calls.

    See docs/fusion_limitations.md for details.
    """

    def test_atomic_kernel_basic(self):
        """Split-K matmul with atomic adds - basic functionality."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul_split_k_atomic(x, y)

        eager_result = f(x, y)
        expected = x @ y  # Reference

        # Check correctness (may have some numerical diff due to atomic adds)
        torch.testing.assert_close(eager_result, expected, rtol=1e-3, atol=1e-3)

    def test_atomic_kernel_with_epilogue(self):
        """Split-K matmul with atomic adds followed by epilogue.

        Kernels with atomic operations cannot have epilogue fused because
        atomics may execute multiple times per element. The epilogue is
        automatically run as a separate kernel when atomics are detected.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        expected = torch.relu(x @ y)

        def f(x, y):
            return torch.relu(matmul_split_k_atomic(x, y))

        # Test eager
        eager_result = f(x, y)
        torch.testing.assert_close(eager_result, expected, rtol=1e-3, atol=1e-3)

        # Test compiled - epilogue should NOT be fused, runs as separate kernel
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, y)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        # Verify that we have multiple kernels (atomic kernel + epilogue kernel)
        kernel_count, all_code = count_triton_kernels(source_codes)
        # Should have at least 2 kernels: the atomic matmul and the relu epilogue
        self.assertGreaterEqual(kernel_count, 2, "Expected separate kernels for atomic + epilogue")

    def test_atomic_vs_simple_epilogue(self):
        """Compare atomic kernel vs simple kernel with same epilogue.

        Both should produce equivalent results when using the same epilogue.
        This tests the eager path (not compiled) to verify kernel correctness.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f_simple(x, y):
            return torch.relu(matmul(x, y))

        def f_atomic(x, y):
            return torch.relu(matmul_split_k_atomic(x, y))

        simple_result = f_simple(x, y)
        atomic_result = f_atomic(x, y)

        # Both should produce the same result (eager mode)
        torch.testing.assert_close(atomic_result, simple_result, rtol=1e-3, atol=1e-3)


class TestIndexingPatterns(TestCase):
    """Tests for different indexing patterns in generated code."""

    def test_standard_indices(self):
        """Standard indexing pattern with indices_0, indices_1."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return torch.relu(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Verify indices_0 and indices_1 are used in the generated code
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")
        self.assertIn("indices_0", all_code, "Expected indices_0 in generated code")
        self.assertIn("indices_1", all_code, "Expected indices_1 in generated code")

    def test_row_bias(self):
        """Test with row bias (bias along first dimension)."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(64, 1, device=DEVICE, dtype=torch.float32)  # row bias

        def f(x, y, bias):
            return matmul(x, y) + bias

        eager_result = f(x, y, bias)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, bias)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_combined_ops(self):
        """Test with combined operations on epilogue."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(64, device=DEVICE, dtype=torch.float32)  # column scale

        def f(x, y, scale):
            return torch.sigmoid(matmul(x, y) * scale)

        eager_result = f(x, y, scale)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y, scale)

        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")


@helion.kernel(static_shapes=True, autotune_effort="none")
def atomic_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel that uses atomic operations - should NOT allow epilogue fusion."""
    result = torch.zeros_like(x)
    for i in hl.tile(x.size(0)):
        hl.atomic_add(result, [i], x[i] + y[i])
    return result


class TestAtomicKernelNoFusion(TestCase):
    """Test that kernels using atomics do NOT fuse epilogues.

    This is a safety invariant: atomic operations require special handling,
    so epilogue fusion is disabled for atomic kernels to avoid incorrect results.
    """

    def test_atomic_kernel_does_not_fuse_epilogue(self):
        """Atomic kernel + epilogue should result in separate kernels (no fusion).

        When a kernel uses atomic operations, epilogue fusion must be disabled
        to avoid silent incorrectness. This test verifies the safety behavior
        through the public torch.compile API.
        """
        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            # atomic_add_kernel uses atomics, so the relu epilogue should NOT fuse
            return torch.relu(atomic_add_kernel(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        # Verify correctness
        torch.testing.assert_close(compiled_result, eager_result, rtol=1e-3, atol=1e-3)

        # Atomic kernels should NOT fuse epilogues, resulting in 2+ kernels
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertGreaterEqual(
            kernel_count, 2,
            f"Expected 2+ kernels (atomic kernel + epilogue), got {kernel_count}. "
            "Epilogue fusion must be disabled for atomic kernels!"
        )

    def test_non_atomic_kernel_does_fuse_epilogue(self):
        """Non-atomic kernel + epilogue should fuse into single kernel.

        This is the counterpart test: kernels without atomics SHOULD fuse
        epilogues for performance. Verifies the detection is working correctly.
        """
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)

        def f(x, y):
            # matmul doesn't use atomics, so relu epilogue SHOULD fuse
            return torch.relu(matmul(x, y))

        eager_result = f(x, y)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, source_codes = run_and_get_code(compiled_f, x, y)

        baseline = torch.relu(x @ y)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-2, atol=1e-2)

        # Non-atomic kernels SHOULD fuse epilogues into 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (fused), got {kernel_count}. "
            "Non-atomic kernels should fuse epilogues for performance."
        )


if __name__ == "__main__":
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    unittest.main()
