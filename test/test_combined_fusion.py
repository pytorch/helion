"""
Test combined prologue and epilogue fusion with a single RMS norm style kernel.

This test verifies that a single Helion kernel can fuse:
- Prologue: pointwise ops applied to inputs BEFORE kernel execution
- Epilogue: pointwise ops applied to outputs AFTER kernel execution

Requirements tested:
- Multiple inputs (x, weight, scale, bias)
- Multiple outputs (normalized tensor, residual tensor, scalar)
- Non-tensor output (scalar value)
- Both prologue and epilogue fusion in one kernel invocation
"""

from __future__ import annotations

import os
import unittest

import torch
from torch._inductor.utils import run_and_get_code

import helion
from helion._testing import DEVICE, TestCase, count_triton_kernels
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_with_different_shapes_same_elements(x):  # pyrefly: ignore
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


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_2d(x, scale):  # pyrefly: ignore
    """Simple 2D elementwise kernel for testing partial flattening.

    Takes [M, N] input, returns [M, N] output.
    The prologue fusion test will pass [d1, d2*d3] -> view -> [d1*d2, d3].

    Returns:
        out: Output tensor [M, N]
        info: Scalar value (42)
    """
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out, 42


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_2d_single_output(x, scale):  # pyrefly: ignore
    """Simple 2D elementwise kernel returning SINGLE output (no scalar).

    Takes [M, N] input, returns [M, N] output.
    Used to test single-output fusion cases.

    Returns:
        out: Output tensor [M, N]
    """
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_3d(x, scale):  # pyrefly: ignore
    """Simple 3D elementwise kernel for testing 3D transpose fusion.

    Takes [D1, D2, D3] input, returns [D1, D2, D3] output.

    Returns:
        out: Output tensor [D1, D2, D3]
        info: Scalar value (42)
    """
    d1, d2, d3 = x.size()
    out = torch.empty_like(x)

    for tile_d1 in hl.tile(d1):
        for tile_d2 in hl.tile(d2):
            x_tile = x[tile_d1, tile_d2, :]
            out[tile_d1, tile_d2, :] = x_tile * scale

    return out, 42


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def elementwise_4d(x, scale):  # pyrefly: ignore
    """Simple 4D elementwise kernel for testing 4D transpose fusion.

    Takes [B, H, S, D] input, returns [B, H, S, D] output.

    Returns:
        out: Output tensor [B, H, S, D]
        info: Scalar value (42)
    """
    b, h, s, d = x.size()
    out = torch.empty_like(x)

    for tile_b in hl.tile(b):
        for tile_h in hl.tile(h):
            for tile_s in hl.tile(s):
                x_tile = x[tile_b, tile_h, tile_s, :]
                out[tile_b, tile_h, tile_s, :] = x_tile * scale

    return out, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True
)
def rms_norm_multi_output(x, weight, eps=1e-5):  # pyrefly: ignore
    """RMS normalization returning multiple outputs including non-tensor.

    Returns:
        out: Normalized tensor [M, N]
        residual: Residual tensor [M, N] (same shape as out for uniform fusion)
        info: Non-tensor scalar value (42)
    """
    m, n = x.size()
    assert weight.size(0) == n
    out = torch.empty_like(x)
    residual = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        result = normalized * weight[:].to(torch.float32)
        out[tile_m, :] = result.to(out.dtype)
        # Residual stores the normalized values before weight scaling
        residual[tile_m, :] = normalized.to(out.dtype)

    return out, residual, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True
)
def rms_norm_multi_output_extended(x, weight, eps=1e-5):  # pyrefly: ignore
    """RMS normalization with extended outputs for combined fusion testing.

    Returns:
        out: Normalized tensor [M, N]
        residual: Residual tensor [M, N] (same shape as out)
        info1: Non-tensor scalar value (42)
        extra: Tensor [M, N] (same shape as out)
        info2: Non-tensor scalar value (99)
    """
    m, n = x.size()
    assert weight.size(0) == n
    out = torch.empty_like(x)
    residual = torch.empty_like(x)
    extra = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        result = normalized * weight[:].to(torch.float32)
        out[tile_m, :] = result.to(out.dtype)
        # Residual stores the normalized values before weight scaling
        residual[tile_m, :] = normalized.to(out.dtype)
        # Extra stores double the normalized values (same shape)
        extra[tile_m, :] = (normalized * 2).to(out.dtype)

    return out, residual, 42, extra, 99


class TestCombinedFusion(TestCase):
    """Test combined prologue and epilogue fusion with RMS norm kernel."""

    def test_rms_norm_epilogue_only(self):
        """RMS norm with epilogue fusion only (no prologue) - verify base case."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, out_bias, res_bias):
            out, residual, info = rms_norm_multi_output(x, weight)
            # Epilogue: apply ops to both outputs
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        inputs = (x, weight, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_rms_norm_prologue_only(self):
        """RMS norm with prologue fusion only (no epilogue) - verify base case."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, x_scale):
            # Prologue: transform input
            x_transformed = torch.sigmoid(torch.relu(x * x_scale))
            return rms_norm_multi_output(x_transformed, weight)

        inputs = (x, weight, x_scale)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_rms_norm_prologue_epilogue_multi_output(self):
        """RMS norm with chained prologue/epilogue ops, multiple outputs including scalars.

        This test includes partial flatten cases for both prologue and epilogue:
        - Partial flatten prologue: [d1, d2*d3] -> view -> [d1*d2, d3] (kernel input)
        - Partial flatten epilogue: [d1*d2, d3] -> view -> [d1, d2*d3] (extra output)

        Prologue chains (3+ ops):
        - x_partial: [d1, d2*d3] -> scale -> relu -> view to [d1*d2, d3]
        - x: [d1*d2, d3] -> add bias -> sigmoid before RMS norm

        Epilogue chains (each tensor output has its own chain):
        - out: tanh((relu(out) + out_bias) * out_scale) - same shape [d1*d2, d3]
        - residual: sigmoid((residual + res_bias) * res_scale) - same shape [d1*d2, d3]
        - extra: view to [d1, d2*d3] -> add bias -> relu - PARTIAL FLATTEN epilogue

        Multiple inputs:
        - x_partial: input tensor [d1, d2*d3] (partial flatten prologue)
        - x_partial_scale: prologue scale [d1, d2*d3]
        - x_bias: prologue bias [d3] (applied after view)
        - weight: RMS norm weight [d3]
        - out_bias, out_scale: epilogue for out [d3]
        - res_bias, res_scale: epilogue for residual [d3]
        - extra_bias: epilogue for extra [d1, d2*d3] (partial flatten epilogue)

        Multiple outputs (5 total, interleaved tensors and scalars):
        - Tensor: processed normalized output [d1*d2, d3]
        - Tensor: processed residual [d1*d2, d3]
        - Non-tensor: scalar value (42)
        - Tensor: processed extra [d1, d2*d3] (partial flatten epilogue)
        - Non-tensor: scalar value (99)
        """
        d1, d2, d3 = 8, 16, 256
        m, n = d1 * d2, d3  # Kernel operates on [128, 256]

        # Partial flatten prologue input [d1, d2*d3] = [8, 4096]
        x_partial = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        x_partial_scale = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        # Same-shape prologue bias [d3] = [256]
        x_bias = torch.randn(d3, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Epilogue closure tensors for main output (same shape)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Epilogue closure tensors for residual output (same shape)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Partial flatten epilogue bias [d1, d2*d3] = [8, 4096]
        extra_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

        def f(x_partial, x_partial_scale, x_bias, weight, out_bias, out_scale, res_bias, res_scale, extra_bias):
            # Partial flatten prologue [d1, d2*d3] -> [d1*d2, d3]
            x_scaled = x_partial * x_partial_scale  # [d1, d2*d3]
            x_relu = torch.relu(x_scaled)  # [d1, d2*d3]
            x_viewed = x_relu.view(m, n)  # [d1*d2, d3]

            # Same-shape prologue on viewed tensor
            x_biased = x_viewed + x_bias  # [d1*d2, d3]
            x_transformed = torch.sigmoid(x_biased)  # [d1*d2, d3]

            # Kernel execution (returns 5 values: 3 tensors + 2 scalars, interleaved)
            out, residual, info1, extra, info2 = rms_norm_multi_output_extended(x_transformed, weight)

            # Epilogue chain for out (4 ops): relu -> add bias -> scale -> tanh
            out_relu = torch.relu(out)
            out_biased = out_relu + out_bias
            out_scaled = out_biased * out_scale
            out_processed = torch.tanh(out_scaled)

            # Epilogue chain for residual (3 ops): add bias -> scale -> sigmoid
            res_biased = residual + res_bias
            res_scaled = res_biased * res_scale
            res_processed = torch.sigmoid(res_scaled)

            # Partial flatten epilogue for extra: [d1*d2, d3] -> [d1, d2*d3] -> ops
            extra_viewed = extra.view(d1, d2 * d3)  # [d1, d2*d3]
            extra_biased = extra_viewed + extra_bias  # [d1, d2*d3]
            extra_processed = torch.relu(extra_biased)  # [d1, d2*d3]

            return out_processed, res_processed, info1, extra_processed, info2

        inputs = (
            x_partial, x_partial_scale, x_bias, weight, out_bias, out_scale, res_bias, res_scale, extra_bias
        )

        # Run eager
        out_eager, res_eager, info1_eager, extra_eager, info2_eager = f(*inputs)
        self.assertEqual(info1_eager, 42)
        self.assertEqual(info2_eager, 99)

        # Run compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        # Verify outputs
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

        out_compiled, res_compiled, info1_compiled, extra_compiled, info2_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(extra_compiled, extra_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info1_compiled, 42)
        self.assertEqual(info2_compiled, 99)

        # Verify output shapes:
        # - out, residual: same shape [d1*d2, d3] = [128, 256]
        # - extra: partial flatten epilogue [d1, d2*d3] = [8, 4096]
        self.assertEqual(list(out_compiled.shape), [m, n])
        self.assertEqual(list(res_compiled.shape), [m, n])
        self.assertEqual(list(extra_compiled.shape), [d1, d2 * d3])

        # Verify fusion: should be single kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 fused kernel, got {kernel_count}"
        )

        # Verify prologue fusion occurred
        self.assertIn(
            "prologue_input",
            all_code,
            "Expected prologue fusion (prologue_input in generated code)",
        )

        # Verify epilogue fusion occurred (out, residual, and extra epilogues)
        epilogue_count = all_code.count("epilogue_input")
        self.assertGreaterEqual(
            epilogue_count,
            3,
            f"Expected at least 3 epilogue_input occurrences (out, residual, extra), got {epilogue_count}",
        )

        # Verify relu is in the fused kernel (prologue and out epilogue)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code,
            "Expected relu (maximum) in fused kernel",
        )

        # Verify tanh is in the fused kernel (out epilogue)
        self.assertTrue(
            "tanh" in all_code or "libdevice.tanh" in all_code,
            "Expected tanh in fused kernel",
        )

        # Verify sigmoid is in the fused kernel (prologue and residual epilogue)
        self.assertTrue(
            "sigmoid" in all_code or "tl.sigmoid" in all_code,
            "Expected sigmoid in fused kernel",
        )

    def test_reshaped_output_epilogue_fuses_to_1_kernel(self):
        """Epilogue on reshaped output [M,N] -> [M*N] fuses into 1 kernel.

        This test verifies that reshaped epilogues can fuse via index transformation:
        1. Kernel stores with 2D indices [row, col]
        2. Epilogue on reshaped output needs 1D indices
        3. View.dynamic_reshape_indexer transforms the indices correctly
        4. Extra store is generated to the epilogue's output buffer

        Shapes:
        - Kernel input: x [M, N]
        - Kernel output: out [M, N], residual [M, N]
        - Same-shaped epilogue: out [M, N] -> relu + add (FUSES)
        - Reshaped epilogue: residual.view(M*N) -> sigmoid + add (FUSES)
        """
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        flat_bias = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, out_bias, flat_bias):
            out, residual, info = rms_norm_multi_output(x, weight)

            # Same-shaped epilogue [M, N] - FUSES
            out_processed = torch.relu(out) + out_bias

            # Reshaped epilogue [M*N] - NOW FUSES via index transformation
            flattened = residual.view(m * n)
            flat_processed = torch.sigmoid(flattened) + flat_bias

            return out_processed, flat_processed, info

        inputs = (x, weight, out_bias, flat_bias)

        # Eager
        out_eager, flat_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, flat_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(flat_compiled, flat_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify shapes - flat output has 1D shape [M*N]
        self.assertEqual(list(out_compiled.shape), [m, n])
        self.assertEqual(list(flat_compiled.shape), [m * n])

        # Should use exactly 1 kernel - both epilogues fused
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (reshaped epilogue fused), got {kernel_count}"
        )

    def test_rms_norm_simple_prologue_epilogue(self):
        """RMS norm with simple single-op prologue and epilogue.

        This reproduces the bug where larger sizes cause 'del buf0' error
        because prologue buffer is fused but scheduler still tries to delete it.
        """
        # Use larger size that triggers the bug
        m, n = 16, 32
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, x_scale, out_bias, res_bias):
            # Simple single-op prologue
            x_transformed = x * x_scale
            # Kernel
            out, residual, info = rms_norm_multi_output(x_transformed, weight)
            # Simple single-op epilogue
            out_processed = out + out_bias
            res_processed = residual + res_bias
            return out_processed, res_processed, info

        inputs = (x, weight, x_scale, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

    def test_rms_norm_chain_prologue_chain_epilogue(self):
        """Chained ops: sigmoid(relu(x)) -> rms_norm -> tanh -> add bias.

        Tests that chained prologue ops (relu, sigmoid) and chained epilogue ops
        (tanh, add) all fuse into a single kernel with rms_norm.
        """
        m, n = 64, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, out_bias, res_bias):
            x_transformed = torch.sigmoid(torch.relu(x))
            out, residual, info = rms_norm_multi_output(x_transformed, weight)
            return torch.tanh(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        inputs = (x, weight, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_flat_prologue_reshaped_to_2d_kernel(self):
        """Flat prologue [M*N] and same-shaped prologue [M,N] both fuse into 1 kernel.

        This test verifies that:
        - Flat prologue [M*N] fuses via View.dynamic_reshape_indexer index transformation
        - Same-shaped prologue [M, N] fuses normally
        - Same-shaped epilogue [M, N] fuses normally
        - All fuse into a single Triton kernel

        Shapes:
        - Flat prologue: x_flat [M*N] -> ops -> view to [M, N]
        - Same-shaped prologue: x_bias [M, N] -> added after view
        - Kernel: operates on [M, N]
        - Epilogue: out [M, N] -> same-shaped ops [M, N]
        """
        m, n = 128, 256
        # Flat prologue input [M*N]
        x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        x_flat_scale = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        # Same-shaped prologue input [M, N]
        x_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Same-shaped epilogue biases
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x_flat, x_flat_scale, x_bias, weight, out_bias, res_bias):
            # Flat prologue [M*N] -> view -> [M, N]
            x_scaled = x_flat * x_flat_scale  # [M*N]
            x_transformed = torch.relu(x_scaled)  # [M*N]
            x = x_transformed.view(m, n)  # [M, N]

            # Same-shaped prologue [M, N]
            x = x + x_bias  # [M, N]

            out, residual, info = rms_norm_multi_output(x, weight)

            # Epilogue on SAME shaped outputs [M, N]
            out_processed = torch.relu(out) + out_bias  # [M, N]
            res_processed = torch.sigmoid(residual) + res_bias  # [M, N]
            return out_processed, res_processed, info

        inputs = (x_flat, x_flat_scale, x_bias, weight, out_bias, res_bias)

        # Eager
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shapes are [M, N]
        self.assertEqual(list(out_compiled.shape), [m, n])
        self.assertEqual(list(res_compiled.shape), [m, n])

        # Should use exactly 1 kernel:
        # - Flat prologue [M*N] fused via index transformation
        # - Same-shaped epilogue [M, N] fused normally
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (flat prologue + same-shaped epilogue fused), got {kernel_count}"
        )

        # Verify prologue fusion occurred
        self.assertIn(
            "prologue_input",
            all_code,
            "Expected prologue fusion (prologue_input in generated code)",
        )

        # Verify epilogue fusion occurred
        self.assertIn(
            "epilogue_input",
            all_code,
            "Expected epilogue fusion (epilogue_input in generated code)",
        )

    def test_multi_output_different_shapes_epilogue(self):
        """Differently-shaped epilogue outputs with same element count - should fuse into 1 kernel.

        This test verifies that multi-output kernels with DIFFERENT shaped outputs
        (but same element count) can fuse epilogues into a single kernel.

        Shapes:
        - out1: [M*N] (flat)
        - out2: [M, N] (2D)
        - Both have M*N elements

        Epilogues:
        - out1 [M*N]: relu
        - out2 [M, N]: sigmoid
        """
        m, n = 64, 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x):
            out1, out2, scalar = elementwise_with_different_shapes_same_elements(x)
            return torch.relu(out1), torch.sigmoid(out2), scalar

        # Eager
        out1_eager, out2_eager, scalar_eager = f(x)
        self.assertEqual(scalar_eager, 42)

        # Compiled
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

        # Should use exactly 1 kernel - both epilogues fused
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (differently-shaped epilogues fused), got {kernel_count}"
        )

    def test_partially_flat_prologue_correctness(self):
        """Partially flat prologue [d1, d2*d3] -> kernel [d1*d2, d3] - fuses to 1 kernel.

        This tests the case where the prologue has a partially flattened shape that
        differs from the kernel's expected shape. View.dynamic_reshape_indexer handles
        the index transformation from kernel indices to prologue indices.

        The partial flattening case [d1, d2*d3] -> [d1*d2, d3]:
        1. Kernel operates on [d1*d2, d3] (2D indices)
        2. Prologue data is [d1, d2*d3]
        3. View.dynamic_reshape_indexer transforms kernel indices to prologue indices
        4. Uses ModularIndexing to compute correct offsets

        This now fuses to 1 kernel thanks to:
        - PyTorch scheduler bypassing can_fuse_vertical for prologue fusion
        - Using TritonPrinter (texpr) to properly convert ModularIndexing to Triton code

        Shapes:
        - Prologue input: x_partial [d1, d2*d3] -> ops -> view to [d1*d2, d3]
        - Kernel: operates on [d1*d2, d3] (2D)
        - Epilogue: out [d1*d2, d3] -> same-shaped ops (fuses)
        """
        d1, d2, d3 = 8, 16, 32
        # Partially flat prologue input [d1, d2*d3]
        x_partial = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        # Same-shaped epilogue bias
        out_bias = torch.randn(d3, device=DEVICE, dtype=torch.float32)

        def f(x_partial, x_scale, out_bias):
            # Prologue on PARTIALLY FLAT input [d1, d2*d3]
            x_scaled = x_partial * x_scale  # [d1, d2*d3]
            x_transformed = torch.relu(x_scaled)  # [d1, d2*d3]
            # Reshape to kernel's expected 2D shape [d1*d2, d3]
            # This is a partial flatten: [d1, d2*d3] -> [d1*d2, d3]
            x = x_transformed.view(d1 * d2, d3)

            out, info = elementwise_2d(x, kernel_scale)

            # Epilogue on SAME shaped output [d1*d2, d3]
            out_processed = torch.sigmoid(out) + out_bias
            return out_processed, info

        inputs = (x_partial, x_scale, out_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [d1*d2, d3]
        self.assertEqual(list(out_compiled.shape), [d1 * d2, d3])

        # Partial flatten prologue [d1, d2*d3] -> [d1*d2, d3] now fuses to 1 kernel!
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (partial flatten prologue DOES fuse), got {kernel_count}"
        )

        # Verify prologue fusion occurred
        self.assertIn(
            "prologue_input",
            all_code,
            "Expected prologue fusion (prologue_input in generated code)",
        )

    def test_partially_flat_epilogue_correctness(self):
        """Partially flat epilogue [d1*d2, d3] -> [d1, d2*d3] - fuses to 1 kernel.

        This tests the case where the epilogue has a partially flattened shape that
        differs from the kernel's output shape. View.dynamic_reshape_indexer handles
        the index transformation from kernel indices to epilogue indices.

        The partial flattening case [d1*d2, d3] -> [d1, d2*d3]:
        1. Kernel operates on [d1*d2, d3] (2D indices) and outputs [d1*d2, d3]
        2. Epilogue reshapes to [d1, d2*d3]
        3. View.dynamic_reshape_indexer transforms kernel indices to epilogue indices
        4. Uses ModularIndexing to compute correct offsets

        This mirrors test_partially_flat_prologue_correctness but for epilogue.

        Shapes:
        - Kernel: operates on [d1*d2, d3], outputs [d1*d2, d3]
        - Epilogue: out [d1*d2, d3] -> view to [d1, d2*d3] -> ops (fuses)
        """
        d1, d2, d3 = 8, 16, 32
        # Kernel input [d1*d2, d3]
        x = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        # Partially flat epilogue bias [d1, d2*d3]
        out_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

        def f(x, out_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Epilogue: reshape to PARTIALLY FLAT [d1, d2*d3] and apply ops
            # This is a partial unflatten: [d1*d2, d3] -> [d1, d2*d3]
            out_reshaped = out.view(d1, d2 * d3)  # [d1, d2*d3]
            out_processed = torch.sigmoid(out_reshaped) + out_bias  # [d1, d2*d3]
            return out_processed, info

        inputs = (x, out_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [d1, d2*d3]
        self.assertEqual(list(out_compiled.shape), [d1, d2 * d3])

        # Partial flatten epilogue [d1*d2, d3] -> [d1, d2*d3] now fuses to 1 kernel!
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (partial flatten epilogue DOES fuse), got {kernel_count}"
        )

        # Verify epilogue fusion occurred
        self.assertIn(
            "epilogue_input",
            all_code,
            "Expected epilogue fusion (epilogue_input in generated code)",
        )


    def test_slice_prologue_column_slice(self):
        """Column slice prologue x[:, half:] -> kernel - should fuse to 1 kernel.

        Tests that slicing the input tensor along columns before passing to kernel
        can fuse the prologue operations into the kernel.

        Shapes:
        - Original input: x [M, N]
        - Sliced input: x[:, half:] = [M, N//2]
        - Kernel: operates on [M, N//2]
        """
        m, n = 128, 256
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Slice prologue: take right half of columns
            x_sliced = x[:, half_n:]  # [M, N//2]
            x_scaled = x_sliced * x_scale  # [M, N//2]
            x_transformed = torch.relu(x_scaled)  # [M, N//2]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, N//2]
        self.assertEqual(list(out_compiled.shape), [m, half_n])

        # Should use 1 kernel - slice prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (slice prologue fused), got {kernel_count}"
        )

    def test_slice_prologue_row_slice(self):
        """Row slice prologue x[half:, :] -> kernel - should fuse to 1 kernel.

        Tests that slicing the input tensor along rows before passing to kernel
        can fuse the prologue operations into the kernel.

        Shapes:
        - Original input: x [M, N]
        - Sliced input: x[half:, :] = [M//2, N]
        - Kernel: operates on [M//2, N]
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Slice prologue: take bottom half of rows
            x_sliced = x[half_m:, :]  # [M//2, N]
            x_scaled = x_sliced * x_scale  # [M//2, N]
            x_transformed = torch.sigmoid(x_scaled)  # [M//2, N]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M//2, N]
        self.assertEqual(list(out_compiled.shape), [half_m, n])

        # Should use 1 kernel - slice prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (slice prologue fused), got {kernel_count}"
        )

    def test_slice_epilogue_column_slice(self):
        """Column slice epilogue out[:, :half] -> ops - fuses to 1 kernel.

        Tests that slicing the kernel output along columns and applying ops.
        Epilogue slicing generates conditional stores with bounds checking.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N]
        - Epilogue: out[:, :half] = [M, N//2] -> ops
        """
        m, n = 128, 256
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Epilogue: slice left half of columns and apply ops
            out_sliced = out[:, :half_n]  # [M, N//2]
            out_processed = torch.relu(out_sliced) + epilogue_bias  # [M, N//2]
            return out_processed, info

        inputs = (x, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, N//2]
        self.assertEqual(list(out_compiled.shape), [m, half_n])

        # Epilogue slice fusion works - fuses to 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (epilogue slice fused), got {kernel_count}"
        )

    def test_slice_epilogue_row_slice(self):
        """Row slice epilogue out[:half, :] -> ops - fuses to 1 kernel.

        Tests that slicing the kernel output along rows and applying ops.
        Epilogue slicing generates conditional stores with bounds checking.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N]
        - Epilogue: out[:half, :] = [M//2, N] -> ops
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Epilogue: slice top half of rows and apply ops
            out_sliced = out[:half_m, :]  # [M//2, N]
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias  # [M//2, N]
            return out_processed, info

        inputs = (x, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M//2, N]
        self.assertEqual(list(out_compiled.shape), [half_m, n])

        # Epilogue slice fusion works - fuses to 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (epilogue slice fused), got {kernel_count}"
        )

    def test_narrow_prologue(self):
        """Narrow prologue x.narrow(1, start, length) -> kernel - should fuse to 1 kernel.

        Tests that using torch.narrow() on input before passing to kernel
        can fuse the prologue operations.

        Shapes:
        - Original input: x [M, N]
        - Narrowed input: x.narrow(1, start, length) = [M, length]
        - Kernel: operates on [M, length]
        """
        m, n = 128, 256
        start, length = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, length, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Narrow prologue: select a contiguous sub-region of columns
            x_narrowed = x.narrow(1, start, length)  # [M, length]
            x_scaled = x_narrowed * x_scale  # [M, length]
            x_transformed = torch.relu(x_scaled)  # [M, length]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, length]
        self.assertEqual(list(out_compiled.shape), [m, length])

        # Should use 1 kernel - narrow prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (narrow prologue fused), got {kernel_count}"
        )

    def test_narrow_epilogue(self):
        """Narrow epilogue out.narrow(1, start, length) -> ops - should fuse to 1 kernel.

        Tests that using torch.narrow() on kernel output and applying ops.
        Supports arbitrary narrow offsets (middle slices), not just edge slices.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N]
        - Epilogue: out.narrow(1, start, length) = [M, length] -> ops
        """
        m, n = 128, 256
        start, length = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(m, length, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Narrow epilogue: select a contiguous sub-region and apply ops
            out_narrowed = out.narrow(1, start, length)  # [M, length]
            out_processed = torch.sigmoid(out_narrowed) + epilogue_bias  # [M, length]
            return out_processed, info

        inputs = (x, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, length]
        self.assertEqual(list(out_compiled.shape), [m, length])

        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (epilogue narrow fused), got {kernel_count}"
        )

    def test_combined_slice_prologue_and_epilogue(self):
        """Combined slice prologue x[:, half:] and slice epilogue out[:half, :].

        Tests that both sliced prologue and sliced epilogue fuse to 1 kernel.
        The epilogue is a LEFT slice (starting at index 0), which is supported.

        Shapes:
        - Prologue: x[:, half_n:] = [M, N//2] -> ops (FUSES)
        - Kernel: operates on [M, N//2], outputs [M, N//2]
        - Epilogue: out[:half_m, :] = [M//2, N//2] -> ops (FUSES - left slice)
        """
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, x_scale, epilogue_bias):
            # Slice prologue: right half of columns
            x_sliced = x[:, half_n:]  # [M, N//2]
            x_scaled = x_sliced * x_scale  # [M, N//2]
            x_transformed = torch.relu(x_scaled)  # [M, N//2]

            out, info = elementwise_2d(x_transformed, kernel_scale)

            # Slice epilogue: top half of rows
            out_sliced = out[:half_m, :]  # [M//2, N//2]
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias  # [M//2, N//2]
            return out_processed, info

        inputs = (x, x_scale, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M//2, N//2]
        self.assertEqual(list(out_compiled.shape), [half_m, half_n])

        # Both prologue and epilogue slice fuse to 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (both prologue and epilogue fused), got {kernel_count}"
        )

    def test_transpose_prologue(self):
        """Transpose prologue x.T -> kernel - should fuse to 1 kernel.

        Tests that transposing input before passing to kernel
        can fuse the prologue operations.

        Shapes:
        - Original input: x [N, M]
        - Transposed input: x.T = [M, N]
        - Kernel: operates on [M, N]
        """
        m, n = 128, 256
        # Input is [N, M], transpose to [M, N] for kernel
        x = torch.randn(n, m, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Transpose prologue
            x_transposed = x.T  # [M, N]
            x_scaled = x_transposed * x_scale  # [M, N]
            x_transformed = torch.relu(x_scaled)  # [M, N]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, N]
        self.assertEqual(list(out_compiled.shape), [m, n])

        # Should use 1 kernel - transpose prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (transpose prologue fused), got {kernel_count}"
        )

    def test_transpose_epilogue(self):
        """Transpose epilogue out.T -> ops - should fuse to 1 kernel.

        Tests that transposing kernel output and applying ops.
        Pure torch.compile also fuses this to 1 kernel, so we match that behavior.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N]
        - Epilogue: out.T = [N, M] -> ops
        """
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(n, m, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Transpose epilogue
            out_transposed = out.T  # [N, M]
            out_processed = torch.sigmoid(out_transposed) + epilogue_bias  # [N, M]
            return out_processed, info

        inputs = (x, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [N, M]
        self.assertEqual(list(out_compiled.shape), [n, m])

        # Should use 1 kernel - transpose epilogue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (transpose epilogue fused), got {kernel_count}"
        )

    def test_permute_3d_prologue(self):
        """3D permute prologue x.permute(0, 2, 1) -> view -> kernel - fuses to 1 kernel.

        Tests that permuting a 3D tensor and then viewing to 2D can fuse.

        Shapes:
        - Original input: x [D1, D3, D2]
        - Permuted: x.permute(0, 2, 1) = [D1, D2, D3]
        - Viewed: [D1*D2, D3] for kernel
        """
        d1, d2, d3 = 8, 16, 32
        # Input is [D1, D3, D2], permute to [D1, D2, D3], view to [D1*D2, D3]
        x = torch.randn(d1, d3, d2, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Permute + view prologue
            x_permuted = x.permute(0, 2, 1)  # [D1, D2, D3]
            x_viewed = x_permuted.reshape(d1 * d2, d3)  # [D1*D2, D3]
            x_scaled = x_viewed * x_scale  # [D1*D2, D3]
            x_transformed = torch.relu(x_scaled)  # [D1*D2, D3]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [D1*D2, D3]
        self.assertEqual(list(out_compiled.shape), [d1 * d2, d3])

        # Should use 1 kernel - permute+view prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (permute+view prologue fused), got {kernel_count}"
        )

    def test_unsqueeze_squeeze_prologue(self):
        """Unsqueeze/squeeze prologue x.unsqueeze(0).squeeze(0) -> kernel.

        Tests that dimension manipulation that preserves shape can fuse.

        Shapes:
        - Original input: x [M, N]
        - Unsqueezed: x.unsqueeze(0) = [1, M, N]
        - Squeezed: .squeeze(0) = [M, N]
        - Kernel: operates on [M, N]
        """
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Unsqueeze/squeeze prologue (no-op reshape, but tests codegen)
            x_unsqueezed = x.unsqueeze(0)  # [1, M, N]
            x_squeezed = x_unsqueezed.squeeze(0)  # [M, N]
            x_scaled = x_squeezed * x_scale  # [M, N]
            x_transformed = torch.relu(x_scaled)  # [M, N]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, N]
        self.assertEqual(list(out_compiled.shape), [m, n])

        # Should use 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (unsqueeze/squeeze prologue fused), got {kernel_count}"
        )


    def test_right_slice_epilogue_column_slice(self):
        """Right column slice epilogue out[:, half:] -> ops - fuses to 1 kernel.

        Tests that slicing the kernel output from a non-zero offset (right slice)
        now fuses correctly with proper index transformation.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N]
        - Epilogue: out[:, half:] = [M, N//2] (starts at half) -> ops
        """
        m, n = 128, 256
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Epilogue: slice RIGHT half of columns (non-zero offset)
            out_sliced = out[:, half_n:]  # [M, N//2] starting at half_n
            out_processed = torch.relu(out_sliced) + epilogue_bias  # [M, N//2]
            return out_processed, info

        inputs = (x, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, N//2]
        self.assertEqual(list(out_compiled.shape), [m, half_n])

        # Right slice epilogue now fuses - uses 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (right slice epilogue fused), got {kernel_count}"
        )

    def test_right_slice_epilogue_row_slice(self):
        """Right row slice epilogue out[half:, :] -> ops - fuses to 1 kernel.

        Tests that slicing the kernel output from a non-zero row offset
        now fuses correctly with proper index transformation.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N]
        - Epilogue: out[half:, :] = [M//2, N] (starts at half_m) -> ops
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)

            # Epilogue: slice BOTTOM half of rows (non-zero offset)
            out_sliced = out[half_m:, :]  # [M//2, N] starting at half_m
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias  # [M//2, N]
            return out_processed, info

        inputs = (x, epilogue_bias)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M//2, N]
        self.assertEqual(list(out_compiled.shape), [half_m, n])

        # Right slice epilogue now fuses - uses 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (right slice epilogue fused), got {kernel_count}"
        )

    # =========================================================================
    # Single-output slice prologue tests (left and right)
    # =========================================================================

    def test_left_slice_prologue_column_slice(self):
        """Left column slice prologue x[:, :half] -> kernel - should fuse to 1 kernel.

        Tests that slicing the input tensor along columns (left slice) before
        passing to kernel can fuse the prologue operations.

        Shapes:
        - Original input: x [M, N]
        - Sliced input: x[:, :half] = [M, N//2]
        - Kernel: operates on [M, N//2]
        """
        m, n = 128, 256
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Slice prologue: take LEFT half of columns
            x_sliced = x[:, :half_n]  # [M, N//2]
            x_scaled = x_sliced * x_scale  # [M, N//2]
            x_transformed = torch.relu(x_scaled)  # [M, N//2]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M, N//2]
        self.assertEqual(list(out_compiled.shape), [m, half_n])

        # Should use 1 kernel - left slice prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (left slice prologue fused), got {kernel_count}"
        )

    def test_left_slice_prologue_row_slice(self):
        """Left row slice prologue x[:half, :] -> kernel - should fuse to 1 kernel.

        Tests that slicing the input tensor along rows (left slice) before
        passing to kernel can fuse the prologue operations.

        Shapes:
        - Original input: x [M, N]
        - Sliced input: x[:half, :] = [M//2, N]
        - Kernel: operates on [M//2, N]
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Slice prologue: take TOP half of rows
            x_sliced = x[:half_m, :]  # [M//2, N]
            x_scaled = x_sliced * x_scale  # [M//2, N]
            x_transformed = torch.sigmoid(x_scaled)  # [M//2, N]

            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify output shape is [M//2, N]
        self.assertEqual(list(out_compiled.shape), [half_m, n])

        # Should use 1 kernel - left slice prologue fuses
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (left slice prologue fused), got {kernel_count}"
        )

    # =========================================================================
    # Single-output slice epilogue tests (left and right)
    # =========================================================================

    def test_single_output_left_slice_epilogue_column(self):
        """Single-output kernel with left column slice epilogue out[:, :half] -> ops.

        Tests that a SINGLE-OUTPUT kernel can fuse left slice epilogue.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N] (single tensor)
        - Epilogue: out[:, :half] = [M, N//2] -> ops
        """
        m, n = 128, 256
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out = elementwise_2d_single_output(x, kernel_scale)

            # Epilogue: slice LEFT half of columns and apply ops
            out_sliced = out[:, :half_n]  # [M, N//2]
            out_processed = torch.relu(out_sliced) + epilogue_bias  # [M, N//2]
            return out_processed

        inputs = (x, epilogue_bias)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [M, N//2]
        self.assertEqual(list(result.shape), [m, half_n])

        # Single-output epilogue fusion - should use 1 kernel when implemented
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (single-output epilogue fused), got {kernel_count}"
        )

    def test_single_output_left_slice_epilogue_row(self):
        """Single-output kernel with left row slice epilogue out[:half, :] -> ops.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N] (single tensor)
        - Epilogue: out[:half, :] = [M//2, N] -> ops
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out = elementwise_2d_single_output(x, kernel_scale)

            # Epilogue: slice TOP half of rows and apply ops
            out_sliced = out[:half_m, :]  # [M//2, N]
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias  # [M//2, N]
            return out_processed

        inputs = (x, epilogue_bias)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [M//2, N]
        self.assertEqual(list(result.shape), [half_m, n])

        # Single-output epilogue fusion - should use 1 kernel when implemented
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (single-output epilogue fused), got {kernel_count}"
        )

    def test_single_output_right_slice_epilogue_column(self):
        """Single-output kernel with right column slice epilogue out[:, half:] -> ops.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N] (single tensor)
        - Epilogue: out[:, half:] = [M, N//2] (starts at half_n) -> ops
        """
        m, n = 128, 256
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out = elementwise_2d_single_output(x, kernel_scale)

            # Epilogue: slice RIGHT half of columns (non-zero offset)
            out_sliced = out[:, half_n:]  # [M, N//2] starting at half_n
            out_processed = torch.relu(out_sliced) + epilogue_bias  # [M, N//2]
            return out_processed

        inputs = (x, epilogue_bias)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [M, N//2]
        self.assertEqual(list(result.shape), [m, half_n])

        # Single-output epilogue fusion - should use 1 kernel when implemented
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (single-output epilogue fused), got {kernel_count}"
        )

    def test_single_output_right_slice_epilogue_row(self):
        """Single-output kernel with right row slice epilogue out[half:, :] -> ops.

        Shapes:
        - Kernel: operates on [M, N], outputs [M, N] (single tensor)
        - Epilogue: out[half:, :] = [M//2, N] (starts at half_m) -> ops
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out = elementwise_2d_single_output(x, kernel_scale)

            # Epilogue: slice BOTTOM half of rows (non-zero offset)
            out_sliced = out[half_m:, :]  # [M//2, N] starting at half_m
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias  # [M//2, N]
            return out_processed

        inputs = (x, epilogue_bias)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [M//2, N]
        self.assertEqual(list(result.shape), [half_m, n])

        # Single-output epilogue fusion - should use 1 kernel when implemented
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (single-output epilogue fused), got {kernel_count}"
        )

    # =========================================================================
    # Single-output reshape tests (prologue and epilogue)
    # =========================================================================

    def test_single_output_reshape_epilogue(self):
        """Single-output kernel with reshaped epilogue [M,N] -> [M*N] -> ops.

        Tests that a SINGLE-OUTPUT kernel can handle reshaped epilogue.

        Shapes:
        - Kernel input: x [M, N]
        - Kernel output: out [M, N] (single tensor)
        - Epilogue: out.view(M*N) -> sigmoid + add
        """
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        flat_bias = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

        def f(x, flat_bias):
            out = elementwise_2d_single_output(x, kernel_scale)

            # Reshaped epilogue [M*N]
            flattened = out.view(m * n)
            flat_processed = torch.sigmoid(flattened) + flat_bias
            return flat_processed

        inputs = (x, flat_bias)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [M*N]
        self.assertEqual(list(result.shape), [m * n])

        # Single-output epilogue fusion - should use 1 kernel when implemented
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (single-output epilogue fused), got {kernel_count}"
        )

    def test_single_output_reshape_prologue(self):
        """Single-output kernel with reshaped prologue [M*N] -> [M, N] -> kernel.

        Tests that a SINGLE-OUTPUT kernel can handle reshaped prologue.

        Shapes:
        - Prologue input: x_flat [M*N] -> ops -> view to [M, N]
        - Kernel: operates on [M, N]
        - Kernel output: out [M, N] (single tensor)
        """
        m, n = 128, 256
        # Flat prologue input [M*N]
        x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        x_flat_scale = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x_flat, x_flat_scale):
            # Flat prologue [M*N] -> view -> [M, N]
            x_scaled = x_flat * x_flat_scale  # [M*N]
            x_transformed = torch.relu(x_scaled)  # [M*N]
            x = x_transformed.view(m, n)  # [M, N]

            out = elementwise_2d_single_output(x, kernel_scale)
            return out

        inputs = (x_flat, x_flat_scale)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [M, N]
        self.assertEqual(list(result.shape), [m, n])

        # Should use 1 kernel - prologue fusion works for single-output
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (reshape prologue fused), got {kernel_count}"
        )

    def test_single_output_partial_flatten_prologue(self):
        """Single-output kernel with partial flatten prologue [d1, d2*d3] -> [d1*d2, d3].

        Tests that a SINGLE-OUTPUT kernel can handle partial flatten prologue.

        Shapes:
        - Prologue input: x_partial [d1, d2*d3] -> ops -> view to [d1*d2, d3]
        - Kernel: operates on [d1*d2, d3]
        """
        d1, d2, d3 = 8, 16, 32
        # Partially flat prologue input [d1, d2*d3]
        x_partial = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x_partial, x_scale):
            # Prologue on PARTIALLY FLAT input [d1, d2*d3]
            x_scaled = x_partial * x_scale  # [d1, d2*d3]
            x_transformed = torch.relu(x_scaled)  # [d1, d2*d3]
            # Reshape to kernel's expected 2D shape [d1*d2, d3]
            x = x_transformed.view(d1 * d2, d3)

            out = elementwise_2d_single_output(x, kernel_scale)
            return out

        inputs = (x_partial, x_scale)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [d1*d2, d3]
        self.assertEqual(list(result.shape), [d1 * d2, d3])

        # Partial flatten prologue fuses to 1 kernel
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (partial flatten prologue fused), got {kernel_count}"
        )

    def test_single_output_partial_flatten_epilogue(self):
        """Single-output kernel with partial flatten epilogue [d1*d2, d3] -> [d1, d2*d3].

        Tests that a SINGLE-OUTPUT kernel can handle partial flatten epilogue.

        Shapes:
        - Kernel: operates on [d1*d2, d3], outputs [d1*d2, d3]
        - Epilogue: out [d1*d2, d3] -> view to [d1, d2*d3] -> ops
        """
        d1, d2, d3 = 8, 16, 32
        # Kernel input [d1*d2, d3]
        x = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        # Partially flat epilogue bias [d1, d2*d3]
        out_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

        def f(x, out_bias):
            out = elementwise_2d_single_output(x, kernel_scale)

            # Epilogue: reshape to PARTIALLY FLAT [d1, d2*d3] and apply ops
            out_reshaped = out.view(d1, d2 * d3)  # [d1, d2*d3]
            out_processed = torch.sigmoid(out_reshaped) + out_bias  # [d1, d2*d3]
            return out_processed

        inputs = (x, out_bias)

        # Eager
        out_eager = f(*inputs)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(result, out_eager, rtol=1e-2, atol=1e-2)

        # Verify output shape is [d1, d2*d3]
        self.assertEqual(list(result.shape), [d1, d2 * d3])

        # Single-output epilogue fusion - should use 1 kernel when implemented
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1,
            f"Expected 1 kernel (single-output epilogue fused), got {kernel_count}"
        )


    # =========================================================================
    # 3D Transpose Tests (Pure permute without reshape)
    # =========================================================================

    def test_3d_transpose_prologue_swap_last_two_dims(self):
        """3D transpose prologue x.permute(0, 2, 1) -> kernel - pure 3D transpose.

        Tests that a pure 3D transpose (swapping last two dims) on input can fuse.
        This is different from test_permute_3d_prologue which also includes a reshape.

        Shapes:
        - Original input: x [D1, D3, D2] (all different sizes)
        - Permuted: x.permute(0, 2, 1) = [D1, D2, D3]
        - Kernel: 3D elementwise on [D1, D2, D3]
        """
        d1, d2, d3 = 8, 16, 32  # All different sizes
        # Input is [D1, D3, D2], permute to [D1, D2, D3] for kernel
        x = torch.randn(d1, d3, d2, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Pure 3D transpose prologue (no reshape)
            x_permuted = x.permute(0, 2, 1)  # [D1, D2, D3]
            x_scaled = x_permuted * x_scale  # [D1, D2, D3]
            x_transformed = torch.relu(x_scaled)  # [D1, D2, D3]

            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)

        # Eager
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_3d_transpose_prologue_swap_first_two_dims(self):
        """3D transpose prologue x.permute(1, 0, 2) -> kernel.

        Tests swapping first two dimensions in a 3D tensor.

        Shapes:
        - Original input: x [D2, D1, D3]
        - Permuted: x.permute(1, 0, 2) = [D1, D2, D3]
        - Kernel: 3D elementwise on [D1, D2, D3]
        """
        d1, d2, d3 = 8, 16, 32
        # Input is [D2, D1, D3], permute to [D1, D2, D3] for kernel
        x = torch.randn(d2, d1, d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_permuted = x.permute(1, 0, 2)  # [D1, D2, D3]
            x_scaled = x_permuted * x_scale
            x_transformed = torch.relu(x_scaled)

            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_3d_transpose_prologue_full_reverse(self):
        """3D transpose prologue x.permute(2, 1, 0) -> kernel - full dimension reversal.

        Tests reversing all dimensions in a 3D tensor.

        Shapes:
        - Original input: x [D3, D2, D1]
        - Permuted: x.permute(2, 1, 0) = [D1, D2, D3]
        - Kernel: 3D elementwise on [D1, D2, D3]
        """
        d1, d2, d3 = 8, 16, 32
        # Input is [D3, D2, D1], permute to [D1, D2, D3] for kernel
        x = torch.randn(d3, d2, d1, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_permuted = x.permute(2, 1, 0)  # [D1, D2, D3]
            x_scaled = x_permuted * x_scale
            x_transformed = torch.sigmoid(x_scaled)

            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_3d_transpose_prologue_cyclic_permute(self):
        """3D transpose prologue x.permute(1, 2, 0) -> kernel - cyclic permutation.

        Tests cyclic permutation (0->2, 1->0, 2->1).

        Shapes:
        - Original input: x [D3, D1, D2]
        - Permuted: x.permute(1, 2, 0) = [D1, D2, D3]
        - Kernel: 3D elementwise on [D1, D2, D3]
        """
        d1, d2, d3 = 8, 16, 32
        # Input is [D3, D1, D2], permute(1,2,0) gives [D1, D2, D3]
        x = torch.randn(d3, d1, d2, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_permuted = x.permute(1, 2, 0)  # [D1, D2, D3]
            x_scaled = x_permuted * x_scale
            x_transformed = torch.tanh(x_scaled)

            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_3d_transpose_epilogue_swap_last_two_dims(self):
        """3D transpose epilogue out.permute(0, 2, 1) -> ops.

        Tests that a pure 3D transpose on output can fuse.

        Shapes:
        - Kernel input/output: [D1, D2, D3]
        - Epilogue permuted: out.permute(0, 2, 1) = [D1, D3, D2] -> ops
        """
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d1, d3, d2, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_3d(x, kernel_scale)
            out_permuted = out.permute(0, 2, 1)  # [D1, D3, D2]
            out_processed = torch.relu(out_permuted) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d3, d2])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_3d_transpose_epilogue_full_reverse(self):
        """3D transpose epilogue out.permute(2, 1, 0) -> ops - full reversal.

        Shapes:
        - Kernel input/output: [D1, D2, D3]
        - Epilogue permuted: out.permute(2, 1, 0) = [D3, D2, D1] -> ops
        """
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d3, d2, d1, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_3d(x, kernel_scale)
            out_permuted = out.permute(2, 1, 0)  # [D3, D2, D1]
            out_processed = torch.sigmoid(out_permuted) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d3, d2, d1])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1)

    def test_3d_transpose_prologue_same_size_dims(self):
        """3D transpose where two dimensions have the SAME size.

        This is a challenging edge case: when dims have the same size,
        the size-matching algorithm in _find_permutation may match incorrectly.

        Shapes:
        - Original input: x [A, A, B] where first two dims are same size
        - Permuted: x.permute(1, 0, 2) = [A, A, B] (same shape, but data reordered!)
        - Kernel: 3D elementwise on [A, A, B]
        """
        a, b = 16, 32  # Two dims with same size
        x = torch.randn(a, a, b, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(a, a, b, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            # Swap first two dims (both size A)
            x_permuted = x.permute(1, 0, 2)  # Still [A, A, B] but data reordered
            x_scaled = x_permuted * x_scale
            x_transformed = torch.relu(x_scaled)

            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        # This may fail if size-matching incorrectly treats as identity permutation
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_3d_transpose_epilogue_same_size_dims(self):
        """3D transpose epilogue where two dimensions have the SAME size.

        Shapes:
        - Kernel input/output: [A, B, A] where first and last dims are same size
        - Epilogue permuted: out.permute(2, 1, 0) = [A, B, A] (same shape, data reordered)
        """
        a, b = 16, 32
        x = torch.randn(a, b, a, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(a, b, a, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_3d(x, kernel_scale)
            out_permuted = out.permute(2, 1, 0)  # [A, B, A] but reordered
            out_processed = torch.sigmoid(out_permuted) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    # =========================================================================
    # 4D Transpose Tests
    # =========================================================================

    def test_4d_transpose_prologue_swap_middle_dims(self):
        """4D transpose prologue x.permute(0, 2, 1, 3) -> kernel.

        Common pattern in attention: swap sequence and head dimensions.

        Shapes:
        - Original: x [B, S, H, D]
        - Permuted: x.permute(0, 2, 1, 3) = [B, H, S, D]
        - Kernel: 4D elementwise on [B, H, S, D]
        """
        b, h, s, d = 2, 8, 64, 32
        # Input is [B, S, H, D], permute to [B, H, S, D] for kernel
        x = torch.randn(b, s, h, d, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(b, h, s, d, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_permuted = x.permute(0, 2, 1, 3)  # [B, H, S, D]
            x_scaled = x_permuted * x_scale
            x_transformed = torch.relu(x_scaled)

            out, info = elementwise_4d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [b, h, s, d])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_4d_transpose_epilogue_swap_middle_dims(self):
        """4D transpose epilogue out.permute(0, 2, 1, 3) -> ops.

        Shapes:
        - Kernel input/output: [B, H, S, D]
        - Epilogue permuted: out.permute(0, 2, 1, 3) = [B, S, H, D] -> ops
        """
        b, h, s, d = 2, 8, 64, 32
        x = torch.randn(b, h, s, d, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(b, s, h, d, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_4d(x, kernel_scale)
            out_permuted = out.permute(0, 2, 1, 3)  # [B, S, H, D]
            out_processed = torch.sigmoid(out_permuted) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [b, s, h, d])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    # =========================================================================
    # Complex View Operations - Multi-step and Combined
    # =========================================================================

    def test_view_then_slice_prologue(self):
        """Prologue: view then slice - x.view(M//2, 2*N)[:, :N] -> kernel.

        Tests chaining view and slice operations with Helion kernel.

        Shapes:
        - Original: x [M, N]
        - Viewed: x.view(M//2, 2*N) = [M//2, 2*N]
        - Sliced: [:, :N] = [M//2, N]
        - Kernel: elementwise_2d on [M//2, N]
        """
        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m // 2, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_viewed = x.view(m // 2, 2 * n)  # [M//2, 2*N]
            x_sliced = x_viewed[:, :n]  # [M//2, N]
            x_scaled = x_sliced * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m // 2, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_slice_then_view_prologue(self):
        """Prologue: slice then view - x[:, :half].reshape(M//2, N) -> kernel.

        Tests chaining slice and view operations with Helion kernel.

        Shapes:
        - Original: x [M, N]
        - Sliced: x[:, :half] = [M, N//2]
        - Viewed: .reshape(M//2, N) = [M//2, N]
        - Kernel: elementwise_2d on [M//2, N]
        """
        m, n = 64, 128
        half_n = n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m // 2, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_sliced = x[:, :half_n]  # [M, N//2]
            x_viewed = x_sliced.reshape(m // 2, n)  # [M//2, N]
            x_scaled = x_viewed * x_scale
            x_transformed = torch.sigmoid(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m // 2, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_transpose_then_slice_prologue(self):
        """Prologue: transpose then slice - x.T[:, :half] -> kernel.

        Tests chaining transpose and slice operations with Helion kernel.

        Shapes:
        - Original: x [N, M]
        - Transposed: x.T = [M, N]
        - Sliced: [:, :half] = [M, N//2]
        - Kernel: elementwise_2d on [M, N//2]
        """
        m, n = 64, 128
        half_n = n // 2
        x = torch.randn(n, m, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_transposed = x.T  # [M, N]
            x_sliced = x_transposed[:, :half_n]  # [M, N//2]
            x_scaled = x_sliced * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m, half_n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_slice_then_transpose_prologue(self):
        """Prologue: slice then transpose - x[:half, :].T -> kernel.

        Tests chaining slice and transpose operations with Helion kernel.

        Shapes:
        - Original: x [M, N]
        - Sliced: x[:half, :] = [M//2, N]
        - Transposed: .T = [N, M//2]
        - Kernel: elementwise_2d on [N, M//2]
        """
        m, n = 64, 128
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(n, half_m, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_sliced = x[:half_m, :]  # [M//2, N]
            x_transposed = x_sliced.T  # [N, M//2]
            x_scaled = x_transposed * x_scale
            x_transformed = torch.sigmoid(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [n, half_m])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_view_then_slice_epilogue(self):
        """Epilogue: view then slice - out.view(M//2, 2*N)[:, :N] -> ops.

        Tests chaining view and slice operations in epilogue with Helion kernel.

        Shapes:
        - Kernel input/output: [M, N]
        - Viewed: out.view(M//2, 2*N) = [M//2, 2*N]
        - Sliced: [:, :N] = [M//2, N] -> ops
        """
        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(m // 2, n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)  # [M, N]
            out_viewed = out.view(m // 2, 2 * n)  # [M//2, 2*N]
            out_sliced = out_viewed[:, :n]  # [M//2, N]
            out_processed = torch.relu(out_sliced) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m // 2, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_transpose_then_slice_epilogue(self):
        """Epilogue: transpose then slice - out.T[:, :half] -> ops.

        Tests chaining transpose and slice operations in epilogue with Helion kernel.

        Shapes:
        - Kernel input/output: [M, N]
        - Transposed: out.T = [N, M]
        - Sliced: [:, :half] = [N, M//2] -> ops
        """
        m, n = 64, 128
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(n, half_m, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)  # [M, N]
            out_transposed = out.T  # [N, M]
            out_sliced = out_transposed[:, :half_m]  # [N, M//2]
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [n, half_m])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_multiple_views_prologue(self):
        """Prologue: multiple consecutive views - x.view().view().view() -> kernel.

        Tests that multiple consecutive view operations can fuse with Helion kernel.

        Shapes:
        - Original: x [D1, D2, D3]
        - View 1: x.view(D1*D2, D3) = [D1*D2, D3]
        - View 2: .view(D1, D2*D3) = [D1, D2*D3]
        - View 3: .view(D1*D2, D3) = [D1*D2, D3] (back to 2D for kernel)
        - Kernel: elementwise_2d on [D1*D2, D3]
        """
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x1 = x.view(d1 * d2, d3)  # [D1*D2, D3]
            x2 = x1.view(d1, d2 * d3)  # [D1, D2*D3]
            x3 = x2.view(d1 * d2, d3)  # [D1*D2, D3] - back to 2D
            x_scaled = x3 * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1 * d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_view_unflatten_prologue(self):
        """Prologue: unflatten operation - x.unflatten(1, (d2, d3)) -> kernel.

        Tests unflatten (inverse of flatten) in prologue with Helion kernel.

        Shapes:
        - Original: x [D1, D2*D3]
        - Unflattened: x.unflatten(1, (D2, D3)) = [D1, D2, D3]
        - Kernel: elementwise_3d on [D1, D2, D3]
        """
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_unflat = x.unflatten(1, (d2, d3))  # [D1, D2, D3]
            x_scaled = x_unflat * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_flatten_epilogue(self):
        """Epilogue: flatten operation - out.flatten(1, 2) -> ops.

        Tests flatten (multi-dim to single-dim) in epilogue with Helion kernel.

        Shapes:
        - Kernel input/output: [D1, D2, D3]
        - Flattened: out.flatten(1, 2) = [D1, D2*D3] -> ops
        """
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_3d(x, kernel_scale)  # [D1, D2, D3]
            out_flat = out.flatten(1, 2)  # [D1, D2*D3]
            out_processed = torch.sigmoid(out_flat) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d2 * d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_expand_prologue(self):
        """Prologue: expand operation - x.expand(B, M, N) -> kernel.

        Tests expand (broadcasting) in prologue with Helion kernel.

        Shapes:
        - Original: x [1, M, N]
        - Expanded: x.expand(B, M, N) = [B, M, N]
        - Kernel: elementwise_3d on [B, M, N]
        """
        b, m, n = 4, 64, 128
        x = torch.randn(1, m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(b, m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_expanded = x.expand(b, m, n)  # [B, M, N]
            x_scaled = x_expanded * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_3d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [b, m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_contiguous_after_transpose_prologue(self):
        """Prologue: transpose + contiguous - x.T.contiguous() -> kernel.

        Tests that making a transposed tensor contiguous can fuse with Helion kernel.

        Shapes:
        - Original: x [N, M]
        - Transposed: x.T = [M, N]
        - Contiguous: .contiguous() = [M, N] (new layout)
        - Kernel: elementwise_2d on [M, N]
        """
        m, n = 64, 128
        x = torch.randn(n, m, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_t = x.T  # [M, N] - non-contiguous
            x_contig = x_t.contiguous()  # [M, N] - contiguous
            x_scaled = x_contig * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_3d_view_to_2d_then_transpose_prologue(self):
        """Prologue: 3D->2D view then transpose - x.view(D1, D2*D3).T -> kernel.

        Tests reshaping 3D to 2D then transposing with Helion kernel.

        Shapes:
        - Original: x [D1, D2, D3]
        - Viewed: x.view(D1, D2*D3) = [D1, D2*D3]
        - Transposed: .T = [D2*D3, D1]
        - Kernel: elementwise_2d on [D2*D3, D1]
        """
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d2 * d3, d1, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_2d = x.view(d1, d2 * d3)  # [D1, D2*D3]
            x_t = x_2d.T  # [D2*D3, D1]
            x_scaled = x_t * x_scale
            x_transformed = torch.sigmoid(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d2 * d3, d1])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_transpose_then_view_to_3d_epilogue(self):
        """Epilogue: transpose then 2D->3D view - out.T.reshape(D1, D2, D3) -> ops.

        Tests transposing then reshaping to 3D in epilogue with Helion kernel.

        Shapes:
        - Kernel input/output: [M, N] where M*N = D1*D2*D3
        - Transposed: out.T = [N, M]
        - Viewed: .reshape(D1, D2, D3) -> ops
        """
        d1, d2, d3 = 8, 16, 32
        m, n = d1 * d2, d3  # M=128, N=32 -> M*N = D1*D2*D3
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d3, d1, d2, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)  # [M, N] = [128, 32]
            out_t = out.T  # [N, M] = [32, 128]
            out_3d = out_t.reshape(d3, d1, d2)  # [32, 8, 16]
            out_processed = torch.relu(out_3d) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d3, d1, d2])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    # =========================================================================
    # Strided View/Slice Tests
    # =========================================================================

    def test_strided_slice_prologue(self):
        """Strided slice prologue x[::2, :] -> kernel.

        Tests slicing with stride > 1 in prologue with Helion kernel.

        Shapes:
        - Original: x [M, N]
        - Strided: x[::2, :] = [M//2, N]
        - Kernel: elementwise_2d on [M//2, N]
        """
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_strided = x[::2, :]  # [M//2, N] - every other row
            x_scaled = x_strided * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            return out, info

        inputs = (x, x_scale)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [half_m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_strided_slice_epilogue(self):
        """Strided slice epilogue out[::2, ::2] -> ops.

        Tests slicing with stride > 1 in epilogue with Helion kernel.

        Shapes:
        - Kernel input/output: [M, N]
        - Strided: out[::2, ::2] = [M//2, N//2] -> ops
        """
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)  # [M, N]
            out_strided = out[::2, ::2]  # [M//2, N//2] - every other row and column
            out_processed = torch.sigmoid(out_strided) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [half_m, half_n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_diagonal_slice_epilogue(self):
        """Diagonal slice epilogue out.diagonal() -> ops.

        Tests extracting diagonal from output in epilogue with Helion kernel.

        Shapes:
        - Kernel input/output: [N, N] (square)
        - Diagonal: out.diagonal() = [N] -> ops
        """
        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)  # [N, N]
            out_diag = out.diagonal()  # [N]
            out_processed = torch.relu(out_diag) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_expand_epilogue_broadcast(self):
        """Broadcasted expand in epilogue: out.view(1, M, N).expand(B, M, N)."""
        b, m, n = 4, 32, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            out = elementwise_2d_single_output(x, kernel_scale)  # [M, N]
            out_view = out.unsqueeze(0).expand(b, m, n)  # [B, M, N]
            return torch.relu(out_view)

        inputs = (x,)
        out_eager = f(*inputs)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        out_compiled, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(list(out_compiled.shape), [b, m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 2, f"Expected 2 kernels, got {kernel_count}")

    def test_expand_permute_epilogue(self):
        """Expand then permute in epilogue: out[None].expand -> permute."""
        b, m, n = 3, 32, 16
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            out = elementwise_2d_single_output(x, kernel_scale)  # [M, N]
            out_view = out.unsqueeze(0).expand(b, m, n)  # [B, M, N]
            out_perm = out_view.permute(1, 0, 2)  # [M, B, N]
            return torch.tanh(out_perm)

        inputs = (x,)
        out_eager = f(*inputs)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        out_compiled, source_codes = run_and_get_code(compiled_f, *inputs)

        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(list(out_compiled.shape), [m, b, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 2, f"Expected 2 kernels, got {kernel_count}")

    def test_expand_epilogue_multi_dim_broadcast(self):
        """Broadcasted expand in epilogue for 3D -> 4D."""
        b, d1, d2, d3 = 3, 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            out, info = elementwise_3d(x, kernel_scale)  # [D1, D2, D3]
            out_view = out.unsqueeze(1).expand(d1, b, d2, d3)  # [D1, B, D2, D3]
            out_processed = torch.relu(out_view)
            return out_processed, info

        inputs = (x,)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, b, d2, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 2, f"Expected 2 kernels, got {kernel_count}")

    def test_as_strided_prologue_overlap(self):
        """Overlapping as_strided in prologue should not fuse."""
        m, n = 32, 64
        x = torch.randn(m + n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            x_view = x.as_strided((m, n), (1, 1))
            x_processed = torch.relu(x_view)
            out, info = elementwise_2d(x_processed, kernel_scale)
            return out, info

        inputs = (x,)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 2, f"Expected 2 kernels, got {kernel_count}")

    def test_as_strided_epilogue_overlap(self):
        """Overlapping as_strided in epilogue should not fuse."""
        m, n = 32, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            out, info = elementwise_2d(x, kernel_scale)
            out_view = out.as_strided((m, n), (1, 1))
            out_processed = torch.sigmoid(out_view)
            return out_processed, info

        inputs = (x,)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 2, f"Expected 2 kernels, got {kernel_count}")

    def test_select_epilogue_rank_drop(self):
        """Rank drop epilogue: out[:, 0, :] -> ops."""
        d1, d2, d3 = 16, 8, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            out, info = elementwise_3d(x, kernel_scale)  # [D1, D2, D3]
            out_selected = out[:, 0, :]  # [D1, D3]
            return torch.relu(out_selected), info

        inputs = (x,)
        out_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [d1, d3])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

if __name__ == "__main__":
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    unittest.main()
