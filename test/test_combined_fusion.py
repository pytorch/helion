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

        Prologue chains (3+ ops):
        - x: sigmoid(relu(x * x_scale + x_bias)) before RMS norm

        Epilogue chains (each tensor output has its own 3+ ops chain):
        - out: tanh((relu(out) + out_bias) * out_scale)
        - residual: sigmoid((residual + res_bias) * res_scale)
        - extra: relu(extra + extra_bias) (same shape [M, N])

        Multiple inputs (10 total):
        - x: input tensor [M, N]
        - weight: RMS norm weight [N]
        - x_scale: prologue scale for x [N]
        - x_bias: prologue bias for x [N]
        - out_bias: epilogue bias for normalized output [N]
        - out_scale: epilogue scale for normalized output [N]
        - res_bias: epilogue bias for residual [N]
        - res_scale: epilogue scale for residual [N]
        - extra_bias: epilogue bias for extra output [N]

        Multiple outputs (5 total, interleaved tensors and scalars):
        - Tensor: processed normalized output [M, N]
        - Tensor: processed residual [M, N]
        - Non-tensor: scalar value (42)
        - Tensor: processed extra [M, N]
        - Non-tensor: scalar value (99)
        """
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Prologue closure tensors for x
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Epilogue closure tensors for main output
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Epilogue closure tensors for residual output
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Epilogue closure tensors for extra output (same shape as out)
        extra_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, x_scale, x_bias, out_bias, out_scale, res_bias, res_scale, extra_bias):
            # Prologue chain for x (4 ops): scale -> add bias -> relu -> sigmoid
            x_scaled = x * x_scale
            x_biased = x_scaled + x_bias
            x_relu = torch.relu(x_biased)
            x_transformed = torch.sigmoid(x_relu)

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

            # Epilogue chain for extra (2 ops): add bias -> relu
            extra_biased = extra + extra_bias
            extra_processed = torch.relu(extra_biased)

            return out_processed, res_processed, info1, extra_processed, info2

        inputs = (
            x, weight, x_scale, x_bias, out_bias, out_scale, res_bias, res_scale, extra_bias
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

        # Verify output shapes - all tensor outputs have same shape [M, N]
        self.assertEqual(list(out_compiled.shape), [m, n])
        self.assertEqual(list(res_compiled.shape), [m, n])
        self.assertEqual(list(extra_compiled.shape), [m, n])

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

    def test_rms_norm_different_shape_outputs_graceful_fallback(self):
        """RMS norm with DIFFERENT shaped prologue and epilogue - tests mixed fusion.

        This test verifies behavior when prologues/epilogues have different shapes:
        - Flat prologues (via ReinterpretView/reshape) ARE fused with index transformation
        - Flat epilogues (via reshape of kernel output) are NOT fused (different indexing)

        The prologue fusion works because:
        1. ReinterpretView provides stride information for the view
        2. The 2D kernel indices are transformed to 1D flat indices via: flat = row * N + col
        3. The load result has the correct 2D shape for kernel operations

        The epilogue cannot fuse because:
        1. Epilogue ops operate on flattened outputs
        2. The kernel stores with 2D indices, epilogue needs 1D indices
        3. These indexing patterns are incompatible for store operations

        Shapes used:
        - Prologue input: x_flat [M*N] -> reshaped to [M, N] for kernel (FUSES)
        - Kernel: operates on [M, N]
        - Epilogue output: residual [M, N] -> reshaped to [M*N] for epilogue (DOES NOT FUSE)
        """
        m, n = 128, 256
        # Different-shaped prologue input (flattened)
        x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        x_flat_scale = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        flat_bias = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

        def f(x_flat, x_flat_scale, weight, out_bias, flat_bias):
            # Prologue on DIFFERENT shaped input [M*N]
            x_scaled = x_flat * x_flat_scale  # [M*N]
            x_transformed = torch.relu(x_scaled)  # [M*N]
            # Reshape to kernel's expected shape [M, N]
            x = x_transformed.view(m, n)

            out, residual, info = rms_norm_multi_output(x, weight)

            # Create flattened view for epilogue (same numel, different shape)
            flattened = residual.view(m * n)
            # Epilogues on DIFFERENT shaped outputs
            out_processed = torch.relu(out) + out_bias  # [M, N]
            flat_processed = torch.sigmoid(flattened) + flat_bias  # [M*N]
            return out_processed, flat_processed, info

        inputs = (x_flat, x_flat_scale, weight, out_bias, flat_bias)

        # Eager
        out_eager, flat_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compiled - should work without crashing (graceful fallback)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, flat_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(flat_compiled, flat_eager, rtol=1e-2, atol=1e-2)
        self.assertEqual(info_compiled, 42)

        # Verify different shapes
        self.assertEqual(list(out_compiled.shape), [m, n])
        self.assertEqual(list(flat_compiled.shape), [m * n])

        # Should use exactly 2 kernels:
        # - One for the Helion kernel with:
        #   - Flat prologue [M*N] fused (mul, relu with index transformation)
        #   - Same-shaped epilogue [M,N] fused (relu, add on out)
        # - One for the different-shaped epilogue (sigmoid, add on [M*N])
        # Note: Differently-shaped prologues ARE now supported via index
        # transformation when the view is a simple reshape (no striding changes).
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 2,
            f"Expected exactly 2 kernels (flat prologue now fuses), got {kernel_count}"
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

    def test_differently_shaped_prologue_same_shaped_epilogue(self):
        """Differently-shaped prologue with same-shaped epilogue - should fuse into 1 kernel.

        This test verifies that:
        - Flat prologue [M*N] is fused via View.dynamic_reshape_indexer index transformation
        - Same-shaped epilogue [M, N] is fused normally
        - Both fuse into a single Triton kernel

        The prologue fusion works because:
        1. ReinterpretView provides the kernel's view size [M, N]
        2. View.dynamic_reshape_indexer converts 2D kernel indices to 1D flat indices
        3. The load result is used directly in the kernel's 2D operations

        Shapes:
        - Prologue input: x_flat [M*N] -> ops -> view to [M, N] for kernel
        - Kernel: operates on [M, N]
        - Epilogue: out [M, N] -> same-shaped ops [M, N]
        """
        m, n = 128, 256
        # Flat prologue input
        x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        x_flat_scale = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Same-shaped epilogue biases
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x_flat, x_flat_scale, weight, out_bias, res_bias):
            # Prologue on DIFFERENTLY shaped input [M*N]
            x_scaled = x_flat * x_flat_scale  # [M*N]
            x_transformed = torch.relu(x_scaled)  # [M*N]
            # Reshape to kernel's expected shape [M, N]
            x = x_transformed.view(m, n)

            out, residual, info = rms_norm_multi_output(x, weight)

            # Epilogue on SAME shaped outputs [M, N]
            out_processed = torch.relu(out) + out_bias  # [M, N]
            res_processed = torch.sigmoid(residual) + res_bias  # [M, N]
            return out_processed, res_processed, info

        inputs = (x_flat, x_flat_scale, weight, out_bias, res_bias)

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

    def test_differently_shaped_epilogue_same_element_count(self):
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


if __name__ == "__main__":
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    unittest.main()
