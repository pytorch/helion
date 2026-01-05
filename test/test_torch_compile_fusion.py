from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import count_triton_kernels
import helion.language as hl
from helion.runtime.kernel import BoundKernel


@helion.kernel(
    static_shapes=True, autotune_effort="quick", allow_torch_compile_fusion=True
)
def elementwise_two_inputs(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Elementwise kernel with two inputs: returns x + y and scalar 42."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        y_tile = y[tile_m, :]
        out[tile_m, :] = x_tile + y_tile

    return out, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def elementwise_with_different_shapes_same_elements(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
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
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def elementwise_2d(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, int]:
    """2D elementwise kernel: returns x * scale and scalar 42."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def elementwise_2d_single_output(x: torch.Tensor, scale: float) -> torch.Tensor:
    """2D elementwise kernel: returns x * scale (single output, no scalar)."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def elementwise_3d(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, int]:
    """3D elementwise kernel: returns x * scale and scalar 42."""
    d1, d2, d3 = x.size()
    out = torch.empty_like(x)

    for tile_d1 in hl.tile(d1):
        for tile_d2 in hl.tile(d2):
            x_tile = x[tile_d1, tile_d2, :]
            out[tile_d1, tile_d2, :] = x_tile * scale

    return out, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def elementwise_2d_explicit_dtype(x: torch.Tensor, scale: float) -> torch.Tensor:
    """2D elementwise kernel using explicit dtype=x.dtype for output allocation."""
    m, n = x.size()
    # Use explicit dtype=x.dtype instead of empty_like to test dtype propagation
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def elementwise_4d(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, int]:
    """4D elementwise kernel: returns x * scale and scalar 42."""
    b, h, s, d = x.size()
    out = torch.empty_like(x)

    for tile_b in hl.tile(b):
        for tile_h in hl.tile(h):
            for tile_s in hl.tile(s):
                x_tile = x[tile_b, tile_h, tile_s, :]
                out[tile_b, tile_h, tile_s, :] = x_tile * scale

    return out, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def rms_norm_multi_output(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """RMS normalization: returns (out, residual, 42)."""
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
        residual[tile_m, :] = normalized.to(out.dtype)

    return out, residual, 42


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def rms_norm_multi_output_extended(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
    """RMS normalization: returns (out, residual, 42, extra, 99)."""
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
        residual[tile_m, :] = normalized.to(out.dtype)
        extra[tile_m, :] = (normalized * 2).to(out.dtype)

    return out, residual, 42, extra, 99


class TestBasicFusion(RefEagerTestDisabled, TestCase):
    """Test basic prologue and epilogue fusion without view operations."""

    def test_rms_norm_epilogue_only(self):
        """Epilogue fusion only (no prologue)."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, out_bias, res_bias):
            out, residual, info = rms_norm_multi_output(x, weight)
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        inputs = (x, weight, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_rms_norm_prologue_only(self):
        """Prologue fusion only (no epilogue)."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, x_scale):
            x_transformed = torch.sigmoid(torch.relu(x * x_scale))
            return rms_norm_multi_output(x_transformed, weight)

        inputs = (x, weight, x_scale)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_rms_norm_prologue_epilogue_multi_output(self):
        """Chained prologue/epilogue ops with multiple outputs including scalars."""
        m, n = 128, 256

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        extra_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(
            x,
            x_scale,
            x_bias,
            weight,
            out_bias,
            out_scale,
            res_bias,
            res_scale,
            extra_bias,
        ):
            x_scaled = x * x_scale
            x_relu = torch.relu(x_scaled)
            x_biased = x_relu + x_bias
            x_transformed = torch.sigmoid(x_biased)

            out, residual, info1, extra, info2 = rms_norm_multi_output_extended(
                x_transformed, weight
            )

            out_relu = torch.relu(out)
            out_biased = out_relu + out_bias
            out_scaled = out_biased * out_scale
            out_processed = torch.tanh(out_scaled)

            res_biased = residual + res_bias
            res_scaled = res_biased * res_scale
            res_processed = torch.sigmoid(res_scaled)

            extra_biased = extra + extra_bias
            extra_processed = torch.relu(extra_biased)

            return out_processed, res_processed, info1, extra_processed, info2

        inputs = (
            x,
            x_scale,
            x_bias,
            weight,
            out_bias,
            out_scale,
            res_bias,
            res_scale,
            extra_bias,
        )

        out_eager, res_eager, info1_eager, extra_eager, info2_eager = f(*inputs)
        self.assertEqual(info1_eager, 42)
        self.assertEqual(info2_eager, 99)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

        out_compiled, res_compiled, info1_compiled, extra_compiled, info2_compiled = (
            result
        )
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(extra_compiled, extra_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info1_compiled, 42)
        self.assertEqual(info2_compiled, 99)

        self.assertEqual(list(out_compiled.shape), [m, n])
        self.assertEqual(list(res_compiled.shape), [m, n])
        self.assertEqual(list(extra_compiled.shape), [m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_rms_norm_simple_prologue_epilogue(self):
        """Simple single-op prologue and epilogue."""
        m, n = 16, 32
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, x_scale, out_bias, res_bias):
            x_transformed = x * x_scale
            out, residual, info = rms_norm_multi_output(x_transformed, weight)
            out_processed = out + out_bias
            res_processed = residual + res_bias
            return out_processed, res_processed, info

        inputs = (x, weight, x_scale, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_rms_norm_chain_prologue_chain_epilogue(self):
        """Chained prologue ops (relu, sigmoid) and chained epilogue ops (tanh, add)."""
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
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_multi_output_different_shapes_epilogue(self):
        """Differently-shaped epilogue outputs with same element count."""
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

        self.assertEqual(list(result[0].shape), [m * n])
        self.assertEqual(list(result[1].shape), [m, n])
        self.assertNotEqual(list(result[0].shape), list(result[1].shape))
        self.assertEqual(result[0].numel(), result[1].numel())

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_reinterpret_view_input(self):
        """Pass a transposed tensor directly to Helion kernel (ReinterpretView)."""
        m, n = 64, 128
        kernel_scale = 2.0
        x_original = torch.randn(n, m, device=DEVICE, dtype=torch.float32)

        def f(x_original):
            x = x_original.T  # ReinterpretView: swapped strides
            return elementwise_2d(x, kernel_scale)

        eager_result = f(x_original)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x_original)
        torch.testing.assert_close(result[0], eager_result[0], rtol=1e-3, atol=1e-3)
        self.assertEqual(result[1], 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_base_view_input(self):
        """Pass a reshaped tensor directly to Helion kernel (BaseView)."""
        m, n = 64, 128
        kernel_scale = 2.0
        x_original = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

        def f(x_original):
            x = x_original.view(m, n)  # BaseView: contiguous reshape
            return elementwise_2d(x, kernel_scale)

        eager_result = f(x_original)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x_original)
        torch.testing.assert_close(result[0], eager_result[0], rtol=1e-3, atol=1e-3)
        self.assertEqual(result[1], 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    def test_autotune_no_fusion_final_has_fusion(self):
        """Test that fusion is applied AFTER autotuning, not during."""
        x = torch.randn(128, 256, device=DEVICE)
        y = torch.randn(128, 256, device=DEVICE)
        bias = torch.randn(128, 256, device=DEVICE)

        autotune_codes = []
        orig = BoundKernel.compile_config

        def track(self, config=None, **kw):
            if config:
                autotune_codes.append(self.to_triton_code(config))
            return orig(self, config, **kw)

        def f(x, y, bias):
            out, _ = elementwise_two_inputs(x, y)
            return torch.relu(out) + bias

        with patch.object(BoundKernel, "compile_config", track):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(f), x, y, bias)

        torch.testing.assert_close(result, torch.relu(x + y) + bias)
        self.assertGreater(len(autotune_codes), 0)
        # Autotune code should NOT have fusion
        for c in autotune_codes:
            self.assertNotIn("relu", c.lower())
            self.assertNotIn("epilogue_input", c)
        # Final code SHOULD have fusion
        self.assertIn("relu", code.lower())
        self.assertIn("epilogue", code.lower())

    def test_inductor_output_code_has_helion_generated_triton_kernel(self):
        """Test that Inductor output code contains Helion-generated Triton code."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Prologue inputs
        x_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        # Epilogue inputs
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, x_scale, out_bias, res_bias):
            # Prologue: operations before Helion kernel that get fused in
            x_transformed = torch.sigmoid(torch.relu(x * x_scale))
            # Helion kernel
            out, residual, info = rms_norm_multi_output(x_transformed, weight)
            # Epilogue: operations after Helion kernel that get fused in
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        # Compute expected result eagerly
        inputs = (x, weight, x_scale, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        # Compile and get generated code
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        # Verify correct results
        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        # Check Inductor generated exactly one source code block
        self.assertEqual(len(source_codes), 1)
        code = source_codes[0]

        # Verify Helion-specific patterns in generated code:
        import re

        # The Helion device function name pattern: _helion_{kernel_name}
        self.assertIn(
            "_helion_rms_norm_multi_output",
            code,
            "Inductor output should contain the Helion device function",
        )

        # The @triton.jit decorator
        self.assertIn(
            "@triton.jit",
            code,
            "Inductor output should contain @triton.jit decorator",
        )

        # The _launcher call pattern (Helion's kernel launch mechanism)
        self.assertIn(
            "_launcher",
            code,
            "Inductor output should contain Helion's _launcher call",
        )

        # Standard Triton imports
        self.assertIn("import triton", code)
        self.assertIn("triton.language", code)

        # Helion's _default_launcher import
        self.assertIn("_default_launcher", code)

        # Verify prologue fusion - prologue_input indicates fused prologue buffer
        self.assertIn(
            "prologue",
            code.lower(),
            "Inductor output should contain prologue fusion",
        )

        # Verify epilogue fusion - epilogue_input indicates fused epilogue buffer
        self.assertIn(
            "epilogue",
            code.lower(),
            "Inductor output should contain epilogue fusion",
        )

        # Verify only 1 Triton kernel (both prologue and epilogue fusion successful)
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 fused kernel, got {kernel_count}"
        )

        # Verify the Triton kernel is defined INLINE with @triton.jit and _helion_ prefix
        # (the generated code contains the full Triton kernel, not just an import)
        triton_kernel_pattern = r"@triton\.jit\s+def\s+(\w+)"
        triton_kernels = re.findall(triton_kernel_pattern, code)
        self.assertGreater(
            len(triton_kernels), 0, "Should have at least one @triton.jit kernel"
        )
        for kernel_name in triton_kernels:
            self.assertTrue(
                kernel_name.startswith("_helion_"),
                f"Triton kernel '{kernel_name}' should start with '_helion_'",
            )

        # Verify the original Helion kernel definition is NOT in the generated code
        # (only the generated Triton kernel should be present)
        # Note: imports from helion.runtime are OK (needed for launcher)
        original_kernel_def_pattern = r"def\s+rms_norm_multi_output\s*\("
        self.assertIsNone(
            re.search(original_kernel_def_pattern, code),
            "Generated code should not contain original Helion kernel definition",
        )

        # Verify the Triton kernel body is actually present (not just a stub/import)
        # Check for tl.load/tl.store which are core Triton operations
        self.assertIn("tl.load", code, "Triton kernel should have tl.load operations")
        self.assertIn("tl.store", code, "Triton kernel should have tl.store operations")


class TestViewFusion(RefEagerTestDisabled, TestCase):
    """Test view-related prologue and epilogue fusion.

    NOTE: These tests currently don't fuse to 1 kernel (which is an upcoming feature).
    """

    def _run_fusion_test(self, f, inputs, expect_one_kernel=False, rtol=1e-3, atol=1e-3):
        """Run eager vs compiled, compare outputs, verify kernel count."""
        eager_result = f(*inputs)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        if isinstance(eager_result, tuple):
            for compiled, eager in zip(result, eager_result, strict=True):
                if isinstance(compiled, torch.Tensor):
                    torch.testing.assert_close(compiled, eager, rtol=rtol, atol=atol)
        else:
            torch.testing.assert_close(result, eager_result, rtol=rtol, atol=atol)

        kernel_count, _ = count_triton_kernels(source_codes)
        if expect_one_kernel:
            self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")
        else:
            self.assertGreater(
                kernel_count, 1, f"Expected >1 kernels, got {kernel_count}"
            )
        return result

    @parametrize("direction", ("prologue", "epilogue"))
    @parametrize("axis", ("row", "column"))
    @parametrize("side", ("left", "right"))
    def test_slice(self, direction, axis, side):
        """Slice tests covering all combinations of direction, axis, and side."""
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        kernel_scale = 2.0

        if direction == "prologue":
            if axis == "column":
                if side == "left":
                    x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                    x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

                    def f(x, x_scale):
                        x_sliced = x[:, :half_n]
                        x_scaled = x_sliced * x_scale
                        x_transformed = torch.relu(x_scaled)
                        return elementwise_2d(x_transformed, kernel_scale)

                    inputs = (x, x_scale)
                else:  # right
                    x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                    x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

                    def f(x, x_scale):
                        x_sliced = x[:, half_n:]
                        x_scaled = x_sliced * x_scale
                        x_transformed = torch.relu(x_scaled)
                        return elementwise_2d(x_transformed, kernel_scale)

                    inputs = (x, x_scale)
            else:  # row
                if side == "left":
                    x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                    x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

                    def f(x, x_scale):
                        x_sliced = x[:half_m, :]
                        x_scaled = x_sliced * x_scale
                        x_transformed = torch.sigmoid(x_scaled)
                        return elementwise_2d(x_transformed, kernel_scale)

                    inputs = (x, x_scale)
                else:  # right
                    x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                    x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

                    def f(x, x_scale):
                        x_sliced = x[half_m:, :]
                        x_scaled = x_sliced * x_scale
                        x_transformed = torch.sigmoid(x_scaled)
                        return elementwise_2d(x_transformed, kernel_scale)

                    inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
            if axis == "column":
                epilogue_bias = torch.randn(
                    m, half_n, device=DEVICE, dtype=torch.float32
                )
                if side == "left":

                    def f(x, epilogue_bias):
                        out, info = elementwise_2d(x, kernel_scale)
                        out_sliced = out[:, :half_n]
                        out_processed = torch.relu(out_sliced) + epilogue_bias
                        return out_processed, info
                else:  # right

                    def f(x, epilogue_bias):
                        out, info = elementwise_2d(x, kernel_scale)
                        out_sliced = out[:, half_n:]
                        out_processed = torch.relu(out_sliced) + epilogue_bias
                        return out_processed, info

                inputs = (x, epilogue_bias)
            else:  # row
                epilogue_bias = torch.randn(
                    half_m, n, device=DEVICE, dtype=torch.float32
                )
                if side == "left":

                    def f(x, epilogue_bias):
                        out, info = elementwise_2d(x, kernel_scale)
                        out_sliced = out[:half_m, :]
                        out_processed = torch.sigmoid(out_sliced) + epilogue_bias
                        return out_processed, info
                else:  # right

                    def f(x, epilogue_bias):
                        out, info = elementwise_2d(x, kernel_scale)
                        out_sliced = out[half_m:, :]
                        out_processed = torch.sigmoid(out_sliced) + epilogue_bias
                        return out_processed, info

                inputs = (x, epilogue_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("axis", ("row", "column"))
    @parametrize("side", ("left", "right"))
    def test_single_output_slice_epilogue(self, axis, side):
        """Single-output slice epilogue tests."""
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        kernel_scale = 2.0
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        if axis == "column":
            epilogue_bias = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
            if side == "left":

                def f(x, epilogue_bias):
                    out = elementwise_2d_single_output(x, kernel_scale)
                    out_sliced = out[:, :half_n]
                    return torch.relu(out_sliced) + epilogue_bias
            else:  # right

                def f(x, epilogue_bias):
                    out = elementwise_2d_single_output(x, kernel_scale)
                    out_sliced = out[:, half_n:]
                    return torch.relu(out_sliced) + epilogue_bias
        else:  # row
            epilogue_bias = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)
            if side == "left":

                def f(x, epilogue_bias):
                    out = elementwise_2d_single_output(x, kernel_scale)
                    out_sliced = out[:half_m, :]
                    return torch.sigmoid(out_sliced) + epilogue_bias
            else:  # right

                def f(x, epilogue_bias):
                    out = elementwise_2d_single_output(x, kernel_scale)
                    out_sliced = out[half_m:, :]
                    return torch.sigmoid(out_sliced) + epilogue_bias

        inputs = (x, epilogue_bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    @parametrize("flatten_type", ("full", "partial"))
    def test_reshape(self, direction, flatten_type):
        """Reshape tests for multi-output kernels."""
        kernel_scale = 2.0

        if flatten_type == "full":
            m, n = 128, 256
            if direction == "prologue":
                x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
                x_flat_scale = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
                x_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
                out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
                res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

                def f(x_flat, x_flat_scale, x_bias, weight, out_bias, res_bias):
                    x_scaled = x_flat * x_flat_scale
                    x_transformed = torch.relu(x_scaled)
                    x = x_transformed.view(m, n) + x_bias
                    out, residual, info = rms_norm_multi_output(x, weight)
                    out_processed = torch.relu(out) + out_bias
                    res_processed = torch.sigmoid(residual) + res_bias
                    return out_processed, res_processed, info

                inputs = (x_flat, x_flat_scale, x_bias, weight, out_bias, res_bias)
            else:  # epilogue
                x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
                out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
                flat_bias = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

                def f(x, weight, out_bias, flat_bias):
                    out, residual, info = rms_norm_multi_output(x, weight)
                    out_processed = torch.relu(out) + out_bias
                    flattened = residual.view(m * n)
                    flat_processed = torch.sigmoid(flattened) + flat_bias
                    return out_processed, flat_processed, info

                inputs = (x, weight, out_bias, flat_bias)
        else:  # partial
            d1, d2, d3 = 8, 16, 32
            if direction == "prologue":
                x_partial = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
                x_scale = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
                out_bias = torch.randn(d3, device=DEVICE, dtype=torch.float32)

                def f(x_partial, x_scale, out_bias):
                    x_scaled = x_partial * x_scale
                    x_transformed = torch.relu(x_scaled)
                    x = x_transformed.view(d1 * d2, d3)
                    out, info = elementwise_2d(x, kernel_scale)
                    out_processed = torch.sigmoid(out) + out_bias
                    return out_processed, info

                inputs = (x_partial, x_scale, out_bias)
            else:  # epilogue
                x = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
                out_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

                def f(x, out_bias):
                    out, info = elementwise_2d(x, kernel_scale)
                    out_reshaped = out.view(d1, d2 * d3)
                    out_processed = torch.sigmoid(out_reshaped) + out_bias
                    return out_processed, info

                inputs = (x, out_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    @parametrize("flatten_type", ("full", "partial"))
    def test_single_output_reshape(self, direction, flatten_type):
        """Reshape tests for single-output kernels."""
        kernel_scale = 2.0

        if flatten_type == "full":
            m, n = 128, 256
            if direction == "prologue":
                x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float32)
                x_flat_scale = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

                def f(x_flat, x_flat_scale):
                    x_scaled = x_flat * x_flat_scale
                    x_transformed = torch.relu(x_scaled)
                    x = x_transformed.view(m, n)
                    return elementwise_2d_single_output(x, kernel_scale)

                inputs = (x_flat, x_flat_scale)
            else:  # epilogue
                x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                flat_bias = torch.randn(m * n, device=DEVICE, dtype=torch.float32)

                def f(x, flat_bias):
                    out = elementwise_2d_single_output(x, kernel_scale)
                    flattened = out.view(m * n)
                    return torch.sigmoid(flattened) + flat_bias

                inputs = (x, flat_bias)
        else:  # partial
            d1, d2, d3 = 8, 16, 32
            if direction == "prologue":
                x_partial = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
                x_scale = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

                def f(x_partial, x_scale):
                    x_scaled = x_partial * x_scale
                    x_transformed = torch.relu(x_scaled)
                    x = x_transformed.view(d1 * d2, d3)
                    return elementwise_2d_single_output(x, kernel_scale)

                inputs = (x_partial, x_scale)
            else:  # epilogue
                x = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
                out_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

                def f(x, out_bias):
                    out = elementwise_2d_single_output(x, kernel_scale)
                    out_reshaped = out.view(d1, d2 * d3)
                    return torch.sigmoid(out_reshaped) + out_bias

                inputs = (x, out_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    def test_narrow(self, direction):
        """Narrow tests for prologue and epilogue."""
        m, n = 128, 256
        start, length = 64, 128
        kernel_scale = 2.0

        if direction == "prologue":
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
            x_scale = torch.randn(m, length, device=DEVICE, dtype=torch.float32)

            def f(x, x_scale):
                x_narrowed = x.narrow(1, start, length)
                x_scaled = x_narrowed * x_scale
                x_transformed = torch.relu(x_scaled)
                return elementwise_2d(x_transformed, kernel_scale)

            inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
            epilogue_bias = torch.randn(m, length, device=DEVICE, dtype=torch.float32)

            def f(x, epilogue_bias):
                out, info = elementwise_2d(x, kernel_scale)
                out_narrowed = out.narrow(1, start, length)
                out_processed = torch.sigmoid(out_narrowed) + epilogue_bias
                return out_processed, info

            inputs = (x, epilogue_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    def test_transpose_2d(self, direction):
        """2D transpose tests."""
        m, n = 128, 256
        kernel_scale = 2.0

        if direction == "prologue":
            x = torch.randn(n, m, device=DEVICE, dtype=torch.float32)
            x_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

            def f(x, x_scale):
                x_transposed = x.T
                x_scaled = x_transposed * x_scale
                x_transformed = torch.relu(x_scaled)
                return elementwise_2d(x_transformed, kernel_scale)

            inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
            epilogue_bias = torch.randn(n, m, device=DEVICE, dtype=torch.float32)

            def f(x, epilogue_bias):
                out, info = elementwise_2d(x, kernel_scale)
                out_transposed = out.T
                out_processed = torch.sigmoid(out_transposed) + epilogue_bias
                return out_processed, info

            inputs = (x, epilogue_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    @parametrize("permute_type", ("swap_last", "swap_first", "reverse", "cyclic"))
    def test_transpose_3d(self, direction, permute_type):
        """3D transpose tests with various permutation types."""
        d1, d2, d3 = 8, 16, 32
        kernel_scale = 2.0

        # Determine permutation and input/output shapes based on permute_type
        if permute_type == "swap_last":
            perm = (0, 2, 1)
            input_shape = (d1, d3, d2) if direction == "prologue" else (d1, d2, d3)
            output_shape = (d1, d3, d2) if direction == "epilogue" else (d1, d2, d3)
        elif permute_type == "swap_first":
            if direction == "epilogue":
                # Skip swap_first for epilogue (not in original tests)
                return
            perm = (1, 0, 2)
            input_shape = (d2, d1, d3)
            output_shape = (d1, d2, d3)
        elif permute_type == "reverse":
            perm = (2, 1, 0)
            input_shape = (d3, d2, d1) if direction == "prologue" else (d1, d2, d3)
            output_shape = (d3, d2, d1) if direction == "epilogue" else (d1, d2, d3)
        else:  # cyclic
            if direction == "epilogue":
                # Skip cyclic for epilogue (not in original tests)
                return
            perm = (1, 2, 0)
            input_shape = (d3, d1, d2)
            output_shape = (d1, d2, d3)

        if direction == "prologue":
            x = torch.randn(*input_shape, device=DEVICE, dtype=torch.float32)
            x_scale = torch.randn(*output_shape, device=DEVICE, dtype=torch.float32)

            def f(x, x_scale):
                x_permuted = x.permute(*perm)
                x_scaled = x_permuted * x_scale
                x_transformed = torch.relu(x_scaled)
                return elementwise_3d(x_transformed, kernel_scale)

            inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(*input_shape, device=DEVICE, dtype=torch.float32)
            epilogue_bias = torch.randn(
                *output_shape, device=DEVICE, dtype=torch.float32
            )

            def f(x, epilogue_bias):
                out, info = elementwise_3d(x, kernel_scale)
                out_permuted = out.permute(*perm)
                out_processed = torch.relu(out_permuted) + epilogue_bias
                return out_processed, info

            inputs = (x, epilogue_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    def test_transpose_3d_same_size_dims(self, direction):
        """3D transpose with two same-size dims (edge case)."""
        a, b = 16, 32
        kernel_scale = 2.0

        if direction == "prologue":
            x = torch.randn(a, a, b, device=DEVICE, dtype=torch.float32)
            x_scale = torch.randn(a, a, b, device=DEVICE, dtype=torch.float32)

            def f(x, x_scale):
                x_permuted = x.permute(1, 0, 2)
                x_scaled = x_permuted * x_scale
                x_transformed = torch.relu(x_scaled)
                return elementwise_3d(x_transformed, kernel_scale)

            inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(a, b, a, device=DEVICE, dtype=torch.float32)
            epilogue_bias = torch.randn(a, b, a, device=DEVICE, dtype=torch.float32)

            def f(x, epilogue_bias):
                out, info = elementwise_3d(x, kernel_scale)
                out_permuted = out.permute(2, 1, 0)
                out_processed = torch.sigmoid(out_permuted) + epilogue_bias
                return out_processed, info

            inputs = (x, epilogue_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    def test_transpose_4d(self, direction):
        """4D transpose tests swapping middle dims."""
        b, h, s, d = 2, 8, 64, 32
        kernel_scale = 2.0

        if direction == "prologue":
            x = torch.randn(b, s, h, d, device=DEVICE, dtype=torch.float32)
            x_scale = torch.randn(b, h, s, d, device=DEVICE, dtype=torch.float32)

            def f(x, x_scale):
                x_permuted = x.permute(0, 2, 1, 3)
                x_scaled = x_permuted * x_scale
                x_transformed = torch.relu(x_scaled)
                return elementwise_4d(x_transformed, kernel_scale)

            inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(b, h, s, d, device=DEVICE, dtype=torch.float32)
            epilogue_bias = torch.randn(b, s, h, d, device=DEVICE, dtype=torch.float32)

            def f(x, epilogue_bias):
                out, info = elementwise_4d(x, kernel_scale)
                out_permuted = out.permute(0, 2, 1, 3)
                out_processed = torch.sigmoid(out_permuted) + epilogue_bias
                return out_processed, info

            inputs = (x, epilogue_bias)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("direction", ("prologue", "epilogue"))
    @parametrize(
        "op_order",
        (
            "view_then_slice",
            "slice_then_view",
            "transpose_then_slice",
            "slice_then_transpose",
        ),
    )
    def test_combined_view_ops(self, direction, op_order):
        """Combined view operations tests."""
        m, n = 64, 128
        half_m, half_n = m // 2, n // 2
        kernel_scale = 2.0

        if direction == "prologue":
            if op_order == "view_then_slice":
                x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

                def f(x, x_scale):
                    x_viewed = x.view(half_m, 2 * n)
                    x_sliced = x_viewed[:, :n]
                    x_scaled = x_sliced * x_scale
                    x_transformed = torch.relu(x_scaled)
                    return elementwise_2d(x_transformed, kernel_scale)

                inputs = (x, x_scale)
            elif op_order == "slice_then_view":
                x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)

                def f(x, x_scale):
                    x_sliced = x[:, :half_n]
                    x_viewed = x_sliced.reshape(half_m, n)
                    x_scaled = x_viewed * x_scale
                    x_transformed = torch.sigmoid(x_scaled)
                    return elementwise_2d(x_transformed, kernel_scale)

                inputs = (x, x_scale)
            elif op_order == "transpose_then_slice":
                x = torch.randn(n, m, device=DEVICE, dtype=torch.float32)
                x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)

                def f(x, x_scale):
                    x_transposed = x.T
                    x_sliced = x_transposed[:, :half_n]
                    x_scaled = x_sliced * x_scale
                    x_transformed = torch.relu(x_scaled)
                    return elementwise_2d(x_transformed, kernel_scale)

                inputs = (x, x_scale)
            else:  # slice_then_transpose
                x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                x_scale = torch.randn(n, half_m, device=DEVICE, dtype=torch.float32)

                def f(x, x_scale):
                    x_sliced = x[:half_m, :]
                    x_transposed = x_sliced.T
                    x_scaled = x_transposed * x_scale
                    x_transformed = torch.sigmoid(x_scaled)
                    return elementwise_2d(x_transformed, kernel_scale)

                inputs = (x, x_scale)
        else:  # epilogue
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
            if op_order == "view_then_slice":
                epilogue_bias = torch.randn(
                    half_m, n, device=DEVICE, dtype=torch.float32
                )

                def f(x, epilogue_bias):
                    out, info = elementwise_2d(x, kernel_scale)
                    out_viewed = out.view(half_m, 2 * n)
                    out_sliced = out_viewed[:, :n]
                    out_processed = torch.relu(out_sliced) + epilogue_bias
                    return out_processed, info

                inputs = (x, epilogue_bias)
            elif op_order == "slice_then_view":
                # Skip - not in original epilogue tests
                return
            elif op_order == "transpose_then_slice":
                epilogue_bias = torch.randn(
                    n, half_m, device=DEVICE, dtype=torch.float32
                )

                def f(x, epilogue_bias):
                    out, info = elementwise_2d(x, kernel_scale)
                    out_transposed = out.T
                    out_sliced = out_transposed[:, :half_m]
                    out_processed = torch.sigmoid(out_sliced) + epilogue_bias
                    return out_processed, info

                inputs = (x, epilogue_bias)
            else:  # slice_then_transpose
                # Skip - not in original epilogue tests
                return

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    @parametrize("expand_type", ("broadcast", "permute", "multi_dim"))
    def test_expand_epilogue(self, expand_type):
        """Expand epilogue tests."""
        kernel_scale = 2.0

        if expand_type == "broadcast":
            b, m, n = 4, 32, 64
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

            def f(x):
                out = elementwise_2d_single_output(x, kernel_scale)
                out_view = out.unsqueeze(0).expand(b, m, n)
                return torch.relu(out_view)

            inputs = (x,)
        elif expand_type == "permute":
            b, m, n = 3, 32, 16
            x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

            def f(x):
                out = elementwise_2d_single_output(x, kernel_scale)
                out_view = out.unsqueeze(0).expand(b, m, n)
                out_perm = out_view.permute(1, 0, 2)
                return torch.tanh(out_perm)

            inputs = (x,)
        else:  # multi_dim
            b, d1, d2, d3 = 3, 8, 16, 32
            x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)

            def f(x):
                out, info = elementwise_3d(x, kernel_scale)
                out_view = out.unsqueeze(1).expand(d1, b, d2, d3)
                out_processed = torch.relu(out_view)
                return out_processed, info

            inputs = (x,)

        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_combined_slice_prologue_and_epilogue(self):
        """Combined slice prologue x[:, half:] and slice epilogue out[:half, :]."""
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, x_scale, epilogue_bias):
            x_sliced = x[:, half_n:]
            x_scaled = x_sliced * x_scale
            x_transformed = torch.relu(x_scaled)
            out, info = elementwise_2d(x_transformed, kernel_scale)
            out_sliced = out[:half_m, :]
            out_processed = torch.sigmoid(out_sliced) + epilogue_bias
            return out_processed, info

        inputs = (x, x_scale, epilogue_bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_permute_3d_prologue(self):
        """3D permute prologue x.permute(0, 2, 1) -> view -> kernel."""
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d3, d2, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_permuted = x.permute(0, 2, 1)
            x_viewed = x_permuted.reshape(d1 * d2, d3)
            x_scaled = x_viewed * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_2d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_unsqueeze_squeeze_prologue(self):
        """Unsqueeze/squeeze prologue x.unsqueeze(0).squeeze(0) -> kernel."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_unsqueezed = x.unsqueeze(0)
            x_squeezed = x_unsqueezed.squeeze(0)
            x_scaled = x_squeezed * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_2d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_multiple_views_prologue(self):
        """Prologue: multiple consecutive views x.view().view().view() -> kernel."""
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1 * d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x1 = x.view(d1 * d2, d3)
            x2 = x1.view(d1, d2 * d3)
            x3 = x2.view(d1 * d2, d3)
            x_scaled = x3 * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_2d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_view_unflatten_prologue(self):
        """Prologue: unflatten operation x.unflatten(1, (d2, d3)) -> kernel."""
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_unflat = x.unflatten(1, (d2, d3))
            x_scaled = x_unflat * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_3d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_flatten_epilogue(self):
        """Epilogue: flatten operation out.flatten(1, 2) -> ops."""
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_3d(x, kernel_scale)
            out_flat = out.flatten(1, 2)
            out_processed = torch.sigmoid(out_flat) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_expand_prologue(self):
        """Prologue: expand operation x.expand(B, M, N) -> kernel."""
        b, m, n = 4, 64, 128
        x = torch.randn(1, m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(b, m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_expanded = x.expand(b, m, n)
            x_scaled = x_expanded * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_3d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_contiguous_after_transpose_prologue(self):
        """Prologue: transpose + contiguous x.T.contiguous() -> kernel."""
        m, n = 64, 128
        x = torch.randn(n, m, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_t = x.T
            x_contig = x_t.contiguous()
            x_scaled = x_contig * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_2d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_3d_view_to_2d_then_transpose_prologue(self):
        """Prologue: 3D->2D view then transpose x.view(D1, D2*D3).T -> kernel."""
        d1, d2, d3 = 8, 16, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(d2 * d3, d1, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_2d = x.view(d1, d2 * d3)
            x_t = x_2d.T
            x_scaled = x_t * x_scale
            x_transformed = torch.sigmoid(x_scaled)
            return elementwise_2d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_transpose_then_view_to_3d_epilogue(self):
        """Epilogue: transpose then 2D->3D view out.T.reshape(D1, D2, D3) -> ops."""
        d1, d2, d3 = 8, 16, 32
        m, n = d1 * d2, d3
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d3, d1, d2, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)
            out_t = out.T
            out_3d = out_t.reshape(d3, d1, d2)
            out_processed = torch.relu(out_3d) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_strided_slice_prologue(self):
        """Strided slice prologue x[::2, :] -> kernel."""
        m, n = 128, 256
        half_m = m // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn(half_m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, x_scale):
            x_strided = x[::2, :]
            x_scaled = x_strided * x_scale
            x_transformed = torch.relu(x_scaled)
            return elementwise_2d(x_transformed, kernel_scale)

        inputs = (x, x_scale)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_strided_slice_epilogue(self):
        """Strided slice epilogue out[::2, ::2] -> ops."""
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(half_m, half_n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)
            out_strided = out[::2, ::2]
            out_processed = torch.sigmoid(out_strided) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_diagonal_slice_epilogue(self):
        """Diagonal slice epilogue out.diagonal() -> ops."""
        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d(x, kernel_scale)
            out_diag = out.diagonal()
            out_processed = torch.relu(out_diag) + epilogue_bias
            return out_processed, info

        inputs = (x, epilogue_bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_as_strided_prologue_overlap(self):
        """Overlapping as_strided in prologue should not fuse."""
        m, n = 32, 64
        x = torch.randn(m + n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            x_view = x.as_strided((m, n), (1, 1))
            x_processed = torch.relu(x_view)
            return elementwise_2d(x_processed, kernel_scale)

        inputs = (x,)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

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
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_select_epilogue_rank_drop(self):
        """Rank drop epilogue: out[:, 0, :] -> ops (no fusion)."""
        d1, d2, d3 = 16, 8, 32
        x = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x):
            out, info = elementwise_3d(x, kernel_scale)
            out_selected = out[:, 0, :]
            return torch.relu(out_selected), info

        inputs = (x,)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_helion_single_kernel_views(self):
        """Helion prologue+epilogue view/slice path should not fuse to one kernel."""
        m, n = 128, 256
        half_m, half_n = m // 2, n // 2
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(m, half_n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(half_m, half_n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, scale, bias):
            x_sliced = x[:, half_n:]
            x_scaled = torch.relu(x_sliced * scale)
            out, info = elementwise_2d(x_scaled, kernel_scale)
            out_sliced = out[:half_m, :]
            out_processed = torch.sigmoid(out_sliced) + bias
            return out_processed, info

        inputs = (x, scale, bias)
        self._run_fusion_test(f, inputs, expect_one_kernel=False)

    def test_partial_prologue_fusion_view_and_pointwise(self):
        """Partial fusion: x has view (no fusion), y has pointwise (should fuse)."""
        d1, d2, d3 = 8, 16, 32
        m, n = d1 * d2, d3

        x_original = torch.randn(d1, d2 * d3, device=DEVICE, dtype=torch.float32)
        y_original = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x_original, y_original, y_scale):
            x = x_original.view(m, n)
            y = torch.relu(y_original) * y_scale
            out, info = elementwise_two_inputs(x, y)
            return out, info

        inputs = (x_original, y_original, y_scale)
        eager_result = f(*inputs)
        self.assertEqual(eager_result[1], 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, eager_result[0], rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)
        self.assertEqual(list(out_compiled.shape), [m, n])

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreater(
            kernel_count, 1, f"Expected >1 kernels due to view, got {kernel_count}"
        )

        # Verify partial fusion
        helion_kernel_found = False
        for code in source_codes:
            if "elementwise_two_inputs" in code or "triton_heuristics" in code:
                if (
                    "relu" in code.lower()
                    or "maximum" in code.lower()
                    or "tl.where" in code
                ):
                    helion_kernel_found = True
                    break

        self.assertTrue(
            helion_kernel_found,
            "Expected y's relu to be fused into Helion kernel (partial fusion)",
        )


instantiate_parametrized_tests(TestViewFusion)


def _run(fn, args):
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, backend="inductor")(*args)


def _get_spec(fn, args):
    torch._dynamo.reset()
    from helion._compiler._dynamo.variables import HelionKernelVariable

    captured = []
    orig = HelionKernelVariable._infer_output_spec
    with patch.object(
        HelionKernelVariable,
        "_infer_output_spec",
        lambda s, a: (spec := orig(s, a), captured.append(spec))[0],
    ):
        torch.compile(fn, fullgraph=True, backend="inductor")(*args)
    return captured[-1] if captured else {}


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_inplace(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_two_mutated(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile in hl.tile(x.size(0)):
        x[tile], y[tile] = x[tile] + 1, y[tile] * 2
    return x, y


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_one_mutated(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + y[tile]
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_mut_and_out(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1
        out[tile] = x[tile] + y[tile]
    return x, out


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_mut_no_return(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1
        out[tile] = y[tile] * 2
    return out


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.store(x, [tile], y[tile] * 2)
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_atomic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile], y[tile])
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_store_kwarg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.store(tensor=x, index=[tile], value=y[tile] * 2)
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_atomic_kwarg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(target=x, index=[tile], value=y[tile])
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_alias(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        x[tile] = y[tile] + 1
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_alias_local(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    alias = x
    tmp = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        tmp[tile] = y[tile] + 1
    return alias


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_alias_multi(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return x, out


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_view(x: torch.Tensor) -> torch.Tensor:
    y = x.view(x.size())
    for tile in hl.tile(x.size(0)):
        y[tile] = y[tile] + 1
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_inline_triton_unknown(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.inline_triton("0", args=(), output_like=None)
        x[tile] = x[tile] + y[tile]
    return x


@helion.kernel(
    static_shapes=True, autotune_effort="none", allow_torch_compile_fusion=True
)
def k_no_mut(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


class TestMutation(RefEagerTestDisabled, TestCase):
    def test_inplace(self):
        def fn(x):
            x = x * 2
            x = k_inplace(x)
            return x + 1

        x = torch.randn(64, device=DEVICE)
        expected = (x * 2 + 1) + 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))
        self.assertIn(
            "x", _get_spec(fn, (torch.randn(64, device=DEVICE),))["mutated_inputs"]
        )

        # Mutation should still allow fusion (pure torch.compile fuses this pattern)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(compiled_fn, x.clone())
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

    def test_mutate_output_alias(self):
        def fn(x):
            y = k_inplace(x)
            y.add_(1)
            return x

        x = torch.randn(64, device=DEVICE)
        expected = x + 2
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))

    def test_mutation_fusion_copy(self):
        def fn(a, b, c):
            a0 = a + c
            b0 = b + a0
            b.copy_(b0)
            a.copy_(a0)
            return k_no_mut(a, b)

        a = torch.randn(64, device=DEVICE)
        b = torch.randn(64, device=DEVICE)
        c = torch.randn(64, device=DEVICE)
        a_expected = a + c
        b_expected = b + a_expected
        out_expected = a_expected + b_expected

        a_actual = a.clone()
        b_actual = b.clone()
        c_actual = c.clone()
        out_actual = _run(fn, (a_actual, b_actual, c_actual))

        self.assertTrue(torch.allclose(out_actual, out_expected))
        self.assertTrue(torch.allclose(a_actual, a_expected))
        self.assertTrue(torch.allclose(b_actual, b_expected))

    def test_input_mutation_view_copy(self):
        def fn(a):
            before = a + 1
            a_flat = a.view(-1)
            a_flat.copy_(
                torch.tensor([66.0], device=a.device, dtype=a.dtype)
            )
            a_flat.add_(1)
            return before, a_flat

        a = torch.randn(1, 64, device=DEVICE)
        a_expected = a.clone()
        before_expected = a_expected + 1
        a_expected_flat = a_expected.view(-1)
        a_expected_flat.copy_(
            torch.tensor([66.0], device=a.device, dtype=a.dtype)
        )
        a_expected_flat.add_(1)
        out_expected = a_expected_flat.clone()

        a_actual = a.clone()
        before_actual, out_actual = _run(fn, (a_actual,))

        self.assertTrue(torch.allclose(before_actual, before_expected))
        self.assertTrue(torch.allclose(out_actual, out_expected))
        self.assertTrue(torch.allclose(a_actual, a_expected))

    def test_mutation_after_kernel_preserves_output(self):
        def fn(x, y):
            out = k_no_mut(x, y)
            x.add_(10)
            return out

        x = torch.randn(64, device=DEVICE)
        y = torch.randn(64, device=DEVICE)
        out_expected = x + y
        x_expected = x + 10

        x_actual = x.clone()
        y_actual = y.clone()
        out_actual = _run(fn, (x_actual, y_actual))

        self.assertTrue(torch.allclose(out_actual, out_expected))
        self.assertTrue(torch.allclose(x_actual, x_expected))
        self.assertTrue(torch.allclose(y_actual, y))

    def test_two_mutated(self):
        def fn(x, y):
            x, y = x + 1, y - 1
            x, y = k_two_mutated(x, y)
            return x * 2, y * 2

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        exp_x, exp_y = ((x + 1) + 1) * 2, ((y - 1) * 2) * 2
        rx, ry = _run(fn, (x.clone(), y.clone()))
        self.assertTrue(torch.allclose(rx, exp_x) and torch.allclose(ry, exp_y))
        spec = _get_spec(
            fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
        )
        self.assertEqual(set(spec["mutated_inputs"]), {"x", "y"})

    def test_one_mutated(self):
        def fn(x, y):
            y = y * 2
            x = k_one_mutated(x, y)
            return x - 1

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        expected = (x + y * 2) - 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        spec = _get_spec(
            fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
        )
        self.assertIn("x", spec["mutated_inputs"])
        self.assertNotIn("y", spec["mutated_inputs"])

        # Mutation should still allow fusion (pure torch.compile fuses this pattern)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(compiled_fn, x.clone(), y.clone())
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

    def test_mut_and_out(self):
        def fn(x, y):
            x, y = x + 1, y + 1
            x, out = k_mut_and_out(x, y)
            return (
                x,
                out,
            )  # no epilogue (epilogue fusion with multi-output has separate bug)

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        x_pre, y_pre = x + 1, y + 1
        exp_x, exp_out = x_pre + 1, x_pre + 1 + y_pre
        rx, rout = _run(fn, (x.clone(), y.clone()))
        self.assertTrue(torch.allclose(rx, exp_x) and torch.allclose(rout, exp_out))

    def test_mutation_not_returned_input(self):
        def fn(x, y):
            x = x + 1
            out = k_mut_no_return(x, y)
            return x + 1, out  # use mutated input after kernel

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        exp_x = x + 3  # +1 before, +1 in kernel, +1 after
        exp_out = y * 2
        rx, rout = _run(fn, (x.clone(), y.clone()))
        self.assertTrue(torch.allclose(rx, exp_x) and torch.allclose(rout, exp_out))
        spec = _get_spec(
            fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
        )
        self.assertIn("x", spec["mutated_inputs"])
        self.assertNotIn("y", spec["mutated_inputs"])

    def test_inplace_ignored_return(self):
        def fn(x):
            k_inplace(x)  # return ignored; still mutates x
            return x + 1

        x = torch.randn(64, device=DEVICE)
        expected = x + 2
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))
        spec = _get_spec(fn, (torch.randn(64, device=DEVICE),))
        self.assertIn("x", spec["mutated_inputs"])

        # Mutation should still allow fusion (pure torch.compile fuses this pattern)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(compiled_fn, x.clone())
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

    def test_store(self):
        def fn(x, y):
            y = y + 1
            x = k_store(x, y)
            return x - 1

        x, y = torch.zeros(64, device=DEVICE), torch.randn(64, device=DEVICE)
        expected = (y + 1) * 2 - 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        self.assertIn(
            "x",
            _get_spec(
                fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
            )["mutated_inputs"],
        )

        # Mutation should still allow fusion (pure torch.compile fuses this pattern)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(compiled_fn, x.clone(), y.clone())
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

    def test_store_kwarg(self):
        def fn(x, y):
            y = y + 1
            x = k_store_kwarg(x, y)
            return x - 1

        x, y = torch.zeros(64, device=DEVICE), torch.randn(64, device=DEVICE)
        expected = (y + 1) * 2 - 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        self.assertIn(
            "x",
            _get_spec(
                fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
            )["mutated_inputs"],
        )

        # Mutation should still allow fusion (pure torch.compile fuses this pattern)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(compiled_fn, x.clone(), y.clone())
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

    def test_atomic(self):
        """Atomic ops should prevent epilogue fusion, resulting in >1 kernels."""

        def fn(x, y):
            y = y * 2
            x = k_atomic(x, y)
            return x + 1

        x, y = torch.zeros(64, device=DEVICE), torch.ones(64, device=DEVICE)
        expected = (y * 2) + 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        self.assertIn(
            "x",
            _get_spec(
                fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
            )["mutated_inputs"],
        )

        # Verify atomics prevent fusion: should have >1 kernel
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(
            compiled_fn,
            torch.zeros(64, device=DEVICE),
            torch.ones(64, device=DEVICE),
        )
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreater(
            kernel_count, 1, f"Atomics should prevent fusion, expected >1 kernels, got {kernel_count}"
        )

    def test_atomic_kwarg(self):
        """Atomic ops (kwarg style) should prevent epilogue fusion, resulting in >1 kernels."""

        def fn(x, y):
            y = y * 2
            x = k_atomic_kwarg(x, y)
            return x + 1

        x, y = torch.zeros(64, device=DEVICE), torch.ones(64, device=DEVICE)
        expected = (y * 2) + 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        self.assertIn(
            "x",
            _get_spec(
                fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
            )["mutated_inputs"],
        )

        # Verify atomics prevent fusion: should have >1 kernel
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(
            compiled_fn,
            torch.zeros(64, device=DEVICE),
            torch.ones(64, device=DEVICE),
        )
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreater(
            kernel_count, 1, f"Atomics should prevent fusion, expected >1 kernels, got {kernel_count}"
        )

    def test_alias_same_tensor(self):
        def fn(x):
            x = x * 2
            x = k_alias(x, x)
            return x + 1

        x = torch.randn(64, device=DEVICE)
        expected = (x * 2 + 1) + 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))
        t = torch.randn(64, device=DEVICE)
        spec = _get_spec(lambda t: k_alias(t, t), (t,))
        self.assertEqual(set(spec["mutated_inputs"]), {"x", "y"})

    def test_alias_view(self):
        def fn(x):
            x = x + 1
            y = x.view(-1)
            x = k_alias(x, y)
            return x * 2

        x = torch.randn(64, device=DEVICE)
        expected = ((x + 1) + 1) * 2
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))
        t = torch.randn(64, device=DEVICE)
        spec = _get_spec(lambda t: k_alias(t, t.view(-1)), (t,))
        self.assertEqual(set(spec["mutated_inputs"]), {"x", "y"})

    def test_aliasing_inputs_used_after(self):
        def fn(x):
            y = x.view(-1)
            k_inplace(x)
            return y + 1

        x = torch.randn(64, device=DEVICE)
        expected = x + 2
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))

    def test_output_alias_local(self):
        def fn(x, y):
            x, y = x + 1, y - 1
            out = k_alias_local(x, y)
            out.add_(2)
            return x

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        expected = x + 3  # +1 before, +2 after via alias
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        spec = _get_spec(
            lambda t, u: k_alias_local(t, u),
            (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)),
        )
        self.assertEqual(spec["mutated_inputs"], [])
        self.assertEqual(spec["output_aliases"], ["x"])
        self.assertEqual(spec["output_alias_is_direct"], [True])

    def test_output_alias_multi(self):
        def fn(x, y):
            x, y = x + 1, y - 1
            x_out, out = k_alias_multi(x, y)
            x_out.add_(2)
            return x, out

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        x_pre, y_pre = x + 1, y - 1
        exp_x, exp_out = x_pre + 2, x_pre + y_pre
        rx, rout = _run(fn, (x.clone(), y.clone()))
        self.assertTrue(torch.allclose(rx, exp_x) and torch.allclose(rout, exp_out))
        spec = _get_spec(
            lambda t, u: k_alias_multi(t, u),
            (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)),
        )
        self.assertEqual(spec["mutated_inputs"], [])
        self.assertEqual(spec["output_aliases"], ["x", None])
        self.assertEqual(spec["output_alias_is_direct"], [True, False])

    def test_output_not_alias(self):
        def fn(x, y):
            out = k_no_mut(x, y)
            out.add_(1)
            return x, out

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        exp_x, exp_out = x, x + y + 1
        rx, rout = _run(fn, (x.clone(), y.clone()))
        self.assertTrue(torch.allclose(rx, exp_x) and torch.allclose(rout, exp_out))

    def test_view_internal(self):
        def fn(x):
            x = x - 1
            x = k_view(x)
            return x * 2

        x = torch.randn(64, device=DEVICE)
        expected = ((x - 1) + 1) * 2
        self.assertTrue(torch.allclose(_run(fn, (x.clone(),)), expected))
        self.assertIn(
            "x", _get_spec(fn, (torch.randn(64, device=DEVICE),))["mutated_inputs"]
        )

        # Note: View ops can prevent fusion, so we just verify correctness (not kernel count)

    def test_no_mutation(self):
        def fn(x, y):
            x, y = x * 2, y * 2
            out = k_no_mut(x, y)
            return out + 1

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        expected = (x * 2 + y * 2) + 1
        self.assertTrue(torch.allclose(_run(fn, (x.clone(), y.clone())), expected))
        spec = _get_spec(
            fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))
        )
        self.assertEqual(spec["mutated_inputs"], [])

        # Verify fusion: prologue + epilogue should fuse to 1 kernel
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        _, source_codes = run_and_get_code(compiled_fn, x.clone(), y.clone())
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

    def test_inline_triton_fallback_marks_all_inputs(self):
        spec = _get_spec(
            lambda a, b: k_inline_triton_unknown(a, b),
            (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)),
        )
        self.assertEqual(set(spec["mutated_inputs"]), {"x", "y"})

    def test_template_buffer_marks_mutations(self):
        def fn(x):
            x = x + 1
            x = k_inplace(x)
            return x * 2

        from helion._compiler._inductor.template_buffer import HelionTemplateBuffer

        captured = []
        orig_init = HelionTemplateBuffer.__init__

        def wrapper(self, *a, **k):
            orig_init(self, *a, **k)
            captured.append(self.mutated_inputs)

        with patch.object(HelionTemplateBuffer, "__init__", wrapper):
            _run(fn, (torch.randn(64, device=DEVICE),))
        self.assertTrue(
            any(captured), "Expected mutated inputs in HelionTemplateBuffer"
        )

    def test_template_buffer_mutations_multi(self):
        def fn(x, y):
            x, y = k_two_mutated(x, y)
            return x + y

        from helion._compiler._inductor.template_buffer import HelionTemplateBuffer

        captured = []
        orig_init = HelionTemplateBuffer.__init__

        def wrapper(self, *a, **k):
            orig_init(self, *a, **k)
            if self.mutated_inputs is not None:
                captured.append(
                    {buf.get_name() for buf in self.mutated_inputs}
                    == {
                        self.named_input_nodes["x"].get_name(),
                        self.named_input_nodes["y"].get_name(),
                    }
                )

        with patch.object(HelionTemplateBuffer, "__init__", wrapper):
            _run(fn, (torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)))
        self.assertTrue(any(captured), "Expected both x and y in mutated_inputs")


class TestDtypePropagation(RefEagerTestDisabled, TestCase):
    """Test dtype propagation in epilogue fusion."""

    def test_fp16_epilogue_sigmoid_fp32_bias(self):
        """Test fp16 kernel output + sigmoid + fp32 bias - requires proper dtype promotion."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        # fp32 bias - should promote result to fp32
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, bias):
            out = elementwise_2d_single_output(x, kernel_scale)  # fp16 output
            out_sigmoid = torch.sigmoid(out)  # sigmoid in epilogue
            return out_sigmoid + bias  # fp16 + fp32 -> should be fp32

        eager_result = f(x, bias)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, bias)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_epilogue_sigmoid_fp32_mul(self):
        """Test sigmoid(fp16) * fp32 - mul needs correct dtype from sigmoid."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, scale):
            out = elementwise_2d_single_output(x, kernel_scale)  # fp16
            out_sigmoid = torch.sigmoid(out)  # fp16, needs dtype tracking
            return out_sigmoid * scale  # fp16 * fp32 -> fp32

        eager_result = f(x, scale)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_epilogue_chained_ops_fp32_mul(self):
        """Test chained ops (sigmoid, relu, tanh) then fp32 mul."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, scale):
            out = elementwise_2d_single_output(x, kernel_scale)  # fp16
            out = torch.sigmoid(out)  # no dtype tracking
            out = torch.relu(out)  # no dtype tracking
            out = torch.tanh(out)  # no dtype tracking
            return out * scale  # fp16 * fp32 -> fp32

        eager_result = f(x, scale)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_epilogue_where_fp32_bias(self):
        """Test where(fp16, fp32) promotes to fp32."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, bias):
            out = elementwise_2d_single_output(x, kernel_scale)
            mask = out > 0
            return torch.where(mask, out, bias)

        eager_result = f(x, bias)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, bias)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_epilogue_maximum_fp32_tensor(self):
        """Test maximum(fp16, fp32) promotes to fp32."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        min_val = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, min_val):
            out = elementwise_2d_single_output(x, kernel_scale)
            return torch.maximum(out, min_val)

        eager_result = f(x, min_val)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, min_val)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_epilogue_cast_fp32_then_fp16_mul(self):
        """Test explicit fp32 cast then fp16 mul keeps fp32 output."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float16)

        def f(x, scale):
            out = elementwise_2d_single_output(x, kernel_scale)
            out = out.float()
            out = torch.sigmoid(out)
            return out * scale

        eager_result = f(x, scale)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_fp32_scale_bias(self):
        """Test prologue mixed-dtype ops preserve fp32 in fused Helion kernel."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, scale, bias):
            x = torch.sigmoid(x * scale + bias)
            return elementwise_2d_single_output(x, kernel_scale)

        eager_result = f(x, scale, bias)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale, bias)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_where_fp32_bias(self):
        """Test where in prologue promotes to fp32."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, bias):
            mask = x > 0
            x = torch.where(mask, x, bias)
            return elementwise_2d_single_output(x, kernel_scale)

        eager_result = f(x, bias)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, bias)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_maximum_fp32_tensor(self):
        """Test maximum in prologue promotes to fp32."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        min_val = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, min_val):
            x = torch.maximum(x, min_val)
            return elementwise_2d_single_output(x, kernel_scale)

        eager_result = f(x, min_val)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, min_val)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_explicit_cast_chain(self):
        """Test explicit fp32 cast in prologue survives chained ops."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)

        def f(x):
            x = x.float()
            x = torch.sigmoid(x)
            x = torch.relu(x)
            x = torch.tanh(x)
            return elementwise_2d_single_output(x, kernel_scale)

        eager_result = f(x)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_addcmul_fp32_bias(self):
        """Test addcmul in prologue promotes to fp32."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x, bias, scale):
            x = torch.addcmul(bias, x, scale)
            return elementwise_2d_single_output(x, kernel_scale)

        eager_result = f(x, bias, scale)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, bias, scale)

        # Complex prologue (addcmul with 3 inputs) doesn't fuse
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreater(kernel_count, 1, f"Expected >1 kernels (no fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_view_fp32_scale_bias(self):
        """Test view + mixed-dtype prologue promotes to fp32."""
        m, n = 64, 128
        kernel_scale = 1.5

        x_flat = torch.randn(m * n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)

        def f(x_flat, scale, bias):
            x = x_flat.view(m, n)
            x = x * scale + bias
            return elementwise_2d_single_output(x, kernel_scale)

        eager_result = f(x_flat, scale, bias)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x_flat, scale, bias)

        # View ops prevent fusion, so we expect >1 kernels
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreater(kernel_count, 1, f"View should prevent fusion, expected >1 kernels, got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    # Tests for explicit dtype=x.dtype pattern (vs empty_like)
    def test_fp16_prologue_explicit_dtype_fp32_scale(self):
        """Test prologue dtype change with kernel using explicit dtype=x.dtype."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, scale):
            x = x * scale  # fp16 * fp32 -> fp32
            return elementwise_2d_explicit_dtype(x, kernel_scale)

        eager_result = f(x, scale)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_prologue_explicit_dtype_float_cast(self):
        """Test explicit .float() cast in prologue with explicit dtype=x.dtype kernel."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)

        def f(x):
            x = x.float()  # fp16 -> fp32
            return elementwise_2d_explicit_dtype(x, kernel_scale)

        eager_result = f(x)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)

    def test_fp16_epilogue_explicit_dtype_fp32_mul(self):
        """Test epilogue dtype change with kernel using explicit dtype=x.dtype."""
        m, n = 64, 128
        kernel_scale = 1.5

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, scale):
            y = elementwise_2d_explicit_dtype(x, kernel_scale)
            return y * scale  # fp16 * fp32 -> fp32

        eager_result = f(x, scale)
        self.assertEqual(eager_result.dtype, torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}")

        self.assertEqual(
            result.dtype,
            eager_result.dtype,
            f"Dtype mismatch: eager={eager_result.dtype}, compiled={result.dtype}",
        )
        torch.testing.assert_close(result, eager_result, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
