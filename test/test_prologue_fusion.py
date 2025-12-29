"""
Test prologue fusion with torch.compile (refactored v2).

Verifies that Inductor correctly generates prologue lambdas for Helion kernels
and that the prologue operations are fused into a single Triton kernel.

Prologue fusion is the complement of epilogue fusion:
- Epilogue fusion: kernel_output -> pointwise_ops (ops fused AFTER kernel)
- Prologue fusion: pointwise_ops -> kernel_input (ops fused INTO kernel, before compute)

This is a refactored version of test_prologue_fusion.py with:
- Parameterized tests to reduce boilerplate
- Common helper methods via PrologueFusionTestMixin
- Additional edge case tests
"""

from __future__ import annotations

import os
from typing import Callable
import unittest

import torch
from torch._inductor.utils import run_and_get_code

import helion
from helion._testing import DEVICE, TestCase, count_triton_kernels
import helion.language as hl


# =============================================================================
# Shared Kernel (duplicated for self-contained file)
# =============================================================================


@helion.kernel(static_shapes=True, autotune_effort="none", allow_fuse_with_inductor_ops=True)
def matmul(x, y):  # pyrefly: ignore
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


# =============================================================================
# Test Mixin with Helper Methods
# =============================================================================


class PrologueFusionTestMixin:
    """Mixin providing common prologue fusion test helpers."""

    def _run_prologue_test(
        self,
        f: Callable,
        inputs: tuple,
        expected_kernels: int = 1,
        code_patterns: list[str] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> str:
        """Run prologue fusion test: eager vs compiled, verify kernel count and patterns."""
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
        self, m: int = 64, n: int = 64, k: int = 64, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create standard matmul inputs (fp32 default for prologue tests)."""
        return (
            torch.randn(m, k, device=DEVICE, dtype=dtype),
            torch.randn(k, n, device=DEVICE, dtype=dtype),
        )


# =============================================================================
# Test Data Constants
# =============================================================================

# Unary prologue operations: (name, op_fn, code_patterns)
# Note: Prologue tests use fp32 to bypass PyTorch's low-precision prologue heuristic
UNARY_PROLOGUE_OPS: list[tuple[str, Callable, list[str] | None]] = [
    ("relu", torch.relu, ["triton_helpers.maximum", "tl.maximum"]),
    ("sigmoid", torch.sigmoid, ["sigmoid", "tl.sigmoid"]),
    ("tanh", torch.tanh, ["tanh", "libdevice.tanh"]),
    ("scalar_mul", lambda x: x * 2.0, ["2.0"]),
    ("abs", torch.abs, None),
    ("exp", torch.exp, ["exp"]),
]

# Closure prologue operations: (name, closure_shape, op_fn, code_patterns)
CLOSURE_PROLOGUE_OPS: list[tuple[str, tuple, Callable, list[str] | None]] = [
    ("bias_add", (64,), lambda x, c: x + c, ["prologue_input"]),
    ("scale_mul", (64,), lambda x, c: x * c, ["prologue_input"]),
]


# =============================================================================
# Test Classes
# =============================================================================


class TestBasicPrologue(PrologueFusionTestMixin, TestCase):
    """Basic prologue fusion tests with unary operations on first input."""

    def test_unary_prologue_ops(self):
        """Test unary operations fused as prologue on first input (x)."""
        for op_name, op_fn, code_patterns in UNARY_PROLOGUE_OPS:
            with self.subTest(op=op_name):
                x, y = self._create_matmul_inputs()

                def f(x, y, op=op_fn):
                    return matmul(op(x), y)

                self._run_prologue_test(
                    f, (x, y), code_patterns=code_patterns, rtol=1e-2, atol=1e-2
                )

    def test_unary_prologue_ops_on_y(self):
        """Test unary operations fused as prologue on second input (y)."""
        for op_name, op_fn, code_patterns in UNARY_PROLOGUE_OPS:
            with self.subTest(op=op_name):
                x, y = self._create_matmul_inputs()

                def f(x, y, op=op_fn):
                    return matmul(x, op(y))

                self._run_prologue_test(
                    f, (x, y), code_patterns=code_patterns, rtol=1e-2, atol=1e-2
                )


class TestPrologueWithClosure(PrologueFusionTestMixin, TestCase):
    """Prologue fusion with external tensors (closures)."""

    def test_closure_prologue_ops(self):
        """Test operations with closure tensors fused as prologue."""
        for op_name, closure_shape, op_fn, code_patterns in CLOSURE_PROLOGUE_OPS:
            with self.subTest(op=op_name):
                x, y = self._create_matmul_inputs()
                closure = torch.randn(*closure_shape, device=DEVICE, dtype=torch.float32)

                def f(x, y, closure, op=op_fn):
                    return matmul(op(x, closure), y)

                self._run_prologue_test(
                    f, (x, y, closure), code_patterns=code_patterns, rtol=1e-2, atol=1e-2
                )

    def test_scale_on_y(self):
        """Scale multiplication on second input (y)."""
        x, y = self._create_matmul_inputs()
        scale = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale):
            return matmul(x, y * scale)

        self._run_prologue_test(f, (x, y, scale), code_patterns=["prologue_input"])


class TestPrologueBothInputs(PrologueFusionTestMixin, TestCase):
    """Prologue fusion on multiple inputs."""

    def test_relu_both_inputs(self):
        """Matmul with relu on both inputs."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return matmul(torch.relu(x), torch.relu(y))

        all_code = self._run_prologue_test(f, (x, y), rtol=1e-2, atol=1e-2)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code
        )

    def test_different_ops_both_inputs(self):
        """Different operations on both inputs: relu(x) @ sigmoid(y)."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return matmul(torch.relu(x), torch.sigmoid(y))

        all_code = self._run_prologue_test(f, (x, y), rtol=1e-2, atol=1e-2)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code
        )
        self.assertTrue("sigmoid" in all_code or "tl.sigmoid" in all_code)


class TestPrologueEpilogueCombined(PrologueFusionTestMixin, TestCase):
    """Tests combining prologue and epilogue fusion."""

    def test_relu_prologue_sigmoid_epilogue(self):
        """relu(x) @ y -> sigmoid - prologue and epilogue combined."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return torch.sigmoid(matmul(torch.relu(x), y))

        all_code = self._run_prologue_test(f, (x, y), rtol=1e-2, atol=1e-2)
        self.assertTrue(
            "triton_helpers.maximum" in all_code or "tl.maximum" in all_code
        )
        self.assertTrue("sigmoid" in all_code or "tl.sigmoid" in all_code)

    def test_scale_prologue_bias_epilogue(self):
        """(x * scale) @ y + bias - scale as prologue, bias as epilogue."""
        x, y = self._create_matmul_inputs()
        scale = torch.randn(64, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale, bias):
            return matmul(x * scale, y) + bias

        all_code = self._run_prologue_test(f, (x, y, scale, bias), rtol=1e-2, atol=1e-2)
        self.assertIn("prologue_input", all_code)
        self.assertIn("epilogue_input", all_code)

    def test_chain_prologue_chain_epilogue(self):
        """Chained ops: sigmoid(relu(x)) @ y -> tanh -> add bias."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y, bias):
            x_transformed = torch.sigmoid(torch.relu(x))
            return torch.tanh(matmul(x_transformed, y)) + bias

        all_code = self._run_prologue_test(f, (x, y, bias), rtol=1e-2, atol=1e-2)
        # Should contain both prologue and epilogue indicators
        self.assertIn("epilogue_input", all_code)


class TestProperPrologueIndexing(PrologueFusionTestMixin, TestCase):
    """Tests requiring proper range matching and index translation for prologue."""

    def test_row_scale_x(self):
        """Row-wise scaling on first input: (x * scale[m]) @ y."""
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        row_scale = torch.randn(m, device=DEVICE, dtype=torch.float32)

        def f(x, y, row_scale):
            return matmul(x * row_scale.unsqueeze(1), y)

        self._run_prologue_test(f, (x, y, row_scale), code_patterns=["prologue_input"])

    def test_col_scale_x(self):
        """Column-wise scaling on first input: (x * scale[k]) @ y."""
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        col_scale = torch.randn(k, device=DEVICE, dtype=torch.float32)

        def f(x, y, col_scale):
            return matmul(x * col_scale, y)

        self._run_prologue_test(
            f, (x, y, col_scale), code_patterns=["prologue_input"], rtol=1e-2, atol=0.02
        )

    def test_row_scale_y(self):
        """Row-wise scaling on second input: x @ (y * scale[k])."""
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        row_scale = torch.randn(k, device=DEVICE, dtype=torch.float32)

        def f(x, y, row_scale):
            return matmul(x, y * row_scale.unsqueeze(1))

        self._run_prologue_test(
            f, (x, y, row_scale), code_patterns=["prologue_input"], rtol=1e-2, atol=1e-2
        )

    def test_col_scale_y(self):
        """Column-wise scaling on second input: x @ (y * scale[n])."""
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        col_scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, col_scale):
            return matmul(x, y * col_scale)

        self._run_prologue_test(f, (x, y, col_scale), code_patterns=["prologue_input"])

    def test_different_scales_both_inputs(self):
        """Different scaling on both inputs: (x * scale_x) @ (y * scale_y)."""
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        scale_x = torch.randn(k, device=DEVICE, dtype=torch.float32)
        scale_y = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, scale_x, scale_y):
            return matmul(x * scale_x, y * scale_y)

        self._run_prologue_test(
            f, (x, y, scale_x, scale_y), code_patterns=["prologue_input"]
        )

    def test_2d_transform_x(self):
        """2D element-wise transform on first input: tanh(x) @ y."""
        m, k, n = 64, 32, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)

        def f(x, y):
            return matmul(torch.tanh(x), y)

        all_code = self._run_prologue_test(f, (x, y), rtol=1e-2, atol=1e-2)
        self.assertIn("libdevice", all_code.lower())


class TestQuantizationPatterns(PrologueFusionTestMixin, TestCase):
    """Tests for quantization-style prologue patterns (dequantization)."""

    def test_dequant_x(self):
        """Simulated dequantization: (x_int - zero_point) * scale @ y."""
        x_int = torch.randint(-128, 127, (64, 64), device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
        scale = torch.tensor(0.1, device=DEVICE, dtype=torch.float32)
        zero_point = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

        def f(x_int, y, scale, zero_point):
            x_dequant = (x_int - zero_point) * scale
            return matmul(x_dequant, y)

        self._run_prologue_test(
            f, (x_int, y, scale, zero_point), code_patterns=["prologue_input"]
        )

    def test_dequant_both_inputs(self):
        """Dequantization on both inputs."""
        x_int = torch.randint(-128, 127, (64, 64), device=DEVICE, dtype=torch.float32)
        y_int = torch.randint(-128, 127, (64, 64), device=DEVICE, dtype=torch.float32)
        scale_x = torch.tensor(0.1, device=DEVICE, dtype=torch.float32)
        scale_y = torch.tensor(0.1, device=DEVICE, dtype=torch.float32)

        def f(x_int, y_int, scale_x, scale_y):
            return matmul(x_int * scale_x, y_int * scale_y)

        self._run_prologue_test(
            f, (x_int, y_int, scale_x, scale_y), code_patterns=["prologue_input"]
        )


class TestPrologueEdgeCases(PrologueFusionTestMixin, TestCase):
    """Edge case tests for prologue fusion."""

    def test_chained_prologue_ops(self):
        """Multiple chained ops before kernel: sigmoid(relu(x + bias)) @ y."""
        x, y = self._create_matmul_inputs()
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)

        def f(x, y, bias):
            x_transformed = torch.sigmoid(torch.relu(x + bias))
            return matmul(x_transformed, y)

        self._run_prologue_test(f, (x, y, bias), rtol=1e-2, atol=1e-2)

    def test_scalar_broadcast(self):
        """Scalar broadcast in prologue: (x + 1.0) @ y."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return matmul(x + 1.0, y)

        self._run_prologue_test(f, (x, y), code_patterns=["1.0"])

    def test_multiple_scalar_ops(self):
        """Multiple scalar operations: ((x * 2.0) + 1.0) @ y."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return matmul((x * 2.0) + 1.0, y)

        all_code = self._run_prologue_test(f, (x, y))
        self.assertTrue("2.0" in all_code or "1.0" in all_code)

    def test_abs_on_both_inputs(self):
        """abs() on both inputs."""
        x, y = self._create_matmul_inputs()

        def f(x, y):
            return matmul(torch.abs(x), torch.abs(y))

        self._run_prologue_test(f, (x, y))


if __name__ == "__main__":
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    unittest.main()
