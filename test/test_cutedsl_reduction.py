from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import TestCase
from helion.runtime.config import Config


def _has_cutlass() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


skipIfNoCutlass = unittest.skipUnless(_has_cutlass(), "cutlass not installed")


def add_combine_fn(x, y):
    return x + y


def max_combine_fn(x, y):
    return torch.maximum(x, y)


def min_combine_fn(x, y):
    return torch.minimum(x, y)


def mul_combine_fn(x, y):
    return x * y


class TestCuteDSLReductionCodegen(TestCase):
    """Tests that CuteDSL backend generates valid code for reduction operations."""

    def _get_code(self, fn, args, **config_kwargs):
        """Helper to get generated code for a CuteDSL kernel."""
        bound = fn.bind(args)
        if config_kwargs:
            config = Config(**config_kwargs)
        else:
            config = bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_reduce_sum_generates_code(self):
        """Test that sum reduction generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def reduce_sum_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_sum_kernel, args)

        # Verify CuteDSL-specific markers
        self.assertIn("@cute.kernel", code)
        self.assertIn("import cutlass", code)
        self.assertIn("import cutlass.cute as cute", code)
        self.assertIn("_default_cutedsl_launcher", code)
        # Verify no triton decorator
        self.assertNotIn("@triton.jit", code)

    def test_reduce_sum_generates_parseable_code(self):
        """Verify generated reduction code is valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl")
        def reduce_sum_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_sum_kernel, args)
        py_ast.parse(code)

    def test_reduce_max_generates_code(self):
        """Test that max reduction generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def reduce_max_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(max_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_max_kernel, args)

        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.reduce", code)

    def test_reduce_min_generates_code(self):
        """Test that min reduction generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def reduce_min_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(min_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_min_kernel, args)

        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.reduce", code)

    def test_reduce_generates_combine_function(self):
        """Verify reduction code includes the combine function helper."""

        @helion.kernel(backend="cutedsl")
        def reduce_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_kernel, args)

        # The combine function should be emitted as a helper
        self.assertIn("add_combine_fn", code)

    def test_reduce_uses_cutlass_types(self):
        """Verify reduction code uses cutlass types for indices."""

        @helion.kernel(backend="cutedsl")
        def reduce_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_kernel, args)

        self.assertIn("cutlass.Int32", code)
        self.assertIn("cutlass.Constexpr[int]", code)

    def test_softmax_pattern_generates_code(self):
        """Test softmax-like pattern (max + exp + sum) generates valid code."""

        @helion.kernel(backend="cutedsl")
        def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                row = x[i, :]
                row_max = hl.reduce(max_combine_fn, row, dim=1, keep_dims=True)
                row_exp = torch.exp(row - row_max)
                row_sum = hl.reduce(add_combine_fn, row_exp, dim=1, keep_dims=True)
                result[i, :] = row_exp / row_sum
            return result

        args = (torch.randn([16, 32], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(softmax_kernel, args)

        self.assertIn("@cute.kernel", code)
        # Should contain two reduce calls (max and sum)
        self.assertIn("tl.reduce", code)

    def test_reduce_product_generates_code(self):
        """Test that product reduction generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def reduce_prod_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(mul_combine_fn, row_data, dim=1)
            return result

        args = (torch.randn([32, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(reduce_prod_kernel, args)

        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.reduce", code)
        self.assertIn("mul_combine_fn", code)


class TestCuteDSLReductionUtils(TestCase):
    """Tests for the CuteDSL reduction utility functions."""

    def test_get_warp_reduce_op_sum(self):
        from helion._compiler.cutedsl_reduction import get_warp_reduce_op

        template = get_warp_reduce_op("sum")
        result = template.format(a="x", b="y")
        self.assertEqual(result, "x + y")

    def test_get_warp_reduce_op_max(self):
        from helion._compiler.cutedsl_reduction import get_warp_reduce_op

        template = get_warp_reduce_op("max")
        result = template.format(a="x", b="y")
        self.assertEqual(result, "max(x, y)")

    def test_get_warp_reduce_op_min(self):
        from helion._compiler.cutedsl_reduction import get_warp_reduce_op

        template = get_warp_reduce_op("min")
        result = template.format(a="x", b="y")
        self.assertEqual(result, "min(x, y)")

    def test_get_warp_reduce_op_prod(self):
        from helion._compiler.cutedsl_reduction import get_warp_reduce_op

        template = get_warp_reduce_op("prod")
        result = template.format(a="x", b="y")
        self.assertEqual(result, "x * y")

    def test_get_warp_reduce_op_invalid(self):
        from helion._compiler.cutedsl_reduction import get_warp_reduce_op

        with self.assertRaises(ValueError, msg="Unsupported reduction op"):
            get_warp_reduce_op("invalid_op")

    def test_compute_reduction_smem_size(self):
        from helion._compiler.cutedsl_reduction import compute_reduction_smem_size

        # 4 warps, 4 bytes (float32), 1 element per warp
        self.assertEqual(compute_reduction_smem_size(4, 4, 1), 16)
        # 8 warps, 2 bytes (float16), 4 elements per warp
        self.assertEqual(compute_reduction_smem_size(8, 2, 4), 64)

    def test_get_reduction_identity(self):
        from helion._compiler.cutedsl_reduction import get_reduction_identity

        self.assertEqual(get_reduction_identity("sum", "cutlass.Float32"), "0.0")
        self.assertEqual(
            get_reduction_identity("max", "cutlass.Float32"), "float('-inf')"
        )
        self.assertEqual(
            get_reduction_identity("min", "cutlass.Float32"), "float('inf')"
        )
        self.assertEqual(get_reduction_identity("prod", "cutlass.Float32"), "1.0")

    def test_get_reduction_identity_invalid(self):
        from helion._compiler.cutedsl_reduction import get_reduction_identity

        with self.assertRaises(ValueError):
            get_reduction_identity("invalid_op", "cutlass.Float32")

    def test_num_shuffle_rounds(self):
        from helion._compiler.cutedsl_reduction import num_shuffle_rounds

        self.assertEqual(num_shuffle_rounds(1), 0)
        self.assertEqual(num_shuffle_rounds(2), 1)
        self.assertEqual(num_shuffle_rounds(4), 2)
        self.assertEqual(num_shuffle_rounds(8), 3)
        self.assertEqual(num_shuffle_rounds(16), 4)
        self.assertEqual(num_shuffle_rounds(32), 5)

    def test_num_shuffle_rounds_non_power_of_2(self):
        from helion._compiler.cutedsl_reduction import num_shuffle_rounds

        with self.assertRaises(AssertionError):
            num_shuffle_rounds(3)

    def test_emit_warp_reduce_snippet_sum(self):
        from helion._compiler.cutedsl_reduction import emit_warp_reduce_snippet

        lines = emit_warp_reduce_snippet("v", "sum", 4)
        self.assertEqual(len(lines), 2)  # log2(4) = 2 rounds
        self.assertIn("shuffle_sync_bfly", lines[0])
        self.assertIn("offset=1", lines[0])
        self.assertIn("offset=2", lines[1])

    def test_emit_warp_reduce_snippet_max(self):
        from helion._compiler.cutedsl_reduction import emit_warp_reduce_snippet

        lines = emit_warp_reduce_snippet("val", "max", 8)
        self.assertEqual(len(lines), 3)  # log2(8) = 3 rounds
        self.assertIn("max(val,", lines[0])

    def test_emit_block_reduce_snippet(self):
        from helion._compiler.cutedsl_reduction import emit_block_reduce_snippet

        lines = emit_block_reduce_snippet("v", "sum", num_warps=4)
        # Should contain warp reduction, smem write, barrier, final reduction
        combined = "\n".join(lines)
        self.assertIn("shuffle_sync_bfly", combined)
        self.assertIn("smem_reduce", combined)
        self.assertIn("syncthreads", combined)
        self.assertIn("warp_id", combined)
        self.assertIn("lane_id", combined)

    def test_get_cutedsl_reduce_fn_name(self):
        from helion._compiler.cutedsl_reduction import get_cutedsl_reduce_fn_name

        self.assertEqual(get_cutedsl_reduce_fn_name("sum"), "fadd_reduce")
        self.assertEqual(get_cutedsl_reduce_fn_name("max"), "fmax_reduce")
        self.assertEqual(get_cutedsl_reduce_fn_name("min"), "fmin_reduce")


if __name__ == "__main__":
    unittest.main()
