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


class TestCuteDSLCodegen(TestCase):
    """Tests that CuteDSL backend generates valid Python code for element-wise ops."""

    def _get_code(self, fn, args, **config_kwargs):
        """Helper to get generated code for a CuteDSL kernel."""
        bound = fn.bind(args)
        if config_kwargs:
            config = Config(**config_kwargs)
        else:
            config = bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_add_generates_code(self):
        @helion.kernel(backend="cutedsl")
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(add_kernel, args)
        # Verify CuteDSL-specific markers are present
        self.assertIn("@cute.kernel", code)
        self.assertIn("cutlass.Constexpr[int]", code)
        self.assertIn("import cutlass", code)
        self.assertIn("import cutlass.cute as cute", code)
        self.assertIn("_default_cutedsl_launcher", code)
        # Verify no triton-specific imports
        self.assertNotIn("import triton\n", code)
        self.assertNotIn("import triton.language", code)

    def test_add_generates_parseable_code(self):
        """Verify generated code is valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl")
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(add_kernel, args)
        # Should parse without errors
        py_ast.parse(code)

    def test_mul_generates_code(self):
        @helion.kernel(backend="cutedsl")
        def mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * y[tile]
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(mul_kernel, args)
        self.assertIn("@cute.kernel", code)

    def test_pointwise_chain_generates_code(self):
        @helion.kernel(backend="cutedsl")
        def chain_kernel(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(a.size()):
                out[tile] = a[tile] * torch.exp(b[tile]) + c[tile]
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(chain_kernel, args)
        self.assertIn("@cute.kernel", code)

    def test_codegen_uses_cutlass_index_type(self):
        @helion.kernel(backend="cutedsl")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        args = (torch.randn([64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(kernel, args)
        # CuteDSL uses cutlass.Constexpr[int] for constexpr integer types
        # and does not emit explicit index type casts like cutlass.Int32
        self.assertIn("cutlass.Constexpr[int]", code)
        # Should NOT have Triton index types
        self.assertNotIn("tl.int32", code)

    def test_codegen_decorator_is_cute_kernel(self):
        @helion.kernel(backend="cutedsl")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0
            return out

        args = (torch.randn([64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(kernel, args)
        self.assertIn("@cute.kernel", code)

    def test_full_generates_code(self):
        @helion.kernel(backend="cutedsl")
        def zeros_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile, :] = hl.zeros([tile, x.size(1)])
            return out

        args = (torch.randn([64, 64], device=DEVICE, dtype=torch.float32),)
        code = self._get_code(zeros_kernel, args)
        self.assertIn("@cute.kernel", code)
        # CuteDSL replaces tl.full with scalar values in per-thread model
        self.assertNotIn("tl.full", code)
        self.assertIn("= 0.0", code)


class TestCuteDSLCodegenRegistration(TestCase):
    """Tests that CuteDSL codegen functions are properly registered."""

    def test_all_api_funcs_have_cutedsl_codegen(self):
        """Verify all essential API functions have cutedsl codegen registered."""
        from helion._compiler.backend import CuteDSLBackend

        # Trigger registration
        CuteDSLBackend()

        from helion.language._tracing_ops import (
            _and,
            _constant_tensor,
            _for_loop,
            _get_symnode,
            _host_tensor,
            _if,
            _mask_to,
            _new_var,
            _not,
            _or,
            _phi,
            _while_loop,
        )
        from helion.language.barrier import barrier
        from helion.language.constexpr import specialize
        from helion.language.creation_ops import full
        from helion.language.loops import grid, tile
        from helion.language.matmul_ops import dot
        from helion.language.memory_ops import load, store
        from helion.language.reduce_ops import _reduce
        from helion.language.tile_ops import (
            tile_begin,
            tile_count,
            tile_end,
            tile_id,
            tile_index,
        )
        from helion.language.tunable_ops import register_block_size, register_tunable
        from helion.language.view_ops import join, split, subscript

        api_funcs = [
            _get_symnode, _host_tensor, _constant_tensor, _for_loop, _while_loop,
            _if, _phi, _and, _or, _not, _mask_to, _new_var,
            tile, grid, store, load,
            subscript, split, join,
            tile_index, tile_begin, tile_end, tile_count, tile_id,
            specialize, register_block_size, register_tunable,
            full, _reduce, dot, barrier,
        ]

        missing = [
            fn.__qualname__
            for fn in api_funcs
            if "cutedsl" not in fn._codegen
        ]
        self.assertEqual(missing, [], f"Missing cutedsl codegen for: {missing}")


if __name__ == "__main__":
    unittest.main()
