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


class TestCuteDSLMatmulCodegen(TestCase):
    """Tests that CuteDSL backend generates valid code for matmul/dot operations."""

    def _get_code(self, fn, args, **config_kwargs):
        """Helper to get generated code for a CuteDSL kernel."""
        bound = fn.bind(args)
        if config_kwargs:
            config = Config(**config_kwargs)
        else:
            config = bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_matmul_generates_code(self):
        """Test that matmul kernel generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(matmul_kernel, args, block_sizes=[16, 16, 16])

        # Verify CuteDSL-specific markers
        self.assertIn("@cute.kernel", code)
        self.assertIn("import cutlass", code)
        self.assertIn("import cutlass.cute as cute", code)
        self.assertIn("_default_cutedsl_launcher", code)
        # Verify no triton decorator
        self.assertNotIn("@triton.jit", code)

    def test_matmul_generates_parseable_code(self):
        """Verify generated matmul code is valid Python."""
        import ast as py_ast

        @helion.kernel(backend="cutedsl")
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(matmul_kernel, args, block_sizes=[16, 16, 16])
        py_ast.parse(code)

    def test_matmul_contains_dot_call(self):
        """Verify generated code contains tl.dot for matrix multiply."""

        @helion.kernel(backend="cutedsl")
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(matmul_kernel, args, block_sizes=[16, 16, 16])
        self.assertIn("tl.dot(", code)

    def test_matmul_with_addmm_generates_code(self):
        """Test matmul with addmm pattern generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def matmul_addmm_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(matmul_addmm_kernel, args, block_sizes=[16, 16, 16])
        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.dot(", code)

    def test_matmul_f16_generates_code(self):
        """Test fp16 matmul generates code with appropriate types."""

        @helion.kernel(backend="cutedsl")
        def matmul_f16_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float16),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float16),
        )
        code = self._get_code(matmul_f16_kernel, args, block_sizes=[16, 16, 16])
        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.dot(", code)

    def test_matmul_uses_cutlass_constexpr(self):
        """Verify matmul code uses cutlass.Constexpr[int] for block sizes."""

        @helion.kernel(backend="cutedsl")
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(matmul_kernel, args, block_sizes=[16, 16, 16])
        self.assertIn("cutlass.Constexpr[int]", code)

    def test_hl_dot_generates_code(self):
        """Test hl.dot generates valid CuteDSL code."""

        @helion.kernel(backend="cutedsl")
        def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc)
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(dot_kernel, args, block_sizes=[16, 16, 16])
        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.dot(", code)

    def test_matmul_larger_sizes_generates_code(self):
        """Test matmul with larger tile sizes generates valid code."""

        @helion.kernel(backend="cutedsl")
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(matmul_kernel, args, block_sizes=[32, 32, 32])
        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.dot(", code)


class TestCuteDSLMMAUtils(TestCase):
    """Tests for the CuteDSL MMA utility functions."""

    def test_get_swizzle_params_128_aligned(self):
        from helion._compiler.cutedsl_mma_utils import get_swizzle_params

        bits, base, k_block = get_swizzle_params(k_dim=64, dtype_bytes=2)
        # 64 * 2 = 128 bytes per row, so 128-byte swizzle
        self.assertEqual(bits, 4)
        self.assertEqual(k_block, 64)  # 128 bytes / 2 bytes = 64 elements

    def test_get_swizzle_params_64_aligned(self):
        from helion._compiler.cutedsl_mma_utils import get_swizzle_params

        bits, base, k_block = get_swizzle_params(k_dim=16, dtype_bytes=4)
        # 16 * 4 = 64 bytes per row
        self.assertEqual(bits, 3)
        self.assertEqual(k_block, 16)  # 64 bytes / 4 bytes = 16 elements

    def test_get_swizzle_params_32_aligned(self):
        from helion._compiler.cutedsl_mma_utils import get_swizzle_params

        bits, base, k_block = get_swizzle_params(k_dim=16, dtype_bytes=2)
        # 16 * 2 = 32 bytes per row
        self.assertEqual(bits, 2)
        self.assertEqual(k_block, 16)  # 32 bytes / 2 bytes = 16 elements

    def test_emit_smem_layout_atom(self):
        from helion._compiler.cutedsl_mma_utils import emit_smem_layout_atom

        code = emit_smem_layout_atom(k_dim=64, dtype_bytes=2)
        self.assertIn("make_composed_layout", code)
        self.assertIn("make_swizzle", code)
        self.assertIn("make_ordered_layout", code)

    def test_get_mma_shape_ampere_fp16(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape

        m, n, k = get_mma_shape(dtype_bytes=2, arch=80)
        self.assertEqual((m, n, k), (16, 8, 16))

    def test_get_mma_shape_ampere_fp32(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape

        m, n, k = get_mma_shape(dtype_bytes=4, arch=80)
        self.assertEqual((m, n, k), (16, 8, 8))

    def test_get_mma_shape_hopper_fp16(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape

        m, n, k = get_mma_shape(dtype_bytes=2, arch=90)
        self.assertEqual((m, n, k), (64, 16, 16))

    def test_emit_tiled_mma_ampere(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma

        code = emit_tiled_mma(
            "cutlass.Float16", "cutlass.Float32", num_warps=4, arch=80
        )
        self.assertIn("make_tiled_mma", code)
        self.assertIn("MmaF16BF16Op", code)

    def test_emit_tiled_mma_hopper(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma

        code = emit_tiled_mma(
            "cutlass.Float16", "cutlass.Float32", num_warps=4, arch=90
        )
        self.assertIn("make_trivial_tiled_mma", code)
        self.assertIn("warpgroup", code)

    def test_emit_gmem_copy_atom_async(self):
        from helion._compiler.cutedsl_mma_utils import emit_gmem_copy_atom

        code = emit_gmem_copy_atom("cutlass.Float16", dtype_bits=16, use_async=True)
        self.assertIn("CopyG2SOp", code)
        self.assertIn("num_bits_per_copy=128", code)

    def test_emit_gmem_copy_atom_sync(self):
        from helion._compiler.cutedsl_mma_utils import emit_gmem_copy_atom

        code = emit_gmem_copy_atom("cutlass.Float16", dtype_bits=16, use_async=False)
        self.assertIn("CopyUniversalOp", code)

    def test_emit_smem_copy_atom_ampere(self):
        from helion._compiler.cutedsl_mma_utils import emit_smem_copy_atom

        code = emit_smem_copy_atom("cutlass.Float16", dtype_bits=16, arch=80)
        self.assertIn("CopyUniversalOp", code)

    def test_emit_smem_copy_atom_hopper(self):
        from helion._compiler.cutedsl_mma_utils import emit_smem_copy_atom

        code = emit_smem_copy_atom("cutlass.Float16", dtype_bits=16, arch=90)
        self.assertIn("StMatrix8x8x16bOp", code)

    def test_compute_smem_size(self):
        from helion._compiler.cutedsl_mma_utils import compute_smem_size

        # tile_m=64, tile_n=64, tile_k=32, fp16 (2 bytes), 2 stages
        size = compute_smem_size(64, 64, 32, dtype_bytes=2, num_stages=2)
        a_size = 64 * 32 * 2  # 4096
        b_size = 32 * 64 * 2  # 4096
        expected = (a_size + b_size) * 2
        self.assertEqual(size, expected)

    def test_emit_tiled_copy_2d(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_copy_2d

        code = emit_tiled_copy_2d(
            "cutlass.Float16",
            dtype_bits=16,
            major_mode_size=64,
            num_threads=128,
        )
        self.assertIn("make_tiled_copy_tv", code)
        self.assertIn("make_ordered_layout", code)


class TestCuteDSLGemmCodegen(TestCase):
    """Tests for the CuteDSL GEMM codegen utility functions."""

    def test_emit_gemm_smem_alloc(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_smem_alloc

        lines = emit_gemm_smem_alloc(
            tile_m=64, tile_n=64, tile_k=32,
            dtype_str="cutlass.Float16", dtype_bytes=2, num_stages=2,
        )
        combined = "\n".join(lines)
        self.assertIn("Shared memory allocation", combined)
        self.assertIn("sA_layout_atom", combined)
        self.assertIn("sB_layout_atom", combined)
        self.assertIn("make_tensor", combined)
        self.assertIn("smem=True", combined)

    def test_emit_gemm_copy_setup(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_copy_setup

        lines = emit_gemm_copy_setup(
            dtype_str="cutlass.Float16", dtype_bits=16, num_threads=128, arch=80,
        )
        combined = "\n".join(lines)
        self.assertIn("gmem_copy_atom", combined)

    def test_emit_gemm_mma_setup(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mma_setup

        lines = emit_gemm_mma_setup(
            dtype_str="cutlass.Float16", acc_dtype_str="cutlass.Float32",
            num_warps=4, arch=80,
        )
        combined = "\n".join(lines)
        self.assertIn("tiled_mma", combined)

    def test_emit_gemm_preamble(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_preamble

        lines = emit_gemm_preamble(
            tile_m=64, tile_n=64, tile_k=32,
            dtype_str="cutlass.Float16", acc_dtype_str="cutlass.Float32",
            dtype_bytes=2, num_stages=2, num_warps=4, arch=80,
        )
        combined = "\n".join(lines)
        self.assertIn("sA_layout_atom", combined)
        self.assertIn("gmem_copy_atom", combined)
        self.assertIn("tiled_mma", combined)

    def test_emit_gemm_mainloop_single_stage(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mainloop

        lines = emit_gemm_mainloop(
            tile_k=32, k_extent="K", num_stages=1, arch=80,
        )
        combined = "\n".join(lines)
        self.assertIn("num_k_tiles", combined)
        self.assertIn("syncthreads", combined)

    def test_emit_gemm_mainloop_pipelined(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mainloop

        lines = emit_gemm_mainloop(
            tile_k=32, k_extent="K", num_stages=2, arch=80,
        )
        combined = "\n".join(lines)
        self.assertIn("cp_async_commit_group", combined)
        self.assertIn("cp_async_wait_group", combined)
        self.assertIn("stage", combined)

    def test_emit_gemm_mainloop_hopper(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mainloop

        lines = emit_gemm_mainloop(
            tile_k=32, k_extent="K", num_stages=2, arch=90,
        )
        combined = "\n".join(lines)
        self.assertIn("num_k_tiles", combined)
        self.assertIn("stage", combined)

    def test_emit_gemm_epilogue(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_epilogue

        lines = emit_gemm_epilogue(arch=80)
        combined = "\n".join(lines)
        self.assertIn("Epilogue", combined)
        self.assertIn("syncthreads", combined)


class TestCuteDSLArchUtils(TestCase):
    """Tests for the CuteDSL architecture utility functions."""

    def test_supports_cp_async(self):
        from helion._compiler.cutedsl_arch_utils import supports_cp_async

        self.assertTrue(supports_cp_async(80))
        self.assertTrue(supports_cp_async(90))
        self.assertTrue(supports_cp_async(100))

    def test_supports_tma(self):
        from helion._compiler.cutedsl_arch_utils import supports_tma

        self.assertFalse(supports_tma(80))
        self.assertTrue(supports_tma(90))
        self.assertTrue(supports_tma(100))

    def test_supports_tcgen05(self):
        from helion._compiler.cutedsl_arch_utils import supports_tcgen05

        self.assertFalse(supports_tcgen05(80))
        self.assertFalse(supports_tcgen05(90))
        self.assertTrue(supports_tcgen05(100))

    def test_select_copy_strategy(self):
        from helion._compiler.cutedsl_arch_utils import select_copy_strategy

        self.assertEqual(select_copy_strategy(80), "cp_async")
        self.assertEqual(select_copy_strategy(90), "tma")
        self.assertEqual(select_copy_strategy(100), "tcgen05")

    def test_select_mma_strategy(self):
        from helion._compiler.cutedsl_arch_utils import select_mma_strategy

        self.assertEqual(select_mma_strategy(80), "warp_mma")
        self.assertEqual(select_mma_strategy(90), "wgmma")
        self.assertEqual(select_mma_strategy(100), "tcgen05_mma")

    def test_get_max_smem_per_block(self):
        from helion._compiler.cutedsl_arch_utils import get_max_smem_per_block

        self.assertEqual(get_max_smem_per_block(80), 164 * 1024)
        self.assertEqual(get_max_smem_per_block(90), 228 * 1024)
        self.assertEqual(get_max_smem_per_block(100), 232 * 1024)

    def test_get_copy_bits(self):
        from helion._compiler.cutedsl_arch_utils import get_copy_bits

        # 16-bit dtype -> 128-bit copy (8 elements)
        self.assertEqual(get_copy_bits(16), 128)
        # 32-bit dtype -> 128-bit copy (4 elements)
        self.assertEqual(get_copy_bits(32), 128)
        # 8-bit dtype -> 128-bit copy (16 elements)
        self.assertEqual(get_copy_bits(8), 128)

    def test_get_copy_bits_invalid(self):
        from helion._compiler.cutedsl_arch_utils import get_copy_bits

        with self.assertRaises(ValueError):
            get_copy_bits(0)

    def test_get_cuda_arch(self):
        """Test that get_cuda_arch returns a reasonable SM version."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        from helion._compiler.cutedsl_arch_utils import get_cuda_arch

        arch = get_cuda_arch()
        # Should be at least SM70 (Volta) for modern GPUs
        self.assertGreaterEqual(arch, 70)
        self.assertLessEqual(arch, 120)


if __name__ == "__main__":
    unittest.main()
