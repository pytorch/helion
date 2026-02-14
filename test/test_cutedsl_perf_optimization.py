"""Tests for Phase 5: CuteDSL performance optimization utilities.

Tests arch utils extensions, MMA utils SM90/SM100 paths,
pipeline utilities, and GEMM codegen arch-aware paths.
"""

from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import TestCase
from helion.runtime.config import Config


class TestArchUtilsExtensions(TestCase):
    """Tests for extended arch utils (Phase 5)."""

    def test_get_num_sms(self):
        from helion._compiler.cutedsl_arch_utils import get_num_sms

        num_sms = get_num_sms()
        self.assertGreater(num_sms, 0)
        self.assertLess(num_sms, 1024)  # Sanity check

    def test_get_l2_cache_size(self):
        from helion._compiler.cutedsl_arch_utils import get_l2_cache_size

        l2_size = get_l2_cache_size()
        self.assertGreaterEqual(l2_size, 0)

    def test_supports_warpgroup_mma(self):
        from helion._compiler.cutedsl_arch_utils import supports_warpgroup_mma

        # Test with explicit arch values
        self.assertFalse(supports_warpgroup_mma(80))
        self.assertFalse(supports_warpgroup_mma(86))
        self.assertTrue(supports_warpgroup_mma(90))
        self.assertTrue(supports_warpgroup_mma(100))

    def test_supports_tmem(self):
        from helion._compiler.cutedsl_arch_utils import supports_tmem

        self.assertFalse(supports_tmem(80))
        self.assertFalse(supports_tmem(90))
        self.assertTrue(supports_tmem(100))

    def test_get_warpgroup_size(self):
        from helion._compiler.cutedsl_arch_utils import get_warpgroup_size

        self.assertEqual(get_warpgroup_size(80), 32)
        self.assertEqual(get_warpgroup_size(86), 32)
        self.assertEqual(get_warpgroup_size(90), 128)
        self.assertEqual(get_warpgroup_size(100), 128)

    def test_select_num_stages_single_stage(self):
        from helion._compiler.cutedsl_arch_utils import select_num_stages

        # Very large tiles should result in fewer stages
        result = select_num_stages(
            tile_m=256, tile_n=256, tile_k=256,
            dtype_bytes=4, num_warps=4, arch=80,
        )
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 4)

    def test_select_num_stages_multi_stage(self):
        from helion._compiler.cutedsl_arch_utils import select_num_stages

        # Small tiles should allow more stages
        result = select_num_stages(
            tile_m=64, tile_n=64, tile_k=32,
            dtype_bytes=2, num_warps=4, arch=80,
        )
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 4)

    def test_select_num_stages_hopper(self):
        from helion._compiler.cutedsl_arch_utils import select_num_stages

        # SM90 allows up to 8 stages
        result = select_num_stages(
            tile_m=64, tile_n=64, tile_k=32,
            dtype_bytes=2, num_warps=4, arch=90,
        )
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 8)

    def test_select_num_stages_zero_smem(self):
        from helion._compiler.cutedsl_arch_utils import select_num_stages

        # Zero-size tile_m means smem_per_stage=0 (only tile_k*tile_n > 0),
        # but tile_m=0 makes A tile 0 bytes; still limited by arch cap
        result = select_num_stages(
            tile_m=0, tile_n=64, tile_k=32,
            dtype_bytes=2, num_warps=4, arch=80,
        )
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 4)

    def test_compute_occupancy_smem(self):
        from helion._compiler.cutedsl_arch_utils import compute_occupancy_smem

        # Small SMEM usage should allow many blocks
        result = compute_occupancy_smem(1024, arch=80)
        self.assertGreater(result, 1)

        # Large SMEM usage should limit blocks
        result_large = compute_occupancy_smem(160 * 1024, arch=80)
        self.assertGreaterEqual(result_large, 1)
        self.assertLessEqual(result_large, result)

    def test_compute_occupancy_smem_zero(self):
        from helion._compiler.cutedsl_arch_utils import compute_occupancy_smem

        result = compute_occupancy_smem(0, arch=80)
        self.assertEqual(result, 32)


class TestMMAUtilsSM90(TestCase):
    """Tests for SM90 (Hopper) MMA utilities."""

    def test_get_mma_shape_sm90_fp16(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape_sm90

        m, n, k = get_mma_shape_sm90(dtype_bytes=2)
        self.assertEqual(m, 64)
        self.assertEqual(n, 16)
        self.assertEqual(k, 16)

    def test_get_mma_shape_sm90_fp32(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape_sm90

        m, n, k = get_mma_shape_sm90(dtype_bytes=4)
        self.assertEqual(m, 64)
        self.assertEqual(n, 16)
        self.assertEqual(k, 8)

    def test_get_mma_shape_sm90_fp8(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape_sm90

        m, n, k = get_mma_shape_sm90(dtype_bytes=1)
        self.assertEqual(m, 64)
        self.assertEqual(n, 16)
        self.assertEqual(k, 32)

    def test_emit_tiled_mma_sm90(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_sm90

        code = emit_tiled_mma_sm90(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=8, dtype_bytes=2,
        )
        self.assertIn("sm90_utils_basic.make_trivial_tiled_mma", code)
        self.assertIn("OperandMajorMode.K", code)
        self.assertIn("cutlass.Float32", code)
        # 8 warps = 2 warpgroups
        self.assertIn("atom_layout_mnk=(2, 1, 1)", code)

    def test_emit_tiled_mma_sm90_rmem_source(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_sm90

        code = emit_tiled_mma_sm90(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=4, dtype_bytes=2,
            a_source="RMEM",
        )
        self.assertIn("OperandSource.RMEM", code)

    def test_emit_smem_layout_atom_sm90(self):
        from helion._compiler.cutedsl_mma_utils import emit_smem_layout_atom_sm90

        code = emit_smem_layout_atom_sm90(64, 2, "cutlass.Float16")
        self.assertIn("warpgroup.make_smem_layout_atom", code)
        self.assertIn("sm90_utils_basic.get_smem_layout_atom", code)
        self.assertIn("LayoutEnum.ROW_MAJOR", code)


class TestMMAUtilsSM100(TestCase):
    """Tests for SM100 (Blackwell) MMA utilities."""

    def test_get_mma_shape_sm100_fp16(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape_sm100

        m, n, k = get_mma_shape_sm100(dtype_bytes=2)
        self.assertEqual(m, 128)
        self.assertEqual(n, 256)
        self.assertEqual(k, 32)

    def test_get_mma_shape_sm100_fp8(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape_sm100

        m, n, k = get_mma_shape_sm100(dtype_bytes=1)
        self.assertEqual(m, 128)
        self.assertEqual(n, 256)
        self.assertEqual(k, 64)

    def test_get_mma_shape_sm100_fp32(self):
        from helion._compiler.cutedsl_mma_utils import get_mma_shape_sm100

        m, n, k = get_mma_shape_sm100(dtype_bytes=4)
        self.assertEqual(m, 128)
        self.assertEqual(n, 256)
        self.assertEqual(k, 16)

    def test_emit_tiled_mma_sm100(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_sm100

        code = emit_tiled_mma_sm100(
            "cutlass.Float16", "cutlass.Float32", dtype_bytes=2,
        )
        self.assertIn("tcgen05.mma.MmaOp", code)
        self.assertIn("shape_mnk=(128, 256, 32)", code)
        self.assertIn("OperandMajorMode.K", code)

    def test_emit_tiled_mma_sm100_tmem(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_sm100

        code = emit_tiled_mma_sm100(
            "cutlass.Float16", "cutlass.Float32",
            dtype_bytes=2, a_source="TMEM",
        )
        self.assertIn("OperandSource.TMEM", code)

    def test_tmem_capacity_constant(self):
        from helion._compiler.cutedsl_mma_utils import SM100_TMEM_CAPACITY_COLUMNS

        self.assertEqual(SM100_TMEM_CAPACITY_COLUMNS, 512)


class TestTMACopyAtoms(TestCase):
    """Tests for TMA copy atom generation."""

    def test_emit_tma_copy_atom(self):
        from helion._compiler.cutedsl_mma_utils import emit_tma_copy_atom

        code = emit_tma_copy_atom("cutlass.Float16")
        self.assertIn("cpasync.tma_partition", code)
        self.assertIn("CopyTmaOp", code)

    def test_emit_tma_store_atom(self):
        from helion._compiler.cutedsl_mma_utils import emit_tma_store_atom

        code = emit_tma_store_atom("cutlass.Float16")
        self.assertIn("CopyTmaStoreOp", code)


class TestSMEMDescriptorLayouts(TestCase):
    """Tests for SM100 SMEM descriptor layout types."""

    def test_get_smem_descriptor_layout_type(self):
        from helion._compiler.cutedsl_mma_utils import (
            SMEM_LAYOUT_SWIZZLE_128B,
            SMEM_LAYOUT_SWIZZLE_128B_BASE32B,
            SMEM_LAYOUT_SWIZZLE_32B,
            SMEM_LAYOUT_SWIZZLE_64B,
            SMEM_LAYOUT_SWIZZLE_NONE,
            get_smem_descriptor_layout_type,
        )

        self.assertEqual(get_smem_descriptor_layout_type(0), SMEM_LAYOUT_SWIZZLE_NONE)
        self.assertEqual(get_smem_descriptor_layout_type(1), SMEM_LAYOUT_SWIZZLE_32B)
        self.assertEqual(get_smem_descriptor_layout_type(2), SMEM_LAYOUT_SWIZZLE_64B)
        self.assertEqual(get_smem_descriptor_layout_type(3), SMEM_LAYOUT_SWIZZLE_128B_BASE32B)
        self.assertEqual(get_smem_descriptor_layout_type(4), SMEM_LAYOUT_SWIZZLE_128B)

    def test_layout_type_values(self):
        from helion._compiler.cutedsl_mma_utils import (
            SMEM_LAYOUT_SWIZZLE_128B,
            SMEM_LAYOUT_SWIZZLE_128B_BASE32B,
            SMEM_LAYOUT_SWIZZLE_32B,
            SMEM_LAYOUT_SWIZZLE_64B,
            SMEM_LAYOUT_SWIZZLE_NONE,
        )

        self.assertEqual(SMEM_LAYOUT_SWIZZLE_NONE, 0)
        self.assertEqual(SMEM_LAYOUT_SWIZZLE_128B_BASE32B, 1)
        self.assertEqual(SMEM_LAYOUT_SWIZZLE_128B, 2)
        self.assertEqual(SMEM_LAYOUT_SWIZZLE_64B, 4)
        self.assertEqual(SMEM_LAYOUT_SWIZZLE_32B, 6)


class TestArchDispatch(TestCase):
    """Tests for architecture-aware MMA/copy dispatch."""

    def test_emit_tiled_mma_for_arch_sm80(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_for_arch

        code = emit_tiled_mma_for_arch(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=4, arch=80, dtype_bytes=2,
        )
        self.assertIn("cute.make_tiled_mma", code)
        self.assertIn("MmaF16BF16Op", code)

    def test_emit_tiled_mma_for_arch_sm90(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_for_arch

        code = emit_tiled_mma_for_arch(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=8, arch=90, dtype_bytes=2,
        )
        self.assertIn("sm90_utils_basic.make_trivial_tiled_mma", code)

    def test_emit_tiled_mma_for_arch_sm100(self):
        from helion._compiler.cutedsl_mma_utils import emit_tiled_mma_for_arch

        code = emit_tiled_mma_for_arch(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=4, arch=100, dtype_bytes=2,
        )
        self.assertIn("tcgen05.mma.MmaOp", code)

    def test_emit_gmem_copy_atom_for_arch_sm80(self):
        from helion._compiler.cutedsl_mma_utils import emit_gmem_copy_atom_for_arch

        code = emit_gmem_copy_atom_for_arch("cutlass.Float16", 16, arch=80)
        self.assertIn("CopyG2SOp", code)
        self.assertIn("cp_async", code.lower().replace("cpasync", "cp_async"))

    def test_emit_gmem_copy_atom_for_arch_sm90(self):
        from helion._compiler.cutedsl_mma_utils import emit_gmem_copy_atom_for_arch

        code = emit_gmem_copy_atom_for_arch("cutlass.Float16", 16, arch=90)
        self.assertIn("tma_partition", code)


class TestPipelineConfig(TestCase):
    """Tests for PipelineConfig."""

    def test_single_stage(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig(num_stages=1, arch=80)
        self.assertFalse(config.is_pipelined)
        self.assertEqual(config.pipeline_mode, "single")
        self.assertEqual(config.effective_buffers, 1)

    def test_cp_async_pipeline(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig(num_stages=3, arch=80)
        self.assertTrue(config.is_pipelined)
        self.assertEqual(config.pipeline_mode, "cp_async")
        self.assertEqual(config.effective_buffers, 3)

    def test_tma_pipeline(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig(num_stages=4, arch=90, use_tma=True)
        self.assertTrue(config.is_pipelined)
        self.assertEqual(config.pipeline_mode, "tma")

    def test_tma_only_on_sm90(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig(num_stages=4, arch=80, use_tma=True)
        self.assertEqual(config.pipeline_mode, "cp_async")

    def test_from_config(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig.from_config(num_stages=2, arch=90)
        self.assertTrue(config.use_tma)
        self.assertEqual(config.pipeline_mode, "tma")

    def test_from_config_no_tma(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig.from_config(num_stages=2, arch=90, prefer_tma=False)
        self.assertFalse(config.use_tma)
        self.assertEqual(config.pipeline_mode, "cp_async")

    def test_custom_buffers(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig

        config = PipelineConfig(num_stages=3, arch=80, num_buffers=2)
        self.assertEqual(config.effective_buffers, 2)


class TestPipelineCodegen(TestCase):
    """Tests for pipeline code generation."""

    def test_emit_pipeline_state_init_single(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_state_init

        config = PipelineConfig(num_stages=1, arch=80)
        lines = emit_pipeline_state_init(config)
        self.assertEqual(lines, [])

    def test_emit_pipeline_state_init_cp_async(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_state_init

        config = PipelineConfig(num_stages=3, arch=80)
        lines = emit_pipeline_state_init(config)
        self.assertTrue(any("pipe_stage" in l for l in lines))
        self.assertTrue(any("cp.async" in l.lower() or "cp_async" in l.lower() for l in lines))

    def test_emit_pipeline_state_init_tma(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_state_init

        config = PipelineConfig(num_stages=3, arch=90, use_tma=True)
        lines = emit_pipeline_state_init(config)
        self.assertTrue(any("pipe_phase" in l for l in lines))
        self.assertTrue(any("mbar" in l for l in lines))

    def test_emit_pipeline_wait_single(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_wait

        config = PipelineConfig(num_stages=1, arch=80)
        lines = emit_pipeline_wait(config)
        self.assertTrue(any("syncthreads" in l for l in lines))

    def test_emit_pipeline_wait_cp_async(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_wait

        config = PipelineConfig(num_stages=3, arch=80)
        lines = emit_pipeline_wait(config)
        self.assertTrue(any("cp_async_wait_group" in l for l in lines))

    def test_emit_pipeline_wait_tma(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_wait

        config = PipelineConfig(num_stages=3, arch=90, use_tma=True)
        lines = emit_pipeline_wait(config)
        self.assertTrue(any("mbarrier_wait" in l for l in lines))

    def test_emit_pipeline_commit_cp_async(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_commit

        config = PipelineConfig(num_stages=3, arch=80)
        lines = emit_pipeline_commit(config)
        self.assertTrue(any("cp_async_commit_group" in l for l in lines))

    def test_emit_pipeline_commit_tma(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_commit

        config = PipelineConfig(num_stages=3, arch=90, use_tma=True)
        lines = emit_pipeline_commit(config)
        self.assertTrue(any("mbarrier_arrive" in l for l in lines))

    def test_emit_pipeline_advance_single(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_advance

        config = PipelineConfig(num_stages=1, arch=80)
        lines = emit_pipeline_advance(config)
        self.assertEqual(lines, [])

    def test_emit_pipeline_advance_multi(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_advance

        config = PipelineConfig(num_stages=3, arch=80)
        lines = emit_pipeline_advance(config)
        self.assertTrue(any("pipe_stage" in l for l in lines))

    def test_emit_pipeline_drain_single(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_drain

        config = PipelineConfig(num_stages=1, arch=80)
        lines = emit_pipeline_drain(config)
        self.assertEqual(lines, [])

    def test_emit_pipeline_drain_cp_async(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipeline_drain

        config = PipelineConfig(num_stages=3, arch=80)
        lines = emit_pipeline_drain(config)
        self.assertTrue(any("cp_async_wait_group(0)" in l for l in lines))


class TestPipelinedMainloop(TestCase):
    """Tests for the full pipelined mainloop generation."""

    def test_single_stage_loop(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipelined_mainloop

        config = PipelineConfig(num_stages=1, arch=80)
        lines = emit_pipelined_mainloop(config, tile_k=32, k_extent="K")
        combined = "\n".join(lines)
        self.assertIn("num_k_tiles", combined)
        self.assertIn("syncthreads", combined)
        # Single stage should not have pipeline state
        self.assertNotIn("pipe_stage", combined)

    def test_cp_async_loop(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipelined_mainloop

        config = PipelineConfig(num_stages=3, arch=80)
        lines = emit_pipelined_mainloop(config, tile_k=32, k_extent="K")
        combined = "\n".join(lines)
        self.assertIn("pipe_stage", combined)
        self.assertIn("cp_async", combined)
        self.assertIn("num_k_tiles", combined)

    def test_tma_loop(self):
        from helion._compiler.cutedsl_pipeline import PipelineConfig, emit_pipelined_mainloop

        config = PipelineConfig(num_stages=4, arch=90, use_tma=True)
        lines = emit_pipelined_mainloop(config, tile_k=32, k_extent="K")
        combined = "\n".join(lines)
        self.assertIn("mbar", combined)
        self.assertIn("pipe_phase", combined)


class TestGEMMCodegenArchAware(TestCase):
    """Tests for architecture-aware GEMM codegen."""

    def test_gemm_copy_setup_sm80(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_copy_setup

        lines = emit_gemm_copy_setup("cutlass.Float16", 16, 128, arch=80)
        combined = "\n".join(lines)
        self.assertIn("cp.async", combined)
        self.assertIn("smem_copy_atom", combined)

    def test_gemm_copy_setup_sm90(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_copy_setup

        lines = emit_gemm_copy_setup("cutlass.Float16", 16, 128, arch=90)
        combined = "\n".join(lines)
        self.assertIn("TMA", combined)

    def test_gemm_mma_setup_sm80(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mma_setup

        lines = emit_gemm_mma_setup(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=4, arch=80,
        )
        combined = "\n".join(lines)
        self.assertIn("warp MMA", combined)

    def test_gemm_mma_setup_sm90(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mma_setup

        lines = emit_gemm_mma_setup(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=8, arch=90,
        )
        combined = "\n".join(lines)
        self.assertIn("warpgroup MMA", combined)

    def test_gemm_mma_setup_sm100(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mma_setup

        lines = emit_gemm_mma_setup(
            "cutlass.Float16", "cutlass.Float32",
            num_warps=4, arch=100,
        )
        combined = "\n".join(lines)
        self.assertIn("tcgen05", combined)

    def test_gemm_mainloop_uses_pipeline(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mainloop

        lines = emit_gemm_mainloop(tile_k=32, k_extent="K", num_stages=3, arch=80)
        combined = "\n".join(lines)
        self.assertIn("pipeline=cp_async", combined)

    def test_gemm_mainloop_tma(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mainloop

        lines = emit_gemm_mainloop(tile_k=32, k_extent="K", num_stages=4, arch=90)
        combined = "\n".join(lines)
        self.assertIn("pipeline=tma", combined)

    def test_gemm_mainloop_single(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_mainloop

        lines = emit_gemm_mainloop(tile_k=32, k_extent="K", num_stages=1, arch=80)
        combined = "\n".join(lines)
        self.assertIn("pipeline=single", combined)

    def test_gemm_epilogue_with_cast(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_epilogue

        lines = emit_gemm_epilogue(
            arch=80,
            acc_dtype_str="cutlass.Float32",
            out_dtype_str="cutlass.Float16",
        )
        combined = "\n".join(lines)
        self.assertIn("Cast", combined)
        self.assertIn("acc_out", combined)

    def test_gemm_epilogue_no_cast(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_epilogue

        lines = emit_gemm_epilogue(
            arch=80,
            acc_dtype_str="cutlass.Float32",
            out_dtype_str="cutlass.Float32",
        )
        combined = "\n".join(lines)
        self.assertNotIn("Cast", combined)

    def test_gemm_epilogue_sm90(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_epilogue

        lines = emit_gemm_epilogue(arch=90)
        combined = "\n".join(lines)
        self.assertIn("SM90", combined)
        self.assertIn("StMatrix", combined)

    def test_gemm_epilogue_sm100(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_epilogue

        lines = emit_gemm_epilogue(arch=100)
        combined = "\n".join(lines)
        self.assertIn("SM100", combined)
        self.assertIn("TMEM", combined)

    def test_gemm_full_sm80(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_full

        lines = emit_gemm_full(
            tile_m=128, tile_n=128, tile_k=32,
            dtype_str="cutlass.Float16", acc_dtype_str="cutlass.Float32",
            out_dtype_str="cutlass.Float16",
            dtype_bytes=2, num_stages=2, num_warps=4, arch=80,
        )
        combined = "\n".join(lines)
        # Preamble
        self.assertIn("Shared memory allocation", combined)
        self.assertIn("MMA atom", combined)
        # Accumulator
        self.assertIn("acc", combined)
        # Main loop
        self.assertIn("GEMM loop", combined)
        # Epilogue
        self.assertIn("Epilogue", combined)

    def test_gemm_full_sm90(self):
        from helion._compiler.cutedsl_gemm_codegen import emit_gemm_full

        lines = emit_gemm_full(
            tile_m=128, tile_n=128, tile_k=32,
            dtype_str="cutlass.Float16", acc_dtype_str="cutlass.Float32",
            out_dtype_str="cutlass.Float16",
            dtype_bytes=2, num_stages=4, num_warps=8, arch=90,
        )
        combined = "\n".join(lines)
        self.assertIn("TMA", combined)
        self.assertIn("warpgroup MMA", combined)
        self.assertIn("StMatrix", combined)


@helion.kernel(backend="cutedsl")
def _regression_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


@helion.kernel(backend="cutedsl")
def _regression_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    result = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        result[tile_m, tile_n] = acc
    return result


class TestExistingTestsStillPass(TestCase):
    """Verify Phase 5 changes don't break existing CuteDSL codegen."""

    def _get_code(self, fn, args, **config_kwargs):
        bound = fn.bind(args)
        if config_kwargs:
            config = Config(**config_kwargs)
        else:
            config = bound.config_spec.default_config()
        return bound.to_triton_code(config)

    def test_add_kernel_still_works(self):
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code = self._get_code(_regression_add_kernel, args)
        self.assertIn("@cute.kernel", code)
        self.assertIn("cutlass", code)

    def test_matmul_kernel_still_works(self):
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
        )
        code = self._get_code(_regression_matmul_kernel, args)
        self.assertIn("@cute.kernel", code)
        self.assertIn("tl.dot", code)


if __name__ == "__main__":
    unittest.main()
