from __future__ import annotations

import ast
import importlib
import os
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import deepgemm_m_grouped as deepgemm_impl
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


def _has_cutlass_cute() -> bool:
    try:
        importlib.import_module("cutlass")
        importlib.import_module("cutlass.cute")
    except ImportError:
        return False
    return True


requires_cutlass_cute = pytest.mark.skipif(
    not _has_cutlass_cute(),
    reason="CUTLASS CuTe is not available",
)

CPU_DEVICE = torch.device("cpu")


def _cute_wrapper_plans_from_code(code: str) -> list[dict[str, object]]:
    marker = "._helion_cute_wrapper_plans = "
    line = next(line for line in code.splitlines() if marker in line)
    payload = line.split(marker, 1)[1]
    freeze_prefix = "helion.runtime._freeze_cute_wrapper_plans("
    if payload.startswith(freeze_prefix) and payload.endswith(")"):
        payload = payload[len(freeze_prefix) : -1]
    return list(ast.literal_eval(payload))


def _deepgemm_segment_test_kernel(body: object) -> helion.Kernel:
    return helion.kernel(
        static_shapes=False,
        config=helion.Config(block_sizes=[64, 64, 64]),
    )(body)


def _deepgemm_segment_bad_store_row_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index + 1, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _deepgemm_segment_bad_store_mask_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = valid_rows & (local_m >= 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _deepgemm_segment_store_m_mask_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _deepgemm_segment_inverted_row_mask_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = valid_m < local_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


_deepgemm_segment_bad_store_row_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_bad_store_row_body
)
_deepgemm_segment_bad_store_mask_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_bad_store_mask_body
)
_deepgemm_segment_store_m_mask_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_store_m_mask_body
)
_deepgemm_segment_inverted_row_mask_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_inverted_row_mask_body
)


class TestCuteDeepGemmMGrouped(TestCase):
    @onlyBackends(["cute"])
    @requires_cutlass_cute
    def test_grouped_gemm_deepgemm_m_grouped_bf16_nt_contiguous(self):
        torch.manual_seed(0)
        use_generated_default = (
            os.environ.get("HELION_CUTE_MMA_IMPL", "").strip().lower() == "tcgen05"
        )
        if use_generated_default:
            from helion._compat import requires_cuda_version
            from helion._compiler.cute.mma_support import get_cute_mma_support

            if DEVICE.type != "cuda":
                self.skipTest("CuTeDSL generated segment path requires CUDA")
            if not requires_cuda_version("13"):
                self.skipTest("CuTeDSL generated segment path requires CUDA >= 13")
            with torch.cuda.device(DEVICE):
                major, _minor = torch.cuda.get_device_capability(DEVICE)
                if major < 10:
                    self.skipTest("CuTeDSL generated segment path requires SM100+")
                if not get_cute_mma_support().tcgen05_f16bf16:
                    self.skipTest(
                        "tcgen05 F16/BF16 MMA is not supported on this machine"
                    )

        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (188, 64, 40, 64) if use_generated_default else (64, 96, 128, 32),
                n=128 if use_generated_default else 64,
                k=64,
                m_alignment=224 if use_generated_default else 64,
                tail_padding=0 if use_generated_default else 64,
                device=DEVICE,
            )
        )
        layout_has_valid_prefix_tiles = (
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                *args
            )
        )
        if use_generated_default:
            self.assertFalse(layout_has_valid_prefix_tiles)
            self.assertTrue(
                deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                    *args,
                    layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
                )
            )
            self.assertTrue(
                deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                    *args
                )
            )

        actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(torch.all(actual[args[2] == -1] == 0).item()))

        row_hole_layout = args[2].clone()
        row_hole_layout[16:32] = -1
        row_hole_layout[32:64] = 0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                args[0],
                args[1],
                row_hole_layout,
            )

    def _skip_unless_deepgemm_segment_tcgen05(self) -> None:
        from helion._compat import requires_cuda_version
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if os.environ.get("HELION_CUTE_MMA_IMPL", "").strip().lower() != "tcgen05":
            self.skipTest("CuTeDSL generated segment path requires tcgen05")
        if not requires_cuda_version("13"):
            self.skipTest("CuTeDSL generated segment path requires CUDA >= 13")
        with torch.cuda.device(DEVICE):
            major, _minor = torch.cuda.get_device_capability(DEVICE)
            if major < 10:
                self.skipTest("CuTeDSL generated segment path requires SM100+")
            if not get_cute_mma_support().tcgen05_f16bf16:
                self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

    def _deepgemm_segment_metadata_fixture(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        torch.manual_seed(0)
        args, _expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (188, 64, 40, 64),
                n=128,
                k=64,
                m_alignment=224,
                tail_padding=0,
                device=DEVICE,
            )
        )
        work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        return args, work_tile_metadata

    def _patch_torch_empty_for_deepgemm_output(
        self,
        replacement: torch.Tensor,
    ) -> object:
        real_empty = torch.empty

        def patched_empty(*shape: object, **kwargs: object) -> torch.Tensor:
            requested_shape = shape
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                requested_shape = tuple(shape[0])
            requested_device = kwargs.get("device")
            if (
                tuple(int(dim) for dim in requested_shape) == tuple(replacement.shape)
                and kwargs.get("dtype") == replacement.dtype
                and requested_device is not None
                and torch.device(requested_device) == replacement.device
            ):
                return replacement
            return real_empty(*shape, **kwargs)

        return patch.object(torch, "empty", side_effect=patched_empty)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("selected generated segment requires compiled CuTe code")
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_source_and_route(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (188, 64, 40, 64),
                n=128,
                k=64,
                m_alignment=224,
                tail_padding=0,
                device=DEVICE,
            )
        )
        A_packed, B_grouped, grouped_layout = args
        A_packed[grouped_layout == -1] = 0
        layout_has_valid_prefix_tiles = (
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                *args
            )
        )
        use_generated_segment_kernel = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
            *args,
            layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
        )
        use_selected_segment_kernel = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
            *args
        )
        work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            *args
        )
        self.assertFalse(layout_has_valid_prefix_tiles)
        self.assertTrue(use_generated_segment_kernel)
        self.assertTrue(use_selected_segment_kernel)
        self.assertIsNotNone(work_tile_metadata)
        self.assertEqual(
            {
                "layout_has_valid_prefix_tiles": layout_has_valid_prefix_tiles,
                "use_generated_segment_kernel": use_generated_segment_kernel,
                "use_selected_segment_kernel": use_selected_segment_kernel,
                "selected_metadata_nonempty": work_tile_metadata is not None
                and int(work_tile_metadata.size(0)) > 0,
                "route": "generated_selected_segment",
            },
            {
                "layout_has_valid_prefix_tiles": False,
                "use_generated_segment_kernel": True,
                "use_selected_segment_kernel": True,
                "selected_metadata_nonempty": True,
                "route": "generated_selected_segment",
            },
        )
        assert work_tile_metadata is not None
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 188, 224],
                [1, 224, 64, 224],
                [2, 448, 40, 224],
                [3, 672, 64, 224],
            ],
        )

        selected_kernel = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_kernel
        bound = selected_kernel.bind((A_packed, B_grouped, work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = selected_kernel.configs
        self.assertEqual(config["block_sizes"], [256, 128, 64])
        self.assertEqual(config["tcgen05_cluster_m"], 2)
        self.assertEqual(config["tcgen05_cluster_n"], 1)
        self.assertEqual(config["tcgen05_ab_stages"], 7)
        self.assertTrue(config["tcgen05_deepgemm_selected"])
        self.assertTrue(config["tcgen05_deepgemm_selected_compact_metadata"])
        self.assertEqual(config["tcgen05_selected_accumulator_view"], "nm")
        self.assertEqual(config["tcgen05_selected_d_store_view"], "nm_transposed")
        code = bound.to_triton_code(config)
        self.assertIn("tcgen05_grouped_selected_sched", code)
        self.assertIn("tcgen05_bSG_gD[None, cutlass.Int32(_tcgen05_subtile)]", code)
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertNotIn("CtaGroup.ONE", code)
        self.assertIn("out = torch.empty(", code)
        self.assertNotIn("out = torch.zeros(", code)
        self.assertNotIn(".zero_(", code)
        for helper_marker in (
            "_cutlass_grouped_gemm_kernel",
            "GroupedGemmKernel",
            "grouped_deepgemm",
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_zero_padding_rows",
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_output",
        ):
            self.assertNotIn(helper_marker, code)
        self.assertFalse(
            hasattr(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_zero_padding_rows",
            )
        )
        self.assertFalse(
            hasattr(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_output",
            )
        )

        grouped_plan = next(
            plan
            for plan in _cute_wrapper_plans_from_code(code)
            if plan["kind"] == "tcgen05_grouped_static_persistent"
        )
        self.assertIs(grouped_plan["deepgemm_selected"], True)
        self.assertIs(grouped_plan["deepgemm_selected_compact_metadata"], True)
        self.assertEqual(grouped_plan["source_m_tile"], 224)
        self.assertEqual(grouped_plan["cluster_m"], 2)
        self.assertEqual(grouped_plan["cluster_n"], 1)
        self.assertEqual(grouped_plan["selected_store_wave"], "nm_explicit_128x32")
        ab_plan = next(
            plan
            for plan in _cute_wrapper_plans_from_code(code)
            if plan["kind"] == "tcgen05_ab_tma"
        )
        self.assertEqual(ab_plan["ab_stage_count"], 7)
        selected_kernel.reset()

        with patch.object(
            deepgemm_impl,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(torch.all(actual[grouped_layout == -1] == 0).item()))

        sentinel = torch.full_like(expected, -77.0)
        with (
            patch.object(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
                side_effect=AssertionError("old segment fallback must not run"),
            ),
            self._patch_torch_empty_for_deepgemm_output(sentinel),
        ):
            actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        self.assertIs(actual, sentinel)
        torch.testing.assert_close(
            actual[grouped_layout != -1],
            expected[grouped_layout != -1],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertTrue(bool(torch.all(actual[grouped_layout == -1] == 0).item()))

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("CUDA graph capture requires compiled kernel")
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_graph_capture_padding(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (188, 64, 40, 64),
                n=128,
                k=64,
                m_alignment=224,
                tail_padding=0,
                device=DEVICE,
            )
        )
        A_packed, _B_grouped, grouped_layout = args
        padding_rows = grouped_layout == -1
        valid_rows = ~padding_rows
        self.assertGreater(int(torch.count_nonzero(padding_rows).item()), 0)
        self.assertTrue(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                *args
            )
        )
        with patch.object(
            deepgemm_impl,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            warm = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(warm, expected, rtol=2e-2, atol=2e-2)

        A_packed[padding_rows] = 7
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with (
            patch.object(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
                side_effect=AssertionError("old segment fallback must not run"),
            ),
            torch.cuda.graph(graph),
        ):
            captured = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()

        captured[valid_rows] = -5
        captured[padding_rows] = 13
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            captured[valid_rows],
            expected[valid_rows],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertTrue(bool(torch.all(captured[padding_rows] == 0).item()))

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("selected generated segment requires compiled CuTe code")
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_full_overwrite(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (224, 224),
                n=128,
                k=64,
                m_alignment=224,
                tail_padding=0,
                device=DEVICE,
            )
        )
        _A_packed, _B_grouped, grouped_layout = args
        self.assertEqual(int(torch.count_nonzero(grouped_layout == -1).item()), 0)
        self.assertTrue(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                *args
            )
        )
        work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        assert work_tile_metadata is not None
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 224, 224],
                [1, 224, 224, 224],
            ],
        )

        with patch.object(
            deepgemm_impl,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            warm = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(warm, expected, rtol=2e-2, atol=2e-2)

        sentinel = torch.full_like(expected, float("nan"))
        with (
            patch.object(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
                side_effect=AssertionError("old segment fallback must not run"),
            ),
            self._patch_torch_empty_for_deepgemm_output(sentinel),
        ):
            actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        self.assertIs(actual, sentinel)
        self.assertFalse(bool(torch.any(torch.isnan(actual)).item()))
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("selected generated segment requires compiled CuTe code")
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_padding_only_chunk(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (224,),
                n=128,
                k=64,
                m_alignment=448,
                tail_padding=0,
                device=DEVICE,
            )
        )
        A_packed, _B_grouped, grouped_layout = args
        A_packed[grouped_layout == -1] = 7
        self.assertTrue(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                *args
            )
        )
        work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        assert work_tile_metadata is not None
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 224, 448],
            ],
        )

        with patch.object(
            deepgemm_impl,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(
            actual[grouped_layout != -1],
            expected[grouped_layout != -1],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertTrue(bool(torch.all(actual[grouped_layout == -1] == 0).item()))

    def _assert_segment_tcgen05_not_admitted(
        self,
        kernel: helion.Kernel,
        args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        work_tile_metadata: torch.Tensor,
    ) -> None:
        bound = kernel.bind((args[0], args[1], work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = kernel.configs
        code = bound.to_triton_code(config)
        self.assertNotIn("cute.nvgpu.tcgen05", code)
        self.assertNotIn("cute.gemm(", code)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("CUDA graph capture requires compiled kernel")
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_mixed_boundary(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (188, 64, 40, 64),
                n=128,
                k=64,
                m_alignment=224,
                tail_padding=0,
                device=DEVICE,
            )
        )
        layout_has_valid_prefix_tiles = (
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                *args
            )
        )
        self.assertFalse(layout_has_valid_prefix_tiles)
        self.assertTrue(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                *args,
                layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
            )
        )
        self.assertFalse(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_output_fully_overwritten(
                *args
            )
        )

        work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 128, 0],
                [0, 128, 60, 0],
                [1, 224, 64, 0],
                [2, 448, 40, 0],
                [3, 672, 64, 0],
            ],
        )
        segment_kernel = (
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_kernel
        )
        bound = segment_kernel.bind((args[0], args[1], work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = segment_kernel.configs
        self.assertEqual(config["block_sizes"], [128, 128, 64])
        self.assertEqual(config["tcgen05_ab_stages"], 3)
        self.assertEqual(config["tcgen05_c_stages"], 4)
        code = bound.to_triton_code(config)
        self.assertIn("'cta_tile_shape_mnk': (128, 128, 64)", code)
        self.assertIn("'bn': 128", code)
        self.assertIn("'ab_stage_count': 3", code)
        self.assertIn("'c_stage_count': 4", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn("PipelineTmaUmma.create", code)
        self.assertIn("StaticPersistentGroupTileScheduler.create", code)
        self.assertIn("TensorMapManager", code)
        self.assertIn("update_tensormap", code)
        self.assertIn("tma_partition", code)
        self.assertIn("CtaGroup.ONE", code)
        self.assertNotIn("CtaGroup.TWO", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)
        self.assertNotIn("_cutlass_grouped_gemm_kernel", code)
        self.assertNotIn("dot_product", code)
        self.assertNotIn("for _load_i", code)
        self.assertIn("work_tile_metadata.size(0)", code)
        self.assertIn("'worklist_metadata': True", code)
        self.assertIn("'real_groups_arg':", code)
        self.assertNotIn("M_total_aligned + _BLOCK_SIZE_0 - 1", code)
        self.assertIn("out = torch.zeros(", code)
        self.assertNotIn("out = torch.empty(", code)

        actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        padding_rows = args[2] == -1
        self.assertTrue(bool(torch.all(actual[padding_rows] == 0).item()))

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()

        captured[padding_rows] = 13
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(torch.all(captured[padding_rows] == 0).item()))

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("CUDA graph capture requires compiled kernel")
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_full_overwrite_uses_empty(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (188, 68),
                n=128,
                k=64,
                m_alignment=1,
                tail_padding=0,
                device=DEVICE,
            )
        )
        self.assertEqual(int(torch.count_nonzero(args[2] == -1).item()), 0)
        layout_has_valid_prefix_tiles = (
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                *args
            )
        )
        self.assertFalse(layout_has_valid_prefix_tiles)
        self.assertTrue(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                *args,
                layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
            )
        )
        self.assertTrue(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_output_fully_overwritten(
                *args
            )
        )

        work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 128, 0],
                [0, 128, 60, 0],
                [1, 188, 68, 0],
            ],
        )
        segment_kernel = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_full_overwrite_kernel
        bound = segment_kernel.bind((args[0], args[1], work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = segment_kernel.configs
        code = bound.to_triton_code(config)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn("StaticPersistentGroupTileScheduler.create", code)
        self.assertIn("out = torch.empty(", code)
        self.assertNotIn("out = torch.zeros(", code)

        actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_rejects_store_mismatch(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        args, work_tile_metadata = self._deepgemm_segment_metadata_fixture()
        for name, kernel in (
            ("row", _deepgemm_segment_bad_store_row_kernel),
            ("mask", _deepgemm_segment_bad_store_mask_kernel),
            ("store_m", _deepgemm_segment_store_m_mask_kernel),
        ):
            with self.subTest(name=name):
                self._assert_segment_tcgen05_not_admitted(
                    kernel,
                    args,
                    work_tile_metadata,
                )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_rejects_inverted_row_mask(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        args, work_tile_metadata = self._deepgemm_segment_metadata_fixture()
        self._assert_segment_tcgen05_not_admitted(
            _deepgemm_segment_inverted_row_mask_kernel,
            args,
            work_tile_metadata,
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_cache_keys_preserve_capture_metadata(
        self,
    ):
        layout_validation_cache = deepgemm_impl._DEEPGEMM_LAYOUT_VALIDATION_CACHE
        layout_segments_cache = deepgemm_impl._DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE
        segment_tensor_cache = deepgemm_impl._DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE
        selected_unsupported_cache = (
            deepgemm_impl._DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE
        )
        saved_layout_validation_cache = layout_validation_cache.copy()
        saved_layout_segments_cache = layout_segments_cache.copy()
        saved_segment_tensor_cache = segment_tensor_cache.copy()
        saved_selected_unsupported_cache = selected_unsupported_cache.copy()
        try:
            layout_validation_cache.clear()
            layout_segments_cache.clear()
            segment_tensor_cache.clear()
            selected_unsupported_cache.clear()

            m_per_group = (188, 64, 40, 64)
            layout: list[int] = []
            for group_id, actual_m in enumerate(m_per_group):
                layout.extend([group_id] * actual_m)
                layout.extend([-1] * ((-actual_m) % 224))
            args = (
                torch.empty(len(layout), 64, device=DEVICE, dtype=torch.bfloat16),
                torch.empty(
                    len(m_per_group),
                    128,
                    64,
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.tensor(layout, device=DEVICE, dtype=torch.int32),
            )
            legacy_metadata_key = deepgemm_impl._deepgemm_layout_validation_cache_key(
                *args,
                block_m=deepgemm_impl.DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
            )
            segment_worklist_key = deepgemm_impl._deepgemm_layout_validation_cache_key(
                *args,
                block_m=deepgemm_impl.DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M,
            )
            selected_worklist_key = deepgemm_impl._deepgemm_layout_validation_cache_key(
                *args,
                block_m=deepgemm_impl.DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_SOURCE_M_TILE,
            )
            self.assertNotEqual(legacy_metadata_key, segment_worklist_key)
            self.assertNotEqual(legacy_metadata_key, selected_worklist_key)
            self.assertNotEqual(segment_worklist_key, selected_worklist_key)

            layout_has_valid_prefix_tiles = (
                deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                    *args
                )
            )
            self.assertFalse(layout_has_valid_prefix_tiles)
            self.assertIsNotNone(
                deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
                    *args
                )
            )
            work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                *args
            )
            self.assertIsNotNone(work_tile_metadata)
            selected_work_tile_metadata = deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
                *args
            )
            self.assertIsNotNone(selected_work_tile_metadata)

            self.assertIn(legacy_metadata_key, layout_segments_cache)
            self.assertNotIn(segment_worklist_key, layout_segments_cache)
            self.assertNotIn(selected_worklist_key, layout_segments_cache)
            self.assertIn(segment_worklist_key, segment_tensor_cache)
            self.assertIn(selected_worklist_key, segment_tensor_cache)
            self.assertNotIn(legacy_metadata_key, segment_tensor_cache)

            with (
                patch.object(
                    deepgemm_impl,
                    "_parse_deepgemm_m_grouped_layout_segments",
                    side_effect=AssertionError(
                        "unexpected layout parse during capture"
                    ),
                ),
                patch.object(
                    deepgemm_impl,
                    "_deepgemm_cuda_graph_capture_active",
                    return_value=True,
                ),
            ):
                self.assertFalse(
                    deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                        *args
                    )
                )
                self.assertIsNotNone(
                    deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
                        *args
                    )
                )
                self.assertTrue(
                    deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                        *args,
                        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
                    )
                )
                self.assertIs(
                    deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                        *args
                    ),
                    work_tile_metadata,
                )
                self.assertIs(
                    deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
                        *args
                    ),
                    selected_work_tile_metadata,
                )
        finally:
            layout_validation_cache.clear()
            layout_validation_cache.update(saved_layout_validation_cache)
            layout_segments_cache.clear()
            layout_segments_cache.update(saved_layout_segments_cache)
            segment_tensor_cache.clear()
            segment_tensor_cache.update(saved_segment_tensor_cache)
            selected_unsupported_cache.clear()
            selected_unsupported_cache.update(saved_selected_unsupported_cache)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @requires_cutlass_cute
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_valid_prefix_uses_legacy_generated_kernel(
        self,
    ):
        torch.manual_seed(0)
        args, expected = (
            deepgemm_impl.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                (64, 96, 128, 32),
                n=64,
                k=64,
                m_alignment=64,
                tail_padding=64,
                device=DEVICE,
            )
        )
        layout_has_valid_prefix_tiles = (
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                *args
            )
        )
        self.assertTrue(layout_has_valid_prefix_tiles)
        self.assertFalse(
            deepgemm_impl._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                *args,
                layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
            )
        )

        with (
            patch.object(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_generated_segment",
                side_effect=AssertionError("segment fallback must not run"),
            ),
            patch.object(
                deepgemm_impl,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel",
                return_value=expected,
            ) as legacy_kernel,
        ):
            actual = deepgemm_impl.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        legacy_kernel.assert_called_once_with(*args)
        self.assertIs(actual, expected)

    @onlyBackends(["cute"])
    def test_grouped_gemm_deepgemm_m_grouped_layout_parser_rejects_invalid(self):
        def validate_layout(layout: list[int]) -> None:
            rows = len(layout)
            a_packed = torch.empty(rows, 8, device=CPU_DEVICE, dtype=torch.bfloat16)
            b_grouped = torch.empty(4, 8, 8, device=CPU_DEVICE, dtype=torch.bfloat16)
            grouped_layout = torch.tensor(layout, device=CPU_DEVICE, dtype=torch.int32)
            deepgemm_impl._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                a_packed,
                b_grouped,
                grouped_layout,
                block_m=4,
            )

        invalid_layouts = (
            ([0, 0, 2, 2, 1, 1], "strictly increasing"),
            ([0, 0, -1, 1, 1, -1, 1, 1], "strictly increasing"),
            ([0, 0, -1, 0], "strictly increasing"),
            ([0, -2, 1], "values must be -1"),
        )
        for layout, error in invalid_layouts:
            with (
                self.subTest(layout=layout),
                self.assertRaisesRegex(ValueError, error),
            ):
                validate_layout(layout)
