from __future__ import annotations

from itertools import starmap
import math
import os
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from benchmarks.cute import blackwell_grouped_gemm_direct as blackwell_benchmark

from helion._compiler.cute import grouped_deepgemm
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA


CPU_DEVICE = torch.device("cpu")


class TestCuteBlackwellGroupedGemmDirectBenchmark(TestCase):
    def _skip_unless_blackwell_grouped_gemm(self) -> None:
        from helion._compat import requires_cuda_version
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if DEVICE.type != "cuda":
            self.skipTest("Blackwell grouped GEMM requires CUDA")
        if not requires_cuda_version("13"):
            self.skipTest("Blackwell grouped GEMM requires CUDA >= 13")
        with torch.cuda.device(DEVICE):
            major, _minor = torch.cuda.get_device_capability(DEVICE)
            if major < 10:
                self.skipTest("Blackwell grouped GEMM requires SM100+")
            if not get_cute_mma_support().tcgen05_f16bf16:
                self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    def test_grouped_gemm_blackwell_nt_accepts_heterogeneous_fp16_metadata(self):
        from helion._compiler.cute import grouped_deepgemm

        group_A = tuple(
            torch.empty(m, k, device=DEVICE, dtype=torch.float16)
            for m, _n, k, _l in blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
        )
        group_B = tuple(
            torch.empty(n, k, device=DEVICE, dtype=torch.float16)
            for _m, n, k, _l in blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
        )
        out_groups = tuple(
            torch.empty(m, n, device=DEVICE, dtype=torch.float16)
            for m, n, _k, _l in blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
        )

        metadata = grouped_deepgemm._blackwell_grouped_gemm_nt_metadata(
            group_A,
            group_B,
            out_groups,
        )
        self.assertEqual(
            metadata.problems,
            tuple(
                starmap(
                    grouped_deepgemm.BlackwellGroupedGemmProblem,
                    blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
                )
            ),
        )
        self.assertEqual(
            grouped_deepgemm._compute_blackwell_total_clusters(
                metadata.problems,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            ),
            sum(
                math.ceil(m / 128) * math.ceil(n / 64)
                for m, n, _k, _l in blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
            ),
        )
        self.assertFalse(metadata.a_mode0_major)
        self.assertFalse(metadata.b_mode0_major)
        self.assertFalse(metadata.c_mode0_major)

    def test_grouped_gemm_blackwell_nt_arg_builder_validates_alignment(self):

        self.assertEqual(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            (
                (128, 128, 128, 1),
                (512, 128, 128, 1),
                (128, 256, 128, 1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "K.*16-byte"):
            blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
                ((128, 128, 130, 1),),
                dtype=torch.float16,
                device=CPU_DEVICE,
            )
        with self.assertRaisesRegex(ValueError, "N.*16-byte"):
            blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
                ((128, 130, 128, 1),),
                dtype=torch.float16,
                device=CPU_DEVICE,
            )
        padded = torch.empty(128, 129, device=CPU_DEVICE, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "leading stride.*16-byte"):
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_check_16b_alignment(
                padded[:, :128],
                major="K",
                mode0="M",
                tensor_name="A",
                group=0,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_direct_source(self):
        self._skip_unless_blackwell_grouped_gemm()
        torch.manual_seed(0)
        args, _expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )
        prepared = blackwell_benchmark._blackwell_grouped_gemm_nt_generated_args(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        self.assertIsNotNone(prepared)
        assert prepared is not None
        kernel_args, _out_tuple = prepared
        direct_pointers = kernel_args[-2]
        self.assertEqual(
            direct_pointers.detach().cpu().tolist(),
            [
                [int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr())]
                for a, b, out in zip(group_A, group_B, out_groups, strict=True)
            ],
        )

        bound = blackwell_benchmark._blackwell_grouped_gemm_nt_generated_kernel.bind(
            kernel_args
        )
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        code = bound.to_triton_code(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_config(
                reserved_sms=4
            )
        )
        self.assertIn("PipelineTmaUmma.create(num_stages=3", code)
        self.assertIn("'external_direct_pointer_metadata': True", code)
        self.assertIn("'tcgen05_grouped_static_reserved_sms': 4", code)
        self.assertIn("tcgen05_grouped_direct_pointers", code)
        self.assertIn("tcgen05_grouped_direct_strides", code)
        self.assertIn("tcgen05_grouped_tensormap_a_addr = (", code)
        self.assertIn("tcgen05_grouped_tensormap_b_addr = (", code)
        self.assertIn("tcgen05_grouped_d_tensormap_addr = (", code)
        self.assertIn("cute.make_ptr(cutlass.Float16", code)
        self.assertNotIn(
            "tcgen05_grouped_tensormap_a_base = a_placeholder.iterator +",
            code,
        )
        self.assertNotIn(
            "tcgen05_grouped_tensormap_b_base = b_placeholder.iterator +",
            code,
        )
        self.assertNotIn(
            "tcgen05_grouped_d_tensormap_base = out_placeholder.iterator +",
            code,
        )
        self.assertNotIn("_cutlass_grouped_gemm_kernel", code)
        self.assertNotIn("GroupedGemmKernel", code)
        self.assertNotIn("torch.cat", code)
        self.assertNotIn("torch.stack", code)

        default_args, _default_expected = (
            blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
                blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
                dtype=torch.float16,
                device=DEVICE,
            )
        )
        default_group_A, default_group_B = default_args
        default_out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(default_group_A, default_group_B, strict=True)
        )
        default_prepared = (
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_args(
                default_group_A,
                default_group_B,
                default_out_groups,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
            )
        )
        self.assertIsNotNone(default_prepared)
        assert default_prepared is not None
        default_kernel_args, _default_out_tuple = default_prepared
        default_bound = (
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_kernel.bind(
                default_kernel_args
            )
        )
        default_bound.env.config_spec.cute_tcgen05_search_enabled = True
        default_code = default_bound.to_triton_code(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_config()
        )
        self.assertIn("PipelineTmaUmma.create(num_stages=2", default_code)
        self.assertNotIn("tcgen05_grouped_static_reserved_sms", default_code)

    def test_grouped_gemm_blackwell_nt_generated_ab_stage_selector(self):
        captured_configs: list[Any] = []
        call_count = 0
        device = torch.device("cpu")

        class FakeConfigSpec:
            cute_tcgen05_search_enabled = False

        class FakeEnv:
            def __init__(self) -> None:
                self.config_spec = FakeConfigSpec()

        class FakeBound:
            def __init__(self) -> None:
                self.env = FakeEnv()

            def set_config(self, config: Any) -> None:
                captured_configs.append(config)

            def __call__(self, *args: object) -> None:
                nonlocal call_count
                call_count += 1

        def fake_bind(args: tuple[torch.Tensor, ...]) -> FakeBound:
            return FakeBound()

        def run_case(k: int) -> None:
            group_A = (torch.empty(128, k, device=device, dtype=torch.float16),)
            group_B = (torch.empty(64, k, device=device, dtype=torch.float16),)
            out_groups = (torch.empty(128, 64, device=device, dtype=torch.float16),)
            kernel_args = (torch.empty(1, device=device, dtype=torch.float16),)
            with patch.object(
                blackwell_benchmark,
                "_blackwell_grouped_gemm_nt_generated_args",
                return_value=(kernel_args, out_groups),
            ) as generated_args:
                result = blackwell_benchmark._blackwell_grouped_gemm_nt_generated(
                    group_A,
                    group_B,
                    out_groups,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                    use_2cta_instrs=False,
                )
            generated_args.assert_called_once_with(
                group_A,
                group_B,
                out_groups,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            )
            self.assertIs(result, out_groups)

        with (
            patch.dict(
                os.environ,
                {
                    "HELION_BACKEND": "cute",
                    "HELION_CUTE_MMA_IMPL": "tcgen05",
                    "HELION_AUTOTUNE_EFFORT": "none",
                },
                clear=False,
            ),
            patch.object(
                blackwell_benchmark._blackwell_grouped_gemm_nt_generated_kernel,
                "bind",
                side_effect=fake_bind,
            ),
            patch.object(
                blackwell_benchmark,
                "_blackwell_grouped_gemm_nt_generated_reserved_sms",
                return_value=0,
            ),
        ):
            run_case(128)
            run_case(192)

        self.assertEqual(call_count, 2)
        self.assertEqual(len(captured_configs), 2)
        small_k_config = captured_configs[0].config
        large_k_config = captured_configs[1].config
        self.assertEqual(small_k_config["block_sizes"], [128, 64, 64])
        self.assertEqual(small_k_config["num_stages"], 2)
        self.assertNotIn("tcgen05_ab_stages", small_k_config)
        self.assertEqual(large_k_config["block_sizes"], [128, 64, 64])
        self.assertEqual(large_k_config["tcgen05_ab_stages"], 4)

    @onlyBackends(["cute"])
    def test_grouped_gemm_blackwell_nt_generated_reserved_sms_selector(self):
        problem = blackwell_benchmark._BlackwellGeneratedGemmProblem
        self.assertEqual(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(16, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 160, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            3,
        )
        self.assertEqual(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(128, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 192, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            0,
        )
        self.assertEqual(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(16, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 192, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            3,
        )
        self.assertEqual(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(16, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 128, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            0,
        )
        self.assertEqual(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8065, 1280, 32),
                    problem(128, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 128, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            2,
        )
        self.assertNotIn(
            "tcgen05_grouped_static_reserved_sms",
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_config().config,
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_declines_common_envelope(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE.clear()
        grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.clear()
        torch.manual_seed(0)
        args, _expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            ((128, 160, 80, 1),),
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )
        prepared = blackwell_benchmark._blackwell_grouped_gemm_nt_generated_args(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        self.assertIsNone(prepared)

        with patch.object(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_kernel,
            "bind",
            side_effect=AssertionError("generated bind should not be reached"),
        ):
            self.assertIsNone(
                blackwell_benchmark._blackwell_grouped_gemm_nt_generated(
                    group_A,
                    group_B,
                    out_groups,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                    use_2cta_instrs=False,
                )
            )

        self.assertEqual(grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE, {})
        self.assertEqual(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE, {})

    def _assert_blackwell_grouped_gemm_nt_correct(
        self,
        problem_sizes_mnkl,
        *,
        mma_tiler_mn,
    ) -> None:
        self._skip_unless_blackwell_grouped_gemm()
        torch.manual_seed(0)
        if problem_sizes_mnkl is None:
            args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
                dtype=torch.float16,
                device=DEVICE,
            )
            problem_sizes_mnkl = (
                blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES
            )
        else:
            args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
                problem_sizes_mnkl,
                dtype=torch.float16,
                device=DEVICE,
            )

        group_A, group_B = args
        self.assertEqual(
            tuple(
                (int(a.size(0)), int(b.size(0)), int(a.size(1)), 1)
                for a, b in zip(group_A, group_B, strict=True)
            ),
            tuple(problem_sizes_mnkl),
        )
        actual = grouped_deepgemm.blackwell_grouped_gemm_nt(
            *args,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        for group, (actual_group, expected_group) in enumerate(
            zip(actual, expected, strict=True)
        ):
            with self.subTest(group=group):
                torch.testing.assert_close(
                    actual_group,
                    expected_group,
                    rtol=1e-2,
                    atol=1e-1,
                )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_official_default_fp16_correct(self):
        self._assert_blackwell_grouped_gemm_nt_correct(
            None,
            mma_tiler_mn=(128, 128),
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_documented_mixed_fp16_correct(self):
        self._assert_blackwell_grouped_gemm_nt_correct(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
            mma_tiler_mn=(128, 64),
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_fp16_correct(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE.clear()
        torch.manual_seed(0)
        args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE, {})
        for actual_group, expected_group, out_group in zip(
            actual,
            expected,
            out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_cache_freshness(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()

        torch.manual_seed(0)
        blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            1,
        )
        cached_launch = next(
            iter(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.values())
        )
        for actual_group, expected_group in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        group_A[0].zero_()
        updated_expected = blackwell_benchmark._reference_blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_dtype=torch.float16,
        )
        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertIs(
            next(
                iter(
                    blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.values()
                )
            ),
            cached_launch,
        )
        for actual_group, expected_group in zip(
            actual,
            updated_expected,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        fresh_group_A = (group_A[0].clone(), *group_A[1:])
        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            fresh_group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            2,
        )
        for actual_group, expected_group in zip(
            actual,
            updated_expected,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        fresh_out_groups = tuple(torch.empty_like(out) for out in out_groups)
        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=fresh_out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            3,
        )
        for actual_group, expected_group, fresh_out in zip(
            actual,
            updated_expected,
            fresh_out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, fresh_out)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            1,
        )
        for actual_group, expected_group in zip(
            actual,
            updated_expected,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_graph_capture(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        torch.manual_seed(0)
        blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            1,
        )

        for out in out_groups:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured: list[tuple[torch.Tensor, ...]] = []
        with (
            patch.object(
                grouped_deepgemm,
                "blackwell_grouped_gemm_nt",
                side_effect=AssertionError(
                    "generated graph capture should not fallback"
                ),
            ),
            torch.cuda.graph(graph),
        ):
            captured.append(
                blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
                    group_A,
                    group_B,
                    out_groups=out_groups,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                )
            )
        for out in out_groups:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(len(captured), 1)
        for actual_group, expected_group, out_group in zip(
            captured[0],
            expected,
            out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_blackwell_nt_generated_explicit_outputs_graph_capture_lru_hit(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        torch.manual_seed(0)
        blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, _expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args

        def new_out_groups() -> tuple[torch.Tensor, ...]:
            return tuple(
                torch.empty(
                    int(a.size(0)),
                    int(b.size(0)),
                    device=DEVICE,
                    dtype=torch.float16,
                )
                for a, b in zip(group_A, group_B, strict=True)
            )

        out_first = new_out_groups()
        out_second = new_out_groups()
        blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_first,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            1,
        )
        first_key = next(
            iter(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE)
        )
        first_launch = blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE[
            first_key
        ]

        blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_second,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            2,
        )
        self.assertIn(
            first_key,
            blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE,
        )
        self.assertIsNot(
            blackwell_benchmark._BLACKWELL_GENERATED_LAST_STABLE_LAUNCH,
            first_launch,
        )

        group_A[0].zero_()
        expected = blackwell_benchmark._reference_blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_dtype=torch.float16,
        )
        torch.cuda.synchronize()

        for out in out_first:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured: list[tuple[torch.Tensor, ...]] = []
        with (
            patch.object(
                grouped_deepgemm,
                "blackwell_grouped_gemm_nt",
                side_effect=AssertionError(
                    "generated graph capture should not fallback"
                ),
            ),
            patch.object(
                blackwell_benchmark._blackwell_grouped_gemm_nt_generated_kernel,
                "bind",
                side_effect=AssertionError("cached graph capture should not rebind"),
            ),
            torch.cuda.graph(graph),
        ):
            captured.append(
                blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
                    group_A,
                    group_B,
                    out_groups=out_first,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                )
            )

        for out in out_first:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(len(captured), 1)
        self.assertIs(
            blackwell_benchmark._BLACKWELL_GENERATED_LAST_STABLE_LAUNCH,
            first_launch,
        )
        for actual_group, expected_group, out_group in zip(
            captured[0],
            expected,
            out_first,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_retains_owned_metadata(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()
        import helion.runtime as helion_runtime

        torch.manual_seed(0)
        blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args

        def new_out_groups() -> tuple[torch.Tensor, ...]:
            return tuple(
                torch.empty(
                    int(a.size(0)),
                    int(b.size(0)),
                    device=DEVICE,
                    dtype=torch.float16,
                )
                for a, b in zip(group_A, group_B, strict=True)
            )

        out_groups = new_out_groups()
        actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            len(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE),
            1,
        )
        first_launch = next(
            iter(blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.values())
        )
        self.assertIsNotNone(first_launch.fast_call)
        self.assertIsNotNone(first_launch.runtime_cache_entry)
        runtime_cache_entry = first_launch.runtime_cache_entry
        retained_owned_tensors = runtime_cache_entry.owned_tensors
        self.assertGreater(len(retained_owned_tensors), 0)
        retained_owned_ptrs = tuple(
            int(tensor.data_ptr()) for tensor in retained_owned_tensors
        )
        for actual_group, expected_group in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        pressure_launches = helion_runtime._CUTE_LAUNCH_ARG_CACHE_LIMIT + 2
        self.assertLess(
            pressure_launches + 1,
            blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE_MAX,
        )
        pressure_out_groups = []
        for _ in range(pressure_launches):
            pressure_out = new_out_groups()
            pressure_out_groups.append(pressure_out)
            blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
                group_A,
                group_B,
                out_groups=pressure_out,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
            )
        torch.cuda.synchronize()
        self.assertIn(
            first_launch.cache_key,
            blackwell_benchmark._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE,
        )
        self.assertIs(first_launch.runtime_cache_entry, runtime_cache_entry)
        self.assertEqual(
            tuple(int(tensor.data_ptr()) for tensor in retained_owned_tensors),
            retained_owned_ptrs,
        )

        junk_tensors = []
        for fill_value in range(64):
            for tensor in retained_owned_tensors:
                junk = torch.empty_like(tensor)
                junk.fill_(fill_value + 1024)
                junk_tensors.append(junk)

        group_A[0].zero_()
        updated_expected = blackwell_benchmark._reference_blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_dtype=torch.float16,
        )
        for out in out_groups:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        with patch.object(
            blackwell_benchmark._blackwell_grouped_gemm_nt_generated_kernel,
            "bind",
            side_effect=AssertionError("relaunch should use cached public launch"),
        ):
            actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
                group_A,
                group_B,
                out_groups=out_groups,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
            )
        torch.cuda.synchronize()
        for actual_group, expected_group, out_group in zip(
            actual,
            updated_expected,
            out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )
        self.assertEqual(len(junk_tensors), 64 * len(retained_owned_tensors))
        self.assertEqual(len(pressure_out_groups), pressure_launches)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_documented_fp16_bf16_correct(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        for dtype, rtol, atol in (
            (torch.float16, 4e-2, 1e-1),
            (torch.bfloat16, 4e-2, 2e-1),
        ):
            with self.subTest(dtype=dtype):
                grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE.clear()
                torch.manual_seed(0)
                args, expected = (
                    blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
                        blackwell_benchmark.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
                        dtype=dtype,
                        device=DEVICE,
                    )
                )
                actual = blackwell_benchmark.blackwell_grouped_gemm_nt_direct(
                    *args,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                )
                torch.cuda.synchronize()
                self.assertEqual(
                    grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE,
                    {},
                )
                for actual_group, expected_group in zip(
                    actual,
                    expected,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_group,
                        expected_group,
                        rtol=rtol,
                        atol=atol,
                    )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_prepares_stable_output_launch(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.clear()
        torch.manual_seed(0)
        args, expected = blackwell_benchmark.make_blackwell_grouped_gemm_nt_args(
            ((128, 128, 128, 1),),
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        actual = grouped_deepgemm.blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-1)
        self.assertEqual(len(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE), 1)
        prepared = next(
            iter(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.values())
        )
        self.assertEqual(len(prepared.stream_executions), 1)

        actual = grouped_deepgemm.blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-1)
        self.assertIs(
            next(iter(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.values())),
            prepared,
        )

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            grouped_deepgemm.blackwell_grouped_gemm_nt(
                group_A,
                group_B,
                out_groups=out_groups,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out_groups[0], expected[0], rtol=1e-2, atol=1e-1)
