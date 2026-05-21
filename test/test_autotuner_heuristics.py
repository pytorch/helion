from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2Heuristic
from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonSkinnyGemmHeuristic
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from helion._hardware import HardwareInfo
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import default_cute_mma_support
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipIfRefEager
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl
from helion.runtime.settings import Settings

HOPPER_HARDWARE = HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA H100",
    runtime_version="12.8",
    compute_capability="sm90",
)
MI350_HARDWARE = HardwareInfo(
    device_kind="rocm",
    hardware_name="AMD MI350",
    runtime_version="7.0",
    compute_capability="gfx950",
)
BLACKWELL_HARDWARE = HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA B200",
    runtime_version="12.8",
    compute_capability="sm100",
)


class TestAutotunerHeuristic(TestCase):
    def test_disable_autotuner_heuristics_setting_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_DISABLE_AUTOTUNER_HEURISTICS", None)
            self.assertFalse(Settings().disable_autotuner_heuristics)

        with patch.dict(
            os.environ,
            {"HELION_DISABLE_AUTOTUNER_HEURISTICS": "1"},
        ):
            self.assertTrue(Settings().disable_autotuner_heuristics)

    def test_compiler_seed_configs_handles_failed_optional_and_duplicate_seeds(
        self,
    ) -> None:
        class FailingAutotunerHeuristic(AutotunerHeuristic):
            name = "failing_autotuner_heuristic"
            backend = "triton"

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                return True

            @classmethod
            def get_seed_config(cls, env: object, device_ir: object) -> helion.Config:
                raise RuntimeError("synthetic compiler seed failure")

        class NoSeedAutotunerHeuristic(AutotunerHeuristic):
            name = "no_seed_autotuner_heuristic"
            backend = "triton"

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                return True

        class ValidAutotunerHeuristic(AutotunerHeuristic):
            name = "valid_autotuner_heuristic"
            backend = "triton"

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                return True

            @classmethod
            def get_seed_config(cls, env: object, device_ir: object) -> helion.Config:
                return helion.Config(block_sizes=[64])

        class DuplicateAutotunerHeuristic(ValidAutotunerHeuristic):
            name = "duplicate_autotuner_heuristic"

        env = MagicMock()
        env.backend_name = "triton"
        env.config_spec = MagicMock()
        env.settings = Settings()
        heuristics = (
            FailingAutotunerHeuristic,
            NoSeedAutotunerHeuristic,
            ValidAutotunerHeuristic,
            DuplicateAutotunerHeuristic,
        )

        with (
            self.assertLogs(
                "helion._compiler.autotuner_heuristics", level="DEBUG"
            ) as logs,
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"triton": heuristics},
            ),
        ):
            configs = compiler_seed_configs(env, MagicMock())

        self.assertEqual([config.config for config in configs], [{"block_sizes": [64]}])
        self.assertEqual(
            env.config_spec.autotuner_heuristics,
            [ValidAutotunerHeuristic.name, DuplicateAutotunerHeuristic.name],
        )
        self.assertIn(FailingAutotunerHeuristic.name, "\n".join(logs.output))
        self.assertIn("synthetic compiler seed failure", "\n".join(logs.output))

    def test_compiler_seed_configs_respects_disable_setting(self) -> None:
        class EnabledAutotunerHeuristic(AutotunerHeuristic):
            name = "enabled_autotuner_heuristic"
            backend = "triton"

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                raise AssertionError("disabled heuristics should not be queried")

        env = MagicMock()
        env.backend_name = "triton"
        env.config_spec = MagicMock()
        env.config_spec.autotuner_heuristics = ["stale"]
        env.settings = Settings(disable_autotuner_heuristics=True)

        with patch(
            "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
            {"triton": (EnabledAutotunerHeuristic,)},
        ):
            configs = compiler_seed_configs(env, MagicMock())

        self.assertEqual(configs, [])
        self.assertEqual(env.config_spec.autotuner_heuristics, [])

    def test_seed_flat_config_pairs_skips_invalid_compiler_seed(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.compiler_seed_configs = [
            helion.Config(block_sizes=["invalid"]),
            helion.Config(block_sizes=[64]),
        ]
        config_gen = spec.create_config_generation()
        messages: list[str] = []

        pairs = config_gen.seed_flat_config_pairs(messages.append)

        self.assertEqual(
            [config.config["block_sizes"] for _flat, config in pairs],
            [[64]],
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("Failed to transfer compiler seed config 1", messages[0])


class TestMatmulFacts(TestCase):
    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler matmul facts are not collected in ref eager mode")
    def test_matmul_facts_record_kernel_structure(self) -> None:
        @helion.kernel(backend="triton")
        def triton_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        @helion.kernel(backend="triton")
        def triton_matmul_epilogue(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
            return out

        @helion.kernel(backend="triton")
        def triton_two_matmuls(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc0 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                acc1 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc0 = torch.addmm(acc0, x[tile_m, tile_k], y[tile_k, tile_n])
                    acc1 = torch.addmm(acc1, x[tile_m, tile_k], z[tile_k, tile_n])
                out[tile_m, tile_n] = (acc0 + acc1).to(x.dtype)
            return out

        @helion.kernel(backend="triton")
        def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m] + y[tile_m]
            return out

        x = torch.empty([1024, 4096], device=DEVICE, dtype=HALF_DTYPE)
        y = torch.empty([4096, 8192], device=DEVICE, dtype=HALF_DTYPE)
        z = torch.empty([4096, 8192], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.empty([8192], device=DEVICE, dtype=HALF_DTYPE)
        add_x = torch.empty([1024], device=DEVICE, dtype=HALF_DTYPE)
        add_y = torch.empty([1024], device=DEVICE, dtype=HALF_DTYPE)

        cases = (
            ("gemm", triton_matmul, (x, y), 1),
            ("gemm_epilogue", triton_matmul_epilogue, (x, y, bias), 1),
            ("gemm_gemm", triton_two_matmuls, (x, y, z), 2),
            ("add", triton_add, (add_x, add_y), 0),
        )

        for name, kernel, args, expected_facts in cases:
            with (
                self.subTest(name=name),
                patch(
                    "helion._hardware.get_hardware_info",
                    return_value=HOPPER_HARDWARE,
                ),
            ):
                bound = kernel.bind(args)

            self.assertEqual(len(bound.config_spec.matmul_facts), expected_facts)
            if expected_facts == 0:
                self.assertEqual(bound.config_spec.compiler_seed_configs, [])
                self.assertEqual(bound.config_spec.autotuner_heuristics, [])
            for fact in bound.config_spec.matmul_facts:
                self.assertEqual(fact.lhs_ndim, 2)
                self.assertEqual(fact.rhs_ndim, 2)
                self.assertEqual(
                    (fact.static_m, fact.static_n, fact.static_k),
                    (1024, 8192, 4096),
                )
                self.assertIsNotNone(fact.m_block_id)
                self.assertIsNotNone(fact.n_block_id)
                self.assertIsNotNone(fact.k_block_id)
                self.assertEqual(fact.lhs_dtype, HALF_DTYPE)
                self.assertEqual(fact.rhs_dtype, HALF_DTYPE)


class TestTritonSkinnyGemmHeuristic(TestCase):
    def _make_triton_env_with_block_sizes(
        self,
        m_max: int = 8192,
        n_max: int = 8192,
        k_max: int = 8192,
    ) -> MagicMock:
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=m_max))
        spec.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=n_max))
        spec.block_sizes.append(BlockSizeSpec(block_id=2, size_hint=k_max))
        env = MagicMock()
        env.backend_name = "triton"
        env.config_spec = spec
        env.device = DEVICE
        env.settings = Settings()
        return env

    def _matmul_fact(
        self,
        static_m: int = 1024,
        static_n: int = 8192,
        static_k: int = 4096,
        *,
        lhs_ndim: int = 2,
        rhs_ndim: int = 2,
        m_block_id: int | None = 0,
        n_block_id: int | None = 1,
        k_block_id: int | None = 2,
    ) -> MatmulFact:
        return MatmulFact(
            lhs_ndim=lhs_ndim,
            rhs_ndim=rhs_ndim,
            m_block_id=m_block_id,
            n_block_id=n_block_id,
            k_block_id=k_block_id,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            lhs_dtype=HALF_DTYPE,
            rhs_dtype=HALF_DTYPE,
        )

    def test_triton_skinny_gemm_seed_eligibility_and_config(
        self,
    ) -> None:
        cases = (
            (
                "hopper",
                HOPPER_HARDWARE,
                [self._matmul_fact()],
                [[64, 64, 256]],
                [TritonSkinnyGemmHeuristic.name],
            ),
            (
                "mi350",
                MI350_HARDWARE,
                [self._matmul_fact()],
                [[64, 64, 256]],
                [TritonSkinnyGemmHeuristic.name],
            ),
            (
                "blackwell",
                BLACKWELL_HARDWARE,
                [self._matmul_fact()],
                [],
                [],
            ),
            (
                "balanced_shape",
                HOPPER_HARDWARE,
                [self._matmul_fact(static_m=4096, static_n=4096)],
                [],
                [],
            ),
            (
                "multiple_matmuls",
                HOPPER_HARDWARE,
                [self._matmul_fact(), self._matmul_fact()],
                [],
                [],
            ),
        )
        for name, hardware, facts, expected_block_sizes, expected_heuristics in cases:
            env = self._make_triton_env_with_block_sizes()
            env.config_spec.matmul_facts.extend(facts)
            with (
                self.subTest(name=name),
                patch(
                    "helion._hardware.get_hardware_info",
                    return_value=hardware,
                ),
            ):
                configs = compiler_seed_configs(env, MagicMock())

            self.assertEqual(
                [config.config["block_sizes"] for config in configs],
                expected_block_sizes,
            )
            self.assertEqual(
                env.config_spec.autotuner_heuristics,
                expected_heuristics,
            )

    def test_triton_skinny_gemm_seed_clamps_to_static_dims(self) -> None:
        env = self._make_triton_env_with_block_sizes(
            m_max=16,
            n_max=8192,
            k_max=128,
        )
        env.config_spec.matmul_facts.append(
            self._matmul_fact(static_m=16, static_n=8192, static_k=128)
        )

        config = TritonSkinnyGemmHeuristic.get_seed_config(env, MagicMock())

        assert config is not None
        self.assertEqual(config.config["block_sizes"], [16, 64, 128])

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler seed configs are not generated in ref eager mode")
    def test_triton_skinny_gemm_seed_in_initial_population(self) -> None:
        @helion.kernel(backend="triton")
        def triton_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        @helion.kernel(backend="triton")
        def triton_matmul_epilogue(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
            return out

        @helion.kernel(backend="triton")
        def triton_two_matmuls(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc0 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                acc1 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc0 = torch.addmm(acc0, x[tile_m, tile_k], y[tile_k, tile_n])
                    acc1 = torch.addmm(acc1, x[tile_m, tile_k], z[tile_k, tile_n])
                out[tile_m, tile_n] = (acc0 + acc1).to(x.dtype)
            return out

        x = torch.empty([1024, 4096], device=DEVICE, dtype=HALF_DTYPE)
        y = torch.empty([4096, 8192], device=DEVICE, dtype=HALF_DTYPE)
        z = torch.empty([4096, 8192], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.empty([8192], device=DEVICE, dtype=HALF_DTYPE)
        cases = (
            ("gemm", triton_matmul, (x, y), True),
            ("gemm_epilogue", triton_matmul_epilogue, (x, y, bias), True),
            ("gemm_gemm", triton_two_matmuls, (x, y, z), False),
        )
        seed_block_sizes = [64, 64, 256]

        def assert_skinny_gemm_seeded(configs: list[helion.Config]) -> None:
            self.assertIn(
                seed_block_sizes,
                [config.config["block_sizes"] for config in configs],
            )

        for name, kernel, args, expect_seed in cases:
            with (
                self.subTest(name=name),
                patch(
                    "helion._hardware.get_hardware_info",
                    return_value=HOPPER_HARDWARE,
                ),
            ):
                bound = kernel.bind(args)
                heuristic = TritonSkinnyGemmHeuristic

                config_gen = bound.config_spec.create_config_generation()
                compiler_seed_block_sizes = [
                    config.config["block_sizes"]
                    for config in bound.config_spec.compiler_seed_configs
                ]

                if expect_seed:
                    self.assertIn(
                        TritonSkinnyGemmHeuristic.name,
                        bound.config_spec.autotuner_heuristics,
                    )
                    self.assertTrue(
                        heuristic.is_eligible(bound.env, bound.host_function.device_ir)
                    )
                    seed_config = heuristic.get_seed_config(
                        bound.env, bound.host_function.device_ir
                    )
                    assert seed_config is not None
                    self.assertEqual(
                        seed_config.config["block_sizes"],
                        seed_block_sizes,
                    )
                    self.assertIn(seed_block_sizes, compiler_seed_block_sizes)
                    assert_skinny_gemm_seeded(config_gen.random_population(2))
                else:
                    self.assertFalse(
                        heuristic.is_eligible(bound.env, bound.host_function.device_ir)
                    )
                    self.assertNotIn(
                        TritonSkinnyGemmHeuristic.name,
                        bound.config_spec.autotuner_heuristics,
                    )


class TestCuteTcgen05ClusterM2Heuristic(TestCase):
    def _assert_cute_tcgen05_cluster_m2_seeded(
        self,
        configs: list[helion.Config],
        *,
        expected_block_k: int,
        expected_indexing_length: int,
    ) -> dict[str, object]:
        seeded = [
            config.config
            for config in configs
            if config.config["tcgen05_cluster_m"] == 2
        ]
        self.assertEqual(len(seeded), 1)
        seed = seeded[0]
        self.assertEqual(
            seed["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                expected_block_k,
            ],
        )
        self.assertEqual(
            seed["indexing"],
            ["tensor_descriptor"] * expected_indexing_length,
        )
        self.assertEqual(seed["pid_type"], "persistent_interleaved")
        self.assertEqual(seed["tcgen05_num_epi_warps"], 4)
        return seed

    def _assert_cute_tcgen05_edge_k_tail_seed_overrides(
        self,
        config: dict[str, object],
        *,
        expected_l2_swizzle_size: int = TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE,
    ) -> None:
        self.assertEqual(
            config["tcgen05_ab_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
        )
        self.assertEqual(
            config["tcgen05_acc_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES,
        )
        self.assertEqual(
            config["tcgen05_c_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES,
        )
        self.assertEqual(
            config["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING],
        )
        self.assertEqual(
            config["tcgen05_l2_swizzle_size"],
            expected_l2_swizzle_size,
        )
        self.assertEqual(
            config[TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY],
            TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
        )
        self.assertEqual(
            config[TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY],
            TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m2_seed_heuristic(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)

            heuristic = CuteTcgen05ClusterM2Heuristic
            self.assertIn(
                CuteTcgen05ClusterM2Heuristic.name,
                bound.config_spec.autotuner_heuristics,
            )
            self.assertTrue(
                heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            )
            seed_config = heuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
            assert seed_config is not None
            self._assert_cute_tcgen05_cluster_m2_seeded(
                [seed_config],
                expected_block_k=128,
                expected_indexing_length=3,
            )
            self.assertEqual(
                seed_config.config["l2_groupings"], [TCGEN05_TWO_CTA_SEED_L2_GROUPING]
            )

        with patch_cute_mma_support(default_cute_mma_support(tcgen05_f16bf16=False)):
            unsupported_args = (
                torch.empty([2048, 2048], device=DEVICE, dtype=HALF_DTYPE),
                torch.empty([2048, 2048], device=DEVICE, dtype=HALF_DTYPE),
            )
            unsupported_bound = cute_matmul_mma.bind(unsupported_args)
            self.assertFalse(
                heuristic.is_eligible(
                    unsupported_bound.env,
                    unsupported_bound.host_function.device_ir,
                )
            )
            self.assertNotIn(
                CuteTcgen05ClusterM2Heuristic.name,
                unsupported_bound.config_spec.autotuner_heuristics,
            )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m2_edge_k_tail_bk_requires_tail(self) -> None:
        valid_tail = Tcgen05ClusterM2SearchConstraints(
            static_k=5000,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
            allow_edge_k_tail_family=True,
        )
        self.assertTrue(
            CuteTcgen05Config.cluster_m2_bk_is_valid(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                valid_tail,
            )
        )
        for static_k in (64, 128, 256):
            with self.subTest(static_k=static_k):
                constraints = Tcgen05ClusterM2SearchConstraints(
                    static_k=static_k,
                    max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
                    allow_edge_k_tail_family=True,
                )
                self.assertFalse(
                    CuteTcgen05Config.cluster_m2_bk_is_valid(
                        TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                        constraints,
                    )
                )

        k_fragment = MagicMock()
        k_fragment.low = 16
        k_fragment.high = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
        k_block = MagicMock()
        k_block._fragment.return_value = k_fragment
        spec = MagicMock()
        spec.block_sizes = [MagicMock(), MagicMock(), k_block]
        spec._tcgen05_cluster_m2_bk_is_valid.side_effect = (
            CuteTcgen05Config.cluster_m2_bk_is_valid
        )
        spec._tcgen05_cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=128,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
            allow_edge_k_tail_family=True,
        )
        env = MagicMock()
        env.config_spec = spec
        self.assertIsNone(CuteTcgen05ClusterM2Heuristic._select_bk(env))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m2_seed_heuristic_for_edge_k_tail_family(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_bias_residual_gelu(
            x: torch.Tensor,
            y: torch.Tensor,
            bias: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.nn.functional.gelu(
                    1.25 * acc + 0.5 * residual[tile_m, tile_n] + bias[tile_n],
                    approximate="tanh",
                ).to(x.dtype)
            return out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_bias_residual_gelu.bind(args)

        spec = bound.config_spec
        self.assertIn(CuteTcgen05ClusterM2Heuristic.name, spec.autotuner_heuristics)
        constraints = spec._tcgen05_cluster_m2_search_constraints
        assert constraints is not None
        self.assertTrue(constraints.allow_edge_k_tail_family)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        direct_seed = CuteTcgen05ClusterM2Heuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        ).config
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(direct_seed)
        raw_seeded = [
            config.config
            for config in spec.compiler_seed_configs
            if config.config.get("tcgen05_cluster_m") == 2
        ]
        self.assertEqual(len(raw_seeded), 2)
        raw_seed = next(
            config
            for config in raw_seeded
            if config.get("tcgen05_strategy")
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        raw_scheduler_seed = next(
            config
            for config in raw_seeded
            if config.get("tcgen05_strategy")
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        self.assertEqual(
            raw_seed["block_sizes"],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(raw_seed["pid_type"], "persistent_interleaved")
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(raw_seed)
        self.assertEqual(
            raw_seed["indexing"], ["tensor_descriptor"] * spec.indexing.length
        )
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(
            raw_scheduler_seed,
            expected_l2_swizzle_size=(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            ),
        )
        self.assertEqual(raw_scheduler_seed["tcgen05_warp_spec_scheduler_warps"], 1)
        self.assertEqual(raw_scheduler_seed["tcgen05_warp_spec_c_input_warps"], 1)

        c_input_seeds = [
            config.config
            for config in spec.autotune_seed_configs()
            if config.config.get("tcgen05_warp_spec_c_input_warps") == 1
        ]
        self.assertEqual(len(c_input_seeds), 1)
        c_input_seed = c_input_seeds[0]
        self.assertEqual(
            c_input_seed["tcgen05_strategy"],
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
        )
        self.assertEqual(c_input_seed["tcgen05_warp_spec_scheduler_warps"], 1)
        self.assertEqual(c_input_seed["tcgen05_warp_spec_c_input_warps"], 1)
        self.assertEqual(
            c_input_seed["block_sizes"],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(
            c_input_seed,
            expected_l2_swizzle_size=(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            ),
        )
        self.assertEqual(
            c_input_seed["indexing"], ["tensor_descriptor"] * spec.indexing.length
        )

        config_gen = spec.create_config_generation()
        seed_pairs = config_gen.seed_flat_config_pairs()
        self.assertEqual(len(seed_pairs), 2)
        normalized_seeds = [normalized.config for _flat, normalized in seed_pairs]
        normalized_seed = next(
            config
            for config in normalized_seeds
            if config["tcgen05_strategy"]
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        normalized_scheduler_seed = next(
            config
            for config in normalized_seeds
            if config["tcgen05_strategy"]
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        for flat_seed, _normalized_seed in seed_pairs:
            config_gen.encode_config(flat_seed)
        self.assertEqual(normalized_seed["pid_type"], "persistent_interleaved")
        self.assertEqual(
            normalized_seed["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(normalized_seed)
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(
            normalized_scheduler_seed,
            expected_l2_swizzle_size=(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            ),
        )

        configs = config_gen.random_population(2)
        self.assertEqual(configs[0].config["tcgen05_cluster_m"], 1)
        population_seed = self._assert_cute_tcgen05_cluster_m2_seeded(
            configs,
            expected_block_k=TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            expected_indexing_length=spec.indexing.length,
        )
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(population_seed)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_c_input_seed_respects_disable_heuristics(self) -> None:
        @helion.kernel(backend="cute", disable_autotuner_heuristics=True)
        def cute_matmul_bias_residual_gelu(
            x: torch.Tensor,
            y: torch.Tensor,
            bias: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.nn.functional.gelu(
                    1.25 * acc + 0.5 * residual[tile_m, tile_n] + bias[tile_n],
                    approximate="tanh",
                ).to(x.dtype)
            return out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_bias_residual_gelu.bind(args)

        self.assertTrue(bound.config_spec.cute_tcgen05_aux_kernel_detected)
        self.assertEqual(bound.config_spec.compiler_seed_configs, [])
        self.assertEqual(bound.config_spec.autotuner_heuristics, [])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m2_edge_ab2_seed_ignores_ab3_budget(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_bias_residual_gelu(
            x: torch.Tensor,
            y: torch.Tensor,
            bias: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.nn.functional.gelu(
                    1.25 * acc + 0.5 * residual[tile_m, tile_n] + bias[tile_n],
                    approximate="tanh",
                ).to(x.dtype)
            return out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_bias_residual_gelu.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        config_dict: dict[str, object] = {
            "block_sizes": [128, 128, 64],
            "indexing": ["tensor_descriptor"] * spec.indexing.length,
            "l2_groupings": [TCGEN05_TWO_CTA_SEED_L2_GROUPING],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
            "tcgen05_ab_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
            "tcgen05_acc_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES,
            "tcgen05_c_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES,
        }
        with patch.object(
            spec._cute_tcgen05_config,
            "ab_stages_three_fits",
            return_value=False,
        ):
            spec._cute_tcgen05_config.fix_search_config(config_dict)

        self.assertEqual(config_dict["tcgen05_cluster_m"], 2)
        self.assertEqual(config_dict["pid_type"], "persistent_interleaved")
        self.assertEqual(
            config_dict["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(
            config_dict["tcgen05_ab_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
        )
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(config_dict)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_seeded_in_initial_populations(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        self.assertIn(
            CuteTcgen05ClusterM2Heuristic.name,
            bound.config_spec.autotuner_heuristics,
        )

        config_gen = bound.config_spec.create_config_generation()
        zero_flat = config_gen.random_population_flat(0)
        self.assertEqual(len(zero_flat), 1)
        zero_config = config_gen.unflatten(zero_flat[0])
        self.assertEqual(zero_config.config["tcgen05_cluster_m"], 1)
        one_flat = config_gen.random_population_flat(1)
        self.assertEqual(len(one_flat), 1)
        one_config = config_gen.unflatten(one_flat[0])
        self.assertEqual(one_config.config["tcgen05_cluster_m"], 1)
        one_config_population = config_gen.random_population(1)
        self.assertEqual(len(one_config_population), 1)
        self.assertEqual(one_config_population[0].config["tcgen05_cluster_m"], 1)
        self._assert_cute_tcgen05_cluster_m2_seeded(
            config_gen.random_population(2),
            expected_block_k=128,
            expected_indexing_length=3,
        )

        acf_config_gen = bound.config_spec.create_config_generation(
            advanced_controls_files=["/tmp/helion-test.acf"]
        )
        acf_configs = acf_config_gen.random_population(2)
        # Future heuristics may add more compiler seeds; this test only
        # requires the CuTe cluster-m2 seed to be present.
        self.assertGreaterEqual(len(acf_configs), 2)
        self.assertEqual(
            {config.config["advanced_controls_file"] for config in acf_configs},
            {"/tmp/helion-test.acf"},
        )
        self._assert_cute_tcgen05_cluster_m2_seeded(
            acf_configs,
            expected_block_k=128,
            expected_indexing_length=3,
        )

        with patch.object(
            PatternSearch, "_find_similar_cached_configs", return_value=[]
        ):
            search = PatternSearch(
                bound,
                args,
                initial_population=30,
                initial_population_strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                best_available_pad_random=False,
            )
            configs = [
                search.config_gen.unflatten(flat)
                for flat in search._generate_initial_population_flat()
            ]
        # Future heuristics may add more compiler seeds; this test only
        # requires the CuTe cluster-m2 seed to be present.
        self.assertGreaterEqual(len(configs), 2)
        self._assert_cute_tcgen05_cluster_m2_seeded(
            configs,
            expected_block_k=128,
            expected_indexing_length=3,
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_seed_indexing_matches_live_spec(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_epilogue(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma_epilogue.bind(args)
        self.assertGreater(bound.config_spec.indexing.length, 3)

        configs = bound.config_spec.create_config_generation().random_population(2)
        seeded = [
            config.config
            for config in configs
            if config.config["tcgen05_cluster_m"] == 2
        ]
        self.assertEqual(len(seeded), 1)
        self.assertEqual(
            len(seeded[0]["indexing"]),
            bound.config_spec.indexing.length,
        )
