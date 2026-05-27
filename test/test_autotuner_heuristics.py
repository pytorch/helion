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
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_STRATEGY_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from helion._compiler.cute.strategies import Tcgen05LayoutStrategy
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_AB_STAGES
from helion._compiler.cute.tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_BLOCK_K
from helion._compiler.cute.tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_C_STAGES
from helion._compiler.cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
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
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING,
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
from helion.autotuner import IntegerFragment
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

    def test_default_config_promotes_compiler_seed(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.compiler_default_config = helion.Config(
            block_sizes=[64], num_warps=8, num_stages=2
        )

        default = spec.default_config()
        config_gen = spec.create_config_generation()
        flat_default = config_gen.unflatten(config_gen.default_flat())

        self.assertEqual(default.config["block_sizes"], [64])
        self.assertEqual(default.config["num_warps"], 8)
        self.assertEqual(default.config["num_stages"], 2)
        self.assertEqual(flat_default.config["block_sizes"], [64])
        self.assertEqual(flat_default.config["num_warps"], 8)
        self.assertEqual(flat_default.config["num_stages"], 2)


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
        expected_l2_grouping: int = TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING,
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
            [expected_l2_grouping],
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

    def _expected_clc_aux_tma_range_knobs(
        self, spec: ConfigSpec
    ) -> tuple[list[bool | None], list[bool | None], list[bool | None]]:
        self.assertEqual(len(spec.matmul_facts), 1)
        k_block_id = spec.matmul_facts[0].k_block_id
        assert k_block_id is not None
        k_range_index = spec.range_flattens.block_id_to_index(k_block_id)
        self.assertEqual(
            k_range_index,
            spec.range_multi_buffers.block_id_to_index(k_block_id),
        )
        self.assertEqual(
            k_range_index,
            spec.range_warp_specialize.block_id_to_index(k_block_id),
        )
        range_flattens: list[bool | None] = [None for _ in spec.range_flattens]
        range_multi_buffers: list[bool | None] = [
            None for _ in spec.range_multi_buffers
        ]
        range_warp_specializes: list[bool | None] = [
            None for _ in spec.range_warp_specialize
        ]
        range_flattens[k_range_index] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN
        )
        range_multi_buffers[k_range_index] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER
        )
        range_warp_specializes[k_range_index] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE
        )
        return range_flattens, range_multi_buffers, range_warp_specializes

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
    def test_cute_tcgen05_target1_tvm_ffi_seed_config(self) -> None:
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
            torch.empty([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
        ):
            bound = cute_matmul_mma.bind(args)
        tvm_ffi_seeds = [
            config.config
            for config in bound.config_spec.autotune_seed_configs()
            if config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
        ]
        self.assertEqual(len(tvm_ffi_seeds), 1)
        seed = tvm_ffi_seeds[0]
        self.assertEqual(
            seed["block_sizes"],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET1_TVM_FFI_BLOCK_K,
            ],
        )
        self.assertEqual(seed["num_warps"], 8)
        self.assertEqual(seed["pid_type"], "persistent_interleaved")
        self.assertEqual(seed["tcgen05_cluster_m"], 2)
        self.assertEqual(seed["tcgen05_cluster_n"], 1)
        self.assertEqual(seed["tcgen05_ab_stages"], TCGEN05_TARGET1_TVM_FFI_AB_STAGES)
        self.assertEqual(seed["tcgen05_c_stages"], TCGEN05_TARGET1_TVM_FFI_C_STAGES)
        self.assertEqual(seed["tcgen05_l2_swizzle_size"], 1)
        search_ab_stages_fragment = bound.config_spec._tcgen05_optional_fragments(
            for_search=True
        )["tcgen05_ab_stages"]
        self.assertIsInstance(search_ab_stages_fragment, IntegerFragment)
        self.assertEqual(search_ab_stages_fragment.high, 2)
        self.assertIn(
            TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY,
            bound.config_spec._tcgen05_optional_fragments(),
        )
        self.assertNotIn(
            TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY,
            bound.config_spec._tcgen05_optional_fragments(for_search=True),
        )
        self.assertEqual(
            seed[TCGEN05_STRATEGY_CONFIG_KEY],
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
        )
        self.assertIsNone(seed.get(TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY))
        self.assertEqual(
            seed[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY],
            Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
        )
        self.assertEqual(seed[TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY], 128)
        self.assertEqual(seed[TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY], 32)
        self.assertEqual(seed[TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY], 32)
        self.assertIs(seed[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY], True)
        self.assertEqual(seed["range_unroll_factors"], [1, 1])

        config_gen = bound.config_spec.create_config_generation()
        transferred_tvm_ffi_seeds = [
            config.config
            for _flat, config in config_gen.seed_flat_config_pairs()
            if config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
        ]
        self.assertEqual(len(transferred_tvm_ffi_seeds), 1)
        transferred_seed = transferred_tvm_ffi_seeds[0]
        self.assertEqual(
            transferred_seed[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY],
            Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
        )
        self.assertEqual(transferred_seed[TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY], 128)
        self.assertEqual(transferred_seed[TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY], 32)
        self.assertEqual(
            transferred_seed[TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY], 32
        )
        self.assertIs(transferred_seed[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY], True)

        bound.config_spec.user_defined_tunables["target1_extra"] = IntegerFragment(
            1, 4, 2
        )
        projected_config = helion.Config(
            block_sizes=[128, 32, 64],
            indexing=["pointer", "pointer", "tensor_descriptor"],
            pid_type="flat",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_tvm_ffi_launch=True,
            epilogue_subtile=2,
            tcgen05_diagnostic_invalid_output=True,
            tcgen05_c_store_mode=TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE,
            tcgen05_acc_producer_mode=TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA,
            tcgen05_aux_load_mode=TCGEN05_AUX_LOAD_MODE_TMA,
            tcgen05_cubin_lineinfo=True,
            tcgen05_epilogue_layout=TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R,
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_warp_spec_c_input_warps=1,
            tcgen05_layout_overrides_smem_swizzle_a=128,
            tcgen05_layout_overrides_smem_swizzle_b=64,
            advanced_controls_file="/tmp/helion-test.acf",
            target1_extra=3,
        )
        bound.config_spec.normalize(projected_config, _fix_invalid=True)
        expected_seed_config = helion.Config(
            **seed,
            advanced_controls_file="/tmp/helion-test.acf",
            target1_extra=3,
        )
        bound.config_spec.normalize(expected_seed_config, _fix_invalid=True)
        self.assertEqual(projected_config.config, expected_seed_config.config)
        self.assertEqual(
            projected_config.config["block_sizes"],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET1_TVM_FFI_BLOCK_K,
            ],
        )
        self.assertIs(projected_config.config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY], True)
        self.assertEqual(
            projected_config.config["tcgen05_ab_stages"],
            TCGEN05_TARGET1_TVM_FFI_AB_STAGES,
        )
        self.assertEqual(
            projected_config.config["tcgen05_c_stages"],
            TCGEN05_TARGET1_TVM_FFI_C_STAGES,
        )
        self.assertIs(
            projected_config.config[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY],
            True,
        )
        self.assertEqual(
            projected_config.config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY],
            Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
        )
        for stale_key in (
            "epilogue_subtile",
            TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
            TCGEN05_C_STORE_MODE_CONFIG_KEY,
            TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
            TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
            TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        ):
            self.assertNotIn(stale_key, projected_config.config)
        self.assertEqual(
            projected_config.config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY], 0
        )
        self.assertEqual(
            projected_config.config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY], 0
        )
        self.assertIsNone(
            projected_config.config[TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY]
        )
        self.assertIsNone(
            projected_config.config[TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY]
        )

        projected_cluster_m2_config = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
        )
        bound.config_spec.normalize(projected_cluster_m2_config, _fix_invalid=True)
        self.assertIs(
            projected_cluster_m2_config.config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY],
            True,
        )
        self.assertEqual(
            projected_cluster_m2_config.config["block_sizes"],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET1_TVM_FFI_BLOCK_K,
            ],
        )

        for requested_ab_stages in (4, 5, 6):
            with self.subTest(requested_ab_stages=requested_ab_stages):
                non_seed_stage_config = helion.Config(
                    block_sizes=[256, 256, 64],
                    indexing=[
                        "tensor_descriptor",
                        "tensor_descriptor",
                        "tensor_descriptor",
                    ],
                    pid_type="persistent_interleaved",
                    tcgen05_cluster_m=1,
                    tcgen05_cluster_n=1,
                    tcgen05_ab_stages=requested_ab_stages,
                )
                bound.config_spec.normalize(non_seed_stage_config, _fix_invalid=True)
                self.assertEqual(non_seed_stage_config.config["tcgen05_ab_stages"], 3)

                invalid_non_seed_stage_config = helion.Config(
                    block_sizes=[256, 256, 64],
                    indexing=[
                        "tensor_descriptor",
                        "tensor_descriptor",
                        "tensor_descriptor",
                    ],
                    pid_type="persistent_interleaved",
                    tcgen05_cluster_m=1,
                    tcgen05_cluster_n=1,
                    tcgen05_ab_stages=requested_ab_stages,
                )
                with self.assertRaisesRegex(
                    helion.exc.InvalidConfig, "validated Target1 TVM-FFI seed"
                ):
                    bound.config_spec.normalize(invalid_non_seed_stage_config)

        non_target_args = (
            torch.empty([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024, 2048], device=DEVICE, dtype=torch.bfloat16),
        )
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
        ):
            non_target_bound = cute_matmul_mma.bind(non_target_args)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
                for config in non_target_bound.config_spec.autotune_seed_configs()
            )
        )
        non_target_ab_stages_fragment = (
            non_target_bound.config_spec._tcgen05_optional_fragments(for_search=True)[
                "tcgen05_ab_stages"
            ]
        )
        self.assertIsInstance(non_target_ab_stages_fragment, IntegerFragment)
        self.assertEqual(non_target_ab_stages_fragment.high, 2)

        @helion.kernel(backend="cute")
        def cute_matmul_mma_no_ab3_budget(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
            patch.object(
                CuteTcgen05Config,
                "per_cta_ab_smem_budget_bytes",
                return_value=0,
            ),
        ):
            no_ab3_budget_bound = cute_matmul_mma_no_ab3_budget.bind(args)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
                for config in no_ab3_budget_bound.config_spec.autotune_seed_configs()
            )
        )
        no_budget_ab_stages_fragment = (
            no_ab3_budget_bound.config_spec._tcgen05_optional_fragments(
                for_search=True
            )["tcgen05_ab_stages"]
        )
        self.assertIsInstance(no_budget_ab_stages_fragment, IntegerFragment)
        self.assertEqual(no_budget_ab_stages_fragment.high, 2)

        @helion.kernel(backend="cute")
        def cute_matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.relu(acc).to(x.dtype)
            return out

        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
        ):
            relu_bound = cute_matmul_relu.bind(args)
        self.assertFalse(
            relu_bound.config_spec.cute_tcgen05_identity_matmul_store_detected
        )
        self.assertEqual(relu_bound.config_spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertFalse(
            any(
                config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
                for config in relu_bound.config_spec.autotune_seed_configs()
            )
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
        self.assertTrue(
            CuteTcgen05Config.cluster_m2_bk_is_valid(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
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
        with (
            patch_cute_mma_support(),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            bound = cute_matmul_bias_residual_gelu.bind(args)

        spec = bound.config_spec
        (
            expected_clc_aux_tma_range_flattens,
            expected_clc_aux_tma_range_multi_buffers,
            expected_clc_aux_tma_range_warp_specializes,
        ) = self._expected_clc_aux_tma_range_knobs(spec)
        self.assertIn(CuteTcgen05ClusterM2Heuristic.name, spec.autotuner_heuristics)
        self.assertTrue(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        constraints = spec._tcgen05_cluster_m2_search_constraints
        assert constraints is not None
        self.assertTrue(constraints.allow_edge_k_tail_family)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        flat_keys = {key for key, _count, _is_sequence in spec.flat_key_layout()}
        self.assertIn(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, flat_keys)
        self.assertIn(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, flat_keys)
        direct_seed = CuteTcgen05ClusterM2Heuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        ).config
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(direct_seed)
        raw_seeded = [
            config.config
            for config in spec.compiler_seed_configs
            if config.config.get("tcgen05_cluster_m") == 2
        ]
        self.assertEqual(len(raw_seeded), 6)
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
            and TCGEN05_AUX_LOAD_MODE_CONFIG_KEY not in config
        )
        raw_aux_tma_seed = next(
            config
            for config in raw_seeded
            if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
        )
        raw_clc_seeds = [
            config
            for config in raw_seeded
            if config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.CLC_PERSISTENT.value
        ]
        self.assertEqual(len(raw_clc_seeds), 3)
        raw_wide_clc_aux_tma_seed = next(
            config
            for config in raw_clc_seeds
            if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
            and config["block_sizes"][1] == TCGEN05_TWO_CTA_BLOCK_N
        )
        raw_narrow_clc_aux_tma_seed = next(
            config
            for config in raw_clc_seeds
            if config["block_sizes"][1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
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
        self.assertEqual(
            raw_aux_tma_seed["tcgen05_strategy"],
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
        )
        self.assertEqual(raw_aux_tma_seed["tcgen05_warp_spec_scheduler_warps"], 1)
        self.assertEqual(raw_aux_tma_seed["tcgen05_warp_spec_c_input_warps"], 1)
        self.assertEqual(
            raw_aux_tma_seed[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            TCGEN05_AUX_LOAD_MODE_TMA,
        )

        c_input_seeds = [
            config.config
            for config in spec.autotune_seed_configs()
            if config.config.get("tcgen05_warp_spec_c_input_warps") == 1
        ]
        self.assertEqual(len(c_input_seeds), 5)
        c_input_seed = next(
            seed
            for seed in c_input_seeds
            if seed.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA
        )
        aux_tma_seed = next(
            seed
            for seed in c_input_seeds
            if seed.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
        )
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
        self.assertEqual(aux_tma_seed["block_sizes"], c_input_seed["block_sizes"])
        self.assertEqual(aux_tma_seed["pid_type"], c_input_seed["pid_type"])
        self.assertEqual(
            aux_tma_seed[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY], TCGEN05_AUX_LOAD_MODE_TMA
        )
        self.assertEqual(
            raw_wide_clc_aux_tma_seed["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING],
        )
        self.assertEqual(
            raw_wide_clc_aux_tma_seed["tcgen05_acc_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES,
        )
        self.assertEqual(
            raw_wide_clc_aux_tma_seed["range_flattens"],
            expected_clc_aux_tma_range_flattens,
        )
        self.assertEqual(
            raw_wide_clc_aux_tma_seed["range_multi_buffers"],
            expected_clc_aux_tma_range_multi_buffers,
        )
        self.assertEqual(
            raw_wide_clc_aux_tma_seed["range_warp_specializes"],
            expected_clc_aux_tma_range_warp_specializes,
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed["block_sizes"],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
            ],
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING],
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed["tcgen05_acc_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES,
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed["range_flattens"],
            expected_clc_aux_tma_range_flattens,
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed["range_multi_buffers"],
            expected_clc_aux_tma_range_multi_buffers,
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed["range_warp_specializes"],
            expected_clc_aux_tma_range_warp_specializes,
        )
        self.assertEqual(
            raw_narrow_clc_aux_tma_seed[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            TCGEN05_AUX_LOAD_MODE_TMA,
        )
        self.assertEqual(
            {
                seed.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, "simt")
                for seed in raw_clc_seeds
            },
            {"simt", TCGEN05_AUX_LOAD_MODE_TMA},
        )

        config_gen = spec.create_config_generation()
        seed_pairs = config_gen.seed_flat_config_pairs()
        self.assertEqual(len(seed_pairs), 6)
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
            and config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
            != TCGEN05_AUX_LOAD_MODE_TMA
        )
        normalized_aux_tma_seed = next(
            config
            for config in normalized_seeds
            if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
        )
        normalized_clc_seeds = [
            config
            for config in normalized_seeds
            if config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.CLC_PERSISTENT.value
        ]
        self.assertEqual(len(normalized_clc_seeds), 3)
        normalized_wide_clc_aux_tma_seed = next(
            config
            for config in normalized_clc_seeds
            if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
            and config["block_sizes"][1] == TCGEN05_TWO_CTA_BLOCK_N
        )
        projected_wide_clc_aux_tma_config = dict(raw_wide_clc_aux_tma_seed)
        projected_wide_clc_aux_tma_config["block_sizes"] = [128, 64, 64]
        projected_wide_clc_aux_tma_config["pid_type"] = "flat"
        projected_wide_clc_aux_tma_config["tcgen05_acc_stages"] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES
        )
        projected_wide_clc_aux_tma_config["l2_groupings"] = [
            TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING
        ]
        projected_wide_clc_aux_tma_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
        for key in (
            "range_flattens",
            "range_multi_buffers",
            "range_warp_specializes",
        ):
            projected_wide_clc_aux_tma_config.pop(key, None)
        spec._cute_tcgen05_config.fix_search_config(projected_wide_clc_aux_tma_config)
        self.assertEqual(
            projected_wide_clc_aux_tma_config["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING],
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config["tcgen05_acc_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES,
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY],
            1,
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY],
            1,
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config["range_flattens"],
            expected_clc_aux_tma_range_flattens,
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config["range_multi_buffers"],
            expected_clc_aux_tma_range_multi_buffers,
        )
        self.assertEqual(
            projected_wide_clc_aux_tma_config["range_warp_specializes"],
            expected_clc_aux_tma_range_warp_specializes,
        )
        for flat_seed, _normalized_seed in seed_pairs:
            config_gen.encode_config(flat_seed)
        persistence_indices, _ = config_gen._key_to_flat_indices[
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
        ]
        legacy_scheduler_seed = dict(raw_scheduler_seed)
        legacy_scheduler_seed.pop(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, None)
        legacy_flat = config_gen.flatten(helion.Config(**legacy_scheduler_seed))
        legacy_normalized = config_gen.unflatten([*legacy_flat]).config
        self.assertEqual(
            legacy_flat[persistence_indices[0]],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        self.assertEqual(
            legacy_normalized["tcgen05_strategy"],
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
        )
        self.assertEqual(legacy_normalized["tcgen05_warp_spec_scheduler_warps"], 1)
        self.assertEqual(legacy_normalized["tcgen05_warp_spec_c_input_warps"], 1)
        self.assertEqual(
            legacy_normalized[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        legacy_minimal_scheduler_seed = dict(raw_scheduler_seed)
        legacy_minimal_scheduler_seed.pop("pid_type", None)
        legacy_minimal_scheduler_seed.pop(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, None)
        legacy_minimal_flat = config_gen.flatten(
            helion.Config(**legacy_minimal_scheduler_seed)
        )
        legacy_minimal_normalized = config_gen.unflatten([*legacy_minimal_flat]).config
        self.assertEqual(
            legacy_minimal_flat[persistence_indices[0]],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        self.assertEqual(
            legacy_minimal_normalized["tcgen05_strategy"],
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
        )
        self.assertEqual(
            legacy_minimal_normalized["pid_type"], "persistent_interleaved"
        )
        self.assertEqual(
            legacy_minimal_normalized[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        invalid_pid_seed_flat = config_gen.flatten(
            helion.Config(pid_type="not_a_valid_pid_type")
        )
        self.assertEqual(
            invalid_pid_seed_flat[persistence_indices[0]],
            Tcgen05PersistenceModel.NON_PERSISTENT.value,
        )
        pid_override_gen = spec.create_config_generation(
            overrides={"pid_type": "persistent_interleaved"}
        )
        pid_override_config = pid_override_gen.unflatten(
            pid_override_gen.default_flat()
        ).config
        self.assertEqual(pid_override_config["pid_type"], "persistent_interleaved")
        self.assertEqual(
            pid_override_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        clc_flat_seed = next(
            flat
            for flat, normalized in seed_pairs
            if normalized.config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )
        pid_override_clc_config = pid_override_gen.unflatten([*clc_flat_seed]).config
        self.assertEqual(pid_override_clc_config["pid_type"], "persistent_interleaved")
        self.assertEqual(
            pid_override_clc_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )
        explicit_bad_override_gen = spec.create_config_generation(
            overrides={
                "pid_type": "persistent_interleaved",
                TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                    Tcgen05PersistenceModel.NON_PERSISTENT.value
                ),
            }
        )
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            "contradicts pid_type='persistent_interleaved'",
        ):
            explicit_bad_override_gen.unflatten(
                explicit_bad_override_gen.default_flat()
            )
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
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(
            normalized_aux_tma_seed,
            expected_l2_swizzle_size=(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            ),
        )
        self.assertEqual(
            normalized_aux_tma_seed[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            TCGEN05_AUX_LOAD_MODE_TMA,
        )
        self.assertEqual(
            normalized_wide_clc_aux_tma_seed["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING],
        )
        self.assertEqual(
            normalized_wide_clc_aux_tma_seed["tcgen05_acc_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES,
        )
        self.assertEqual(
            normalized_wide_clc_aux_tma_seed["range_flattens"],
            expected_clc_aux_tma_range_flattens,
        )
        self.assertEqual(
            normalized_wide_clc_aux_tma_seed["range_multi_buffers"],
            expected_clc_aux_tma_range_multi_buffers,
        )
        self.assertEqual(
            normalized_wide_clc_aux_tma_seed["range_warp_specializes"],
            expected_clc_aux_tma_range_warp_specializes,
        )
        normalized_narrow_clc_aux_tma_seed = next(
            config
            for config in normalized_clc_seeds
            if config["block_sizes"][1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
            ],
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING],
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed["tcgen05_acc_stages"],
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES,
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed["range_flattens"],
            expected_clc_aux_tma_range_flattens,
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed["range_multi_buffers"],
            expected_clc_aux_tma_range_multi_buffers,
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed["range_warp_specializes"],
            expected_clc_aux_tma_range_warp_specializes,
        )
        self.assertEqual(
            normalized_narrow_clc_aux_tma_seed[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            TCGEN05_AUX_LOAD_MODE_TMA,
        )
        self.assertEqual(
            {
                seed.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, "simt")
                for seed in normalized_clc_seeds
            },
            {"simt", TCGEN05_AUX_LOAD_MODE_TMA},
        )
        for seed in normalized_clc_seeds:
            self.assertEqual(
                seed["tcgen05_strategy"],
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            )
            self.assertEqual(seed["tcgen05_warp_spec_scheduler_warps"], 1)
            self.assertEqual(seed["tcgen05_warp_spec_c_input_warps"], 1)
            self.assertEqual(seed["tcgen05_cluster_m"], 2)
            self.assertEqual(seed["tcgen05_cluster_n"], 1)
            self.assertEqual(
                seed["indexing"], ["tensor_descriptor"] * spec.indexing.length
            )

        configs = config_gen.random_population(7)
        self.assertEqual(configs[0].config["tcgen05_cluster_m"], 1)
        cluster_m2_population = [
            config.config
            for config in configs
            if config.config["tcgen05_cluster_m"] == 2
        ]
        self.assertEqual(len(cluster_m2_population), 6)
        self.assertTrue(
            any(
                config["block_sizes"][1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
                for config in cluster_m2_population
            )
        )
        population_seed = next(
            config
            for config in cluster_m2_population
            if config.get("tcgen05_strategy")
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        self.assertEqual(
            population_seed["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(population_seed["pid_type"], "persistent_interleaved")
        self.assertEqual(population_seed["tcgen05_num_epi_warps"], 4)
        self.assertEqual(
            population_seed["indexing"],
            ["tensor_descriptor"] * spec.indexing.length,
        )
        self._assert_cute_tcgen05_edge_k_tail_seed_overrides(population_seed)
        self.assertTrue(
            any(
                config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                == Tcgen05PersistenceModel.CLC_PERSISTENT.value
                for config in cluster_m2_population
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_clc_search_normalizes_valid_and_invalid_cases(
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
        with (
            patch_cute_mma_support(),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            bound = cute_matmul_bias_residual_gelu.bind(args)
        with (
            patch_cute_mma_support(),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            sm90_bound = cute_matmul_bias_residual_gelu.bind(args)
        self.assertIsNot(sm90_bound, bound)
        (
            expected_clc_aux_tma_range_flattens,
            expected_clc_aux_tma_range_multi_buffers,
            expected_clc_aux_tma_range_warp_specializes,
        ) = self._expected_clc_aux_tma_range_knobs(bound.config_spec)

        valid_config: dict[str, object] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
            "indexing": ["tensor_descriptor"] * bound.config_spec.indexing.length,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_strategy": Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_warp_spec_c_input_warps": 1,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.CLC_PERSISTENT.value
            ),
        }
        bound.config_spec.normalize(valid_config, _fix_invalid=True)
        self.assertEqual(
            valid_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )

        def make_minimal_preprojection_clc_aux_tma_config() -> dict[str, object]:
            return {
                "block_sizes": [128, 64, 64],
                "indexing": ["tensor_descriptor"] * bound.config_spec.indexing.length,
                "pid_type": "flat",
                "tcgen05_cluster_m": 2,
                "tcgen05_cluster_n": 1,
                "tcgen05_strategy": Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
                TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 0,
                TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: TCGEN05_AUX_LOAD_MODE_TMA,
                TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value
                ),
            }

        minimal_preprojection_clc_aux_tma_config = (
            make_minimal_preprojection_clc_aux_tma_config()
        )
        bound.config_spec.normalize(
            minimal_preprojection_clc_aux_tma_config,
            _fix_invalid=True,
        )
        self.assertEqual(
            minimal_preprojection_clc_aux_tma_config["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(
            minimal_preprojection_clc_aux_tma_config["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING],
        )
        self.assertEqual(
            minimal_preprojection_clc_aux_tma_config["range_flattens"],
            expected_clc_aux_tma_range_flattens,
        )
        self.assertEqual(
            minimal_preprojection_clc_aux_tma_config["range_multi_buffers"],
            expected_clc_aux_tma_range_multi_buffers,
        )
        self.assertEqual(
            minimal_preprojection_clc_aux_tma_config["range_warp_specializes"],
            expected_clc_aux_tma_range_warp_specializes,
        )
        unresolved_range_clc_aux_tma_config = (
            make_minimal_preprojection_clc_aux_tma_config()
        )
        with patch.object(
            bound.config_spec._cute_tcgen05_config,
            "_clc_aux_tma_matmul_k_range_index",
            return_value=None,
        ):
            bound.config_spec.normalize(
                unresolved_range_clc_aux_tma_config,
                _fix_invalid=True,
            )
        self.assertEqual(
            unresolved_range_clc_aux_tma_config["l2_groupings"],
            [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING],
        )
        self.assertNotIn("range_flattens", unresolved_range_clc_aux_tma_config)
        self.assertNotIn("range_multi_buffers", unresolved_range_clc_aux_tma_config)
        self.assertNotIn("range_warp_specializes", unresolved_range_clc_aux_tma_config)

        invalid_cluster_n_config = dict(valid_config)
        invalid_cluster_n_config["tcgen05_cluster_n"] = 2
        bound.config_spec.normalize(invalid_cluster_n_config, _fix_invalid=True)
        self.assertEqual(
            invalid_cluster_n_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        narrow_invalid_cluster_n_config = dict(valid_config)
        narrow_invalid_cluster_n_config["block_sizes"] = [
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
        ]
        narrow_invalid_cluster_n_config["tcgen05_cluster_n"] = 2
        narrow_invalid_cluster_n_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = (
            TCGEN05_AUX_LOAD_MODE_TMA
        )
        bound.config_spec.normalize(narrow_invalid_cluster_n_config, _fix_invalid=True)
        self.assertEqual(
            narrow_invalid_cluster_n_config["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(
            narrow_invalid_cluster_n_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        reset_config = dict(valid_config)
        reset_config["pid_type"] = "flat"
        bound.config_spec._cute_tcgen05_config.normalize_strategy(
            reset_config,
            fix_invalid=True,
        )
        self.assertEqual(
            reset_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.NON_PERSISTENT.value,
        )

        sm90_config = dict(valid_config)
        sm90_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
        sm90_bound.config_spec.normalize(sm90_config, _fix_invalid=True)
        self.assertEqual(
            sm90_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        self.assertEqual(
            sm90_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            TCGEN05_AUX_LOAD_MODE_TMA,
        )
        sm90_flat_keys = {
            key
            for key, _count, _is_sequence in sm90_bound.config_spec.flat_key_layout()
        }
        sm90_seeds = sm90_bound.config_spec.autotune_seed_configs()
        self.assertNotIn(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, sm90_flat_keys)
        self.assertFalse(
            any(
                seed.config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                == Tcgen05PersistenceModel.CLC_PERSISTENT.value
                for seed in sm90_seeds
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_narrow_n_seed_requires_n_edge_at_128(self) -> None:
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

        # N=4224 is an edge for block_n=256 but a full tile for block_n=128.
        # Keep the narrow-N seed out of this family so aux-TMA never turns the
        # validated double-output-edge + K-tail split into M-edge + K-tail.
        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 4224], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4224], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 4224], device=DEVICE, dtype=HALF_DTYPE),
        )
        with (
            patch_cute_mma_support(),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            bound = cute_matmul_bias_residual_gelu.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        seed_block_sizes = [
            config.config.get("block_sizes") for config in spec.autotune_seed_configs()
        ]
        self.assertFalse(
            any(
                isinstance(block_sizes, list)
                and len(block_sizes) > 1
                and block_sizes[1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
                for block_sizes in seed_block_sizes
            )
        )

        narrow_config: dict[str, object] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
            "indexing": ["tensor_descriptor"] * spec.indexing.length,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_strategy": Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_warp_spec_c_input_warps": 1,
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: TCGEN05_AUX_LOAD_MODE_TMA,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.CLC_PERSISTENT.value
            ),
        }
        spec.normalize(narrow_config, _fix_invalid=True)
        self.assertEqual(
            narrow_config["block_sizes"][:3],
            [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            ],
        )
        self.assertEqual(
            narrow_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_clc_force_persistent_hides_persistence_flat_axis(
        self,
    ) -> None:
        @helion.kernel(backend="cute", autotune_force_persistent=True)
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
        with (
            patch_cute_mma_support(),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            bound = cute_matmul_bias_residual_gelu.bind(args)

        spec = bound.config_spec
        self.assertEqual(spec.allowed_pid_types, ("persistent_interleaved",))
        flat_keys = {key for key, _count, _is_sequence in spec.flat_key_layout()}
        self.assertNotIn(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, flat_keys)
        self.assertFalse(
            any(
                seed.config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                == Tcgen05PersistenceModel.CLC_PERSISTENT.value
                for seed in spec.autotune_seed_configs()
            )
        )

        config_gen = spec.create_config_generation()
        default_flat = config_gen.default_flat()
        pid_indices, _ = config_gen._key_to_flat_indices["pid_type"]
        self.assertEqual(default_flat[pid_indices[0]], "persistent_interleaved")
        minimal_seed_flat = config_gen.flatten(helion.Config())
        self.assertEqual(minimal_seed_flat[pid_indices[0]], "persistent_interleaved")
        config = config_gen.unflatten([*default_flat])
        # Force-persistent removes "flat" from the pid fragment, so flattening
        # encodes persistent_interleaved. Unflatten normalization rewrites the
        # cluster_m=1 persistent pid back to flat, which derives NON_PERSISTENT.
        # The CLC persistence axis is hidden for this non-identity path.
        self.assertEqual(
            config.config["pid_type"],
            "flat",
        )
        self.assertEqual(
            config.config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.NON_PERSISTENT.value,
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_aux_tma_seed_requires_exact_shape_aux(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_bias(
            x: torch.Tensor,
            y: torch.Tensor,
            bias: torch.Tensor,
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
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_bias.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        flat_keys = {key for key, _count, _is_sequence in spec.flat_key_layout()}
        self.assertNotIn(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, flat_keys)
        self.assertNotIn(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, flat_keys)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.compiler_seed_configs
            )
        )
        self.assertFalse(
            any(
                config.config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                == Tcgen05PersistenceModel.CLC_PERSISTENT.value
                for config in spec.compiler_seed_configs
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_aux_tma_seed_rejects_mixed_exact_aux_dtype(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_two_residuals(
            x: torch.Tensor,
            y: torch.Tensor,
            residual_bf16: torch.Tensor,
            residual_fp32: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (
                    acc + residual_bf16[tile_m, tile_n] + residual_fp32[tile_m, tile_n]
                ).to(x.dtype)
            return out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=torch.float32),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_two_residuals.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.compiler_seed_configs
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_aux_tma_seed_rejects_unrelated_exact_aux_store(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_and_elementwise_store(
            x: torch.Tensor,
            y: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            aux_out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
                aux_out[tile_m, tile_n] = (residual[tile_m, tile_n] + 1).to(x.dtype)
            return out, aux_out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_and_elementwise_store.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.compiler_seed_configs
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_aux_tma_seed_rejects_partial_rank2_aux_load(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_partial_residual(
            x: torch.Tensor,
            y: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + residual[tile_m, 0][:, None]).to(x.dtype)
            return out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_partial_residual.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.compiler_seed_configs
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_aux_tma_seed_rejects_scrambled_exact_aux_index(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_scrambled_residual(
            x: torch.Tensor,
            y: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + residual[tile_n, tile_m]).to(x.dtype)
            return out

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_scrambled_residual.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.compiler_seed_configs
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_aux_tma_seed_rejects_multi_store_exact_aux(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_residual_fanout(
            x: torch.Tensor,
            y: torch.Tensor,
            residual_a: torch.Tensor,
            residual_b: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = x.size()
            _, n = y.size()
            out_a = torch.empty([m, n], dtype=x.dtype, device=x.device)
            out_b = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out_a[tile_m, tile_n] = (acc + residual_a[tile_m, tile_n]).to(x.dtype)
                out_b[tile_m, tile_n] = (acc + residual_b[tile_m, tile_n]).to(x.dtype)
            return out_a, out_b

        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_residual_fanout.bind(args)

        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertFalse(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.compiler_seed_configs
            )
        )

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
