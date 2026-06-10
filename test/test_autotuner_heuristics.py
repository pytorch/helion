from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2Heuristic
from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonReductionTileHeuristic
from helion._compiler.autotuner_heuristics.triton import (
    TritonReductionUserTileHeuristic,
)
from helion._compiler.autotuner_heuristics.triton import TritonSkinnyGemmHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonSplitJoinRotateHeuristic
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from helion._compiler.cute.strategies import Tcgen05LayoutStrategy
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
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
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
)
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
from helion._compiler.cute.tcgen05_constants import tcgen05_default_epilogue_tile_size
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
from helion.autotuner.config_spec import ReductionFact
from helion.autotuner.config_spec import ReductionLoopSpec
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


class TestTritonSplitJoinRotateHeuristic(TestCase):
    """Rope split/join "rotate" heuristic: seeds all-ones ``block_sizes`` and
    fires only for a split/join rotate kernel, not a plain elementwise one.
    """

    def test_seed_config_is_all_ones(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=2048))
        spec.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=2048))
        env = MagicMock()
        env.config_spec = spec
        config = TritonSplitJoinRotateHeuristic.get_seed_config(env, MagicMock())
        self.assertEqual(config.config["block_sizes"], [1, 1])

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler heuristics are not collected in ref eager mode")
    def test_fires_for_rope_not_elementwise(self) -> None:
        @helion.kernel(backend="triton")
        def rope_like(
            q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ) -> torch.Tensor:
            batch, heads, seq_len, head_dim = q.size()
            half_dim = head_dim // 2
            out = torch.empty_like(q)
            for tile_b, tile_t in hl.tile([batch, seq_len]):
                cos_pair = (
                    cos[tile_b, tile_t, :]
                    .to(torch.float32)
                    .reshape([tile_b, tile_t, 2, half_dim])
                    .permute(0, 1, 3, 2)
                )
                cos_first, cos_second = hl.split(cos_pair)
                q_pair = (
                    q[tile_b, :, tile_t, :]
                    .to(torch.float32)
                    .reshape([tile_b, heads, tile_t, 2, half_dim])
                    .permute(0, 1, 2, 4, 3)
                )
                q_first, q_second = hl.split(q_pair)
                out[tile_b, :, tile_t, :] = (
                    hl.join(
                        q_first * cos_first[:, None, :, :],
                        q_second * cos_second[:, None, :, :],
                    )
                    .permute(0, 1, 2, 4, 3)
                    .reshape([tile_b, heads, tile_t, head_dim])
                    .to(out.dtype)
                )
            return out

        @helion.kernel(backend="triton")
        def elementwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] * y[tile_m, tile_n]
            return out

        q = torch.randn(2, 8, 256, 64, device=DEVICE, dtype=HALF_DTYPE)
        angles = torch.randn(2, 256, 64, device=DEVICE, dtype=HALF_DTYPE)
        rope = rope_like.bind((q, torch.cos(angles), torch.sin(angles)))
        self.assertTrue(
            TritonSplitJoinRotateHeuristic.is_eligible(
                rope.env, rope.host_function.device_ir
            )
        )
        seed = TritonSplitJoinRotateHeuristic.get_seed_config(
            rope.env, rope.host_function.device_ir
        )
        self.assertEqual(
            seed.config["block_sizes"], [1] * len(rope.config_spec.block_sizes)
        )

        xy = (
            torch.randn(512, 512, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(512, 512, device=DEVICE, dtype=HALF_DTYPE),
        )
        ew = elementwise.bind(xy)
        self.assertFalse(
            TritonSplitJoinRotateHeuristic.is_eligible(
                ew.env, ew.host_function.device_ir
            )
        )


class TestTritonReductionTileHeuristic(TestCase):
    """Triton T1 row-reduction heuristic: seeds the "one row per program"
    skeleton with an rnumel-scaled ``num_warps`` ramp and faithful per-slot load
    eviction, fires only for a canonical row reduction, and its persistent seed
    survives flatten/unflatten (the config_spec sentinel round-trip fix).
    """

    def _reduction_spec(
        self,
        *,
        reduction_size_hint: int,
        num_load: int = 1,
        itemsize: int = 4,
    ) -> ConfigSpec:
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.reduction_loops.append(
            ReductionLoopSpec(block_id=1, size_hint=reduction_size_hint)
        )
        # The deepened heuristic reads a ReductionFact (the workload facts it keys
        # the warp ramp / eviction / persist decision on); the reduction axis is
        # block_id=1 (the rolled reduction loop above), the row axis block_id=0.
        spec.reduction_facts.append(
            ReductionFact(
                block_id=1,
                size_hint=reduction_size_hint,
                m_block_ids=(0,),
                static_rnumel=reduction_size_hint,
                itemsize=itemsize,
                num_load=num_load,
            )
        )
        return spec

    def _reduction_env(self, spec: ConfigSpec) -> MagicMock:
        # The deepened heuristic reads env.backend.max_tensor_numel (the structural
        # persistent cap) — provide the real Triton cap so a sub-cap rnumel stays
        # persistent.
        from helion.autotuner.config_generation import TRITON_MAX_TENSOR_NUMEL

        env = MagicMock()
        env.backend_name = "triton"
        env.backend.max_tensor_numel = TRITON_MAX_TENSOR_NUMEL
        env.config_spec = spec
        env.device = DEVICE
        return env

    def test_seed_is_persistent_one_row(self) -> None:
        # The structural seed: one row per program + persistent reduction. The
        # deepened heuristic ALSO seeds num_warps via the rnumel ramp (rnumel=1024
        # -> 4 warps) and num_stages=1, rather than leaving them to the autotuner.
        env = self._reduction_env(self._reduction_spec(reduction_size_hint=1024))
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            seed = TritonReductionTileHeuristic.get_seed_config(env, MagicMock())
        self.assertEqual(seed.config["block_sizes"], [1])
        self.assertEqual(seed.config["reduction_loops"], [None])
        # rnumel ramp: 1024 falls in the <=1024 band -> 4 warps.
        self.assertEqual(seed.config["num_warps"], 4)
        self.assertEqual(seed.config["num_stages"], 1)

    def test_single_load_seeds_stream_eviction_over_load_slots(self) -> None:
        # A single-load streaming reduction (num_load==1: e.g. sum) is read once
        # and never reused, so every load slot -> 'first' (evict_first frees L2),
        # broadcast over the spec's load slots. Build the fragment explicitly so
        # the test does not depend on the host backend's eviction choices.
        from helion.autotuner.config_fragment import EnumFragment
        from helion.autotuner.config_fragment import ListOf

        spec = self._reduction_spec(reduction_size_hint=1024, num_load=1)
        spec.load_eviction_policies = ListOf(
            EnumFragment(choices=("", "first", "last")), length=4
        )
        env = self._reduction_env(spec)
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            seed = TritonReductionTileHeuristic.get_seed_config(env, MagicMock())
        self.assertEqual(
            seed.config["load_eviction_policies"],
            ["first", "first", "first", "first"],
        )

    def test_persistent_seed_round_trips_through_config_generation(self) -> None:
        # reduction_loops=[None] (persistent) MUST survive flatten/unflatten. For
        # a wide reduction (size_hint 32000) a sentinel < size_hint would decode
        # back to the SLOW looped family this heuristic exists to avoid; the
        # config_spec fix encodes None as the fragment's ``high`` (>= size_hint).
        from helion.autotuner.config_generation import ConfigGeneration

        spec = self._reduction_spec(reduction_size_hint=32000)
        env = self._reduction_env(spec)
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            seed = TritonReductionTileHeuristic.get_seed_config(env, MagicMock())
        spec.compiler_seed_configs = [seed]
        pairs = ConfigGeneration(spec).seed_flat_config_pairs()
        self.assertEqual(len(pairs), 1)
        _flat, normalized = pairs[0]
        self.assertEqual(normalized.config["reduction_loops"], [None])

    def test_not_eligible_without_single_reduction_tile(self) -> None:
        env = MagicMock()
        # No reduction loop -> not a reduction.
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        env.config_spec = spec
        self.assertFalse(TritonReductionTileHeuristic.is_eligible(env, MagicMock()))
        # A matmul fact disqualifies even a 1-tile/1-reduction shape.
        spec_mm = self._reduction_spec(reduction_size_hint=1024)
        spec_mm.matmul_facts = [MagicMock()]
        env.config_spec = spec_mm
        self.assertFalse(TritonReductionTileHeuristic.is_eligible(env, MagicMock()))

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler heuristics are not collected in ref eager mode")
    def test_fires_for_reduction_not_matmul(self) -> None:
        @helion.kernel(backend="triton")
        def row_reduction(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :]
                shifted = row - torch.amax(row, dim=-1, keepdim=True)
                out[tile_m] = torch.log(torch.sum(torch.exp(shifted), dim=-1))
            return out

        @helion.kernel(backend="triton")
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        red = row_reduction.bind(
            (torch.randn(1024, 1024, device=DEVICE, dtype=HALF_DTYPE),)
        )
        self.assertTrue(
            TritonReductionTileHeuristic.is_eligible(
                red.env, red.host_function.device_ir
            )
        )
        seed = TritonReductionTileHeuristic.get_seed_config(
            red.env, red.host_function.device_ir
        )
        self.assertEqual(seed.config["block_sizes"], [1])
        self.assertEqual(seed.config["reduction_loops"], [None])

        mm = matmul.bind(
            (
                torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE),
                torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE),
            )
        )
        self.assertFalse(
            TritonReductionTileHeuristic.is_eligible(mm.env, mm.host_function.device_ir)
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
        # For an FFI-eligible 16-bit shape BOTH the DEFAULT-layout cluster_m=2
        # heuristic and the generalized TVM-FFI direct-entry heuristic emit a
        # cluster_m=2 seed; the FFI search projection then normalizes both onto
        # the same validated CtaGroup.TWO envelope. Require at least one and
        # check that every cluster_m=2 seed matches that envelope.
        self.assertGreaterEqual(len(seeded), 1)
        for seed in seeded:
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
        return seeded[0]

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
    def test_cute_tcgen05_full_tile_ffi_seed_config(self) -> None:
        # The generalized FFI direct-entry seed drives ANY eligible bf16
        # full-tile CtaGroup.TWO matmul (it replaced the bank of per-shape
        # ``_target{N}`` seeds). The seed itself now lives on the
        # ConfigSpec/CuteTcgen05Config and is emitted into the autotuner
        # population by ``CuteTcgen05ClusterM2FfiHeuristic``; it is no longer
        # part of ``autotune_seed_configs()`` (that chain is now only the
        # c-input family). This asserts the eligibility gate + the generalized
        # seed envelope plus the surviving search projection behavior.
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
        spec = bound.config_spec
        self.assertTrue(spec._tcgen05_full_tile_direct_entry_seed_eligible())
        seed_config = spec._tcgen05_full_tile_direct_entry_seed_config()
        self.assertIsNotNone(seed_config)
        seed = seed_config.config
        self.assertIs(seed[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY], True)
        self.assertEqual(
            seed[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY],
            Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
        )
        bk = spec._tcgen05_full_tile_direct_entry_seed_bk()
        self.assertIsNotNone(bk)
        self.assertEqual(
            seed["block_sizes"],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, bk],
        )
        self.assertEqual(seed["tcgen05_ab_stages"], 3)
        self.assertEqual(seed["tcgen05_cluster_m"], 2)
        self.assertEqual(seed["tcgen05_cluster_n"], 1)
        self.assertEqual(seed["tcgen05_c_stages"], 2)
        self.assertEqual(seed["num_warps"], 8)
        self.assertEqual(seed["pid_type"], "persistent_interleaved")
        self.assertEqual(seed[TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY], 128)
        self.assertEqual(seed[TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY], 32)
        self.assertEqual(seed[TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY], 32)
        self.assertIs(seed[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY], True)

        # The same generalized seed is what the search projection uses to map
        # FFI-requesting cluster_m=2 candidates onto the validated envelope.
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
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, bk],
        )

        # ab > 3 is only valid on the TVM-FFI direct-entry path for the
        # (bk, ab, c) stage tuples the codegen accepts
        # (``TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK``: bk=64 admits the deep
        # (ab=6, c=4) tuple). ab=4 and ab=5 are admitted by NO bk, so they are
        # rejected everywhere; a bare ab>3 config (no FFI launch) is likewise
        # rejected. ``_fix_invalid=True`` clamps any such config down to ab=3.
        def _non_seed_stage_config(requested_ab_stages: int) -> helion.Config:
            return helion.Config(
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

        for requested_ab_stages in (4, 5, 6):
            with self.subTest(requested_ab_stages=requested_ab_stages):
                fixed = _non_seed_stage_config(requested_ab_stages)
                bound.config_spec.normalize(fixed, _fix_invalid=True)
                self.assertEqual(fixed.config["tcgen05_ab_stages"], 3)

                with self.assertRaisesRegex(
                    helion.exc.InvalidConfig,
                    "tcgen05_ab_stages > 3 is not supported",
                ):
                    bound.config_spec.normalize(
                        _non_seed_stage_config(requested_ab_stages)
                    )

        # ab=6 IS accepted on the FFI direct-entry path at bk=64 with c=4: that
        # is the (ab=6, c=4) tuple ``TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK``
        # admits for bk=64. Plain normalize (no ``_fix_invalid``) leaves it at 6.
        ffi_direct_entry_ab6 = helion.Config(
            block_sizes=[256, 256, 64],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=6,
            tcgen05_c_stages=4,
            **{TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True},
        )
        bound.config_spec.normalize(ffi_direct_entry_ab6)
        self.assertEqual(ffi_direct_entry_ab6.config["tcgen05_ab_stages"], 6)

        # ab=6 is rejected for bk=128 even on the FFI direct-entry path: bk=128
        # only admits the (ab=3, c=2) tuple.
        ffi_direct_entry_ab6_bk128 = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=6,
            tcgen05_c_stages=4,
            **{TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True},
        )
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig, "tcgen05_ab_stages > 3 is not supported"
        ):
            bound.config_spec.normalize(ffi_direct_entry_ab6_bk128)

        search_ab_stages_fragment = bound.config_spec._tcgen05_optional_fragments(
            for_search=True
        )["tcgen05_ab_stages"]
        self.assertIsInstance(search_ab_stages_fragment, IntegerFragment)
        # Cycle 97: the for_search ab cap is BUDGET-AWARE — lifted to 3 wherever
        # ab=3 is admissible (the SMEM-budget constraints were recorded at bind
        # time, i.e. bf16/fp16 on a B200-class optin cap), else 2. Conditioning on
        # the recorded constraints keeps the assertion deterministic across hosts.
        expected_search_ab_high = (
            3
            if bound.config_spec._cute_tcgen05_config.ab_stages_three_search_constraints
            is not None
            else 2
        )
        self.assertEqual(search_ab_stages_fragment.high, expected_search_ab_high)

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
        # With no recorded SMEM budget the generalized FFI seed is ineligible
        # (ab=3 cannot fit) and the for_search ab cap stays at 2.
        self.assertFalse(
            no_ab3_budget_bound.config_spec._tcgen05_full_tile_direct_entry_seed_eligible()
        )
        no_budget_ab_stages_fragment = (
            no_ab3_budget_bound.config_spec._tcgen05_optional_fragments(
                for_search=True
            )["tcgen05_ab_stages"]
        )
        self.assertIsInstance(no_budget_ab_stages_fragment, IntegerFragment)
        self.assertEqual(no_budget_ab_stages_fragment.high, 2)

        # An fp16 matmul IS eligible for the FFI seed: the direct-entry TMA
        # descriptors / SMEM layout / epilogue tile are dtype-general for any
        # 16-bit operand, so fp16 (matching operand dtypes) at a structurally
        # valid shape is admitted exactly like bf16. Only fp32 stays excluded.
        fp16_args = (
            torch.empty([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.float16),
        )
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
        ):
            fp16_bound = cute_matmul_mma.bind(fp16_args)
        self.assertTrue(
            fp16_bound.config_spec._tcgen05_full_tile_direct_entry_seed_eligible()
        )
        self.assertIsNotNone(
            fp16_bound.config_spec._tcgen05_full_tile_direct_entry_seed_config()
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_full_tile_ffi_seed_rejects_structurally_invalid_shape(
        self,
    ) -> None:
        # The structural shape guard for the generalized FFI seed lives entirely
        # in ``_tcgen05_full_tile_direct_entry_seed_eligible`` (the per-shape
        # TargetN codegen gate and the runtime direct-entry validator were
        # removed). A shape whose N is not a multiple of the 256 CtaGroup.TWO
        # CTA tile is not a full-tile matmul, so the seed must be ineligible and
        # emit no config even though the dtype is a supported 16-bit type.
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

        # N = 4080 is not divisible by the 256 CTA tile -> edge tile, not a
        # full-tile CtaGroup.TWO matmul.
        invalid_args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 4080], device=DEVICE, dtype=torch.bfloat16),
        )
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
        ):
            invalid_bound = cute_matmul_mma.bind(invalid_args)
        spec = invalid_bound.config_spec
        self.assertFalse(spec._tcgen05_full_tile_direct_entry_seed_eligible())
        self.assertIsNone(spec._tcgen05_full_tile_direct_entry_seed_config())

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
    def test_cute_tcgen05_aux_tma_full_tile_search_projection(self) -> None:
        # Cycle 88 (Workstream B): on the residual full-tile cluster_m=2
        # family (T20-shape 6144³ bf16 residual_add), the search projection
        # ``_fix_aux_tma_full_tile_search_config`` forces cluster_m=2 SIMT
        # candidates onto the validated aux-TMA producer regime so the
        # +14 pp aux-TMA gain is banked deterministically. cluster_m=1
        # candidates stay untouched.
        @helion.kernel(backend="cute")
        def cute_matmul_residual_add(
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
                out[tile_m, tile_n] = (acc + residual[tile_m, tile_n]).to(x.dtype)
            return out

        args = (
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
        )
        # Mock the SMEM budget to the B200 value at bind time (when
        # ``allow_ab_stages_three_search`` records the budget into the
        # constraints) so the c=4 lift's ``c_stages_fits`` gate is deterministic
        # on any cute host (``@onlyBackends`` does not imply B200).
        b200_budget = 232448 - 28 * 1024
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
            patch.object(
                CuteTcgen05Config,
                "per_cta_ab_smem_budget_bytes",
                return_value=b200_budget,
            ),
        ):
            bound = cute_matmul_residual_add.bind(args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertTrue(spec._cute_tcgen05_config._aux_tma_full_tile_search_enabled())
        # The aux-TMA seed is present in the compiler seed pool.
        self.assertTrue(
            any(
                config.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                for config in spec.autotune_seed_configs()
            )
        )
        # A cluster_m=2 SIMT monolithic ab=3 candidate is projected onto the
        # aux-TMA regime (role_local_with_scheduler + warps + ab=2 + tma).
        cm2 = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=3,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            tcgen05_persistence_model="static_persistent",
        )
        # The budget was recorded into the constraints at bind time (mocked to
        # B200 above), so ``c_stages_fits`` is deterministic here.
        spec.normalize(cm2, _fix_invalid=True)
        self.assertEqual(
            cm2.config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY], TCGEN05_AUX_LOAD_MODE_TMA
        )
        self.assertEqual(
            cm2.config["tcgen05_strategy"],
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
        )
        self.assertEqual(cm2.config["tcgen05_ab_stages"], 2)
        self.assertEqual(cm2.config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY], 1)
        self.assertEqual(cm2.config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY], 1)
        # Cycle 90 (Workstream A Stage 2): the same projection deepens the C
        # ring to 4 (foundation for the Stage-4 store-warp split). At ab=2 the
        # c=4 ring fits under the 232 KB B200 cap, so the budget gate admits it.
        self.assertEqual(cm2.config["tcgen05_c_stages"], 4)
        # A cluster_m=1 candidate is left in its own regime (not forced to TMA),
        # and the deeper C ring is NOT projected onto it.
        cm1 = helion.Config(
            block_sizes=[128, 256, 64],
            indexing=["pointer", "tensor_descriptor", "tensor_descriptor"],
            pid_type="flat",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
        )
        spec.normalize(cm1, _fix_invalid=True)
        self.assertEqual(cm1.config["tcgen05_cluster_m"], 1)
        self.assertNotEqual(
            cm1.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY),
            TCGEN05_AUX_LOAD_MODE_TMA,
        )
        self.assertEqual(cm1.config["tcgen05_c_stages"], 2)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_c_stages_budget_gate(self) -> None:
        # Cycle 90 (Workstream A Stage 2): the budget-aware ``c_stages_fits``
        # check sums AB + C SMEM against the same 232 KB B200 envelope as the
        # ab=3 gate, using the REAL DEFAULT epilogue tile — which depends on
        # source-C presence: 256x256 16-bit is (128, 64) WITH source C (residual
        # family, 16 KB/stage) but (128, 32) WITHOUT one (plain matmul, 8
        # KB/stage). At the canonical 256x256x128 cluster_m=2 tile: ab=2 + c=4
        # fits (the foundation depth); ab=3 + c=4 overflows (matching the
        # cycle-90 probe where a directly sampled 256x256 ab=3 + c=4 hit a raw
        # ``ptxas: too much shared`` error). Uses a plain (no-epilogue) matmul so
        # the residual aux-TMA projection (which forces ab=2) does not claim the
        # sampled candidate — the admission gate is the only thing acting on c=4.
        # The SMEM budget is MOCKED to the B200 value so the gate is exercised
        # deterministically on any cute host (``@onlyBackends`` does not imply
        # B200).
        #
        # An fp16 (NOT bf16) matmul is used so the generalized TVM-FFI
        # direct-entry seed — which is bf16-only — is INELIGIBLE here. On a
        # bf16-eligible shape ``_fix_target1_tvm_ffi_search_config`` claims every
        # cluster_m=2 candidate and projects it onto the validated FFI envelope
        # (ab=3, c=2), which would shadow the c-stages gate before it could act.
        # fp16 still records the ab=3 SMEM-budget constraints (16-bit), so the
        # c-stages gate is exercised in isolation at the canonical cluster_m=2
        # 256x256x128 tile.
        b200_budget = 232448 - 28 * 1024  # optin cap - ab=3 reservation

        @helion.kernel(backend="cute")
        def cute_matmul_plain(
            x: torch.Tensor,
            y: torch.Tensor,
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

        args = (
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.float16),
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.float16),
        )
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
            patch.object(
                CuteTcgen05Config,
                "per_cta_ab_smem_budget_bytes",
                return_value=b200_budget,
            ),
        ):
            bound = cute_matmul_plain.bind(args)
        spec = bound.config_spec
        tcfg = spec._cute_tcgen05_config
        # The DEFAULT epilogue tile for a 256x256 16-bit tile depends on
        # source-C presence (N shrinks when no C tile competes for SMEM).
        self.assertEqual(
            tcgen05_default_epilogue_tile_size(
                256, 256, elem_width_d=16, elem_width_c=16
            ),
            (128, 64),
        )
        self.assertEqual(
            tcgen05_default_epilogue_tile_size(
                256, 256, elem_width_d=16, elem_width_c=None
            ),
            (128, 32),
        )
        # With source-C (residual family, 16 KB/stage): ab=2 + c=4 = 192 KB fits;
        # ab=3 + c=4 = 256 KB overflows.
        self.assertTrue(
            tcfg.c_stages_fits(
                bm=256,
                bn=256,
                bk=128,
                cluster_m=2,
                ab_stages=2,
                c_stages=4,
                has_source_c=True,
            )
        )
        self.assertFalse(
            tcfg.c_stages_fits(
                bm=256,
                bn=256,
                bk=128,
                cluster_m=2,
                ab_stages=3,
                c_stages=4,
                has_source_c=True,
            )
        )
        # Without source-C (plain matmul, 8 KB/stage): ab=3 + c=4 = 224 KB still
        # overflows the conservative budget (the cycle-90 probe confirmed the
        # plain 256x256 ab=3 + c=4 hits raw ptxas ``too much shared``).
        self.assertFalse(
            tcfg.c_stages_fits(
                bm=256,
                bn=256,
                bk=128,
                cluster_m=2,
                ab_stages=3,
                c_stages=4,
                has_source_c=False,
            )
        )
        # True admission gate: a DIRECTLY sampled 256x256 ab=3 + c=4 candidate
        # (no projection claims it — plain matmul, no aux) is demoted to c=2 so
        # tuning never reaches the raw ptxas overflow. ab=3 alone fits, so the
        # ab-stages gate keeps it — only c is demoted.
        #
        # Exercise the c-stages admission gate (``_fix_c_stages_search_config``)
        # DIRECTLY rather than through the full ``fix_search_config`` chain.
        # Since the generalized TVM-FFI direct-entry seed is now eligible for
        # ANY structurally-valid 16-bit shape (including fp16 6144³), the FFI
        # projection in ``fix_search_config`` would otherwise claim every
        # cluster_m=2 candidate and project it onto the validated (ab=3, c=2)
        # EXPLICIT_EPI_TILE envelope — shadowing the DEFAULT-layout c-stages
        # gate. Calling the gate directly keeps the test focused on the
        # c-stages budget demotion it is meant to validate.
        tcfg.search_enabled = True
        ab3_c4 = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=3,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=4,
            tcgen05_strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            tcgen05_persistence_model="static_persistent",
        )
        # The budget is already recorded in the constraints from bind (mocked to
        # B200 above), so the gate is deterministic here.
        tcfg._fix_c_stages_search_config(ab3_c4.config)
        self.assertEqual(ab3_c4.config["tcgen05_ab_stages"], 3)
        self.assertEqual(ab3_c4.config["tcgen05_c_stages"], 2)
        # ab=2 + c=4 (fits) is preserved by the gate.
        ab2_c4 = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=4,
            tcgen05_strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            tcgen05_persistence_model="static_persistent",
        )
        tcfg._fix_c_stages_search_config(ab2_c4.config)
        self.assertEqual(ab2_c4.config["tcgen05_c_stages"], 4)
        # Fail CLOSED: with no recorded SMEM budget (non-B200 / CPU host, where
        # ``ab_stages_three_search_constraints`` is None) a sampled c=4 cannot be
        # proven to fit, so it is demoted to 2 rather than left to overflow.
        tcfg.ab_stages_three_search_constraints = None
        ab2_c4_no_budget = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=4,
            tcgen05_strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            tcgen05_persistence_model="static_persistent",
        )
        tcfg._fix_c_stages_search_config(ab2_c4_no_budget.config)
        self.assertEqual(ab2_c4_no_budget.config["tcgen05_c_stages"], 2)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_ab_stages_budget_gate(self) -> None:
        # Cycle 97: ab=3 is BUDGET-AWARE-SEARCHABLE. The for_search ab cap is
        # lifted to 3 wherever ab=3 is admissible (constraints recorded), and
        # ``_fix_ab_stages_search_config`` demotes a sampled ab=3 that does not fit
        # — fail-CLOSED, mirroring the c-stages budget gate. The new dimension over
        # the bare-AB gate is REAL source-C presence, keyed on the PRECISE
        # ``exact_shape_aux_kernel_detected`` (rank-2 exact-shape residual_add), NOT
        # the broad ``aux_kernel_detected`` (which is also True for a rowvec bias
        # that has no source-C ring). A real source-C kernel materializes the larger
        # (128, 64) C ring, so AB(ab=3) + C overflows the 232 KiB B200 cap even at
        # c=2 and MUST demote; the plain / rowvec-bias family (no source-C ring)
        # keeps the calibrated bare-AB admission so its ab=3 cluster_m=2 winner stays
        # searchable (cycle-97 force-config: bias 256x256x128 cluster_m=2 ab=3
        # compiles + runs, T16 639.7 / T2 460.1 TF). The SMEM budget is MOCKED to the
        # B200 value so the gate is deterministic on any cute host.
        b200_budget = 232448 - 28 * 1024  # optin cap - ab=3 reservation

        @helion.kernel(backend="cute")
        def cute_matmul_plain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        @helion.kernel(backend="cute")
        def cute_matmul_bias(
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

        @helion.kernel(backend="cute")
        def cute_matmul_residual_add(
            x: torch.Tensor, y: torch.Tensor, residual: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = (acc + residual[tile_m, tile_n]).to(x.dtype)
            return out

        plain_args = (
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
        )
        bias_args = (
            torch.empty([1024, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([1024], device=DEVICE, dtype=torch.bfloat16),
        )
        # T16 shape (4096x4096x512): K=512 -> bk=128 -> 4 divisible k-tiles, so
        # cluster_m=2 search IS admitted (passes ``cluster_m2_bk_is_valid``). Used
        # for the END-TO-END bias guard below — unlike the 1024x4096x1024 bias
        # above (cluster_m=2 search OFF -> reprojects to cluster_m=1).
        bias_t16_args = (
            torch.empty([4096, 512], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([512, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096], device=DEVICE, dtype=torch.bfloat16),
        )
        residual_args = (
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([6144, 6144], device=DEVICE, dtype=torch.bfloat16),
        )
        with (
            patch_cute_mma_support(),
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=132),
            patch.object(
                CuteTcgen05Config,
                "per_cta_ab_smem_budget_bytes",
                return_value=b200_budget,
            ),
        ):
            plain_bound = cute_matmul_plain.bind(plain_args)
            bias_bound = cute_matmul_bias.bind(bias_args)
            bias_t16_bound = cute_matmul_bias.bind(bias_t16_args)
            residual_bound = cute_matmul_residual_add.bind(residual_args)

        plain_tcfg = plain_bound.config_spec._cute_tcgen05_config
        bias_tcfg = bias_bound.config_spec._cute_tcgen05_config
        bias_t16_tcfg = bias_t16_bound.config_spec._cute_tcgen05_config
        residual_tcfg = residual_bound.config_spec._cute_tcgen05_config
        # Plain: no aux at all. Bias: broad aux True but NO source-C ring (rowvec).
        # Residual: real rank-2 exact-shape source-C.
        self.assertFalse(plain_tcfg.aux_kernel_detected)
        self.assertFalse(plain_tcfg.exact_shape_aux_kernel_detected)
        self.assertTrue(bias_tcfg.aux_kernel_detected)
        self.assertFalse(bias_tcfg.exact_shape_aux_kernel_detected)
        self.assertTrue(residual_tcfg.aux_kernel_detected)
        self.assertTrue(residual_tcfg.exact_shape_aux_kernel_detected)

        # The for_search ab fragment is lifted to 3 (the budget was recorded at
        # bind via the mocked B200 cap) for every family.
        for tcfg in (plain_tcfg, bias_tcfg, residual_tcfg):
            ab_fragment = tcfg.optional_fragments(for_search=True)["tcgen05_ab_stages"]
            self.assertEqual(ab_fragment.high, 3)

        def _ab3_config(cluster_m: int = 2) -> helion.Config:
            return helion.Config(
                block_sizes=[256, 256, 128],
                indexing=[
                    "tensor_descriptor",
                    "tensor_descriptor",
                    "tensor_descriptor",
                ],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=cluster_m,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=3,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                tcgen05_strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                tcgen05_persistence_model="static_persistent",
            )

        # PLAIN (no source-C): the bare AB pipeline (192 KiB at cluster_m=2) fits
        # the 199 KiB budget and the small no-source-C epilogue D ring rides the
        # non-AB reservation — the ab=3 winner is PRESERVED.
        plain_ab3 = _ab3_config()
        plain_tcfg.fix_search_config(plain_ab3.config)
        self.assertEqual(plain_ab3.config["tcgen05_ab_stages"], 3)

        # BIAS (broad-aux True, source-C False): the rowvec bias has NO source-C
        # ring, so the gate's NON-source-C branch admits the same 192 KiB bare-AB
        # ab=3 at cluster_m=2. This is the case that exercises the precise-signal
        # fix — under the old broad ``aux_kernel_detected`` branch the gate would
        # have wrongly demoted it (cycle-97 force-config proved bias 256x256x128
        # cluster_m=2 ab=3 fits + runs). Call the gate in ISOLATION because the
        # full chain's cluster_m=2 projection (``_fix_cluster_m2_search_config``)
        # can reproject a bias candidate to cluster_m=1 for K-cap reasons unrelated
        # to this gate; the gate itself must KEEP the bias cluster_m=2 ab=3.
        bias_ab3 = _ab3_config()
        bias_tcfg._fix_ab_stages_search_config(bias_ab3.config)
        self.assertEqual(bias_ab3.config["tcgen05_ab_stages"], 3)

        # BIAS END-TO-END (the P2 guard): on a bias shape where cluster_m=2 search
        # is genuinely admitted (T16 = 4096x4096x512, K=512 -> 4 divisible k-tiles),
        # the FULL ``fix_search_config`` chain must KEEP a cluster_m=2 256x256x128
        # ab=3 bias candidate at ab=3 cluster_m=2 — NOT reprojected to cluster_m=1
        # (the cluster_m=2 projection accepts it) and NOT demoted to ab=2 (no
        # source-C ring). This locks in the bias-family "now admits cluster_m=2 ab=3"
        # claim that GATE 2 confirmed empirically (T5/T9/T16), guarding it against a
        # future regression the way the plain/silu winner is already covered.
        self.assertTrue(bias_t16_tcfg.aux_kernel_detected)
        self.assertFalse(bias_t16_tcfg.exact_shape_aux_kernel_detected)
        self.assertIsNotNone(bias_t16_tcfg.cluster_m2_search_constraints)
        bias_t16_ab3 = _ab3_config()
        bias_t16_tcfg.fix_search_config(bias_t16_ab3.config)
        self.assertEqual(bias_t16_ab3.config["tcgen05_ab_stages"], 3)
        self.assertEqual(bias_t16_ab3.config["tcgen05_cluster_m"], 2)
        self.assertEqual(bias_t16_ab3.config["block_sizes"][:3], [256, 256, 128])

        # RESIDUAL source-C branch, in ISOLATION. The exact-shape aux-TMA full-tile
        # projection forces ab=2 on a cluster_m=2 candidate BEFORE the gate runs, so
        # to exercise the gate's source-C branch directly we call it on a cluster_m=1
        # residual candidate (which no projection claims): AB(ab=3) + (128, 64) C
        # ring overflows even at cluster_m=1, so it DEMOTES to 2.
        residual_cm1_ab3 = _ab3_config(cluster_m=1)
        residual_tcfg._fix_ab_stages_search_config(residual_cm1_ab3.config)
        self.assertEqual(residual_cm1_ab3.config["tcgen05_ab_stages"], 2)

        # And the same residual source-C branch demotes a cluster_m=2 candidate too
        # (independent of the aux-TMA projection): call the gate in isolation.
        residual_cm2_ab3 = _ab3_config(cluster_m=2)
        residual_tcfg._fix_ab_stages_search_config(residual_cm2_ab3.config)
        self.assertEqual(residual_cm2_ab3.config["tcgen05_ab_stages"], 2)

        # Full chain on the cluster_m=2 residual: the aux-TMA projection forces ab=2
        # first, and the gate is consistent (still 2).
        residual_full = _ab3_config(cluster_m=2)
        residual_tcfg.fix_search_config(residual_full.config)
        self.assertEqual(residual_full.config["tcgen05_ab_stages"], 2)

        # cluster_m=1 256x256 plain ab=3 overflows bare-AB (384 KiB > budget) and is
        # demoted even without a source-C.
        plain_cm1_ab3 = _ab3_config(cluster_m=1)
        plain_tcfg.fix_search_config(plain_cm1_ab3.config)
        self.assertEqual(plain_cm1_ab3.config["tcgen05_ab_stages"], 2)

        # Fail CLOSED: with no recorded SMEM budget the sampled plain ab=3 cannot be
        # proven to fit, so it is demoted to 2 rather than left to overflow.
        plain_tcfg.ab_stages_three_search_constraints = None
        plain_ab3_no_budget = _ab3_config()
        plain_tcfg.fix_search_config(plain_ab3_no_budget.config)
        self.assertEqual(plain_ab3_no_budget.config["tcgen05_ab_stages"], 2)

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

        # fp16 4096³ is now FFI-eligible, so the leading compiler seed is the
        # generalized TVM-FFI direct-entry cluster_m=2 seed (previously fp16 was
        # bf16-only for the FFI seed, so the leading seed was the cluster_m=1
        # universal default). The DEFAULT-layout cluster_m=2 seed is also
        # emitted; both normalize onto the validated CtaGroup.TWO envelope.
        config_gen = bound.config_spec.create_config_generation()
        zero_flat = config_gen.random_population_flat(0)
        self.assertEqual(len(zero_flat), 1)
        zero_config = config_gen.unflatten(zero_flat[0])
        self.assertEqual(zero_config.config["tcgen05_cluster_m"], 2)
        one_flat = config_gen.random_population_flat(1)
        self.assertEqual(len(one_flat), 1)
        one_config = config_gen.unflatten(one_flat[0])
        self.assertEqual(one_config.config["tcgen05_cluster_m"], 2)
        one_config_population = config_gen.random_population(1)
        self.assertEqual(len(one_config_population), 1)
        self.assertEqual(one_config_population[0].config["tcgen05_cluster_m"], 2)
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
        # This test only requires the CuTe cluster-m2 seed to be present. For an
        # FFI-eligible shape the DEFAULT and TVM-FFI cluster_m=2 seeds normalize
        # to the same validated config, so the dedup'd FROM_BEST_AVAILABLE
        # initial population can collapse to a single distinct config.
        self.assertGreaterEqual(len(configs), 1)
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
        # The bias epilogue is an FFI-supported family, so both the DEFAULT and
        # the generalized TVM-FFI cluster_m=2 seeds are emitted; each must carry
        # an indexing list matching the live spec's (wider-than-3) length.
        self.assertGreaterEqual(len(seeded), 1)
        for seed in seeded:
            self.assertEqual(
                len(seed["indexing"]),
                bound.config_spec.indexing.length,
            )


class TestTritonReductionHeuristic(TestCase):
    """Lock the reduction seed heuristics' branch decisions on two kernels, one per
    track:

    - rms_norm wide (rnumel=16384): the T1 path (``TritonReductionTileHeuristic``)
      seeds a persistent reduction (``reduction_loops=[None]``) with the rnumel-ramp
      warp count.
    - kl_div wide (rnumel=131072): the Band-B T2 path
      (``TritonReductionUserTileHeuristic``) caps R_BLOCK by the accumulator footprint
      instead of going full-N persistent, with M at floor 1.
    """

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler reduction facts are not collected in ref eager mode")
    def test_rms_norm_wide_seeds_persistent_with_warps(self) -> None:
        from examples.rms_norm import rms_norm_fwd

        m, n = 2048, 16384
        args = (
            torch.randn([m, n], device=DEVICE, dtype=torch.float32),
            torch.randn([n], device=DEVICE, dtype=torch.float32),
            1e-5,
        )
        heuristic = TritonReductionTileHeuristic

        # Pin the kernel to the triton backend. autotuner_heuristics is populated by
        # iterating HEURISTICS_BY_BACKEND[env.backend_name], which only has triton /
        # cute / pallas entries. @onlyBackends(["triton"]) still RUNS this test on the
        # tileir lane (tileir is triton-compatible there), where env.backend_name would
        # be "tileir" -> no registered heuristics -> autotuner_heuristics == [] and the
        # assertIn below fails. Binding a triton-pinned kernel keeps backend_name
        # "triton" on every lane, matching test_t1_reduction_then_normalize_loop_widens_tile.
        kernel = helion.kernel(rms_norm_fwd.fn, backend="triton")

        # Force the sm90 deep path so the test exercises the H100-tuned seed on any
        # runner (off-sm90 the heuristic falls back to the conservative narrow seed).
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            bound = kernel.bind(args)

            # The reduction heuristic registered a single workload fact and fired.
            self.assertEqual(len(bound.config_spec.reduction_facts), 1)
            fact = bound.config_spec.reduction_facts[0]
            self.assertEqual(fact.size_hint, n)
            # rms_norm has no separate apply/normalize loop (its apply is over the full
            # row in the reduction scope), so no reduce-then-apply tile is captured.
            self.assertEqual(fact.non_reduction_loop_block_ids, ())
            self.assertIn(
                TritonReductionTileHeuristic.name,
                bound.config_spec.autotuner_heuristics,
            )
            self.assertTrue(
                heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            )

            # Exactly one compiler seed, and it is the *persistent* T1 config.
            seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        self.assertEqual(len(seeds), 1)
        seed = seeds[0].config
        # rnumel ramp: 16384 falls in the (4096, 16384] band -> 16 warps.
        self.assertEqual(seed["block_sizes"], [1])
        self.assertEqual(seed["reduction_loops"], [None])
        self.assertEqual(seed["num_warps"], 16)
        self.assertEqual(seed["num_stages"], 1)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler reduction facts are not collected in ref eager mode")
    def test_kl_div_wide_seeds_band_b_r_block_cap(self) -> None:
        from examples.kl_div import kl_div_forward

        m, n = 4096, 131072
        log_q = torch.log_softmax(torch.randn([m, n], device=DEVICE), dim=-1)
        p = torch.softmax(torch.randn([m, n], device=DEVICE), dim=-1)
        args = (log_q, p)
        heuristic = TritonReductionUserTileHeuristic

        # Pin the kernel to the triton backend. autotuner_heuristics is populated by
        # iterating HEURISTICS_BY_BACKEND[env.backend_name], which only has triton /
        # cute / pallas entries. @onlyBackends(["triton"]) still RUNS this test on the
        # tileir lane (tileir is triton-compatible there), where env.backend_name would
        # be "tileir" -> no registered heuristics -> autotuner_heuristics == [] and the
        # assertIn below fails. Binding a triton-pinned kernel keeps backend_name
        # "triton" on every lane, matching test_t1_reduction_then_normalize_loop_widens_tile.
        kernel = helion.kernel(kl_div_forward.fn, backend="triton")

        # Force the sm90 deep path so the Band-B seed is exercised on any runner
        # (off-sm90 the heuristic declines to the conservative narrow seed).
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            bound = kernel.bind(args)

            # Single workload fact carrying a 2D [M, R] tile -> Band B.
            self.assertEqual(len(bound.config_spec.reduction_facts), 1)
            fact = bound.config_spec.reduction_facts[0]
            self.assertEqual(fact.size_hint, n)
            self.assertGreaterEqual(fact.num_carried_2d_tiles, 1)
            self.assertEqual(fact.non_reduction_loop_block_ids, ())
            self.assertIn(
                TritonReductionUserTileHeuristic.name,
                bound.config_spec.autotuner_heuristics,
            )
            self.assertTrue(
                heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            )

            # Exactly one seed; R_BLOCK is capped, NOT full-N persistent, and the
            # grid (M) axis sits at its floor of 1.
            seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        self.assertEqual(len(seeds), 1)
        seed = seeds[0].config
        expected_cap = max(
            1,
            TritonReductionUserTileHeuristic.BANDB_R_BLOCK_BYTES
            // max(1, fact.itemsize),
        )
        # block_sizes is [R_BLOCK, M_BLOCK]; the reduction axis is capped well
        # below next_pow2(131072) and M stays at 1.
        self.assertEqual(seed["block_sizes"], [expected_cap, 1])
        self.assertLess(expected_cap, n)
        # rnumel 131072 > the 16384 warps-32 breakpoint -> 32 warps.
        self.assertEqual(seed["num_warps"], 32)
        self.assertEqual(seed["num_stages"], 1)
        # Band B must NOT use the T1 reduction_loops knob.
        self.assertNotIn("reduction_loops", seed)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler reduction facts are not collected in ref eager mode")
    def test_t1_reduction_then_normalize_loop_widens_tile(self) -> None:
        # A T1 rollable reduction (hl.sum over the full inner dim) IMMEDIATELY followed
        # by a SEPARATE non-reduction hl.tile(n) loop that normalizes the row. No
        # curriculum kernel has this shape, so it pins the T1 non-reduction-loop path:
        # - the fact captures the normalize loop as non_reduction_loop_block_ids and
        #   keeps m_block_ids grid-only (the normalize tile is NOT a row axis), and
        # - the seed emits a full-length block_sizes with that tile widened (without it
        #   the T1 seed would emit a wrong-length [grid_floor] and crash) — for BOTH the
        #   persistent and the looped (wide-N) reduction cases.
        # NOTE: this T1+normalize seed is NOT performance-validated (no oracle); it is
        # only a seed (worse tile => more autotuning, never wrong results), so the test
        # asserts only that the emitted config is well-formed in both regimes.
        @helion.kernel(backend="triton")
        def t1_then_normalize(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                s = torch.sum(x[tile_m, :], dim=-1)
                for tile_n in hl.tile(n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] / s[:, None]
            return out

        def check(m: int, n: int, expect_looped: bool) -> None:
            # Force the sm90 deep path so the T1+normalize seed is exercised on any
            # runner (off-sm90 the heuristic falls back to the narrow seed, which does
            # not widen the normalize tile).
            with patch(
                "helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE
            ):
                bound = t1_then_normalize.bind(
                    (torch.randn([m, n], device=DEVICE, dtype=torch.float32),)
                )
                # One workload fact: reduction axis + grid-only row axis + the normalize
                # loop captured as a non-reduction loop tile (NOT a row axis).
                self.assertEqual(len(bound.config_spec.reduction_facts), 1)
                fact = bound.config_spec.reduction_facts[0]
                self.assertEqual(fact.size_hint, n)
                self.assertEqual(fact.m_block_ids, (0,))
                self.assertEqual(len(fact.non_reduction_loop_block_ids), 1)
                self.assertNotIn(fact.block_id, fact.non_reduction_loop_block_ids)
                self.assertEqual(fact.num_carried_2d_tiles, 0)
                self.assertIn(
                    TritonReductionTileHeuristic.name,
                    bound.config_spec.autotuner_heuristics,
                )
                # Exactly one seed; block_sizes has an entry per tiled dim (grid +
                # normalize loop), the grid axis at its floor and the normalize tile
                # widened (> 1), and it normalizes without error (the crux: a valid,
                # full-length config).
                seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
            self.assertEqual(len(seeds), 1)
            seed = seeds[0].config
            self.assertEqual(
                len(seed["block_sizes"]), len(bound.config_spec.block_sizes)
            )
            norm_idx = bound.config_spec.block_sizes.block_id_to_index(
                fact.non_reduction_loop_block_ids[0]
            )
            self.assertGreater(seed["block_sizes"][norm_idx], 1)
            # Persistent (narrow row) -> reduction_loops=[None]; looped (wide row past
            # the byte cap) -> reduction_loops=[LOOPED_CHUNK].
            if expect_looped:
                self.assertEqual(
                    seed["reduction_loops"],
                    [TritonReductionTileHeuristic.LOOPED_CHUNK],
                )
            else:
                self.assertEqual(seed["reduction_loops"], [None])
            # The emitted seed must round-trip through normalize() without raising.
            bound.config_spec.normalize(dict(seed))

        # 1024x4096: 16 KB/row < 240 KB byte cap -> persistent.
        check(1024, 4096, expect_looped=False)
        # 1024x131072: 512 KB/row > 240 KB byte cap -> looped (the case the old guard
        # wrongly declined into a wrong-length crash; now emits a widened looped seed).
        check(1024, 131072, expect_looped=True)

    def test_dynamic_extent_normalize_tile_matches_reduction_tile(self) -> None:
        # When the reduction extent is NOT statically known (static_rnumel is None,
        # e.g. a dynamic/jagged reduce-then-apply reduction), the per-row-bytes cap has
        # no extent to key on, so the non-reduction loop tile falls back to "match the
        # reduction tile". This default is NOT tuned on any kernel (no curriculum kernel
        # has a dynamic-extent non-reduction loop); the test pins the fallback's two
        # shapes.
        from helion.autotuner.config_spec import BlockSizeSpec

        H = TritonReductionUserTileHeuristic
        size_hint = 4096  # next_pow2(size_hint) == 4096

        def spec_with(reduction_bid: int, norm_bid: int) -> ConfigSpec:
            spec = ConfigSpec(backend=TritonBackend())
            # grid (block 0), reduction axis, normalize-loop axis — all block_sizes.
            spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
            spec.block_sizes.append(
                BlockSizeSpec(block_id=reduction_bid, size_hint=size_hint)
            )
            spec.block_sizes.append(
                BlockSizeSpec(block_id=norm_bid, size_hint=size_hint)
            )
            return spec

        def fact(block_id: int, norm_bid: int) -> ReductionFact:
            return ReductionFact(
                block_id=block_id,
                size_hint=size_hint,
                m_block_ids=(0,),
                static_rnumel=None,  # <-- dynamic extent: triggers the fallback
                itemsize=4,
                num_load=1,
                non_reduction_loop_block_ids=(norm_bid,),
            )

        # T2: the reduction axis IS a block_sizes entry (red_value given). The normalize
        # tile matches that red_value (777, an arbitrary sentinel), NOT a byte-cap value.
        spec = spec_with(reduction_bid=1, norm_bid=2)
        bs = H._build_block_sizes(spec, fact(1, 2), 1, 777, non_reduction_loop_ids={2})
        red_idx = spec.block_sizes.block_id_to_index(1)
        norm_idx = spec.block_sizes.block_id_to_index(2)
        self.assertEqual(bs[red_idx], 777)
        self.assertEqual(bs[norm_idx], 777)  # normalize tile == reduction tile

        # T1: the reduction rides reduction_loops (red_block_id=None, red_value=None), so
        # the normalize tile matches next_pow2(size_hint) instead — must NOT floor to 1.
        bs_t1 = H._build_block_sizes(
            spec, fact(1, 2), None, None, non_reduction_loop_ids={2}
        )
        self.assertEqual(bs_t1[norm_idx], 4096)  # next_pow2(4096)
        self.assertNotEqual(bs_t1[norm_idx], 1)
        self.assertEqual(bs_t1[0], H._block_floor(spec.block_sizes[0]))  # grid floored
