from __future__ import annotations

import dataclasses
import math
import os
from typing import Any
from typing import Callable
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

import helion
from helion._argument_device import _ArgumentDeviceResolver as _DeviceResolver
from helion._compiler.autotuner_heuristics import compiler_promotion_specialization_key
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics import compiler_seed_specialization_facts
from helion._compiler.autotuner_heuristics.cute import (
    CuteFlashAttentionCausalLptHeuristic,
)
from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteFp8GemmSkinnyMHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2Heuristic
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedWorklistHeuristic,
)
from helion._compiler.autotuner_heuristics.cute import (
    _filter_reachable_block_size_configs,
)
from helion._compiler.autotuner_heuristics.cute import _tcgen05_grouped_fact
from helion._compiler.autotuner_heuristics.cute import _tcgen05_grouped_worklist_fact
from helion._compiler.autotuner_heuristics.cute import (
    tcgen05_grouped_worklist_seed_configs,
)
from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
from helion._compiler.autotuner_heuristics.registry import (
    CompilerHeuristicSpecializationFact,
)
from helion._compiler.autotuner_heuristics.triton import TritonH100MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonNarrowReductionHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonPointwiseSeedHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonSkinnyGemmHeuristic
from helion._compiler.autotuner_heuristics.triton import (
    TritonStandardReductionHeuristicSM90,
)
from helion._compiler.autotuner_heuristics.triton import (
    TritonStandardReductionHeuristicSM100,
)
from helion._compiler.autotuner_heuristics.triton import (
    TritonUserTiledReductionHeuristicSM90,
)
from helion._compiler.autotuner_heuristics.triton import (
    TritonUserTiledReductionHeuristicSM100,
)
from helion._compiler.autotuner_heuristics.triton import _h100_matmul_tile
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.cute_flash import FLASH_AUTOTUNE_CONFIG_KEYS
from helion._compiler.cute.cute_flash import FLASH_CAUSAL_KV_ORDER_KEY
from helion._compiler.cute.cute_flash import FLASH_CAUSAL_LOOP_SPLIT_KEY
from helion._compiler.cute.cute_flash import FLASH_CAUSAL_LPT_SWIZZLE_KEY
from helion._compiler.cute.cute_flash import FLASH_CGA2_LOCAL_KEY
from helion._compiler.cute.cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY
from helion._compiler.cute.cute_flash import FLASH_CLC_KEY
from helion._compiler.cute.cute_flash import FLASH_CLC_PDL_KEY
from helion._compiler.cute.cute_flash import FLASH_CLC_STAGES_KEY
from helion._compiler.cute.cute_flash import FLASH_CONFIG_KEYS
from helion._compiler.cute.cute_flash import FLASH_CORR_REGS_KEY
from helion._compiler.cute.cute_flash import FLASH_CORR_TILE_SIZE_KEY
from helion._compiler.cute.cute_flash import FLASH_DERIVED_CONFIG_KEYS
from helion._compiler.cute.cute_flash import FLASH_DISC_PIPE_KEY
from helion._compiler.cute.cute_flash import FLASH_E2E_OFFSET0_KEY
from helion._compiler.cute.cute_flash import FLASH_E2E_OFFSET_KEY
from helion._compiler.cute.cute_flash import FLASH_E2E_SCHEDULE_KEY
from helion._compiler.cute.cute_flash import FLASH_EPI_STG_GMEM_KEY
from helion._compiler.cute.cute_flash import FLASH_EPI_STG_KEY
from helion._compiler.cute.cute_flash import FLASH_EPI_STG_STORE_KEY
from helion._compiler.cute.cute_flash import FLASH_EPI_TMA_KEY
from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
from helion._compiler.cute.cute_flash import FLASH_FIRST_LOAD_ORDER_KEY
from helion._compiler.cute.cute_flash import FLASH_KV_ORDER_KEY
from helion._compiler.cute.cute_flash import FLASH_KV_STAGE_KEY
from helion._compiler.cute.cute_flash import FLASH_LEGACY_CONFIG_KEYS
from helion._compiler.cute.cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
from helion._compiler.cute.cute_flash import FLASH_LOCAL_TMA_PARTITION_KEY
from helion._compiler.cute.cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
from helion._compiler.cute.cute_flash import FLASH_MMA_INTERLEAVE_KEY
from helion._compiler.cute.cute_flash import FLASH_OTHER_REGS_KEY
from helion._compiler.cute.cute_flash import FLASH_P_STORE_REP_KEY
from helion._compiler.cute.cute_flash import FLASH_PACKED_REDUCE_KEY
from helion._compiler.cute.cute_flash import FLASH_PERSISTENT_CTAS_PER_SM_KEY
from helion._compiler.cute.cute_flash import FLASH_PERSISTENT_KEY
from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILIES
from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
from helion._compiler.cute.cute_flash import FLASH_PRECOMPUTE_QK_DESC_KEY
from helion._compiler.cute.cute_flash import FLASH_Q_TILE_COUNT_KEY
from helion._compiler.cute.cute_flash import FLASH_RECOMPUTE_TILE_COORDS_KEY
from helion._compiler.cute.cute_flash import FLASH_RESCALE_CHUNK_COLS_KEY
from helion._compiler.cute.cute_flash import FLASH_RESCALE_THRESHOLD_KEY
from helion._compiler.cute.cute_flash import FLASH_ROLE_MAP_KEY
from helion._compiler.cute.cute_flash import FLASH_S_LOAD_REP_KEY
from helion._compiler.cute.cute_flash import FLASH_S_STAGE_KEY
from helion._compiler.cute.cute_flash import FLASH_SKIP_RESCALE_STATS_KEY
from helion._compiler.cute.cute_flash import FLASH_SMALL_BIASED_KEY
from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
from helion._compiler.cute.cute_flash import FLASH_SPLIT_P_ARRIVE_KEY
from helion._compiler.cute.cute_flash import FLASH_STAT_TRANSPORT_KEY
from helion._compiler.cute.cute_flash import FLASH_TENSOR_4D_TMA_KEY
from helion._compiler.cute.cute_flash import FLASH_TOPOLOGY_KEY
from helion._compiler.cute.cute_flash import FLASH_USE_2CTA_KEY
from helion._compiler.cute.cute_flash import FLASH_WAIT_HINT_KEY
from helion._compiler.cute.cute_flash import flash_attention_seed_config
from helion._compiler.cute.cute_flash import flash_attention_seed_configs
from helion._compiler.cute.cute_flash import flash_autotune_fragments
from helion._compiler.cute.cute_flash import flash_effective_config_values
from helion._compiler.cute.cute_flash import resolve_flash_config
from helion._compiler.cute.flash_policy import get_flash_target_policy
from helion._compiler.cute.flash_tuning import FlashCausalTuningPolicy
from helion._compiler.cute.flash_tuning import FlashDenseTuningPolicy
from helion._compiler.cute.flash_tuning import FlashSoftmaxLowering
from helion._compiler.cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
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
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
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
from helion._testing import skipIfTileIR
from helion.autotuner import IntegerFragment
from helion.autotuner.base_cache import BoundKernelInMemoryCacheKey
from helion.autotuner.base_cache import LooseAutotuneCacheKey
from helion.autotuner.base_cache import StrictAutotuneCacheKey
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CoResidencyGroup
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import ReductionCategory
from helion.autotuner.config_spec import ReductionDescriptor
from helion.autotuner.config_spec import ReductionKernelFact
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl
from helion.runtime.kernel import _find_device as runtime_find_device
from helion.runtime.kernel import _input_tensor_metadata
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
GB300_HARDWARE = HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA GB300",
    runtime_version="12.8",
    compute_capability="sm100",
)
A100_HARDWARE = HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA A100",
    runtime_version="12.8",
    compute_capability="sm80",
)


class TestAutotunerHeuristic(TestCase):
    @staticmethod
    def _heuristic(name: str, **attributes: object) -> type[AutotunerHeuristic]:
        return type(
            f"_{name.title().replace('_', '')}",
            (AutotunerHeuristic,),
            {"name": name, "backend": "triton", **attributes},
        )

    @staticmethod
    def _grouped_worklist_seed_families() -> dict[str, list[helion.Config]]:
        def seeds(
            groups: int, n: int, k: int, b_major: str, source_m_tile: int
        ) -> list[helion.Config]:
            return tcgen05_grouped_worklist_seed_configs(
                groups=groups,
                packed_m=groups * (32 if source_m_tile == 32 else 4096),
                n=n,
                k=k,
                b_major=b_major,
                source_m_tile=source_m_tile,
                num_sm=148,
            )

        return {
            "small_k": seeds(6, 4096, 4096, "k", 32),
            "small_n": seeds(6, 4096, 4096, "n", 32),
            "source224": tcgen05_grouped_worklist_seed_configs(
                groups=6,
                packed_m=6 * 224,
                n=7168,
                k=3072,
                b_major="k",
                source_m_tile=224,
                num_sm=148,
            ),
            "source256_k": seeds(8, 4096, 2048, "k", 256),
            "source256_n": seeds(8, 4096, 2048, "n", 256),
        }

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

    def test_compiler_fact_hook_runs_once_before_disabled_return(self) -> None:
        fact_calls: list[tuple[object, object]] = []

        class FactOwningHeuristic(AutotunerHeuristic):
            name = "fact_owning_heuristic"
            backend = "synthetic"

            @classmethod
            def register_facts(
                cls, env: object, device_ir: object
            ) -> frozenset[CompilerHeuristicSpecializationFact]:
                fact_calls.append((env, device_ir))
                return frozenset({"input_tensor_metadata"})

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                raise AssertionError("disabled heuristics should not be queried")

        class PassiveHeuristic(AutotunerHeuristic):
            name = "passive_heuristic"
            backend = "synthetic"

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                raise AssertionError("disabled heuristics should not be queried")

        env = MagicMock()
        env.backend_name = "synthetic"
        env.config_spec = MagicMock()
        env.settings = Settings(disable_autotuner_heuristics=True)
        device_ir = MagicMock()

        with patch(
            "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
            {"synthetic": (FactOwningHeuristic, PassiveHeuristic)},
        ):
            self.assertEqual(compiler_seed_configs(env, device_ir), [])

        self.assertEqual(fact_calls, [(env, device_ir)])
        self.assertEqual(
            env.compiler_fact_specialization_facts,
            frozenset({"input_tensor_metadata"}),
        )

    def test_compiler_fact_hook_failure_propagates(self) -> None:
        class FailingFactHeuristic(AutotunerHeuristic):
            name = "failing_fact_heuristic"
            backend = "synthetic"

            @classmethod
            def register_facts(
                cls, env: object, device_ir: object
            ) -> frozenset[CompilerHeuristicSpecializationFact]:
                raise RuntimeError("synthetic correctness-fact failure")

            @classmethod
            def is_eligible(cls, env: object, device_ir: object) -> bool:
                raise AssertionError("fact registration must fail first")

        env = MagicMock()
        env.backend_name = "synthetic"
        env.config_spec = MagicMock()
        env.settings = Settings(disable_autotuner_heuristics=True)

        with (
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"synthetic": (FailingFactHeuristic,)},
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "synthetic correctness-fact failure",
            ),
        ):
            compiler_seed_configs(env, MagicMock())

    def test_cute_flash_disable_heuristics_keeps_value_priors(self) -> None:
        spec = ConfigSpec(backend=CuteBackend())
        self.assertIsNone(spec.compiler_seed_timeout_retry_repetitions)
        for block_id, size_hint in enumerate((1, 128, 128)):
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=size_hint)
            )
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=64,
            block_size_targets={0: 1, 1: 128, 2: 128},
            is_causal=True,
        )
        self.assertIsNone(spec.compiler_seed_timeout_retry_repetitions)
        env = MagicMock()
        env.backend_name = "cute"
        env.config_spec = spec
        env.settings = Settings(disable_autotuner_heuristics=True)

        self.assertEqual(compiler_seed_configs(env, MagicMock()), [])
        self.assertIsNone(spec.compiler_seed_timeout_retry_repetitions)
        self.assertEqual(spec.compiler_seed_configs, [])
        self.assertEqual(spec.autotuner_heuristics, [])
        self.assertTrue(spec.autotune_seed_configs())

        config_gen = spec.create_config_generation()
        self.assertIn(FLASH_PIPELINE_FAMILY_KEY, config_gen._config_value_priors)
        self.assertIn(FLASH_CAUSAL_KV_ORDER_KEY, config_gen._config_value_priors)
        population = config_gen.random_population(4)

        self.assertGreaterEqual(len(population), 4)
        self.assertIn(spec.default_config(), population)

    def test_cute_flash_seed_enables_bounded_timeout_retry(self) -> None:
        spec = ConfigSpec(backend=CuteBackend())
        for block_id, size_hint in enumerate((1, 128, 128)):
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=size_hint)
            )
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=64,
            block_size_targets={0: 1, 1: 128, 2: 128},
        )
        env = MagicMock()
        env.config_spec = spec

        seed = CuteFlashAttentionHeuristic.get_seed_config(env, MagicMock())

        self.assertIsNotNone(seed)
        self.assertEqual(spec.compiler_seed_timeout_retry_repetitions, 3)

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

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler heuristic metadata is not collected in ref eager mode")
    def test_register_heuristic_metadata_is_literal_identity(self) -> None:
        @helion.kernel(backend="triton")
        def metadata_kernel(
            x: torch.Tensor,
            semantic_value: hl.constexpr,
            label: hl.constexpr,
            enabled: hl.constexpr,
            threshold: hl.constexpr,
        ) -> torch.Tensor:
            value = hl.register_heuristic_metadata(
                "test.semantic_value", semantic_value
            )
            hl.register_heuristic_metadata("test.semantic_value", value)
            hl.register_heuristic_metadata("test.label", label)
            hl.register_heuristic_metadata("test.enabled", enabled)
            hl.register_heuristic_metadata("test.threshold", threshold)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] + value  # pyrefly: ignore[unsupported-operation]
            return out

        bound = metadata_kernel.bind(
            (
                torch.empty([8], device=DEVICE, dtype=torch.float32),
                7,
                "layout-aware",
                True,
                float("-inf"),
            )
        )

        self.assertEqual(
            bound.config_spec.compiler_heuristic_metadata,
            {
                "test.semantic_value": 7,
                "test.label": "layout-aware",
                "test.enabled": True,
                "test.threshold": float("-inf"),
            },
        )
        bound.to_triton_code(bound.config_spec.default_config())

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler heuristic metadata is not collected in ref eager mode")
    def test_register_heuristic_metadata_rejects_conflicting_duplicate(self) -> None:
        @helion.kernel(backend="triton")
        def conflicting_metadata(
            x: torch.Tensor,
            first: hl.constexpr,
            second: hl.constexpr,
        ) -> torch.Tensor:
            hl.register_heuristic_metadata("test.semantic_value", first)
            hl.register_heuristic_metadata("test.semantic_value", second)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile]
            return out

        with self.assertRaisesRegex(
            Exception,
            "conflicting compiler heuristic metadata",
        ):
            conflicting_metadata.bind(
                (torch.empty([8], device=DEVICE, dtype=torch.float32), 7, 8)
            )

    def test_should_promote_gate(self) -> None:
        # should_promote() gates a seed's PROMOTION (becoming the autotune-off
        # default) on PROMOTE_TARGETS, independently of where the seed fires.
        env = MagicMock()
        env.device = "cuda"
        all_arches = self._heuristic("all_arches", promote_seed_to_default=True)
        sm90_only = self._heuristic(
            "sm90_only",
            promote_seed_to_default=True,
            PROMOTE_TARGETS=(("cuda", "sm90"),),
        )
        not_promoting = self._heuristic(
            "not_promoting",
            promote_seed_to_default=False,
            PROMOTE_TARGETS=(("cuda", "sm90"),),
        )
        b200_only = self._heuristic(
            "b200_only",
            promote_seed_to_default=True,
            PROMOTE_TARGETS=(("cuda", "sm100"),),
            PROMOTE_HARDWARE_NAMES=frozenset({"NVIDIA B200"}),
        )

        # PROMOTE_TARGETS=None promotes without consulting hardware.
        self.assertTrue(all_arches.should_promote(env))
        for name, heuristic, hardware, expected in (
            ("matching arch", sm90_only, HOPPER_HARDWARE, True),
            ("off-target arch", sm90_only, BLACKWELL_HARDWARE, False),
            ("promotion disabled", not_promoting, HOPPER_HARDWARE, False),
            ("matching hardware", b200_only, BLACKWELL_HARDWARE, True),
            ("different hardware", b200_only, GB300_HARDWARE, False),
        ):
            with (
                self.subTest(name=name),
                patch("helion._hardware.get_hardware_info", return_value=hardware),
            ):
                self.assertIs(heuristic.should_promote(env), expected)

    def test_should_promote_declines_on_unclassifiable_device(self) -> None:
        # On a device Helion cannot classify (e.g. MTIA), get_hardware_info raises
        # RuntimeError. should_promote must degrade to "do not promote" rather than
        # letting the exception escape and crash compilation.
        env = MagicMock()
        env.device = "mtia"

        sm90_only = self._heuristic(
            "sm90_only",
            promote_seed_to_default=True,
            PROMOTE_TARGETS=(("cuda", "sm90"),),
        )

        with patch(
            "helion._hardware.get_hardware_info",
            side_effect=RuntimeError(
                "No supported GPU or TPU device found. "
                "Helion requires CUDA, ROCm, XPU, or TPU."
            ),
        ):
            self.assertFalse(sm90_only.should_promote(env))

    def test_promotion_specialization_key_tracks_only_exact_name_gates(self) -> None:
        arch_only = self._heuristic(
            "arch_only",
            promote_seed_to_default=True,
            PROMOTE_TARGETS=(("cuda", "sm100"),),
        )
        b200_only = self._heuristic(
            "b200_only",
            promote_seed_to_default=True,
            PROMOTE_TARGETS=(("cuda", "sm100"),),
            PROMOTE_HARDWARE_NAMES=frozenset({"NVIDIA B200"}),
        )

        device = torch.device("cuda:0")
        with (
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"triton": (arch_only,)},
            ),
            patch("helion._hardware.get_hardware_info") as hardware_info,
        ):
            self.assertEqual(
                compiler_promotion_specialization_key("triton", device), ()
            )
            hardware_info.assert_not_called()

        with (
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"triton": (arch_only, b200_only)},
            ),
            patch(
                "helion._hardware.get_hardware_info",
                side_effect=(
                    BLACKWELL_HARDWARE,
                    GB300_HARDWARE,
                    BLACKWELL_HARDWARE,
                ),
            ) as hardware_info,
        ):
            b200_key = compiler_promotion_specialization_key("triton", device)
            gb300_key = compiler_promotion_specialization_key("triton", device)
            b200_key_again = compiler_promotion_specialization_key("triton", device)
            self.assertEqual(hardware_info.call_count, 3)

        self.assertEqual(b200_key, (("b200_only", "NVIDIA B200"),))
        self.assertEqual(gb300_key, (("b200_only", None),))
        self.assertEqual(b200_key_again, b200_key)

    def test_promotion_key_canonicalizes_indexless_current_device(self) -> None:
        b200_only = self._heuristic(
            "b200_only",
            promote_seed_to_default=True,
            PROMOTE_TARGETS=(("cuda", "sm100"),),
            PROMOTE_HARDWARE_NAMES=frozenset({"NVIDIA B200"}),
        )

        devices = (torch.device("cuda:0"), torch.device("cuda:1"))
        hardware_by_device = {
            devices[0]: BLACKWELL_HARDWARE,
            devices[1]: GB300_HARDWARE,
        }
        observed_devices: list[torch.device] = []

        def get_hardware_info(device: torch.device) -> HardwareInfo:
            observed_devices.append(device)
            return hardware_by_device[device]

        with (
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"triton": (b200_only,)},
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", side_effect=(0, 1)),
            patch(
                "helion._hardware.get_hardware_info",
                side_effect=get_hardware_info,
            ),
        ):
            indexless = torch.device("cuda")
            b200_key = compiler_promotion_specialization_key("triton", indexless)
            gb300_key = compiler_promotion_specialization_key("triton", indexless)

        self.assertEqual(b200_key, (("b200_only", "NVIDIA B200"),))
        self.assertEqual(gb300_key, (("b200_only", None),))
        self.assertEqual(observed_devices, list(devices))

    def test_promotion_key_cache_tracks_mutated_registry_policy(self) -> None:
        mutable_promotion = self._heuristic(
            "mutable_promotion",
            promote_seed_to_default=True,
            PROMOTE_HARDWARE_NAMES=frozenset({"NVIDIA B200"}),
        )

        device = torch.device("cuda:0")
        with (
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"triton": (mutable_promotion,)},
            ),
            patch(
                "helion._hardware.get_hardware_info",
                side_effect=(BLACKWELL_HARDWARE, GB300_HARDWARE),
            ) as hardware_info,
        ):
            self.assertEqual(
                compiler_promotion_specialization_key("triton", device),
                (("mutable_promotion", "NVIDIA B200"),),
            )
            mutable_promotion.PROMOTE_HARDWARE_NAMES = frozenset({"NVIDIA GB300"})
            self.assertEqual(
                compiler_promotion_specialization_key("triton", device),
                (("mutable_promotion", "NVIDIA GB300"),),
            )
            mutable_promotion.promote_seed_to_default = False
            self.assertEqual(
                compiler_promotion_specialization_key("triton", device), ()
            )
            self.assertEqual(hardware_info.call_count, 2)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler seed specialization is not used in ref eager mode")
    def test_compiler_seed_specialization_rebinds_only_eligible_kernels(
        self,
    ) -> None:
        device_num_sm_sensitive = self._heuristic(
            "device_num_sm_sensitive",
            CACHE_SPECIALIZATION_FACTS=frozenset({"device_num_sm"}),
        )
        config_num_sm_sensitive = self._heuristic(
            "config_num_sm_sensitive",
            CACHE_SPECIALIZATION_FACTS=frozenset({"config_num_sm"}),
        )
        input_metadata_sensitive = self._heuristic(
            "input_tensor_metadata_sensitive",
            CACHE_SPECIALIZATION_FACTS=frozenset({"input_tensor_metadata"}),
        )
        unrelated = self._heuristic("unrelated")

        self.assertEqual(
            CuteTcgen05GroupedWorklistHeuristic.CACHE_SPECIALIZATION_FACTS,
            frozenset({"config_num_sm", "input_tensor_metadata"}),
        )
        for heuristic in (
            CuteTcgen05ClusterM2Heuristic,
            TritonH100MatmulHeuristic,
            TritonPointwiseSeedHeuristic,
            TritonStandardReductionHeuristicSM90,
            TritonStandardReductionHeuristicSM100,
            TritonUserTiledReductionHeuristicSM90,
            TritonUserTiledReductionHeuristicSM100,
        ):
            self.assertEqual(
                heuristic.CACHE_SPECIALIZATION_FACTS,
                frozenset({"device_num_sm"}),
            )

        heuristics = (
            device_num_sm_sensitive,
            config_num_sm_sensitive,
            input_metadata_sensitive,
            unrelated,
        )
        with patch(
            "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
            {"triton": heuristics},
        ):
            for heuristic, expected in (
                (device_num_sm_sensitive, frozenset({"device_num_sm"})),
                (config_num_sm_sensitive, frozenset({"config_num_sm"})),
                (
                    input_metadata_sensitive,
                    frozenset({"input_tensor_metadata"}),
                ),
                (unrelated, frozenset()),
            ):
                self.assertEqual(
                    compiler_seed_specialization_facts("triton", [heuristic.name]),
                    expected,
                )

        current_num_sm = 148

        def get_num_sm(
            _device: torch.device,
            *,
            reserved_sms: int = 0,
        ) -> int:
            return max(current_num_sm - reserved_sms, 1)

        def make_kernel(
            heuristic_name: str,
            *,
            persistent_reserved_sms: int = 0,
            static_shapes: bool = True,
        ) -> tuple[
            helion.Kernel,
            Callable[[MagicMock, object], list[helion.Config]],
        ]:
            @helion.kernel(
                backend="triton",
                persistent_reserved_sms=persistent_reserved_sms,
                static_shapes=static_shapes,
            )
            def identity(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(x.shape):
                    out[tile] = x[tile]
                return out

            def seed_configs(env: MagicMock, _device_ir: object) -> list[helion.Config]:
                env.config_spec.autotuner_heuristics = [heuristic_name]
                return []

            return identity, seed_configs

        x = torch.empty([8], device=DEVICE)
        wider_x = torch.empty([16], device=DEVICE)

        def bind_kernel(
            heuristic_name: str,
            cases: tuple[tuple[tuple[torch.Tensor, ...], int], ...],
            *,
            persistent_reserved_sms: int = 0,
            static_shapes: bool = True,
        ) -> tuple[helion.Kernel, list[Any]]:
            nonlocal current_num_sm
            kernel, seed_configs = make_kernel(
                heuristic_name,
                persistent_reserved_sms=persistent_reserved_sms,
                static_shapes=static_shapes,
            )
            bounds = []
            with patch(
                "helion.runtime.kernel.compiler_seed_configs",
                side_effect=seed_configs,
            ):
                for args, num_sm in cases:
                    current_num_sm = num_sm
                    bounds.append(kernel.bind(args))
            return kernel, bounds

        def specialization_values(kernel: helion.Kernel, fact: str) -> set[object]:
            values = set()
            for key in kernel._bound_kernels:
                [(name, value)] = cast(
                    "tuple[tuple[str, object], ...]",
                    key.compiler_seed_results,
                )
                self.assertEqual(name, fact)
                values.add(value)
            return values

        def assert_bind_case(
            heuristic_name: str,
            cases: tuple[tuple[tuple[torch.Tensor, ...], int], ...],
            *,
            same: tuple[tuple[int, int], ...] = (),
            different: tuple[tuple[int, int], ...] = (),
            fact: str | None = None,
            expected_values: set[object] | None = None,
            persistent_reserved_sms: int = 0,
            static_shapes: bool = True,
        ) -> tuple[helion.Kernel, list[Any]]:
            kernel, bounds = bind_kernel(
                heuristic_name,
                cases,
                persistent_reserved_sms=persistent_reserved_sms,
                static_shapes=static_shapes,
            )
            for pairs, assertion in (
                (same, self.assertIs),
                (different, self.assertIsNot),
            ):
                for first_index, second_index in pairs:
                    assertion(bounds[first_index], bounds[second_index])
            if fact is None:
                self.assertFalse(
                    any(key.compiler_seed_results for key in kernel._bound_kernels)
                )
            else:
                self.assertEqual(specialization_values(kernel, fact), expected_values)
            return kernel, bounds

        with (
            patch(
                "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                {"triton": heuristics},
            ),
            patch(
                "helion.runtime.get_num_sm", side_effect=get_num_sm
            ) as get_num_sm_mock,
        ):
            sensitive, (first, second, rebound) = bind_kernel(
                device_num_sm_sensitive.name,
                (((x,), 148), ((x,), 132), ((x,), 148)),
            )

            self.assertIsNot(first, second)
            self.assertIs(rebound, first)
            self.assertEqual(len(sensitive._bound_kernels), 2)
            self.assertEqual(
                specialization_values(sensitive, "device_num_sm"),
                {132, 148},
            )
            self.assertTrue(
                all(not key.extra_results for key in sensitive._bound_kernels)
            )
            first._run = lambda value: value
            second._run = lambda value: value
            current_num_sm = 148
            get_num_sm_calls = get_num_sm_mock.call_count
            with (
                patch.object(sensitive, "bind", wraps=sensitive.bind) as bind,
                patch(
                    "helion.runtime.kernel._find_device",
                    wraps=runtime_find_device,
                ) as find_device,
            ):
                self.assertIs(first(x), x)
                self.assertIs(first(x), x)
            self.assertEqual(bind.call_count, 0)
            self.assertEqual(find_device.call_count, 0)
            self.assertEqual(get_num_sm_mock.call_count, get_num_sm_calls)

            current_num_sm = 132
            different_device = torch.device("cuda", (x.device.index or 0) + 1)
            first._compiler_seed_device_resolver = MagicMock(
                return_value=different_device
            )
            with patch.object(sensitive, "bind", return_value=second) as bind:
                self.assertIs(first(x), x)
            bind.assert_called_once_with((x,))
            self.assertEqual(get_num_sm_mock.call_count, get_num_sm_calls + 1)

            current_num_sm = 148
            same_num_sm_device = torch.device("cuda", (x.device.index or 0) + 2)
            first._compiler_seed_device_resolver = MagicMock(
                return_value=same_num_sm_device
            )
            same_num_sm_calls = get_num_sm_mock.call_count
            with patch.object(sensitive, "bind", wraps=sensitive.bind) as bind:
                self.assertIs(first(x), x)
                calls_after_new_device = get_num_sm_mock.call_count
                self.assertIs(first(x), x)
            bind.assert_called_once_with((x,))
            self.assertEqual(calls_after_new_device, same_num_sm_calls + 2)
            self.assertEqual(get_num_sm_mock.call_count, calls_after_new_device)

            assert_bind_case(
                config_num_sm_sensitive.name,
                (((x,), 148), ((x,), 132)),
                different=((0, 1),),
                fact="config_num_sm",
                expected_values={124, 140},
                persistent_reserved_sms=8,
            )

            _metadata_sensitive, metadata_bounds = assert_bind_case(
                input_metadata_sensitive.name,
                (
                    ((x,), 148),
                    ((wider_x,), 148),
                    ((torch.empty_like(x),), 148),
                ),
                same=((0, 2),),
                different=((0, 1),),
                fact="input_tensor_metadata",
                expected_values={
                    (((("sequence", tuple, 0),), (8,), (1,)),),
                    (((("sequence", tuple, 0),), (16,), (1,)),),
                },
                static_shapes=False,
            )
            first_metadata, second_metadata, _rebound_metadata = metadata_bounds
            first_metadata._run = lambda _value: x
            second_metadata._run = lambda _value: wider_x
            self.assertIs(first_metadata(wider_x), wider_x)

            assert_bind_case(
                input_metadata_sensitive.name,
                (((x,), 148), ((wider_x,), 148)),
                different=((0, 1),),
            )

            for static_shapes, cases in (
                (True, (((x,), 148), ((x,), 132))),
                (False, (((x,), 148), ((wider_x,), 148))),
            ):
                with self.subTest(unrelated_static_shapes=static_shapes):
                    kernel, _bounds = assert_bind_case(
                        unrelated.name,
                        cases,
                        same=((0, 1),),
                        static_shapes=static_shapes,
                    )
                    self.assertEqual(len(kernel._bound_kernels), 1)

    def test_argument_device_resolver_canonicalizes_indexless_cuda(self) -> None:
        resolver = _DeviceResolver.from_values((torch.device("cuda"),))

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", side_effect=(0, 1)) as current_device,
        ):
            self.assertEqual(
                resolver((torch.device("cuda"),)),
                torch.device("cuda", 0),
            )
            self.assertEqual(
                resolver((torch.device("cuda"),)),
                torch.device("cuda", 1),
            )

        self.assertEqual(current_device.call_count, 2)

    def test_argument_device_resolver_relearns_invalid_mapping_paths(self) -> None:
        cpu = torch.device("cpu")
        meta = torch.device("meta")
        resolver = _DeviceResolver.from_values(({"empty": None, "device": meta},))
        first_path = resolver.path

        self.assertEqual(resolver(({"empty": None, "device": meta},)), meta)
        self.assertEqual(resolver.path, first_path)

        # Moving the selected key invalidates its learned mapping position even
        # when the selected device is unchanged.
        self.assertEqual(resolver(({"device": meta, "empty": None},)), meta)
        reordered_path = resolver.path
        self.assertNotEqual(reordered_path, first_path)

        # A missing key and a changed container type both trigger discovery of
        # the first device under the new structure.
        self.assertEqual(resolver(({"replacement": cpu},)), cpu)
        self.assertEqual(resolver(([None, meta],)), meta)

    def test_input_tensor_metadata_tracks_roles_not_mapping_order(self) -> None:
        narrow = torch.empty_strided((8, 4), (4, 1))
        wide = torch.empty_strided((16, 4), (5, 1))
        original = _input_tensor_metadata(
            ({"lhs": narrow, "rhs": wide},),
        )
        reordered = _input_tensor_metadata(
            ({"rhs": wide, "lhs": narrow},),
        )
        swapped_roles = _input_tensor_metadata(
            ({"rhs": narrow, "lhs": wide},),
        )

        self.assertEqual(original, reordered)
        self.assertNotEqual(original, swapped_roles)
        self.assertEqual(
            {record[0][-1][-1] for record in original},
            {"lhs", "rhs"},
        )

    def test_input_tensor_metadata_tracks_dataclass_field_roles(self) -> None:
        @dataclasses.dataclass
        class Inputs:
            lhs: torch.Tensor
            rhs: torch.Tensor

        narrow = torch.empty_strided((8, 4), (4, 1))
        wide = torch.empty_strided((16, 4), (5, 1))
        original = _input_tensor_metadata((Inputs(narrow, wide),))
        swapped_roles = _input_tensor_metadata((Inputs(wide, narrow),))

        self.assertNotEqual(original, swapped_roles)
        self.assertEqual(
            {record[0][-1][-1] for record in original},
            {"lhs", "rhs"},
        )

    def test_compiler_seed_specialization_is_in_cache_hash(self) -> None:
        bound_keys = [
            BoundKernelInMemoryCacheKey(
                (),
                (),
                compiler_seed_results=(("config_num_sm", num_sm),),
            )
            for num_sm in (132, 148)
        ]

        self.assertNotEqual(bound_keys[0].stable_hash(), bound_keys[1].stable_hash())
        self.assertIn("compiler_seed_results", repr(bound_keys[0]))
        self.assertEqual(
            repr(BoundKernelInMemoryCacheKey((), ())),
            "BoundKernelInMemoryCacheKey(specialization_key=(), extra_results=())",
        )

        common = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "source",
            "hardware": "NVIDIA B200",
            "runtime_name": "13.0",
            "backend": "cute",
        }
        keys = [
            LooseAutotuneCacheKey(
                **common,
                compiler_seed_results=(
                    ("config_num_sm", 148),
                    (
                        "input_tensor_metadata",
                        (((("sequence", tuple, 0),), (size,), (1,)),),
                    ),
                ),
            )
            for size in (8, 16)
        ]

        self.assertNotEqual(keys[0].stable_hash(), keys[1].stable_hash())
        self.assertIn("compiler_seed_results", repr(keys[0]))

        strict_keys = [
            StrictAutotuneCacheKey(
                **common,
                compiler_seed_results=(("config_num_sm", num_sm),),
                helion_key="helion",
                torch_key="torch",
                triton_key="triton",
            )
            for num_sm in (132, 148)
        ]
        self.assertNotEqual(
            strict_keys[0].stable_hash(),
            strict_keys[1].stable_hash(),
        )
        self.assertIn("compiler_seed_results", repr(strict_keys[0]))

    def test_grouped_worklist_seeds_are_packing_compatible_and_ranked(self) -> None:
        spec = ConfigSpec(backend=CuteBackend())
        spec.cute_tcgen05_search_enabled = True
        for block_id, size_hint in enumerate((256, 4096, 4096)):
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=size_hint)
            )
        fact = MatmulFact(
            lhs_ndim=2,
            rhs_ndim=3,
            m_block_id=0,
            n_block_id=1,
            k_block_id=2,
            static_m=256,
            static_n=4096,
            static_k=4096,
            lhs_dtype=torch.bfloat16,
            rhs_dtype=torch.bfloat16,
        )
        spec.matmul_facts = [fact]
        env = MagicMock(config_spec=spec)
        self.assertIs(_tcgen05_grouped_fact(env), fact)
        self.assertIs(_tcgen05_grouped_worklist_fact(env), fact)
        dynamic_fact = fact._replace(static_n=None, static_k=None)
        spec.matmul_facts = [dynamic_fact]
        self.assertIsNone(_tcgen05_grouped_fact(env))
        self.assertIs(_tcgen05_grouped_worklist_fact(env), dynamic_fact)
        fp16_fact = fact._replace(
            lhs_dtype=torch.float16,
            rhs_dtype=torch.float16,
        )
        spec.matmul_facts = [fp16_fact]
        self.assertIs(_tcgen05_grouped_fact(env), fp16_fact)
        self.assertIsNone(_tcgen05_grouped_worklist_fact(env))

        families = self._grouped_worklist_seed_families()
        small = families["small_k"]
        k_major = families["source256_k"]
        legacy = families["source224"]
        n_major = families["source256_n"]

        for configs, key, expected in (
            (small, TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, 52),
            (k_major, TCGEN05_CONSUMER_REGS_CONFIG_KEY, 240),
            (n_major, TCGEN05_CONSUMER_REGS_CONFIG_KEY, 240),
            (legacy, TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 8),
            (legacy, TCGEN05_CONSUMER_REGS_CONFIG_KEY, 256),
        ):
            self.assertEqual(configs[0].config[key], expected)
        self.assertNotIn(
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
            k_major[0].config,
        )
        self.assertEqual(
            n_major[0].config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY],
            Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )
        for source_m_tile, configs in (
            (32, small),
            (224, legacy),
            (256, k_major),
            (256, n_major),
        ):
            self.assertTrue(configs)
            self.assertTrue(
                all(
                    config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY]
                    == TCGEN05_GROUPED_MODE_WORKLIST_NM
                    and config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY]
                    == source_m_tile
                    for config in configs
                )
            )

    def test_grouped_worklist_seed_family_covers_reviewed_variants(self) -> None:
        def block_k(values: dict[str, object]) -> int:
            return cast("list[int]", values["block_sizes"])[2]

        families = self._grouped_worklist_seed_families()
        small = families["small_n"]
        source224 = families["source224"]
        source256 = families["source256_n"]

        small_values = [config.config for config in small]
        self.assertTrue(
            {32, 52}.issubset(
                {
                    values.get(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY)
                    for values in small_values
                    if block_k(values) == 128
                }
            )
        )
        self.assertTrue(
            any(
                block_k(values) == 128
                and TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY not in values
                for values in small_values
            )
        )
        self.assertTrue(
            {4, 8}.issubset(
                {
                    values.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
                    for values in small_values
                    if values.get(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY) == 20
                }
            )
        )
        self.assertTrue(
            any(
                block_k(values) == 64
                and values.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY) is True
                and TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY not in values
                for values in small_values
            )
        )
        self.assertTrue(
            any(
                values.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY) is not True
                for values in small_values
            )
        )

        source224_values = [config.config for config in source224]
        self.assertEqual(
            {
                values["tcgen05_ab_stages"]
                for values in source224_values
                if values.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY) is not True
            },
            {4, 5, 6, 7},
        )
        self.assertTrue(
            any(
                values.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY) == 8
                and values.get(TCGEN05_CONSUMER_REGS_CONFIG_KEY) == 256
                for values in source224_values
            )
        )
        self.assertTrue(any(block_k(values) == 128 for values in source224_values))

        source256_values = [config.config for config in source256]
        self.assertEqual(
            {
                (block_k(values), values["tcgen05_ab_stages"])
                for values in source256_values
            },
            {(64, 5), (64, 6), (128, 3)},
        )
        self.assertTrue(
            {224, 240, 256}.issubset(
                {
                    values.get(TCGEN05_CONSUMER_REGS_CONFIG_KEY)
                    for values in source256_values
                }
            )
        )
        self.assertTrue(
            {8, 16}.issubset(
                {
                    values.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
                    for values in source256_values
                }
            )
        )
        self.assertTrue(
            any(
                values.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                == Tcgen05PersistenceModel.CLC_PERSISTENT.value
                for values in source256_values
            )
        )

    def test_grouped_worklist_clc_requires_full_physical_n_tile(self) -> None:
        configs = tcgen05_grouped_worklist_seed_configs(
            groups=8,
            packed_m=20_480,
            n=416,
            k=192,
            b_major="n",
            source_m_tile=256,
            num_sm=148,
        )

        self.assertTrue(configs)
        self.assertTrue(
            all(
                config.config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                != Tcgen05PersistenceModel.CLC_PERSISTENT.value
                for config in configs
            )
        )
        self.assertNotEqual(
            configs[0].config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY),
            Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )

    def test_grouped_worklist_clc_respects_cuda_grid_z_limit(self) -> None:
        for m_tiles, n_tiles, expect_clc in (
            (257, 255, True),
            (258, 255, False),
        ):
            with self.subTest(
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                expect_clc=expect_clc,
            ):
                configs = tcgen05_grouped_worklist_seed_configs(
                    groups=1,
                    packed_m=m_tiles * 256,
                    n=n_tiles * 256,
                    k=65_536,
                    b_major="n",
                    source_m_tile=256,
                    num_sm=148,
                )
                clc_configs = [
                    config
                    for config in configs
                    if config.config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                    == Tcgen05PersistenceModel.CLC_PERSISTENT.value
                ]

                self.assertEqual(bool(clc_configs), expect_clc)
                self.assertEqual(
                    m_tiles * n_tiles
                    <= TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS,
                    expect_clc,
                )
                if expect_clc:
                    self.assertEqual(
                        configs[0].config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY),
                        Tcgen05PersistenceModel.CLC_PERSISTENT.value,
                    )
                else:
                    self.assertNotEqual(
                        configs[0].config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY),
                        Tcgen05PersistenceModel.CLC_PERSISTENT.value,
                    )

    def test_grouped_worklist_seeds_filter_unreachable_block_k(self) -> None:
        configs = tcgen05_grouped_worklist_seed_configs(
            groups=8,
            packed_m=8 * 4096,
            n=4096,
            k=2048,
            b_major="n",
            source_m_tile=256,
            num_sm=148,
        )

        for block_k in (64, 128):
            spec = ConfigSpec(backend=CuteBackend())
            for block_id, value in enumerate((256, 128, block_k)):
                spec.block_sizes.append(
                    BlockSizeSpec(
                        block_id=block_id,
                        size_hint=value,
                        min_size=value,
                        max_size=value,
                    )
                )
            reachable = _filter_reachable_block_size_configs(spec, configs)
            self.assertTrue(reachable)
            self.assertTrue(
                all(
                    cast("list[int]", config.config["block_sizes"])[2] == block_k
                    for config in reachable
                )
            )
        bk128_only = ConfigSpec(
            backend=CuteBackend(),
            target_device_capability=(10, 0),
        )
        bk128_only.cute_tcgen05_search_enabled = True
        for block_id, value in enumerate((256, 128, 128)):
            bk128_only.block_sizes.append(
                BlockSizeSpec(
                    block_id=block_id,
                    size_hint=value,
                    min_size=value,
                    max_size=value,
                )
            )
        bk128_only.matmul_facts = [
            MatmulFact(
                lhs_ndim=2,
                rhs_ndim=3,
                m_block_id=0,
                n_block_id=1,
                k_block_id=2,
                static_m=256,
                static_n=None,
                static_k=None,
                lhs_dtype=torch.bfloat16,
                rhs_dtype=torch.bfloat16,
            )
        ]
        env = MagicMock(config_spec=bk128_only)
        bk128_only.compiler_heuristic_metadata[
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
        ] = 224
        with patch(
            "helion._compiler.autotuner_heuristics.cute."
            "tcgen05_runtime_n_ptx_compatible",
            return_value=True,
        ):
            self.assertIsNotNone(
                CuteTcgen05GroupedWorklistHeuristic._eligible_inputs(env, MagicMock())
            )
            bk128_only.matmul_facts = [bk128_only.matmul_facts[0]._replace(static_n=33)]
            self.assertIsNone(
                CuteTcgen05GroupedWorklistHeuristic._eligible_inputs(env, MagicMock())
            )
            bk128_only.matmul_facts = [
                bk128_only.matmul_facts[0]._replace(static_n=None)
            ]
            bk128_only.compiler_heuristic_metadata[
                TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
            ] = 256
            self.assertIsNotNone(
                CuteTcgen05GroupedWorklistHeuristic._eligible_inputs(env, MagicMock())
            )

    def test_grouped_worklist_seed_only_controls_survive_flattening(self) -> None:
        spec = ConfigSpec(backend=CuteBackend())
        spec.cute_tcgen05_search_enabled = True
        spec.allowed_pid_types = ("flat",)
        families = self._grouped_worklist_seed_families()
        spec.compiler_seed_configs = [
            *families["small_k"],
            *families["source256_n"],
            *families["source256_k"],
        ]
        fields = spec._cute_tcgen05_config.flat_fields()

        def enum(key: str) -> EnumFragment:
            fragment = fields[key]
            self.assertIsInstance(fragment, EnumFragment)
            return cast("EnumFragment", fragment)

        ab_stages = cast("IntegerFragment", fields["tcgen05_ab_stages"])
        self.assertIsInstance(ab_stages, IntegerFragment)
        grouped_mode = enum(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        runtime_direct = enum(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY)
        reserved = enum(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY)
        consumer_regs = enum(TCGEN05_CONSUMER_REGS_CONFIG_KEY)
        strategy = enum(TCGEN05_STRATEGY_CONFIG_KEY)
        persistence = enum(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        scheduler_warps = enum(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY)
        source_m_tile = enum(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
        l2_swizzle = enum(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
        pid_type = enum("pid_type")
        seed_ab_stages = max(
            cast("int", seed.config["tcgen05_ab_stages"])
            for seed in spec.compiler_seed_configs
        )
        self.assertGreater(seed_ab_stages, ab_stages.high)
        self.assertEqual(ab_stages.default(), 2)
        self.assertEqual(ab_stages.cardinality(), 2)
        self.assertEqual(ab_stages.search_values(), [1, 2])
        self.assertEqual(ab_stages.dim(), 1)
        self.assertEqual(ab_stages.encode(seed_ab_stages), [float(seed_ab_stages)])
        self.assertEqual(ab_stages.pattern_neighbors(seed_ab_stages), [2, 1])
        self.assertEqual(
            ab_stages.differential_mutation(seed_ab_stages, 1, 1),
            seed_ab_stages,
        )
        self.assertEqual(ab_stages.differential_mutation(seed_ab_stages, 1, 2), 2)
        self.assertEqual(ab_stages.differential_mutation(seed_ab_stages, 2, 1), 2)
        self.assertTrue(all(ab_stages.random() in (1, 2) for _ in range(32)))
        self.assertEqual(grouped_mode.search_choices, (None,))
        self.assertEqual(runtime_direct.search_choices, (False,))
        self.assertIn(52, reserved.choices)
        self.assertEqual(
            reserved.search_choices,
            TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES,
        )
        self.assertEqual(consumer_regs.search_choices, (256,))
        self.assertEqual(
            strategy.search_choices,
            (Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,),
        )
        self.assertIn(Tcgen05PersistenceModel.CLC_PERSISTENT.value, persistence.choices)
        self.assertEqual(
            scheduler_warps.search_choices,
            (0,),
        )
        self.assertEqual(set(source_m_tile.choices), {None, 32, 256})
        self.assertEqual(source_m_tile.search_choices, (None,))
        self.assertIn(16, l2_swizzle.choices)
        self.assertEqual(l2_swizzle.search_choices, (1, 2, 4, 8))
        self.assertEqual(pid_type.search_choices, ("flat",))

    def test_grouped_worklist_primary_promotes_only_on_b200(self) -> None:
        spec = ConfigSpec(backend=CuteBackend())
        spec.cute_tcgen05_search_enabled = True
        for block_id, size_hint in enumerate((256, 256, 128)):
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=size_hint)
            )
        static_fact = MatmulFact(
            lhs_ndim=2,
            rhs_ndim=2,
            m_block_id=0,
            n_block_id=1,
            k_block_id=2,
            static_m=256,
            static_n=256,
            static_k=128,
            lhs_dtype=torch.bfloat16,
            rhs_dtype=torch.bfloat16,
        )
        spec.matmul_facts = [static_fact]
        env = MagicMock(device=DEVICE, config_spec=spec)
        with (
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
        ):
            self.assertTrue(CuteTcgen05GroupedWorklistHeuristic.should_promote(env))
            spec.matmul_facts = [static_fact._replace(static_k=None)]
            self.assertFalse(CuteTcgen05GroupedWorklistHeuristic.should_promote(env))
            spec.matmul_facts = [
                static_fact._replace(
                    lhs_dtype=torch.float16,
                    rhs_dtype=torch.float16,
                )
            ]
            self.assertFalse(CuteTcgen05GroupedWorklistHeuristic.should_promote(env))
        spec.matmul_facts = [static_fact]
        with (
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch("helion._hardware.get_hardware_info", return_value=GB300_HARDWARE),
        ):
            self.assertFalse(CuteTcgen05GroupedWorklistHeuristic.should_promote(env))

    def test_grouped_worklist_seeds_require_validated_runtime_n_ptx(self) -> None:
        spec = ConfigSpec(
            backend=CuteBackend(),
            target_device_capability=(10, 0),
            num_sm=148,
        )
        spec.cute_tcgen05_search_enabled = True
        for block_id, size_hint in enumerate((256, 256, 128)):
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=size_hint)
            )
        spec.matmul_facts = [
            MatmulFact(
                lhs_ndim=2,
                rhs_ndim=3,
                m_block_id=0,
                n_block_id=1,
                k_block_id=2,
                static_m=256,
                static_n=256,
                static_k=128,
                lhs_dtype=torch.bfloat16,
                rhs_dtype=torch.bfloat16,
            )
        ]
        spec.compiler_heuristic_metadata[
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
        ] = 256
        env = MagicMock(device=DEVICE, config_spec=spec)
        device_ir = MagicMock()
        grouped_fact = spec.matmul_facts[0]

        with (
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=False,
            ),
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "warn_tcgen05_runtime_n_ptx_fallback"
            ) as warn_fallback,
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
        ):
            spec.matmul_facts = []
            self.assertFalse(
                CuteTcgen05GroupedWorklistHeuristic.is_eligible(env, device_ir)
            )
            warn_fallback.assert_not_called()

            spec.matmul_facts = [grouped_fact]
            self.assertFalse(
                CuteTcgen05GroupedWorklistHeuristic.is_eligible(env, device_ir)
            )
            warn_fallback.assert_called_once_with()
            self.assertIsNone(
                CuteTcgen05GroupedWorklistHeuristic.get_seed_config(env, device_ir)
            )
            self.assertEqual(
                CuteTcgen05GroupedWorklistHeuristic.get_seed_configs(env, device_ir),
                [],
            )
            self.assertFalse(CuteTcgen05GroupedWorklistHeuristic.should_promote(env))

    @onlyBackends(["cute"])
    def test_grouped_worklist_dynamic_hints_do_not_narrow_search(self) -> None:
        from pretuned_kernels.grouped_gemm_deepgemm.grouped_gemm_deepgemm import (
            _GROUPED_GEMM_DEEPGEMM_BODY,
        )

        from helion.language.matmul_ops import plan_cute_tcgen05_search

        def make_args(packed_m: int) -> tuple[object, ...]:
            return (
                torch.empty(
                    [packed_m, 128],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    [6, 256, 128],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.empty([6, 4], device=DEVICE, dtype=torch.int32),
                None,
                32,
            )

        kernel = helion.kernel(
            _GROUPED_GEMM_DEEPGEMM_BODY,
            backend="cute",
            static_shapes=False,
        )
        plans = []

        def capture_plan(*args: object, **kwargs: object):
            plan = plan_cute_tcgen05_search(*args, **kwargs)  # pyrefly: ignore
            if plan is not None:
                plans.append(plan)
            return plan

        first_args = make_args(6 * 32)
        rebound_args = make_args(12 * 32)

        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion.language.matmul_ops.plan_cute_tcgen05_search",
                side_effect=capture_plan,
            ),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
        ):
            first = kernel.bind(first_args)
            rebound = kernel.bind(rebound_args)
            first._run = lambda *_args: cast("torch.Tensor", first_args[0])
            rebound._run = lambda *_args: cast("torch.Tensor", rebound_args[0])
            self.assertIs(first(*rebound_args), rebound_args[0])

        self.assertIsNot(rebound, first)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(
                (plan.static_m, plan.static_n, plan.static_k),
                (256, None, None),
            )
        fact = first.config_spec.matmul_facts[0]
        self.assertEqual(
            (fact.static_m, fact.static_n, fact.static_k),
            (256, None, None),
        )
        self.assertEqual(first.config_spec.allowed_pid_types, ("flat",))
        self.assertEqual(first.config_spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertIsNone(first.config_spec._tcgen05_cluster_m2_search_constraints)
        self.assertIsNone(first.config_spec.compiler_default_config)
        # The worklist's group/start/extent loads feed scheduling, indices, and
        # masks rather than the matmul's store value. They must not be mistaken
        # for per-subtile epilogue aux loads, which would expose a
        # ``pre_acc_wait`` search point that every identity-store config rejects.
        self.assertFalse(first.config_spec.cute_tcgen05_aux_kernel_detected)
        self.assertNotIn(
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            first.config_spec._flat_fields(),
        )
        self.assertTrue(
            any(
                config.config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
                == TCGEN05_GROUPED_MODE_WORKLIST_NM
                for config in first.config_spec.compiler_seed_configs
            )
        )
        transfer_errors: list[str] = []
        generation = first.config_spec.create_config_generation()
        seed_pairs = generation.seed_flat_config_pairs(transfer_errors.append)
        self.assertFalse(transfer_errors)
        self.assertTrue(seed_pairs)
        for flat, normalized in seed_pairs:
            self.assertEqual(generation.unflatten([*flat]), normalized)

    @onlyBackends(["cute"])
    def test_grouped_worklist_dynamic_inputs_rebind_without_promotion(self) -> None:
        from pretuned_kernels.grouped_gemm_deepgemm.grouped_gemm_deepgemm import (
            _GROUPED_GEMM_DEEPGEMM_BODY,
        )

        def make_args(
            *,
            n: int = 256,
            k: int = 256,
            b_major: str = "k",
        ) -> tuple[object, ...]:
            if b_major == "k":
                b_grouped = torch.empty(
                    [6, n, k],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                )
            else:
                self.assertEqual(b_major, "n")
                b_grouped = torch.empty(
                    [6, k, n],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ).transpose(1, 2)
            return (
                torch.empty([6 * 32, k], device=DEVICE, dtype=torch.bfloat16),
                b_grouped,
                torch.empty([6, 4], device=DEVICE, dtype=torch.int32),
                None,
                32,
            )

        kernel = helion.kernel(
            _GROUPED_GEMM_DEEPGEMM_BODY,
            backend="cute",
            static_shapes=False,
        )
        current_num_sm = 148

        def get_num_sm(
            _device: torch.device,
            *,
            reserved_sms: int = 0,
        ) -> int:
            return max(current_num_sm - reserved_sms, 1)

        first_args = make_args()
        different_k_args = make_args(k=192)
        different_n_args = make_args(n=240)
        different_layout_args = make_args(b_major="n")

        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
            patch("helion.runtime.get_num_sm", side_effect=get_num_sm),
        ):
            first = kernel.bind(first_args)
            different_k = kernel.bind(different_k_args)
            different_n = kernel.bind(different_n_args)
            different_layout = kernel.bind(different_layout_args)

            first_output = cast("torch.Tensor", first_args[0])
            different_k_output = cast("torch.Tensor", different_k_args[0])
            different_n_output = cast("torch.Tensor", different_n_args[0])
            different_layout_output = cast("torch.Tensor", different_layout_args[0])
            first._run = lambda *_args: first_output
            different_k._run = lambda *_args: different_k_output
            different_n._run = lambda *_args: different_n_output
            different_layout._run = lambda *_args: different_layout_output
            self.assertIs(first(*different_k_args), different_k_output)
            self.assertIs(first(*different_n_args), different_n_output)
            self.assertIs(first(*different_layout_args), different_layout_output)

            current_num_sm = 132
            smaller_sm = kernel.bind(different_k_args)

        self.assertIsNot(different_k, first)
        self.assertIsNot(different_n, first)
        self.assertIsNot(different_layout, first)
        self.assertIsNot(smaller_sm, first)
        self.assertIsNot(smaller_sm, different_k)
        self.assertEqual(len(kernel._bound_kernels), 5)

        def specialization_results(bound: object) -> dict[str, object]:
            matches = [
                dict(
                    cast(
                        "tuple[tuple[str, object], ...]",
                        key.compiler_seed_results,
                    )
                )
                for key, candidate in kernel._bound_kernels.items()
                if candidate is bound
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        first_results = specialization_results(first)
        different_k_results = specialization_results(different_k)
        different_n_results = specialization_results(different_n)
        different_layout_results = specialization_results(different_layout)
        smaller_sm_results = specialization_results(smaller_sm)
        self.assertEqual(first_results["config_num_sm"], 148)
        self.assertEqual(different_k_results["config_num_sm"], 148)
        self.assertEqual(different_n_results["config_num_sm"], 148)
        self.assertEqual(different_layout_results["config_num_sm"], 148)
        self.assertEqual(smaller_sm_results["config_num_sm"], 132)
        first_metadata = first_results["input_tensor_metadata"]
        self.assertNotEqual(
            different_k_results["input_tensor_metadata"], first_metadata
        )
        self.assertNotEqual(
            different_n_results["input_tensor_metadata"], first_metadata
        )
        self.assertNotEqual(
            different_layout_results["input_tensor_metadata"], first_metadata
        )
        self.assertEqual(
            smaller_sm_results["input_tensor_metadata"],
            different_k_results["input_tensor_metadata"],
        )
        self.assertIsNone(first.config_spec.matmul_facts[0].static_k)
        self.assertIsNone(first.config_spec.compiler_default_config)
        self.assertEqual(
            {
                cast("list[int]", config.config["block_sizes"])[2]
                for config in first.config_spec.compiler_seed_configs
                if config.config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
                == TCGEN05_GROUPED_MODE_WORKLIST_NM
            },
            {64, 128},
        )

    @onlyBackends(["cute"])
    def test_grouped_worklist_smem_facts_register_with_heuristics_disabled(
        self,
    ) -> None:
        from pretuned_kernels.grouped_gemm_deepgemm.grouped_gemm_deepgemm import (
            _GROUPED_GEMM_DEEPGEMM_BODY,
        )

        source_m_tile = 224
        k = 128

        def make_args(groups: int, *, n: int = 256) -> tuple[object, ...]:
            return (
                torch.empty(
                    [groups * source_m_tile, k],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    [groups, n, k],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.empty([groups, 4], device=DEVICE, dtype=torch.int32),
                None,
                source_m_tile,
            )

        groups = 6
        empty_args = make_args(groups, n=0)
        args = make_args(groups)
        different_groups = 7
        different_args = make_args(different_groups)
        kernel = helion.kernel(
            _GROUPED_GEMM_DEEPGEMM_BODY,
            backend="cute",
            static_shapes=False,
            disable_autotuner_heuristics=True,
        )
        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
        ):
            empty_bound = kernel.bind(empty_args)
            bound = kernel.bind(args)
            rebound = kernel.bind(different_args)

        self.assertIsNone(
            empty_bound.config_spec._cute_tcgen05_config.grouped_worklist_smem_facts
        )
        self.assertIsNot(bound, empty_bound)
        self.assertEqual(bound.config_spec.compiler_seed_configs, [])
        self.assertEqual(bound.config_spec.autotuner_heuristics, [])
        self.assertEqual(
            compiler_seed_configs(bound.env, bound.host_function.device_ir),
            [],
        )
        self.assertEqual(
            bound.config_spec._cute_tcgen05_config.grouped_worklist_smem_facts,
            (groups, False),
        )
        self.assertIsNot(rebound, bound)
        self.assertEqual(
            rebound.config_spec._cute_tcgen05_config.grouped_worklist_smem_facts,
            (different_groups, False),
        )
        self.assertEqual(
            [
                extractor.fact
                for extractor in empty_bound._compiler_seed_specialization_extractors
            ],
            ["input_tensor_metadata"],
        )
        bound_output = cast("torch.Tensor", args[0])
        bound._run = lambda *_args: bound_output
        self.assertIs(empty_bound(*args), bound_output)
        rebound_output = cast("torch.Tensor", different_args[0])
        rebound._run = lambda *_args: rebound_output
        self.assertIs(bound(*different_args), rebound_output)

    @onlyBackends(["cute"])
    def test_grouped_worklist_fact_registration_ignores_plain_matmul(self) -> None:
        @helion.kernel(
            backend="cute",
            static_shapes=False,
            disable_autotuner_heuristics=True,
        )
        def plain_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(
                        acc,
                        x[tile_m, tile_k],
                        y[tile_k, tile_n],
                    )
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        def make_args(size: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.empty(
                    [size, 128],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    [128, size],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
            )

        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
        ):
            first = plain_matmul.bind(make_args(128))
            rebound = plain_matmul.bind(make_args(256))

        self.assertIs(rebound, first)
        self.assertEqual(len(plain_matmul._bound_kernels), 1)
        self.assertEqual(first._compiler_seed_specialization_extractors, ())
        self.assertIsNone(
            first.config_spec._cute_tcgen05_config.grouped_worklist_smem_facts
        )

    @onlyBackends(["cute"])
    def test_grouped_worklist_bound_metadata_proof_and_effective_default(self) -> None:
        from helion._compiler.cute.cute_mma import tcgen05_grouped_worklist_seed_facts

        @helion.kernel(backend="cute", static_shapes=True)
        def renamed_packing_kernel(
            a_packed: torch.Tensor,
            b_grouped: torch.Tensor,
            worklist: torch.Tensor,
            packing_rows: hl.constexpr,
        ) -> torch.Tensor:
            packing_rows = hl.register_heuristic_metadata(
                "tcgen05_grouped_worklist_source_m_tile",
                packing_rows,
            )
            m_total, k = a_packed.shape
            _groups, n, k2 = b_grouped.shape
            assert k == k2
            assert packing_rows in (32, 224, 256)
            block_m = hl.register_block_size(256)
            block_n = hl.register_block_size(128)
            block_k = hl.register_block_size(64, 128)
            out = torch.empty(
                [m_total, n],
                dtype=a_packed.dtype,
                device=a_packed.device,
            )
            for work_tile, tile_m, tile_n in hl.tile(
                [worklist.size(0), 256, n],
                block_size=[1, block_m, block_n],
            ):
                work_id = work_tile.begin
                group_id = worklist[work_id, 0]
                start = worklist[work_id, 1]
                valid_m = worklist[work_id, 2]
                store_m = worklist[work_id, 3]
                local_m = tile_m.index
                row = start + local_m
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k, block_size=block_k):
                    a_block = hl.load(
                        a_packed,
                        [row, tile_k],
                        extra_mask=(local_m < valid_m)[:, None],  # pyrefly: ignore[bad-index]
                    )
                    acc = torch.addmm(
                        acc,
                        a_block,
                        b_grouped[group_id, tile_n, tile_k].T,
                    )
                hl.store(
                    out,
                    [row, tile_n],
                    acc.to(out.dtype),
                    extra_mask=(local_m < store_m)[:, None],  # pyrefly: ignore[bad-index]
                )
            return out

        def make_args(
            source_m_tile: int,
            b_major: str,
            k: int = 128,
            packed_m: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
            groups, n = 6, 256
            a_packed = torch.empty(
                [
                    groups * source_m_tile if packed_m is None else packed_m,
                    k,
                ],
                device=DEVICE,
                dtype=torch.bfloat16,
            )
            if b_major == "k":
                b_grouped = torch.empty(
                    [groups, n, k],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                )
            else:
                b_grouped = torch.empty(
                    [groups, k, n],
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ).transpose(1, 2)
            worklist = torch.empty(
                [groups, 4],
                device=DEVICE,
                dtype=torch.int32,
            )
            return a_packed, b_grouped, worklist, source_m_tile

        def grouped_seeds(spec: ConfigSpec, source_m_tile: int) -> list[helion.Config]:
            return [
                config
                for config in spec.compiler_seed_configs
                if config.config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
                == TCGEN05_GROUPED_MODE_WORKLIST_NM
                and config.config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
                == source_m_tile
            ]

        def transferred_seed_pairs(spec: ConfigSpec) -> tuple[Any, list[Any]]:
            transfer_errors: list[str] = []
            generation = spec.create_config_generation()
            transferred = generation.seed_flat_config_pairs(transfer_errors.append)
            self.assertFalse(transfer_errors)
            for flat, normalized in transferred:
                self.assertEqual(generation.unflatten([*flat]), normalized)
            return generation, transferred

        def seed_pipeline(
            bound: Any,
            source_m_tile: int,
            *,
            clear_cluster_constraints: bool = False,
            expected_block_k: int | None = None,
            compile_default: bool = False,
        ) -> tuple[ConfigSpec, list[helion.Config], helion.Config, Any, list[Any]]:
            spec = bound.config_spec
            self.assertTrue(spec.cute_tcgen05_search_enabled)
            self.assertEqual(
                spec.compiler_heuristic_metadata[
                    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
                ],
                source_m_tile,
            )
            self.assertIn(
                CuteTcgen05GroupedWorklistHeuristic.name,
                spec.autotuner_heuristics,
            )
            seeds = grouped_seeds(spec, source_m_tile)
            self.assertTrue(seeds)
            if expected_block_k is not None:
                self.assertTrue(
                    all(
                        cast("list[int]", seed.config["block_sizes"])[2]
                        == expected_block_k
                        for seed in seeds
                    )
                )
            if clear_cluster_constraints:
                spec._cute_tcgen05_config.cluster_m2_search_constraints = None
            generation, transferred = transferred_seed_pairs(spec)
            self.assertGreaterEqual(len(transferred), len(seeds))
            promoted = spec.compiler_default_config
            self.assertIsNotNone(promoted)
            assert promoted is not None
            self.assertEqual(promoted, seeds[0])
            self.assertIn(
                generation.canonicalize_flat(generation.flatten(promoted)),
                transferred,
            )
            effective = spec.default_config()
            self.assertEqual(
                effective.config["block_sizes"], promoted.config["block_sizes"]
            )
            if compile_default:
                self.assertTrue(
                    callable(bound.compile_config(effective, allow_print=False))
                )
            return spec, seeds, promoted, effective, transferred

        bounds = []
        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cute_mma.tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch(
                "helion._hardware.get_hardware_info",
                return_value=BLACKWELL_HARDWARE,
            ),
        ):
            for source_m_tile, b_major in (
                (32, "k"),
                (32, "n"),
                (224, "k"),
                (224, "n"),
                (256, "k"),
                (256, "n"),
            ):
                with self.subTest(source_m_tile=source_m_tile, b_major=b_major):
                    bound = renamed_packing_kernel.bind(
                        make_args(source_m_tile, b_major)
                    )
                    bounds.append(bound)
                    expect_cluster2 = source_m_tile in (224, 256)
                    spec, _seeds, promoted, effective, transferred = seed_pipeline(
                        bound,
                        source_m_tile,
                        clear_cluster_constraints=expect_cluster2,
                        compile_default=True,
                    )
                    for _flat, normalized in transferred:
                        if expect_cluster2:
                            self.assertEqual(
                                normalized.config["tcgen05_cluster_m"],
                                2,
                            )
                    self.assertIsNot(effective, promoted)
                    self.assertGreater(len(effective.config), len(promoted.config))
                    self.assertEqual(
                        effective.config[
                            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
                        ],
                        source_m_tile,
                    )
                    self.assertEqual(
                        effective.config["block_sizes"],
                        promoted.config["block_sizes"],
                    )
                    self.assertEqual(
                        cast("list[int]", effective.config["block_sizes"])[:2],
                        [256, 128],
                    )
                    if expect_cluster2:
                        self.assertIsNone(
                            spec._cute_tcgen05_config.cluster_m2_search_constraints
                        )
                        self.assertEqual(promoted.config["tcgen05_cluster_m"], 2)
                        self.assertEqual(effective.config["tcgen05_cluster_m"], 2)

            from pretuned_kernels.grouped_gemm_deepgemm.grouped_gemm_deepgemm import (
                grouped_gemm_deepgemm,
            )

            renamed_args = make_args(224, "n")
            original_bound = grouped_gemm_deepgemm.bind(
                (*renamed_args[:3], None, renamed_args[3])
            )
            self.assertIsNotNone(original_bound.config_spec.compiler_default_config)
            self.assertEqual(
                grouped_seeds(original_bound.config_spec, 224),
                grouped_seeds(bounds[3].config_spec, 224),
            )
            b200_256_bound = bounds[-1]
            self.assertIsNotNone(b200_256_bound.config_spec.compiler_default_config)

            for source_m_tile in (32, 256):
                with self.subTest(source_m_tile=source_m_tile, k=192):
                    bound = renamed_packing_kernel.bind(
                        make_args(source_m_tile, "n", k=192)
                    )
                    seed_pipeline(
                        bound,
                        source_m_tile,
                        expected_block_k=64,
                        compile_default=True,
                    )

            for source_m_tile in (224, 256):
                with self.subTest(
                    source_m_tile=source_m_tile,
                    packed_m="non_aligned",
                ):
                    exact_packed_m = 6 * source_m_tile + 1
                    bound = renamed_packing_kernel.bind(
                        make_args(
                            source_m_tile,
                            "n",
                            packed_m=exact_packed_m,
                        )
                    )
                    spec, _seeds, _promoted, _effective, _transferred = seed_pipeline(
                        bound,
                        source_m_tile,
                    )
                    host_function = bound.host_function
                    self.assertIsNotNone(host_function)
                    assert host_function is not None
                    seed_facts = tcgen05_grouped_worklist_seed_facts(
                        bound.env,
                        host_function.device_ir,
                        spec.matmul_facts[0],
                        source_m_tile=source_m_tile,
                    )
                    self.assertIsNotNone(seed_facts)
                    assert seed_facts is not None
                    self.assertEqual(seed_facts.packed_m_hint, exact_packed_m)

        source32_host = bounds[0].host_function
        self.assertIsNotNone(source32_host)
        assert source32_host is not None
        source32_facts = tcgen05_grouped_worklist_seed_facts(
            bounds[0].env,
            source32_host.device_ir,
            bounds[0].config_spec.matmul_facts[0],
            source_m_tile=32,
        )
        self.assertIsNotNone(source32_facts)
        assert source32_facts is not None
        self.assertEqual(source32_facts.packed_m_hint, 6 * 32)
        with (
            bounds[1].env,
            self.assertRaisesRegex(
                AssertionError,
                "must use the provided compile environment",
            ),
        ):
            tcgen05_grouped_worklist_seed_facts(
                bounds[0].env,
                source32_host.device_ir,
                bounds[0].config_spec.matmul_facts[0],
                source_m_tile=32,
            )
        with (
            patch_cute_mma_support(),
            patch(
                "helion._compiler.autotuner_heuristics.cute."
                "tcgen05_runtime_n_ptx_compatible",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cutedsl_compat.check_cute_backend_requirements"
            ),
            patch("helion._hardware.get_hardware_info", return_value=GB300_HARDWARE),
        ):
            gb300_bound = renamed_packing_kernel.bind(make_args(256, "n"))
        self.assertIsNot(gb300_bound, b200_256_bound)
        self.assertTrue(grouped_seeds(gb300_bound.config_spec, 256))
        self.assertIsNone(gb300_bound.config_spec.compiler_default_config)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_pointwise_seed_promotes_only_on_target_arch(self) -> None:
        # The pointwise seed fires arch-agnostically but its PROMOTION to the
        # autotune-off default is gated to PROMOTE_TARGETS (sm90/sm100). On a
        # target arch the promoted seed replaces the base default; off-target it
        # is still offered as a search candidate but the base default is kept.
        def make_add() -> object:
            @helion.kernel(backend="triton")
            def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(x.shape):
                    out[tile] = x[tile] + y[tile]
                return out

            return triton_add

        x = torch.empty([1024, 1024], device=DEVICE, dtype=torch.float32)
        y = torch.empty([1024, 1024], device=DEVICE, dtype=torch.float32)

        self.assertIn(("cuda", "sm90"), TritonPointwiseSeedHeuristic.PROMOTE_TARGETS)
        self.assertIn(("cuda", "sm100"), TritonPointwiseSeedHeuristic.PROMOTE_TARGETS)

        # A fresh kernel object per arch: bind() caches by args, so reusing one
        # kernel would return the first arch's config on the second bind.
        for name, hardware, promotes in (
            ("sm90", HOPPER_HARDWARE, True),
            ("sm100", BLACKWELL_HARDWARE, True),
            ("sm80", A100_HARDWARE, False),
        ):
            with (
                self.subTest(arch=name),
                patch("helion._hardware.get_hardware_info", return_value=hardware),
            ):
                bound = make_add().bind((x, y))
                spec = bound.config_spec
                # The seed fires and is offered as a search candidate on every arch.
                self.assertEqual(spec.autotuner_heuristics, ["triton_pointwise"])
                self.assertEqual(len(spec.compiler_seed_configs), 1)
                seed = spec.compiler_seed_configs[0]
                if promotes:
                    self.assertEqual(spec.compiler_default_config, seed)
                else:
                    self.assertIsNone(spec.compiler_default_config)


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
                # No matmul fact: a pure-pointwise kernel (triton_add) is instead seeded by
                # TritonPointwiseSeedHeuristic. Assert it routes there (one seed config), not
                # the pre-pointwise-heuristic expectation of no seed at all.
                self.assertEqual(
                    bound.config_spec.autotuner_heuristics, ["triton_pointwise"]
                )
                self.assertEqual(len(bound.config_spec.compiler_seed_configs), 1)
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
        dtype: torch.dtype = HALF_DTYPE,
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
            lhs_dtype=dtype,
            rhs_dtype=dtype,
        )

    def test_triton_skinny_gemm_seed_eligibility_and_config(
        self,
    ) -> None:
        # The dense TritonH100MatmulHeuristic ALSO fires on every clean 2-D static matmul on
        # sm90 (by design — it is the H100 dense seed). So this test checks the SKINNY
        # heuristic's OWN contribution (its name + [64,64,256] config present-or-absent),
        # robust to the H100 seeds co-existing in the list. expected_skinny = the skinny config
        # when the skinny rule fires, else None.
        cases = (
            ("hopper", HOPPER_HARDWARE, [self._matmul_fact()], [64, 64, 256]),
            ("mi350", MI350_HARDWARE, [self._matmul_fact()], [64, 64, 256]),
            ("blackwell", BLACKWELL_HARDWARE, [self._matmul_fact()], None),
            (
                "balanced_shape",
                HOPPER_HARDWARE,
                [self._matmul_fact(static_m=4096, static_n=4096)],
                None,
            ),
            (
                "multiple_matmuls",
                HOPPER_HARDWARE,
                [self._matmul_fact(), self._matmul_fact()],
                None,
            ),
        )
        for name, hardware, facts, expected_skinny in cases:
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

            block_size_lists = [config.config["block_sizes"] for config in configs]
            if expected_skinny is not None:
                self.assertIn(
                    TritonSkinnyGemmHeuristic.name,
                    env.config_spec.autotuner_heuristics,
                )
                self.assertIn(expected_skinny, block_size_lists)
            else:
                self.assertNotIn(
                    TritonSkinnyGemmHeuristic.name,
                    env.config_spec.autotuner_heuristics,
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
                    # The initial population includes every compiler seed; the dense
                    # H100 heuristic now plants its own seeds alongside skinny's, so ask
                    # for a population large enough to contain all seeds (not just 2).
                    assert_skinny_gemm_seeded(
                        config_gen.random_population(len(compiler_seed_block_sizes) + 2)
                    )
                else:
                    self.assertFalse(
                        heuristic.is_eligible(bound.env, bound.host_function.device_ir)
                    )
                    self.assertNotIn(
                        TritonSkinnyGemmHeuristic.name,
                        bound.config_spec.autotuner_heuristics,
                    )


class TestTritonH100MatmulHeuristic(TestCase):
    """The H100 (sm90) dense-matmul budget-formula seed heuristic."""

    def _matmul_fact(
        self,
        static_m: int = 4096,
        static_n: int = 4096,
        static_k: int = 4096,
        *,
        lhs_ndim: int = 2,
        rhs_ndim: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ) -> MatmulFact:
        return MatmulFact(
            lhs_ndim=lhs_ndim,
            rhs_ndim=rhs_ndim,
            m_block_id=0,
            n_block_id=1,
            k_block_id=2,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            lhs_dtype=dtype,
            rhs_dtype=dtype,
        )

    def _make_env(self) -> MagicMock:
        spec = ConfigSpec(backend=TritonBackend())
        for bid in range(3):
            spec.block_sizes.append(BlockSizeSpec(block_id=bid, size_hint=8192))
        env = MagicMock()
        env.backend_name = "triton"
        env.config_spec = spec
        env.device = DEVICE
        env.settings = Settings()
        return env

    def test_budget_formula_is_deterministic_per_regime(self) -> None:
        # Pure formula (fixed num_sm=132, H100), exercising every lever. Returns the tile
        # tuple (bm, bn, bk, num_warps, num_stages, l2_grouping).
        sm = 132
        # big compute-bound cube: wide-N [128,256], num_warps=8, num_stages=4; block_k scales
        # with operand WIDTH via SMEM (fp8 1B -> 128, bf16 2B -> 64, fp32 4B -> 32).
        self.assertEqual(
            _h100_matmul_tile(4096, 4096, 4096, 2, sm, 1), (128, 256, 64, 8, 4, 1)
        )
        self.assertEqual(
            _h100_matmul_tile(4096, 4096, 4096, 1, sm, 1), (128, 256, 128, 8, 4, 1)
        )
        self.assertEqual(
            _h100_matmul_tile(4096, 4096, 4096, 4, sm, 1), (128, 256, 32, 8, 4, 1)
        )
        # tall tile-grid (grid_m >> grid_n) -> l2_grouping=2.
        self.assertEqual(_h100_matmul_tile(16384, 512, 8192, 2, sm, 1)[5], 2)
        # deep-K small-MN -> deepen the pipeline (num_stages 6, SMEM-permitting).
        self.assertEqual(_h100_matmul_tile(256, 256, 12288, 2, sm, 1)[4], 6)
        # saturated fused batched dot (huge pinned grid): tile capped to <=[64,128] for
        # occupancy AND num_stages capped to 2 (deep pipeline is redundant when occupancy
        # already hides latency).
        bm, bn, _bk, _w, ns, _l2 = _h100_matmul_tile(64, 128, 256, 2, sm, 10240)
        self.assertLessEqual(bm, 64)
        self.assertLessEqual(bn, 128)
        self.assertEqual(ns, 2)
        # a bare GEMM (pinned_grid==1) at the SAME tile keeps the deep pipeline.
        self.assertGreater(_h100_matmul_tile(64, 128, 256, 2, sm, 1)[4], 2)

    def test_eligibility(self) -> None:
        cases = (
            ("hopper_single_static", HOPPER_HARDWARE, [self._matmul_fact()], True),
            ("blackwell_off", BLACKWELL_HARDWARE, [self._matmul_fact()], False),
            ("mi350_off", MI350_HARDWARE, [self._matmul_fact()], False),
            (
                "multiple_matmuls",
                HOPPER_HARDWARE,
                [self._matmul_fact(), self._matmul_fact()],
                False,
            ),
            (
                "non_static",
                HOPPER_HARDWARE,
                [self._matmul_fact(static_m=None)],
                False,
            ),
            # fp8 (both operands 1-byte float) is declined: the budget tile would trigger the
            # Triton fp8-accumulator bug (block_m>=64 -> never-promoted QGMMA accumulator). See
            # TritonH100MatmulHeuristic.is_eligible. bf16/fp16/fp32 stay eligible above.
            (
                "fp8_declined",
                HOPPER_HARDWARE,
                [self._matmul_fact(dtype=torch.float8_e4m3fn)],
                False,
            ),
        )
        for name, hardware, facts, eligible in cases:
            env = self._make_env()
            env.config_spec.matmul_facts.extend(facts)
            with (
                self.subTest(name=name),
                patch("helion._hardware.get_hardware_info", return_value=hardware),
            ):
                self.assertEqual(
                    TritonH100MatmulHeuristic.is_eligible(env, MagicMock()), eligible
                )

    def test_ranked_multi_seed_and_width_merge(self) -> None:
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            # rank-0 (Product A) == get_seed_config; the ranked list has diverse alternates.
            env = self._make_env()
            env.config_spec.matmul_facts.append(self._matmul_fact())
            ranked = TritonH100MatmulHeuristic.get_seed_configs(env, MagicMock())
            primary = TritonH100MatmulHeuristic.get_seed_config(env, MagicMock())
            assert ranked is not None and primary is not None
            self.assertGreaterEqual(len(ranked), 1)
            self.assertEqual(
                ranked[0].config["block_sizes"], primary.config["block_sizes"]
            )
            self.assertEqual(len(ranked), len({repr(dict(c)) for c in ranked}))

            # 16-bit merge: bf16 and fp16 produce the IDENTICAL seed (width key, not dtype kind).
            env_bf16 = self._make_env()
            env_bf16.config_spec.matmul_facts.append(
                self._matmul_fact(dtype=torch.bfloat16)
            )
            env_fp16 = self._make_env()
            env_fp16.config_spec.matmul_facts.append(
                self._matmul_fact(dtype=torch.float16)
            )
            bf16 = TritonH100MatmulHeuristic.get_seed_config(env_bf16, MagicMock())
            fp16 = TritonH100MatmulHeuristic.get_seed_config(env_fp16, MagicMock())
            assert bf16 is not None and fp16 is not None
            self.assertEqual(dict(bf16), dict(fp16))

    @onlyBackends(["triton"])
    @skipIfTileIR(
        "seed heuristics dispatch on backend_name, which is 'tileir' not 'triton'"
    )
    @skipIfRefEager("Compiler heuristics are not collected in ref eager mode")
    def test_fires_on_batched_dot_and_pins_batch_to_one(self) -> None:
        # A genuine BATCHED dot (bmm's baddbmm: 3-D operands + a tunable batch axis) fires, and
        # the seed pins every batch/outer axis to 1 (one CTA per batch — a no-reuse parallel axis)
        # while sizing only the dot's M/N/K by the budget. fp32 inputs keep this torch-version
        # independent (16-bit baddbmm needs torch>=2.8); the levers tested here are dtype-agnostic.
        from examples.bmm import bmm

        a = torch.randn(16, 512, 512, device=DEVICE, dtype=torch.float32)
        b = torch.randn(16, 512, 512, device=DEVICE, dtype=torch.float32)
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            k = helion.kernel(bmm.fn, static_shapes=True)
            bound = k.bind(k.normalize_args(a, b))
            spec = bound.env.config_spec
            fact = spec.matmul_facts[0]
            self.assertGreaterEqual(fact.lhs_ndim, 3)  # a 3-D (batched) dot
            self.assertGreater(len(spec.block_sizes), 3)  # batch axis is tunable
            self.assertIn(TritonH100MatmulHeuristic.name, spec.autotuner_heuristics)
            seed = TritonH100MatmulHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
            assert seed is not None
            block_sizes = seed.config["block_sizes"]
            mnk = {fact.m_block_id, fact.n_block_id, fact.k_block_id}
            batch_axes = [
                i
                for i in range(len(spec.block_sizes))
                if spec.block_sizes[i].block_id not in mnk
            ]
            self.assertTrue(batch_axes)  # there is a batch/outer axis
            for i in batch_axes:
                self.assertEqual(block_sizes[i], 1)  # pinned to 1


class TestRopePointwiseSeed(TestCase):
    """Rope (split/join rotate) is handled by the pointwise seed: its heavy untiled
    [heads, head_dim] slab collapses the tunable tile, while a plain elementwise kernel
    of the same dtype gets a bandwidth-sized tile.
    """

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler heuristics are not collected in ref eager mode")
    def test_rope_collapses_tile_elementwise_does_not(self) -> None:
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
            TritonPointwiseSeedHeuristic.is_eligible(
                rope.env, rope.host_function.device_ir
            )
        )
        with rope.env:
            rope_seed = TritonPointwiseSeedHeuristic.get_seed_config(
                rope.env, rope.host_function.device_ir
            )
        # Heavy untiled [heads, head_dim] slab → the tunable tile collapses (not tiled past ~1).
        self.assertLessEqual(math.prod(rope_seed.config["block_sizes"]), 8)

        xy = (
            torch.randn(512, 512, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(512, 512, device=DEVICE, dtype=HALF_DTYPE),
        )
        ew = elementwise.bind(xy)
        self.assertTrue(
            TritonPointwiseSeedHeuristic.is_eligible(ew.env, ew.host_function.device_ir)
        )
        with ew.env:
            ew_seed = TritonPointwiseSeedHeuristic.get_seed_config(
                ew.env, ew.host_function.device_ir
            )
        # Plain elementwise (no slab) → a bandwidth-sized tile, not collapsed.
        self.assertGreater(math.prod(ew_seed.config["block_sizes"]), 8)


class TestPointwiseComputeItemsize(TestCase):
    """``compute_itemsize`` measures DATA width, not index width.

    The fact walks only what reaches a load/store's data path, so int64 address
    arithmetic does not inflate it. The split is data-vs-address, not
    float-vs-integer -- both directions are asserted here.
    """

    @staticmethod
    def _fact(kernel: object, args: tuple[object, ...]) -> object:
        bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
        facts = bound.config_spec.pointwise_facts
        assert len(facts) == 1, facts
        return facts[0]

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_int64_indexing_does_not_inflate_compute_itemsize(self) -> None:
        # int64-INDEXED but half-precision DATA (promoting to fp32 -> 4).
        # The SCALED subscript is load-bearing: a plain ``x[tile]`` lowers to a
        # ``_get_symnode`` with no tensor val, so no int64 node exists and the test
        # would pass even with the fix reverted.
        @helion.kernel(backend="triton", index_dtype=torch.int64, static_shapes=False)
        def add_int64_index(x: torch.Tensor) -> torch.Tensor:
            m, n2 = x.shape
            n2 = hl.specialize(n2)
            n = n2 // 2
            out = x.new_empty(m, n)
            for tile_m, tile_n in hl.tile([m, n]):
                a = x[tile_m, 2 * tile_n.index].to(torch.float32)
                b = x[tile_m, 2 * tile_n.index + 1].to(torch.float32)
                out[tile_m, tile_n] = (a * b).to(out.dtype)
            return out

        x = torch.randn(256, 512, device=DEVICE, dtype=HALF_DTYPE)
        bound = add_int64_index.bind((x,))
        self.assertEqual(bound.env.index_dtype, torch.int64)
        # Guard the guard: an int64 tensor is actually present to be picked up.
        self.assertTrue(
            any(
                isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].dtype == torch.int64
                for graph_info in bound.host_function.device_ir.graphs
                for node in graph_info.graph.nodes
            )
        )
        facts = bound.config_spec.pointwise_facts
        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertEqual(fact.storage_itemsize, 2)
        # 4 = the fp32 compute promotion, not 8 = the int64 index arithmetic.
        self.assertEqual(fact.compute_itemsize, 4)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_integer_compute_kernel_reports_its_data_width(self) -> None:
        # Guards against filtering to float dtypes: this DATA is genuinely int64, so
        # 8 is correct and a float-only walk would report 1 and emit a spilling tile.
        @helion.kernel(backend="triton")
        def add_int64_data(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.ones(256, 256, device=DEVICE, dtype=torch.int64)
        y = torch.ones(256, 256, device=DEVICE, dtype=torch.int64)
        fact = self._fact(add_int64_data, (x, y))
        self.assertEqual(fact.storage_itemsize, 8)
        self.assertEqual(fact.compute_itemsize, 8)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_gather_index_tensor_does_not_inflate_compute_itemsize(self) -> None:
        # A gather whose index was itself loaded as int64: the cut at loads means the
        # walk never reaches it, so the data width stays half-precision.
        @helion.kernel(backend="triton")
        def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[idx[tile_m], tile_n]
            return out

        x = torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE)
        idx = torch.zeros(256, device=DEVICE, dtype=torch.int64)
        fact = self._fact(gather_rows, (x, idx))
        self.assertEqual(fact.compute_itemsize, 2)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_intermediate_wider_than_every_buffer_is_counted(self) -> None:
        # Why the walk cannot just read buffer dtypes: every buffer is half-precision
        # but an fp64 intermediate is register-resident, and under-reporting it
        # over-estimates ``reg_cap``. ``storage_itemsize`` already covers buffers.
        @helion.kernel(backend="triton")
        def fp64_intermediate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                a = x[tile].to(torch.float64)
                b = y[tile].to(torch.float64)
                out[tile] = (a * b + a).to(out.dtype)
            return out

        x = torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE)
        fact = self._fact(fp64_intermediate, (x, y))
        self.assertEqual(fact.storage_itemsize, 2)
        self.assertEqual(fact.compute_itemsize, 8)


class TestPointwiseGatherStride(TestCase):
    """``gather_stride`` is the address-expression scale, not the layout stride.

    Two kernels can index one row-major tensor -- identical layout strides -- while
    reading it at different gather strides, and their optimal tiles then differ.
    """

    @staticmethod
    def _fact(kernel: object, args: tuple[object, ...]) -> object:
        bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
        facts = bound.config_spec.pointwise_facts
        assert len(facts) == 1, facts
        return facts[0]

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_interleaved_and_contiguous_halves_differ(self) -> None:
        @helion.kernel(backend="triton", static_shapes=False)
        def interleaved(x: torch.Tensor) -> torch.Tensor:
            m, n2 = x.shape
            n2 = hl.specialize(n2)
            n = n2 // 2
            out = x.new_empty(m, n)
            for tile_m, tile_n in hl.tile([m, n]):
                a = x[tile_m, 2 * tile_n.index]
                b = x[tile_m, 2 * tile_n.index + 1]
                out[tile_m, tile_n] = a * b
            return out

        @helion.kernel(backend="triton", static_shapes=False)
        def halves(x: torch.Tensor) -> torch.Tensor:
            m, n2 = x.shape
            n2 = hl.specialize(n2)
            n = n2 // 2
            out = x.new_empty(m, n)
            for tile_m, tile_n in hl.tile([m, n]):
                a = x[tile_m, tile_n.index]
                b = x[tile_m, tile_n.index + n]
                out[tile_m, tile_n] = a * b
            return out

        x = torch.randn(256, 512, device=DEVICE, dtype=HALF_DTYPE)
        inter = self._fact(interleaved, (x,))
        contig = self._fact(halves, (x,))
        # The stride-2 gather is recognized...
        self.assertEqual(inter.gather_stride, 2)
        # ...and the stride-1 control is not penalized.
        self.assertEqual(contig.gather_stride, 1)
        # Layout stride is 1 for both, so ``subscript_strides`` cannot tell them apart.
        for fact_owner in (interleaved, halves):
            bound = fact_owner.bind((x,))
            inner = [
                m.subscript_strides[-1]
                for m in bound.config_spec.memory_op_facts
                if m.subscript_strides
            ]
            self.assertTrue(all(s == 1 for s in inner), inner)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler pointwise facts are not collected in ref eager mode")
    def test_plain_elementwise_is_stride_one(self) -> None:
        @helion.kernel(backend="triton")
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE)
        self.assertEqual(self._fact(add, (x, x)).gather_stride, 1)

    @onlyBackends(["triton"])
    def test_elems_per_thread_is_a_three_band_ladder(self) -> None:
        # Bands are 16 coalesced / 4 at strides 2-4 / 2 beyond. Guards two wrong
        # forms: saturate-at-2 (returns 4 past stride 4) and a continuous
        # ``16 // stride`` (returns 8 at stride 2).
        eps = TritonPointwiseSeedHeuristic._elems_per_thread
        self.assertEqual(eps(1), 16)
        for stride in (2, 3, 4):
            self.assertEqual(eps(stride), 4, f"stride {stride}")
        for stride in (8, 16, 32):
            self.assertEqual(eps(stride), 2, f"stride {stride}")
        # Monotonically non-increasing, and never below the wide-gather floor.
        values = [eps(s) for s in (1, 2, 4, 8, 16, 64, 1024)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreaterEqual(
            min(values), TritonPointwiseSeedHeuristic.ELEMS_PER_THREAD_WIDE_GATHER
        )
        # A degenerate/absent stride reads as coalesced rather than penalized.
        self.assertEqual(eps(0), 16)


class TestPointwiseArchConstants(TestCase):
    """The seed's byte constants are arch-keyed; unmeasured arches keep the sm90 path."""

    @onlyBackends(["triton"])
    def test_tile_bytes_and_waves_are_arch_keyed(self) -> None:
        from helion._compiler.compile_environment import CompileEnvironment

        env = MagicMock(spec=CompileEnvironment)
        env.device = torch.device(DEVICE)
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            self.assertEqual(
                TritonPointwiseSeedHeuristic.tile_bytes_for(env),
                TritonPointwiseSeedHeuristic.TILE_BYTES,
            )
            self.assertEqual(
                TritonPointwiseSeedHeuristic.min_waves_for(env),
                TritonPointwiseSeedHeuristic.MIN_WAVES,
            )
        with patch(
            "helion._hardware.get_hardware_info", return_value=BLACKWELL_HARDWARE
        ):
            self.assertEqual(
                TritonPointwiseSeedHeuristic.tile_bytes_for(env),
                TritonPointwiseSeedHeuristic.TILE_BYTES_SM100,
            )
            self.assertEqual(
                TritonPointwiseSeedHeuristic.min_waves_for(env),
                TritonPointwiseSeedHeuristic.MIN_WAVES_SM100,
            )
        # An arch the seed FIRES on but was never tuned for keeps the conservative
        # sm90-calibrated values, not the B200 ones. Covers both directions of the
        # allow-list: an OLDER arch (sm80), and -- the case a ``>= sm100`` version
        # compare would get wrong -- a NEWER/adjacent one. sm120 (consumer Blackwell,
        # GB202) and a hypothetical future sm110 both sort above sm100 numerically
        # while having entirely different SM counts and HBM bandwidth, so neither may
        # inherit B200's fitted constants.
        for name, cc in (("sm80", "sm80"), ("sm120", "sm120"), ("sm110", "sm110")):
            hardware = HardwareInfo(
                device_kind="cuda",
                hardware_name=f"untuned-{name}",
                runtime_version="12.8",
                compute_capability=cc,
            )
            with (
                self.subTest(arch=name),
                patch("helion._hardware.get_hardware_info", return_value=hardware),
            ):
                self.assertEqual(
                    TritonPointwiseSeedHeuristic.tile_bytes_for(env),
                    TritonPointwiseSeedHeuristic.TILE_BYTES,
                )
                self.assertEqual(
                    TritonPointwiseSeedHeuristic.min_waves_for(env),
                    TritonPointwiseSeedHeuristic.MIN_WAVES,
                )


class TestTritonStandardReductionHeuristic(TestCase):
    """Triton standard row-reduction heuristic: seeds the "one row per program"
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
        row_reread: bool = False,
    ) -> ConfigSpec:
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.reduction_loops.append(
            ReductionLoopSpec(block_id=1, size_hint=reduction_size_hint)
        )
        # The deepened heuristic reads the primary ReductionDescriptor (the workload facts it
        # keys the warp ramp / eviction / persist decision on) off the ReductionKernelFact; the
        # reduction axis is block_id=1 (the rolled reduction loop above, so a FULL_SLICE), the
        # row/grid axis is block_id=0.
        desc = ReductionDescriptor(
            category=ReductionCategory.FULL_SLICE,
            block_id=1,
            graph_id=0,
            size_hint=reduction_size_hint,
            itemsize=itemsize,
            input_load_itemsize=itemsize,
            row_reread=row_reread,
            num_load=num_load,
        )
        spec.reduction_kernel_fact = ReductionKernelFact(
            reductions=(desc,),
            coresidency_groups=(
                CoResidencyGroup(
                    graph_id=0,
                    descriptor_indices=(0,),
                    # The resident live tiles of a one-row reduction (softmax/rms_norm-like): the
                    # ``[grid_M, rdim]`` read/compute tile + a ``[grid_M]`` scalar carry. The grid
                    # axis (block_id 0) APPEARS in a live tile -> it is register-RESIDENT, so
                    # ``_has_reduced_away_grid`` is False (it is NOT a grad-parameter ``.sum(0)``
                    # collapse). Without this the grid axis is in no tile and the residency test
                    # wrongly flags a collapse, tripping the num_warps>=8 grad-param floor.
                    live_tiles=((0, 1), (0,)),
                ),
            ),
            grid_axis_block_ids=(0,),
        )
        return spec

    def _reduction_env(self, spec: ConfigSpec) -> MagicMock:
        # The deepened heuristic reads env.backend.max_tensor_numel (the structural
        # persistent cap) — provide the real Triton cap so a sub-cap rnumel stays
        # persistent.
        from types import SimpleNamespace

        from helion.autotuner.config_generation import TRITON_MAX_TENSOR_NUMEL

        env = MagicMock()
        env.backend_name = "triton"
        env.backend.max_tensor_numel = TRITON_MAX_TENSOR_NUMEL
        env.config_spec = spec
        env.device = DEVICE
        # ``_primary_descriptor_selected`` filters the sized descriptors to the BACKED axes
        # (``free_unbacked_symbols(env.block_sizes[bid].size)``), so the descriptor axes'
        # ``env.block_sizes[bid].size`` must be a real (backed) int — a bare MagicMock can't be
        # fed to sympy. Resolve each descriptor's block_id to its static ``size_hint``. For any
        # OTHER block_id (the grid/M axis) keep a non-int so ``_grid_rows`` returns 0 (no static
        # grid -> the occupancy-gated narrow-w1 warps lever stays disabled, as it was when the
        # mock had no configured sizes at all).
        sizes = {d.block_id: d.size_hint for d in spec.reduction_kernel_fact.reductions}
        env.block_sizes.__getitem__.side_effect = lambda bid: SimpleNamespace(
            size=sizes[bid] if bid in sizes else MagicMock()
        )
        return env

    def test_seed_is_persistent_one_row(self) -> None:
        # The structural seed: one row per program + persistent reduction. The
        # deepened heuristic ALSO seeds num_warps via the rnumel ramp (rnumel=1024
        # -> 4 warps) and num_stages=1, rather than leaving them to the autotuner.
        env = self._reduction_env(self._reduction_spec(reduction_size_hint=1024))
        # The mock env has no real GPU device, so patch hardware info / SM count (this
        # heuristic only fires on GPU in production and the SM count is irrelevant here).
        with (
            patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE),
            patch("helion.runtime.get_num_sm", return_value=132),
        ):
            seed = TritonStandardReductionHeuristicSM90.get_seed_config(
                env, MagicMock()
            )
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
        # The mock env has no real GPU device, so patch hardware info / SM count (this
        # heuristic only fires on GPU in production and the SM count is irrelevant here).
        with (
            patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE),
            patch("helion.runtime.get_num_sm", return_value=132),
        ):
            seed = TritonStandardReductionHeuristicSM90.get_seed_config(
                env, MagicMock()
            )
        self.assertEqual(
            seed.config["load_eviction_policies"],
            ["first", "first", "first", "first"],
        )

    def test_persistent_seed_round_trips_through_config_generation(self) -> None:
        # reduction_loops=[None] (persistent) MUST survive flatten/unflatten. For
        # a wide reduction (size_hint 32000) a sentinel < size_hint would decode
        # back to the SLOW looped family this heuristic exists to avoid; the
        # config_spec fix encodes None as the fragment's ``high`` (>= size_hint).
        # row_reread=True makes the wide reduction persist under the read-once persist
        # gate (read-once reductions deliberately loop), so there is a [None] to round-trip.
        from helion.autotuner.config_generation import ConfigGeneration

        spec = self._reduction_spec(reduction_size_hint=32000, row_reread=True)
        env = self._reduction_env(spec)
        # The mock env has no real GPU device, so patch hardware info / SM count (this
        # heuristic only fires on GPU in production and the SM count is irrelevant here).
        with (
            patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE),
            patch("helion.runtime.get_num_sm", return_value=132),
        ):
            seed = TritonStandardReductionHeuristicSM90.get_seed_config(
                env, MagicMock()
            )
        spec.compiler_seed_configs = [seed]
        pairs = ConfigGeneration(spec).seed_flat_config_pairs()
        self.assertEqual(len(pairs), 1)
        _flat, normalized = pairs[0]
        self.assertEqual(normalized.config["reduction_loops"], [None])

    def test_not_eligible_without_single_reduction_tile(self) -> None:
        env = MagicMock()
        # Pin the sm90 target so this exercises the STRUCTURAL gate, not the hardware gate
        # (is_eligible now checks hardware first).
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            # No reduction loop -> not a reduction.
            spec = ConfigSpec(backend=TritonBackend())
            spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
            env.config_spec = spec
            self.assertFalse(
                TritonStandardReductionHeuristicSM90.is_eligible(env, MagicMock())
            )
            # A matmul fact disqualifies even a 1-tile/1-reduction shape.
            spec_mm = self._reduction_spec(reduction_size_hint=1024)
            spec_mm.matmul_facts = [MagicMock()]
            env.config_spec = spec_mm
            self.assertFalse(
                TritonStandardReductionHeuristicSM90.is_eligible(env, MagicMock())
            )

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
        mm = matmul.bind(
            (
                torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE),
                torch.randn(256, 256, device=DEVICE, dtype=HALF_DTYPE),
            )
        )
        # Pin the sm90/H100 target so the sm90 heuristic is exercised regardless of the CI
        # runner's GPU: the hardware gate now lives in ``is_eligible`` (sm100/B200 routes to
        # ``TritonStandardReductionHeuristicSM100``, other GPUs to the narrow fallback), so on a
        # B200 runner the unpatched ``is_eligible`` would be False.
        with (
            patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE),
            patch("helion.runtime.get_num_sm", return_value=132),
        ):
            self.assertTrue(
                TritonStandardReductionHeuristicSM90.is_eligible(
                    red.env, red.host_function.device_ir
                )
            )
            seed = TritonStandardReductionHeuristicSM90.get_seed_config(
                red.env, red.host_function.device_ir
            )
            # A matmul is not a reduction, so the reduction seed declines even on its own target.
            self.assertFalse(
                TritonStandardReductionHeuristicSM90.is_eligible(
                    mm.env, mm.host_function.device_ir
                )
            )
        self.assertEqual(seed.config["block_sizes"], [1])
        self.assertEqual(seed.config["reduction_loops"], [None])

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler heuristics are not collected in ref eager mode")
    def test_exactly_one_reduction_track_eligible_per_hardware(self) -> None:
        # The hardware gate lives in ``is_eligible``: for a standard reduction, EXACTLY one of
        # the three standard-track classes fires per GPU — sm90 -> SM90, sm100 -> SM100, anything
        # else -> the narrow fallback — and none of them return None-for-deferral from
        # ``get_seed_config``. This is the invariant the class split exists to guarantee.
        @helion.kernel(backend="triton")
        def row_reduction(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :]
                shifted = row - torch.amax(row, dim=-1, keepdim=True)
                out[tile_m] = torch.log(torch.sum(torch.exp(shifted), dim=-1))
            return out

        red = row_reduction.bind(
            (torch.randn(1024, 1024, device=DEVICE, dtype=HALF_DTYPE),)
        )
        env, device_ir = red.env, red.host_function.device_ir
        # (hardware, the one class expected to fire).
        cases = [
            (HOPPER_HARDWARE, TritonStandardReductionHeuristicSM90),
            (BLACKWELL_HARDWARE, TritonStandardReductionHeuristicSM100),
            (
                HardwareInfo(
                    device_kind="cuda",
                    hardware_name="NVIDIA A10G",
                    runtime_version="12.8",
                    compute_capability="sm86",
                ),
                TritonNarrowReductionHeuristic,
            ),
        ]
        tracks = [
            TritonStandardReductionHeuristicSM90,
            TritonStandardReductionHeuristicSM100,
            TritonNarrowReductionHeuristic,
        ]
        for hardware, expected in cases:
            with (
                patch("helion._hardware.get_hardware_info", return_value=hardware),
                patch("helion.runtime.get_num_sm", return_value=132),
            ):
                eligible = [t for t in tracks if t.is_eligible(env, device_ir)]
                self.assertEqual(
                    eligible,
                    [expected],
                    f"{hardware.compute_capability}: expected only {expected.__name__}",
                )
                # The eligible class always yields a real Config (never None-for-deferral).
                seed = expected.get_seed_config(env, device_ir)
                self.assertIsNotNone(seed)
                self.assertEqual(seed.config["block_sizes"], [1])


_FP8_SKINNY_M_SEED_BLOCK_SIZES = [1, 256]
_FP8_SKINNY_M_SEED_NUM_THREADS = [0, 32]
_FP8_SKINNY_M_SEED_VECTOR_WIDTHS = [4, 8]


class TestCuteFp8GemmSkinnyMHeuristic(TestCase):
    """Skinny-M FP8 GEMM heuristic: fires only for a single FP8 matmul with
    static M <= 16 and seeds the [1, 256] / nt=[0, 32] / vec=[4, 8] config that
    the full autotune converges to for the decode / small-batch regime.
    """

    def _make_cute_env(self) -> MagicMock:
        spec = ConfigSpec(backend=CuteBackend())
        env = MagicMock()
        env.backend_name = "cute"
        env.config_spec = spec
        env.device = DEVICE
        env.settings = Settings()
        return env

    def _matmul_fact(
        self,
        *,
        static_m: int = 1,
        static_n: int = 4096,
        static_k: int = 4096,
        lhs_dtype: torch.dtype = torch.float8_e4m3fn,
        rhs_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> MatmulFact:
        return MatmulFact(
            lhs_ndim=2,
            rhs_ndim=2,
            m_block_id=0,
            n_block_id=1,
            k_block_id=2,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
        )

    def test_eligibility_cases(self) -> None:
        # (name, facts, expected_eligible)
        cases = (
            ("m1_fp8", [self._matmul_fact(static_m=1)], True),
            ("m16_fp8", [self._matmul_fact(static_m=16)], True),
            ("e5m2_fp8", [self._matmul_fact(lhs_dtype=torch.float8_e5m2)], True),
            ("m17_too_large", [self._matmul_fact(static_m=17)], False),
            ("m1024_gemm", [self._matmul_fact(static_m=1024)], False),
            (
                "bf16_not_fp8",
                [self._matmul_fact(lhs_dtype=torch.bfloat16, rhs_dtype=torch.bfloat16)],
                False,
            ),
            ("mixed_fp8_bf16", [self._matmul_fact(rhs_dtype=torch.bfloat16)], False),
            ("dynamic_m", [self._matmul_fact(static_m=None)], False),
            ("no_matmul", [], False),
            ("two_matmuls", [self._matmul_fact(), self._matmul_fact()], False),
        )
        for name, facts, expected in cases:
            env = self._make_cute_env()
            env.config_spec.matmul_facts.extend(facts)
            with self.subTest(name=name):
                self.assertEqual(
                    CuteFp8GemmSkinnyMHeuristic.is_eligible(env, MagicMock()),
                    expected,
                )

    def test_seed_config_contents(self) -> None:
        env = self._make_cute_env()
        env.config_spec.matmul_facts.append(self._matmul_fact(static_m=1))
        seed = CuteFp8GemmSkinnyMHeuristic.get_seed_config(env, MagicMock())
        assert seed is not None
        self.assertEqual(seed.config["block_sizes"], _FP8_SKINNY_M_SEED_BLOCK_SIZES)
        self.assertEqual(seed.config["num_threads"], _FP8_SKINNY_M_SEED_NUM_THREADS)
        self.assertEqual(
            seed.config["cute_vector_widths"], _FP8_SKINNY_M_SEED_VECTOR_WIDTHS
        )

    def test_compiler_seed_configs_records_heuristic(self) -> None:
        # The heuristic must be wired into the cute backend registry so the
        # generic compiler_seed_configs() path emits its seed and records it.
        env = self._make_cute_env()
        env.config_spec.matmul_facts.append(self._matmul_fact(static_m=1))
        configs = compiler_seed_configs(env, MagicMock())
        self.assertIn(
            _FP8_SKINNY_M_SEED_BLOCK_SIZES,
            [config.config["block_sizes"] for config in configs],
        )
        self.assertIn(
            CuteFp8GemmSkinnyMHeuristic.name,
            env.config_spec.autotuner_heuristics,
        )

    def test_compiler_seed_configs_skips_large_m(self) -> None:
        env = self._make_cute_env()
        env.config_spec.matmul_facts.append(self._matmul_fact(static_m=1024))
        configs = compiler_seed_configs(env, MagicMock())
        self.assertNotIn(
            CuteFp8GemmSkinnyMHeuristic.name,
            env.config_spec.autotuner_heuristics,
        )
        self.assertNotIn(
            _FP8_SKINNY_M_SEED_BLOCK_SIZES,
            [config.config["block_sizes"] for config in configs],
        )

    @onlyBackends(["cute"])
    @skipIfRefEager("Compiler seed configs are not generated in ref eager mode")
    def test_seed_in_initial_population_for_skinny_m(self) -> None:
        @helion.kernel(backend="cute", static_shapes=True)
        def fp8_gemm_skinny_m(
            x: torch.Tensor,
            y: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_n in hl.tile(n):
                acc = hl.zeros([m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[:, tile_k], y[tile_k, tile_n], acc=acc)
                acc = acc * scale_a[:, tile_n] * scale_b[tile_n]
                out[:, tile_n] = acc.to(torch.bfloat16)
            return out

        m, k, n = 1, 4096, 4096
        x = torch.randn([m, k], device=DEVICE, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        y = torch.randn([k, n], device=DEVICE, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        scale_a = torch.ones([m, n], device=DEVICE)
        scale_b = torch.ones([n], device=DEVICE)

        with patch_cute_mma_support():
            bound = fp8_gemm_skinny_m.bind((x, y, scale_a, scale_b))

        device_ir = bound.host_function.device_ir
        self.assertTrue(CuteFp8GemmSkinnyMHeuristic.is_eligible(bound.env, device_ir))
        self.assertIn(
            CuteFp8GemmSkinnyMHeuristic.name,
            bound.config_spec.autotuner_heuristics,
        )
        seed = CuteFp8GemmSkinnyMHeuristic.get_seed_config(bound.env, device_ir)
        assert seed is not None
        self.assertEqual(seed.config["block_sizes"], _FP8_SKINNY_M_SEED_BLOCK_SIZES)
        self.assertIn(
            _FP8_SKINNY_M_SEED_BLOCK_SIZES,
            [
                config.config["block_sizes"]
                for config in bound.config_spec.compiler_seed_configs
            ],
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
        # FFI-eligible shapes have both DEFAULT-layout and direct-entry seeds.
        # Callers decide whether both are expected in the supplied population;
        # every cluster_m=2 seed must still match the common tile envelope.
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

    def test_cute_flash_accepts_extra_knobs(self) -> None:
        self.assertIn(FLASH_PIPELINE_FAMILY_KEY, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertIn(FLASH_PIPELINE_FAMILY_KEY, FLASH_CONFIG_KEYS)
        self.assertEqual(len(FLASH_PIPELINE_FAMILIES), 15)
        self.assertEqual(
            set(FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS),
            {
                FLASH_TOPOLOGY_KEY,
                FLASH_CGA2_LOCAL_KEY,
                FLASH_CLC_KEY,
                FLASH_LOCAL_TMA_PARTITION_KEY,
                FLASH_TENSOR_4D_TMA_KEY,
                FLASH_USE_2CTA_KEY,
            },
        )
        for legacy_key in FLASH_LEGACY_CONFIG_KEYS:
            self.assertIn(legacy_key, FLASH_CONFIG_KEYS)
            self.assertNotIn(legacy_key, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertIn(FLASH_MMA_INTERLEAVE_KEY, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertIn(FLASH_Q_TILE_COUNT_KEY, FLASH_DERIVED_CONFIG_KEYS)
        self.assertNotIn(FLASH_Q_TILE_COUNT_KEY, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertNotIn(FLASH_Q_TILE_COUNT_KEY, FLASH_LEGACY_CONFIG_KEYS)
        self.assertIn(FLASH_WAIT_HINT_KEY, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertIn(FLASH_EXP2_PACKET_KEY, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertIn(FLASH_STAT_TRANSPORT_KEY, FLASH_AUTOTUNE_CONFIG_KEYS)
        self.assertIn(FLASH_PERSISTENT_CTAS_PER_SM_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_P_STORE_REP_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_S_LOAD_REP_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_PRECOMPUTE_QK_DESC_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_RECOMPUTE_TILE_COORDS_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_FIRST_LOAD_ORDER_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_KV_ORDER_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_SOFTMAX_DISC_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_EPI_STG_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_EPI_STG_STORE_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_EPI_STG_GMEM_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_CORR_TILE_SIZE_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_RESCALE_CHUNK_COLS_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_SKIP_RESCALE_STATS_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_CLC_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_CLC_HEADS_PER_BATCH_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_CLC_PDL_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_CLC_STAGES_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_LOCAL_TMA_PARTITION_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_TENSOR_4D_TMA_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_SOFTMAX_REGS_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_CORR_REGS_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_OTHER_REGS_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_MASKED_E2E_SCHEDULE_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_ROLE_MAP_KEY, FLASH_CONFIG_KEYS)
        self.assertIn(FLASH_SMALL_BIASED_KEY, FLASH_CONFIG_KEYS)

    def test_cute_flash_seed_helper_dense_hd64_families(self) -> None:
        for num_kv in (16, 32, 62):
            seed = flash_attention_seed_config(64, num_kv)
            assert seed is not None
            config = seed.config
            self.assertEqual(config["block_sizes"], [1, 128, 128])
            self.assertEqual(config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
            self.assertEqual(config[FLASH_S_STAGE_KEY], 2)
            self.assertEqual(config[FLASH_KV_STAGE_KEY], 3)
            self.assertTrue(config[FLASH_PERSISTENT_KEY])
            self.assertEqual(config[FLASH_PERSISTENT_CTAS_PER_SM_KEY], 1)
            self.assertEqual(config[FLASH_E2E_SCHEDULE_KEY], "8/2")
            self.assertEqual(config[FLASH_E2E_OFFSET_KEY], 2)
            self.assertEqual(config[FLASH_E2E_OFFSET0_KEY], 2)
            self.assertTrue(config[FLASH_SOFTMAX_DISC_KEY])
            self.assertEqual(config[FLASH_DISC_PIPE_KEY], 4)
            self.assertEqual(config[FLASH_P_STORE_REP_KEY], 16)
            self.assertEqual(config[FLASH_S_LOAD_REP_KEY], 32)
            self.assertEqual(config[FLASH_PRECOMPUTE_QK_DESC_KEY], num_kv == 512)
            self.assertFalse(config[FLASH_RECOMPUTE_TILE_COORDS_KEY])
            self.assertEqual(config[FLASH_FIRST_LOAD_ORDER_KEY], 0)
            self.assertEqual(config[FLASH_KV_ORDER_KEY], "ascending")
            self.assertEqual(config[FLASH_SOFTMAX_REGS_KEY], 184)
            self.assertEqual(config[FLASH_OTHER_REGS_KEY], 48)
            self.assertTrue(config[FLASH_EPI_TMA_KEY])
            self.assertFalse(config[FLASH_EPI_STG_KEY])
            self.assertEqual(config[FLASH_CORR_TILE_SIZE_KEY], 16)
            self.assertEqual(config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
            self.assertEqual(config[FLASH_RESCALE_THRESHOLD_KEY], 8.0)
            self.assertFalse(config[FLASH_SKIP_RESCALE_STATS_KEY])
            self.assertFalse(config[FLASH_PACKED_REDUCE_KEY])
            for legacy_key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
                self.assertNotIn(legacy_key, config)

        for (
            num_kv,
            e2e_schedule,
            offset,
            offset0,
            disc_pipe,
            epi_tma,
            rescale_threshold,
            rescale_chunk_cols,
            skip_rescale_stats,
            persistent_ctas_per_sm,
            kv_order,
            corr_tile_size,
            softmax_disc,
            first_load_order,
            softmax_regs,
            corr_regs,
            other_regs,
            local_tma_partition,
            cga2_local,
            clc,
            clc_heads_per_batch,
        ) in (
            (
                64,
                "8/2",
                2,
                2,
                3,
                True,
                8.0,
                16,
                False,
                1,
                "ascending",
                16,
                True,
                0,
                184,
                64,
                48,
                False,
                False,
                False,
                0,
            ),
            (
                128,
                "8/2",
                3,
                2,
                3,
                False,
                8.0,
                16,
                False,
                1,
                "ascending",
                16,
                True,
                0,
                184,
                64,
                48,
                False,
                False,
                False,
                0,
            ),
            (
                256,
                "8/2",
                0,
                1,
                1,
                True,
                8.0,
                8,
                False,
                1,
                "descending",
                8,
                False,
                0,
                200,
                64,
                40,
                False,
                False,
                False,
                0,
            ),
            (
                384,
                "8/2",
                0,
                3,
                1,
                True,
                8.0,
                8,
                False,
                1,
                "ascending",
                8,
                False,
                0,
                200,
                72,
                32,
                False,
                False,
                False,
                0,
            ),
            (
                512,
                "8/2",
                0,
                0,
                3,
                False,
                8.0,
                8,
                False,
                1,
                "descending",
                16,
                False,
                4,
                200,
                80,
                32,
                False,
                False,
                False,
                0,
            ),
            (
                1024,
                "16/4",
                0,
                0,
                1,
                True,
                8.0,
                8,
                False,
                1,
                "descending",
                8,
                False,
                0,
                200,
                72,
                40,
                False,
                False,
                True,
                32,
            ),
        ):
            seed = flash_attention_seed_config(64, num_kv)
            assert seed is not None
            config = seed.config
            self.assertEqual(config["block_sizes"], [1, 128, 128])
            if cga2_local:
                expected_family = "fa4_cga2_local"
            elif clc:
                expected_family = (
                    "fa4_clc_local_tma" if local_tma_partition else "fa4_clc"
                )
            elif local_tma_partition:
                expected_family = "fa4_local_tma"
            else:
                expected_family = "fa4"
            self.assertEqual(config[FLASH_PIPELINE_FAMILY_KEY], expected_family)
            self.assertEqual(config[FLASH_S_STAGE_KEY], 2)
            self.assertEqual(config[FLASH_KV_STAGE_KEY], 2)
            self.assertTrue(config[FLASH_PERSISTENT_KEY])
            self.assertEqual(
                config[FLASH_PERSISTENT_CTAS_PER_SM_KEY], persistent_ctas_per_sm
            )
            self.assertEqual(config[FLASH_E2E_SCHEDULE_KEY], e2e_schedule)
            self.assertEqual(config[FLASH_E2E_OFFSET0_KEY], offset0)
            self.assertEqual(config[FLASH_E2E_OFFSET_KEY], offset)
            self.assertEqual(config[FLASH_SOFTMAX_DISC_KEY], softmax_disc)
            self.assertEqual(config[FLASH_DISC_PIPE_KEY], disc_pipe)
            self.assertEqual(config[FLASH_SPLIT_P_ARRIVE_KEY], num_kv == 512)
            self.assertEqual(config[FLASH_P_STORE_REP_KEY], 16)
            self.assertEqual(config[FLASH_S_LOAD_REP_KEY], 32)
            self.assertEqual(config[FLASH_PRECOMPUTE_QK_DESC_KEY], num_kv == 512)
            self.assertFalse(config[FLASH_RECOMPUTE_TILE_COORDS_KEY])
            self.assertEqual(config[FLASH_FIRST_LOAD_ORDER_KEY], first_load_order)
            self.assertEqual(config[FLASH_KV_ORDER_KEY], kv_order)
            self.assertEqual(config[FLASH_SOFTMAX_REGS_KEY], softmax_regs)
            self.assertEqual(config[FLASH_CORR_REGS_KEY], corr_regs)
            self.assertEqual(config[FLASH_OTHER_REGS_KEY], other_regs)
            self.assertEqual(config[FLASH_EPI_TMA_KEY], epi_tma)
            self.assertEqual(config[FLASH_EPI_STG_KEY], num_kv >= 512 and not epi_tma)
            self.assertEqual(config[FLASH_CORR_TILE_SIZE_KEY], corr_tile_size)
            self.assertEqual(config[FLASH_RESCALE_THRESHOLD_KEY], rescale_threshold)
            self.assertEqual(config[FLASH_RESCALE_CHUNK_COLS_KEY], rescale_chunk_cols)
            self.assertEqual(config[FLASH_SKIP_RESCALE_STATS_KEY], skip_rescale_stats)
            self.assertTrue(config[FLASH_PACKED_REDUCE_KEY])
            self.assertEqual(config[FLASH_CLC_HEADS_PER_BATCH_KEY], clc_heads_per_batch)
            for legacy_key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
                self.assertNotIn(legacy_key, config)

        very_long_seed = flash_attention_seed_config(64, 2048)
        assert very_long_seed is not None
        very_long_config = very_long_seed.config
        self.assertEqual(very_long_config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertEqual(very_long_config[FLASH_KV_STAGE_KEY], 3)
        self.assertEqual(very_long_config[FLASH_E2E_SCHEDULE_KEY], "16/4")
        self.assertEqual(very_long_config[FLASH_E2E_OFFSET_KEY], 0)
        self.assertEqual(very_long_config[FLASH_E2E_OFFSET0_KEY], 0)
        self.assertEqual(very_long_config[FLASH_DISC_PIPE_KEY], 1)
        self.assertFalse(very_long_config[FLASH_SPLIT_P_ARRIVE_KEY])
        self.assertFalse(very_long_config[FLASH_PRECOMPUTE_QK_DESC_KEY])
        self.assertFalse(very_long_config[FLASH_RECOMPUTE_TILE_COORDS_KEY])
        self.assertEqual(very_long_config[FLASH_FIRST_LOAD_ORDER_KEY], 4)
        self.assertEqual(very_long_config[FLASH_KV_ORDER_KEY], "descending")
        self.assertEqual(very_long_config[FLASH_SOFTMAX_REGS_KEY], 192)
        self.assertEqual(very_long_config[FLASH_CORR_REGS_KEY], 80)
        self.assertEqual(very_long_config[FLASH_OTHER_REGS_KEY], 32)
        self.assertFalse(very_long_config[FLASH_EPI_TMA_KEY])
        self.assertTrue(very_long_config[FLASH_EPI_STG_KEY])
        self.assertEqual(very_long_config[FLASH_CORR_TILE_SIZE_KEY], 8)
        self.assertEqual(very_long_config.get(FLASH_ROLE_MAP_KEY, "helion"), "helion")
        self.assertEqual(very_long_config[FLASH_RESCALE_THRESHOLD_KEY], 8.0)
        self.assertEqual(very_long_config[FLASH_RESCALE_CHUNK_COLS_KEY], 8)
        self.assertTrue(very_long_config[FLASH_PACKED_REDUCE_KEY])
        for legacy_key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            self.assertNotIn(legacy_key, very_long_config)

        dense_sp_seed = flash_attention_seed_config(
            64,
            256,
            seed_kind="dense_sp",
        )
        assert dense_sp_seed is not None
        dense_sp_config = dense_sp_seed.config
        self.assertFalse(dense_sp_config[FLASH_SOFTMAX_DISC_KEY])
        self.assertFalse(dense_sp_config[FLASH_SPLIT_P_ARRIVE_KEY])
        self.assertEqual(dense_sp_config[FLASH_P_STORE_REP_KEY], 32)
        self.assertEqual(dense_sp_config[FLASH_S_LOAD_REP_KEY], 32)
        self.assertTrue(dense_sp_config[FLASH_PRECOMPUTE_QK_DESC_KEY])
        self.assertEqual(dense_sp_config[FLASH_FIRST_LOAD_ORDER_KEY], 1)
        self.assertEqual(dense_sp_config[FLASH_KV_ORDER_KEY], "descending")
        self.assertEqual(dense_sp_config[FLASH_CORR_REGS_KEY], 80)
        self.assertEqual(dense_sp_config[FLASH_OTHER_REGS_KEY], 32)
        self.assertEqual(dense_sp_config[FLASH_PIPELINE_FAMILY_KEY], "fa4_local_tma_4d")
        dense_sp_512_seed = flash_attention_seed_config(
            64,
            512,
            seed_kind="dense_sp",
        )
        assert dense_sp_512_seed is not None
        dense_sp_512_config = dense_sp_512_seed.config
        self.assertFalse(dense_sp_512_config[FLASH_EPI_TMA_KEY])
        self.assertTrue(dense_sp_512_config[FLASH_EPI_STG_KEY])
        dense_sp_2048_seed = flash_attention_seed_config(
            64,
            2048,
            seed_kind="dense_sp",
        )
        assert dense_sp_2048_seed is not None
        dense_sp_2048_config = dense_sp_2048_seed.config
        self.assertEqual(dense_sp_2048_config[FLASH_RESCALE_THRESHOLD_KEY], 8.0)
        self.assertEqual(dense_sp_2048_config[FLASH_SOFTMAX_REGS_KEY], 200)
        self.assertIsNone(flash_attention_seed_config(64, 128, seed_kind="dense_sp"))

        short_seed = flash_attention_seed_config(64, 8)
        assert short_seed is not None
        self.assertEqual(short_seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertNotIn(FLASH_E2E_OFFSET_KEY, short_seed.config)

        sparse_seed = flash_attention_seed_config(
            64,
            64,
            has_kv_tile_pruning=True,
            requires_ws_overlap=True,
        )
        assert sparse_seed is not None
        self.assertEqual(sparse_seed.config[FLASH_PIPELINE_FAMILY_KEY], "ws_overlap")
        self.assertTrue(sparse_seed.config[FLASH_PACKED_REDUCE_KEY])
        self.assertNotIn(FLASH_E2E_OFFSET_KEY, sparse_seed.config)

        small_seed = flash_attention_seed_config(
            64,
            1,
            small_biased_candidate=True,
        )
        assert small_seed is not None
        self.assertTrue(small_seed.config[FLASH_SMALL_BIASED_KEY])

    def test_cute_flash_seed_helper_causal_lpt_family(self) -> None:
        expected = {
            32: (0, 2, 200, 8, "inherit", "helion", True, 16),
            64: (0, 2, 200, 8, "inherit", "helion", True, 16),
            128: (0, 2, 192, 8, "16/4", "fa4", True, 16),
            256: (1, 4, 184, 4, "16/4", "helion", False, 16),
            512: (9, 4, 184, 1, "16/4", "helion", False, 32),
            1024: (9, 4, 184, 1, "16/4", "helion", False, 32),
            4096: (9, 4, 184, 1, "16/4", "helion", False, 32),
        }
        for (
            num_kv,
            (
                offset,
                disc_pipe,
                softmax_regs,
                swizzle,
                masked_schedule,
                role_map,
                epi_tma,
                rescale_chunk_cols,
            ),
        ) in expected.items():
            seed = flash_attention_seed_config(
                64,
                num_kv,
                is_causal=True,
                seed_kind="causal_lpt",
            )
            assert seed is not None
            config = seed.config
            self.assertEqual(config["block_sizes"], [1, 128, 128])
            self.assertEqual(config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
            self.assertEqual(config[FLASH_S_STAGE_KEY], 2)
            self.assertEqual(config[FLASH_KV_STAGE_KEY], 2)
            self.assertFalse(config[FLASH_PERSISTENT_KEY])
            self.assertEqual(config[FLASH_E2E_SCHEDULE_KEY], "8/2")
            self.assertEqual(config[FLASH_MASKED_E2E_SCHEDULE_KEY], masked_schedule)
            self.assertEqual(config[FLASH_ROLE_MAP_KEY], role_map)
            self.assertEqual(config[FLASH_E2E_OFFSET_KEY], offset)
            self.assertEqual(config[FLASH_DISC_PIPE_KEY], disc_pipe)
            self.assertEqual(config[FLASH_EPI_TMA_KEY], epi_tma)
            self.assertEqual(config[FLASH_RESCALE_CHUNK_COLS_KEY], rescale_chunk_cols)
            self.assertEqual(config[FLASH_RESCALE_THRESHOLD_KEY], 8.0)
            self.assertTrue(config[FLASH_PACKED_REDUCE_KEY])
            self.assertEqual(config[FLASH_CAUSAL_LPT_SWIZZLE_KEY], swizzle)
            self.assertEqual(config[FLASH_CAUSAL_KV_ORDER_KEY], "descending")
            self.assertEqual(config[FLASH_SOFTMAX_REGS_KEY], softmax_regs)

        self.assertIsNone(
            flash_attention_seed_config(64, 16, is_causal=True, seed_kind="causal_lpt")
        )
        self.assertIsNone(
            flash_attention_seed_config(64, 96, is_causal=True, seed_kind="causal_lpt")
        )
        self.assertIsNone(
            flash_attention_seed_config(
                64,
                64,
                is_causal=True,
                requires_ws_overlap=True,
                seed_kind="causal_lpt",
            )
        )
        split_seed = flash_attention_seed_config(
            64, 64, is_causal=True, seed_kind="causal_split"
        )
        assert split_seed is not None
        lpt_seed = flash_attention_seed_config(
            64, 64, is_causal=True, seed_kind="causal_lpt"
        )
        assert lpt_seed is not None
        self.assertNotEqual(split_seed, lpt_seed)
        self.assertEqual(split_seed.config[FLASH_KV_STAGE_KEY], 2)
        self.assertEqual(split_seed.config[FLASH_E2E_SCHEDULE_KEY], "8/2")
        self.assertEqual(split_seed.config[FLASH_MASKED_E2E_SCHEDULE_KEY], "16/4")
        self.assertEqual(split_seed.config[FLASH_ROLE_MAP_KEY], "fa4")
        self.assertTrue(split_seed.config[FLASH_CAUSAL_LOOP_SPLIT_KEY])
        self.assertEqual(split_seed.config[FLASH_E2E_OFFSET_KEY], 0)
        self.assertEqual(split_seed.config[FLASH_E2E_OFFSET0_KEY], 0)
        self.assertEqual(split_seed.config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
        self.assertEqual(split_seed.config[FLASH_CAUSAL_LPT_SWIZZLE_KEY], 8)
        split_seed_256 = flash_attention_seed_config(
            64, 256, is_causal=True, seed_kind="causal_split"
        )
        assert split_seed_256 is not None
        self.assertEqual(split_seed_256.config[FLASH_MASKED_E2E_SCHEDULE_KEY], "16/4")
        self.assertEqual(split_seed_256.config[FLASH_E2E_OFFSET0_KEY], 11)
        self.assertEqual(split_seed_256.config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
        self.assertEqual(split_seed_256.config[FLASH_CAUSAL_LPT_SWIZZLE_KEY], 4)
        split_seed_512 = flash_attention_seed_config(
            64, 512, is_causal=True, seed_kind="causal_split"
        )
        assert split_seed_512 is not None
        lpt_seed_512 = flash_attention_seed_config(
            64, 512, is_causal=True, seed_kind="causal_lpt"
        )
        assert lpt_seed_512 is not None
        self.assertNotEqual(split_seed_512, lpt_seed_512)
        self.assertNotEqual(
            resolve_flash_config(64, 512, split_seed_512.config, is_causal=True),
            resolve_flash_config(64, 512, lpt_seed_512.config, is_causal=True),
        )
        self.assertEqual(split_seed_512.config[FLASH_E2E_SCHEDULE_KEY], "16/4")
        self.assertEqual(split_seed_512.config[FLASH_MASKED_E2E_SCHEDULE_KEY], "8/2")
        self.assertEqual(split_seed_512.config[FLASH_E2E_OFFSET_KEY], 0)
        self.assertEqual(split_seed_512.config[FLASH_E2E_OFFSET0_KEY], 14)
        self.assertEqual(split_seed_512.config[FLASH_DISC_PIPE_KEY], 3)
        self.assertEqual(split_seed_512.config[FLASH_EXP2_PACKET_KEY], "4x1")
        self.assertEqual(split_seed_512.config[FLASH_WAIT_HINT_KEY], 0)
        self.assertFalse(split_seed_512.config[FLASH_EPI_TMA_KEY])
        self.assertTrue(split_seed_512.config[FLASH_EPI_STG_KEY])
        self.assertEqual(split_seed_512.config[FLASH_EPI_STG_GMEM_KEY], "pair")
        self.assertEqual(split_seed_512.config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
        self.assertEqual(split_seed_512.config[FLASH_ROLE_MAP_KEY], "fa4")
        split_seed_4096 = flash_attention_seed_config(
            64, 4096, is_causal=True, seed_kind="causal_split"
        )
        assert split_seed_4096 is not None
        self.assertEqual(split_seed_4096.config, split_seed_512.config)

        causal_seeds = flash_attention_seed_configs(64, 64, is_causal=True)
        self.assertLessEqual(
            {"fa4", "ws_overlap"},
            {seed.config.get(FLASH_PIPELINE_FAMILY_KEY) for seed in causal_seeds},
        )

    def test_cute_flash_family_seeds_cover_legal_search_surface(self) -> None:
        cases = (
            (64, 256, torch.float16, False, False, True),
            (64, 1024, torch.float16, False, False, True),
            (64, 2048, torch.float16, False, False, True),
            (64, 512, torch.float16, True, False, False),
            (128, 256, torch.float16, False, False, True),
            (64, 256, torch.bfloat16, False, False, False),
            (64, 256, torch.float16, False, True, False),
        )
        for (
            head_dim,
            num_kv,
            dtype,
            is_causal,
            requires_ws_overlap,
            standard_dense_output,
        ) in cases:
            with self.subTest(
                head_dim=head_dim,
                num_kv=num_kv,
                dtype=str(dtype),
                is_causal=is_causal,
                requires_ws_overlap=requires_ws_overlap,
            ):
                fragments = flash_autotune_fragments(
                    head_dim,
                    num_kv,
                    dtype=dtype,
                    is_causal=is_causal,
                    requires_ws_overlap=requires_ws_overlap,
                    standard_dense_output=standard_dense_output,
                )
                family_fragment = fragments[FLASH_PIPELINE_FAMILY_KEY]
                self.assertIsInstance(family_fragment, EnumFragment)
                assert isinstance(family_fragment, EnumFragment)
                legal_families = set(
                    family_fragment.choices
                    if family_fragment.search_choices is None
                    else family_fragment.search_choices
                )

                spec = ConfigSpec(backend=CuteBackend())
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=head_dim,
                    num_kv=num_kv,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    dtype=dtype,
                    is_causal=is_causal,
                    requires_ws_overlap=requires_ws_overlap,
                    standard_dense_output=standard_dense_output,
                )
                spec.compiler_seed_configs = list(
                    flash_attention_seed_configs(
                        head_dim,
                        num_kv,
                        dtype=dtype,
                        is_causal=is_causal,
                        requires_ws_overlap=requires_ws_overlap,
                        standard_dense_output=standard_dense_output,
                    )
                )
                config_gen = ConfigGeneration(spec)
                normalized_seeds = [
                    config for _flat, config in config_gen.seed_flat_config_pairs()
                ]
                self.assertLessEqual(len(normalized_seeds), 6)
                self.assertLessEqual(
                    legal_families,
                    {
                        config.config[FLASH_PIPELINE_FAMILY_KEY]
                        for config in normalized_seeds
                    },
                )
                for family in legal_families:
                    family_gen = spec.create_config_generation(
                        overrides={FLASH_PIPELINE_FAMILY_KEY: family}
                    )
                    family_default = family_gen.unflatten(family_gen.default_flat())
                    self.assertIn(family_default, normalized_seeds)
                if (
                    head_dim == 64
                    and dtype is torch.float16
                    and not is_causal
                    and standard_dense_output
                ):
                    self.assertFalse(
                        any(
                            config.config[FLASH_EXP2_PACKET_KEY]
                            in {"deg1_8x2_corr10", "deg1_16x8"}
                            for config in normalized_seeds
                        )
                    )
                    self.assertEqual(
                        {
                            config.config[FLASH_STAT_TRANSPORT_KEY]
                            for config in normalized_seeds
                            if config.config[FLASH_EXP2_PACKET_KEY] == "deg2_16x6"
                        },
                        {"ring2", "single_final"},
                    )

    def test_cute_flash_sm103_rank0_seed_policy(self) -> None:
        sm103_policy = get_flash_target_policy((10, 3)).tuning
        dense_policies = {
            shape_policy.num_kv: shape_policy
            for shape_policy in sm103_policy.dense_policies
        }
        causal_policies = {
            shape_policy.num_kv: shape_policy
            for shape_policy in sm103_policy.causal_policies
        }
        cases = [(False, num_kv) for num_kv in dense_policies] + [
            (True, num_kv) for num_kv in causal_policies
        ]
        for is_causal, num_kv in cases:
            with self.subTest(is_causal=is_causal, num_kv=num_kv):
                seed_kwargs = {
                    "dtype": torch.float16,
                    "is_causal": is_causal,
                    "standard_dense_output": not is_causal,
                    "standard_causal_output": is_causal,
                }
                seed = flash_attention_seed_config(
                    64,
                    num_kv,
                    target_device_capability=(10, 3),
                    **seed_kwargs,
                )
                assert seed is not None

                self.assertEqual(seed.config[FLASH_P_STORE_REP_KEY], 16)
                self.assertEqual(seed.config[FLASH_S_LOAD_REP_KEY], 32)
                self.assertTrue(seed.config[FLASH_SPLIT_P_ARRIVE_KEY])
                self.assertEqual(seed.config[FLASH_RESCALE_THRESHOLD_KEY], 8.0)

                if is_causal:
                    resident_policy = causal_policies[num_kv]
                    self.assertEqual(seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
                    self.assertFalse(seed.config[FLASH_PERSISTENT_KEY])
                    self.assertEqual(seed.config[FLASH_E2E_SCHEDULE_KEY], "16/6")
                    self.assertEqual(seed.config[FLASH_MASKED_E2E_SCHEDULE_KEY], "16/6")
                    self.assertEqual(seed.config[FLASH_EXP2_PACKET_KEY], "deg2_16x6")
                    self.assertEqual(seed.config[FLASH_STAT_TRANSPORT_KEY], "ring2")
                    self.assertEqual(seed.config[FLASH_Q_TILE_COUNT_KEY], 2)
                    self.assertEqual(seed.config[FLASH_CORR_REGS_KEY], 64)
                    self.assertEqual(seed.config[FLASH_OTHER_REGS_KEY], 48)
                    self.assertEqual(
                        (
                            seed.config[FLASH_KV_STAGE_KEY],
                            seed.config[FLASH_E2E_OFFSET_KEY],
                            seed.config[FLASH_E2E_OFFSET0_KEY],
                            seed.config[FLASH_SOFTMAX_REGS_KEY],
                            seed.config[FLASH_FIRST_LOAD_ORDER_KEY],
                            seed.config[FLASH_ROLE_MAP_KEY],
                            seed.config[FLASH_EPI_TMA_KEY],
                        ),
                        (
                            resident_policy.kv_stage,
                            resident_policy.e2e_offset,
                            resident_policy.e2e_offset0,
                            resident_policy.softmax_regs,
                            resident_policy.first_load_order,
                            resident_policy.role_map,
                            resident_policy.epi_tma,
                        ),
                    )
                else:
                    resident_policy = dense_policies[num_kv]
                    self.assertEqual(
                        (
                            seed.config[FLASH_EXP2_PACKET_KEY],
                            seed.config[FLASH_E2E_SCHEDULE_KEY],
                            seed.config[FLASH_E2E_OFFSET_KEY],
                            seed.config[FLASH_E2E_OFFSET0_KEY],
                            seed.config[FLASH_STAT_TRANSPORT_KEY],
                            seed.config[FLASH_PIPELINE_FAMILY_KEY],
                            seed.config[FLASH_KV_STAGE_KEY],
                            seed.config[FLASH_PERSISTENT_KEY],
                            seed.config[FLASH_CORR_REGS_KEY],
                            seed.config[FLASH_OTHER_REGS_KEY],
                        ),
                        (
                            resident_policy.exp2_packet,
                            resident_policy.e2e_schedule,
                            resident_policy.e2e_offset,
                            resident_policy.e2e_offset0,
                            resident_policy.stat_transport,
                            resident_policy.pipeline_family,
                            resident_policy.kv_stage,
                            resident_policy.persistent,
                            resident_policy.corr_regs,
                            resident_policy.other_regs,
                        ),
                    )
                self.assertEqual(seed.config[FLASH_Q_TILE_COUNT_KEY], 2)

                ranked = flash_attention_seed_configs(
                    64,
                    num_kv,
                    target_device_capability=(10, 3),
                    **seed_kwargs,
                )
                self.assertEqual(ranked[0], seed)
                if is_causal and num_kv == 1024:
                    self.assertEqual(
                        sum(
                            candidate.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
                            and candidate.config.get(FLASH_KV_STAGE_KEY) == 2
                            for candidate in ranked
                        ),
                        1,
                    )

                if (is_causal, num_kv) not in ((False, 512), (True, 512)):
                    continue

                spec = ConfigSpec(
                    backend=CuteBackend(),
                    target_device_capability=(10, 3),
                )
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=64,
                    num_kv=num_kv,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    **seed_kwargs,
                )
                env = MagicMock()
                env.backend_name = "cute"
                env.config_spec = spec
                env.settings = Settings()
                with patch(
                    "helion._compiler.autotuner_heuristics.HEURISTICS_BY_BACKEND",
                    {
                        "cute": (
                            CuteFlashAttentionHeuristic,
                            CuteFlashAttentionCausalLptHeuristic,
                        )
                    },
                ):
                    compiler_seeds = compiler_seed_configs(env, MagicMock())
                self.assertEqual(compiler_seeds[0], seed)
                self.assertIsNone(spec.compiler_default_config)
                self.assertEqual(spec.autotune_seed_configs()[0], seed)

                spec.compiler_seed_configs = compiler_seeds
                config_gen = ConfigGeneration(spec)
                roundtrip = config_gen.unflatten(config_gen.flatten(seed))
                self.assertEqual(roundtrip, seed)

    def test_cute_flash_sm103_seed_policy_changes_cache_identity(self) -> None:
        def make_spec(
            target: tuple[int, int], *, is_causal: bool = False
        ) -> ConfigSpec:
            spec = ConfigSpec(
                backend=CuteBackend(),
                target_device_capability=target,
            )
            for block_id, size_hint in enumerate((1, 128, 128)):
                spec.block_sizes.append(
                    BlockSizeSpec(block_id=block_id, size_hint=size_hint)
                )
            spec.enable_cute_flash_search(
                head_dim=64,
                num_kv=1024,
                block_size_targets={0: 1, 1: 128, 2: 128},
                dtype=torch.float16,
                is_causal=is_causal,
                standard_dense_output=not is_causal,
                standard_causal_output=is_causal,
            )
            spec.compiler_seed_configs = list(
                flash_attention_seed_configs(
                    64,
                    1024,
                    dtype=torch.float16,
                    is_causal=is_causal,
                    standard_dense_output=not is_causal,
                    standard_causal_output=is_causal,
                    target_device_capability=target,
                )
            )
            return spec

        sm100 = make_spec((10, 0))
        sm103 = make_spec((10, 3))

        self.assertEqual(sm100.structural_fingerprint(), sm103.structural_fingerprint())
        self.assertNotEqual(
            sm100.compiler_seed_configs,
            sm103.compiler_seed_configs,
        )
        self.assertNotEqual(
            sm100.cache_fingerprint_hash(),
            sm103.cache_fingerprint_hash(),
        )

        original_hash = sm103.cache_fingerprint_hash()
        target_policy = get_flash_target_policy((10, 3))
        unrelated_tuning = dataclasses.replace(
            target_policy.tuning,
            dense_policies=tuple(
                dataclasses.replace(policy, e2e_offset=policy.e2e_offset + 1)
                if policy.num_kv == 256
                else policy
                for policy in target_policy.tuning.dense_policies
            ),
        )
        with patch(
            "helion._compiler.cute.flash_policy.get_flash_target_policy",
            return_value=dataclasses.replace(target_policy, tuning=unrelated_tuning),
        ):
            self.assertEqual(sm103.cache_fingerprint_hash(), original_hash)

        dense_policy = target_policy.tuning.dense_policy(1024)
        assert dense_policy is not None
        changed_tuning = dataclasses.replace(
            target_policy.tuning,
            dense_policies=tuple(
                dataclasses.replace(
                    policy,
                    softmax_lowering=FlashSoftmaxLowering.STANDARD,
                    probability_log2_shift=0,
                )
                if policy.num_kv == 1024
                else policy
                for policy in target_policy.tuning.dense_policies
            ),
        )
        with patch(
            "helion._compiler.cute.flash_policy.get_flash_target_policy",
            return_value=dataclasses.replace(target_policy, tuning=changed_tuning),
        ):
            self.assertNotEqual(sm103.cache_fingerprint_hash(), original_hash)

        causal_sm103 = make_spec((10, 3), is_causal=True)
        causal_original_hash = causal_sm103.cache_fingerprint_hash()
        with patch(
            "helion._compiler.cute.flash_policy.get_flash_target_policy",
            return_value=dataclasses.replace(target_policy, tuning=unrelated_tuning),
        ):
            self.assertEqual(
                causal_sm103.cache_fingerprint_hash(), causal_original_hash
            )

        changed_causal_tuning = dataclasses.replace(
            target_policy.tuning,
            causal_policies=tuple(
                dataclasses.replace(
                    policy,
                    softmax_lowering=FlashSoftmaxLowering.STANDARD,
                )
                if policy.num_kv == 1024
                else policy
                for policy in target_policy.tuning.causal_policies
            ),
        )
        with patch(
            "helion._compiler.cute.flash_policy.get_flash_target_policy",
            return_value=dataclasses.replace(
                target_policy, tuning=changed_causal_tuning
            ),
        ):
            self.assertNotEqual(
                causal_sm103.cache_fingerprint_hash(), causal_original_hash
            )

    def test_cute_flash_registered_causal_shape_builds_direct_target_seed(self) -> None:
        target_policy = get_flash_target_policy((10, 3))
        extra_policy = FlashCausalTuningPolicy(
            num_kv=768,
            kv_stage=4,
            e2e_offset=2,
            e2e_offset0=3,
            softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            softmax_regs=184,
        )
        extended_tuning = dataclasses.replace(
            target_policy.tuning,
            causal_policies=(*target_policy.tuning.causal_policies, extra_policy),
        )
        with patch(
            "helion._compiler.cute.cute_flash.get_flash_target_policy",
            return_value=dataclasses.replace(
                target_policy,
                tuning=extended_tuning,
            ),
        ):
            seed = flash_attention_seed_config(
                64,
                768,
                dtype=torch.float16,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 3),
            )

        assert seed is not None
        self.assertEqual(seed.config[FLASH_KV_STAGE_KEY], 4)
        self.assertEqual(seed.config[FLASH_E2E_OFFSET_KEY], 2)
        self.assertEqual(seed.config[FLASH_E2E_OFFSET0_KEY], 3)
        self.assertTrue(seed.config[FLASH_CAUSAL_LOOP_SPLIT_KEY])
        self.assertEqual(seed.config[FLASH_CAUSAL_KV_ORDER_KEY], "descending")

    def test_cute_flash_target_policy_seed_must_round_trip(self) -> None:
        target_policy = get_flash_target_policy((10, 3))
        invalid_dense_policy = FlashDenseTuningPolicy(
            num_kv=256,
            exp2_packet="deg1_16x8",
            e2e_schedule="8/2",
            e2e_offset=0,
            e2e_offset0=0,
            stat_transport="single",
        )
        invalid_tuning = dataclasses.replace(
            target_policy.tuning,
            dense_policies=(
                invalid_dense_policy,
                *target_policy.tuning.dense_policies[1:],
            ),
        )
        with (
            patch(
                "helion._compiler.cute.cute_flash.get_flash_target_policy",
                return_value=dataclasses.replace(
                    target_policy,
                    tuning=invalid_tuning,
                ),
            ),
            self.assertRaisesRegex(ValueError, "does not round-trip"),
        ):
            flash_attention_seed_config(
                64,
                256,
                dtype=torch.float16,
                standard_dense_output=True,
                target_device_capability=(10, 3),
            )

    def test_cute_flash_target_policy_preserves_b200_fragment_surface(self) -> None:
        def fragment_signature(
            target: tuple[int, int] | None, *, is_causal: bool
        ) -> tuple[tuple[object, ...], ...]:
            fragments = flash_autotune_fragments(
                64,
                512,
                dtype=torch.float16,
                is_causal=is_causal,
                standard_dense_output=not is_causal,
                standard_causal_output=is_causal,
                target_device_capability=target,
            )
            return tuple(
                (
                    key,
                    fragment.fingerprint(),
                    tuple(cast("EnumFragment", fragment).choices),
                    cast("EnumFragment", fragment).search_choices,
                )
                for key, fragment in fragments.items()
            )

        for is_causal in (False, True):
            baseline = fragment_signature(None, is_causal=is_causal)
            self.assertEqual(baseline, fragment_signature((10, 0), is_causal=is_causal))
            self.assertEqual(
                baseline, fragment_signature((999, 999), is_causal=is_causal)
            )

        sm103_fragments = flash_autotune_fragments(
            64,
            512,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 3),
        )
        first_load = cast("EnumFragment", sm103_fragments[FLASH_FIRST_LOAD_ORDER_KEY])
        self.assertEqual(first_load.choices, (0, 2))
        self.assertEqual(first_load.search_choices, (0,))

    def test_cute_flash_non_sm103_seed_order_is_unchanged(self) -> None:
        targets = (None, (10, 0), (999, 999))
        cases = (
            {
                "num_kv": 1024,
                "is_causal": False,
                "standard_dense_output": True,
            },
            {
                "num_kv": 2048,
                "is_causal": True,
                "standard_causal_output": True,
            },
        )
        for case in cases:
            baseline = flash_attention_seed_configs(
                64,
                case["num_kv"],
                dtype=torch.float16,
                is_causal=case["is_causal"],
                standard_dense_output=case.get("standard_dense_output", False),
                standard_causal_output=case.get("standard_causal_output", False),
            )
            for target in targets:
                with self.subTest(case=case, target=target):
                    actual = flash_attention_seed_configs(
                        64,
                        case["num_kv"],
                        dtype=torch.float16,
                        is_causal=case["is_causal"],
                        standard_dense_output=case.get("standard_dense_output", False),
                        standard_causal_output=case.get(
                            "standard_causal_output", False
                        ),
                        target_device_capability=target,
                    )
                    self.assertEqual(actual, baseline)

    def test_cute_flash_sm103_seed_policy_gates(self) -> None:
        cases = (
            {"standard_dense_output": False},
            {"standard_causal_output": False, "is_causal": True, "num_kv": 512},
            {"standard_dense_output": True, "dtype": torch.bfloat16},
            {"standard_dense_output": True, "head_dim": 128},
            {"standard_dense_output": True, "has_kv_tile_pruning": True},
            {"standard_dense_output": True, "requires_ws_overlap": True},
            {"standard_dense_output": True, "small_biased_candidate": True},
            {"standard_dense_output": True, "num_kv": 768},
            {
                "standard_dense_output": True,
                "block_size_targets": (1, 64, 128),
            },
        )
        for raw_case in cases:
            case = dict(raw_case)
            head_dim = int(case.pop("head_dim", 64))
            num_kv = int(case.pop("num_kv", 256))
            subtest_case = {
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in raw_case.items()
            }
            with self.subTest(case=subtest_case):
                baseline = flash_attention_seed_config(
                    head_dim,
                    num_kv,
                    **case,
                )
                actual = flash_attention_seed_config(
                    head_dim,
                    num_kv,
                    target_device_capability=(10, 3),
                    **case,
                )
                self.assertEqual(actual, baseline)

    def test_cute_flash_dense_degree2_seed_uses_validated_schedule(self) -> None:
        degree1_packets = {"deg1_8x2_corr10", "deg1_16x8"}
        for num_kv in (252, 256, 258, 260, 512, 516, 768, 1024, 1536, 2048, 2052, 4096):
            with self.subTest(num_kv=num_kv):
                seeds = flash_attention_seed_configs(
                    64,
                    num_kv,
                    dtype=torch.float16,
                    standard_dense_output=True,
                )
                self.assertFalse(
                    any(
                        seed.config.get(FLASH_EXP2_PACKET_KEY) in degree1_packets
                        for seed in seeds
                    )
                )
                degree2_seeds = [
                    seed
                    for seed in seeds
                    if seed.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
                ]
                if num_kv < 256 or num_kv % 4:
                    self.assertFalse(degree2_seeds)
                    continue
                self.assertEqual(len(degree2_seeds), 2)
                self.assertEqual(
                    {seed.config[FLASH_STAT_TRANSPORT_KEY] for seed in degree2_seeds},
                    {"ring2", "single_final"},
                )
                config = next(
                    seed.config
                    for seed in degree2_seeds
                    if seed.config[FLASH_STAT_TRANSPORT_KEY] == "single_final"
                )
                ring2_config = next(
                    seed.config
                    for seed in degree2_seeds
                    if seed.config[FLASH_STAT_TRANSPORT_KEY] == "ring2"
                )
                self.assertEqual(
                    ring2_config,
                    {**config, FLASH_STAT_TRANSPORT_KEY: "ring2"},
                )
                fragments = flash_autotune_fragments(
                    64,
                    num_kv,
                    dtype=torch.float16,
                    standard_dense_output=True,
                    pipeline_family_override="fa4_2cta",
                )

                self.assertEqual(config[FLASH_PIPELINE_FAMILY_KEY], "fa4_2cta")
                expected = {
                    FLASH_EXP2_PACKET_KEY: "deg2_16x6",
                    FLASH_E2E_SCHEDULE_KEY: "16/6",
                    FLASH_E2E_OFFSET_KEY: 14 if num_kv <= 512 else 12,
                    FLASH_E2E_OFFSET0_KEY: 0 if num_kv <= 512 else 2,
                    FLASH_KV_STAGE_KEY: 2,
                    FLASH_PERSISTENT_KEY: False,
                    FLASH_STAT_TRANSPORT_KEY: "single_final",
                    FLASH_WAIT_HINT_KEY: 0,
                    FLASH_SOFTMAX_REGS_KEY: 192,
                    FLASH_CORR_REGS_KEY: 72,
                    FLASH_OTHER_REGS_KEY: 40,
                    FLASH_ROLE_MAP_KEY: "fa4",
                    FLASH_FIRST_LOAD_ORDER_KEY: 4,
                    FLASH_CORR_TILE_SIZE_KEY: 16 if num_kv <= 512 else 8,
                    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
                    FLASH_RESCALE_CHUNK_COLS_KEY: 8,
                }
                for key, value in expected.items():
                    self.assertEqual(config[key], value)

                for key, fragment in fragments.items():
                    if key not in expected:
                        self.assertEqual(config[key], fragment.default())
                packet_fragment = fragments[FLASH_EXP2_PACKET_KEY]
                self.assertIsInstance(packet_fragment, EnumFragment)
                assert isinstance(packet_fragment, EnumFragment)
                self.assertIn(
                    config[FLASH_EXP2_PACKET_KEY],
                    packet_fragment.search_choices or (),
                )

                effective_configs = {
                    tuple(
                        sorted(
                            flash_effective_config_values(
                                resolve_flash_config(
                                    64,
                                    num_kv,
                                    seed.config,
                                    dtype=torch.float16,
                                    standard_dense_output=True,
                                )
                            ).items()
                        )
                    )
                    for seed in degree2_seeds
                }
                self.assertEqual(len(effective_configs), 2)

    def test_cute_flash_dense_degree2_seed_gates(self) -> None:
        def has_degree2_seed(
            head_dim: int = 64,
            num_kv: int = 2048,
            **kwargs: object,
        ) -> bool:
            return any(
                seed.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
                for seed in flash_attention_seed_configs(
                    head_dim,
                    num_kv,
                    **kwargs,
                )
            )

        cases = (
            {"standard_dense_output": False},
            {"standard_dense_output": True, "is_causal": True},
            {"standard_dense_output": True, "dtype": torch.bfloat16},
            {"standard_dense_output": True, "head_dim": 128},
            {"standard_dense_output": True, "has_kv_tile_pruning": True},
            {"standard_dense_output": True, "requires_ws_overlap": True},
            {"standard_dense_output": True, "small_biased_candidate": True},
            {
                "standard_dense_output": True,
                "block_size_targets": (1, 64, 128),
            },
            {"standard_dense_output": True, "num_kv": 252},
            {"standard_dense_output": True, "num_kv": 258},
            {"standard_dense_output": True, "num_kv": 2050},
        )
        for case in cases:
            subtest_case = {
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in case.items()
            }
            with self.subTest(case=subtest_case):
                case = dict(case)
                head_dim = int(case.pop("head_dim", 64))
                num_kv = int(case.pop("num_kv", 2048))
                self.assertFalse(has_degree2_seed(head_dim, num_kv, **case))

    def test_cute_flash_degree2_packet_search_is_eligibility_gated(self) -> None:
        def search_choices(
            head_dim: int = 64,
            num_kv: int = 260,
            **kwargs: object,
        ) -> tuple[object, ...]:
            fragment = flash_autotune_fragments(
                head_dim,
                num_kv,
                **kwargs,
            )[FLASH_EXP2_PACKET_KEY]
            self.assertIsInstance(fragment, EnumFragment)
            assert isinstance(fragment, EnumFragment)
            return fragment.search_choices or ()

        self.assertIn(
            "deg2_16x6",
            search_choices(dtype=torch.float16, standard_dense_output=True),
        )
        self.assertIn(
            "deg2_16x6",
            search_choices(
                num_kv=8192,
                dtype=torch.float16,
                is_causal=True,
                standard_causal_output=True,
            ),
        )
        excluded = (
            {"dtype": torch.float16, "standard_dense_output": False},
            {"dtype": torch.bfloat16, "standard_dense_output": True},
            {"head_dim": 128, "dtype": torch.float16, "standard_dense_output": True},
            {
                "num_kv": 252,
                "dtype": torch.float16,
                "standard_dense_output": True,
            },
            {
                "num_kv": 258,
                "dtype": torch.float16,
                "standard_dense_output": True,
            },
            {"num_kv": 768, "dtype": torch.float16, "is_causal": True},
            {"num_kv": 8192, "dtype": torch.float16, "is_causal": True},
            {
                "num_kv": 256,
                "dtype": torch.float16,
                "is_causal": True,
                "standard_causal_output": True,
            },
            {
                "num_kv": 8192,
                "dtype": torch.float16,
                "is_causal": True,
                "standard_causal_output": True,
                "has_kv_tile_pruning": True,
            },
        )
        for case in excluded:
            subtest_case = {
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in case.items()
            }
            with self.subTest(case=subtest_case):
                case = dict(case)
                head_dim = int(case.pop("head_dim", 64))
                num_kv = int(case.pop("num_kv", 260))
                self.assertNotIn(
                    "deg2_16x6",
                    search_choices(head_dim, num_kv, **case),
                )

    def test_cute_flash_dense_degree2_seed_is_in_full_initial_population(
        self,
    ) -> None:
        for num_kv in (256, 260, 516, 768, 1536, 2052, 4096):
            with self.subTest(num_kv=num_kv):
                spec = ConfigSpec(backend=CuteBackend())
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=64,
                    num_kv=num_kv,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    dtype=torch.float16,
                    is_causal=False,
                    standard_dense_output=True,
                )
                spec.compiler_seed_configs = list(
                    flash_attention_seed_configs(
                        64,
                        num_kv,
                        dtype=torch.float16,
                        standard_dense_output=True,
                    )
                )
                config_gen = ConfigGeneration(spec)
                expected = [
                    config
                    for _flat, config in config_gen.seed_flat_config_pairs()
                    if config.config[FLASH_EXP2_PACKET_KEY] == "deg2_16x6"
                ]
                self.assertEqual(len(expected), 2)
                self.assertEqual(
                    {config.config[FLASH_STAT_TRANSPORT_KEY] for config in expected},
                    {"ring2", "single_final"},
                )
                for config in expected:
                    canonical_flat, canonical = config_gen.canonicalize_flat(
                        config_gen.flatten(config)
                    )
                    self.assertEqual(canonical, config)
                    self.assertEqual(config_gen.unflatten(canonical_flat), config)

                profile = get_effort_profile("full").lfbo_pattern_search
                assert profile is not None
                self.assertEqual(profile.initial_population_strategy, "from_random")
                population = config_gen.random_population(profile.initial_population)
                seed_count = len(config_gen.seed_flat_config_pairs())
                for config in expected:
                    self.assertEqual(population.count(config), 1)
                    self.assertIn(config, population[:seed_count])

    def test_cute_flash_does_not_automatically_seed_degree1_causal_packet(
        self,
    ) -> None:
        for num_kv in (256, 512, 768, 4096, 8192):
            with self.subTest(num_kv=num_kv):
                seeds = flash_attention_seed_configs(
                    64,
                    num_kv,
                    is_causal=True,
                    standard_causal_output=True,
                )
                self.assertFalse(
                    any(
                        seed.config.get(FLASH_EXP2_PACKET_KEY) == "hybrid_deg1_16x8"
                        for seed in seeds
                    )
                )

        spec = ConfigSpec(backend=CuteBackend())
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=8192,
            block_size_targets={0: 1, 1: 128, 2: 128},
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
        )
        spec.compiler_seed_configs = list(
            flash_attention_seed_configs(
                64,
                8192,
                is_causal=True,
                standard_causal_output=True,
            )
        )
        normalized = [
            config for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        ]
        self.assertEqual(len(normalized), 4)
        self.assertEqual(
            {
                (
                    config.config[FLASH_PIPELINE_FAMILY_KEY],
                    config.config[FLASH_EXP2_PACKET_KEY],
                )
                for config in normalized
            },
            {
                ("fa4", "1x1"),
                ("fa4", "4x1"),
                ("fa4", "deg2_16x6"),
                ("ws_overlap", "1x1"),
            },
        )

    def test_cute_flash_causal_degree2_seed_uses_validated_schedule(self) -> None:
        expected_params = {
            512: ("fa4", 15, 3, True),
            1024: ("fa4", 1, 14, False),
            2048: ("fa4", 14, 12, False),
            4096: ("helion", 14, 12, False),
            8192: ("helion", 14, 12, False),
        }
        common_configs: list[dict[str, object]] = []
        shape_specific_keys = {
            FLASH_ROLE_MAP_KEY,
            FLASH_E2E_OFFSET_KEY,
            FLASH_E2E_OFFSET0_KEY,
            FLASH_EPI_TMA_KEY,
        }
        for num_kv in (256, *expected_params):
            with self.subTest(num_kv=num_kv):
                seeds = flash_attention_seed_configs(
                    64,
                    num_kv,
                    is_causal=True,
                    standard_causal_output=True,
                )
                degree2_seeds = [
                    seed
                    for seed in seeds
                    if seed.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
                ]
                if num_kv < 512:
                    self.assertFalse(degree2_seeds)
                    continue
                self.assertEqual(len(degree2_seeds), 1)
                config = degree2_seeds[0].config
                role_map, offset, offset0, epi_tma = expected_params[num_kv]
                expected = {
                    FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    FLASH_EXP2_PACKET_KEY: "deg2_16x6",
                    FLASH_E2E_SCHEDULE_KEY: "16/6",
                    FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
                    FLASH_E2E_OFFSET_KEY: offset,
                    FLASH_E2E_OFFSET0_KEY: offset0,
                    FLASH_WAIT_HINT_KEY: 0,
                    FLASH_DISC_PIPE_KEY: 3,
                    FLASH_EPI_TMA_KEY: epi_tma,
                    FLASH_EPI_STG_KEY: False,
                    FLASH_RESCALE_CHUNK_COLS_KEY: 16,
                    FLASH_CAUSAL_LPT_SWIZZLE_KEY: 0,
                    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
                    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
                    FLASH_SOFTMAX_REGS_KEY: 184,
                    FLASH_CORR_REGS_KEY: 64,
                    FLASH_OTHER_REGS_KEY: 48,
                    FLASH_ROLE_MAP_KEY: role_map,
                }
                for key, value in expected.items():
                    self.assertEqual(config[key], value)
                self.assertNotIn(FLASH_USE_2CTA_KEY, config)
                common_configs.append(
                    {
                        key: value
                        for key, value in config.items()
                        if key not in shape_specific_keys
                    }
                )
        self.assertTrue(
            all(config == common_configs[0] for config in common_configs[1:])
        )

    def test_cute_flash_causal_degree2_seed_gates(self) -> None:
        cases = (
            {},
            {"dtype": torch.bfloat16, "standard_causal_output": True},
            {"head_dim": 128, "standard_causal_output": True},
            {"has_kv_tile_pruning": True, "standard_causal_output": True},
            {"requires_ws_overlap": True, "standard_causal_output": True},
            {"small_biased_candidate": True, "standard_causal_output": True},
            {
                "block_size_targets": (1, 64, 128),
                "standard_causal_output": True,
            },
        )
        for case in cases:
            subtest_case = {
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in case.items()
            }
            with self.subTest(case=subtest_case):
                case = dict(case)
                head_dim = int(case.pop("head_dim", 64))
                self.assertFalse(
                    any(
                        seed.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
                        for seed in flash_attention_seed_configs(
                            head_dim,
                            4096,
                            is_causal=True,
                            **case,
                        )
                    )
                )

    def test_cute_flash_causal_degree2_seed_is_in_full_initial_population(
        self,
    ) -> None:
        expected_params = {
            512: ("fa4", 15, 3, True),
            1024: ("fa4", 1, 14, False),
            2048: ("fa4", 14, 12, False),
            4096: ("helion", 14, 12, False),
            8192: ("helion", 14, 12, False),
        }
        for num_kv, (role_map, offset, offset0, epi_tma) in expected_params.items():
            with self.subTest(num_kv=num_kv):
                spec = ConfigSpec(backend=CuteBackend())
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=64,
                    num_kv=num_kv,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    dtype=torch.float16,
                    is_causal=True,
                    standard_causal_output=True,
                )
                spec.compiler_seed_configs = list(
                    flash_attention_seed_configs(
                        64,
                        num_kv,
                        is_causal=True,
                        standard_causal_output=True,
                    )
                )
                config_gen = ConfigGeneration(spec)
                raw_seed = next(
                    config
                    for config in spec.compiler_seed_configs
                    if config.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
                )
                roundtrip = config_gen.unflatten(config_gen.flatten(raw_seed))
                self.assertEqual(
                    config_gen.unflatten(config_gen.flatten(roundtrip)), roundtrip
                )
                expected = next(
                    config
                    for _flat, config in config_gen.seed_flat_config_pairs()
                    if config.config[FLASH_EXP2_PACKET_KEY] == "deg2_16x6"
                )
                self.assertEqual(expected, roundtrip)
                self.assertEqual(expected.config[FLASH_ROLE_MAP_KEY], role_map)
                self.assertEqual(expected.config[FLASH_E2E_OFFSET_KEY], offset)
                self.assertEqual(expected.config[FLASH_E2E_OFFSET0_KEY], offset0)
                self.assertEqual(expected.config[FLASH_EPI_TMA_KEY], epi_tma)
                profile = get_effort_profile("full").lfbo_pattern_search
                assert profile is not None
                search = PatternSearch.__new__(PatternSearch)
                search.config_gen = config_gen
                search.settings = Settings()
                search.log = MagicMock()
                search.initial_population_strategy = (
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE
                )
                search.best_available_pad_random = True
                search.initial_population = profile.initial_population
                search._best_available_seed_configs = []
                search._pinned_finalist_configs = set()
                search._autotune_seed_configs = lambda: ()
                search._find_similar_cached_configs = lambda _max_configs: []
                population = [
                    config_gen.unflatten(flat)
                    for flat in search._generate_initial_population_flat()
                ]
                seed_configs = [
                    config for _flat, config in config_gen.seed_flat_config_pairs()
                ]
                seed_count = len(seed_configs)
                self.assertEqual(population.count(expected), 1)
                self.assertIn(expected, population[:seed_count])
                self.assertEqual(population[:seed_count], seed_configs)
                self.assertLessEqual(len(population), profile.initial_population)
                self.assertEqual(len(set(population)), len(population))
                self.assertIn(expected, search._pinned_finalist_configs)

    @onlyBackends(["cute"])
    def test_cute_flash_attention_seed_heuristic(self) -> None:
        # The flash seed promotes block_sizes=[1, 128, 128] for a detected
        # fp16 dense/causal online-softmax attention kernels so the autotuner
        # exercises the fused tcgen05 flash path. fp32 attention is NOT eligible
        # and keeps the scalar path.
        @helion.kernel(backend="cute", static_shapes=True)
        def flash_attn(
            q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        @helion.kernel(backend="cute", static_shapes=True)
        def causal_flash_attn(
            q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
        ) -> torch.Tensor:
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    qk = torch.where(
                        tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                        qk,
                        float("-inf"),
                    )
                    m_ij_keepdim = torch.maximum(
                        m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
                    )
                    qk = qk - m_ij_keepdim
                    m_ij = m_ij_keepdim.squeeze(-1)
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        heuristic = CuteFlashAttentionHeuristic
        fp16_args = tuple(
            torch.randn(2, 32, 1024, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = flash_attn.bind(fp16_args)
        self.assertIn(
            CuteFlashAttentionHeuristic.name,
            bound.config_spec.autotuner_heuristics,
        )
        self.assertTrue(heuristic.is_eligible(bound.env, bound.host_function.device_ir))
        seed = heuristic.get_seed_config(bound.env, bound.host_function.device_ir)
        assert seed is not None
        self.assertEqual(seed.config["block_sizes"], [1, 128, 128])
        self.assertEqual(seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertNotIn(FLASH_E2E_OFFSET_KEY, seed.config)
        self.assertIn(seed, bound.config_spec.compiler_seed_configs)
        self.assertIsNone(bound.config_spec.compiler_default_config)
        self.assertEqual(
            bound.config_spec.compiler_seed_timeout_retry_repetitions,
            3,
        )

        bf16_args = tuple(
            torch.empty(1, 1, 49152, 64, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        )
        bf16_bound = flash_attn.bind(bf16_args)
        self.assertIs(bf16_bound.config_spec._cute_flash_dtype, torch.bfloat16)
        bf16_seed = heuristic.get_seed_config(
            bf16_bound.env, bf16_bound.host_function.device_ir
        )
        assert bf16_seed is not None
        self.assertEqual(bf16_seed.config[FLASH_RESCALE_THRESHOLD_KEY], 32.0)

        dense_2048_args = tuple(
            torch.empty(1, 1, 2048, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        dense_2048_bound = flash_attn.bind(dense_2048_args)
        dense_2048_seed = heuristic.get_seed_config(
            dense_2048_bound.env, dense_2048_bound.host_function.device_ir
        )
        assert dense_2048_seed is not None
        self.assertEqual(dense_2048_seed.config[FLASH_S_STAGE_KEY], 2)
        self.assertEqual(dense_2048_seed.config[FLASH_KV_STAGE_KEY], 3)
        self.assertTrue(dense_2048_seed.config[FLASH_PERSISTENT_KEY])
        self.assertEqual(dense_2048_seed.config[FLASH_PERSISTENT_CTAS_PER_SM_KEY], 1)
        self.assertEqual(dense_2048_seed.config[FLASH_KV_ORDER_KEY], "ascending")
        self.assertEqual(dense_2048_seed.config[FLASH_E2E_SCHEDULE_KEY], "8/2")
        self.assertEqual(dense_2048_seed.config[FLASH_E2E_OFFSET_KEY], 2)
        self.assertEqual(dense_2048_seed.config[FLASH_E2E_OFFSET0_KEY], 2)
        self.assertEqual(dense_2048_seed.config[FLASH_DISC_PIPE_KEY], 4)
        self.assertEqual(dense_2048_seed.config[FLASH_OTHER_REGS_KEY], 48)
        self.assertTrue(dense_2048_seed.config[FLASH_EPI_TMA_KEY])
        self.assertFalse(dense_2048_seed.config[FLASH_EPI_STG_KEY])
        self.assertEqual(dense_2048_seed.config[FLASH_CORR_TILE_SIZE_KEY], 16)
        self.assertEqual(dense_2048_seed.config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
        self.assertFalse(dense_2048_seed.config[FLASH_PACKED_REDUCE_KEY])
        self.assertEqual(dense_2048_seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertIn(
            dense_2048_seed, dense_2048_bound.config_spec.compiler_seed_configs
        )
        self.assertIsNone(dense_2048_bound.config_spec.compiler_default_config)

        dense_8192_args = tuple(
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        dense_8192_bound = flash_attn.bind(dense_8192_args)
        dense_8192_seed = heuristic.get_seed_config(
            dense_8192_bound.env, dense_8192_bound.host_function.device_ir
        )
        assert dense_8192_seed is not None
        self.assertEqual(dense_8192_seed.config[FLASH_E2E_SCHEDULE_KEY], "8/2")
        self.assertEqual(dense_8192_seed.config[FLASH_E2E_OFFSET_KEY], 2)
        self.assertEqual(dense_8192_seed.config[FLASH_E2E_OFFSET0_KEY], 2)
        self.assertEqual(dense_8192_seed.config[FLASH_DISC_PIPE_KEY], 3)
        self.assertEqual(dense_8192_seed.config[FLASH_PERSISTENT_CTAS_PER_SM_KEY], 1)
        self.assertEqual(dense_8192_seed.config[FLASH_KV_ORDER_KEY], "ascending")
        self.assertEqual(dense_8192_seed.config[FLASH_SOFTMAX_REGS_KEY], 184)
        self.assertEqual(dense_8192_seed.config[FLASH_OTHER_REGS_KEY], 48)
        self.assertEqual(dense_8192_seed.config[FLASH_CORR_REGS_KEY], 64)
        self.assertTrue(dense_8192_seed.config[FLASH_EPI_TMA_KEY])
        self.assertFalse(dense_8192_seed.config[FLASH_EPI_STG_KEY])
        self.assertEqual(dense_8192_seed.config[FLASH_CORR_TILE_SIZE_KEY], 16)
        self.assertEqual(dense_8192_seed.config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
        self.assertEqual(dense_8192_seed.config[FLASH_RESCALE_THRESHOLD_KEY], 8.0)
        self.assertTrue(dense_8192_seed.config[FLASH_PACKED_REDUCE_KEY])
        self.assertEqual(dense_8192_seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertNotIn(FLASH_CAUSAL_LPT_SWIZZLE_KEY, dense_8192_seed.config)
        dense_8192_gen = ConfigGeneration(dense_8192_bound.config_spec)
        dense_8192_roundtrip = dense_8192_gen.unflatten(
            dense_8192_gen.flatten(dense_8192_seed)
        )
        self.assertEqual(
            dense_8192_roundtrip.config[FLASH_SOFTMAX_REGS_KEY],
            184,
        )
        self.assertEqual(
            dense_8192_roundtrip.config[FLASH_CORR_REGS_KEY],
            64,
        )
        self.assertEqual(
            dense_8192_roundtrip.config[FLASH_OTHER_REGS_KEY],
            48,
        )

        causal_hd64_bound = causal_flash_attn.bind(fp16_args)
        causal_hd64_seed = heuristic.get_seed_config(
            causal_hd64_bound.env, causal_hd64_bound.host_function.device_ir
        )
        assert causal_hd64_seed is not None
        self.assertEqual(causal_hd64_seed.config["block_sizes"], [1, 128, 128])
        self.assertEqual(causal_hd64_seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertTrue(causal_hd64_seed.config[FLASH_PACKED_REDUCE_KEY])
        self.assertIn(
            causal_hd64_seed, causal_hd64_bound.config_spec.compiler_seed_configs
        )
        self.assertIsNone(causal_hd64_bound.config_spec.compiler_default_config)

        causal_8192_args = tuple(
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        causal_8192_bound = causal_flash_attn.bind(causal_8192_args)
        self.assertIn(
            CuteFlashAttentionCausalLptHeuristic.name,
            causal_8192_bound.config_spec.autotuner_heuristics,
        )
        causal_lpt_seed = CuteFlashAttentionCausalLptHeuristic.get_seed_config(
            causal_8192_bound.env, causal_8192_bound.host_function.device_ir
        )
        assert causal_lpt_seed is not None
        self.assertEqual(causal_lpt_seed.config[FLASH_CAUSAL_LPT_SWIZZLE_KEY], 8)
        self.assertEqual(
            causal_lpt_seed.config[FLASH_CAUSAL_KV_ORDER_KEY], "descending"
        )
        self.assertEqual(
            causal_lpt_seed.config[FLASH_MASKED_E2E_SCHEDULE_KEY], "inherit"
        )
        self.assertEqual(causal_lpt_seed.config[FLASH_ROLE_MAP_KEY], "helion")
        self.assertEqual(causal_lpt_seed.config[FLASH_E2E_OFFSET_KEY], 0)
        self.assertEqual(causal_lpt_seed.config[FLASH_SOFTMAX_REGS_KEY], 200)
        self.assertTrue(causal_lpt_seed.config[FLASH_PACKED_REDUCE_KEY])
        self.assertIn(
            causal_lpt_seed, causal_8192_bound.config_spec.compiler_seed_configs
        )
        causal_split_seed = flash_attention_seed_config(
            64, 64, is_causal=True, seed_kind="causal_split"
        )
        assert causal_split_seed is not None
        self.assertIn(
            causal_split_seed, causal_8192_bound.config_spec.compiler_seed_configs
        )
        self.assertIsNone(causal_8192_bound.config_spec.compiler_default_config)

        causal_65536_args = tuple(
            torch.empty(1, 1, 65536, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        causal_65536_bound = causal_flash_attn.bind(causal_65536_args)
        self.assertTrue(
            causal_65536_bound.config_spec._cute_flash_standard_causal_output
        )
        self.assertIn(
            CuteFlashAttentionCausalLptHeuristic.name,
            causal_65536_bound.config_spec.autotuner_heuristics,
        )
        causal_65536_seed = CuteFlashAttentionCausalLptHeuristic.get_seed_config(
            causal_65536_bound.env, causal_65536_bound.host_function.device_ir
        )
        assert causal_65536_seed is not None
        self.assertEqual(causal_65536_seed.config[FLASH_CAUSAL_LPT_SWIZZLE_KEY], 1)
        self.assertEqual(causal_65536_seed.config[FLASH_E2E_OFFSET_KEY], 9)
        self.assertEqual(causal_65536_seed.config[FLASH_SOFTMAX_REGS_KEY], 184)
        self.assertEqual(causal_65536_seed.config[FLASH_DISC_PIPE_KEY], 4)
        self.assertTrue(causal_65536_seed.config[FLASH_PACKED_REDUCE_KEY])
        self.assertIn(
            causal_65536_seed, causal_65536_bound.config_spec.compiler_seed_configs
        )
        self.assertFalse(
            any(
                seed.config.get(FLASH_EXP2_PACKET_KEY) == "hybrid_deg1_16x8"
                for seed in causal_65536_bound.config_spec.compiler_seed_configs
            )
        )
        causal_65536_degree2 = next(
            seed
            for seed in causal_65536_bound.config_spec.compiler_seed_configs
            if seed.config.get(FLASH_EXP2_PACKET_KEY) == "deg2_16x6"
        )
        self.assertEqual(causal_65536_degree2.config[FLASH_E2E_OFFSET_KEY], 15)
        self.assertEqual(causal_65536_degree2.config[FLASH_E2E_OFFSET0_KEY], 3)
        self.assertTrue(causal_65536_degree2.config[FLASH_EPI_TMA_KEY])
        with patch.object(
            PatternSearch,
            "_find_similar_cached_configs",
            return_value=[],
        ):
            quick_search = PatternSearch(
                causal_65536_bound,
                causal_65536_args,
                initial_population=30,
                initial_population_strategy=(
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE
                ),
                best_available_pad_random=False,
            )
            quick_configs = [
                quick_search.config_gen.unflatten(flat)
                for flat in quick_search._generate_initial_population_flat()
            ]
        causal_65536_degree2_normalized = quick_search.config_gen.unflatten(
            quick_search.config_gen.flatten(causal_65536_degree2)
        )
        expected_quick_configs = [
            config for _flat, config in quick_search.config_gen.seed_flat_config_pairs()
        ]
        default_config = quick_search.config_gen.unflatten(
            quick_search.config_gen.default_flat()
        )
        if default_config not in expected_quick_configs:
            expected_quick_configs.append(default_config)
        self.assertEqual(quick_configs, expected_quick_configs)
        self.assertIn(causal_65536_degree2_normalized, quick_configs)
        self.assertIsNone(causal_65536_bound.config_spec.compiler_default_config)
        causal_65536_gen = ConfigGeneration(causal_65536_bound.config_spec)
        causal_65536_default = causal_65536_gen.unflatten(
            causal_65536_gen.default_flat()
        )
        self.assertEqual(
            causal_65536_default.config[FLASH_SOFTMAX_REGS_KEY],
            184,
        )
        self.assertEqual(causal_65536_default.config[FLASH_E2E_OFFSET_KEY], 9)
        causal_65536_roundtrip = causal_65536_gen.unflatten(
            causal_65536_gen.flatten(causal_65536_seed)
        )
        self.assertEqual(
            causal_65536_roundtrip.config[FLASH_SOFTMAX_REGS_KEY],
            184,
        )

        causal_hd128_args = tuple(
            torch.randn(2, 32, 1024, 128, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        causal_hd128_bound = causal_flash_attn.bind(causal_hd128_args)
        causal_hd128_seed = heuristic.get_seed_config(
            causal_hd128_bound.env, causal_hd128_bound.host_function.device_ir
        )
        assert causal_hd128_seed is not None
        self.assertEqual(causal_hd128_seed.config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertEqual(causal_hd128_seed.config[FLASH_KV_STAGE_KEY], 2)

        # fp32 attention is not flash-eligible (the kernel hardcodes fp16).
        fp32_args = tuple(
            torch.randn(2, 32, 1024, 64, dtype=torch.float32, device=DEVICE)
            for _ in range(3)
        )
        fp32_bound = flash_attn.bind(fp32_args)
        self.assertFalse(
            heuristic.is_eligible(fp32_bound.env, fp32_bound.host_function.device_ir)
        )
        self.assertNotIn(
            CuteFlashAttentionHeuristic.name,
            fp32_bound.config_spec.autotuner_heuristics,
        )

        with patch_cute_mma_support(default_cute_mma_support(tcgen05_f16bf16=False)):
            unsupported_args = tuple(
                torch.randn(1, 16, 512, 64, dtype=torch.float16, device=DEVICE)
                for _ in range(3)
            )
            unsupported_bound = flash_attn.bind(unsupported_args)
            self.assertFalse(
                heuristic.is_eligible(
                    unsupported_bound.env,
                    unsupported_bound.host_function.device_ir,
                )
            )
            self.assertNotIn(
                CuteFlashAttentionHeuristic.name,
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

        # An ordinary cluster_m=2 candidate remains on the DEFAULT-layout path
        # so the autotuner can measure it independently from the FFI seed.
        default_cluster_m2_config = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
        )
        bound.config_spec.normalize(default_cluster_m2_config, _fix_invalid=True)
        self.assertIs(
            default_cluster_m2_config.config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY],
            False,
        )
        self.assertEqual(
            default_cluster_m2_config.config["block_sizes"],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, bk],
        )

        # The clustered scheduler still assigns physical cluster M/N to work-tile
        # coordinates 0/1. Autotuning repairs an alternate loop order to the
        # default; an explicit user config gets a clear validation error.
        clustered_alt_order = helion.Config.from_dict(default_cluster_m2_config.config)
        clustered_alt_order.config["loop_orders"] = [[1, 0]]
        bound.config_spec.normalize(clustered_alt_order, _fix_invalid=True)
        self.assertEqual(clustered_alt_order.config["tcgen05_cluster_m"], 2)
        self.assertEqual(clustered_alt_order.loop_orders, [[0, 1]])

        clustered_alt_order.config["loop_orders"] = [[1, 0]]
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            r"non-default loop_orders require tcgen05 cluster shape \(1, 1\)",
        ):
            bound.config_spec.normalize(clustered_alt_order)

        # An explicit FFI request still projects onto the generalized seed.
        requested_ffi_config = helion.Config(
            block_sizes=[256, 256, 128],
            indexing=["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            **{TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True},
        )
        bound.config_spec.normalize(requested_ffi_config, _fix_invalid=True)
        self.assertIs(
            requested_ffi_config.config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY], True
        )
        self.assertEqual(
            requested_ffi_config.config["block_sizes"],
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
                    "tcgen05_ab_stages > 3 is only supported",
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
            helion.exc.InvalidConfig, "tcgen05_ab_stages > 3 is only supported"
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
        spec = MagicMock()
        spec._tcgen05_matmul_block_fragments.return_value = (
            MagicMock(),
            MagicMock(),
            k_fragment,
        )
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
            # target_device_capability is memoized (is_hip / _is_hip pattern),
            # so torch.cuda.get_device_capability alone no longer reaches its
            # consumers. Patch each seam the bind path reads: the bound-kernel
            # cache key (runtime.kernel) and the ConfigSpec arch capability,
            # which CompileEnvironment captures onto config_spec at build time.
            patch(
                "helion.runtime.kernel.target_device_capability",
                return_value=(10, 0),
            ),
            patch(
                "helion._compiler.compile_environment.target_device_capability",
                return_value=(10, 0),
            ),
        ):
            bound = cute_matmul_bias_residual_gelu.bind(args)
        with (
            patch_cute_mma_support(),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch(
                "helion.runtime.kernel.target_device_capability",
                return_value=(9, 0),
            ),
            patch(
                "helion._compiler.compile_environment.target_device_capability",
                return_value=(9, 0),
            ),
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
        # The residual load belongs to a separate elementwise store, not the
        # matmul epilogue, so no aux-only matmul search axis is productive.
        self.assertFalse(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertNotIn(
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            {key for key, _count, _is_sequence in spec.flat_key_layout()},
        )
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
        # The partial rank-2 load is outside the accepted per-subtile epilogue
        # contract, so codegen cannot use the C-input or placement controls.
        self.assertFalse(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertNotIn(
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            {key for key, _count, _is_sequence in spec.flat_key_layout()},
        )
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
        # Reordered tile indices are rejected by the same chain analyzer used
        # at codegen, so they must not widen the aux search surface at bind.
        self.assertFalse(spec.cute_tcgen05_aux_kernel_detected)
        self.assertFalse(spec.cute_tcgen05_exact_shape_aux_kernel_detected)
        self.assertNotIn(
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            {key for key, _count, _is_sequence in spec.flat_key_layout()},
        )
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
        # emitted and remains distinct after normalization.
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
        seeded_configs = config_gen.random_population(3)
        self._assert_cute_tcgen05_cluster_m2_seeded(
            seeded_configs,
            expected_block_k=128,
            expected_indexing_length=3,
        )
        expected_seed_modes = {
            (Tcgen05LayoutStrategy.DEFAULT.value, False),
            (Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value, True),
        }
        seeded_modes = {
            (
                config.config.get("tcgen05_layout_strategy"),
                config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True,
            )
            for config in seeded_configs
        }
        self.assertTrue(expected_seed_modes.issubset(seeded_modes))

        acf_config_gen = bound.config_spec.create_config_generation(
            advanced_controls_files=["/tmp/helion-test.acf"]
        )
        acf_configs = acf_config_gen.random_population(2)
        # Future heuristics may add more compiler seeds; this test only
        # requires the CuTe cluster-m2 seed to be present.
        self.assertGreaterEqual(len(acf_configs), 2)
        acf_seed_configs = [
            config
            for config in acf_configs
            if config.config["advanced_controls_file"] == "/tmp/helion-test.acf"
        ]
        self.assertGreaterEqual(len(acf_seed_configs), 1)
        self.assertLessEqual(
            {config.config["advanced_controls_file"] for config in acf_configs},
            {"", "/tmp/helion-test.acf"},
        )
        self._assert_cute_tcgen05_cluster_m2_seeded(
            acf_seed_configs,
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
        # FROM_BEST_AVAILABLE must retain both compiler-owned strategies so the
        # autotuner measures the ordinary and direct-entry kernels.
        self.assertGreaterEqual(len(configs), 2)
        self._assert_cute_tcgen05_cluster_m2_seeded(
            configs,
            expected_block_k=128,
            expected_indexing_length=3,
        )
        best_available_modes = {
            (
                config.config.get("tcgen05_layout_strategy"),
                config.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True,
            )
            for config in configs
        }
        self.assertTrue(expected_seed_modes.issubset(best_available_modes))

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

    - rms_norm wide (rnumel=16384): the standard path
      (``TritonStandardReductionHeuristicSM90``) seeds a persistent reduction
      (``reduction_loops=[None]``) with the rnumel-ramp warp count.
    - kl_div wide (rnumel=131072): the Band-B (user-tiled) path
      (``TritonUserTiledReductionHeuristicSM90``) caps R_BLOCK by the accumulator footprint
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
        heuristic = TritonStandardReductionHeuristicSM90

        # Pin the kernel to the triton backend: autotuner_heuristics is keyed on
        # env.backend_name, and the tileir lane (where @onlyBackends(["triton"]) still
        # runs) has no registered heuristics, so an unpinned kernel yields [] and the
        # assertIn below fails. Pinning keeps backend_name "triton" on every lane.
        kernel = helion.kernel(rms_norm_fwd.fn, backend="triton")

        # Force the sm90 deep path so the test exercises the H100-tuned seed on any
        # runner (off-sm90 a different class fires: SM100 on B200, the narrow fallback
        # elsewhere).
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            bound = kernel.bind(args)

            # The reduction heuristic registered a single reduction descriptor and fired.
            kf = bound.config_spec.reduction_kernel_fact
            self.assertIsNotNone(kf)
            self.assertEqual(len(kf.reductions), 1)
            self.assertEqual(kf.reductions[0].size_hint, n)
            # rms_norm has no separate apply/normalize loop (its apply is over the full
            # row in the reduction scope), so no reduce-then-apply tile is captured.
            self.assertEqual(kf.non_reduction_loop_block_ids, ())
            self.assertIn(
                TritonStandardReductionHeuristicSM90.name,
                bound.config_spec.autotuner_heuristics,
            )
            self.assertTrue(
                heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            )

            # Exactly one compiler seed, and it is the *persistent* standard config.
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
        heuristic = TritonUserTiledReductionHeuristicSM90

        # Pin the kernel to the triton backend: autotuner_heuristics is keyed on
        # env.backend_name, and the tileir lane (where @onlyBackends(["triton"]) still
        # runs) has no registered heuristics, so an unpinned kernel yields [] and the
        # assertIn below fails. Pinning keeps backend_name "triton" on every lane.
        kernel = helion.kernel(kl_div_forward.fn, backend="triton")

        # Force the sm90 deep path so the Band-B seed is exercised on any runner
        # (off-sm90 a different class fires: SM100 on B200; the narrow fallback covers
        # only the standard track, so user-tiled simply does not seed elsewhere).
        with patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE):
            bound = kernel.bind(args)

            # Single reduction descriptor carrying a 2D [M, R] tile -> Band B.
            kf = bound.config_spec.reduction_kernel_fact
            self.assertIsNotNone(kf)
            self.assertEqual(len(kf.reductions), 1)
            fact = kf.reductions[0]
            self.assertEqual(fact.size_hint, n)
            self.assertGreaterEqual(fact.carried_2d_count, 1)
            self.assertEqual(kf.non_reduction_loop_block_ids, ())
            self.assertIn(
                TritonUserTiledReductionHeuristicSM90.name,
                bound.config_spec.autotuner_heuristics,
            )
            self.assertTrue(
                heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            )

            # Exactly one seed; R_BLOCK is capped (NOT full-N persistent) by the ONE budget
            # allocator, and the grid (M) axis sits at its floor of 1.
            seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        self.assertEqual(len(seeds), 1)
        seed = seeds[0].config
        # The budget allocator sizes the carried [M_BLOCK, R_BLOCK] tile against ONE group budget
        # (num_live × itemsize footprint vs the CARRIED budget), NOT a bespoke carried byte cap.
        # The carried accumulator is live the whole loop with body_live_tiles copies, so the budget
        # depletes to R_BLOCK = pow2(CARRIED_PERSIST_MAX_BYTES / (num_live × itemsize)). A carried
        # reduction holds its [M, R] tile resident across the whole loop (not streamed-then-released),
        # so it sizes against the TIGHTER CARRIED_PERSIST_MAX_BYTES (= ROW_PERSIST // 2 = 122880), a
        # single budget CONSTANT — the footprint FORMULA is the same uniform num_live × ∏(working
        # tile) as every other kernel (no buffer-count multiplier: body_live_tiles already counts the
        # carried buffers). For kl_div (body_live_tiles == 6, fp32) that is pow2(122880 / (6 × 4)) =
        # pow2(5120) = 4096 — capped well below next_pow2(131072) and M floored to 1 (budget spent).
        # 4096 is the MEASURED optimum (~+2% vs the old 8192). Floor-vs-resident falls out of
        # depletion: no carried recognizer, no separate CARRIED_TILE_MAX_BYTES.
        r_block = seed["block_sizes"][0]
        self.assertEqual(seed["block_sizes"], [4096, 1])
        self.assertLess(r_block, n)
        # rnumel 131072 > the 16384 warps-32 breakpoint -> 32 warps.
        self.assertEqual(seed["num_warps"], 32)
        self.assertEqual(seed["num_stages"], 1)
        # The carried-tile path must NOT use the standard reduction_loops knob.
        self.assertNotIn("reduction_loops", seed)

    @onlyBackends(["triton"])
    @skipIfRefEager("Compiler reduction facts are not collected in ref eager mode")
    def test_t1_reduction_then_normalize_loop_widens_tile(self) -> None:
        # A standard rollable reduction (hl.sum over the full inner dim) IMMEDIATELY
        # followed by a SEPARATE non-reduction hl.tile(n) loop that normalizes the row.
        # No example kernel has this shape, so it pins the standard
        # non-reduction-loop path:
        # - the fact captures the normalize loop as non_reduction_loop_block_ids and
        #   keeps m_block_ids grid-only (the normalize tile is NOT a row axis), and
        # - the seed emits a full-length block_sizes with that tile widened (without it
        #   the standard seed would emit a wrong-length [grid_floor] and crash) — for
        #   BOTH the persistent and the looped (wide-N) reduction cases.
        # NOTE: this standard+normalize seed is NOT performance-validated (no oracle);
        # it is only a seed (worse tile => more autotuning, never wrong results), so the
        # test asserts only that the emitted config is well-formed in both regimes.
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
            # Force the sm90 deep path so the standard+normalize seed is exercised on
            # any runner (off-sm90 the narrow fallback fires instead, which does not
            # widen the normalize tile).
            with patch(
                "helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE
            ):
                bound = t1_then_normalize.bind(
                    (torch.randn([m, n], device=DEVICE, dtype=torch.float32),)
                )
                # One reduction descriptor: reduction axis + grid-only row axis + the
                # normalize loop captured as a non-reduction loop tile (NOT a row axis).
                kf = bound.config_spec.reduction_kernel_fact
                self.assertIsNotNone(kf)
                self.assertEqual(len(kf.reductions), 1)
                fact = kf.reductions[0]
                self.assertEqual(fact.size_hint, n)
                self.assertEqual(kf.grid_axis_block_ids, (0,))
                self.assertEqual(len(kf.non_reduction_loop_block_ids), 1)
                self.assertNotIn(fact.block_id, kf.non_reduction_loop_block_ids)
                self.assertEqual(fact.carried_2d_count, 0)
                self.assertIn(
                    TritonStandardReductionHeuristicSM90.name,
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
                kf.non_reduction_loop_block_ids[0]
            )
            self.assertGreater(seed["block_sizes"][norm_idx], 1)
            # Persistent (narrow row) -> reduction_loops=[None]; looped (wide row past
            # the byte cap) -> reduction_loops=[LOOPED_CHUNK].
            if expect_looped:
                self.assertEqual(
                    seed["reduction_loops"],
                    [TritonStandardReductionHeuristicSM90.LOOPED_CHUNK],
                )
                # At m_block==1 the normalize tile is clamped to the SAME ÷M_BLOCK
                # register-resident footprint as the reduction tile, NOT left at
                # next_pow2(N). With m_block==1 / fp32 the budget is prev_pow2(
                # ROW_PERSIST_MAX_BYTES // (1 * 4)) == 32768, which is < next_pow2(131072).
                # This is the only test cell that exercises the cap at M_BLOCK==1, so pin
                # the value (a > 1 check would also pass on the OLD M_BLOCK>1-gated cap that
                # left this tile uncapped).
                from helion._utils import prev_power_of_2

                expected_norm = prev_power_of_2(
                    TritonStandardReductionHeuristicSM90.ROW_PERSIST_MAX_BYTES
                    // (1 * 4)
                )
                self.assertEqual(expected_norm, 32768)
                self.assertEqual(seed["block_sizes"][norm_idx], expected_norm)
            else:
                self.assertEqual(seed["reduction_loops"], [None])
            # The emitted seed must round-trip through normalize() without raising.
            bound.config_spec.normalize(dict(seed))

        # 1024x4096: 16 KB/row < 240 KB byte cap -> persistent.
        check(1024, 4096, expect_looped=False)
        # 1024x131072: 512 KB/row > 240 KB byte cap -> looped (the case the old guard
        # wrongly declined into a wrong-length crash; now emits a widened looped seed).
        check(1024, 131072, expect_looped=True)

    def test_independent_loops_not_floored_by_budget_allocator(self) -> None:
        # The ONE budget allocator (``size_reduction_tiles``) sizes a USER_TILE reduction
        # axis AND a co-occurring non-reduction (normalize) loop / secondary reducing axis
        # so neither FLOORS to 1 (the [..., 1] serialization catastrophe). No example
        # kernel has a dynamic-extent non-reduction loop, so this pins the behavior on a
        # constructed spec (bare-spec, no active env — the allocator reads stored hints).
        from helion.autotuner.config_spec import BlockSizeSpec

        H = TritonUserTiledReductionHeuristicSM90
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
            # USER_TILE reduction (rdim is a block_sizes entry), grid row block 0, and a
            # non-reduction normalize loop ``norm_bid`` captured on the kernel fact.
            desc = ReductionDescriptor(
                category=ReductionCategory.USER_TILE,
                block_id=reduction_bid,
                graph_id=0,
                size_hint=size_hint,
                itemsize=4,
                input_load_itemsize=4,
                num_load=1,
            )
            spec.reduction_kernel_fact = ReductionKernelFact(
                reductions=(desc,),
                coresidency_groups=(
                    CoResidencyGroup(graph_id=0, descriptor_indices=(0,)),
                ),
                non_reduction_loop_block_ids=(norm_bid,),
                grid_axis_block_ids=(0,),
            )
            return spec

        def pd(reduction_bid: int) -> ReductionDescriptor:
            return ReductionDescriptor(
                category=ReductionCategory.USER_TILE,
                block_id=reduction_bid,
                graph_id=0,
                size_hint=size_hint,
                itemsize=4,
                input_load_itemsize=4,
                num_load=1,
            )

        # The allocator runs without an active CompileEnvironment (it reads stored hints);
        # device_ir is only consulted for materialized features (none here), so a MagicMock
        # whose attribute access yields empty iterables is fine.
        from unittest.mock import MagicMock

        # reduce-then-apply: the reduction axis is sized to its full extent (persistent /
        # budget-admitted) and the normalize loop is sized to its own extent — NOT 1.
        spec = spec_with(reduction_bid=1, norm_bid=2)
        device_ir = MagicMock()
        device_ir.grid_block_ids = []
        with (
            patch("helion._hardware.get_hardware_info", return_value=HOPPER_HARDWARE),
            patch("helion.runtime.get_num_sm", return_value=132),
        ):
            alloc = H.size_reduction_tiles(MagicMock(), spec, device_ir, pd(1))
        red_idx = spec.block_sizes.block_id_to_index(1)
        norm_idx = spec.block_sizes.block_id_to_index(2)
        self.assertEqual(alloc.block_sizes[red_idx], 4096)  # rdim sized to extent
        self.assertEqual(
            alloc.block_sizes[norm_idx], 4096
        )  # normalize loop NOT floored
        self.assertNotEqual(alloc.block_sizes[norm_idx], 1)
        # grid (M) row axis floored (no widen headroom is required; it just must be valid).
        self.assertGreaterEqual(alloc.block_sizes[0], 1)
