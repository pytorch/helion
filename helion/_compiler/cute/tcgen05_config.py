from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import torch

from ...autotuner.config_fragment import BooleanFragment
from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_fragment import IntegerFragment
from ...exc import InvalidConfig
from ...runtime.config import Config
from .strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from .strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from .strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY
from .strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from .strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from .strategies import TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES
from .strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from .strategies import TCGEN05_STRATEGY_CONFIG_KEY
from .strategies import TCGEN05_STRATEGY_CONFIG_KEYS
from .strategies import TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY
from .strategies import TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_MMA_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY
from .strategies import TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY
from .strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from .strategies import Tcgen05LayoutStrategy
from .strategies import Tcgen05PersistenceModel
from .strategies import Tcgen05Strategy
from .strategies import derive_persistence_model_from_pid_type
from .strategies import layout_overrides_from_config
from .strategies import validate_tcgen05_strategy_invariants
from .strategies import warp_spec_from_config
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODES
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODES
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODES
from .tcgen05_constants import TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN
from .tcgen05_constants import TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODES
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODES
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from .tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_C_STORE_MODES
from .tcgen05_constants import TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides

if TYPE_CHECKING:
    from ...autotuner.block_id_sequence import BlockIdSequence
    from ...autotuner.config_fragment import BlockSizeFragment
    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import PidTypeLiteral


class Tcgen05ClusterM2SearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_cluster_m=2`` is validated."""

    static_k: int
    max_k_tiles: int
    allow_edge_k_tail_family: bool = False


class Tcgen05AbStagesThreeSearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_ab_stages=3`` is admitted.

    The 3-stage AB pipeline is only safe to search when its larger SMEM
    allocation fits after reserving space for CuTe runtime/barrier scratch.
    """

    dtype_bytes: int
    per_cta_smem_budget_bytes: int


CUTE_TCGEN05_TUNABLE_KEYS: tuple[str, ...] = (
    "tcgen05_cluster_m",
    "tcgen05_cluster_n",
    "tcgen05_ab_stages",
    "tcgen05_acc_stages",
    "tcgen05_c_stages",
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_STORE_MODE_CONFIG_KEY,
    "tcgen05_num_epi_warps",
    TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
)
CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
        TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
        TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
        TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
        TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_LARGE_BN_PROOF_CONFIG_KEY,
        TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
    }
)
CUTE_TCGEN05_STRATEGY_CONFIG_KEYS: frozenset[str] = frozenset(
    TCGEN05_STRATEGY_CONFIG_KEYS
)


class CuteTcgen05Config:
    """CuTe-owned tcgen05 ConfigSpec state and normalization hooks."""

    def __init__(self, config_spec: ConfigSpec) -> None:
        self.config_spec = config_spec
        self.search_enabled: bool = False
        self.aux_kernel_detected: bool = False
        self.cluster_m_search_choices: tuple[int, ...] | None = None
        self.cluster_m2_search_constraints: Tcgen05ClusterM2SearchConstraints | None = (
            None
        )
        self.ab_stages_three_search_constraints: (
            Tcgen05AbStagesThreeSearchConstraints | None
        ) = None
        self.num_epi_warps_search_choices: tuple[int, ...] | None = None
        self.num_epi_warps_validation_choices: tuple[int, ...] | None = None

    @property
    def allowed_pid_types(self) -> tuple[PidTypeLiteral, ...]:
        return self.config_spec.allowed_pid_types

    @allowed_pid_types.setter
    def allowed_pid_types(self, value: tuple[PidTypeLiteral, ...]) -> None:
        self.config_spec.allowed_pid_types = value

    @staticmethod
    def _validate_optional_fragment_value(
        name: str, fragment: ConfigSpecFragment, value: object
    ) -> object:
        if isinstance(fragment, BooleanFragment):
            if type(value) is not bool:
                raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
            return value
        if isinstance(fragment, EnumFragment):
            if value not in fragment.choices:
                raise InvalidConfig(
                    f"{name} must be one of {fragment.choices!r}, got {value!r}"
                )
            return value
        if isinstance(fragment, IntegerFragment):
            if type(value) is not int:
                raise InvalidConfig(f"{name} must be an integer, got {value!r}")
            if value < fragment.low or value > fragment.high:
                raise InvalidConfig(
                    f"{name} must be in [{fragment.low}, {fragment.high}], got {value!r}"
                )
            return value
        raise InvalidConfig(f"Unsupported optional fragment type for {name}")

    def restrict_cluster_m_search(self, choices: tuple[int, ...]) -> None:
        assert choices, "tcgen05_cluster_m search must allow at least one value"
        self.cluster_m_search_choices = choices
        if 2 not in choices:
            self.cluster_m2_search_constraints = None

    def allow_cluster_m2_search(
        self,
        *,
        static_k: int,
        max_k_tiles: int = TCGEN05_TWO_CTA_MAX_K_TILES,
        allow_edge_k_tail_family: bool = False,
    ) -> None:
        assert static_k > 0, "static_k is required for cluster_m=2 K-cap checks"
        assert max_k_tiles > 0, "cluster_m=2 max K tiles must be positive"
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
        )
        self.restrict_cluster_m_search((1, 2))

    @staticmethod
    def cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        if bk <= 0:
            return False
        if constraints.allow_edge_k_tail_family:
            return (
                bk == TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
                and constraints.static_k > bk
                and constraints.static_k % bk != 0
                and (constraints.static_k + bk - 1) // bk <= constraints.max_k_tiles
            )
        if constraints.static_k % bk == 0:
            return constraints.static_k // bk <= constraints.max_k_tiles
        return False

    def _c_input_seed_config(self) -> Config | None:
        if not self.aux_kernel_detected:
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None

        bm_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[0]._fragment(self.config_spec),
        )
        bn_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[1]._fragment(self.config_spec),
        )
        bk_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[2]._fragment(self.config_spec),
        )
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        m_tile_reachable = (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            or (edge_k_tail_family and bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M)
        )
        n_tile_reachable = (
            bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
            or (edge_k_tail_family and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N)
        )
        if not (m_tile_reachable and n_tile_reachable):
            return None

        if edge_k_tail_family:
            bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
            if not (
                bk_fragment.low <= bk <= bk_fragment.high
                and self.cluster_m2_bk_is_valid(bk, constraints)
            ):
                return None
        else:
            bk = bk_fragment.high
            while bk >= bk_fragment.low:
                if self.cluster_m2_bk_is_valid(bk, constraints):
                    break
                bk //= 2
            else:
                return None

        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": 2,
            TCGEN05_STRATEGY_CONFIG_KEY: (
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            ),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 1,
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: 1,
        }
        if edge_k_tail_family:
            seed_config.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
            seed_config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            )
            seed_config["indexing"] = [
                "tensor_descriptor"
            ] * self.config_spec.indexing.length
        else:
            seed_config["l2_groupings"] = [1]
            if self.config_spec.indexing.length == 3:
                seed_config["indexing"] = [
                    "tensor_descriptor",
                    "tensor_descriptor",
                    "tensor_descriptor",
                ]
        return Config(**seed_config)

    def autotune_seed_configs(self) -> list[Config]:
        seeds: list[Config] = []
        c_input_seed = self._c_input_seed_config()
        if c_input_seed is not None:
            seeds.append(c_input_seed)
        return seeds

    def _fix_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        if not (self.search_enabled and config.get("tcgen05_cluster_m") == 2):
            return
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            config["tcgen05_cluster_m"] = 1
            return
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            config["tcgen05_cluster_m"] = 1
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 3:
            config["tcgen05_cluster_m"] = 1
            return
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        if edge_k_tail_family:
            block_sizes[2] = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
        bk = block_sizes[2]
        if not isinstance(bk, int) or isinstance(bk, bool):
            config["tcgen05_cluster_m"] = 1
            return
        if not self.cluster_m2_bk_is_valid(bk, constraints):
            config["tcgen05_cluster_m"] = 1
            return
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
        block_sizes[1] = TCGEN05_TWO_CTA_BLOCK_N
        if edge_k_tail_family:
            # This family is pinned to the measured production tuple after
            # search projection. The placement keys remain available for
            # non-edge diagnostic/search paths, but edge+K-tail candidates do
            # not explore partially mutated placement variants.
            config.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
            if self.aux_kernel_detected and self._has_any_matmul_fact_edge_tile(config):
                self._set_aux_edge_cluster_m2_prefix(config)

    def allow_ab_stages_three_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        assert dtype_bytes > 0, "dtype_bytes must be positive"
        if len(self.config_spec.block_sizes) != 3:
            self.ab_stages_three_search_constraints = None
            return
        budget_bytes = self.per_cta_ab_smem_budget_bytes(device)
        if budget_bytes <= 0:
            self.ab_stages_three_search_constraints = None
            return
        self.ab_stages_three_search_constraints = Tcgen05AbStagesThreeSearchConstraints(
            dtype_bytes=dtype_bytes,
            per_cta_smem_budget_bytes=budget_bytes,
        )

    @staticmethod
    def per_cta_ab_smem_budget_bytes(device: torch.device) -> int:
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(device)
        optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
        device_cap = max(props.shared_memory_per_block, optin_shared)
        if device_cap < TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN:
            return 0
        # Keep a fixed headroom reservation: CuTe's raw opt-in limit does not
        # include every barrier/runtime byte the 3-stage AB pipeline needs.
        return device_cap - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES

    def ab_stages_three_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
    ) -> bool:
        constraints = self.ab_stages_three_search_constraints
        if constraints is None:
            return False
        if cluster_m not in (1, 2):
            return False
        if bm <= 0 or bn <= 0 or bk <= 0:
            return False
        bytes_per_cta = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=3,
            cluster_m=cluster_m,
        )
        return bytes_per_cta <= constraints.per_cta_smem_budget_bytes

    def _fix_ab_stages_three_search_config(self, config: dict[str, object]) -> None:
        if self.ab_stages_three_search_constraints is None:
            return
        if not self.search_enabled:
            return
        if config.get("tcgen05_ab_stages") != 3:
            return
        block_sizes = cast("list[int]", config["block_sizes"])
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if not self.ab_stages_three_fits(
            bm=block_sizes[0],
            bn=block_sizes[1],
            bk=block_sizes[2],
            cluster_m=cluster_m,
        ):
            config["tcgen05_ab_stages"] = 2

    def _fix_with_scheduler_search_config(self, config: dict[str, object]) -> None:
        if not (self.search_enabled and self.aux_kernel_detected):
            return
        strategy = config.get(TCGEN05_STRATEGY_CONFIG_KEY)
        scheduler_warps = config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY)
        c_input_warps = config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY)
        ab_stages = config.get("tcgen05_ab_stages")
        if strategy == Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value:
            if scheduler_warps != 0:
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
            if c_input_warps != 0:
                config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        elif strategy == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value:
            if scheduler_warps != 1:
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        if ab_stages == 3 and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1:
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0

    def _matmul_fact_has_edge_tile(
        self, config: dict[str, object], *, fact_index: int
    ) -> bool:
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return False
        fact = self.config_spec.matmul_facts[fact_index]
        for static_size, block_id in (
            (fact.static_m, fact.m_block_id),
            (fact.static_n, fact.n_block_id),
            (fact.static_k, fact.k_block_id),
        ):
            if static_size is None or block_id is None:
                continue
            try:
                block_idx = self.config_spec.block_sizes.block_id_to_index(block_id)
            except KeyError:
                continue
            if block_idx >= len(block_sizes):
                continue
            block_size = block_sizes[block_idx]
            if (
                not isinstance(block_size, int)
                or isinstance(block_size, bool)
                or block_size <= 0
            ):
                continue
            if static_size % block_size != 0:
                return True
        return False

    def _has_any_matmul_fact_edge_tile(self, config: dict[str, object]) -> bool:
        return any(
            self._matmul_fact_has_edge_tile(config, fact_index=i)
            for i in range(len(self.config_spec.matmul_facts))
        )

    def _fix_aux_edge_search_config(self, config: dict[str, object]) -> None:
        if not (self.search_enabled and self.aux_kernel_detected):
            return
        if not self._has_any_matmul_fact_edge_tile(config):
            return

        if self._is_validated_cluster_m2_edge_search_candidate(config):
            self._set_aux_edge_cluster_m2_prefix(config)
            return

        self._set_aux_edge_monolithic_prefix(config)
        config["tcgen05_acc_stages"] = 2
        config["tcgen05_c_stages"] = 4
        config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
        if isinstance(config.get("l2_groupings"), list):
            config["l2_groupings"] = [1]
        if isinstance(config.get("indexing"), list):
            indexing = cast("list[object]", config["indexing"])
            for i in range(len(indexing)):
                indexing[i] = "pointer"
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return
        for fact in self.config_spec.matmul_facts:
            if fact.static_m is not None and fact.m_block_id is not None:
                try:
                    m_idx = self.config_spec.block_sizes.block_id_to_index(
                        fact.m_block_id
                    )
                except KeyError:
                    m_idx = -1
                if 0 <= m_idx < len(block_sizes):
                    bm = block_sizes[m_idx]
                    if (
                        isinstance(bm, int)
                        and not isinstance(bm, bool)
                        and bm > 128
                        and fact.static_m % bm != 0
                    ):
                        block_sizes[m_idx] = 128
            # K-only bk=128 tails keep the default A SMEM atom; output-edge
            # bk=128 fallback is handled in cute_mma by forcing A INTER.
            # Both cases have runtime coverage in test_cute_lowerings.

    @staticmethod
    def _set_aux_edge_monolithic_prefix(config: dict[str, object]) -> None:
        config[TCGEN05_STRATEGY_CONFIG_KEY] = (
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
        )
        config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
        config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        if config.get("tcgen05_ab_stages") == 1:
            config["tcgen05_ab_stages"] = 2
        config.pop("epilogue_subtile", None)

    @staticmethod
    def _set_aux_edge_cluster_m2_prefix(config: dict[str, object]) -> None:
        if (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
            config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            )
        else:
            CuteTcgen05Config._set_aux_edge_monolithic_prefix(config)
            return
        if config.get("tcgen05_ab_stages") == 1:
            config["tcgen05_ab_stages"] = 2
        config.pop("epilogue_subtile", None)

    def _is_validated_cluster_m2_edge_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if config.get("tcgen05_cluster_m") != 2:
            return False
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            return False
        return constraints.allow_edge_k_tail_family

    def _fix_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        if not (
            self.search_enabled
            and config.get("tcgen05_cluster_m", 1) == 1
            and config.get("pid_type")
            in {"persistent_blocked", "persistent_interleaved"}
        ):
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or not block_sizes:
            return
        constraints = self.cluster_m2_search_constraints
        if constraints is not None and constraints.allow_edge_k_tail_family:
            # persistent_interleaved stays in the flat enum so cluster_m=2
            # edge-family samples can encode; cluster_m=1 samples from the
            # same surface must use the validated flat edge fallback.
            config["pid_type"] = "flat"
            return
        bm = block_sizes[0]
        if isinstance(bm, int) and not isinstance(bm, bool):
            block_sizes[0] = min(bm, TCGEN05_ONE_CTA_MAX_BLOCK_M)

    def restrict_num_epi_warps_search(self, choices: tuple[int, ...]) -> None:
        assert choices, "tcgen05_num_epi_warps search must allow at least one value"
        self.num_epi_warps_search_choices = choices

    def restrict_num_epi_warps_validation(self, choices: tuple[int, ...]) -> None:
        assert choices, "tcgen05_num_epi_warps validation must allow at least one value"
        self.num_epi_warps_validation_choices = choices

    def narrow_autotune_to_validated_configs(
        self,
        *,
        allow_persistent_pid_types: bool = False,
        allow_cluster_m2_search: bool = False,
        cluster_m2_static_k: int | None = None,
        allow_cluster_m2_edge_k_tail_family: bool = False,
        ab_stages_three_dtype_bytes: int | None = None,
        ab_stages_three_device: torch.device | None = None,
    ) -> None:
        # Keep the default tcgen05 surface to combinations with runtime
        # coverage. Some unvalidated combinations fail loudly at CuTe
        # construction/launch, while diagnostic pipeline modes can compile and
        # intentionally produce wrong output.
        if allow_cluster_m2_edge_k_tail_family:
            assert allow_cluster_m2_search, (
                "cluster_m=2 edge/K-tail admission requires cluster_m=2 search"
            )
        cluster_m2_static_k_int: int | None = None
        if allow_cluster_m2_search:
            assert allow_persistent_pid_types or allow_cluster_m2_edge_k_tail_family, (
                "cluster_m=2 search requires persistent pid types or the "
                "validated output-edge + K-tail admission"
            )
            if cluster_m2_static_k is None:
                raise AssertionError("cluster_m=2 search requires a static K extent")
            cluster_m2_static_k_int = cluster_m2_static_k
        if allow_cluster_m2_edge_k_tail_family and (
            TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types
        ):
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )
        if not allow_persistent_pid_types:
            self.config_spec.disallow_pid_type("persistent_blocked")
            if not allow_cluster_m2_edge_k_tail_family:
                self.config_spec.disallow_pid_type("persistent_interleaved")
        if allow_cluster_m2_search:
            assert cluster_m2_static_k_int is not None
            self.allow_cluster_m2_search(
                static_k=cluster_m2_static_k_int,
                allow_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
            )
        else:
            self.restrict_cluster_m_search((1,))
        self.restrict_num_epi_warps_search((4,))
        self.restrict_num_epi_warps_validation((4,))
        if ab_stages_three_dtype_bytes is not None:
            assert ab_stages_three_device is not None, (
                "ab_stages_three_dtype_bytes requires ab_stages_three_device "
                "so the SMEM-budget gate consults the operand's device, not "
                "the host's current CUDA device"
            )
            self.allow_ab_stages_three_search(
                dtype_bytes=ab_stages_three_dtype_bytes,
                device=ab_stages_three_device,
            )

    def optional_fragments(
        self, *, for_search: bool = False
    ) -> dict[str, ConfigSpecFragment]:
        if for_search and self.cluster_m_search_choices is not None:
            cluster_m_choices = self.cluster_m_search_choices
        else:
            cluster_m_choices = (1, 2)
        cluster_n_choices: tuple[int, ...] = (1,) if for_search else (1, 2)
        if for_search and self.num_epi_warps_search_choices is not None:
            num_epi_warps_fragment: ConfigSpecFragment = EnumFragment(
                self.num_epi_warps_search_choices
            )
        elif not for_search and self.num_epi_warps_validation_choices is not None:
            num_epi_warps_fragment = EnumFragment(self.num_epi_warps_validation_choices)
        else:
            num_epi_warps_fragment = IntegerFragment(1, 4, 4)
        if not for_search or self.ab_stages_three_search_constraints is not None:
            ab_stages_max = 3
        else:
            ab_stages_max = 2
        if for_search:
            l2_swizzle_choices = tuple(
                v for v in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if v <= 8
            )
        else:
            l2_swizzle_choices = TCGEN05_LEGAL_L2_SWIZZLE_SIZES
        return {
            "tcgen05_cluster_m": EnumFragment(cluster_m_choices),
            "tcgen05_cluster_n": EnumFragment(cluster_n_choices),
            "tcgen05_ab_stages": IntegerFragment(1, ab_stages_max, 2),
            "tcgen05_acc_stages": IntegerFragment(1, 2, 2),
            "tcgen05_c_stages": EnumFragment((2, 4)),
            "tcgen05_num_epi_warps": num_epi_warps_fragment,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: EnumFragment(l2_swizzle_choices),
        }

    def strategy_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        # Aux kernels are the only current trigger for scheduler/c_input warp
        # search. The surface is derived from aux_kernel_detected so repeated
        # detection or repeated fragment construction stays idempotent.
        if self.aux_kernel_detected:
            strategy_choices: tuple[str, ...] = (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            )
            scheduler_warps_choices: tuple[int, ...] = (0, 1)
            c_input_warps_choices: tuple[int, ...] = (0, 1)
        else:
            strategy_choices = (Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,)
            scheduler_warps_choices = (0,)
            c_input_warps_choices = (0,)
        return {
            TCGEN05_STRATEGY_CONFIG_KEY: EnumFragment(strategy_choices),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: EnumFragment(
                (Tcgen05LayoutStrategy.DEFAULT.value,)
            ),
            TCGEN05_WARP_SPEC_MMA_WARPS_KEY: EnumFragment((1,)),
            TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY: EnumFragment((1,)),
            TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY: EnumFragment((0,)),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: EnumFragment(
                scheduler_warps_choices
            ),
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: EnumFragment(c_input_warps_choices),
            TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY: EnumFragment(
                (ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC.register_split[0],)
            ),
            TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY: EnumFragment(
                (ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC.register_split[1],)
            ),
        }

    def strategy_validation_fragments(self) -> dict[str, ConfigSpecFragment]:
        fragments = self.strategy_autotune_fragments()
        fragments[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = EnumFragment(
            (
                Tcgen05PersistenceModel.NON_PERSISTENT.value,
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
                Tcgen05PersistenceModel.CLC_PERSISTENT.value,
            )
        )
        fragments[TCGEN05_STRATEGY_CONFIG_KEY] = EnumFragment(
            (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            )
        )
        fragments[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = EnumFragment((0, 1))
        fragments[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = EnumFragment((0, 1))
        return fragments

    @staticmethod
    def strategy_field_default(key: str, *, pid_type: object = None) -> object:
        if key == TCGEN05_STRATEGY_CONFIG_KEY:
            return Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
        if key == TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY:
            return derive_persistence_model_from_pid_type(pid_type).value
        if key == TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY:
            return Tcgen05LayoutStrategy.DEFAULT.value
        if key in TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY:
            return TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY[key]
        raise KeyError(f"Unknown tcgen05 strategy field: {key!r}")

    def validate_strategy_invariants(
        self,
        config: dict[str, object],
        *,
        fix_invalid: bool,
    ) -> None:
        strategy = Tcgen05Strategy(config[TCGEN05_STRATEGY_CONFIG_KEY])
        persistence_model = Tcgen05PersistenceModel(
            config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
        )
        layout_strategy = Tcgen05LayoutStrategy(
            config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY]
        )
        warp_spec = warp_spec_from_config(config)
        layout_overrides = layout_overrides_from_config(config)

        cluster_m_raw = config.get("tcgen05_cluster_m", 1)
        cluster_m = int(cluster_m_raw) if isinstance(cluster_m_raw, int) else 1
        cluster_n_raw = config.get("tcgen05_cluster_n", 1)
        cluster_n = int(cluster_n_raw) if isinstance(cluster_n_raw, int) else 1
        arch_major: int | None = None
        if torch.cuda.is_available():
            arch_major = torch.cuda.get_device_capability(torch.cuda.current_device())[
                0
            ]
        errors = validate_tcgen05_strategy_invariants(
            strategy=strategy,
            persistence_model=persistence_model,
            layout_strategy=layout_strategy,
            warp_spec=warp_spec,
            layout_overrides=layout_overrides,
            pid_type=config.get("pid_type"),
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            arch_major=arch_major,
        )
        if not errors:
            return
        if fix_invalid:
            pid_type = config.get("pid_type")
            for key in TCGEN05_STRATEGY_CONFIG_KEYS:
                if key in TCGEN05_LAYOUT_OVERRIDES_KEYS:
                    config[key] = None
                else:
                    config[key] = self.strategy_field_default(key, pid_type=pid_type)
            return
        message = "; ".join(errors)
        raise InvalidConfig(f"tcgen05 strategy invariants violated: {message}")

    def _clamp_l2_swizzle_size_to_shape(self, config: dict[str, object]) -> None:
        # CuTe layout construction assumes the L2 swizzle does not exceed the
        # number of N tile-clusters; clamp before layout objects are built.
        swizzle_value = config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
        if swizzle_value is None or swizzle_value == 1:
            return
        if len(self.config_spec.block_sizes) < 2:
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 2:
            return
        bn = block_sizes[1]
        if not isinstance(bn, int) or isinstance(bn, bool) or bn <= 0:
            return
        n_hint = self.config_spec.block_sizes[1].size_hint
        if n_hint <= 0:
            return
        cluster_n_raw = config.get("tcgen05_cluster_n", 1)
        cluster_n = (
            cluster_n_raw
            if isinstance(cluster_n_raw, int) and not isinstance(cluster_n_raw, bool)
            else 1
        )
        cluster_n = max(cluster_n, 1)
        ncluster_n = max(((n_hint + bn - 1) // bn) // cluster_n, 1)
        if not isinstance(swizzle_value, int) or isinstance(swizzle_value, bool):
            return
        if swizzle_value <= ncluster_n:
            return
        clamped = max(
            (v for v in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if v <= ncluster_n),
            default=1,
        )
        config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = clamped

    def normalize_pre_pid_type(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        optional_fragments = self.optional_fragments()
        optional_search_fragments = self.optional_fragments(for_search=True)
        if self.search_enabled:
            for key, fragment in optional_fragments.items():
                if key in config:
                    config[key] = self._validate_optional_fragment_value(
                        key, fragment, config[key]
                    )
                else:
                    config[key] = optional_search_fragments[key].default()
            self._clamp_l2_swizzle_size_to_shape(config)
        else:
            for key in optional_fragments:
                if key not in config:
                    continue
                if fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
                    )

        strategy_validation_fragments = self.strategy_validation_fragments()
        if not self.search_enabled:
            for key in (
                *strategy_validation_fragments.keys(),
                *TCGEN05_LAYOUT_OVERRIDES_KEYS,
            ):
                if key not in config:
                    continue
                if fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
                    )

        self._validate_enum_config(
            config,
            TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
            TCGEN05_C_ACQUIRE_PLACEMENTS,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
            TCGEN05_ACC_WAIT_PLACEMENTS,
            fix_invalid=fix_invalid,
        )
        self._validate_bool_config(
            config,
            TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )

        for key, modes, normal_mode in (
            (
                TCGEN05_C_STORE_MODE_CONFIG_KEY,
                TCGEN05_C_STORE_MODES,
                TCGEN05_C_STORE_MODE_NORMAL,
            ),
            (
                TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
                TCGEN05_ACC_PRODUCER_MODES,
                TCGEN05_ACC_PRODUCER_MODE_NORMAL,
            ),
            (
                TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
                TCGEN05_ACC_PRODUCER_ADVANCE_MODES,
                TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
                TCGEN05_AB_PRODUCER_ACQUIRE_MODES,
                TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
                TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES,
                TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
                TCGEN05_AB_PRODUCER_ADVANCE_MODES,
                TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
                TCGEN05_AB_CONSUMER_WAIT_MODES,
                TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
                TCGEN05_AB_CONSUMER_PHASE_MODES,
                TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL,
            ),
        ):
            self._validate_diagnostic_mode(
                config, key, modes, normal_mode, fix_invalid=fix_invalid
            )

        self._validate_bool_config(
            config, TCGEN05_CUBIN_LINEINFO_CONFIG_KEY, fix_invalid=fix_invalid
        )
        self._validate_bool_config(
            config, TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, fix_invalid=fix_invalid
        )
        if config.get(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY) is True:
            proof_envelope_matches = (
                tuple(cast("list[int]", config.get("block_sizes", [])))
                == TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
                and config.get("tcgen05_cluster_m", 1)
                == TCGEN05_LARGE_BN_PROOF_CLUSTER_M
                and config.get("pid_type", "flat") == TCGEN05_LARGE_BN_PROOF_PID_TYPE
                and all(
                    config.get(key) == expected
                    for key, expected in TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
                )
            )
            if not proof_envelope_matches:
                if fix_invalid:
                    config.pop(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY}=True requires "
                        f"block_sizes={list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES)}, "
                        f"tcgen05_cluster_m={TCGEN05_LARGE_BN_PROOF_CLUSTER_M}, "
                        f"pid_type={TCGEN05_LARGE_BN_PROOF_PID_TYPE!r}, "
                        "tcgen05_ab_stages=2, tcgen05_acc_stages=1, "
                        "and tcgen05_c_stages=2"
                    )
        self._validate_bool_config(
            config,
            TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
            TCGEN05_EPILOGUE_LAYOUTS,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
            TCGEN05_SCHED_CONSUMER_WAIT_MODES,
            fix_invalid=fix_invalid,
        )
        if (
            TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY in config
            and config.get(TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY)
            != TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
            and config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            if fix_invalid:
                config.pop(TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY, None)
            else:
                raise InvalidConfig(
                    f"{TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY} is only "
                    "supported with "
                    f"{TCGEN05_STRATEGY_CONFIG_KEY}="
                    f"{Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value!r}"
                )

    def _validate_enum_config(
        self,
        config: dict[str, object],
        key: str,
        choices: tuple[str, ...],
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        if config[key] not in choices:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} must be one of {choices!r}, got {config[key]!r}"
            )

    def _validate_bool_config(
        self,
        config: dict[str, object],
        key: str,
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        if not isinstance(config[key], bool):
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(f"{key} must be a boolean")

    def _validate_diagnostic_mode(
        self,
        config: dict[str, object],
        key: str,
        modes: tuple[str, ...],
        normal_mode: str,
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        if config[key] not in modes:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(f"{key} must be one of {modes!r}, got {config[key]!r}")
        if (
            config[key] != normal_mode
            and config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY) is not True
        ):
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key}={config[key]!r} changes output correctness; set "
                f"{TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY}=True "
                "only for diagnostic invalid-output runs"
            )

    def fix_search_config(self, config: dict[str, object]) -> None:
        self._fix_aux_edge_search_config(config)
        self._fix_cluster_m2_search_config(config)
        self._fix_cluster_m1_persistent_search_config(config)
        self._fix_ab_stages_three_search_config(config)
        self._fix_with_scheduler_search_config(config)

    def normalize_strategy(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        if not self.search_enabled:
            return
        pid_type_for_default = config.get("pid_type")
        strategy_validation_fragments = self.strategy_validation_fragments()
        for key, fragment in strategy_validation_fragments.items():
            if key in config:
                config[key] = self._validate_optional_fragment_value(
                    key, fragment, config[key]
                )
            else:
                config[key] = self.strategy_field_default(
                    key, pid_type=pid_type_for_default
                )
        swizzle_keys = {
            TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY,
            TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY,
        }
        for key in TCGEN05_LAYOUT_OVERRIDES_KEYS:
            if key in config:
                value = config[key]
                if value is None:
                    continue
                if key in swizzle_keys:
                    if (
                        type(value) is not int
                        or value not in TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES
                    ):
                        if fix_invalid:
                            config[key] = None
                        else:
                            raise InvalidConfig(
                                f"{key} must be one of "
                                f"{TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r} "
                                f"or None, got {value!r}"
                            )
                elif not (type(value) is int and value >= 0):
                    if fix_invalid:
                        config[key] = None
                    else:
                        raise InvalidConfig(
                            f"{key} must be a non-negative integer or None, "
                            f"got {value!r}"
                        )
            else:
                config[key] = None
        self.validate_strategy_invariants(config, fix_invalid=fix_invalid)

    def flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "l2_groupings": self.config_spec.l2_groupings,
        }
        fields.update(self.optional_fragments(for_search=True))
        fields.update(self.strategy_autotune_fragments())
        if self.config_spec.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
        if (
            self.config_spec.supports_config_key("indexing")
            and self.config_spec.indexing.length > 0
        ):
            fields["indexing"] = self.config_spec.indexing
        return fields
