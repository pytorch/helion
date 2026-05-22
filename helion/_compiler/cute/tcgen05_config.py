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
from .strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
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
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODES
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from .tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from .tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_C_STORE_MODES
from .tcgen05_constants import TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNTS
from .tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN
from .tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER,
)
from .tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE,
)
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
        TCGEN05_AUX_STAGES_CONFIG_KEY,
        TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
        TCGEN05_CONSUMER_REGS_CONFIG_KEY,
        TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
        TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
        TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
        TCGEN05_LARGE_BN_PROOF_CONFIG_KEY,
        TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY,
        TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY,
        TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY,
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
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
        self.exact_shape_aux_kernel_detected: bool = False
        self.identity_matmul_store_detected: bool = False
        self.relu_matmul_store_detected: bool = False
        self.bias_matmul_store_detected: bool = False
        self.bias_relu_matmul_store_detected: bool = False
        self.target1_tvm_ffi_seed_enabled: bool = False
        self.target2_tvm_ffi_seed_enabled: bool = False
        self.target3_tvm_ffi_seed_enabled: bool = False
        self.target4_tvm_ffi_seed_enabled: bool = False
        self.target5_tvm_ffi_seed_enabled: bool = False
        self.target6_tvm_ffi_seed_enabled: bool = False
        self.target7_tvm_ffi_seed_enabled: bool = False
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

    def allow_target1_tvm_ffi_seed(self) -> None:
        if not self.identity_matmul_store_detected:
            return
        if not self._has_target1_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET1_TVM_FFI_SHAPE[2]
        self.target1_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    def allow_target4_tvm_ffi_seed(self) -> None:
        if not self.relu_matmul_store_detected:
            return
        if not self._has_target4_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET4_TVM_FFI_SHAPE[2]
        self.target4_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    def allow_target5_tvm_ffi_seed(self) -> None:
        # T5 mirrors T4 at the transposed (1024, 8192, 1024) identity-store
        # shape; the shape fact gate distinguishes it from T1.
        if not self.identity_matmul_store_detected:
            return
        if not self._has_target5_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET5_TVM_FFI_SHAPE[2]
        self.target5_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    def allow_target3_tvm_ffi_seed(self) -> None:
        # T3 reuses T4/T5's envelope at the larger (2048, 4096, 2048)
        # identity-store shape; K=2048 -> k_tile_count=16, well below
        # ``TCGEN05_TWO_CTA_MAX_K_TILES``. Shape fact distinguishes T1/T5.
        if not self.identity_matmul_store_detected:
            return
        if not self._has_target3_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET3_TVM_FFI_SHAPE[2]
        self.target3_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    def allow_target2_tvm_ffi_seed(self) -> None:
        # T2 reuses T3/T4/T5's envelope at (4096, 2048, 2048) with a
        # rank-1 rowvec bias epilogue. The bias-store gate is unique to
        # T2 (mutually exclusive with identity/relu/bias_relu).
        if not self.bias_matmul_store_detected:
            return
        if not self._has_target2_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET2_TVM_FFI_SHAPE[2]
        self.target2_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    def allow_target6_tvm_ffi_seed(self) -> None:
        # T6 mirrors T2's envelope at (8192, 2048, 2048) with a fused
        # ``relu(acc + bias[n])`` epilogue. The bias-relu-store gate is
        # unique to T6 (mutually exclusive with identity/relu/bias).
        if not self.bias_relu_matmul_store_detected:
            return
        if not self._has_target6_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET6_TVM_FFI_SHAPE[2]
        self.target6_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    def allow_target7_tvm_ffi_seed(self) -> None:
        # T7 mirrors T3/T5's envelope at (2048, 8192, 2048) identity-store
        # shape; shape fact distinguishes it from T1/T3/T5.
        if not self.identity_matmul_store_detected:
            return
        if not self._has_target7_tvm_ffi_matmul_fact():
            return
        target_k = TCGEN05_TARGET7_TVM_FFI_SHAPE[2]
        self.target7_tvm_ffi_seed_enabled = True
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=target_k,
            max_k_tiles=TCGEN05_TWO_CTA_MAX_K_TILES,
        )
        self.cluster_m_search_choices = (1, 2)
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )

    @staticmethod
    def cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        if bk <= 0:
            return False
        if constraints.allow_edge_k_tail_family:
            return (
                bk
                in (
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
                )
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
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
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

    def _aux_tma_search_enabled(self) -> bool:
        # The TMA aux producer is currently admitted only for the validated
        # Target8-style edge+K-tail family. Exact-shape aux tensors use the
        # aux-TMA producer on both full and partial-output tiles; non-staged
        # aux operands remain on the direct guarded load path.
        constraints = self.cluster_m2_search_constraints
        return (
            self.exact_shape_aux_kernel_detected
            and constraints is not None
            and constraints.allow_edge_k_tail_family
        )

    def _aux_tma_seed_config(self, c_input_seed: Config) -> Config | None:
        if not self._aux_tma_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(c_input_seed.config)
        seed_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
        return Config(**seed_config)

    def _clc_persistence_seed_config(self, base_seed: Config) -> Config | None:
        if not self._clc_persistence_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(base_seed.config)
        seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )
        return Config(**seed_config)

    def _set_clc_aux_tma_edge_perf_knobs(self, config: dict[str, object]) -> None:
        config["tcgen05_acc_stages"] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES
        )
        config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING]
        range_knobs = self._clc_aux_tma_edge_range_knobs()
        if range_knobs is not None:
            (
                config["range_flattens"],
                config["range_multi_buffers"],
                config["range_warp_specializes"],
            ) = range_knobs

    def _clc_aux_tma_wide_n_seed_config(self, clc_aux_tma_seed: Config) -> Config:
        seed_config: dict[str, Any] = dict(clc_aux_tma_seed.config)
        self._set_clc_aux_tma_edge_perf_knobs(seed_config)
        return Config(**seed_config)

    def _clc_aux_tma_edge_range_knobs(
        self,
    ) -> tuple[list[bool | None], list[bool | None], list[bool | None]] | None:
        k_range_index = self._clc_aux_tma_matmul_k_range_index()
        if k_range_index is None:
            return None
        range_flattens: list[bool | None] = [
            None for _ in self.config_spec.range_flattens
        ]
        range_multi_buffers: list[bool | None] = [
            None for _ in self.config_spec.range_multi_buffers
        ]
        range_warp_specializes: list[bool | None] = [
            None for _ in self.config_spec.range_warp_specialize
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

    def _clc_aux_tma_matmul_k_range_index(self) -> int | None:
        k_range_indices: set[int] = set()
        range_flattens_ids = self.config_spec.range_flattens.valid_block_ids()
        range_multi_buffers_ids = self.config_spec.range_multi_buffers.valid_block_ids()
        range_warp_specialize_ids = (
            self.config_spec.range_warp_specialize.valid_block_ids()
        )
        for fact in self.config_spec.matmul_facts:
            k_block_id = fact.k_block_id
            if k_block_id is None:
                continue
            in_range_maps = (
                k_block_id in range_flattens_ids,
                k_block_id in range_multi_buffers_ids,
                k_block_id in range_warp_specialize_ids,
            )
            if not any(in_range_maps):
                continue
            if not all(in_range_maps):
                return None
            range_index = self.config_spec.range_flattens.block_id_to_index(k_block_id)
            if range_index != self.config_spec.range_multi_buffers.block_id_to_index(
                k_block_id
            ):
                return None
            if range_index != self.config_spec.range_warp_specialize.block_id_to_index(
                k_block_id
            ):
                return None
            k_range_indices.add(range_index)
        if len(k_range_indices) != 1:
            return None
        return next(iter(k_range_indices))

    def _clc_aux_tma_narrow_n_seed_config(
        self, clc_aux_tma_seed: Config
    ) -> Config | None:
        if not self._has_any_matmul_fact_n_edge_for_block_n(
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        ):
            return None
        bn_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[1]._fragment(self.config_spec),
        )
        if not (
            bn_fragment.low
            <= TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
            <= bn_fragment.high
        ):
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or not self.cluster_m2_bk_is_valid(
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
            constraints,
        ):
            return None
        seed_config: dict[str, Any] = dict(clc_aux_tma_seed.config)
        seed_config["block_sizes"] = [
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
        ]
        seed_config["tcgen05_acc_stages"] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
        )
        seed_config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING]
        return Config(**seed_config)

    def _has_target1_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET1_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _has_target4_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET4_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _has_target5_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET5_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _has_target3_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET3_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _has_target2_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET2_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _has_target6_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET6_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _has_target7_tvm_ffi_matmul_fact(self) -> bool:
        target_m, target_n, target_k = TCGEN05_TARGET7_TVM_FFI_SHAPE
        return any(
            fact.static_m == target_m
            and fact.static_n == target_n
            and fact.static_k == target_k
            and fact.lhs_dtype == torch.bfloat16
            and fact.rhs_dtype == torch.bfloat16
            for fact in self.config_spec.matmul_facts
        )

    def _target1_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target1_tvm_ffi_seed_enabled:
            return None
        if self.aux_kernel_detected:
            return None
        if not self.identity_matmul_store_detected:
            return None
        if not self._has_target1_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        if self.config_spec.indexing.length != 3:
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET1_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET1_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET1_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET1_TVM_FFI_BLOCK_K,
            ],
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "l2_groupings": [1],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET1_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET1_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def _target4_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target4_tvm_ffi_seed_enabled:
            return None
        if self.aux_kernel_detected:
            return None
        if not self.relu_matmul_store_detected:
            return None
        if not self._has_target4_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        if self.config_spec.indexing.length != 3:
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET4_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET4_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET4_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        # ``l2_groupings=[2]`` is the T4 autotune-selected value; keep
        # ``_is_target4_tvm_ffi_seed_config`` matching and let
        # ``implicit_default_keys_to_preserve`` retain it across overrides.
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET4_TVM_FFI_BLOCK_K,
            ],
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "l2_groupings": [2],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET4_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET4_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def _target5_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target5_tvm_ffi_seed_enabled:
            return None
        if self.aux_kernel_detected:
            return None
        # T5 shares the identity-store gate with T1 but at the T5 shape.
        # The fact check below distinguishes T1 from T5.
        if not self.identity_matmul_store_detected:
            return None
        if not self._has_target5_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        if self.config_spec.indexing.length != 3:
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET5_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET5_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET5_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        # T5 mirrors T4's seed knobs at the transposed M/N identity-store
        # shape; ``l2_groupings=[2]`` is inherited pending a T5 sweep.
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET5_TVM_FFI_BLOCK_K,
            ],
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "l2_groupings": [2],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET5_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET5_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def _target2_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target2_tvm_ffi_seed_enabled:
            return None
        # T2's rowvec bias sets ``aux_kernel_detected`` True; the SIMT
        # aux pipeline doesn't need the c_input_warp/aux-TMA wiring
        # used by exact-shape rank-2 aux tensors. The bias-store gate
        # keeps T2 mutually exclusive with identity/relu stores.
        if not self.bias_matmul_store_detected:
            return None
        if not self._has_target2_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        # Accept indexing length 3 (no closure) or 4 (with bias closure).
        if self.config_spec.indexing.length not in (3, 4):
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET2_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET2_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET2_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        # T2 mirrors T3/T4/T5's seed knobs at the (4096, 2048, 2048)
        # bias-store shape; ``l2_groupings=[2]`` is inherited pending a
        # T2 sweep. The 4th indexing slot (bias) uses ``tensor_descriptor``.
        indexing_length = self.config_spec.indexing.length
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET2_TVM_FFI_BLOCK_K,
            ],
            "indexing": ["tensor_descriptor"] * indexing_length,
            "l2_groupings": [2],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET2_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET2_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def _target6_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target6_tvm_ffi_seed_enabled:
            return None
        # T6's rowvec bias + relu sets ``aux_kernel_detected`` True (same
        # as T2). The bias-relu-store gate keeps T6 mutually exclusive
        # with identity/relu/bias stores.
        if not self.bias_relu_matmul_store_detected:
            return None
        if not self._has_target6_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        # Accept indexing length 3 (no closure) or 4 (with bias closure).
        if self.config_spec.indexing.length not in (3, 4):
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET6_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET6_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET6_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        # T6 mirrors T2/T3/T4/T5's seed knobs at the (8192, 2048, 2048)
        # bias-relu-store shape; ``l2_groupings=[2]`` is inherited pending
        # a T6 sweep. The 4th indexing slot (bias) uses ``tensor_descriptor``.
        indexing_length = self.config_spec.indexing.length
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET6_TVM_FFI_BLOCK_K,
            ],
            "indexing": ["tensor_descriptor"] * indexing_length,
            "l2_groupings": [2],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET6_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET6_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def _target3_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target3_tvm_ffi_seed_enabled:
            return None
        if self.aux_kernel_detected:
            return None
        # T3 shares the identity-store gate with T1/T5 but at the T3
        # shape. The fact check below distinguishes T3 from T1/T5.
        if not self.identity_matmul_store_detected:
            return None
        if not self._has_target3_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        if self.config_spec.indexing.length != 3:
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET3_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET3_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET3_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        # T3 mirrors T4/T5's seed knobs at the larger 2048x4096x2048
        # identity-store shape; ``l2_groupings=[2]`` is inherited pending
        # a T3 sweep.
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET3_TVM_FFI_BLOCK_K,
            ],
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "l2_groupings": [2],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET3_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET3_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def _target7_tvm_ffi_seed_config(self) -> Config | None:
        if not self.target7_tvm_ffi_seed_enabled:
            return None
        if self.aux_kernel_detected:
            return None
        # T7 shares the identity-store gate with T1/T3/T5 but at the T7
        # shape. The fact check below distinguishes T7 from T1/T3/T5.
        if not self.identity_matmul_store_detected:
            return None
        if not self._has_target7_tvm_ffi_matmul_fact():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None
        if self.config_spec.indexing.length != 3:
            return None
        if not self.cluster_m2_bk_is_valid(
            TCGEN05_TARGET7_TVM_FFI_BLOCK_K, constraints
        ):
            return None
        if not self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=TCGEN05_TARGET7_TVM_FFI_BLOCK_K,
            cluster_m=2,
            ab_stages=TCGEN05_TARGET7_TVM_FFI_AB_STAGES,
        ):
            return None
        range_count = len(self.config_spec.range_unroll_factors)
        # T7 mirrors T3/T4/T5's seed knobs at the larger 2048x8192x2048
        # identity-store shape; ``l2_groupings=[2]`` is inherited pending
        # a T7 sweep.
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET7_TVM_FFI_BLOCK_K,
            ],
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "l2_groupings": [2],
            "loop_orders": [[0, 1]],
            "num_stages": 4,
            "num_warps": 8,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "range_flattens": [None] * range_count,
            "range_multi_buffers": [None] * range_count,
            "range_num_stages": [0] * range_count,
            "range_unroll_factors": [1] * range_count,
            "range_warp_specializes": [None] * range_count,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_TARGET7_TVM_FFI_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_TARGET7_TVM_FFI_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        return Config(**seed_config)

    def autotune_seed_configs(self) -> list[Config]:
        seeds: list[Config] = []
        target1_tvm_ffi_seed = self._target1_tvm_ffi_seed_config()
        if target1_tvm_ffi_seed is not None:
            seeds.append(target1_tvm_ffi_seed)
        target2_tvm_ffi_seed = self._target2_tvm_ffi_seed_config()
        if target2_tvm_ffi_seed is not None:
            seeds.append(target2_tvm_ffi_seed)
        target3_tvm_ffi_seed = self._target3_tvm_ffi_seed_config()
        if target3_tvm_ffi_seed is not None:
            seeds.append(target3_tvm_ffi_seed)
        target4_tvm_ffi_seed = self._target4_tvm_ffi_seed_config()
        if target4_tvm_ffi_seed is not None:
            seeds.append(target4_tvm_ffi_seed)
        target5_tvm_ffi_seed = self._target5_tvm_ffi_seed_config()
        if target5_tvm_ffi_seed is not None:
            seeds.append(target5_tvm_ffi_seed)
        target6_tvm_ffi_seed = self._target6_tvm_ffi_seed_config()
        if target6_tvm_ffi_seed is not None:
            seeds.append(target6_tvm_ffi_seed)
        target7_tvm_ffi_seed = self._target7_tvm_ffi_seed_config()
        if target7_tvm_ffi_seed is not None:
            seeds.append(target7_tvm_ffi_seed)
        c_input_seed = self._c_input_seed_config()
        if c_input_seed is not None:
            seeds.append(c_input_seed)
            clc_c_input_seed = self._clc_persistence_seed_config(c_input_seed)
            if clc_c_input_seed is not None:
                seeds.append(clc_c_input_seed)
            aux_tma_seed = self._aux_tma_seed_config(c_input_seed)
            if aux_tma_seed is not None:
                seeds.append(aux_tma_seed)
                clc_aux_tma_seed = self._clc_persistence_seed_config(aux_tma_seed)
                if clc_aux_tma_seed is not None:
                    clc_aux_tma_seed = self._clc_aux_tma_wide_n_seed_config(
                        clc_aux_tma_seed
                    )
                    seeds.append(clc_aux_tma_seed)
                    clc_aux_tma_narrow_n_seed = self._clc_aux_tma_narrow_n_seed_config(
                        clc_aux_tma_seed
                    )
                    if clc_aux_tma_narrow_n_seed is not None:
                        seeds.append(clc_aux_tma_narrow_n_seed)
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
        is_narrow_clc_aux_tma = self._is_clc_aux_tma_narrow_n_request(config)
        if edge_k_tail_family:
            block_sizes[2] = (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
                if is_narrow_clc_aux_tma
                else TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
            )
        bk = block_sizes[2]
        if not isinstance(bk, int) or isinstance(bk, bool):
            config["tcgen05_cluster_m"] = 1
            return
        if not self.cluster_m2_bk_is_valid(bk, constraints):
            config["tcgen05_cluster_m"] = 1
            return
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
        # Only the fully validated narrow-N CLC+aux-TMA seed may keep
        # block_n=128; other candidates use the canonical block_n=256.
        if is_narrow_clc_aux_tma:
            block_sizes[1] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        else:
            block_sizes[1] = TCGEN05_TWO_CTA_BLOCK_N
        if edge_k_tail_family:
            # This family is pinned to measured production stage/pipeline
            # values after search projection.
            # Placement keys remain available for non-edge diagnostic/search
            # paths, but edge+K-tail candidates do not explore partially
            # mutated placement variants.
            config.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
            if self.aux_kernel_detected and self._has_any_matmul_fact_edge_tile(config):
                self._set_aux_edge_cluster_m2_prefix(config)
            if self._is_clc_aux_tma_config(config):
                self._set_clc_aux_tma_edge_perf_knobs(config)
            if is_narrow_clc_aux_tma:
                config["tcgen05_acc_stages"] = (
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
                )
                config["l2_groupings"] = [
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
                ]

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
        ab_stages: int = 3,
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
            ab_stages=ab_stages,
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
        if strategy in (
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            Tcgen05Strategy.PURE_MATMUL_ROLE_LIFECYCLE.value,
        ):
            if scheduler_warps != 0:
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
            if c_input_warps != 0:
                config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        elif strategy == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value:
            if scheduler_warps != 1:
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        if ab_stages == 3 and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1:
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0

    def _fix_aux_tma_search_config(self, config: dict[str, object]) -> None:
        if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA:
            return
        if not self._aux_tma_search_enabled():
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
            return
        if not (
            self._is_validated_cluster_m2_edge_search_candidate(config)
            and self._is_with_scheduler_c_input_config(config)
        ):
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT

    @staticmethod
    def _is_with_scheduler_c_input_config(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) == 1
            and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
        )

    def _clc_persistence_search_enabled(self) -> bool:
        """CLC search is the sm100+ slice of the aux-TMA edge+K-tail gate."""
        if not self._aux_tma_search_enabled():
            return False
        capability = self.config_spec.target_device_capability
        if capability is None:
            return False
        return capability[0] >= 10 and "flat" in self.allowed_pid_types

    def _is_clc_aux_tma_request(self, config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
            and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.CLC_PERSISTENT.value
            and self._clc_persistence_search_enabled()
        )

    def _is_clc_aux_tma_config(self, config: dict[str, object]) -> bool:
        return self._is_clc_aux_tma_request(
            config
        ) and self._is_validated_clc_persistence_search_candidate(config)

    def _is_clc_aux_tma_narrow_n_request(self, config: dict[str, object]) -> bool:
        block_sizes = config.get("block_sizes")
        if not (
            isinstance(block_sizes, list)
            and len(block_sizes) >= 3
            and block_sizes[1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
            and self._is_clc_aux_tma_request(config)
        ):
            return False
        projected_config = dict(config)
        projected_block_sizes = list(block_sizes)
        projected_config["block_sizes"] = projected_block_sizes
        projected_config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        projected_block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
        projected_block_sizes[2] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
        if (
            self.aux_kernel_detected
            and self._has_any_matmul_fact_edge_tile(projected_config)
            and projected_config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            projected_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
            projected_config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
        return self._is_validated_clc_persistence_search_candidate(projected_config)

    def implicit_default_keys_to_preserve(self, config: dict[str, object]) -> set[str]:
        if (
            self._is_target1_tvm_ffi_seed_config(config)
            or self._is_target2_tvm_ffi_seed_config(config)
            or self._is_target3_tvm_ffi_seed_config(config)
            or self._is_target4_tvm_ffi_seed_config(config)
            or self._is_target5_tvm_ffi_seed_config(config)
            or self._is_target6_tvm_ffi_seed_config(config)
            or self._is_target7_tvm_ffi_seed_config(config)
        ):
            return {
                "indexing",
                "l2_groupings",
                "loop_orders",
                "num_stages",
                "num_warps",
                "pid_type",
                "range_flattens",
                "range_multi_buffers",
                "range_num_stages",
                "range_unroll_factors",
                "range_warp_specializes",
            }
        if not self._is_clc_aux_tma_config(config):
            return set()
        preserve_keys = {"l2_groupings"}
        if self._clc_aux_tma_matmul_k_range_index() is not None:
            preserve_keys.update(
                {
                    "range_flattens",
                    "range_multi_buffers",
                    "range_warp_specializes",
                }
            )
        return preserve_keys

    @staticmethod
    def _is_target1_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET1_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET1_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET1_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    @staticmethod
    def _is_target4_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET4_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET4_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET4_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    @staticmethod
    def _is_target5_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        # T5 mirrors T4 at the bk=128 stage tuple (shape gated upstream).
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET5_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET5_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET5_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    @staticmethod
    def _is_target2_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        # T2 mirrors T3/T4/T5 at the bk=128 stage tuple; the bias-store
        # gate happens upstream in ``_target2_tvm_ffi_seed_config``.
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET2_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET2_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET2_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    @staticmethod
    def _is_target3_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        # T3 mirrors T4/T5 at the bk=128 stage tuple; the identity-store
        # gate happens upstream in ``_target3_tvm_ffi_seed_config``.
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET3_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET3_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET3_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    @staticmethod
    def _is_target6_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        # T6 mirrors T2/T3/T4/T5 at the bk=128 stage tuple; the
        # bias-relu-store gate happens upstream.
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET6_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET6_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET6_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    @staticmethod
    def _is_target7_tvm_ffi_seed_config(config: dict[str, object]) -> bool:
        # T7 mirrors T3/T4/T5/T6 at the bk=128 stage tuple; the
        # identity-store gate happens upstream.
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            and config.get("block_sizes")
            == [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                TCGEN05_TARGET7_TVM_FFI_BLOCK_K,
            ]
            and config.get("tcgen05_ab_stages") == TCGEN05_TARGET7_TVM_FFI_AB_STAGES
            and config.get("tcgen05_c_stages") == TCGEN05_TARGET7_TVM_FFI_C_STAGES
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n") == 1
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) == 128
            and config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) == 32
            and config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) == 32
        )

    def _validate_target1_ab_stage_envelope(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        ab_stages = config.get("tcgen05_ab_stages")
        if type(ab_stages) is not int or ab_stages <= 3:
            return
        if self._is_target1_tvm_ffi_seed_config(config):
            return
        if fix_invalid:
            config["tcgen05_ab_stages"] = 3
            return
        raise InvalidConfig(
            "tcgen05_ab_stages > 3 is only supported by the validated "
            "Target1 TVM-FFI seed"
        )

    def _is_validated_clc_persistence_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if not self._clc_persistence_search_enabled():
            return False
        if not self._is_validated_cluster_m2_edge_search_candidate(config):
            return False
        if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
            return False
        if config.get("tcgen05_cluster_n", 1) != 1:
            return False
        if not self._is_with_scheduler_c_input_config(config):
            return False
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 3:
            return False
        if block_sizes[0] != TCGEN05_TWO_CTA_BLOCK_M:
            return False
        if block_sizes[1] not in (
            TCGEN05_TWO_CTA_BLOCK_N,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
        ):
            return False
        is_narrow_n = block_sizes[1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        if is_narrow_n:
            if not self._has_any_matmul_fact_n_edge_for_block_n(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
            ):
                return False
            if (
                config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                != TCGEN05_AUX_LOAD_MODE_TMA
            ):
                return False
            if block_sizes[2] not in (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
            ):
                return False
        elif block_sizes[2] != TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K:
            return False
        if self.config_spec.supports_config_key("indexing"):
            indexing = config.get("indexing")
            if (
                not isinstance(indexing, list)
                or indexing != ["tensor_descriptor"] * self.config_spec.indexing.length
            ):
                return False
        return True

    def _fix_clc_persistence_search_config(self, config: dict[str, object]) -> None:
        if (
            config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            != Tcgen05PersistenceModel.CLC_PERSISTENT.value
        ):
            return
        if self._is_validated_clc_persistence_search_candidate(config):
            return
        config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            self.persistence_model_default_from_config(config).value
        )

    def _validate_sched_stage_count_config(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        self._validate_int_enum_config(
            config,
            TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY,
            TCGEN05_SCHED_STAGE_COUNTS,
            fix_invalid=fix_invalid,
        )
        block_sizes = config.get("block_sizes")
        is_full_role_local_two_cta_shape = (
            isinstance(block_sizes, list)
            and len(block_sizes) >= 3
            and block_sizes[0] == TCGEN05_TWO_CTA_BLOCK_M
            and config.get("pid_type") == TCGEN05_TWO_CTA_SEED_PID_TYPE
            and config.get("tcgen05_cluster_m", 1) == 2
            and config.get("tcgen05_cluster_n", 1) == 1
            and config.get(TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY) is not True
        )
        if (
            TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY in config
            and config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY) != 1
            and (
                config.get(TCGEN05_STRATEGY_CONFIG_KEY)
                != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
                or config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                != Tcgen05PersistenceModel.CLC_PERSISTENT.value
                or config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY, 0) == 0
                or not is_full_role_local_two_cta_shape
            )
        ):
            if fix_invalid:
                config.pop(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY, None)
            else:
                raise InvalidConfig(
                    f"{TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY}=2 is only supported "
                    "with "
                    f"{TCGEN05_STRATEGY_CONFIG_KEY}="
                    f"{Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value!r} and "
                    f"{TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY}="
                    f"{Tcgen05PersistenceModel.CLC_PERSISTENT.value!r} and "
                    "the omitted shared-loop full role-local CtaGroup.TWO "
                    "shape: pid_type='persistent_interleaved', "
                    "tcgen05_cluster_m=2, tcgen05_cluster_n=1, block_m=256, "
                    f"and {TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY} > 0"
                )

    def prepare_override_normalization(
        self,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        if "pid_type" not in overrides:
            return
        if TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY in overrides:
            return
        persistence_value = config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        if persistence_value is None:
            return
        try:
            persistence_model = Tcgen05PersistenceModel(persistence_value)
        except ValueError:
            return
        pid_type = overrides["pid_type"]
        if pid_type not in self.allowed_pid_types:
            return
        derived = derive_persistence_model_from_pid_type(pid_type)
        compatible = persistence_model is derived or (
            persistence_model is Tcgen05PersistenceModel.CLC_PERSISTENT
            and derived is Tcgen05PersistenceModel.STATIC_PERSISTENT
        )
        if not compatible:
            config.pop(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, None)

    def persistence_model_default_from_config(
        self,
        config: dict[str, object],
    ) -> Tcgen05PersistenceModel:
        """Derive default persistence from pid_type."""
        pid_type = config.get("pid_type", self.allowed_pid_types[0])
        if pid_type not in self.allowed_pid_types:
            pid_type = self.allowed_pid_types[0]
        return derive_persistence_model_from_pid_type(pid_type)

    def flatten_missing_field_default(
        self,
        key: str,
        config: dict[str, object],
    ) -> tuple[bool, object]:
        if key != TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY:
            return False, None
        projected_config = {
            config_key: [*value] if isinstance(value, list) else value
            for config_key, value in config.items()
        }
        self.fix_search_config(projected_config)
        return True, self.persistence_model_default_from_config(projected_config).value

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

    def _has_any_matmul_fact_n_edge_for_block_n(self, block_n: int) -> bool:
        for fact in self.config_spec.matmul_facts:
            if fact.static_n is None or fact.n_block_id is None:
                continue
            try:
                # Presence check: skip facts whose N block id is not registered
                # in this config spec.
                self.config_spec.block_sizes.block_id_to_index(fact.n_block_id)
            except KeyError:
                continue
            if fact.static_n % block_n != 0:
                return True
        return False

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
        if not for_search and self._target1_tvm_ffi_seed_config() is not None:
            ab_stages_max = TCGEN05_TARGET1_TVM_FFI_AB_STAGES
        elif not for_search and self._target2_tvm_ffi_seed_config() is not None:
            ab_stages_max = max(3, TCGEN05_TARGET2_TVM_FFI_AB_STAGES)
        elif not for_search and self._target3_tvm_ffi_seed_config() is not None:
            ab_stages_max = max(3, TCGEN05_TARGET3_TVM_FFI_AB_STAGES)
        elif not for_search and self._target4_tvm_ffi_seed_config() is not None:
            ab_stages_max = max(3, TCGEN05_TARGET4_TVM_FFI_AB_STAGES)
        elif not for_search and self._target5_tvm_ffi_seed_config() is not None:
            ab_stages_max = max(3, TCGEN05_TARGET5_TVM_FFI_AB_STAGES)
        elif not for_search and self._target6_tvm_ffi_seed_config() is not None:
            ab_stages_max = max(3, TCGEN05_TARGET6_TVM_FFI_AB_STAGES)
        elif not for_search and self._target7_tvm_ffi_seed_config() is not None:
            ab_stages_max = max(3, TCGEN05_TARGET7_TVM_FFI_AB_STAGES)
        elif not for_search:
            ab_stages_max = 3
        else:
            ab_stages_max = 2
        if for_search:
            l2_swizzle_choices = tuple(
                v for v in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if v <= 8
            )
        else:
            l2_swizzle_choices = TCGEN05_LEGAL_L2_SWIZZLE_SIZES
        fragments: dict[str, ConfigSpecFragment] = {
            "tcgen05_cluster_m": EnumFragment(cluster_m_choices),
            "tcgen05_cluster_n": EnumFragment(cluster_n_choices),
            "tcgen05_ab_stages": IntegerFragment(1, ab_stages_max, 2),
            "tcgen05_acc_stages": IntegerFragment(1, 2, 2),
            "tcgen05_c_stages": EnumFragment((2, 4)),
            "tcgen05_num_epi_warps": num_epi_warps_fragment,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: EnumFragment(l2_swizzle_choices),
        }
        if (
            self._target1_tvm_ffi_seed_config() is not None
            or self._target2_tvm_ffi_seed_config() is not None
            or self._target3_tvm_ffi_seed_config() is not None
            or self._target4_tvm_ffi_seed_config() is not None
            or self._target5_tvm_ffi_seed_config() is not None
            or self._target6_tvm_ffi_seed_config() is not None
            or self._target7_tvm_ffi_seed_config() is not None
        ):
            fragments.update(
                {
                    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: BooleanFragment(),
                    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: BooleanFragment(),
                    TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: EnumFragment((None, 128)),
                    TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: EnumFragment((None, 32)),
                    TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: EnumFragment(
                        (None, 32)
                    ),
                }
            )
            if not for_search:
                fragments[TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY] = BooleanFragment()
                fragments[TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY] = (
                    BooleanFragment()
                )
                # Pure-dynamic scheduler-object key is in the validation
                # surface only — user configs that set it round-trip through
                # normalize, but the autotune ``for_search`` surface excludes
                # it until the productive codegen lands.
                fragments[TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY] = (
                    BooleanFragment()
                )
        return fragments

    @staticmethod
    def _target1_tvm_ffi_promotion_requested(
        config: dict[str, object], *, seed_enabled: bool
    ) -> bool:
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            or config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            or config.get(TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY) is True
            or config.get(TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY) is True
            or config.get(TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY) is True
            or (seed_enabled and config.get("tcgen05_cluster_m") == 2)
            or config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            or config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) is not None
            or config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) is not None
            or config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) is not None
        )

    @staticmethod
    def _clear_target1_tvm_ffi_promotion_surface(config: dict[str, object]) -> None:
        for key in list(config):
            if key.startswith("tcgen05_") or key == "epilogue_subtile":
                config.pop(key, None)

    def _fix_target1_tvm_ffi_search_config(self, config: dict[str, object]) -> None:
        # T1-T7 share the TVM-FFI direct-entry promotion surface; pick
        # the seed matching the detected (shape, store) so search
        # candidates project onto the right envelope. The shape+store
        # gates are mutually exclusive so at most one seed is non-None.
        seed = self._target1_tvm_ffi_seed_config()
        if seed is None:
            seed = self._target2_tvm_ffi_seed_config()
        if seed is None:
            seed = self._target3_tvm_ffi_seed_config()
        if seed is None:
            seed = self._target4_tvm_ffi_seed_config()
        if seed is None:
            seed = self._target5_tvm_ffi_seed_config()
        if seed is None:
            seed = self._target6_tvm_ffi_seed_config()
        if seed is None:
            seed = self._target7_tvm_ffi_seed_config()
        if not self._target1_tvm_ffi_promotion_requested(
            config, seed_enabled=seed is not None
        ):
            return
        if seed is None:
            config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY] = False
            config[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY] = False
            config[TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY] = False
            config[TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY] = False
            config[TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY] = False
            if (
                config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
                == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ):
                config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = (
                    Tcgen05LayoutStrategy.DEFAULT.value
                )
            for key in (
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY,
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY,
                TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY,
            ):
                config[key] = None
            return
        pure_clc_scheduler_requested = (
            config.get(TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY) is True
        )
        pure_dynamic_scheduler_requested = (
            config.get(TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY) is True
        )
        direct_entry_requested = (
            config.get(TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY) is True
        )
        self._clear_target1_tvm_ffi_promotion_surface(config)
        config.update(seed.config)
        if direct_entry_requested:
            config[TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY] = True
        if pure_clc_scheduler_requested:
            config[TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY] = True
        if pure_dynamic_scheduler_requested:
            config[TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY] = True

    def aux_load_mode_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        if not self._aux_tma_search_enabled():
            return {}
        return {
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: EnumFragment(
                (TCGEN05_AUX_LOAD_MODE_SIMT, TCGEN05_AUX_LOAD_MODE_TMA)
            )
        }

    def aux_stages_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Per-config aux-pipeline stage-count knob.

        Admitted only under ``_aux_tma_search_enabled``, which pins the
        surface to the validated edge+K-tail family with ``cluster_m=2``
        and the c-input warp + aux-TMA combination. Configs outside that
        gate never see the knob; codegen at the default of 2 is unchanged.
        """
        if not self._aux_tma_search_enabled():
            return {}
        return {
            TCGEN05_AUX_STAGES_CONFIG_KEY: EnumFragment(TCGEN05_AUX_STAGE_COUNT_CHOICES)
        }

    def consumer_regs_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Per-config consumer-warp ``setmaxregister_increase`` ceiling knob.

        Admission mirrors ``aux_stages_autotune_fragments``: the
        ``_aux_tma_search_enabled`` gate pins the search to the validated
        wide-N CLC + aux-TMA seed family with the c-input warp + aux-TMA
        combination. Configs outside that gate never see the knob. The
        default value (256) is included in ``TCGEN05_CONSUMER_REGS_CHOICES``
        so default-with-knob emits the same code as default-without-knob.
        """
        if not self._aux_tma_search_enabled():
            return {}
        return {
            TCGEN05_CONSUMER_REGS_CONFIG_KEY: EnumFragment(
                TCGEN05_CONSUMER_REGS_CHOICES
            )
        }

    def persistence_model_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        if not self._clc_persistence_search_enabled():
            return {}
        default_model = derive_persistence_model_from_pid_type(
            self.allowed_pid_types[0]
        ).value
        choices = tuple(
            dict.fromkeys(
                (
                    default_model,
                    Tcgen05PersistenceModel.NON_PERSISTENT.value,
                    Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value,
                )
            )
        )
        return {TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: EnumFragment(choices)}

    def strategy_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        # Aux kernels are the only current trigger for scheduler/c_input warp
        # search. The surface is derived from aux_kernel_detected so repeated
        # detection or repeated fragment construction stays idempotent.
        target1_tvm_ffi_seed_enabled = (
            self._target1_tvm_ffi_seed_config() is not None
            or self._target2_tvm_ffi_seed_config() is not None
            or self._target3_tvm_ffi_seed_config() is not None
            or self._target4_tvm_ffi_seed_config() is not None
            or self._target5_tvm_ffi_seed_config() is not None
            or self._target6_tvm_ffi_seed_config() is not None
            or self._target7_tvm_ffi_seed_config() is not None
        )
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
        if target1_tvm_ffi_seed_enabled:
            layout_choices = (
                Tcgen05LayoutStrategy.DEFAULT.value,
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
            )
        else:
            layout_choices = (Tcgen05LayoutStrategy.DEFAULT.value,)
        return {
            TCGEN05_STRATEGY_CONFIG_KEY: EnumFragment(strategy_choices),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: EnumFragment(layout_choices),
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
                Tcgen05Strategy.PURE_MATMUL_ROLE_LIFECYCLE.value,
            )
        )
        fragments[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = EnumFragment(
            (
                Tcgen05LayoutStrategy.DEFAULT.value,
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
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
        capability = self.config_spec.target_device_capability
        arch_major = capability[0] if capability is not None else None
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
                elif key in optional_search_fragments:
                    config[key] = optional_search_fragments[key].default()
            self._clamp_l2_swizzle_size_to_shape(config)
            self._validate_target1_ab_stage_envelope(config, fix_invalid=fix_invalid)
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
        self._validate_enum_config(
            config,
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
            TCGEN05_AUX_LOAD_MODES,
            fix_invalid=fix_invalid,
        )
        self._validate_int_enum_config(
            config,
            TCGEN05_AUX_STAGES_CONFIG_KEY,
            TCGEN05_AUX_STAGE_COUNT_CHOICES,
            fix_invalid=fix_invalid,
        )
        self._validate_int_enum_config(
            config,
            TCGEN05_CONSUMER_REGS_CONFIG_KEY,
            TCGEN05_CONSUMER_REGS_CHOICES,
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
            config, TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY, fix_invalid=fix_invalid
        )
        self._validate_bool_config(
            config, TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, fix_invalid=fix_invalid
        )
        self._validate_bool_config(
            config,
            TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )
        self._validate_bool_config(
            config,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )
        self._validate_bool_config(
            config,
            TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )
        self._validate_bool_config(
            config,
            TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY,
            fix_invalid=fix_invalid,
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
        self._validate_sched_stage_count_config(config, fix_invalid=fix_invalid)

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

    def _validate_int_enum_config(
        self,
        config: dict[str, object],
        key: str,
        choices: tuple[int, ...],
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
        value = config[key]
        # ``bool`` is an ``int`` subclass, but it is not a valid stage count.
        if type(value) is not int or value not in choices:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(f"{key} must be one of {choices!r}, got {value!r}")

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
        self._fix_target1_tvm_ffi_search_config(config)
        self._fix_with_scheduler_search_config(config)
        self._fix_aux_tma_search_config(config)
        # CLC admission depends on the projected cluster/pid/strategy tuple
        # above, including the scheduler/c-input warp fix-ups.
        self._fix_clc_persistence_search_config(config)
        self._validate_sched_stage_count_config(config, fix_invalid=True)

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
                elif not (type(value) is int and value > 0):
                    if fix_invalid:
                        config[key] = None
                    else:
                        raise InvalidConfig(
                            f"{key} must be a positive integer or None, got {value!r}"
                        )
            else:
                config[key] = None
        self.validate_strategy_invariants(config, fix_invalid=fix_invalid)
        if fix_invalid:
            # Strategy validation can reset scheduler/c-input fields for
            # inconsistent user configs. Revalidate aux-TMA after that reset so
            # TMA aux loads do not outlive their producer warp. CLC does not
            # need a matching second pass because the reset path never produces
            # a CLC persistence model.
            self._fix_aux_tma_search_config(config)

    def flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "l2_groupings": self.config_spec.l2_groupings,
        }
        fields.update(self.optional_fragments(for_search=True))
        fields.update(self.strategy_autotune_fragments())
        fields.update(self.aux_load_mode_autotune_fragments())
        fields.update(self.aux_stages_autotune_fragments())
        fields.update(self.consumer_regs_autotune_fragments())
        fields.update(self.persistence_model_autotune_fragments())
        if self.config_spec.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
        if (
            self.config_spec.supports_config_key("indexing")
            and self.config_spec.indexing.length > 0
        ):
            fields["indexing"] = self.config_spec.indexing
        return fields
