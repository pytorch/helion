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
from .strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
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
from .tcgen05_constants import TCGEN05_AB_STAGES_MIN_DEVICE_SMEM_OPTIN
from .tcgen05_constants import TCGEN05_AB_STAGES_RESERVED_SMEM_BYTES
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODES
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODES
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from .tcgen05_constants import TCGEN05_AUX_EDGE_SEED_BLOCK_K
from .tcgen05_constants import TCGEN05_AUX_EDGE_SEED_BLOCK_N
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODES
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENTS
from .tcgen05_constants import TCGEN05_AUX_PRODUCER_WARP_MAX_AB_STAGES
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from .tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from .tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_C_STORE_MODES
from .tcgen05_constants import TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CLUSTER_M2_REPAIR_BLOCK_K_ORDER
from .tcgen05_constants import TCGEN05_CLUSTER_M2_SEED_MIN_AB_STAGES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIRECT_ENTRY_LEGAL_BK
from .tcgen05_constants import TCGEN05_DIRECT_ENTRY_SEED_AB_STAGES
from .tcgen05_constants import TCGEN05_DIRECT_ENTRY_SEED_C_STAGES
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from .tcgen05_constants import TCGEN05_EXPLICIT_D_STORE_BOX_N
from .tcgen05_constants import TCGEN05_EXPLICIT_EPI_TILE_M
from .tcgen05_constants import TCGEN05_EXPLICIT_EPI_TILE_N
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_DYNAMIC_MODES
from .tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_MODE_DIRECT
from .tcgen05_constants import TCGEN05_GROUPED_MODE_DYNAMIC
from .tcgen05_constants import TCGEN05_GROUPED_MODE_STATIC
from .tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from .tcgen05_constants import TCGEN05_GROUPED_MODES
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from .tcgen05_constants import TCGEN05_MIN_CONCEDED_BLOCK_K
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNTS
from .tcgen05_constants import TCGEN05_STAGED_WORK_TILE_MAILBOX_SCHED_STAGES
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
from .tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_default_epilogue_tile_size
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
    # When True, a sampled bm<=128 cluster_m=2 candidate is projected onto the
    # fp8 small-grid 2-CTA tile (bm=128/bn=128, per-CTA 64xbn) instead of the
    # bm=256 full tile. Gated to fp8 by the caller, mirroring
    # ``_tcgen05_use_2cta_instrs`` (``bm == 128 and is_fp8``). See the
    # ``TCGEN05_TWO_CTA_FP8_SMALL_GRID_*`` constants.
    allow_fp8_small_grid: bool = False


class Tcgen05AbStagesSearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_ab_stages=3`` is admitted."""

    dtype_bytes: int
    per_cta_smem_budget_bytes: int
    per_cta_smem_capacity_bytes: int = 0


TCGEN05_GROUPED_DYNAMIC_AB4_STAGE = 4
TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES = ((4, 2), (8, 4))
# The generated grouped kernel's non-operand allocations are about 1.6 KiB
# (pipeline barriers, TensorMap staging, and TMEM bookkeeping). Keep a small
# margin while still admitting CUTLASS's max-fit AB8/C4 pipeline on B200.
TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES = 2 * 1024


CUTE_TCGEN05_TUNABLE_KEYS: tuple[str, ...] = (
    "tcgen05_cluster_m",
    "tcgen05_cluster_n",
    "tcgen05_ab_stages",
    "tcgen05_acc_stages",
    "tcgen05_c_stages",
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
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
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
        TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY,
        TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY,
        TCGEN05_GROUPED_MODE_CONFIG_KEY,
        TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY,
        TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
        TCGEN05_LARGE_BN_PROOF_CONFIG_KEY,
        TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY,
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
    }
)


def parse_tcgen05_grouped_static_problem_signature(
    value: object,
) -> tuple[tuple[int, int, int], ...]:
    """Parse ``[group_count, M0, N0, K0, ...]`` from an AOT config."""
    key = TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
    if not isinstance(value, list) or not value:
        raise InvalidConfig(f"{key} must be a non-empty list of integers")
    if any(type(item) is not int for item in value):
        raise InvalidConfig(f"{key} must contain only integers (not booleans)")
    group_count = value[0]
    if group_count <= 0 or len(value) != 1 + 3 * group_count:
        raise InvalidConfig(
            f"{key} must have the form [group_count, M0, N0, K0, ...] "
            "with exactly three positive sizes per group"
        )
    shapes = tuple(
        (value[offset], value[offset + 1], value[offset + 2])
        for offset in range(1, len(value), 3)
    )
    if any(size <= 0 for shape in shapes for size in shape):
        raise InvalidConfig(f"{key} requires every M/N/K size to be positive")
    return shapes


CUTE_TCGEN05_STRATEGY_CONFIG_KEYS: frozenset[str] = frozenset(
    TCGEN05_STRATEGY_CONFIG_KEYS
)


class CuteTcgen05Config:
    """CuTe-owned tcgen05 ConfigSpec state and normalization hooks."""

    def __init__(self, config_spec: ConfigSpec) -> None:
        self.config_spec = config_spec
        self.search_enabled: bool = False
        self.matmul_block_ids: tuple[int, int, int] | None = None
        self.matmul_input_dtype: torch.dtype | None = None
        self.matmul_has_leading_passthrough: bool = False
        self.matmul_explicit_epi_tile_compatible: bool | None = None
        self.aux_kernel_detected: bool = False
        self.exact_shape_aux_kernel_detected: bool = False
        # True when the kernel feeds a matmul an operand sourced from a load
        # whose dtype is not a tcgen05-native MMA dtype (e.g. an int16 tensor
        # cast to bf16, ``w.to(bfloat16)``). Such an operand cannot be TMA-staged
        # for the SMEM tcgen05 MMA, so the dot lowers through the non-tcgen05
        # fallback and the FFI/flat-role direct-entry seed must stay ineligible.
        self.matmul_has_non_tcgen05_operand: bool = False
        self.cluster_m_search_choices: tuple[int, ...] | None = None
        self.cluster_m2_search_constraints: Tcgen05ClusterM2SearchConstraints | None = (
            None
        )
        self.ab_stages_search_constraints: Tcgen05AbStagesSearchConstraints | None = (
            None
        )
        self.deep_direct_entry_validation_enabled: bool = False
        self.num_epi_warps_search_choices: tuple[int, ...] | None = None
        self.num_epi_warps_validation_choices: tuple[int, ...] | None = None

    @property
    def allowed_pid_types(self) -> tuple[PidTypeLiteral, ...]:
        return self.config_spec.allowed_pid_types

    @allowed_pid_types.setter
    def allowed_pid_types(self, value: tuple[PidTypeLiteral, ...]) -> None:
        self.config_spec.allowed_pid_types = value

    def _config_block_index(self, block_id: int | None) -> int | None:
        if (
            block_id is None
            or block_id not in self.config_spec.block_sizes.valid_block_ids()
        ):
            return None
        return self.config_spec.block_sizes.block_id_to_index(block_id)

    def register_mma_analysis(
        self,
        *,
        m_block_id: int,
        n_block_id: int,
        k_block_id: int,
        input_dtype: torch.dtype,
        has_leading_passthrough: bool,
        explicit_epi_tile_compatible: bool,
    ) -> None:
        """Record semantic axes from the structurally accepted MMA candidate."""
        assert self.matmul_block_ids is None, "tcgen05 MMA analysis registered twice"
        self.matmul_block_ids = (m_block_id, n_block_id, k_block_id)
        self.matmul_input_dtype = input_dtype
        self.matmul_has_leading_passthrough = has_leading_passthrough
        self.matmul_explicit_epi_tile_compatible = explicit_epi_tile_compatible

    def _matmul_block_indices(self) -> tuple[int, int, int] | None:
        if self.matmul_block_ids is None:
            return None
        indices = tuple(
            self._config_block_index(block_id) for block_id in self.matmul_block_ids
        )
        if any(index is None for index in indices):
            return None
        return cast("tuple[int, int, int]", indices)

    def _matmul_config_view(
        self, config: dict[str, object]
    ) -> tuple[list[object], int, int, int] | None:
        block_sizes = config.get("block_sizes")
        indices = self._matmul_block_indices()
        if not isinstance(block_sizes, list) or indices is None:
            return None
        m_index, n_index, k_index = indices
        if max(indices) >= len(block_sizes):
            return None
        return block_sizes, m_index, n_index, k_index

    def _matmul_block_fragments(
        self,
    ) -> tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment] | None:
        indices = self._matmul_block_indices()
        if indices is None:
            return None
        return cast(
            "tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment]",
            tuple(
                cast(
                    "BlockSizeFragment",
                    self.config_spec.block_sizes[index]._fragment(self.config_spec),
                )
                for index in indices
            ),
        )

    def _matmul_seed_block_sizes(
        self, *, bm: int, bn: int, bk: int
    ) -> list[int] | None:
        indices = self._matmul_block_indices()
        if indices is None:
            return None
        block_sizes = [
            cast(
                "BlockSizeFragment",
                spec._fragment(self.config_spec),
            ).default()
            for spec in self.config_spec.block_sizes
        ]
        m_index, n_index, k_index = indices
        block_sizes[m_index] = bm
        block_sizes[n_index] = bn
        block_sizes[k_index] = bk
        return block_sizes

    def _direct_entry_k_block_index(self) -> int | None:
        if self.matmul_has_non_tcgen05_operand or self.matmul_input_dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            return None
        indices = self._matmul_block_indices()
        return indices[2] if indices is not None else None

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
        allow_fp8_small_grid: bool = False,
    ) -> None:
        assert static_k > 0, "static_k is required for cluster_m=2 K-cap checks"
        assert max_k_tiles > 0, "cluster_m=2 max K tiles must be positive"
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
            allow_fp8_small_grid=allow_fp8_small_grid,
        )
        self.restrict_cluster_m_search((1, 2))

    def _cluster_m2_capability_holds(self, config: dict[str, object]) -> bool:
        """Whether ``cluster_m=2`` is EMITTABLE for this config, ignoring search scope."""
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return False
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return False
        block_sizes, m_index, _, k_index = config_view
        bm = block_sizes[m_index]
        if type(bm) is not int:
            return False
        if bm == TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M:
            if self.matmul_input_dtype is not torch.float8_e4m3fn:
                return False
        elif bm != TCGEN05_TWO_CTA_BLOCK_M:
            return False
        bk = block_sizes[k_index]
        if type(bk) is not int or bk <= 0:
            return False
        for fact in self.config_spec.matmul_facts:
            static_k = fact.static_k
            if static_k is None:
                continue
            if -(-static_k // bk) > TCGEN05_TWO_CTA_MAX_K_TILES:
                return False
        return True

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

    def full_tile_direct_entry_seed_bk(self) -> int | None:
        """Largest valid full-tile direct-entry K tile for the live shape."""
        constraints = self.cluster_m2_search_constraints
        fragments = self._matmul_block_fragments()
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or fragments is None
        ):
            return None
        bk_fragment = fragments[2]
        bk = min(bk_fragment.high, max(TCGEN05_DIRECT_ENTRY_LEGAL_BK))
        while bk >= bk_fragment.low:
            if bk in TCGEN05_DIRECT_ENTRY_LEGAL_BK and self.cluster_m2_bk_is_valid(
                bk, constraints
            ):
                return bk
            bk //= 2
        return None

    def explicit_epi_tile_family_exists(self) -> bool:
        """Bind-time facts: can the ``explicit_epi_tile`` family exist on this shape?"""
        # A non-tcgen05-native operand forces the dot through the fallback, which
        # rejects the flat-role / FFI config.
        if self.matmul_has_non_tcgen05_operand:
            return False
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return False
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return False
        # 16-bit only: there is no tcgen05 fp32 SMEM-staged MMA path.
        if self.matmul_input_dtype not in (torch.bfloat16, torch.float16):
            return False
        if self.matmul_explicit_epi_tile_compatible is not True:
            return False
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return False
        bm_fragment, bn_fragment, _ = fragments
        if not (bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high):
            return False
        if not (bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high):
            return False
        bk = self.full_tile_direct_entry_seed_bk()
        if bk is None:
            return False
        # cute_mma.py raises for any other bk: the flat-role launch shape and the
        # D-descriptor box are built only for {64, 128}.
        return bk in TCGEN05_DIRECT_ENTRY_LEGAL_BK

    def full_tile_direct_entry_seed_emittable(self) -> bool:
        """Can the FFI direct-entry SEED actually be emitted on this shape?"""
        if not self.explicit_epi_tile_family_exists():
            return False
        bk = self.full_tile_direct_entry_seed_bk()
        if bk is None:
            return False
        return self.ab_stages_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            cluster_m=2,
            ab_stages=TCGEN05_DIRECT_ENTRY_SEED_AB_STAGES,
        )

    def full_tile_direct_entry_seed_config(self) -> Config | None:
        """Generalized TVM-FFI direct-entry seed config for the live shape."""
        if not self.full_tile_direct_entry_seed_emittable():
            return None
        bk = self.full_tile_direct_entry_seed_bk()
        if bk is None:
            return None
        block_sizes = self._matmul_seed_block_sizes(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "l2_groupings": [2],
            "num_warps": 8,
            "num_stages": 4,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": TCGEN05_DIRECT_ENTRY_SEED_AB_STAGES,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": TCGEN05_DIRECT_ENTRY_SEED_C_STAGES,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            # The flat-role launch path uses this fixed explicit subtile: it is the
            # only triple the D-descriptor codegen accepts, so it is the same for
            # every eligible shape.
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: TCGEN05_EXPLICIT_EPI_TILE_M,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: TCGEN05_EXPLICIT_EPI_TILE_N,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: (
                TCGEN05_EXPLICIT_D_STORE_BOX_N
            ),
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        if self.config_spec.indexing.length in (3, 4):
            seed["indexing"] = ["tensor_descriptor"] * self.config_spec.indexing.length
        return Config(**seed)

    def _c_input_seed_config(self) -> Config | None:
        if not self.aux_kernel_detected:
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if self.matmul_has_leading_passthrough:
            return None
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bm_fragment, bn_fragment, bk_fragment = fragments
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
            bk = None
            for require_ab_headroom in (True, False):
                candidate = bk_fragment.high
                while candidate >= bk_fragment.low:
                    if self.cluster_m2_bk_is_valid(candidate, constraints) and (
                        not require_ab_headroom
                        or self.ab_stages_fits(
                            bm=TCGEN05_TWO_CTA_BLOCK_M,
                            bn=TCGEN05_TWO_CTA_BLOCK_N,
                            bk=candidate,
                            cluster_m=2,
                            ab_stages=TCGEN05_CLUSTER_M2_SEED_MIN_AB_STAGES,
                        )
                    ):
                        bk = candidate
                        break
                    candidate //= 2
                if bk is not None:
                    break
            if bk is None:
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

    def _aux_tma_edge_search_enabled(self) -> bool:
        # The TMA aux producer's original admission: the validated Target8-style
        # double-edge + K-tail family with ``cluster_m=2``. The CLC-persistent
        # variants and edge-perf knobs remain pinned to this slice.
        constraints = self.cluster_m2_search_constraints
        return (
            self.exact_shape_aux_kernel_detected
            and constraints is not None
            and constraints.allow_edge_k_tail_family
        )

    def _aux_tma_full_tile_search_enabled(self) -> bool:
        # (T14/T20/T25/T28 residual_add family). The codegen-side gate at
        # ``cute_mma.py`` ``tcgen05_static_output_tiles`` already accepts
        # full-tile shapes; only the search-space gate was excluding them.
        # Edge-perf knobs (``_set_clc_aux_tma_edge_perf_knobs``) and
        # CLC-persistent variants stay pinned to ``_aux_tma_edge_search_enabled``
        # so this widening does not perturb the 5000³ T12 family.
        constraints = self.cluster_m2_search_constraints
        return (
            self.exact_shape_aux_kernel_detected
            and not self.matmul_has_leading_passthrough
            and constraints is not None
            and not constraints.allow_edge_k_tail_family
        )

    def _aux_tma_search_enabled(self) -> bool:
        # The TMA aux producer is admitted on either the edge+K-tail family or
        # the full-tile cluster_m=2 family. Exact-shape aux tensors use the
        # aux-TMA producer on both full and partial-output tiles; non-staged
        # aux operands remain on the direct guarded load path.
        return (
            self._aux_tma_edge_search_enabled()
            or self._aux_tma_full_tile_search_enabled()
        )

    def _aux_tma_seed_config(self, c_input_seed: Config) -> Config | None:
        """The aux-TMA producer regime as a SEED, not a projection."""
        if not self._aux_tma_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(c_input_seed.config)
        seed_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
        # real resource decision rather than a drawn-knob overwrite: it is the
        # foundation for the store-warp split (the store warp drains tile N's TMA-D
        # from the ring while the 4 epi warps run tile N+1's T2R; a 2-stage ring
        if self._aux_tma_full_tile_search_enabled():
            config_view = self._matmul_config_view(seed_config)
            if config_view is not None:
                block_sizes, m_index, n_index, k_index = config_view
                if self.c_stages_fits(
                    bm=cast("int", block_sizes[m_index]),
                    bn=cast("int", block_sizes[n_index]),
                    bk=cast("int", block_sizes[k_index]),
                    cluster_m=2,
                    ab_stages=cast("int", seed_config.get("tcgen05_ab_stages", 2)),
                    c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
                    has_source_c=True,
                ):
                    seed_config["tcgen05_c_stages"] = (
                        TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES
                    )
        return Config(**seed_config)

    def _clc_persistence_seed_config(self, base_seed: Config) -> Config | None:
        if not self._clc_persistence_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(base_seed.config)
        seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )
        return Config(**seed_config)

    def _staged_mailbox_seed_config(self, clc_seed: Config) -> Config | None:
        """``tcgen05_sched_stage_count = 2`` as a SEED — closing a coverage gap."""
        if not self._clc_persistence_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(clc_seed.config)
        if (
            seed_config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            != Tcgen05PersistenceModel.CLC_PERSISTENT.value
        ):
            return None
        if seed_config.get(TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY) is True:
            return None
        block_sizes = seed_config.get("block_sizes")
        if (
            not isinstance(block_sizes, list)
            or len(block_sizes) < 3
            or block_sizes[0] != TCGEN05_TWO_CTA_BLOCK_M
        ):
            return None
        seed_config[TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY] = (
            TCGEN05_STAGED_WORK_TILE_MAILBOX_SCHED_STAGES
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
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bn_fragment = fragments[1]
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
        narrow_ab = seed_config.get("tcgen05_ab_stages")
        narrow_c = seed_config.get("tcgen05_c_stages")
        if not (
            type(narrow_ab) is int
            and type(narrow_c) is int
            and self.c_stages_fits(
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
                bk=TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
                cluster_m=2,
                ab_stages=narrow_ab,
                c_stages=narrow_c,
                has_source_c=self.exact_shape_aux_kernel_detected,
            )
        ):
            return None
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

    def autotune_seed_configs(self) -> list[Config]:
        seeds: list[Config] = []
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
                    # The STAGED WORK-TILE MAILBOX. Derived from the WIDE-N CLC aux-TMA
                    # seed because that one carries ``bm == 256``; the narrow-N variant
                    # carries ``bm == 256`` too, but seeding both would put two configs
                    # differing in one diagnostic key into the population. See
                    # ``_staged_mailbox_seed_config`` for why the extra envelope terms are
                    # CHECKED there rather than written.
                    staged_mailbox_seed = self._staged_mailbox_seed_config(
                        clc_aux_tma_seed
                    )
                    if staged_mailbox_seed is not None:
                        seeds.append(staged_mailbox_seed)
        return seeds

    def _fix_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        # ── A bm=256 EXPLICIT-EPI-TILE CONFIG IS CtaGroup.TWO ──
        #
        # This stage OWNS ``tcgen05_cluster_m``: the six demotes below plus this promotion
        if (
            self.search_enabled
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
        ):
            explicit_view = self._matmul_config_view(config)
            if explicit_view is not None:
                explicit_block_sizes, explicit_m_index, _, _ = explicit_view
                if explicit_block_sizes[explicit_m_index] == TCGEN05_TWO_CTA_BLOCK_M:
                    config["tcgen05_cluster_m"] = 2
        if not (self.search_enabled and config.get("tcgen05_cluster_m") == 2):
            return
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            # ── SCOPE IS NOT CAPABILITY ──
            #
            # ``cluster_m2_search_constraints is None`` means the SAMPLER does not draw
            if not self._cluster_m2_capability_holds(config):
                config["tcgen05_cluster_m"] = 1
            return
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            config["tcgen05_cluster_m"] = 1
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            config["tcgen05_cluster_m"] = 1
            return
        block_sizes, m_index, n_index, k_index = config_view
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        drawn_bn = block_sizes[n_index]
        layout = config.get(
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
            Tcgen05LayoutStrategy.DEFAULT.value,
        )
        if (
            layout != Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            and isinstance(drawn_bn, int)
            and not isinstance(drawn_bn, bool)
            and drawn_bn <= TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        ):
            block_sizes[n_index] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        else:
            block_sizes[n_index] = TCGEN05_TWO_CTA_BLOCK_N
        # Derived from the SETTLED tile. Every read of this flag below is now reading
        # a predicate over the same ``bn`` that will reach codegen, so a second pass
        # computes the same value and the stage is idempotent by construction.
        bk = block_sizes[k_index]
        if not isinstance(bk, int) or isinstance(bk, bool):
            config["tcgen05_cluster_m"] = 1
            return
        #
        # was inconsistent with the function's own behaviour on every other axis:
        # ``bm`` is snapped, ``bn`` is snapped, and the joint solve re-tunes
        if bk not in TCGEN05_CLUSTER_M2_REPAIR_BLOCK_K_ORDER:
            if not self.cluster_m2_bk_is_valid(bk, constraints):
                config["tcgen05_cluster_m"] = 1
                return
        elif not self.cluster_m2_bk_is_valid(bk, constraints):
            repaired_bk = min(
                (
                    candidate
                    for candidate in TCGEN05_CLUSTER_M2_REPAIR_BLOCK_K_ORDER
                    if self.cluster_m2_bk_is_valid(candidate, constraints)
                ),
                key=lambda candidate: (abs(candidate - bk), -candidate),
                default=None,
            )
            if repaired_bk is None:
                config["tcgen05_cluster_m"] = 1
                return
            block_sizes[k_index] = repaired_bk
            bk = repaired_bk
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        # unconditionally, at the top of ``fix_search_config`` -- the key is incompatible
        # cluster_m=2 ones, so a cluster_m-scoped pop was hiding a subset of the failures.
        # ── The fp8 small-grid family is a SCOPE EXCLUSION, not a special case ──
        sampled_bm = block_sizes[m_index]
        fp8_small_grid_family = (
            constraints.allow_fp8_small_grid
            and not edge_k_tail_family
            and isinstance(sampled_bm, int)
            and not isinstance(sampled_bm, bool)
            and sampled_bm <= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
        )
        if fp8_small_grid_family:
            # ``bm`` is LEGALITY and stays pinned: ``_tcgen05_use_2cta_instrs`` admits
            # ``bm == 256`` at any 16-bit dtype but ``bm == 128`` only for fp8, and at
            # bf16 that config point belongs to the legacy clustered ``CtaGroup.ONE``
            # family. This is a family-ownership pin, not a tuned value. Note both ``bm``
            # values already COMPETE on an fp8 shape: this branch keys on the DRAWN ``bm``,
            # so a drawn 256 falls through to the main path below and gets the 256-wide
            # 2-CTA tile (verified on fp8 512x2048x4096: drawn ``bm=256`` exits as
            # ``[256, {64,128,256}, *]`` with ``use_2cta=True``).
            block_sizes[m_index] = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
            # the SETTLE above, which is the same rule the non-fp8 arm gets: ``bn <= 128``
            # snaps to 128, otherwise 256. So the fp8 arm's band goes ``{128}`` ->
            # ``{128, 256}`` and stops being a special case.
            return
        block_sizes[m_index] = TCGEN05_TWO_CTA_BLOCK_M
        # Block-N shaping for the surviving cluster_m=2 candidates. A CtaGroup.TWO
        # matmul has exactly two validated N tiles: 128 (a 256x128 output tile)
        # and 256. Both are hardware-legal because the 2-CTA MMA decision keys

    def prepare_normalization(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        source_m_tile_key = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
        source_m_tile = config.get(source_m_tile_key)
        if source_m_tile_key in config and (
            type(source_m_tile) is not int
            or source_m_tile not in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
            or config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            != TCGEN05_GROUPED_MODE_WORKLIST_NM
        ):
            if fix_invalid:
                config.pop(source_m_tile_key)
            else:
                raise InvalidConfig(
                    f"{source_m_tile_key} requires "
                    f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                    f"{TCGEN05_GROUPED_MODE_WORKLIST_NM!r} and one of "
                    f"{TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES}, got "
                    f"{source_m_tile!r}"
                )
        signature_key = TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
        if signature_key in config:
            try:
                parse_tcgen05_grouped_static_problem_signature(config[signature_key])
            except InvalidConfig:
                if fix_invalid:
                    config.pop(signature_key)
                else:
                    raise
            else:
                if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) not in (
                    TCGEN05_GROUPED_MODE_STATIC,
                    TCGEN05_GROUPED_MODE_DIRECT,
                    TCGEN05_GROUPED_MODE_DYNAMIC,
                ):
                    if fix_invalid:
                        config.pop(signature_key)
                    else:
                        raise InvalidConfig(
                            f"{signature_key} requires "
                            f"{TCGEN05_GROUPED_MODE_CONFIG_KEY} to be "
                            f"{TCGEN05_GROUPED_MODE_STATIC!r}, "
                            f"{TCGEN05_GROUPED_MODE_DIRECT!r} or "
                            f"{TCGEN05_GROUPED_MODE_DYNAMIC!r}"
                        )
        reserved_sms_key = TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
        reserved_sms = config.get(reserved_sms_key)
        if reserved_sms_key in config and (
            type(reserved_sms) is not int
            or reserved_sms < 0
            or reserved_sms > TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
        ):
            if fix_invalid:
                config.pop(reserved_sms_key)
            else:
                raise InvalidConfig(
                    f"{reserved_sms_key} must be an "
                    f"integer in [0, {TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX}], "
                    f"got {reserved_sms!r}"
                )
        if reserved_sms == 0:
            config.pop(reserved_sms_key, None)
        if (
            fix_invalid
            and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) not in TCGEN05_GROUPED_MODES
        ):
            config.pop(TCGEN05_GROUPED_MODE_CONFIG_KEY, None)

    @staticmethod
    def _uses_grouped_static_reserved_sms(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) in TCGEN05_GROUPED_DYNAMIC_MODES
            and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.STATIC_PERSISTENT.value
        )

    def _normalize_grouped_static_reserved_sms(
        self,
        config: dict[str, object],
    ) -> None:
        reserved_sms_key = TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
        if reserved_sms_key not in config:
            return
        if not self._uses_grouped_static_reserved_sms(config):
            config.pop(reserved_sms_key, None)

    def allow_ab_stages_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        assert dtype_bytes > 0, "dtype_bytes must be positive"
        if self._matmul_block_indices() is None:
            self.ab_stages_search_constraints = None
            return
        budget_bytes = self.per_cta_ab_smem_budget_bytes(device)
        if budget_bytes <= 0:
            self.ab_stages_search_constraints = None
            return
        self.ab_stages_search_constraints = Tcgen05AbStagesSearchConstraints(
            dtype_bytes=dtype_bytes,
            per_cta_smem_budget_bytes=budget_bytes,
            per_cta_smem_capacity_bytes=self.per_cta_smem_capacity_bytes(device),
        )

    def allow_deep_direct_entry_validation(self, *, device: torch.device) -> None:
        self.deep_direct_entry_validation_enabled = (
            self._direct_entry_k_block_index() is not None
            and self.per_cta_ab_smem_budget_bytes(device) > 0
        )

    @staticmethod
    def per_cta_smem_capacity_bytes(device: torch.device) -> int:
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(device)
        optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
        return max(props.shared_memory_per_block, optin_shared)

    @classmethod
    def per_cta_smem_budget_bytes(cls, device: torch.device) -> int:
        device_cap = cls.per_cta_smem_capacity_bytes(device)
        return max(0, device_cap - TCGEN05_AB_STAGES_RESERVED_SMEM_BYTES)

    @classmethod
    def per_cta_ab_smem_budget_bytes(cls, device: torch.device) -> int:
        device_cap = cls.per_cta_smem_capacity_bytes(device)
        if device_cap < TCGEN05_AB_STAGES_MIN_DEVICE_SMEM_OPTIN:
            return 0
        # Keep a fixed headroom reservation: CuTe's raw opt-in limit does not
        # include every barrier/runtime byte the 3-stage AB pipeline needs.
        return device_cap - TCGEN05_AB_STAGES_RESERVED_SMEM_BYTES

    def ab_stages_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int = 3,
    ) -> bool:
        constraints = self.ab_stages_search_constraints
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

    def c_stages_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
        c_stages: int,
        has_source_c: bool,
    ) -> bool:
        # C-store ring. Reuse the ``tcgen05_ab_stages=3`` SMEM-budget envelope
        # (same dtype_bytes + per-CTA budget after the non-AB reservation) and
        # require AB + C to fit together. This is the gate that keeps a deeper
        # C ring (``tcgen05_c_stages=4``) out of the ab=3 regime, where AB+C
        # overshoots the 232 KB B200 cap and ptxas raises a raw
        # ``too much shared`` error during tuning. The C bytes use the REAL
        constraints = self.ab_stages_search_constraints
        if constraints is None:
            return False
        if cluster_m not in (1, 2):
            return False
        if bm <= 0 or bn <= 0 or bk <= 0:
            return False
        if ab_stages <= 0 or c_stages <= 0:
            return False
        ab_bytes = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=ab_stages,
            cluster_m=cluster_m,
        )
        # The epilogue processes the full per-CTA output tile (bm, bn); unlike
        # the AB operands it is NOT split across the cluster, so the C-ring
        # bytes do not depend on cluster_m. ``elem_width`` is the operand /
        # output element width in bits (the validated families are uniform
        # 16-bit). ``elem_width_c`` is None for no-source-C (plain) kernels so
        # the helper picks the smaller no-source-C epilogue tile.
        elem_width = constraints.dtype_bytes * 8
        epi_tile_m, epi_tile_n = tcgen05_default_epilogue_tile_size(
            bm,
            bn,
            elem_width_d=elem_width,
            elem_width_c=elem_width if has_source_c else None,
        )
        c_bytes = tcgen05_c_smem_bytes_per_cta(
            epi_tile_m=epi_tile_m,
            epi_tile_n=epi_tile_n,
            dtype_bytes=constraints.dtype_bytes,
            c_stages=c_stages,
        )
        capacity = constraints.per_cta_smem_capacity_bytes
        if capacity <= 0:
            # No recorded capacity (older constraint record / non-CUDA host): fall back to
            # the budget, which fails closed the way every other gate here does.
            return ab_bytes + c_bytes <= constraints.per_cta_smem_budget_bytes
        return ab_bytes + c_bytes <= capacity

    def _fix_aux_producer_depth_feasibility_search_config(
        self, config: dict[str, object]
    ) -> None:
        """A productive aux producer warp cannot coexist with a deep AB pipeline."""
        if not (self.search_enabled and self.aux_kernel_detected):
            return
        if (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            return
        ab_stages = config.get("tcgen05_ab_stages")
        if (
            type(ab_stages) is int
            and ab_stages > TCGEN05_AUX_PRODUCER_WARP_MAX_AB_STAGES
            and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
        ):
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0

    @staticmethod
    def _grouped_dynamic_deep_config_matches(config: dict[str, object]) -> bool:
        block_sizes = config.get("block_sizes")
        defaults: dict[str, object] = {
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_acc_stages": 2,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: Tcgen05LayoutStrategy.DEFAULT.value,
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 0,
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: 0,
            TCGEN05_WARP_SPEC_STORE_WARPS_KEY: 0,
        }
        ab_stages = config.get("tcgen05_ab_stages")
        c_stages = config.get("tcgen05_c_stages", 2)
        return (
            type(ab_stages) is int
            and type(c_stages) is int
            and (ab_stages, c_stages) in TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES
            and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            in TCGEN05_GROUPED_DYNAMIC_MODES
            and isinstance(block_sizes, list)
            and block_sizes[:3] == [128, 64, 64]
            and config.get("pid_type") == TCGEN05_TWO_CTA_SEED_PID_TYPE
            and all(
                config.get(key, expected) == expected
                for key, expected in defaults.items()
            )
            and all(config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
        )

    def _grouped_worklist_nm_deep_ab_config_matches(
        self, config: dict[str, object], ab_stages: object
    ) -> bool:
        block_sizes = config.get("block_sizes")
        if not (
            type(ab_stages) is int
            and 4 <= ab_stages <= 7
            and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            == TCGEN05_GROUPED_MODE_WORKLIST_NM
            and config.get("tcgen05_cluster_m") == 2
            and config.get("tcgen05_cluster_n", 1) == 1
            and config.get("tcgen05_acc_stages", 2) == 2
            and config.get("tcgen05_c_stages", 2) == 2
            and isinstance(block_sizes, list)
            and block_sizes[:3] == [TCGEN05_TWO_CTA_BLOCK_M, 128, 64]
        ):
            return False
        if self.ab_stages_search_constraints is None:
            # Fixed configs can be normalized before their input device is known.
            # CuTe MMA selection applies the real target's SMEM limit at codegen.
            return True
        return self.ab_stages_fits(
            bm=block_sizes[0],
            bn=block_sizes[1],
            bk=block_sizes[2],
            cluster_m=2,
            ab_stages=ab_stages,
        )

    def grouped_dynamic_stages_fit_for_target(
        self,
        *,
        dtype_bytes: int,
        output_dtype_bytes: int,
        device: torch.device,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
        c_stages: int,
    ) -> bool:
        if dtype_bytes != 2 or output_dtype_bytes <= 0:
            return False
        if (bm, bn, bk, cluster_m) != (128, 64, 64, 1):
            return False
        if (ab_stages, c_stages) not in TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES:
            return False
        cap_bytes = self.per_cta_smem_capacity_bytes(device)
        if cap_bytes <= 0:
            return False
        elem_width = output_dtype_bytes * 8
        epi_tile_m, epi_tile_n = tcgen05_default_epilogue_tile_size(
            bm,
            bn,
            elem_width_d=elem_width,
            elem_width_c=None,
        )
        ab_bytes = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=dtype_bytes,
            ab_stages=ab_stages,
            cluster_m=cluster_m,
        )
        c_bytes = tcgen05_c_smem_bytes_per_cta(
            epi_tile_m=epi_tile_m,
            epi_tile_n=epi_tile_n,
            dtype_bytes=output_dtype_bytes,
            c_stages=c_stages,
        )
        return (
            ab_bytes + c_bytes + TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES
            <= cap_bytes
        )

    def _fix_c_stages_search_config(self, config: dict[str, object]) -> None:
        # ring. ``tcgen05_c_stages`` is an ``EnumFragment((2, 4))`` knob, so the
        # autotuner can SAMPLE c=4 independently of any projection — a directly
        if not self.search_enabled:
            return
        if config.get("tcgen05_c_stages") != TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES:
            return
        if not self._is_default_layout_config(config):
            return
        if self.aux_kernel_detected and self._has_any_matmul_fact_edge_tile(config):
            return
        # Fail CLOSED: the ``(2, 4)`` c-stages fragment is offered on every
        # device, but with no SMEM budget recorded (non-B200 / CPU host) we
        # cannot prove c=4 fits — demote rather than leave the ptxas-overflow
        # window open. ``c_stages_fits`` itself returns False when constraints
        # are absent, so a single ``not c_stages_fits`` check covers both the
        # over-budget and the no-budget arms.
        config_view = self._matmul_config_view(config)
        if config_view is None:
            config["tcgen05_c_stages"] = 2
            return
        block_sizes, m_index, n_index, k_index = config_view
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        ab_stages = cast("int", config.get("tcgen05_ab_stages", 2))
        if not self.c_stages_fits(
            bm=cast("int", block_sizes[m_index]),
            bn=cast("int", block_sizes[n_index]),
            bk=cast("int", block_sizes[k_index]),
            cluster_m=cluster_m,
            ab_stages=ab_stages,
            c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
            has_source_c=self.aux_kernel_detected,
        ):
            config["tcgen05_c_stages"] = 2

    @staticmethod
    def _is_default_layout_config(config: dict[str, object]) -> bool:
        # DEFAULT-layout role-local, ANY tile. This is where the bare-AB and AB+C
        # per-CTA SMEM models apply, because they key on (bm, bn, bk, cluster_m)
        # and the DEFAULT epilogue tile. EXPLICIT_EPI_TILE configs use a separate
        # (128, 32) epilogue tile with its own admission
        # (``_validate_direct_entry_ab_stage_envelope``) and a different store
        # topology, so the AB+C model is not calibrated for them; an absent layout
        # key defaults to DEFAULT.
        return (
            config.get(
                TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
                Tcgen05LayoutStrategy.DEFAULT.value,
            )
            == Tcgen05LayoutStrategy.DEFAULT.value
        )

    def _is_default_layout_full_tile_config(self, config: dict[str, object]) -> bool:
        # The canonical 256x256 DEFAULT-layout role-local tile, where the C-ring
        # AB+C SMEM model is calibrated. EXPLICIT_EPI_TILE configs use a separate
        # tile/admission and are excluded; an absent layout key defaults to
        # DEFAULT.
        if not self._is_default_layout_config(config):
            return False
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return False
        block_sizes, m_index, n_index, _ = config_view
        return (
            block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
            and block_sizes[n_index] == TCGEN05_TWO_CTA_BLOCK_N
        )

    @staticmethod
    def _get_dtype_ab_stages_hard_cap(dtype_bytes: int) -> int:
        """Get hardware-validated maximum ab_stages for a dtype."""
        if dtype_bytes <= 0:
            return 0
        if dtype_bytes == 1:  # FP8
            return 12
        if dtype_bytes == 2:  # FP16/BF16
            return 8
        # FP32 or larger
        return 3

    def max_ab_stages_that_fit(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        hard_cap: int | None = None,
    ) -> int:
        """Compute maximum ab_stages that fits in per-CTA SMEM budget."""
        constraints = self.ab_stages_search_constraints
        if constraints is None or bm <= 0 or bn <= 0 or bk <= 0:
            return 0
        if cluster_m not in (1, 2):
            return 0

        # Calculate SMEM cost for ab_stages=1 (baseline)
        bytes_per_stage = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=1,
            cluster_m=cluster_m,
        )

        # Edge cases: invalid calculation or even ab_stages=1 doesn't fit
        if bytes_per_stage <= 0:
            return 0
        if bytes_per_stage > constraints.per_cta_smem_budget_bytes:
            return 0

        # Direct calculation: SMEM usage scales linearly with ab_stages
        # Solve: N * bytes_per_stage <= budget
        max_from_budget = constraints.per_cta_smem_budget_bytes // bytes_per_stage

        # Apply hard cap (dtype-specific default if not provided)
        if hard_cap is None:
            hard_cap = self._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes)

        # Return clamped value: at least 1, at most hard_cap or budget limit
        return max(1, min(max_from_budget, hard_cap))

    def _fix_ab_stages_search_config(self, config: dict[str, object]) -> None:
        # Budget-aware deep-AB admission for the lifted ``for_search`` cap (see
        # Mirror ``_fix_c_stages_search_config`` (fail-CLOSED, cast-based): on ANY
        # DEFAULT-layout tile, demote a directly-sampled ab>=3 to the deepest depth
        if not self.search_enabled:
            return
        ab_stages = config.get("tcgen05_ab_stages")
        if type(ab_stages) is not int or ab_stages < 1:
            return
        #   ``if not self._is_default_layout_config(config): return``
        # which excluded the ``explicit_epi_tile`` family because the AB+C byte model is
        # calibrated for the DEFAULT epilogue tile. That exclusion was only safe while
        config_view = self._matmul_config_view(config)
        if config_view is None:
            # ``min``, not a bare ``= 2``: with no readable tile we cannot prove the
            # drawn depth fits, so fail closed to the conservative depth — but this
            # stage must never RAISE ``tcgen05_ab_stages``. A bare write would take
            # a drawn ab=1 UP to 2, which falsifies the monotonicity invariant that
            # ``_fix_with_scheduler_search_config``'s ``>= 3`` guard relies on to be
            # safe at its earlier position (see the comment there). It cannot cross
            config["tcgen05_ab_stages"] = min(ab_stages, 2)
            return
        block_sizes, m_index, n_index, k_index = config_view
        bm = cast("int", block_sizes[m_index])
        bn = cast("int", block_sizes[n_index])
        bk = cast("int", block_sizes[k_index])
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if self.exact_shape_aux_kernel_detected:
            # Real source-C present: require AB + the (128, 64) C ring to fit
            # together at the sampled depth. ``c_stages_fits`` fails CLOSED when no
            # SMEM budget is recorded (non-B200 / CPU host), so the over-budget and
            # no-budget arms are both covered.
            c_stages = cast("int", config.get("tcgen05_c_stages", 2))

            def _fits_at(ab: int, bk_: int) -> bool:
                return self.c_stages_fits(
                    bm=bm,
                    bn=bn,
                    bk=bk_,
                    cluster_m=cluster_m,
                    ab_stages=ab,
                    c_stages=c_stages,
                    has_source_c=True,
                )
        else:
            # Plain / rowvec-bias store (no source-C ring): the bare-AB gate is the
            # calibrated admission — the small no-source-C epilogue D ring rides the
            # non-AB reservation. ``ab_stages_fits`` returns False with no
            # budget recorded, so this also fails CLOSED.
            def _fits_at(ab: int, bk_: int) -> bool:
                return self.ab_stages_fits(
                    bm=bm, bn=bn, bk=bk_, cluster_m=cluster_m, ab_stages=ab
                )

        def _fits(ab: int) -> bool:
            return _fits_at(ab, bk)

        # Demote to the deepest depth that fits. The floor is 1 when a real budget
        # is recorded — ab=1 is a legal depth the fragment offers, and a tile that
        # does not fit even at 2 (e.g. [256,256,128] cluster_m=1 = 262144 B at ab=2,
        # fitting only at ab=1) must still be
        # brought under the cap rather than left to fail at ptxas. ab=1 is an
        # unpipelined AB ring and therefore slow, but the point is that it is LEGAL:
        floor = 1 if self.ab_stages_search_constraints is not None else 2
        while ab_stages > floor and not _fits(ab_stages):
            ab_stages -= 1
        config["tcgen05_ab_stages"] = ab_stages
        if not _fits(ab_stages):
            candidate_bk = bk
            while candidate_bk > TCGEN05_MIN_CONCEDED_BLOCK_K:
                candidate_bk //= 2
                if _fits_at(ab_stages, candidate_bk):
                    block_sizes[k_index] = candidate_bk
                    break
        # ⚠ HISTORY, because the ``bk`` concession above was once REMOVED and the
        #
        # An earlier version shrank ``block_sizes[k]`` here and was reverted, on two

    def _fix_with_scheduler_search_config(self, config: dict[str, object]) -> None:
        """Strategy <-> warp-count repair. DO NOT FOLD THE TWO WARP KEYS TOGETHER.

        determined by the strategy -- biconditional -- so deriving it is correct
        * ``c_input_warps`` is an **accept set**
        Fold them and the free axis silently disappears: any helper that derives a
        populated, with no spare warp, which is why their accept set is ``{0}`` and
        """
        # ── FLAT-ROLE WARP TOPOLOGY, UNCONDITIONAL AND FIRST ──
        #
        # made the warp group have TWO writers in two different stages. They live here
        if config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True:
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
            config[TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY] = 1
            config["tcgen05_num_epi_warps"] = 4
        if not (self.search_enabled and self.aux_kernel_detected):
            return
        strategy = config.get(TCGEN05_STRATEGY_CONFIG_KEY)
        scheduler_warps = config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY)
        c_input_warps = config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY)
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
            # ⚠ THE ``ab >= 3 => c_input_warps = 0`` GUARD MOVED OUT OF THIS STAGE
            # which runs AFTER the AB depth walk. Here it read ``tcgen05_ab_stages``
            # before ``_fix_ab_stages_search_config`` had settled it, which cost this

    def _fix_aux_tma_search_config(self, config: dict[str, object]) -> None:
        """Complete the aux-TMA regime a config REQUESTED, or decline it.

        | ``c_input_warps = 1`` | **required by the determinator** | ``tma`` with no producer warp raises ``BackendUnsupported`` ("requires a productive aux producer warp") — nobody issues the copy. NOT a derivation: under ``simt`` the key is a real axis (see below), so this write is scoped to the ``tma`` arm alone |
        | ``ab_stages = min(ab, CAP)`` | **required by the determinator** | the aux SMEM ring + AB pipeline overshoot the 232 KB B200 cap past the cap depth; ``cute_mma.py`` raises. A ``min`` so it can only ever LOWER the key |
        """
        if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA:
            return
        if self._aux_tma_request_is_satisfiable(config):
            # Supply exactly the two values ``aux_load_mode=tma`` cannot run
            # without. Both are unconditional inside this arm: writing the value
            # the request requires is completing the request, not steering it.
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
            ab_stages = config.get("tcgen05_ab_stages")
            if type(ab_stages) is int:
                config["tcgen05_ab_stages"] = min(
                    ab_stages, TCGEN05_AUX_PRODUCER_WARP_MAX_AB_STAGES
                )
            return
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT

    def _aux_tma_request_is_satisfiable(self, config: dict[str, object]) -> bool:
        """Can an ``aux_load_mode=tma`` request be completed on *config*?

        Every clause is a **precondition** — a fact or a knob this stage refuses to
        write. False means the request must be declined (demoted to SIMT), because
        no value this stage is willing to supply can rescue it.
        """
        # PRECONDITION-FACT: the family. All four of §3's eligibility terms
        # (``exact_shape_aux_kernel_detected``, no leading passthrough, cm2
        # constraints exist, edge-vs-full-tile) already live inside this one
        # accessor; do not re-implement them here.
        if not self._aux_tma_search_enabled():
            return False
        # Checked, never written: the explicit epilogue tile is incompatible with a
        # productive aux producer warp (cute_mma.py raises on that pairing).
        if (
            config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
        ):
            return False
        # PRECONDITION-KNOB: only ROLE_LOCAL_WITH_SCHEDULER's 7-role/8-launched warp
        # shape has the inert padding slot the aux producer occupies. The other two
        # strategies are fully-populated 6-warp shapes, so
        if (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            return False
        # PRECONDITION-KNOB: ``scheduler_warps`` is strategy-DETERMINED
        # (``_STRATEGY_REQUIRED_SCHEDULER_WARPS``, read as an equality test in
        # ``validate_tcgen05_strategy_invariants``), and stage 5 already derives it
        # on the search path. Checked rather than written so this stage stays out of
        # the warp-topology business; an explicit config that disagrees with its own
        # strategy is declined rather than silently repaired here.
        if config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) != 1:
            return False
        # PRECONDITION-KNOB: ``c_input_warps`` and ``store_warps`` share the SINGLE
        # padding slot, so ``c_input + store > 1`` is rejected by
        # ``validate_tcgen05_strategy_invariants`` (``role_warp_count`` would sum to
        # 9, rounding the launch to 12 warps / 384 threads). A drawn config cannot
        # reach this — ``strategy_autotune_fragments`` pins ``store_warps`` to
        # ``(0,)`` — but this stage also runs on explicit configs and cache
        # transfers, where a store warp is expressible. With one live, the aux
        # producer has no slot, so the request is declined rather than made illegal.
        if config.get(TCGEN05_WARP_SPEC_STORE_WARPS_KEY, 0) != 0:
            return False
        #
        # Evaluated as a PRECONDITION of the write arm rather than as a trailing
        # demotion, which is a deliberate change of position. As a trailing clause
        return config.get("tcgen05_cluster_n", 1) == 1

    @staticmethod
    def _is_with_scheduler_c_input_config(config: dict[str, object]) -> bool:
        """The full aux-TMA producer trio, as a READ-ONLY predicate."""
        return (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) == 1
            and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
        )

    def _clc_persistence_search_enabled(self) -> bool:
        """CLC search is the sm100+ slice of the aux-TMA edge+K-tail gate.

        Cycle 46 widened ``_aux_tma_search_enabled`` to also admit the full-tile
        cluster_m=2 family, but the CLC-persistent perf knobs and validated
        candidate shape are still scoped to ``_aux_tma_edge_search_enabled``.
        """
        if not self._aux_tma_edge_search_enabled():
            return False
        capability = self.config_spec.target_device_capability
        if capability is None:
            return False
        return capability[0] >= 10 and "flat" in self.allowed_pid_types

    def _clc_persistence_codegen_supported(self) -> bool:
        """Can a CLC-persistent kernel be EMITTED on this shape at all?

        (``strategies.py``'s persistence arch gate raises below it).
        """
        if self.cluster_m2_search_constraints is None:
            return False
        capability = self.config_spec.target_device_capability
        if capability is None:
            return False
        # ``"flat" in allowed_pid_types`` is the FORCE-PERSISTENT DETECTOR, not a claim
        # that CLC wants a flat pid. ``autotune_force_persistent`` (or an initialized
        # process group) disallows ``flat``/``xyz`` in ``compile_environment.py``, and
        # the persistence axis this gate opens is built from
        # ``derive_persistence_model_from_pid_type``, which cannot offer
        # ``NON_PERSISTENT`` once those are gone -- so the axis must stay hidden there.
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

    # the two writes it gated in ``_fix_cluster_m2_search_config``: the narrow K pin and the
    # narrow ``acc_stages``/``l2_groupings`` arm of the L cascade. It had no other caller.

    def implicit_default_keys_to_preserve(self, config: dict[str, object]) -> set[str]:
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

    def _validate_direct_entry_ab_stage_envelope(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        ab_stages = config.get("tcgen05_ab_stages")
        if type(ab_stages) is not int or ab_stages <= 3:
            return
        if self._grouped_dynamic_deep_config_matches(config):
            return
        if self._grouped_worklist_nm_deep_ab_config_matches(config, ab_stages):
            return
        # ab>3 is only valid on the TVM-FFI direct-entry path, and only for the
        # (bk, ab, c) stage tuples the direct-entry codegen accepts (bk=64
        # admits (ab=6, c=4)). Everything else clamps (or rejects) to ab=3.
        block_sizes = config.get("block_sizes")
        k_block_index = self._direct_entry_k_block_index()
        bk = (
            block_sizes[k_block_index]
            if isinstance(block_sizes, list)
            and k_block_index is not None
            and k_block_index < len(block_sizes)
            else None
        )
        # ``c_stages`` is read only to confirm the key is present and well-typed:
        # its VALUE no longer gates admission (that was the enumerated ``(ab, c)``
        # pairing), but a config missing a valid c_stages is still not a
        # well-formed direct-entry config.
        c_stages = config.get("tcgen05_c_stages")
        if (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and isinstance(bk, int)
            and not isinstance(bk, bool)
            and bk in TCGEN05_DIRECT_ENTRY_LEGAL_BK
            and type(c_stages) is int
        ):
            config_view = self._matmul_config_view(config)
            if config_view is not None:
                view_block_sizes, m_index, n_index, k_index = config_view
                if self.ab_stages_fits(
                    bm=cast("int", view_block_sizes[m_index]),
                    bn=cast("int", view_block_sizes[n_index]),
                    bk=cast("int", view_block_sizes[k_index]),
                    cluster_m=cast("int", config.get("tcgen05_cluster_m", 1)),
                    ab_stages=ab_stages,
                ):
                    return
        # A deeper AB pipeline than the bf16-tuned cap of 3 is admitted whenever
        # the per-CTA AB SMEM fits the budget, for two families:
        #   * FP8 (1-byte) operands, any tile — lets Helion emit the same
        #     deeply-pipelined CtaGroup.TWO kernel CUTLASS uses for fp8
        constraints = self.ab_stages_search_constraints
        is_fp8 = constraints is not None and constraints.dtype_bytes == 1
        layout = config.get(
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
            Tcgen05LayoutStrategy.DEFAULT.value,
        )
        is_16bit_default = (
            constraints is not None
            and constraints.dtype_bytes == 2
            and config.get("tcgen05_cluster_m") in (1, 2)
            and layout == Tcgen05LayoutStrategy.DEFAULT.value
        )
        if is_fp8 or is_16bit_default:
            config_view = self._matmul_config_view(config)
            cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
            if config_view is not None:
                block_sizes, m_index, n_index, k_index = config_view
                fit_max = self.max_ab_stages_that_fit(
                    bm=cast("int", block_sizes[m_index]),
                    bn=cast("int", block_sizes[n_index]),
                    bk=cast("int", block_sizes[k_index]),
                    cluster_m=cluster_m,
                )
                if fit_max > 0 and ab_stages <= fit_max:
                    return
                if fix_invalid and fit_max > 0:
                    config["tcgen05_ab_stages"] = fit_max
                    return
        if fix_invalid:
            config["tcgen05_ab_stages"] = 3
            return
        raise InvalidConfig(
            "tcgen05_ab_stages > 3 is only supported by the validated "
            "TVM-FFI direct-entry path, a 16-bit cluster_m=2 DEFAULT-layout "
            "tile within the SMEM budget, or fp8 within the SMEM budget"
        )

    def _is_validated_clc_persistence_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if not self._clc_persistence_codegen_supported():
            return False
        # ``cluster_m == 2`` ONLY -- not the aux/edge family tag this used to read
        # through ``_is_validated_cluster_m2_edge_search_candidate``. CLC needs the
        # clustered shape (the query publishes to peer CTAs, and the staged-mailbox
        # assert wants ``cluster_m > 1``); it does not need an aux tensor or an output
        # edge. Verified bit-exact with CLC emitted on a plain full-tile 4096^3 matmul.
        if config.get("tcgen05_cluster_m") != 2:
            return False
        if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
            return False
        if config.get("tcgen05_cluster_n", 1) != 1:
            return False
        return (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) == 1
        )

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
        if key == TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY:
            return True, 0
        if key == TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY:
            # The search surface for this key draws False only
            # (``EnumFragment((True, False), search_choices=(False,))``), so an
            # absent key can only come from a config that did not go through the
            # search: a user-supplied config, a partial seed, or a cache entry.
            # Absence there means "no FFI promotion requested" — matching the
            # validation-view default and the special case at
            return True, False
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

    def _matmul_fact_has_double_edge_output(
        self, config: dict[str, object], *, fact_index: int
    ) -> bool:
        """``M % bm != 0`` AND ``N % bn != 0`` on the DRAWN tile.

        raise BackendUnsupported("tcgen05 SIMT edge epilogue double-edge output
        Two things this deliberately does NOT do, both because codegen does not:
        """
        # Every unreadable case answers the same way -- an axis we cannot read is an axis
        # we cannot PROVE partial, and this predicate gates a DEMOTION, so it fails toward
        # NOT demoting. That makes the whole body one comprehension over the two axes.
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return False
        fact = self.config_spec.matmul_facts[fact_index]

        def axis_is_partial(static_size: int | None, block_id: int | None) -> bool:
            index = self._config_block_index(block_id)  # None if absent/unregistered
            if static_size is None or index is None or index >= len(block_sizes):
                return False
            block_size = block_sizes[index]
            if type(block_size) is not int or block_size <= 0:
                return False  # ``type(...) is int`` also rejects bool
            return static_size % block_size != 0

        return axis_is_partial(fact.static_m, fact.m_block_id) and axis_is_partial(
            fact.static_n, fact.n_block_id
        )

    def _has_any_matmul_fact_double_edge_output(
        self, config: dict[str, object]
    ) -> bool:
        return any(
            self._matmul_fact_has_double_edge_output(config, fact_index=i)
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

    def aux_edge_seed_shape_eligible(self) -> bool:
        """Shape-structural eligibility for the aux-edge SEED (§2.3)."""
        if not self.aux_kernel_detected:
            return False
        probe: dict[str, object] = {
            "block_sizes": self._aux_edge_seed_block_sizes(),
        }
        if not isinstance(probe["block_sizes"], list):
            return False
        return self._has_any_matmul_fact_edge_tile(probe)

    def _aux_edge_seed_block_sizes(self) -> list[int] | None:
        """The seed's tile."""
        return self._matmul_seed_block_sizes(
            bm=TCGEN05_ONE_CTA_MAX_BLOCK_M,
            bn=TCGEN05_AUX_EDGE_SEED_BLOCK_N,
            bk=TCGEN05_AUX_EDGE_SEED_BLOCK_K,
        )

    def aux_edge_seed_config(self) -> Config | None:
        """The aux-edge PERF regime, as a competing seed instead of a projection."""
        block_sizes = self._aux_edge_seed_block_sizes()
        if block_sizes is None:
            return None
        seed: dict[str, object] = {
            "block_sizes": block_sizes,
            TCGEN05_STRATEGY_CONFIG_KEY: (Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 0,
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: 0,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 4,
            "tcgen05_ab_stages": 2,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "l2_groupings": [1],
        }
        if self.config_spec.indexing.length > 0:
            # Kernel-wide TMA addressing off, matching the regime this seed
            # encodes. Unlike the stage's blanket rewrite this is a SEED value on
            # a fresh config, so it strands nothing: the search still draws every
            # other ``indexing`` combination.
            seed["indexing"] = ["pointer"] * self.config_spec.indexing.length
        try:
            return Config(**seed)  # type: ignore[arg-type]
        except (InvalidConfig, ValueError, TypeError, KeyError):
            return None

    @staticmethod
    def _set_aux_edge_prefix(
        config: dict[str, object], *, allow_with_scheduler: bool
    ) -> None:
        """The aux-edge LEGALITY prefix — one entry point, hoisted (§2.3).

        MONOLITHIC with both warp counts 0, which is the only accept set for that
        """
        if allow_with_scheduler and (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
            # the entry in ``tcgen05_two_cta_edge_k_tail_seed_overrides()``). The two
            # had to go as a PAIR and the ordering is why:
        else:
            # ab=2 is the aux-TMA producer's validated stage depth for this family
            # (the aux SMEM ring forces ab<=2 under the 232 KB B200 cap; the
            # cycle-86/88 measurements ran at ab=2). ``_c_input_seed_config``
            # already emits exactly this regime as a seed, so the shape envelope
            # validated above is sufficient — no extra SMEM-fit check is needed.
            config[TCGEN05_STRATEGY_CONFIG_KEY] = (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
            )
            config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        # aux-edge PERF regime now enters the population as ``aux_edge_seed_config``,
        # which already carries ``tcgen05_ab_stages=2``, so raising a drawn ab=1 was
        # redundant steering rather than legality: codegen has no ``ab >= 2``

    def _is_validated_cluster_m2_edge_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if config.get("tcgen05_cluster_m") != 2:
            return False
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            return False
        return constraints.allow_edge_k_tail_family

    # ``_fix_aux_tma_search_config``'s arm-4 tile-shape envelope (8 clauses:
    # cluster_m, constraints/edge-family, pid_type, cluster_n, config_view,
    # bm == 256, bn == 256, bk validity) and that arm-4 call site was its ONLY

    def _fix_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        #
        #     and config.get("pid_type") in {"persistent_blocked", "persistent_interleaved"}
        if not (self.search_enabled and config.get("tcgen05_cluster_m", 1) == 1):
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return
        block_sizes, m_index, _, _ = config_view
        # TWO INDEPENDENT legality clamps. They must NOT sit in an if/else: they
        # constrain different keys for different reasons, and either one being
        # skipped is a defect.
        bm = block_sizes[m_index]
        if isinstance(bm, int) and not isinstance(bm, bool):
            block_sizes[m_index] = min(bm, TCGEN05_ONE_CTA_MAX_BLOCK_M)
        # (2) pid_type: ``persistent_interleaved`` stays in the flat enum so
        #     cluster_m=2 edge-family samples can encode; cluster_m=1 samples from
        #     the same surface must use the validated flat edge fallback.
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is not None
            and constraints.allow_edge_k_tail_family
            and config.get("pid_type")
            in {"persistent_blocked", "persistent_interleaved"}
        ):
            # persistent_interleaved stays in the flat enum so cluster_m=2
            # edge-family samples can encode; cluster_m=1 samples from the
            # same surface must use the validated flat edge fallback.
            config["pid_type"] = "flat"

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
        allow_cluster_m2_fp8_small_grid: bool = False,
        ab_stages_dtype_bytes: int | None = None,
        ab_stages_device: torch.device | None = None,
    ) -> None:
        # Keep the default tcgen05 surface to combinations with runtime
        # coverage. Some unvalidated combinations fail loudly at CuTe
        # construction/launch, while diagnostic pipeline modes can compile and
        # intentionally produce wrong output.
        if allow_cluster_m2_edge_k_tail_family:
            assert allow_cluster_m2_search, (
                "cluster_m=2 edge/K-tail admission requires cluster_m=2 search"
            )
        if allow_cluster_m2_fp8_small_grid:
            assert allow_cluster_m2_search, (
                "cluster_m=2 fp8 small-grid admission requires cluster_m=2 search"
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
            self.config_spec.disallow_pid_type(
                "persistent_blocked",
                reason="tcgen05 two-CTA launch-grid contract does not allow "
                "persistent pid types here",
            )
            if not allow_cluster_m2_edge_k_tail_family:
                self.config_spec.disallow_pid_type(
                    "persistent_interleaved",
                    reason="tcgen05 two-CTA launch-grid contract does not allow "
                    "persistent pid types here",
                )
        if allow_cluster_m2_search:
            assert cluster_m2_static_k_int is not None
            self.allow_cluster_m2_search(
                static_k=cluster_m2_static_k_int,
                allow_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
                allow_fp8_small_grid=allow_cluster_m2_fp8_small_grid,
            )
        else:
            self.restrict_cluster_m_search((1,))
        self.restrict_num_epi_warps_search((4,))
        self.restrict_num_epi_warps_validation((4,))
        if ab_stages_dtype_bytes is not None:
            assert ab_stages_device is not None, (
                "ab_stages_dtype_bytes requires ab_stages_device "
                "so the SMEM-budget gate consults the operand's device, not "
                "the host's current CUDA device"
            )
            self.allow_deep_direct_entry_validation(device=ab_stages_device)
            self.allow_ab_stages_search(
                dtype_bytes=ab_stages_dtype_bytes,
                device=ab_stages_device,
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
        if not for_search:
            # Validation admits the dtype's deepest AB pipeline; the depth walk trims it.
            constraints = self.ab_stages_search_constraints
            if constraints is None:
                # Cycle 97: make ab=3 BUDGET-AWARE-SEARCHABLE. Where the device/dtype
                # admits ab=3 at all (the SMEM-budget constraints were recorded by
                # ``allow_ab_stages_three_search`` at bind time — B200-class optin cap,
                # bf16/fp16), lift the ``for_search`` cap to 3 so the autotuner can
                # SAMPLE ab=3 directly instead of reaching it only through the per-shape
                # FFI / gelu seeds. ``_fix_ab_stages_search_config`` then demotes any
                # sampled ab=3 that does not fit (the residual/source-C ring overflows;
                # cluster_m=1 256x256 overflows bare-AB) before codegen, so admission is
                # free but an overflowing kernel is never generated.
                ab_stages_max = 3
            else:
                ab_stages_max = self._get_dtype_ab_stages_hard_cap(
                    constraints.dtype_bytes
                )
        elif self.ab_stages_search_constraints is not None:
            constraints = self.ab_stages_search_constraints
            ab_stages_max = self._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes)
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
        if self.aux_kernel_detected or not for_search:
            fragments[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = EnumFragment(
                TCGEN05_AUX_LOAD_PLACEMENTS
            )
        if not for_search:
            fragments[TCGEN05_GROUPED_MODE_CONFIG_KEY] = EnumFragment(
                TCGEN05_GROUPED_MODES
            )
            fragments[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = EnumFragment(
                TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
            )
        direct_entry_seed_eligible = self.explicit_epi_tile_family_exists()
        if direct_entry_seed_eligible or (
            not for_search and self._direct_entry_k_block_index() is not None
        ):
            # Validation exposes the two direct-entry controls for explicit
            # configs. Layout overrides already have a generic validation path
            # below; only the seed/search surface narrows them to its fixed tile.
            tvm_ffi_launch_fragment: ConfigSpecFragment = (
                EnumFragment((False, True)) if for_search else BooleanFragment()
            )
            fragments.update(
                {
                    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: BooleanFragment(),
                    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: tvm_ffi_launch_fragment,
                }
            )
            if direct_entry_seed_eligible:
                fragments.update(
                    {
                        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: EnumFragment(
                            (None, TCGEN05_EXPLICIT_EPI_TILE_M),
                        ),
                        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: EnumFragment(
                            (None, TCGEN05_EXPLICIT_EPI_TILE_N),
                        ),
                        TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: EnumFragment(
                            (None, TCGEN05_EXPLICIT_D_STORE_BOX_N),
                        ),
                    }
                )
        return fragments

    def _derive_layout_override_bundle(self, config: dict[str, object]) -> None:
        """Make the three epi-tile overrides agree with ``layout_strategy``."""
        present = [
            key
            for key in (
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY,
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY,
                TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY,
            )
            if key in config
        ]
        if not present:
            return
        if (
            config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
        ):
            derived = {
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: TCGEN05_EXPLICIT_EPI_TILE_M,
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: TCGEN05_EXPLICIT_EPI_TILE_N,
                TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: (
                    TCGEN05_EXPLICIT_D_STORE_BOX_N
                ),
            }
            for key in present:
                config[key] = derived[key]
            return
        for key in present:
            config[key] = None

    @staticmethod
    def _target1_tvm_ffi_promotion_requested(config: dict[str, object]) -> bool:
        """Does *config* request the LAYOUT family this stage governs?"""
        return (
            config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            or config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            or config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) is not None
            or config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) is not None
            or config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) is not None
        )

    def _strip_target1_tvm_ffi_promotion_surface(
        self, config: dict[str, object]
    ) -> None:
        """Repair a config AWAY from the LAYOUT envelope, in place.

        has no reason to lose its launch mechanism. Clearing it here would silently
        """
        config[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY] = False
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

    # Keys that define the LAYOUT direct-entry envelope: the five promotion-request
    _TARGET1_TVM_FFI_ENVELOPE_KEYS: tuple[str, ...] = (
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
        TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY,
        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY,
        TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY,
        "block_sizes",
        "tcgen05_cluster_m",
        "tcgen05_cluster_n",
    )

    def _fix_target1_tvm_ffi_search_config(self, config: dict[str, object]) -> None:
        self._settle_layout_group(config)
        if not self._target1_tvm_ffi_promotion_requested(config):
            return
        #
        # The remaining population asked for ``explicit_epi_tile`` (possibly with
        # ``flat_role``) but does not match the seed's layout envelope. Stripping it
        if self._fix_towards_explicit_epi_tile_envelope(config):
            return
        self._strip_target1_tvm_ffi_promotion_surface(config)

    def _fix_towards_explicit_epi_tile_envelope(
        self, config: dict[str, object]
    ) -> bool:
        """Complete an ``explicit_epi_tile`` request in place. True if completed.

        | ``cluster_m = 2`` | the envelope is CtaGroup.TWO (``tcgen05_is_two_cta``). At ``cluster_m=1`` a ``bm=256`` tile silently emits **Triton**, so this is legality, and it is derivable from a tile the draw already chose |
        """
        # PRECONDITION-FACT: the shape. ``static_full_tiles``, 16-bit dtype and
        # aux-descriptor compatibility are bind-time facts no knob can repair.
        if not self._flat_role_shape_facts_hold():
            return False
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return False
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return False
        block_sizes, m_index, n_index, k_index = config_view
        # PRECONDITION-KNOB: the CTA tile. Checked, never written — see the
        # docstring's measurement for why.
        if (
            block_sizes[m_index] != TCGEN05_TWO_CTA_BLOCK_M
            or block_sizes[n_index] != TCGEN05_TWO_CTA_BLOCK_N
        ):
            return False
        if config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True:
            # flat-role's guard adds two TILE conjuncts over plain explicit-epi-tile:
            # ``bk in {64, 128}`` and ``cluster_n == 1``. Both are drawn values, so
            # both are checked here; a draw that misses either keeps ``flat_role``
            # cleared by the caller rather than having its K tile rewritten.
            #
            # The guard's remaining config-dependent conjuncts — the FLAT warp topology
            bk = block_sizes[k_index]
            # ``_flat_role_shape_facts_hold`` above already requires the cm2
            # constraints, so this narrowing is what makes that invariant explicit
            # rather than an unstated non-local assumption.
            cm2_constraints = self.cluster_m2_search_constraints
            if not (
                cm2_constraints is not None
                and isinstance(bk, int)
                and not isinstance(bk, bool)
                and bk in TCGEN05_DIRECT_ENTRY_LEGAL_BK
                and self.cluster_m2_bk_is_valid(bk, cm2_constraints)
            ):
                return False
            if config.get("tcgen05_cluster_n", 1) != 1:
                return False
        # WRITES NOTHING. Every term above is a precondition, checked and never written,
        # so this function only ever ANSWERS "is this request completable" — the layout
        # keys were already settled by ``_settle_layout_group`` and the ``cluster_m = 2``
        # the envelope needs is written by ``_fix_cluster_m2_search_config``, which owns
        # that key (see its entry block). Returning True is what tells the caller to keep
        # the request instead of stripping it.
        return True

    def _settle_layout_group(self, config: dict[str, object]) -> None:
        """Settle the layout group: promote or demote flat_role, then derive the overrides."""
        if config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True:
            if self._flat_role_config_can_hold(config):
                # COMPLETE the request: ``flat_role=True`` is what the draw asked for and
                # is left alone; ``explicit_epi_tile`` is the value it REQUIRES.
                config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = (
                    Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
                )
                # The flat WARP topology this request also needs is supplied at the top of
                # ``_fix_with_scheduler_search_config`` (stage 5), keyed on the same
                # ``flat_role is True`` test, so all four warp keys are written in one
                # place. Nothing between here and there reads any of them.
            else:
                # DEMOTE. The unmet term is either a shape fact (static_full_tiles,
                # 16-bit dtype, aux-descriptor compatibility) or the strategy, which this
                # stage does not own, so the flag is what gives.
                config[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY] = False
        # Tier 2, from the FINAL ``layout_strategy``. Keep this last, and keep its
        # ``present`` guard: on a non-matmul kernel the override keys are absent by
        # design and ``normalize_strategy`` raises ``InvalidConfig`` for any that
        # appears, so writing ``None`` is not harmless (it broke 12 pointwise tests).
        self._derive_layout_override_bundle(config)

    def _flat_role_config_can_hold(self, config: dict[str, object]) -> bool:
        """Can ``flat_role_coordinates=True`` hold for THIS CONFIG (not just this shape)?

        Not checked here, deliberately: the tile terms (``bm``/``bn``/``bk``/
        """
        if not self._flat_role_shape_facts_hold():
            return False
        # The STRATEGY only. The warp keys it determines are verified at the end of
        # ``_settle_layout_group``, once stage 5's derivation has had its say.
        return config.get(TCGEN05_STRATEGY_CONFIG_KEY) in (
            None,
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            Tcgen05Strategy.PURE_MATMUL_ROLE_LIFECYCLE.value,
        )

    def _flat_role_shape_facts_hold(self) -> bool:
        """Can ``flat_role_coordinates=True`` hold on THIS SHAPE at all?"""
        return self.explicit_epi_tile_family_exists()

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

        Admitted only under ``_aux_tma_edge_search_enabled``, which pins the
        surface to the validated edge+K-tail family with ``cluster_m=2``
        and the c-input warp + aux-TMA combination. Configs outside that
        gate never see the knob; codegen at the default of 2 is unchanged.

        Cycle 46 intentionally keeps this scoped to the edge+K-tail gate
        even though ``_aux_tma_search_enabled`` was widened to admit the
        full-tile cluster_m=2 family. The stage-count choices were tuned
        on the T8/CLC edge rows; exposing them to T14/T20/T25/T28 would
        let autotune sample stage counts on shapes they were not measured
        on.
        """
        if not self._aux_tma_edge_search_enabled():
            return {}
        return {
            TCGEN05_AUX_STAGES_CONFIG_KEY: EnumFragment(TCGEN05_AUX_STAGE_COUNT_CHOICES)
        }

    def sched_stage_count_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Per-config depth of the CLC scheduler's SMEM work-tile mailbox."""
        if not self._clc_persistence_search_enabled():
            return {}
        return {
            TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY: EnumFragment(
                TCGEN05_SCHED_STAGE_COUNTS
            )
        }

    def consumer_regs_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Per-config consumer-warp ``setmaxregister_increase`` ceiling knob.

        Admission mirrors ``aux_stages_autotune_fragments``: the
        ``_aux_tma_edge_search_enabled`` gate pins the search to the
        validated wide-N CLC + aux-TMA seed family with the c-input warp +
        aux-TMA combination. Configs outside that gate never see the
        knob. The default value (256) is included in
        ``TCGEN05_CONSUMER_REGS_CHOICES`` so default-with-knob emits the
        same code as default-without-knob.

        Cycle 46 intentionally keeps this scoped to the edge+K-tail gate
        even though ``_aux_tma_search_enabled`` was widened (see
        ``aux_stages_autotune_fragments``).
        """
        if not self._aux_tma_edge_search_enabled():
            return {}
        return {
            TCGEN05_CONSUMER_REGS_CONFIG_KEY: EnumFragment(
                TCGEN05_CONSUMER_REGS_CHOICES
            )
        }

    def persistence_model_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        # Paired with stage 10's ``_is_validated_clc_persistence_search_candidate``:
        # both read ``_clc_persistence_codegen_supported``, so the surface offers
        # ``clc_persistent`` exactly where the repair will let it stand.
        if not self._clc_persistence_codegen_supported():
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
        direct_entry_seed_eligible = self.explicit_epi_tile_family_exists()
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
        # The store-warp slot stays narrowed to ``0`` in the autotune surface —
        # only an explicit ``helion.Config(tcgen05_warp_spec_store_warps=1)``
        # activates it. Cycle 93 (Workstream A Stage 4) landed the productive
        # decouple body (the C-store edge + tail split, +1.1 % on T20), but the
        # autotune surface stays at ``0`` until Stage 5 wires store=1 into the
        # residual family's production config + runs the full regression sweep,
        # so no passing target can pick it before it is characterized per-family.
        store_warps_choices: tuple[int, ...] = (0,)
        if direct_entry_seed_eligible:
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
            TCGEN05_WARP_SPEC_STORE_WARPS_KEY: EnumFragment(store_warps_choices),
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
        # Cycle 91 (Workstream A Stage 3): the user-config validation surface
        # accepts ``{0, 1}`` so an explicit ``store_warps=1`` round-trips; the
        # per-strategy accept set in ``_STRATEGY_SUPPORTED_STORE_WARPS`` still
        # pins it to ``{0}`` outside ROLE_LOCAL_WITH_SCHEDULER.
        fragments[TCGEN05_WARP_SPEC_STORE_WARPS_KEY] = EnumFragment((0, 1))
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
            # The loop just put ``layout_strategy`` back to DEFAULT, and
            # ``flat_role_coordinates=True`` is illegal there -- ``cute_mma.py`` raises
            # ``BackendUnsupported`` for the flag outside the guarded explicit-epi-tile
            # path, which is why ``_settle_layout_group`` promotes layout whenever the
            # flag holds. The flag is NOT in ``TCGEN05_STRATEGY_CONFIG_KEYS`` (and
            # ``strategy_field_default`` has no branch for it), so the loop cannot clear
            # it. Clearing it here keeps the pair settled: left True, the next
            # ``normalize`` pass re-promotes layout off the surviving flag and the
            # pipeline has no fixed point -- which the post-fix write-back requires.
            #
            # Only rewritten when already PRESENT: the key is absent by design on a
            # family that cannot draw it (``optional_fragments(for_search=True)`` omits
            # it off the direct-entry surface), and introducing it there makes the
            # config differ from what the same vector decodes to, which is what the
            # write-back checks before adopting.
            if TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY in config:
                config[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY] = False
            return
        message = "; ".join(errors)
        raise InvalidConfig(f"tcgen05 strategy invariants violated: {message}")

    def _clamp_l2_swizzle_size_to_shape(self, config: dict[str, object]) -> None:
        # CuTe layout construction assumes the L2 swizzle does not exceed the
        # number of N tile-clusters; clamp before layout objects are built.
        swizzle_value = config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
        if swizzle_value is None or swizzle_value == 1:
            return
        indices = self._matmul_block_indices()
        if indices is None:
            config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
            return
        n_block_index = indices[1]
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or n_block_index >= len(block_sizes):
            return
        bn = block_sizes[n_block_index]
        if not isinstance(bn, int) or isinstance(bn, bool) or bn <= 0:
            return
        n_hint = self.config_spec.block_sizes[n_block_index].size_hint
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
        reserved_sms_key = TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
        if reserved_sms_key in config and not self.search_enabled:
            if fix_invalid:
                config.pop(reserved_sms_key, None)
            else:
                raise InvalidConfig(
                    f"{reserved_sms_key} is only supported for tcgen05-enabled "
                    "CuTe matmul kernels"
                )
        optional_fragments = self.optional_fragments()
        optional_search_fragments = self.optional_fragments(for_search=True)
        if self.search_enabled:
            for key, fragment in optional_fragments.items():
                if key in config:
                    if key == "tcgen05_ab_stages" and (
                        self._grouped_dynamic_deep_config_matches(config)
                        or self._grouped_worklist_nm_deep_ab_config_matches(
                            config,
                            config[key],
                        )
                    ):
                        config[key] = int(cast("Any", config[key]))
                    else:
                        config[key] = self._validate_optional_fragment_value(
                            key, fragment, config[key]
                        )
                elif key in optional_search_fragments:
                    if key == TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY:
                        # An omitted user-config means "no FFI promotion
                        # requested" — fill from the validation surface
                        # so the non-seed envelope and promotion gates
                        # that key off ``config.get(...) is True`` stay
                        # consistent.
                        config[key] = optional_fragments[key].default()
                    else:
                        config[key] = optional_search_fragments[key].default()
            self._clamp_l2_swizzle_size_to_shape(config)
            # ── SETTLE THE LAYOUT GROUP BEFORE JUDGING AB DEPTH ──
            #
            # ``_validate_direct_entry_ab_stage_envelope`` below reads
            if fix_invalid:
                self._settle_layout_group(config)
            self._validate_direct_entry_ab_stage_envelope(
                config, fix_invalid=fix_invalid
            )
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
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            TCGEN05_AUX_LOAD_PLACEMENTS,
            fix_invalid=fix_invalid,
        )
        aux_load_placement = config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
        if (
            aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            and not self.aux_kernel_detected
        ):
            raise InvalidConfig(
                f"invalid {TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
                f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}: the kernel has "
                "no per-subtile auxiliary loads to place"
            )
        if (
            aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            and config.get(
                TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
                TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
            )
            == TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
        ):
            raise InvalidConfig(
                f"invalid {TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
                f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}: requires "
                f"{TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY}="
                f"{TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP!r}; per-subtile "
                "auxiliary loads cannot precede an accumulator wait emitted "
                "before the subtile loop"
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
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
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
        # epilogue_subtile is incompatible with EVERY tcgen05 matmul config: the FX split
        # clones the store per piece, and a tcgen05-fed store off the splice whitelist hits
        # the SIMT fallback, which needs indices_<n>/mask_<n> names the tcgen05 grid never
        # binds (memory_ops.py:1644 raises BackendUnsupported).
        if self.search_enabled and self.matmul_block_ids is not None:
            # The tcgen05 CtaGroup.TWO MMA path does not emit the per-block-id
            # indices/masks that a fused epilogue subtile needs, so a sampled
            # ``epilogue_subtile`` on a cluster_m=2 candidate raises
            # ``BackendUnsupported`` at codegen. Drop it here (rather than letting
            # the candidate fail to compile and waste autotune budget) -- every
            # cluster_m=2 search candidate that survives to this point is committed
            # to the 2-CTA path. The edge-family prefixes below also pop it for
            # their sub-paths; doing it once here covers the full-tile and
            # small-grid paths too.
            config.pop("epilogue_subtile", None)
        # The FFI repair runs FIRST, before the cluster_m=2 block shaping. It
        # only rewrites the FFI-only controls + layout keys, so the shaping below
        # then sees a settled layout: a repaired candidate is DEFAULT-layout and
        # so its sampled bn<=128 can snap to the un-seeded [256,128,*] tile,
        # instead of being forced to 256 by the ``layout == DEFAULT`` guard in
        # ``_fix_cluster_m2_search_config`` reading a pre-repair
        # ``explicit_epi_tile``. (When this ran after the shaping, every repaired
        # candidate landed on the already-seeded 256x256 tile.)
        self._fix_target1_tvm_ffi_search_config(config)
        self._fix_cluster_m2_search_config(config)
        self._fix_cluster_m1_persistent_search_config(config)
        # ── NO AUX-EDGE STAGE ──
        #
        # c_input_warps=0)`` on an aux kernel whose tile left a partial tile on BOTH M
        self._fix_with_scheduler_search_config(config)
        self._fix_aux_tma_search_config(config)
        # Budget-aware ab-stages admission for the lifted for_search cap, and the
        # ONLY AB-depth budget walk on this path (it absorbed the old bare-AB
        # ``ab == 3``-exactly stage). Runs after the family projections (FFI / gelu
        # / aux-TMA full-tile) have set their validated stage tuple AND after the
        # cluster_m=2 shaping, so it judges the tile that actually reaches codegen:
        # a directly-SAMPLED ab>=3 that no projection claimed and that does not fit
        self._fix_ab_stages_search_config(config)
        # Immediately AFTER the depth walk, on purpose: it judges the aux producer warp
        # against the SETTLED ``tcgen05_ab_stages``. Moved out of
        # ``_fix_with_scheduler_search_config`` (2026-08-10), whose early position made that
        # read stale and cost the pipeline its fixed point.
        self._fix_aux_producer_depth_feasibility_search_config(config)
        # Final c-stages admission: demote a directly-sampled (or any unclaimed)
        # deeper C ring on the canonical 256x256 DEFAULT-layout path that does
        # not fit AB+C under the B200 cap. Runs after the family fixups so their
        # validated c=4 (residual ab=2 projection above; edge/seed families,
        # which this gate's scope excludes) is set first.
        self._fix_c_stages_search_config(config)
        # CLC admission depends on the projected cluster/pid/strategy tuple
        # above, including the scheduler/c-input warp fix-ups.
        self._fix_clc_persistence_search_config(config)
        # together with the method. It cleared ``tvm_ffi_launch`` when the FINAL tile would
        # not lower on tcgen05, mirroring ``cute/backend.py``'s raise (``--enable-tvm-ffi``
        # is emitted only for a kernel whose specialized MMA plan is ``tcgen05``).
        self._validate_sched_stage_count_config(config, fix_invalid=True)

    def normalize_strategy(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        if not self.search_enabled:
            return
        default_loop_orders = [
            spec._fill_missing() for spec in self.config_spec.loop_orders
        ]
        loop_orders = config.get("loop_orders", default_loop_orders)
        cluster_shape = (
            config.get("tcgen05_cluster_m", 1),
            config.get("tcgen05_cluster_n", 1),
        )
        if cluster_shape != (1, 1) and loop_orders != default_loop_orders:
            # The clustered scheduler binds physical M/N dimensions to work-tile
            # coordinates 0/1. Reordering is safe for a single CTA, but clustered
            # scheduling must become block-ID-aware before those coordinates move.
            if fix_invalid:
                config["loop_orders"] = default_loop_orders
            else:
                raise InvalidConfig(
                    "non-default loop_orders require tcgen05 cluster shape (1, 1)"
                )
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
        self._normalize_grouped_static_reserved_sms(config)

    def flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "l2_groupings": self.config_spec.l2_groupings,
        }
        if (
            self.config_spec.supports_config_key("loop_orders")
            and len(self.config_spec.loop_orders) > 0
        ):
            fields["loop_orders"] = self.config_spec.loop_orders
        fields.update(self.optional_fragments(for_search=True))
        grouped_seed_modes = tuple(
            dict.fromkeys(
                mode
                for config in self.config_spec.compiler_seed_configs
                if (mode := config.config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY))
                in TCGEN05_GROUPED_MODES
            )
        )
        has_grouped_dynamic_seed = any(
            mode in TCGEN05_GROUPED_DYNAMIC_MODES for mode in grouped_seed_modes
        )
        if has_grouped_dynamic_seed:
            fields[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] = EnumFragment(
                TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES
            )
        if grouped_seed_modes:
            # Seed-only encoding: random/default search stays off the grouped
            # path, while exact-proof compiler seeds survive flatten/unflatten.
            fields[TCGEN05_GROUPED_MODE_CONFIG_KEY] = EnumFragment(
                (None, *grouped_seed_modes),
                search_choices=(None,),
            )
        grouped_source_m_tiles = tuple(
            dict.fromkeys(
                source_m_tile
                for config in self.config_spec.compiler_seed_configs
                if (
                    source_m_tile := config.config.get(
                        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
                    )
                )
                and type(source_m_tile) is int
                and source_m_tile in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
            )
        )
        if grouped_source_m_tiles:
            fields[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = EnumFragment(
                (None, *grouped_source_m_tiles),
                search_choices=(None,),
            )
        fields.update(self.strategy_autotune_fragments())
        fields.update(self.aux_load_mode_autotune_fragments())
        fields.update(self.aux_stages_autotune_fragments())
        fields.update(self.sched_stage_count_autotune_fragments())
        fields.update(self.consumer_regs_autotune_fragments())
        fields.update(self.persistence_model_autotune_fragments())
        if self.config_spec.supports_config_key("pid_type"):
            seed_pid_types = tuple(
                cast("PidTypeLiteral", pid_type)
                for seed in self.config_spec.compiler_seed_configs
                if isinstance(pid_type := seed.config.get("pid_type"), str)
            )
            pid_type_choices = tuple(
                dict.fromkeys((*self.allowed_pid_types, *seed_pid_types))
            )
            pid_type_search_choices = (
                self.allowed_pid_types
                if pid_type_choices != self.allowed_pid_types
                else None
            )
            fields["pid_type"] = EnumFragment(
                pid_type_choices, search_choices=pid_type_search_choices
            )
        if (
            self.config_spec.supports_config_key("indexing")
            and self.config_spec.indexing.length > 0
        ):
            fields["indexing"] = self.config_spec.indexing
        return fields
