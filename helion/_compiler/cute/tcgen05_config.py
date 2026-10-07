from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import TypeVar
from typing import cast

import torch

from ...autotuner.config_fragment import BooleanFragment
from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_fragment import IntegerFragment
from ...autotuner.config_fragment import ListOf
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_warp_mma_gemm import MATMUL_FAMILIES
from .cute_warp_mma_gemm import MATMUL_FAMILY_TCGEN05
from .cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
from .cute_warp_mma_gemm import WARP_MMA_DEFAULT_WARPS
from .cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY
from .cute_warp_mma_gemm import WARP_MMA_MAX_BM
from .cute_warp_mma_gemm import WARP_MMA_MAX_BN
from .cute_warp_mma_gemm import WARP_MMA_MIN_BM
from .cute_warp_mma_gemm import WARP_MMA_MIN_BN
from .cute_warp_mma_gemm import WARP_MMA_SEED_TILES
from .cute_warp_mma_gemm import WARP_MMA_WARP_CHOICES
from .cute_warp_mma_gemm import WARP_MMA_WARPS_KEY
from .cute_warp_mma_gemm import warp_mma_k_step
from .cute_warp_mma_gemm import warp_mma_max_legal_warps
from .cute_warp_mma_gemm import warp_mma_tile_reason
from .cute_warp_mma_gemm import warp_mma_warp_layout
from .epilogue_fanout import FANOUT_CONFIG_KEY
from .epilogue_fanout import FANOUT_MODES
from .epilogue_fanout import schedule_supported as fanout_schedule_supported
from .grouped_full_coverage import full_coverage_pipeline_supported
from .grouped_full_coverage import full_coverage_smem_upper_bound
from .grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from .grouped_row_union import LEGACY_SCHEDULE
from .grouped_row_union import PAIRED_CLC_SCHEDULE
from .grouped_row_union import RESIDENT_CTAS_KEY as GROUPED_RESIDENT_CTAS_KEY
from .grouped_row_union import SCHEDULE_KEY as GROUPED_ROW_UNION_SCHEDULE_KEY
from .grouped_row_union import STARTUP_PREFILL_KEY
from .grouped_row_union import TRANSPOSED_SCHEDULE
from .grouped_row_union import physical_schedule
from .grouped_row_union import schedule_supported as row_union_schedule_supported
from .pipeline_smem import TCGEN05_CTA_GROUP_CHOICES
from .pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from .pipeline_smem import TCGEN05_SMEM_AWARE_MAX_AB_STAGES
from .pipeline_smem import Tcgen05PipelineSmemFacts
from .pipeline_smem import max_pipeline_ab_stages
from .pipeline_smem import pipeline_smem_bytes
from .strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from .strategies import TCGEN05_BATCH_RASTER_CHOICES
from .strategies import TCGEN05_BATCH_RASTER_CONFIG_KEY
from .strategies import TCGEN05_BATCH_RASTER_SLOWEST
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
from .strategies import TCGEN05_PERSISTENCE_MODEL_PID_TYPES
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
from .strategies import tcgen05_explicit_epilogue_tile_supported
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
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODES
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from .tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENTS
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from .tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from .tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from .tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_C_STORE_MODES
from .tcgen05_constants import TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_DEFAULT
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_DYNAMIC_MODES
from .tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_DENSE
from .tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL
from .tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_MODES
from .tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_OFF
from .tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_MODE_DIRECT
from .tcgen05_constants import TCGEN05_GROUPED_MODE_DYNAMIC
from .tcgen05_constants import TCGEN05_GROUPED_MODE_STATIC
from .tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from .tcgen05_constants import TCGEN05_GROUPED_MODES
from .tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_N
from .tcgen05_constants import TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNTS
from .tcgen05_constants import TCGEN05_SMALL_GRID_DIRECT_STORE_MAX_BN
from .tcgen05_constants import TCGEN05_SMEM_ROW_STAGE_CHUNK_BYTES
from .tcgen05_constants import TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_DEEP_AB_STAGES
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES
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
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
from .tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .tcgen05_constants import TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from .tcgen05_constants import Tcgen05RowvecAuxFacts
from .tcgen05_constants import resolve_tcgen05_grouped_worklist_mma_profile
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_default_epilogue_tile_size
from .tcgen05_constants import tcgen05_direct_entry_stage_tuple_allowed
from .tcgen05_constants import tcgen05_fixed_smem_overhead_bytes
from .tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from .tcgen05_constants import tcgen05_round_up_smem_bytes
from .tcgen05_constants import tcgen05_rowvec_stage_smem_bytes
from .tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...autotuner.block_id_sequence import BlockIdSequence
    from ...autotuner.config_fragment import BlockSizeFragment
    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import PidTypeLiteral
    from .epilogue_fanout import PairedFanoutPlan
    from .materialized_pdl import MaterializedOperandPdl


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
    # When True, sampled bm=256 / bn in TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
    # cluster_m=2 candidates keep their narrow N tile instead of the 256x256
    # projection (GEMMs whose 256x256 grid cannot fill the SMs).
    allow_one_wave_tiles: bool = False
    # When True, the one-wave tiles are the only reason cluster_m=2 search is
    # on: the plain 256x256 two-CTA grid would not fill a quarter of the SMs,
    # so its families stay off (the 2x2 / CLC / M-pair / C-input seeds, the
    # FFI direct-entry coordinate, the aux-TMA surface) and a sampled
    # cluster_m=2 candidate that is not a one-wave tile runs on one CTA.
    one_wave_only: bool = False


class Tcgen05AbStagesThreeSearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_ab_stages=3`` is admitted.

    The 3-stage AB pipeline is only safe to search when its larger SMEM
    allocation fits after reserving space for CuTe runtime/barrier scratch.
    """

    dtype_bytes: int
    per_cta_smem_budget_bytes: int


class Tcgen05GroupedWorklistSmemFacts(NamedTuple):
    group_count: int
    device_split_sizes: bool


TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH = (
    "is outside the grouped N,M worklist envelope (a 256x128 tile with a "
    "resolvable MMA profile, tcgen05_cluster_n=1, tcgen05_acc_stages=2, "
    "tcgen05_c_stages=2 and 4 <= tcgen05_ab_stages <= 7)"
)
TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH = (
    "exceeds the grouped N,M worklist per-CTA SMEM footprint"
)
TCGEN05_GROUPED_DYNAMIC_AB4_STAGE = 4
TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES = ((4, 2), (8, 4))


_SeedValue = TypeVar("_SeedValue", bound=Hashable)


def _compiler_seed_values(
    seeds: Sequence[Config],
    key: str,
    value_type: type[_SeedValue],
    is_valid: Callable[[_SeedValue], bool],
    *,
    exact_type: bool = True,
    is_valid_for_seed: Callable[[Config, _SeedValue], bool] | None = None,
) -> tuple[_SeedValue, ...]:
    return tuple(
        dict.fromkeys(
            cast("_SeedValue", value)
            for seed in seeds
            for value in (seed.config.get(key),)
            if (
                type(value) is value_type
                if exact_type
                else isinstance(value, value_type)
            )
            and is_valid(cast("_SeedValue", value))
            and (
                is_valid_for_seed is None
                or is_valid_for_seed(seed, cast("_SeedValue", value))
            )
        )
    )


def _integer_fragment_with_seed_values(
    fragment: IntegerFragment,
    seed_values: tuple[int, ...],
) -> ConfigSpecFragment:
    seed_only_values = tuple(
        value
        for value in dict.fromkeys(seed_values)
        if value < fragment.low or value > fragment.high
    )
    return (
        fragment
        if not seed_only_values
        else _CompilerSeedIntegerFragment(fragment, seed_only_values)
    )


class _CompilerSeedIntegerFragment(IntegerFragment):
    """An integer search range that can also encode frozen compiler seeds."""

    def __init__(
        self,
        fragment: IntegerFragment,
        seed_only_values: tuple[int, ...],
    ) -> None:
        super().__init__(fragment.low, fragment.high, fragment.default_val)
        self.seed_only_values = seed_only_values

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if type(current) is not int:
            raise TypeError(f"Expected int, got {type(current).__name__}")
        if self.low <= current <= self.high:
            return super().pattern_neighbors(current, radius)
        if current not in self.seed_only_values:
            raise ValueError(f"{current!r} is not a compiler-seed integer value")
        boundary = self.clamp(current)
        return [boundary, *super().pattern_neighbors(boundary, radius)]

    def encode(self, value: object) -> list[float]:
        if type(value) is not int:
            raise TypeError(f"Expected int, got {type(value).__name__}")
        if not (self.low <= value <= self.high or value in self.seed_only_values):
            raise ValueError(f"{value!r} is not a compiler-seed integer value")
        return [float(value)]

    def fingerprint(self) -> tuple[str | int, ...]:
        return (
            "compiler_seed_integer",
            self.low,
            self.high,
            self.default_val,
            *self.seed_only_values,
        )


def _enum_fragment_with_seed_values(
    base_choices: tuple[object, ...],
    seed_values: tuple[Hashable, ...],
    *,
    search_choices: tuple[object, ...],
    original: EnumFragment | None = None,
    search_only_if_widened: bool = False,
) -> EnumFragment:
    choices = tuple(dict.fromkeys((*base_choices, *seed_values)))
    if original is not None and choices == base_choices:
        return original
    return EnumFragment(
        choices,
        search_choices=(
            None
            if search_only_if_widened and choices == base_choices
            else search_choices
        ),
    )


# The generated grouped kernel's non-operand allocations are about 1.6 KiB
# (pipeline barriers, TensorMap staging, and TMEM bookkeeping). Keep a small
# margin while still admitting CUTLASS's max-fit AB8/C4 pipeline on B200.
TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES = 2 * 1024


CUTE_TCGEN05_TUNABLE_KEYS: tuple[str, ...] = (
    GROUPED_ROW_UNION_KEY,
    GROUPED_ROW_UNION_SCHEDULE_KEY,
    STARTUP_PREFILL_KEY,
    GROUPED_RESIDENT_CTAS_KEY,
    FANOUT_CONFIG_KEY,
    TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
    TCGEN05_CTA_GROUP_CONFIG_KEY,
    "tcgen05_cluster_m",
    "tcgen05_cluster_n",
    "tcgen05_ab_stages",
    "tcgen05_region_ab_stages",
    "tcgen05_region_c_stages",
    "tcgen05_materialized_pdl",
    "tcgen05_acc_stages",
    "tcgen05_c_stages",
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_STORE_MODE_CONFIG_KEY,
    "tcgen05_num_epi_warps",
    TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
    TCGEN05_BATCH_RASTER_CONFIG_KEY,
    WARP_MMA_FAMILY_KEY,
    WARP_MMA_WARPS_KEY,
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
        TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
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
        # Separate device launches may own independent native contractions.
        # They share pipeline knobs, but never a single matrix-axis projection.
        self.materialized_matmul_block_ids: tuple[tuple[int, int, int], ...] = ()
        self.materialized_matmul_shapes: tuple[tuple[int, int, int], ...] = ()
        self.materialized_pipeline_facts: tuple[
            Tcgen05PipelineSmemFacts | None, ...
        ] = ()
        self.materialized_pair_budget_facts: tuple[
            Tcgen05PipelineSmemFacts | None, ...
        ] = ()
        # DeviceIR may later replace missing MatmulFact extents with runtime
        # hints. Keep the preflight plan's compile-time provenance for CuTe.
        self.matmul_compile_time_static_extents: (
            tuple[int | None, int | None, int | None] | None
        ) = None
        self.matmul_input_dtype: torch.dtype | None = None
        self.matmul_has_leading_passthrough: bool = False
        # Product of the leading passthrough (batch) extents: the work-tile
        # multiplier of every (m, n) tile grid.
        self.matmul_leading_work_multiplier: int = 1
        self.matmul_explicit_epi_tile_compatible: bool | None = None
        self.aux_kernel_detected: bool = False
        self.exact_shape_aux_kernel_detected: bool = False
        # Row-vector aux rows of the analyzed stores (``aux_tensor``): the SMEM
        # model of the ``pre_acc_wait`` row stages the seeds request.
        self.rowvec_aux_facts: Tcgen05RowvecAuxFacts | None = None
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
        self.ab_stages_three_search_constraints: (
            Tcgen05AbStagesThreeSearchConstraints | None
        ) = None
        self.grouped_worklist_smem_facts: Tcgen05GroupedWorklistSmemFacts | None = None
        self.grouped_full_coverage_supported: bool = False
        self.grouped_full_coverage_eligible: bool = False
        self.grouped_row_union_supported: bool = False
        self.grouped_row_union_cluster4_supported: bool = False
        self.grouped_row_union_cluster4_eligible: bool = False
        self.grouped_row_union_paired_clc_supported: bool = False
        self.grouped_row_union_paired_clc_eligible: bool = False
        self.grouped_row_union_eligible: bool = False
        self.grouped_row_union_multi_resident_supported: bool = False
        self.epilogue_fanout_plans: tuple[PairedFanoutPlan, ...] = ()
        self.epilogue_fanout_search_enabled: bool = False
        self.deep_direct_entry_validation_enabled: bool = False
        self.pipeline_smem_facts: Tcgen05PipelineSmemFacts | None = None
        # Root identities from the typed operand-materialization dependency proof.
        self.materialized_operand_pdl_roots: MaterializedOperandPdl | None = None
        self.num_epi_warps_search_choices: tuple[int, ...] | None = None
        self.num_epi_warps_validation_choices: tuple[int, ...] | None = None
        # SM count of the bound device (0 when unknown); small-grid seeds use
        # it to decide whether the standard tiling can fill the machine.
        self.device_sm_count: int = 0
        # The register-MMA GEMM family (``cute_matmul_family="warp_mma"``):
        # admitted by the search plan (static latency-bound problem, one
        # matmul, supported layouts); its 16x8 tiles widen the M / N block
        # floors, which ``_normalize_warp_mma_family`` clamps back to the
        # recorded tcgen05 floors for tcgen05-family configs.
        self.warp_mma_admitted: bool = False
        self.tcgen05_min_search_m: int = 64
        self.tcgen05_min_search_n: int = 8
        # Block id of the leading passthrough (batch) axis, when any.
        self.matmul_leading_block_id: int | None = None

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
        compile_time_static_extents: tuple[int | None, int | None, int | None],
        input_dtype: torch.dtype,
        has_leading_passthrough: bool,
        explicit_epi_tile_compatible: bool,
        leading_work_multiplier: int = 1,
    ) -> None:
        """Record semantic axes from the structurally accepted MMA candidate."""
        assert self.matmul_block_ids is None, "tcgen05 MMA analysis registered twice"
        self.matmul_block_ids = (m_block_id, n_block_id, k_block_id)
        self.matmul_compile_time_static_extents = compile_time_static_extents
        self.matmul_input_dtype = input_dtype
        self.matmul_has_leading_passthrough = has_leading_passthrough
        self.matmul_leading_work_multiplier = max(1, leading_work_multiplier)
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

    def row_union_profile_supported(self, name: str) -> bool:
        if name == TRANSPOSED_SCHEDULE:
            return self.grouped_row_union_cluster4_supported
        if name == PAIRED_CLC_SCHEDULE:
            return self.grouped_row_union_paired_clc_supported
        return False

    def project_row_union_codegen(self, config: Config) -> Config:
        profile = physical_schedule(config)
        if profile is None:
            return config
        indices = self._matmul_block_indices()
        if (
            not self.row_union_profile_supported(profile.name)
            or indices is None
            or config.get(GROUPED_ROW_UNION_KEY, False) is not True
            or not row_union_schedule_supported(
                config,
                cast(
                    "tuple[int, int, int]",
                    tuple(config.block_sizes[i] for i in indices),
                ),
            )
        ):
            raise InvalidConfig(
                "row-union physical codegen projection lacks its typed schedule proof"
            )
        blocks = list(config.block_sizes)
        for index, value in zip(indices, profile.source_tile, strict=True):
            blocks[index] = value
        return Config.from_dict(
            config.config | profile.codegen_values() | {"block_sizes": blocks}
        )

    def register_pipeline_smem_facts(self, facts: Tcgen05PipelineSmemFacts) -> None:
        self.pipeline_smem_facts = facts
        if self.paired_pipeline_search_enabled():
            # The explicit instruction family owns its own 128-row geometry.
            # It need not satisfy the older 256x256 search projection.
            for pid_type in ("persistent_blocked", "persistent_interleaved"):
                if pid_type not in self.allowed_pid_types:
                    self.allowed_pid_types = (
                        *self.allowed_pid_types,
                        cast("PidTypeLiteral", pid_type),
                    )

    def paired_pipeline_search_enabled(self) -> bool:
        facts = self.pipeline_smem_facts
        extents = self.matmul_compile_time_static_extents
        fragments = self._matmul_block_fragments()
        if (
            not self.search_enabled
            or facts is None
            or extents is None
            or fragments is None
            or any(value is None or value <= 0 for value in extents)
        ):
            return False
        m, n, k = cast("tuple[int, int, int]", extents)
        return (
            m % 128 == 0
            and n % 64 == 0
            and k % 64 == 0
            and all(
                fragment.low <= value <= fragment.high
                for fragment, value in zip(fragments, (128, 64, 64), strict=True)
            )
        )

    def _paired_pipeline_parameters(
        self, config: dict[str, object]
    ) -> tuple[int, int, int, int, int] | None:
        """Exact allocation family supported by the smaller SMEM reservation."""
        facts = self.pipeline_smem_facts
        view = self._matmul_config_view(config)
        if (
            facts is None
            or view is None
            or config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) != "two"
            or config.get("tcgen05_cluster_m", 1) != 2
            or config.get("tcgen05_cluster_n", 1) != 1
            or config.get("pid_type")
            not in (
                "persistent_blocked",
                "persistent_interleaved",
            )
            or config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is not None
            or config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY)
            or config.get(TCGEN05_STRATEGY_CONFIG_KEY, "role_local_monolithic")
            != "role_local_monolithic"
            or config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, "static_persistent")
            != "static_persistent"
            or config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, "default") != "default"
            or any(config.get(key) is not None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
            or any(
                config.get(key, 0) != 0
                for key in (
                    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
                    TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
                    TCGEN05_WARP_SPEC_STORE_WARPS_KEY,
                )
            )
        ):
            return None
        blocks, mi, ni, ki = view
        bm, bn, bk = blocks[mi], blocks[ni], blocks[ki]
        c_stages = config.get("tcgen05_c_stages", 2)
        acc_stages = config.get("tcgen05_acc_stages", 2)
        if (
            bm not in (128, 256)
            or bn not in (64, 128, 256)
            or bk not in (64, 128, 256)
            or c_stages not in (2, 4)
            or acc_stages not in (1, 2)
        ):
            return None
        return cast(
            "tuple[int, int, int, int, int]", (bm, bn, bk, c_stages, acc_stages)
        )

    def _paired_pipeline_stage_limit(self, config: dict[str, object]) -> int | None:
        parameters = self._paired_pipeline_parameters(config)
        if parameters is None:
            return None
        facts = self.pipeline_smem_facts
        assert facts is not None
        bm, bn, bk, c_stages, acc_stages = parameters
        return max_pipeline_ab_stages(
            facts,
            bm=bm,
            bn=bn,
            bk=bk,
            c_stages=c_stages,
            acc_stages=acc_stages,
        )

    def _materialized_paired_stage_limits(
        self, config: dict[str, object]
    ) -> list[int] | None:
        if not self.materialized_matmul_block_ids or not self.epilogue_fanout_plans:
            return None
        if (
            config.get(FANOUT_CONFIG_KEY) != "shared"
            or config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) != "two"
            or config.get("tcgen05_cluster_m", 1) != 2
            or not fanout_schedule_supported(config)
            or not self.epilogue_fanout_config_supported(config)
        ):
            return None
        blocks = config.get("block_sizes")
        if not isinstance(blocks, list):
            return None
        c_counts = config.get(
            "tcgen05_region_c_stages", [0] * len(self.materialized_matmul_block_ids)
        )
        assert isinstance(c_counts, list)
        limits = []
        for region, (axes, shape, facts, pair_facts) in enumerate(
            zip(
                self.materialized_matmul_block_ids,
                self.materialized_matmul_shapes,
                self.materialized_pipeline_facts,
                self.materialized_pair_budget_facts,
                strict=True,
            )
        ):
            indices = [self._config_block_index(axis) for axis in axes]
            if any(index is None or index >= len(blocks) for index in indices):
                return None
            bm, bn, bk = (blocks[cast("int", index)] for index in indices)
            if (
                bm not in (128, 256)
                or bn not in (64, 128, 256)
                or bk not in (64, 128, 256)
                or any(
                    extent % tile
                    for extent, tile in zip(shape, (bm, bn, bk), strict=True)
                )
            ):
                return None
            c_count = c_counts[region] or config.get("tcgen05_c_stages", 2)
            if c_count not in (2, 4):
                return None
            pairs = [
                pair
                for pair in self.epilogue_fanout_plans
                if pair.block_ids == axes[:2]
            ]
            if pairs:
                if (
                    len(pairs) != 1
                    or pairs[0].output_dtype.itemsize != 2
                    or c_count != 2
                    or pair_facts is None
                ):
                    return None
                facts = pair_facts
            elif facts is None:
                return None
            assert facts is not None
            # Equal-size full output rings have 1024B-multiple sizes, so the
            # second ring adds no alignment gap beyond the existing reservation.
            base = pipeline_smem_bytes(
                facts, bm=bm, bn=bn, bk=bk, ab_stages=1, c_stages=c_count, acc_stages=2
            )
            ab_one = tcgen05_ab_smem_bytes_per_cta(
                bm=bm,
                bn=bn,
                bk=bk,
                dtype_bytes=facts.input_dtype_bytes,
                ab_stages=1,
                cluster_m=2,
            )
            ring = base - ab_one - (2 * 1024 + 16 + 32)
            extra = ring if pairs else 0
            limits.append(
                max(
                    (
                        count
                        for count in range(1, 17)
                        if pipeline_smem_bytes(
                            facts,
                            bm=bm,
                            bn=bn,
                            bk=bk,
                            ab_stages=count,
                            c_stages=c_count,
                            acc_stages=2,
                        )
                        + extra
                        <= facts.capacity_bytes
                    ),
                    default=0,
                )
            )
        return limits if limits and all(limits) else None

    def _validate_cta_group_config(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) != "two":
            return
        region_limits = self._materialized_paired_stage_limits(config)
        if region_limits is not None:
            counts = config.get("tcgen05_region_ab_stages", [0] * len(region_limits))
            assert isinstance(counts, list)
            effective = [
                count or config.get("tcgen05_ab_stages", 2) for count in counts
            ]
            if all(
                type(count) is int and 1 <= count <= limit
                for count, limit in zip(effective, region_limits, strict=True)
            ):
                return
            if fix_invalid:
                config["tcgen05_region_ab_stages"] = [
                    min(cast("int", count), limit)
                    for count, limit in zip(effective, region_limits, strict=True)
                ]
                return
            raise InvalidConfig(
                "materialized two-CTA AB pipelines exceed their complete per-region SMEM budgets"
            )
        limit = self._paired_pipeline_stage_limit(config)
        if limit is not None and limit > 0:
            return
        if fix_invalid:
            config[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
            return
        raise InvalidConfig(
            "tcgen05_cta_group='two' requires one row-major half/bfloat16 MMA "
            "with a single output, no auxiliary tensor loads, a static "
            "persistent two-CTA schedule, and supported shared-memory geometry"
        )

    def _paired_pipeline_seed_configs(self, *, acc_stages: int = 2) -> list[Config]:
        if not self.paired_pipeline_search_enabled():
            return []
        facts = self.pipeline_smem_facts
        extents = self.matmul_compile_time_static_extents
        fragments = self._matmul_block_fragments()
        assert facts is not None and extents is not None and fragments is not None
        m, n, k = cast("tuple[int, int, int]", extents)
        seeds: list[Config] = []
        for bm in (128, 256):
            for bn in (64, 128, 256):
                for bk in (64, 128):
                    values = (bm, bn, bk)
                    if any(
                        extent % value != 0
                        or not fragment.low <= value <= fragment.high
                        for extent, value, fragment in zip(
                            (m, n, k), values, fragments, strict=True
                        )
                    ):
                        continue
                    block_sizes = self._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
                    assert block_sizes is not None
                    maximum = max_pipeline_ab_stages(
                        facts, bm=bm, bn=bn, bk=bk, c_stages=2, acc_stages=acc_stages
                    )
                    # Include a shallow ring so the tuner can choose occupancy
                    # over latency hiding. Geometry and memory capacity decide
                    # these seeds; no workload name or problem-size table does.
                    for stages in dict.fromkeys((min(2, maximum), maximum)):
                        if stages <= 0:
                            continue
                        seeds.append(
                            Config(
                                block_sizes=block_sizes,
                                pid_type="persistent_blocked",
                                tcgen05_cta_group="two",
                                tcgen05_cluster_m=2,
                                tcgen05_cluster_n=1,
                                tcgen05_ab_stages=stages,
                                tcgen05_c_stages=2,
                                tcgen05_acc_stages=acc_stages,
                                tcgen05_persistence_model="static_persistent",
                                tcgen05_strategy="role_local_monolithic",
                                tcgen05_layout_strategy="default",
                                tcgen05_tvm_ffi_launch=False,
                                tcgen05_flat_role_coordinates=False,
                                **dict.fromkeys(TCGEN05_LAYOUT_OVERRIDES_KEYS),
                            )
                        )
        return seeds

    def _small_grid_seed_configs(self) -> list[Config]:
        """One-CTA seeds for problems the standard 128x128 tiling cannot fill.

        A GEMM with fewer 128x128 output tiles than SMs is bound by per-CTA
        fixed latency and by each SM's L2->SMEM bandwidth, not by MMA
        throughput. Small one-CTA tiles spread the operand re-reads over more
        SMs (the cuBLAS heuristic picks 64x8 .. 64x32 tiles there); the paired
        two-CTA seeds above cover the machine-filling shapes. The tiles are
        chosen from the problem's geometry only: whole tiles, the widest K
        packet the search admits, and an AB ring no deeper than the K loop.

        Where the search admits both 64- and 128-row tiles, the 64-row tiles
        are seeded only while their grid stays within one wave: the row split
        halves each CTA's A tile and its share of the cold-L2 fetch (fp8 256^3
        on 64x16 tiles: 2.8 -> 2.6 us device span, cuBLAS's 64x8 tiles 2.6),
        while a second wave of narrow tiles only re-runs the per-CTA prologue.
        """
        extents = self.matmul_compile_time_static_extents
        fragments = self._matmul_block_fragments()
        if (
            not self.search_enabled
            or self.device_sm_count <= 0
            or extents is None
            or fragments is None
            or any(value is None or value <= 0 for value in extents)
            or "persistent_interleaved" not in self.allowed_pid_types
        ):
            return []
        m, n, k = cast("tuple[int, int, int]", extents)
        bm_fragment, bn_fragment, bk_fragment = fragments
        # A leading passthrough (batch) axis multiplies the tile count: the
        # batched 4 x 64x128x128 GEMM has four 128x128 tiles and wants the
        # same 64x{16,32,64} spread (cuBLAS runs it as 64 CTAs of 64x8).
        batch = self.matmul_leading_work_multiplier
        standard_tiles = batch * -(-m // 128) * -(-n // 128)
        if standard_tiles >= self.device_sm_count:
            return []
        bk = bk_fragment.high
        while bk >= bk_fragment.low and (k % bk or bk % 16):
            bk //= 2
        if bk < bk_fragment.low or bk < 16:
            return []
        rowvec_aux_only = (
            self.aux_kernel_detected and not self.exact_shape_aux_kernel_detected
        )
        seeds: list[Config] = []
        wide_rows_admitted = bm_fragment.low <= 128 <= bm_fragment.high and m % 128 == 0
        for bm in (64, 128):
            if not bm_fragment.low <= bm <= bm_fragment.high or m % bm:
                continue
            narrow_rows_need_one_wave = bm == 64 and wide_rows_admitted
            for bn in (16, 32, 64):
                if not bn_fragment.low <= bn <= bn_fragment.high or n % bn:
                    continue
                if (
                    narrow_rows_need_one_wave
                    and batch * (m // bm) * (n // bn) > self.device_sm_count
                ):
                    continue
                block_sizes = self._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
                assert block_sizes is not None
                fit = self.max_ab_stages_that_fit(bm=bm, bn=bn, bk=bk, cluster_m=1)
                stages = max(1, min(k // bk, fit if fit > 0 else 2))
                seed_config: dict[str, Any] = {
                    "block_sizes": block_sizes,
                    "pid_type": "persistent_interleaved",
                    "tcgen05_cta_group": "auto",
                    "tcgen05_cluster_m": 1,
                    "tcgen05_cluster_n": 1,
                    "tcgen05_ab_stages": stages,
                    "tcgen05_c_stages": 2,
                    "tcgen05_acc_stages": 2,
                    "tcgen05_persistence_model": "static_persistent",
                    "tcgen05_strategy": "role_local_monolithic",
                    "tcgen05_layout_strategy": "default",
                    **dict.fromkeys(TCGEN05_LAYOUT_OVERRIDES_KEYS),
                }
                # A row-vector epilogue (bias) on a one-tile-per-CTA kernel has
                # nothing to hide the row's cold-L2 load behind unless it is
                # issued before the accumulator wait: the promoted row stage
                # loads it at the start of the epilogue role (fp16 256^3 +
                # bias: 4.0 -> 3.0 us device time with ``pre_acc_wait``).
                if rowvec_aux_only and self.rowvec_aux_stage_fits(
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    cluster_m=1,
                    ab_stages=stages,
                    c_stages=2,
                ):
                    seed_config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                        TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                    )
                seeds.append(Config(**seed_config))
                if bn <= TCGEN05_SMALL_GRID_DIRECT_STORE_MAX_BN:
                    # Narrow tiles store one or two 16-byte chunks per thread:
                    # the register-direct store beats the SMEM-staged TMA
                    # store's fixed cost there (fp8 256^3, 128x16 tiles:
                    # 3.25 -> 2.88 us device span, cuBLAS 2.91), while wide
                    # tiles keep the TMA store's coalescing.  Seed the twin and
                    # let the timer decide.
                    direct_seed_config: dict[str, Any] = dict(seed_config)
                    direct_seed_config[TCGEN05_C_STORE_MODE_CONFIG_KEY] = (
                        TCGEN05_C_STORE_MODE_DIRECT
                    )
                    seeds.append(Config(**direct_seed_config))
        return seeds

    def _warp_mma_problem(
        self,
    ) -> tuple[int, int, int, torch.dtype] | None:
        """``(m, n, k, dtype)`` of the admitted register-MMA GEMM, or None."""
        extents = self.matmul_compile_time_static_extents
        dtype = self.matmul_input_dtype
        if (
            not self.search_enabled
            or not self.warp_mma_admitted
            or extents is None
            or dtype is None
            or any(value is None or value <= 0 for value in extents)
        ):
            return None
        m, n, k = cast("tuple[int, int, int]", extents)
        return m, n, k, dtype

    def _warp_mma_seed_bk(self, k: int, dtype: torch.dtype) -> int | None:
        """The widest K chunk the K fragment admits that divides K."""
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bk_fragment = fragments[2]
        k_step = warp_mma_k_step(dtype)
        bk = bk_fragment.high
        while bk > k_step and (k % bk or bk % k_step):
            bk //= 2
        if k % bk or bk % k_step or bk < bk_fragment.low or bk > k:
            return None
        return bk

    def _warp_mma_seed_configs(self) -> list[Config]:
        """Register-MMA tiles for the admitted latency-bound GEMMs.

        One-warp 16x8 tiles (the Helion-Triton fp8 256^3 winner's structure:
        2.24 us vs cuBLAS 2.62) and the four-warp 64x8 / 32x32 and two-warp
        32x16 tiles that won the fp16 256^3 + bias and (4, 64, 128, 128)
        probes (2.64 vs 2.88, 2.19 vs 2.66); the timer decides against the
        tcgen05 small-grid seeds.
        """
        problem = self._warp_mma_problem()
        fragments = self._matmul_block_fragments()
        if problem is None or fragments is None:
            return []
        m, n, k, dtype = problem
        bk = self._warp_mma_seed_bk(k, dtype)
        if bk is None:
            return []
        bm_fragment, bn_fragment, _bk_fragment = fragments
        leading_index = self._config_block_index(self.matmul_leading_block_id)
        seeds: list[Config] = []
        for bm, bn, warps in WARP_MMA_SEED_TILES:
            if not (
                bm_fragment.low <= bm <= bm_fragment.high
                and bn_fragment.low <= bn <= bn_fragment.high
            ):
                continue
            if (
                warp_mma_tile_reason(
                    m=m, n=n, k=k, bm=bm, bn=bn, bk=bk, warps=warps, dtype=dtype
                )
                is not None
            ):
                continue
            block_sizes = self._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
            assert block_sizes is not None
            if leading_index is not None:
                block_sizes[leading_index] = 1
            seed_config: dict[str, Any] = {
                "block_sizes": block_sizes,
                WARP_MMA_FAMILY_KEY: MATMUL_FAMILY_WARP_MMA,
                WARP_MMA_WARPS_KEY: warps,
            }
            seeds.append(Config(**seed_config))
        return seeds

    def _pin_warp_mma_inert_knobs(self, config: dict[str, object]) -> None:
        """Pin every knob the register-MMA body ignores to its default so equal
        kernels share one autotune identity: the tcgen05 pipeline knobs, the
        rasterization knobs of the (replaced) Helion grid, the per-block thread /
        vector / lane-layout knobs, the indexing choice and the occupancy hint."""
        for key, fragment in self.optional_fragments(for_search=True).items():
            if key in (WARP_MMA_FAMILY_KEY, WARP_MMA_WARPS_KEY):
                continue
            if key in config:
                config[key] = fragment.default()
        spec = self.config_spec
        if "l2_groupings" in config:
            config["l2_groupings"] = [1 for _ in spec.l2_groupings]
        if "loop_orders" in config:
            config["loop_orders"] = [item._fill_missing() for item in spec.loop_orders]
        for key, default in (
            ("num_threads", 0),
            ("cute_vector_widths", 1),
            ("cute_lane_layouts", "blocked"),
            ("indexing", "pointer"),
        ):
            value = config.get(key)
            if isinstance(value, list):
                config[key] = [default for _ in value]
        if "cute_min_blocks_per_mp" in config:
            config["cute_min_blocks_per_mp"] = 0

    @staticmethod
    def _warp_mma_largest_divisor(
        extent: int, requested: int, *, low: int, high: int
    ) -> int | None:
        """The largest power of two in ``[low, min(requested, high)]`` dividing ``extent``."""
        value = 1 << (max(min(requested, high), 1).bit_length() - 1)
        while value >= low:
            if extent % value == 0:
                return value
            value //= 2
        return None

    def _normalize_warp_mma_family(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """Keep the register-MMA family and the tcgen05 tiles mutually legal.

        ``cute_matmul_family="warp_mma"`` needs the admitted problem, a tile
        the family can run (16..64 rows, 8..64 columns, a K chunk of whole
        mma steps dividing K, a warp count tiling the block) and a unit batch
        block; its tcgen05-only knobs are inert and pinned to their defaults
        so equal kernels share one identity.  The tcgen05 family keeps its
        own block floors, which the admitted family's search widened.
        """
        if not self.search_enabled:
            return
        family = config.get(WARP_MMA_FAMILY_KEY)
        if family is not None and family not in MATMUL_FAMILIES:
            if fix_invalid:
                config[WARP_MMA_FAMILY_KEY] = MATMUL_FAMILY_TCGEN05
                family = MATMUL_FAMILY_TCGEN05
            else:
                raise InvalidConfig(
                    f"{WARP_MMA_FAMILY_KEY} must be one of {list(MATMUL_FAMILIES)!r}, "
                    f"got {family!r}"
                )
        view = self._matmul_config_view(config)
        if family == MATMUL_FAMILY_WARP_MMA:
            problem = self._warp_mma_problem()
            if problem is None or view is None:
                if not fix_invalid:
                    raise InvalidConfig(
                        f"{WARP_MMA_FAMILY_KEY}={family!r} is not admitted for this "
                        "kernel (the register-MMA family needs one static, "
                        "latency-bound dense GEMM with K-contiguous A)"
                    )
                config[WARP_MMA_FAMILY_KEY] = MATMUL_FAMILY_TCGEN05
                family = MATMUL_FAMILY_TCGEN05
            else:
                m, n, k, dtype = problem
                block_sizes, m_index, n_index, k_index = view
                bm, bn, bk = (
                    block_sizes[m_index],
                    block_sizes[n_index],
                    block_sizes[k_index],
                )
                warps = config.get(WARP_MMA_WARPS_KEY, WARP_MMA_DEFAULT_WARPS)
                values_are_ints = all(
                    type(value) is int for value in (bm, bn, bk, warps)
                )
                reason = (
                    warp_mma_tile_reason(
                        m=m,
                        n=n,
                        k=k,
                        bm=cast("int", bm),
                        bn=cast("int", bn),
                        bk=cast("int", bk),
                        warps=cast("int", warps),
                        dtype=dtype,
                    )
                    if values_are_ints
                    else "tile sizes and the warp count must be integers"
                )
                if reason is not None:
                    if not fix_invalid:
                        raise InvalidConfig(
                            f"{WARP_MMA_FAMILY_KEY}={family!r}: {reason}"
                        )
                    k_step = warp_mma_k_step(dtype)
                    new_bm = self._warp_mma_largest_divisor(
                        m,
                        bm if type(bm) is int else WARP_MMA_MAX_BM,
                        low=WARP_MMA_MIN_BM,
                        high=WARP_MMA_MAX_BM,
                    )
                    new_bn = self._warp_mma_largest_divisor(
                        n,
                        bn if type(bn) is int else WARP_MMA_MAX_BN,
                        low=WARP_MMA_MIN_BN,
                        high=WARP_MMA_MAX_BN,
                    )
                    new_bk = self._warp_mma_largest_divisor(
                        k, bk if type(bk) is int else k, low=k_step, high=k
                    )
                    if new_bk is not None and new_bk % k_step:
                        new_bk = None
                    if new_bm is None or new_bn is None or new_bk is None:
                        config[WARP_MMA_FAMILY_KEY] = MATMUL_FAMILY_TCGEN05
                        family = MATMUL_FAMILY_TCGEN05
                    else:
                        block_sizes[m_index] = new_bm
                        block_sizes[n_index] = new_bn
                        block_sizes[k_index] = new_bk
                        requested = (
                            warps if type(warps) is int else WARP_MMA_DEFAULT_WARPS
                        )
                        legal = [
                            choice
                            for choice in WARP_MMA_WARP_CHOICES
                            if choice <= requested
                            and warp_mma_warp_layout(new_bm, new_bn, choice) is not None
                        ]
                        config[WARP_MMA_WARPS_KEY] = (
                            max(legal)
                            if legal
                            else warp_mma_max_legal_warps(new_bm, new_bn)
                        )
                if family == MATMUL_FAMILY_WARP_MMA:
                    leading_index = self._config_block_index(
                        self.matmul_leading_block_id
                    )
                    if leading_index is not None and block_sizes[leading_index] != 1:
                        if not fix_invalid:
                            raise InvalidConfig(
                                f"{WARP_MMA_FAMILY_KEY}={family!r} requires a batch block of 1"
                            )
                        block_sizes[leading_index] = 1
                    # One CTA per output tile on the flat grid: the persistent
                    # program-id strategies wrap the device body in a
                    # ``virtual_pid`` loop (shared memory allocated inside a
                    # dynamic loop, a ``_NUM_SM`` parameter the body ignores).
                    if "flat" not in self.allowed_pid_types:
                        if not fix_invalid:
                            raise InvalidConfig(
                                f"{WARP_MMA_FAMILY_KEY}={family!r} requires the flat "
                                "program-id grid, which this kernel disallows"
                            )
                        config[WARP_MMA_FAMILY_KEY] = MATMUL_FAMILY_TCGEN05
                        family = MATMUL_FAMILY_TCGEN05
                    else:
                        config["pid_type"] = "flat"
                        self._pin_warp_mma_inert_knobs(config)
                        return
        # tcgen05 family (explicit or implied): the admitted register-MMA
        # search widened the M / N floors; clamp them back the way the block
        # floors always clamped an under-sized request (``BlockSizeSpec``
        # raises a value below ``min_size`` silently, so a pinned
        # ``block_sizes=[16, 16, 32]`` on a tcgen05 kernel keeps running the
        # 64-row tile it always ran).
        if view is None or not self.warp_mma_admitted:
            return
        block_sizes, m_index, n_index, _k_index = view
        for index, floor in (
            (m_index, self.tcgen05_min_search_m),
            (n_index, self.tcgen05_min_search_n),
        ):
            value = block_sizes[index]
            if type(value) is int and value < floor:
                block_sizes[index] = floor
        if fix_invalid and WARP_MMA_WARPS_KEY in config:
            config[WARP_MMA_WARPS_KEY] = WARP_MMA_DEFAULT_WARPS

    def _one_wave_two_cta_tile_is_valid(self, bm: object, bn: object) -> bool:
        """Whole 256 x {128, 64} CtaGroup.TWO tiles of the static problem."""
        extents = self.matmul_compile_time_static_extents
        if extents is None:
            return False
        m, n, _k = extents
        return (
            type(bm) is int
            and type(bn) is int
            and bm == TCGEN05_TWO_CTA_BLOCK_M
            and bn in TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
            and isinstance(m, int)
            and isinstance(n, int)
            and m % bm == 0
            and n % bn == 0
        )

    def _one_wave_cluster_n2_is_valid(self, bn: object) -> bool:
        extents = self.matmul_compile_time_static_extents
        if extents is None or type(bn) is not int or bn <= 0:
            return False
        n = extents[1]
        return (
            not self.matmul_has_leading_passthrough
            and isinstance(n, int)
            and n % bn == 0
            and (n // bn) % 2 == 0
        )

    def _batched_multi_tile_seed_configs(self) -> list[Config]:
        """Persistent two-CTA seeds for batched GEMMs with several tiles per pair.

        cuBLAS runs 16 x 512x768x1024 fp16 as ``nvjet 128x256_64x6_2x2_2cta``:
        the 256x256 pair tile in 2x2 clusters with a six-deep 64-wide K ring,
        and a batch-major grid (batch on grid z) so the four clusters of a
        batch share its A tile across the N peers and its B tile across the
        M clusters at the same time.  The persistent equivalent is the
        ``batch_slowest`` raster at the co-resident 4-CTA cluster count: its
        per-stage cost matches cuBLAS's (394 vs 389 ns per 64-K per tile on
        the K sweep; the batch-fastest 2x2 ring paid 471 and re-fetched B from
        DRAM), 16.3 vs 15.0 us at the campaign shape (2x1 pair 16.9).  The
        search's timer cannot tell these structures apart inside its ~2 us
        quantum at this shape, so the recipe arrives as a seed, next to the
        same ring on the 2x1 pair.  Geometry only: batched 16-bit GEMMs
        whose 256x256 two-CTA grid fills the machine, with even M/N tile
        counts and K in whole 64-wide steps.
        """
        extents = self.matmul_compile_time_static_extents
        fragments = self._matmul_block_fragments()
        dtype = self.matmul_input_dtype
        constraints = self.cluster_m2_search_constraints
        if (
            not self.search_enabled
            or not self.matmul_has_leading_passthrough
            or self.aux_kernel_detected
            or self.device_sm_count <= 0
            or extents is None
            or fragments is None
            or dtype not in (torch.float16, torch.bfloat16)
            or constraints is None
            or constraints.one_wave_only
            or any(value is None or value <= 0 for value in extents)
            or TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types
        ):
            return []
        m, n, k = cast("tuple[int, int, int]", extents)
        bm_fragment, bn_fragment, bk_fragment = fragments
        batch = self.matmul_leading_work_multiplier
        bm, bn, bk = (
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_BLOCK_N,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K,
        )
        if (
            m % bm
            or n % bn
            or k % bk
            or (n // bn) % 2
            or not (bm_fragment.low <= bm <= bm_fragment.high)
            or not (bn_fragment.low <= bn <= bn_fragment.high)
            or not (bk_fragment.low <= bk <= bk_fragment.high)
            or not self.cluster_m2_bk_is_valid(bk, constraints)
            or batch * (m // bm) * (n // bn) * 2 < self.device_sm_count
        ):
            return []
        ab_stages = 6
        if not self.ab_stages_three_fits(
            bm=bm, bn=bn, bk=bk, cluster_m=2, ab_stages=ab_stages
        ):
            return []
        block_sizes = self._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
        assert block_sizes is not None
        seeds: list[Config] = []
        for cluster_n in (2, 1):
            seed_config: dict[str, Any] = {
                "block_sizes": block_sizes,
                "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
                "tcgen05_cta_group": "auto",
                "tcgen05_cluster_m": 2,
                "tcgen05_cluster_n": cluster_n,
                "tcgen05_ab_stages": ab_stages,
                "tcgen05_c_stages": 2,
                "tcgen05_acc_stages": 2,
                "l2_groupings": [1],
                TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
                TCGEN05_BATCH_RASTER_CONFIG_KEY: TCGEN05_BATCH_RASTER_SLOWEST,
                "tcgen05_persistence_model": "static_persistent",
                "tcgen05_strategy": "role_local_monolithic",
                "tcgen05_layout_strategy": "default",
                **dict.fromkeys(TCGEN05_LAYOUT_OVERRIDES_KEYS),
            }
            seeds.append(Config(**seed_config))
        return seeds

    def _one_wave_seed_configs(self) -> list[Config]:
        """One-tile-per-CTA seeds for GEMMs whose 256x256 grid leaves SMs idle.

        cuBLAS's small-shape heuristic (fp16 1024^3, batched 8x256x256x512,
        fp8 512x1024x512) picks 2-CTA 128x64 / 256x64 tiles in 2x2 clusters
        with a ten-deep 64-wide K ring: every SM gets one tile and the ring
        keeps the whole K loop in flight.  The search's timer cannot separate
        the small-tile families inside its ~2 us quantum, so the recipe has
        to arrive as a seed.  Geometry only: the one-CTA tile and the
        two-CTA tile whose one-tile-per-CTA grid comes closest to the SM
        count without exceeding it, at the narrowest K step the search
        admits, with the deepest AB ring that fits the SMEM budget (no deeper
        than the K loop).  fp16 1024^3 with a bias epilogue: the seeded
        2-CTA 256x64x64 ring of 9 runs 6.24 us vs 7.8 for the one-CTA
        128x64x128 winner the unseeded search found (cuBLAS 5.6).
        """
        extents = self.matmul_compile_time_static_extents
        fragments = self._matmul_block_fragments()
        dtype = self.matmul_input_dtype
        if (
            not self.search_enabled
            or self.device_sm_count <= 0
            or extents is None
            or fragments is None
            or dtype is None
            or any(value is None or value <= 0 for value in extents)
            or (self.aux_kernel_detected and self.exact_shape_aux_kernel_detected)
            or "persistent_interleaved" not in self.allowed_pid_types
        ):
            return []
        m, n, k = cast("tuple[int, int, int]", extents)
        bm_fragment, bn_fragment, bk_fragment = fragments
        batch = self.matmul_leading_work_multiplier
        sm_count = self.device_sm_count
        if batch * -(-m // 256) * -(-n // 256) * 2 >= sm_count:
            # The 256x256 two-CTA seeds already fill the machine.
            return []
        dtype_bytes = torch.empty((), dtype=dtype).element_size()
        bk_choices = (64, 128) if dtype_bytes >= 2 else (128, 64)
        bk = next(
            (
                value
                for value in bk_choices
                if k % value == 0 and bk_fragment.low <= value <= bk_fragment.high
            ),
            None,
        )
        if bk is None:
            return []
        rowvec_aux_only = (
            self.aux_kernel_detected and not self.exact_shape_aux_kernel_detected
        )
        seeds: list[Config] = []

        def emit(bm: int, bn: int, cluster_m: int, cluster_n: int) -> None:
            block_sizes = self._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
            assert block_sizes is not None
            fit = self.max_ab_stages_that_fit(bm=bm, bn=bn, bk=bk, cluster_m=cluster_m)
            stages = max(1, min(k // bk, fit if fit > 0 else 2))
            seed_config: dict[str, Any] = {
                "block_sizes": block_sizes,
                "pid_type": "persistent_interleaved",
                "tcgen05_cta_group": "auto",
                "tcgen05_cluster_m": cluster_m,
                "tcgen05_cluster_n": cluster_n,
                "tcgen05_ab_stages": stages,
                "tcgen05_c_stages": 2,
                "tcgen05_acc_stages": 2,
                "l2_groupings": [1],
                "tcgen05_l2_swizzle_size": 1,
                "tcgen05_persistence_model": "static_persistent",
                "tcgen05_strategy": "role_local_monolithic",
                "tcgen05_layout_strategy": "default",
                **dict.fromkeys(TCGEN05_LAYOUT_OVERRIDES_KEYS),
            }
            if rowvec_aux_only and self.rowvec_aux_stage_fits(
                bm=bm,
                bn=bn,
                bk=bk,
                cluster_m=cluster_m,
                ab_stages=stages,
                c_stages=2,
            ):
                seed_config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                    TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                )
            seeds.append(Config(**seed_config))

        def in_range(fragment: BlockSizeFragment, value: int) -> bool:
            return fragment.low <= value <= fragment.high

        one_cta: tuple[tuple[int, int], int, int] | None = None
        for bm in (128, 64):
            for bn in (128, 64, 32):
                if (
                    m % bm
                    or n % bn
                    or not in_range(bm_fragment, bm)
                    or not in_range(bn_fragment, bn)
                ):
                    continue
                ctas = batch * (m // bm) * (n // bn)
                if ctas > sm_count:
                    continue
                key = (ctas, bm * bn)
                if one_cta is None or key > one_cta[0]:
                    one_cta = (key, bm, bn)
        if one_cta is not None:
            emit(one_cta[1], one_cta[2], 1, 1)
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is not None
            and constraints.allow_one_wave_tiles
            and self.cluster_m2_bk_is_valid(bk, constraints)
            and m % TCGEN05_TWO_CTA_BLOCK_M == 0
            and in_range(bm_fragment, TCGEN05_TWO_CTA_BLOCK_M)
        ):
            two_cta: tuple[int, int] | None = None
            for bn in TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS:
                if n % bn or not in_range(bn_fragment, bn):
                    continue
                ctas = batch * (m // TCGEN05_TWO_CTA_BLOCK_M) * (n // bn) * 2
                if ctas > sm_count:
                    continue
                if two_cta is None or ctas > two_cta[0]:
                    two_cta = (ctas, bn)
            if two_cta is not None:
                emit(TCGEN05_TWO_CTA_BLOCK_M, two_cta[1], 2, 1)
                if self._one_wave_cluster_n2_is_valid(two_cta[1]):
                    emit(TCGEN05_TWO_CTA_BLOCK_M, two_cta[1], 2, 2)
        return seeds

    def _one_cta_persistent_parameters(
        self, config: dict[str, object]
    ) -> tuple[int, int, int] | None:
        """Plain one-CTA static-persistent role-local family (cluster 1x1).

        The same role-local ``while`` builder that emits the two-CTA TMA-role
        dependency wait serves this family, so its TMA warp can wait on a
        programmatic dependency before its first operand read.
        """
        view = self._matmul_config_view(config)
        if (
            view is None
            or config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto") != "auto"
            or config.get("tcgen05_cluster_m", 1) != 1
            or config.get("tcgen05_cluster_n", 1) != 1
            or config.get("pid_type")
            not in (
                "persistent_blocked",
                "persistent_interleaved",
            )
            or config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is not None
            or config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY)
            or config.get(TCGEN05_STRATEGY_CONFIG_KEY, "role_local_monolithic")
            != "role_local_monolithic"
            or config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, "static_persistent")
            != "static_persistent"
            or config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, "default") != "default"
            or any(config.get(key) is not None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
            or any(
                config.get(key, 0) != 0
                for key in (
                    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
                    TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
                    TCGEN05_WARP_SPEC_STORE_WARPS_KEY,
                )
            )
        ):
            return None
        blocks, mi, ni, ki = view
        bm, bn, bk = blocks[mi], blocks[ni], blocks[ki]
        if (
            bm not in (64, 128)
            or not isinstance(bn, int)
            or not isinstance(bk, int)
            or not 8 <= bn <= 256
            or bn % 8
            or not 16 <= bk <= 256
            or bk % 16
        ):
            return None
        return cast("tuple[int, int, int]", (bm, bn, bk))

    def materialized_operand_pdl_supported(self, config: dict[str, object]) -> bool:
        # The static persistent role-local schedules (the paired TWO family and
        # the plain one-CTA family) emit a TMA-role dependency wait before all
        # initial/refill operand reads. Alternative/fused schedules do not
        # inherit this proof or the separate producer launch.
        parameters = self._paired_pipeline_parameters(config)
        tiles = (
            parameters[:3]
            if parameters is not None
            else self._one_cta_persistent_parameters(config)
        )
        extents = self.matmul_compile_time_static_extents
        if (
            self.materialized_operand_pdl_roots is None
            or tiles is None
            or extents is None
            or any(
                extent is None or extent % tile != 0
                for extent, tile in zip(extents, tiles, strict=True)
            )
        ):
            return False
        if parameters is not None:
            limit = self._paired_pipeline_stage_limit(config)
            if limit is None or limit <= 0:
                return False
        return (
            config.get(FANOUT_CONFIG_KEY, "off") == "off"
            and config.get("cute_materialized_operand_schedule", "off") == "off"
            and config.get("cute_materialized_schedule", "off") == "off"
            and not config.get("cute_split_k_workspace")
            and config.get("cute_split_k_schedule", "legacy") == "legacy"
        )

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
        if isinstance(fragment, ListOf):
            if not isinstance(value, list) or len(value) != fragment.length:
                raise InvalidConfig(
                    f"{name} requires a list of length {fragment.length}"
                )
            return [
                CuteTcgen05Config._validate_optional_fragment_value(
                    f"{name}[{index}]", fragment.inner, item
                )
                for index, item in enumerate(value)
            ]
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
        allow_one_wave_tiles: bool = False,
        one_wave_only: bool = False,
    ) -> None:
        assert static_k > 0, "static_k is required for cluster_m=2 K-cap checks"
        assert max_k_tiles > 0, "cluster_m=2 max K tiles must be positive"
        assert allow_one_wave_tiles or not one_wave_only
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
            allow_fp8_small_grid=allow_fp8_small_grid,
            allow_one_wave_tiles=allow_one_wave_tiles,
            one_wave_only=one_wave_only,
        )
        self.restrict_cluster_m_search((1, 2))

    @staticmethod
    def cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        if bk <= 0:
            return False
        if constraints.allow_edge_k_tail_family:
            # bk=64 halves the per-stage AB SMEM so a 16-bit edge kernel can
            # run a 5-deep AB pipeline (bf16 5000^3: 948 vs 815 TFLOP/s at the
            # bk=128 seed); the K tail stays a clamped TMA box either way.
            return (
                bk
                in (
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K,
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

    def _m_pair_block_m_is_valid(self) -> bool:
        """Whether block_m=512 (M-paired tiles) is admissible for this kernel.

        The codegen envelope is the plain 16-bit full-tile static family:
        M divisible by 512 with the canonical 256-wide N tile (the fix pass
        projects bn to 256 and validates bk). Aux kernels and edge families
        keep block_m=256.
        """
        if self.aux_kernel_detected or self.matmul_has_leading_passthrough:
            return False
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or constraints.one_wave_only
        ):
            return False
        for fact in self.config_spec.matmul_facts:
            if (
                fact.static_m is not None
                and fact.static_m % (2 * TCGEN05_TWO_CTA_BLOCK_M) == 0
                and fact.static_n is not None
                and fact.static_n % TCGEN05_TWO_CTA_BLOCK_N == 0
                and fact.lhs_dtype in (torch.float16, torch.bfloat16)
            ):
                return True
        return False

    def full_tile_direct_entry_seed_bk(self) -> int | None:
        """Largest valid full-tile direct-entry K tile for the live shape.

        Mirrors the full-tile branch of the heuristic bk selection: the highest
        power-of-two ``bk`` within the K block-size fragment that divides
        ``static_k`` within the ``max_k_tiles`` cap.
        """
        constraints = self.cluster_m2_search_constraints
        fragments = self._matmul_block_fragments()
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or constraints.one_wave_only
            or fragments is None
        ):
            return None
        bk_fragment = fragments[2]
        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if self.cluster_m2_bk_is_valid(bk, constraints):
                return bk
            bk //= 2
        return None

    def full_tile_direct_entry_seed_eligible(self) -> bool:
        """Structural eligibility for the generalized TVM-FFI direct-entry seed.

        The direct-entry codegen + runtime validator build their A/B/D TMA
        descriptors from the runtime tensor shapes, so the fast launch path is
        shape-general; the constraints are purely structural: a full-tile (NOT
        edge+K-tail) CtaGroup.TWO 16-bit (bf16/fp16) GEMM, the 256x256 CTA tile
        reachable, a direct-entry-valid ``bk``, and the ``(ab=3, c=2)`` stage
        tuple admitted
        and SMEM-fitting. Both ``optional_fragments`` (to add the FFI search
        surface) and ``CuteTcgen05ClusterM2FfiHeuristic`` (to gate the seed) use
        this so they cannot disagree.
        """
        # A non-tcgen05-native matmul operand (e.g. an int16 tensor cast to
        # bf16) forces the dot through the non-tcgen05 fallback, where the
        # flat-role / FFI seed config is rejected.
        if self.matmul_has_non_tcgen05_operand:
            return False
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or constraints.one_wave_only
        ):
            return False
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return False
        # The direct-entry TMA descriptors, SMEM layout, and epilogue tile are
        # dtype-general for any 16-bit operand (the byte math keys on
        # ``dtype_bytes``, and bf16/fp16 are both 2 bytes), so admit bf16 and
        # fp16 with matching operand dtypes. fp32 stays excluded (no tcgen05
        # fp32 SMEM-staged MMA path).
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
        if not tcgen05_direct_entry_stage_tuple_allowed(
            bk=bk, ab_stage_count=3, c_stage_count=2
        ):
            return False
        return self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            cluster_m=2,
        )

    def full_tile_direct_entry_seed_config(
        self, *, l2_groupings: list[object] | None = None
    ) -> Config | None:
        """Generalized TVM-FFI direct-entry seed config for the live shape.

        Single source of truth for the FFI ``explicit_epi_tile`` + flat-role +
        ``tvm_ffi_launch`` config: emitted into the autotuner population by
        ``CuteTcgen05ClusterM2FfiHeuristic`` AND used by
        ``_fix_target1_tvm_ffi_search_config`` to project FFI-requesting search
        candidates onto the validated CtaGroup.TWO envelope (this is what the
        per-shape ``_target{N}`` seeds used to do, generalized to any eligible
        shape). Returns ``None`` for ineligible shapes.
        """
        if not self.full_tile_direct_entry_seed_eligible():
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
        # A materializer or another root owns its own grouping coordinate.
        # Preserve those values when projecting an existing configuration.
        l2_groupings = (
            [item._fill_missing() for item in self.config_spec.l2_groupings]
            if l2_groupings is None
            else list(l2_groupings)
        )
        assert self.matmul_block_ids is not None
        l2_groupings[
            self.config_spec.l2_groupings.block_id_to_index(self.matmul_block_ids[0])
        ] = 2
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "l2_groupings": l2_groupings,
            "num_warps": 8,
            "num_stages": 4,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 3,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            # The flat-role launch path uses this fixed explicit subtile.
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
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
        if constraints is None or constraints.one_wave_only:
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

    def _plain_clc_seed_config(self) -> Config | None:
        """Autotune seed for the plain full-tile cluster_m=2 CLC family.

        Mirrors ``_c_input_seed_config``'s full-tile branch but for kernels
        without aux operands: ROLE_LOCAL_WITH_SCHEDULER + a scheduler warp
        driving CLC dynamic persistence, no C-input warp. Besides giving the
        search a head start, this seed widens the strategy/scheduler-warps/
        persistence fragments via the compiler-seed mechanism so neighboring
        configs stay explorable.
        """
        if not self._plain_clc_persistence_search_enabled():
            return None
        if not self._clc_persistence_search_enabled():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.one_wave_only:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bm_fragment, bn_fragment, bk_fragment = fragments
        if not (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
        ):
            return None
        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if self.cluster_m2_bk_is_valid(bk, constraints):
                break
            bk //= 2
        else:
            return None
        ab_stages = (
            3
            if self.ab_stages_three_fits(
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=bk,
                cluster_m=2,
            )
            else 2
        )
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            # L2 grouping matters even under the hardware scheduler: CLC
            # preserves the interleaved rasterization order it cancels into,
            # and l2_groupings=[1] costs ~13% at fp16 16384^3 (780 vs 899
            # TFLOP/s pinned) vs the measured-good [4].
            "l2_groupings": [4],
            "tcgen05_cluster_m": 2,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": ab_stages,
            TCGEN05_STRATEGY_CONFIG_KEY: (
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            ),
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.CLC_PERSISTENT.value
            ),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 1,
        }
        if self.config_spec.indexing.length == 3:
            seed_config["indexing"] = ["tensor_descriptor"] * 3
        return Config(**seed_config)

    def _plain_m_pair_seed_config(self) -> Config | None:
        """Autotune seed for block_m=512 M-paired tiles (nvjet's B-reuse).

        Two 256-row CtaGroup.TWO subtiles share each K stage's B buffer,
        halving B's SMEM/L2/DRAM traffic. Measured fp16 16384^3: 986-990
        TFLOP/s vs the block_m=256 CLC winner's 908 (co-timed vs cuBLAS's
        971-981). The doubled A staging fits ab=2 at bk=128 under the B200
        SMEM cap. Rides the CLC WITH_SCHEDULER shape when enabled, else the
        static persistent default.
        """
        if not self._m_pair_block_m_is_valid():
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.one_wave_only:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bm_fragment, bn_fragment, bk_fragment = fragments
        pair_bm = 2 * TCGEN05_TWO_CTA_BLOCK_M
        if not (
            bm_fragment.low <= pair_bm <= bm_fragment.high
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
        ):
            return None
        bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
        if not (bk_fragment.low <= bk <= bk_fragment.high):
            return None
        if not self.cluster_m2_bk_is_valid(bk, constraints):
            return None
        seed_config: dict[str, Any] = {
            "block_sizes": [pair_bm, TCGEN05_TWO_CTA_BLOCK_N, bk],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "l2_groupings": [4],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": 2,
            "tcgen05_acc_stages": 2,
        }
        if self._plain_clc_persistence_search_enabled() and (
            self._clc_persistence_search_enabled()
        ):
            seed_config[TCGEN05_STRATEGY_CONFIG_KEY] = (
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            )
            seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
                Tcgen05PersistenceModel.CLC_PERSISTENT.value
            )
            seed_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        if self.config_spec.indexing.length == 3:
            seed_config["indexing"] = ["tensor_descriptor"] * 3
        return Config(**seed_config)

    def _plain_edge_deep_ab_stages(self, bk: int) -> int:
        """Deepest AB pipeline that leaves room for the TMA-store epilogue's
        C ring on the 256x256 cluster_m=2 tile.

        Overflowing the C ring silently demotes the kernel to the much slower
        all-SIMT store (ab=6 fits bare AB at bk=64 but not AB + C, and
        measures far below ab=5 + TMA store), so walk down from the bare-AB
        maximum until AB + C fits.
        """
        ab_stages = self.max_ab_stages_that_fit(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            cluster_m=2,
        )
        while (
            ab_stages > TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES
            and not self.c_stages_fits(
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=bk,
                cluster_m=2,
                ab_stages=ab_stages,
                c_stages=2,
                has_source_c=False,
            )
        ):
            ab_stages -= 1
        return ab_stages

    def _plain_edge_seed_config(self) -> Config | None:
        """Autotune seed for the plain (no-aux) edge+K-tail cluster_m=2 family.

        The deep-AB edge regime: bk=64 halves per-stage AB SMEM so a 16-bit
        kernel fits a 5-stage AB pipeline (bf16 5000^3: 948 TFLOP/s vs 815 at
        the bk=128/ab=2 projection). Partial stripes and the K tail stay
        clamped TMA boxes; the TMA-store epilogue covers fringe output tiles
        via descriptor clamping. Seeding also widens the ab-stages fragment
        via the compiler-seed mechanism so nearby depths stay explorable.
        """
        if self.aux_kernel_detected or self.matmul_has_leading_passthrough:
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or not constraints.allow_edge_k_tail_family:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bm_fragment, bn_fragment, bk_fragment = fragments
        if not (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N
        ):
            return None
        bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
        if not (
            bk_fragment.low <= bk <= bk_fragment.high
            and self.cluster_m2_bk_is_valid(bk, constraints)
        ):
            bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
            if not (
                bk_fragment.low <= bk <= bk_fragment.high
                and self.cluster_m2_bk_is_valid(bk, constraints)
            ):
                return None
        ab_stages = self._plain_edge_deep_ab_stages(bk)
        if ab_stages <= 0:
            return None
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "l2_groupings": [4],
            "tcgen05_cluster_m": 2,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": ab_stages,
        }
        if self._clc_persistence_search_enabled():
            # CLC dynamic persistence also wins on the deep-AB edge family
            # (bf16 5000^3: 883 vs 859 TFLOP/s static, same-GPU pair).
            seed_config[TCGEN05_STRATEGY_CONFIG_KEY] = (
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            )
            seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
                Tcgen05PersistenceModel.CLC_PERSISTENT.value
            )
            seed_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        if self.config_spec.indexing.length == 3:
            seed_config["indexing"] = ["tensor_descriptor"] * 3
        return Config(**seed_config)

    def _plain_cluster_n2_seed_config(self, *, deep: bool = False) -> Config | None:
        """Autotune seed for the 4-CTA (cluster 2x2) multicast family.

        B multicast on top of the 2-CTA A multicast halves B traffic, which
        wins on B-heavy full-tile shapes (fp16 4096x4096x32768: 911 vs 824
        TFLOP/s); cuBLAS's nvjet picks 2x2 clusters for the same shape. This
        seeds the static-persistent monolithic family; a CLC-persistent
        variant is layered on in ``autotune_seed_configs`` and the search's
        terminal refinement arbitrates between them.

        ``deep`` is nvjet's ``64x6`` staging of the same tile: bk=64 with a
        six-deep AB ring (same SMEM as bk=128/ab=3). The finer stages land the
        first MMA earlier and shorten the drain after the last stage, which is
        the whole difference on one-wave shapes where every CTA runs a single
        tile (fp16 2048x4096x2048 with a bias epilogue: 28.3 vs 29.0 us device
        time, cuBLAS 26.7).

        Row-vector broadcast epilogues (``acc + bias[n]``) ride the same
        family: their per-tile aux is a 512 B row staged once per warp in SMEM,
        so they take the plain seed plus the ``pre_acc_wait`` placement that
        enables the stage. Exact-shape (source-C) epilogues keep their own
        C-input seeds.
        """
        rowvec_aux_only = (
            self.aux_kernel_detected and not self.exact_shape_aux_kernel_detected
        )
        if (
            self.aux_kernel_detected and not rowvec_aux_only
        ) or self.matmul_has_leading_passthrough:
            return None
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or constraints.one_wave_only
        ):
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        fragments = self._matmul_block_fragments()
        if fragments is None:
            return None
        bm_fragment, bn_fragment, bk_fragment = fragments
        if not (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
        ):
            return None
        if deep:
            bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
            ab_stages = self._two_cta_deep_ab_stages(bk)
            if not (
                bk_fragment.low <= bk <= bk_fragment.high
                and self.cluster_m2_bk_is_valid(bk, constraints)
                and self.ab_stages_three_fits(
                    bm=TCGEN05_TWO_CTA_BLOCK_M,
                    bn=TCGEN05_TWO_CTA_BLOCK_N,
                    bk=bk,
                    cluster_m=2,
                    ab_stages=ab_stages,
                )
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
            # ab=3 only where the SMEM-budget gate admits it (B200-class
            # optin); sub-B200 devices keep the known-good ab=2 envelope.
            ab_stages = 3 if self.ab_stages_three_search_constraints is not None else 2
        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "l2_groupings": [4],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 2,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": ab_stages,
        }
        if deep:
            seed_config["tcgen05_c_stages"] = 2
        if rowvec_aux_only and self.rowvec_aux_stage_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            cluster_m=2,
            ab_stages=ab_stages,
            c_stages=2,
        ):
            seed_config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            )
        if self.config_spec.indexing.length == 3:
            seed_config["indexing"] = ["tensor_descriptor"] * 3
        return Config(**seed_config)

    def _two_cta_deep_ab_stages(self, bk: int) -> int:
        """AB depth of the deep (bk=64) 256x256 two-CTA seed.

        nvjet's 64x6 ring for 16-bit operands; fp8 operands halve the
        per-stage SMEM, so the ring goes as deep as the budget admits (12 at
        256x256x64, nvjet's fp8 kernels run 128x6 = the same bytes in flight).
        Row-vector aux kernels whose seed stages the row hand it one AB
        stage where that is what admits the stage
        (``_deep_ab_stages_with_rowvec_stage``).
        """
        constraints = self.ab_stages_three_search_constraints
        if constraints is None or constraints.dtype_bytes != 1:
            ab_stages = TCGEN05_TWO_CTA_DEEP_AB_STAGES
        else:
            fit = self.max_ab_stages_that_fit(
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=bk,
                cluster_m=2,
            )
            ab_stages = max(TCGEN05_TWO_CTA_DEEP_AB_STAGES, fit)
        return self._deep_ab_stages_with_rowvec_stage(ab_stages, bk=bk)

    def _deep_ab_stages_with_rowvec_stage(self, ab_stages: int, *, bk: int) -> int:
        """Make room for the deep seed's ``pre_acc_wait`` row stage.

        The AB-only depth fills the budget the seed gates see, so a row stage
        that the full SMEM model (``rowvec_aux_stage_fits``: AB ring + C ring
        + row stage) does not admit next to it would cost the seed its
        placement. Where one stage less admits the stage, take it: a 32-bit
        row's 4 KiB warp-private stage overflows next to the 192 KiB ring and
        the 32 KiB (128, 64) C ring at ab=6 (fp16) and ab=12 (fp8), and ab=5
        / ab=11 with the stage measured 2.1 us faster than the nominal depth
        without it at 2048x4096x2048 (fp16 33.8 vs 35.8 us, fp8 20.6 vs
        22.7 us; the shallower ring alone costs nothing there, and the same
        holds on the 2x1 twin). Rows that take no stage (unpromoted 16-bit
        rows) and exact-shape epilogues keep the nominal depth, as does a
        stage that one AB stage does not make room for.
        """
        facts = self.rowvec_aux_facts
        if (
            facts is None
            or not self.aux_kernel_detected
            or self.exact_shape_aux_kernel_detected
            or ab_stages <= 1
            or tcgen05_rowvec_stage_smem_bytes(
                rows=facts.rows, bn=TCGEN05_TWO_CTA_BLOCK_N, epi_warps=4
            )
            == 0
        ):
            return ab_stages

        def fits(depth: int) -> bool:
            return self.rowvec_aux_stage_fits(
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=bk,
                cluster_m=2,
                ab_stages=depth,
                c_stages=2,
            )

        if fits(ab_stages) or not fits(ab_stages - 1):
            return ab_stages
        return ab_stages - 1

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
        # Cycle 46: also admit aux-TMA on full-tile cluster_m=2 problems
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
            and not constraints.one_wave_only
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
        if (
            constraints is None
            or constraints.one_wave_only
            or not self.cluster_m2_bk_is_valid(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
                constraints,
            )
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

    def autotune_seed_configs(self) -> list[Config]:
        seeds = self._paired_pipeline_seed_configs()
        seeds.extend(self._small_grid_seed_configs())
        seeds.extend(self._warp_mma_seed_configs())
        seeds.extend(self._one_wave_seed_configs())
        seeds.extend(self._batched_multi_tile_seed_configs())
        plain_clc_seed = self._plain_clc_seed_config()
        if plain_clc_seed is not None:
            seeds.append(plain_clc_seed)
        plain_edge_seed = self._plain_edge_seed_config()
        if plain_edge_seed is not None:
            seeds.append(plain_edge_seed)
            # Edge-family 4-CTA multicast variant: with the generalized CLC
            # broadcast this is quack's winning 5000^3 shape (dynamic
            # persistence + cluster 2x2), measured 907 vs the cluster_n=1
            # edge seed's 852 TFLOP/s.
            edge_n2_seed_config: dict[str, Any] = dict(plain_edge_seed.config)
            edge_n2_seed_config["tcgen05_cluster_n"] = 2
            seeds.append(Config(**edge_n2_seed_config))
        if plain_clc_seed is not None:
            # Deep-staged bk=64 variant of the plain CLC seed: halving the K
            # tile and doubling the AB pipeline depth absorbs UMMA latency on
            # SHORT-K shapes where the per-tile K loop is otherwise too short
            # to hide the drain (fp16 8192x6144x4096: ratio 1.002 vs cuBLAS
            # at ab=6/bk=64/c=2, from 0.980 at the ab=3/bk=128 pick). cuBLAS
            # nvjet uses the same 64x6 staging for this shape class.
            deep_seed_config: dict[str, Any] = dict(plain_clc_seed.config)
            deep_block_sizes = list(cast("list[int]", deep_seed_config["block_sizes"]))
            deep_block_sizes[-1] = 64
            constraints = self.cluster_m2_search_constraints
            if (
                constraints is not None
                and self.cluster_m2_bk_is_valid(64, constraints)
                and self.ab_stages_three_fits(
                    bm=deep_block_sizes[0],
                    bn=deep_block_sizes[1],
                    bk=64,
                    cluster_m=2,
                    ab_stages=6,
                )
            ):
                deep_seed_config["block_sizes"] = deep_block_sizes
                deep_seed_config["tcgen05_ab_stages"] = 6
                deep_seed_config["tcgen05_c_stages"] = 2
                seeds.append(Config(**deep_seed_config))
        plain_cluster_n2_seed = self._plain_cluster_n2_seed_config()
        if plain_cluster_n2_seed is not None:
            seeds.append(plain_cluster_n2_seed)
            deep_cluster_n2_seed = self._plain_cluster_n2_seed_config(deep=True)
            if deep_cluster_n2_seed is not None:
                seeds.append(deep_cluster_n2_seed)
                # The same deep ring on the 2x1 cluster: for fp8
                # 2048x4096x2048 it lands 190/200 flushed replays in cuBLAS's
                # timer bucket where the 2x2 rings straddle it.
                deep_cluster_n1_seed_config: dict[str, Any] = dict(
                    deep_cluster_n2_seed.config
                )
                deep_cluster_n1_seed_config["tcgen05_cluster_n"] = 1
                seeds.append(Config(**deep_cluster_n1_seed_config))
            if plain_clc_seed is not None:
                # 4-CTA multicast + CLC dynamic persistence (Quack's
                # dynamic-persistent 2x2 topology): reuse the validated CLC
                # seed shape (WITH_SCHEDULER + scheduler warp) and add the
                # cluster-N multicast on top. l2_groupings=[8] keeps the
                # concurrent wave square in TILES (G rows x ~2G tile-cols of
                # half-width boxes): measured best across the full-tile
                # cn2+CLC probes (fp16 6144^3 994 vs 962 at [4]; fp16 deepK
                # 934 vs 898; fp16 16384^3 899 vs 881).
                clc_n2_seed_config: dict[str, Any] = dict(plain_clc_seed.config)
                clc_n2_seed_config["tcgen05_cluster_n"] = 2
                clc_n2_seed_config["l2_groupings"] = [8]
                seeds.append(Config(**clc_n2_seed_config))
        plain_m_pair_seed = self._plain_m_pair_seed_config()
        if plain_m_pair_seed is not None:
            seeds.append(plain_m_pair_seed)
            # 2x2 super-tile: A multicast across the cluster-N pairs on top
            # of the SMEM-shared B within each M-paired tile. Wins on
            # K-dominated shapes (fp16 4096x4096x32768: ratio 1.016 vs
            # cuBLAS, from 0.919 at block_m=256/cluster_n=1).
            m_pair_cn2_seed_config: dict[str, Any] = dict(plain_m_pair_seed.config)
            m_pair_cn2_seed_config["tcgen05_cluster_n"] = 2
            seeds.append(Config(**m_pair_cn2_seed_config))
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
        config_view = self._matmul_config_view(config)
        if config_view is None:
            config["tcgen05_cluster_m"] = 1
            return
        block_sizes, m_index, n_index, k_index = config_view
        if (
            config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) == "two"
            and self.paired_pipeline_search_enabled()
        ):
            # This family uses the sampled MMA geometry rather than the
            # historical automatic 256x256 cluster projection.
            block_sizes[m_index] = 256 if block_sizes[m_index] == 256 else 128
            if block_sizes[n_index] not in (64, 128, 256):
                block_sizes[n_index] = 128
            if block_sizes[k_index] not in (64, 128, 256):
                block_sizes[k_index] = 64
            config["tcgen05_cluster_n"] = 1
            config["pid_type"] = "persistent_blocked"
            config.pop("epilogue_subtile", None)
            return

        def is_grouped_worklist_two_cta() -> bool:
            return (
                config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
                == TCGEN05_GROUPED_MODE_WORKLIST_NM
                and config.get("tcgen05_cluster_n", 1) == 1
                and block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
                and block_sizes[n_index] == 128
                and block_sizes[k_index] in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
                and config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
                in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
            )

        constraints = self.cluster_m2_search_constraints
        if is_grouped_worklist_two_cta():
            # The selected worklist source-M family owns the validated physical
            # 256x32/224/256 MMA profile and its K envelope independently of
            # the generic cluster-M2 policy. Compiler seeds widen the
            # otherwise-narrow pid fragment, so keep this exact family even
            # when generic constraints reject it or are absent.
            config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
            config.pop("epilogue_subtile", None)
            return
        if constraints is None:
            config["tcgen05_cluster_m"] = 1
            return
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            config["tcgen05_cluster_m"] = 1
            return
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        is_narrow_clc_aux_tma = self._is_clc_aux_tma_narrow_n_request(config)
        if edge_k_tail_family:
            # Plain (no-aux) kernels may keep a sampled deep-AB bk (the
            # bk=64/ab=5 family, see _plain_edge_deep_ab_stages); aux kernels
            # stay projected onto their validated bk=128/256 regimes.
            sampled_bk = block_sizes[k_index]
            if not (
                not self.aux_kernel_detected
                and isinstance(sampled_bk, int)
                and not isinstance(sampled_bk, bool)
                and self.cluster_m2_bk_is_valid(sampled_bk, constraints)
            ):
                block_sizes[k_index] = (
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
                    if is_narrow_clc_aux_tma
                    else TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
                )
        bk = block_sizes[k_index]
        if not isinstance(bk, int) or isinstance(bk, bool):
            config["tcgen05_cluster_m"] = 1
            return
        if not self.cluster_m2_bk_is_valid(bk, constraints):
            config["tcgen05_cluster_m"] = 1
            return
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
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
        # fp8 small-grid family: a sampled bm<=128 routes to the fp8-validated
        # per-CTA 64xbn 2-CTA tile (bm=128/bn=128) instead of the bm=256 full
        # tile, which underfills the device on small/wave-limited fp8 GEMMs. The
        # bm=256 full tile is still reachable from a sampled bm>128. The codegen
        # + runtime own this tile via ``_tcgen05_use_2cta_instrs``
        # (``bm == 128 and is_fp8``). Edge+K-tail candidates keep the bm=256
        # double-edge family unconditionally.
        sampled_bm = block_sizes[m_index]
        if (
            constraints.allow_fp8_small_grid
            and not constraints.one_wave_only
            and not edge_k_tail_family
            and isinstance(sampled_bm, int)
            and not isinstance(sampled_bm, bool)
            and sampled_bm <= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
        ):
            block_sizes[m_index] = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
            block_sizes[n_index] = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
            return
        if (
            constraints.allow_one_wave_tiles
            and not edge_k_tail_family
            and not is_narrow_clc_aux_tma
            # Exact-shape (source-C) epilogues keep the validated 256x256
            # aux-TMA regime, as ``_one_wave_seed_configs`` does.
            and not self.exact_shape_aux_kernel_detected
            and self._one_wave_two_cta_tile_is_valid(
                block_sizes[m_index], block_sizes[n_index]
            )
        ):
            # One-wave family: a sampled 256 x {128, 64} CtaGroup.TWO tile
            # keeps its narrow N (the 256x256 projection would leave most SMs
            # idle on these shapes).  cluster_n=2 needs an even N tile count.
            if config.get("tcgen05_cluster_n", 1) == 2 and not (
                self._one_wave_cluster_n2_is_valid(block_sizes[n_index])
            ):
                config["tcgen05_cluster_n"] = 1
            return
        if constraints.one_wave_only:
            # The one-wave tiles are the only two-CTA geometry admitted on
            # this shape (its 256x256 grid would idle most SMs); any other
            # sample keeps its tile on one CTA.
            config["tcgen05_cluster_m"] = 1
            return
        if (
            block_sizes[m_index] == 2 * TCGEN05_TWO_CTA_BLOCK_M
            and not edge_k_tail_family
            and self._m_pair_block_m_is_valid()
        ):
            # Keep the sampled block_m=512 (M-paired tiles) and pin the knob
            # its codegen envelope requires: both TMEM acc stages repurposed
            # as the pair's accumulators. cluster_n stays searchable in
            # {1, 2} (the 2x2 super-tile).
            config["tcgen05_acc_stages"] = 2
        else:
            block_sizes[m_index] = TCGEN05_TWO_CTA_BLOCK_M
        # Only the fully validated narrow-N CLC+aux-TMA seed may keep
        # block_n=128; other candidates use the canonical block_n=256.
        if is_narrow_clc_aux_tma:
            block_sizes[n_index] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        else:
            block_sizes[n_index] = TCGEN05_TWO_CTA_BLOCK_N
        self._fix_batched_cluster_n2_search_config(config, block_sizes[n_index])
        if edge_k_tail_family:
            if (
                not self.aux_kernel_detected
                and not is_narrow_clc_aux_tma
                and bk == TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
            ):
                # Plain deep-AB edge family (bk=64): pin the AB depth to the
                # deepest stage count that still fits next to the TMA-store
                # epilogue's C ring (the measured winner; bf16 5000^3 runs
                # 948 TFLOP/s at ab=5 vs 815 for the legacy bk=128/ab=2
                # projection below). Other knobs keep their sampled values —
                # the legacy placement overrides were calibrated for the
                # shallow bk=128 pipeline and measure neutral here.
                config["tcgen05_ab_stages"] = self._plain_edge_deep_ab_stages(bk)
                return
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

    def _fix_batched_cluster_n2_search_config(
        self, config: dict[str, object], bn: object
    ) -> None:
        """Project a batched cluster_n=2 sample onto what its codegen admits.

        The batched cluster_n=2 kernel pairs N tiles on the scheduler's
        swapped dim 1 (``program_id``), so it needs whole N-tile pairs and
        the plain raster: the L2-grouped and L2-swizzled pid decodes pair the
        cluster lanes on the first two grid dims, which are (batch, m) on a
        batched grid, and an odd M-tile count under ``l2_groupings > 1``
        hangs the multicast peers (``cute_mma`` rejects both). Odd N-tile
        counts fall back to cluster_n=1; the other samples keep cluster_n=2
        with ``l2_groupings`` of 1 and no L2 swizzle.
        """
        if not (
            self.matmul_has_leading_passthrough
            and config.get("tcgen05_cluster_n", 1) == 2
        ):
            return
        extents = self.matmul_compile_time_static_extents
        n = extents[1] if extents is not None else None
        if not (
            type(bn) is int
            and isinstance(n, int)
            and bn > 0
            and n % bn == 0
            and (n // bn) % 2 == 0
        ):
            config["tcgen05_cluster_n"] = 1
            return
        groupings = config.get("l2_groupings")
        if isinstance(groupings, list) and any(g != 1 for g in groupings):
            config["l2_groupings"] = [1] * len(groupings)
        if config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 1) != 1:
            config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1

    def _fix_grouped_worklist_search_config(self, config: dict[str, object]) -> None:
        """Project worklist search neighbors onto the validated logical tile."""
        if not self.search_enabled:
            return
        if (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            != TCGEN05_GROUPED_MODE_WORKLIST_NM
        ):
            return
        source_m_tile = config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
        if (
            type(source_m_tile) is not int
            or source_m_tile
            not in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
        ):
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return
        block_sizes, m_index, n_index, k_index = config_view
        block_sizes[m_index] = TCGEN05_TWO_CTA_BLOCK_M
        block_sizes[n_index] = 128
        sampled_bk = block_sizes[k_index]
        block_k_choices = (
            (128,)
            if source_m_tile == TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
            else TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
        )
        known_static_ks = {
            fact.static_k
            for fact in self.config_spec.matmul_facts
            if self._config_block_index(fact.k_block_id) == k_index
            and fact.static_k is not None
        }
        constraints = self.cluster_m2_search_constraints
        if constraints is not None:
            known_static_ks.add(constraints.static_k)
        if known_static_ks:
            # Worklists require exact K divisibility but own an independent
            # stage/tile-count envelope from generic cluster-M2 search. Filter
            # only by that shared structural requirement: importing generic
            # max-tile or edge-tail policy would rewrite valid worklist BKs.
            constrained_choices = tuple(
                block_k
                for block_k in block_k_choices
                if all(static_k % block_k == 0 for static_k in known_static_ks)
            )
            if not constrained_choices:
                static_k_values = tuple(sorted(known_static_ks))
                raise InvalidConfig(
                    f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                    f"{TCGEN05_GROUPED_MODE_WORKLIST_NM!r} has no supported "
                    f"block_k in {TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES} "
                    f"that divides every known static K in {static_k_values}"
                )
            block_k_choices = constrained_choices
        block_sizes[k_index] = (
            min(
                block_k_choices,
                key=lambda block_k: (abs(block_k - sampled_bk), block_k),
            )
            if type(sampled_bk) is int
            else block_k_choices[0]
        )
        if source_m_tile == TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE:
            config["tcgen05_cluster_m"] = 1
        elif source_m_tile != TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE:
            # The established compact source-32 profile also supports CtaGroup.ONE. The
            # reviewed source-224/256 profiles are CtaGroup.TWO even when a
            # pattern neighbor independently mutates the cluster fragment.
            config["tcgen05_cluster_m"] = 2

    @staticmethod
    def _is_grouped_clc_config(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) in TCGEN05_GROUPED_MODES
            and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )

    @staticmethod
    def _is_grouped_runtime_direct_clc_config(config: dict[str, object]) -> bool:
        return (
            CuteTcgen05Config._is_grouped_clc_config(config)
            and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            == TCGEN05_GROUPED_MODE_WORKLIST_NM
            and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY) is True
            and TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY not in config
        )

    def _normalize_grouped_full_coverage(
        self,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        validate_schedule: bool = False,
    ) -> None:
        key = TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY
        value = config.get(key, TCGEN05_GROUPED_FULL_COVERAGE_OFF)
        source_m_tile = config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
        if (
            source_m_tile == TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
            and value != TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL
        ):
            raise InvalidConfig(
                "source64 requires the proved grouped dense-local profile"
            )
        if value == TCGEN05_GROUPED_FULL_COVERAGE_OFF:
            config.pop(key, None)
            return
        valid = (
            type(value) is str
            and value
            in (
                TCGEN05_GROUPED_FULL_COVERAGE_DENSE,
                TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL,
            )
            and self.grouped_full_coverage_supported
        )
        if valid and validate_schedule:
            blocks = config.get("block_sizes")
            indices = self._matmul_block_indices()
            valid = (
                isinstance(blocks, list)
                and indices is not None
                and max(indices) < len(blocks)
                and full_coverage_pipeline_supported(
                    blocks[indices[2]],
                    config.get("tcgen05_ab_stages", 2),
                    config.get(
                        TCGEN05_CONSUMER_REGS_CONFIG_KEY,
                        TCGEN05_CONSUMER_REGS_DEFAULT,
                    ),
                    source_m_tile=source_m_tile,
                )
                and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
                == TCGEN05_GROUPED_MODE_WORKLIST_NM
                and source_m_tile
                in TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES
                and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False)
                is False
                and config.get(
                    TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY, False
                )
                is False
                and config.get(
                    TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY, False
                )
                is False
                and config.get(TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY)
                is None
                and not self._is_grouped_clc_config(config)
                and config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto") in ("auto", "one")
                and config.get("tcgen05_cluster_m", 1) == 1
                and config.get("tcgen05_cluster_n", 1) == 1
                and config.get("tcgen05_acc_stages", 2) == 2
                and config.get("tcgen05_c_stages", 2) == 2
                and config.get("tcgen05_num_epi_warps", 4) == 4
                and config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY, 1) == 1
                and config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 1) == 1
                and config.get(TCGEN05_WARP_SPEC_STORE_WARPS_KEY, 0) == 0
                and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY, 0) == 0
                and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY, 0) == 0
                and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, "default")
                == "default"
                and all(
                    config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS
                )
                and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False) is False
                and config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY, False)
                is False
                and all(
                    config.get(key, "normal") == "normal"
                    for key in (
                        TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
                        TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
                        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
                        TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
                        TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
                        TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
                        TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
                        TCGEN05_C_STORE_MODE_CONFIG_KEY,
                        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
                        TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
                    )
                )
            )
            if valid and config.get("tcgen05_ab_stages", 2) == 4:
                facts = self.grouped_worklist_smem_facts
                constraints = self.ab_stages_three_search_constraints
                if facts is not None and constraints is not None:
                    valid = full_coverage_smem_upper_bound(
                        facts.group_count,
                        128,
                        4,
                        source_m_tile=cast("int", source_m_tile),
                    ) <= (
                        constraints.per_cta_smem_budget_bytes
                        + TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
                    )
        if valid:
            return
        if fix_invalid and source_m_tile != TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE:
            config.pop(key, None)
        else:
            raise InvalidConfig(
                f"{key}={value!r} requires a proved pure shared-RHS BF16 "
                "Int32-offset worklist with ONE/source32, BK64/AB2/regs240 or "
                "BK128/AB4/regs256 (also ONE/source64 dense-local), ACC2/C2, four epilogue warps, one scheduler "
                "stage and sufficient per-CTA shared memory"
            )

    def _normalize_grouped_row_union(
        self,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        validate_schedule: bool = False,
    ) -> None:
        value = config.get(GROUPED_ROW_UNION_KEY, False)
        schedule = config.get(GROUPED_ROW_UNION_SCHEDULE_KEY, LEGACY_SCHEDULE)
        if schedule == LEGACY_SCHEDULE:
            config.pop(GROUPED_ROW_UNION_SCHEDULE_KEY, None)
        elif not (
            isinstance(schedule, str)
            and self.row_union_profile_supported(schedule)
            and value is True
        ):
            if not fix_invalid:
                raise InvalidConfig(
                    "row-union physical schedule requires its typed BF16 shared-RHS proof"
                )
            config.pop(GROUPED_ROW_UNION_SCHEDULE_KEY, None)
            schedule = LEGACY_SCHEDULE
        prefill = config.get(STARTUP_PREFILL_KEY, False)
        if prefill is False:
            config.pop(STARTUP_PREFILL_KEY, None)
        elif not (
            prefill is True
            and value is True
            and schedule == PAIRED_CLC_SCHEDULE
            and self.grouped_row_union_paired_clc_supported
        ):
            if not fix_invalid:
                raise InvalidConfig(
                    "AB startup prefill requires the proved linear-record paired CLC profile"
                )
            config.pop(STARTUP_PREFILL_KEY, None)
        ctas = config.get(GROUPED_RESIDENT_CTAS_KEY, 1)
        valid_ctas = type(ctas) is int and (
            ctas == 1
            or (
                ctas == 2
                and value is True
                and self.grouped_row_union_multi_resident_supported
            )
        )
        if not valid_ctas:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{GROUPED_RESIDENT_CTAS_KEY}={ctas!r} requires the proved "
                    "single-CTA row-union resource domain"
                )
            ctas = 1
        if ctas == 1:
            config.pop(GROUPED_RESIDENT_CTAS_KEY, None)
        if value is False:
            config.pop(GROUPED_ROW_UNION_KEY, None)
            return
        valid = value is True and (
            self.grouped_row_union_supported
            if schedule == LEGACY_SCHEDULE
            else self.row_union_profile_supported(cast("str", schedule))
        )
        if valid and validate_schedule:
            indices = self._matmul_block_indices()
            blocks = config.get("block_sizes")
            valid = (
                indices is not None
                and isinstance(blocks, list)
                and max(indices) < len(blocks)
                and row_union_schedule_supported(
                    config, tuple(blocks[index] for index in indices)
                )
            )
        if valid:
            return
        if fix_invalid:
            config.pop(GROUPED_ROW_UNION_KEY, None)
            config.pop(STARTUP_PREFILL_KEY, None)
            config.pop(GROUPED_ROW_UNION_SCHEDULE_KEY, None)
            config.pop(GROUPED_RESIDENT_CTAS_KEY, None)
        else:
            raise InvalidConfig(
                f"{GROUPED_ROW_UNION_KEY}={value!r} requires a proved pure "
                "shared-RHS BF16 Int32-offset union, fresh contiguous output, "
                "full 128x64x128 tiles and the ordinary single-CTA AB2/ACC2 "
                "role-local dense pipeline"
            )

    def epilogue_fanout_config_supported(self, config: dict[str, object]) -> bool:
        if not self.epilogue_fanout_plans or not fanout_schedule_supported(config):
            return False
        blocks = config.get("block_sizes")
        if not isinstance(blocks, list):
            return False
        for plan in self.epilogue_fanout_plans:
            for block_id, extent in zip(plan.block_ids, plan.shape, strict=True):
                index = self._config_block_index(block_id)
                if index is None or index >= len(blocks):
                    return False
                block = blocks[index]
                if type(block) is not int or block <= 0 or extent % block:
                    return False
            if config.get("tcgen05_cluster_m", 1) == 2:
                index = self._config_block_index(plan.block_ids[0])
                assert index is not None
                n_index = self._config_block_index(plan.block_ids[1])
                assert n_index is not None
                if (
                    blocks[index] != 256
                    or blocks[n_index] != 128
                    or plan.output_dtype.itemsize != 2
                ):
                    return False
            if (
                config.get("tcgen05_aux_load_placement") == "pre_acc_wait"
                and not plan.pre_wait_aux_safe
            ):
                return False
        return True

    def _normalize_epilogue_fanout(
        self,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        validate_schedule: bool = False,
    ) -> None:
        for key, domain in (
            ("tcgen05_region_ab_stages", range(17)),
            ("tcgen05_region_c_stages", (0, 2, 4)),
        ):
            if key not in config:
                continue
            stages = config[key]
            valid = (
                isinstance(stages, list)
                and len(stages) == len(self.materialized_matmul_block_ids)
                and len(stages) >= 2
                and all(type(stage) is int and stage in domain for stage in stages)
            )
            if not valid:
                if not fix_invalid:
                    raise InvalidConfig(
                        f"{key} requires one valid stage count per materialized MMA region"
                    )
                config.pop(key)
            elif not any(stages):
                config.pop(key)
        if "tcgen05_materialized_pdl" in config:
            value = config["tcgen05_materialized_pdl"]
            if type(value) is not bool or (
                len(self.materialized_matmul_block_ids) < 2
                and self.materialized_operand_pdl_roots is None
            ):
                if not fix_invalid:
                    raise InvalidConfig(
                        "tcgen05_materialized_pdl requires a proved materialized producer/consumer edge"
                    )
                config.pop("tcgen05_materialized_pdl")
            elif not value:
                config.pop("tcgen05_materialized_pdl")
        region_keys = (
            "tcgen05_region_ab_stages",
            "tcgen05_region_c_stages",
            *(
                ("tcgen05_materialized_pdl",)
                if len(self.materialized_matmul_block_ids) >= 2
                else ()
            ),
        )
        if validate_schedule and any(key in config for key in region_keys):
            if self._materialized_paired_stage_limits(config) is None:
                if not fix_invalid:
                    raise InvalidConfig(
                        "per-region pipelines and PDL require the complete typed paired materialized fanout schedule"
                    )
                for key in region_keys:
                    config.pop(key, None)
        if (
            validate_schedule
            and len(self.materialized_matmul_block_ids) < 2
            and config.get("tcgen05_materialized_pdl")
        ):
            supported = self.materialized_operand_pdl_supported(config)
            if not supported:
                if not fix_invalid:
                    raise InvalidConfig(
                        "tcgen05_materialized_pdl requires a proved materialized "
                        "dependency and its native TMA dependency-wait schedule"
                    )
                config.pop("tcgen05_materialized_pdl")
        if config.get(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY) == "pre_loop":
            # Keep the inert default implicit in both fanout controls so carriers
            # also round-trip when fanout search and its coordinate are disabled.
            config.pop(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY)
        value = config.get(FANOUT_CONFIG_KEY, "off")
        if type(value) is str and value == "off":
            if config.get(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY) == "before_store":
                if not fix_invalid:
                    raise InvalidConfig("before_store requires a proved shared fanout")
                config.pop(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY)
            config.pop(FANOUT_CONFIG_KEY, None)
            return
        if (
            type(value) is str
            and value == "shared"
            and self.epilogue_fanout_plans
            and (not validate_schedule or self.epilogue_fanout_config_supported(config))
        ):
            return
        if fix_invalid:
            config.pop(FANOUT_CONFIG_KEY, None)
        else:
            raise InvalidConfig(
                f"{FANOUT_CONFIG_KEY}={value!r} requires a proved equal-layout "
                "fresh-output fanout and a static-full standard "
                "role-local TMA epilogue"
            )

    def _normalize_batch_raster(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """``tcgen05_batch_raster`` names a walk over a batched grid only.

        Plain grids raster through ``loop_orders``; a raster carried onto a
        plain bind (config reuse across kernels) is dropped by the fix pass
        and rejected by validation.  Batched binds validate the value through
        the optional fragment.
        """
        if TCGEN05_BATCH_RASTER_CONFIG_KEY not in config:
            return
        if self.matmul_has_leading_passthrough:
            return
        if fix_invalid:
            config.pop(TCGEN05_BATCH_RASTER_CONFIG_KEY, None)
        else:
            raise InvalidConfig(
                f"{TCGEN05_BATCH_RASTER_CONFIG_KEY} requires a batched "
                "(leading passthrough) tcgen05 GEMM"
            )

    def prepare_normalization(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        self._normalize_warp_mma_family(config, fix_invalid=fix_invalid)
        self._normalize_batch_raster(config, fix_invalid=fix_invalid)
        # An explicit physical schedule owns its projection.  Validate the
        # requested carrier before generic block normalization can repair it.
        self._normalize_grouped_row_union(
            config,
            fix_invalid=fix_invalid,
            validate_schedule=physical_schedule(config) is not None,
        )
        self._normalize_grouped_full_coverage(config, fix_invalid=fix_invalid)
        self._normalize_epilogue_fanout(config, fix_invalid=fix_invalid)
        grouped_mode = config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        if (
            grouped_mode in TCGEN05_GROUPED_MODES
            and config.get("num_sm_multiplier", 1) != 1
        ):
            if fix_invalid:
                config.pop("num_sm_multiplier", None)
            else:
                raise InvalidConfig(
                    "tcgen05 grouped kernels require num_sm_multiplier=1"
                )
        grouped_clc = self._is_grouped_clc_config(config)
        if grouped_clc and not self._is_grouped_runtime_direct_clc_config(config):
            if fix_invalid:
                config.pop(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, None)
                config[TCGEN05_STRATEGY_CONFIG_KEY] = (
                    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
                )
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
            else:
                raise InvalidConfig(
                    "tcgen05 grouped CLC persistence requires "
                    f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                    f"{TCGEN05_GROUPED_MODE_WORKLIST_NM!r}, "
                    f"{TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY}=True, and no "
                    f"{TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY}; the "
                    "launcher must build an exact one-record-per-cluster tile table"
                )
        grouped_clc = self._is_grouped_clc_config(config)
        if grouped_clc and (
            config.get(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, 0) != 0
        ):
            if fix_invalid:
                config.pop(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, None)
            else:
                raise InvalidConfig(
                    "tcgen05 grouped CLC launches its exact full tile-record grid; "
                    "reserved_sms cannot limit that grid"
                )
        source_m_tile_key = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
        source_m_tile = config.get(source_m_tile_key)
        if source_m_tile_key in config and (
            type(source_m_tile) is not int
            or source_m_tile
            not in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
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
                    f"{TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES}, got "
                    f"{source_m_tile!r}"
                )
        signature_key = TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
        runtime_direct_key = TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
        if config.get(runtime_direct_key) is True and (
            grouped_mode != TCGEN05_GROUPED_MODE_WORKLIST_NM or signature_key in config
        ):
            if fix_invalid:
                config.pop(runtime_direct_key)
            else:
                raise InvalidConfig(
                    f"{runtime_direct_key}=True requires "
                    f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                    f"{TCGEN05_GROUPED_MODE_WORKLIST_NM!r} and no "
                    f"{signature_key}; unsupported requests must not silently "
                    "fall back to the legacy grouped scheduler"
                )
        l2_swizzle_size = config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 1)
        if (
            grouped_mode == TCGEN05_GROUPED_MODE_WORKLIST_NM
            and type(l2_swizzle_size) is int
            and l2_swizzle_size > 1
            and config.get(runtime_direct_key) is not True
        ):
            if fix_invalid:
                config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
            else:
                raise InvalidConfig(
                    f"{TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY}>1 for "
                    f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                    f"{TCGEN05_GROUPED_MODE_WORKLIST_NM!r} requires "
                    f"{runtime_direct_key}=True so the host runtime tile table "
                    "owns panel rastering"
                )
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

    def allow_ab_stages_three_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        assert dtype_bytes > 0, "dtype_bytes must be positive"
        if self._matmul_block_indices() is None:
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

    def register_grouped_worklist_smem_facts(
        self, *, group_count: int, device_split_sizes: bool
    ) -> None:
        if group_count <= 0:
            raise ValueError(
                "grouped worklist SMEM facts require a positive group count"
            )
        facts = Tcgen05GroupedWorklistSmemFacts(group_count, device_split_sizes)
        if self.grouped_worklist_smem_facts not in (None, facts):
            raise RuntimeError(
                "conflicting grouped worklist SMEM facts were registered for one "
                "ConfigSpec"
            )
        self.grouped_worklist_smem_facts = facts

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
        return max(0, device_cap - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES)

    @classmethod
    def per_cta_ab_smem_budget_bytes(cls, device: torch.device) -> int:
        device_cap = cls.per_cta_smem_capacity_bytes(device)
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
        return self._ab_stages_fit_constraints(
            constraints=self.ab_stages_three_search_constraints,
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            ab_stages=ab_stages,
        )

    @staticmethod
    def _ab_stages_fit_constraints(
        *,
        constraints: Tcgen05AbStagesThreeSearchConstraints | None,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
    ) -> bool:
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

    def rowvec_aux_stage_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
        c_stages: int,
        epi_warps: int = 4,
        acc_stages: int = 2,
    ) -> bool:
        """Whether ``pre_acc_wait`` row stages fit next to a tile's rings.

        Row-vector aux seeds and samples add the placement that stages the
        row in SMEM (``memory_ops``); the stage shares the per-CTA opt-in
        capacity with the AB ring and the C ring, which the AB-only gates
        (``ab_stages_three_fits``) do not count. ``default_layout_smem_fits``
        models the kernel's whole static arena against the raw capacity: a
        32-bit row on 16-bit inputs at the 2x2 256x256 ab=3 seed (192 KiB AB
        + 32 KiB (128, 64) C + 4 KiB warp-private stage) overflows, so that
        seed keeps the default placement, while the promoted 16-bit row
        ((128, 32) subtile, 1 KiB stage) fits, as does the 2 KiB stage next
        to the 192 KiB ring and 32 KiB c=4 ring of a 256x128x64 tile.
        """
        return self.default_layout_smem_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            ab_stages=ab_stages,
            c_stages=c_stages,
            stage_rows=True,
            epi_warps=epi_warps,
            acc_stages=acc_stages,
        )

    def default_layout_smem_bytes(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
        c_stages: int,
        stage_rows: bool,
        epi_warps: int = 4,
        acc_stages: int = 2,
    ) -> int | None:
        """Modelled static SMEM arena of a DEFAULT-layout tile, in bytes.

        The AB ring, the TMA-store C ring, with ``stage_rows`` the
        ``pre_acc_wait`` row stages of the store lowering
        (``tcgen05_rowvec_stage_smem_bytes``, occupying whole KiB ahead of
        the C ring's alignment) and the fixed allocations
        (``tcgen05_fixed_smem_overhead_bytes``), laid out as ptxas lays the
        kernel's allocations out (``tcgen05_constants``); exact for the
        measured plain and row-vector kernels. The C subtile follows the
        matmul plan (``cute_mma``): (128, 32) for a 16-bit output whose
        row-vector rows are all promoted and staged, or for a plain 16-bit
        store on the 256-wide two-CTA tile, otherwise CuTe's with-source rule
        at the output width, which the role-local epilogue applies whether
        or not a source-C tensor exists ((128, 64) on a 256-wide 16-bit
        tile, (128, 32) on narrower ones). ``None`` without a
        recorded SMEM budget, and without store facts when rows are staged;
        rings without facts are sized for the widest (32-bit) output the
        epilogue stores (the with-source subtile rule makes that ring at
        least as large as any narrower output's), so an unanalyzed store
        chain is judged conservatively rather than refused.
        """
        constraints = self.ab_stages_three_search_constraints
        facts = self.rowvec_aux_facts
        if constraints is None or cluster_m not in (1, 2):
            return None
        if facts is None and stage_rows:
            return None
        if bm <= 0 or bn <= 0 or bk <= 0 or ab_stages <= 0 or c_stages <= 0:
            return None
        if acc_stages <= 0 or epi_warps <= 0:
            return None
        ab_bytes = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=ab_stages,
            cluster_m=cluster_m,
        )
        if facts is None:
            output_itemsize = 4
            narrow_subtile = False
            stage_bytes = 0
        else:
            output_itemsize = facts.output_itemsize
            # The matmul plan's (128, 32) subtile: promoted 16-bit rows that
            # are staged, or a plain 16-bit store on the 256-wide two-CTA
            # tile (``tcgen05_plain_narrow_subtile``; the plan also asks for
            # a single static-full-tile store, which every plain kernel the
            # gates see has).
            narrow_subtile = (
                facts.output_itemsize == 2
                and (
                    (
                        stage_rows
                        and bool(facts.rows)
                        and all(row.promoted for row in facts.rows)
                    )
                    or (
                        not facts.rows
                        and constraints.dtype_bytes == 2
                        and cluster_m == 2
                        and bn == TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_N
                    )
                )
                and tcgen05_explicit_epilogue_tile_supported(
                    is_two_cta=cluster_m == 2,
                    bm=bm,
                    bn=bn,
                    tile_shape=(128, 32, 32),
                )
            )
            stage_bytes = (
                tcgen05_rowvec_stage_smem_bytes(
                    rows=facts.rows, bn=bn, epi_warps=epi_warps
                )
                if stage_rows
                else 0
            )
        if stage_bytes:
            # The row stages lead the arena and the C ring's 1 KiB alignment
            # follows them: a 256 B or 512 B promoted row occupies a whole KiB.
            stage_bytes = tcgen05_round_up_smem_bytes(
                stage_bytes, TCGEN05_SMEM_ROW_STAGE_CHUNK_BYTES
            )
        out_bits = output_itemsize * 8
        if narrow_subtile:
            epi_tile_m, epi_tile_n = 128, 32
        else:
            epi_tile_m, epi_tile_n = tcgen05_default_epilogue_tile_size(
                bm, bn, elem_width_d=out_bits, elem_width_c=out_bits
            )
        c_bytes = tcgen05_c_smem_bytes_per_cta(
            epi_tile_m=epi_tile_m,
            epi_tile_n=epi_tile_n,
            dtype_bytes=output_itemsize,
            c_stages=c_stages,
        )
        return (
            ab_bytes
            + c_bytes
            + stage_bytes
            + tcgen05_fixed_smem_overhead_bytes(
                ab_stages=ab_stages, acc_stages=acc_stages, cluster_m=cluster_m
            )
        )

    def default_layout_smem_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
        c_stages: int,
        stage_rows: bool,
        epi_warps: int = 4,
        acc_stages: int = 2,
    ) -> bool:
        """Whether a DEFAULT-layout tile's SMEM plan fits the opt-in capacity.

        ``default_layout_smem_bytes`` plus the allowance for the small
        allocations the model does not enumerate
        (``TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES``) against the raw
        per-CTA capacity, which the arena may fill exactly. Fails closed:
        ``False`` without a recorded SMEM budget, or without store facts
        when rows are staged.
        """
        arena_bytes = self.default_layout_smem_bytes(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            ab_stages=ab_stages,
            c_stages=c_stages,
            stage_rows=stage_rows,
            epi_warps=epi_warps,
            acc_stages=acc_stages,
        )
        if arena_bytes is None:
            return False
        constraints = self.ab_stages_three_search_constraints
        assert constraints is not None
        # ``per_cta_smem_budget_bytes`` is the opt-in capacity less the fixed
        # AB-gate reservation (``per_cta_ab_smem_budget_bytes``); the arena is
        # modelled in full here, so compare against the capacity itself.
        capacity = (
            constraints.per_cta_smem_budget_bytes
            + TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        return arena_bytes + TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES <= capacity

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
        # Workstream A Stage 2 (cycle 90): budget-aware admission for the deeper
        # C-store ring. Reuse the ``tcgen05_ab_stages=3`` SMEM-budget envelope
        # (same dtype_bytes + per-CTA budget after the non-AB reservation) and
        # require AB + C to fit together. This is the gate that keeps a deeper
        # C ring (``tcgen05_c_stages=4``) out of the ab=3 regime, where AB+C
        # overshoots the 232 KB B200 cap and ptxas raises a raw
        # ``too much shared`` error during tuning. The C bytes use the REAL
        # ``Tcgen05LayoutStrategy.DEFAULT`` epilogue subtile, not the (128, 32)
        # EXPLICIT_EPI_TILE direct-entry tile, so the byte count matches the
        # role-local codegen: a 256x256 16-bit tile is ``(128, 64)`` WITH a
        # source-C (residual family) but ``(128, 32)`` WITHOUT one (plain
        # matmul) -- ``compute_epilogue_tile_size`` shrinks N when no C tile
        # competes for SMEM. ``has_source_c`` threads that distinction through.
        constraints = self.ab_stages_three_search_constraints
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
        return ab_bytes + c_bytes <= constraints.per_cta_smem_budget_bytes

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

    def _grouped_worklist_nm_ab_config_matches(
        self, config: dict[str, object], ab_stages: object
    ) -> bool:
        return self._grouped_worklist_nm_ab_config_mismatch(config, ab_stages) is None

    def _grouped_worklist_nm_ab_config_mismatch(
        self, config: dict[str, object], ab_stages: object
    ) -> str | None:
        """Why ``config`` at ``ab_stages`` is not an admitted N,M worklist ring.

        ``None`` when it is: the worklist shape (a 256x128 logical tile with a
        resolvable MMA profile, ``tcgen05_cluster_n=1``, two accumulator and
        C stages, four to seven AB stages) whose exact per-CTA footprint
        (``tcgen05_grouped_worklist_smem_bytes``) fits the target capacity.
        The shape and the footprint are reported apart, so a rejected explicit
        config learns which one it missed.
        """
        config_view = self._matmul_config_view(config)
        if config_view is None:
            block_sizes = config.get("block_sizes")
            if (
                self.matmul_block_ids is not None
                or not isinstance(block_sizes, list)
                or len(block_sizes) != 3
            ):
                return TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
            # Reviewed/AOT configs can be normalized by a standalone ConfigSpec
            # before compiler MMA analysis registers semantic block IDs.  That
            # schema is exactly the canonical [M, N, K] triple.  Real kernels
            # always use the registered semantic indices above, including when
            # their block-size order is permuted.
            config_view = (block_sizes, 0, 1, 2)
        block_sizes, m_index, n_index, k_index = config_view
        block_k = block_sizes[k_index]
        profile = resolve_tcgen05_grouped_worklist_mma_profile(
            config,
            block_k=block_k,
        )
        if not (
            type(ab_stages) is int
            and 4 <= ab_stages <= 7
            and profile is not None
            and config.get("tcgen05_cluster_n", 1) == 1
            and config.get("tcgen05_acc_stages", 2) == 2
            and config.get("tcgen05_c_stages", 2) == 2
            and block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
            and block_sizes[n_index] == 128
        ):
            return TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
        constraints = self.ab_stages_three_search_constraints
        if constraints is None:
            # Fixed configs can be normalized before their input device is known.
            # CuTe MMA selection applies the real target's SMEM limit at codegen.
            return None
        target_capacity_bytes = (
            constraints.per_cta_smem_budget_bytes
            + TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        smem_facts = self.grouped_worklist_smem_facts
        if smem_facts is None:
            # Compiler-owned seeds register scheduler-specific allocation facts
            # and are rejected here when their exact footprint is too large.  An
            # explicit config may be normalized without a discovered worklist
            # contract, so those facts can legitimately be absent. Admit only when the
            # physical AB ring itself fits the raw target capacity, then defer
            # scheduler/mailbox allocations to the resolved worklist codegen
            # check.  That check proves the single grouped matmul and computes
            # its exact footprint before emitting any allocations.
            required_ab_bytes = tcgen05_ab_smem_bytes_per_cta(
                bm=profile.mma_m,
                bn=profile.mma_n,
                bk=cast("int", block_k),
                dtype_bytes=constraints.dtype_bytes,
                ab_stages=ab_stages,
                cluster_m=profile.cluster_m,
            )
            if required_ab_bytes <= target_capacity_bytes:
                return None
            return TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH
        sched_stage_count = config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY, 1)
        if type(sched_stage_count) is not int or sched_stage_count <= 0:
            return TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
        physical_bm, physical_bn = profile.mma_m, profile.mma_n
        # The generic AB search budget reserves 28 KiB, which is deliberately
        # conservative for general matmuls but would reject the valid
        # BK64/source-224/AB7 worklist at B200's exact 227-KiB cap. Reconstruct
        # the raw target cap and apply the same conservative worklist upper bound
        # across scheduler modes as codegen.
        required_bytes = tcgen05_grouped_worklist_smem_bytes(
            group_count=smem_facts.group_count,
            device_split_sizes=smem_facts.device_split_sizes,
            sched_stage_count=sched_stage_count,
            bm=physical_bm,
            bn=physical_bn,
            bk=cast("int", block_k),
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=ab_stages,
            acc_stages=2,
            c_stages=2,
            cluster_m=profile.cluster_m,
        )
        if required_bytes <= target_capacity_bytes:
            return None
        return TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH

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
        # Workstream A Stage 2 (cycle 90): true admission gate for the deeper C
        # ring. ``tcgen05_c_stages`` is an ``EnumFragment((2, 4))`` knob, so the
        # autotuner can SAMPLE c=4 independently of any projection — a directly
        # sampled 256x256 cluster_m=2 ab=3 + c=4 reaches ptxas and fails with a
        # raw ``too much shared`` error (verified cycle-90). Mirror
        # ``_fix_ab_stages_three_search_config``: when a config carries c=4 from
        # ANY source and ``c_stages_fits`` is False, demote it to 2.
        #
        # Scope: the canonical full-tile 256x256 DEFAULT-layout path, which is
        # exactly where the AB+C arithmetic is calibrated (the validated
        # CtaGroup.TWO cosize shapes) and where the role-local C ring lives,
        # plus the one-CTA bm=128 tiles (``_fix_one_cta_c_stages_search_config``).
        # The narrow-N / bm=128 edge family (``_fix_aux_edge_search_config`` sets
        # its own validated c=4 at a different cosize that the analytic model
        # would mis-judge) and the EXPLICIT_EPI_TILE direct-entry seeds (separate
        # (128, 32) tile + own admission) keep their c=4 untouched.
        if not self.search_enabled:
            return
        if config.get("tcgen05_c_stages") != TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES:
            return
        if not self._is_default_layout_full_tile_config(config):
            self._fix_one_cta_c_stages_search_config(config)
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
        bm = cast("int", block_sizes[m_index])
        bn = cast("int", block_sizes[n_index])
        bk = cast("int", block_sizes[k_index])
        # Judge at the depth the AB envelope leaves, like the one-CTA branch
        # and the row-stage gate: a sampled ab=12 reaches codegen at the
        # deepest AB-only fit, and that is the ring the C ring shares SMEM
        # with.
        if not self.c_stages_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            ab_stages=self._projected_ab_stages(
                config, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
            ),
            c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
            has_source_c=self.aux_kernel_detected,
        ):
            config["tcgen05_c_stages"] = 2

    def _fix_one_cta_c_stages_search_config(self, config: dict[str, object]) -> None:
        """Demote a sampled c=4 that overflows SMEM on a one-CTA bm=128 tile.

        The 256x256 gate never sees the one-CTA ``[128, 256, bk]`` tiles that
        the persistent bm clamp (``_fix_cluster_m1_persistent_search_config``)
        and the one-wave-only fallback land samples on, while the AB envelope
        fills their budget on its own (128x256x64 ab=4 and 128x256x128 ab=2
        are 192 KiB) and a 4-stage (128, 64) 16-bit C ring adds 64 KiB: NVVM
        rejected 2 of the 308 configs projected from an fp16 1024^3 sweep
        (the c=2 twins and ab<=3 compile). Judge the tile with the measured
        DEFAULT-layout arena model at the depth the envelope leaves, counting
        the row stages a kept ``pre_acc_wait`` adds: 128x128x64 ab=6 keeps
        c=4 with a 2 KiB fp32 row stage (231 588 B of 232 448), 128x256x64
        ab=4 does not. Grouped kernels keep their own stage admission, and
        the aux output-edge family keeps its validated c=4
        (``_fix_aux_edge_search_config``).
        """
        if (
            config.get(
                TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
                Tcgen05LayoutStrategy.DEFAULT.value,
            )
            != Tcgen05LayoutStrategy.DEFAULT.value
            or config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            or config.get("tcgen05_cluster_m", 1) != 1
        ):
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return
        block_sizes, m_index, n_index, k_index = config_view
        bm = block_sizes[m_index]
        bn = block_sizes[n_index]
        bk = block_sizes[k_index]
        if not (
            isinstance(bm, int)
            and isinstance(bn, int)
            and isinstance(bk, int)
            and bm == TCGEN05_ONE_CTA_MAX_BLOCK_M
        ):
            return
        if self.aux_kernel_detected and self._has_any_matmul_fact_edge_tile(config):
            return
        if not self.default_layout_smem_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=1,
            ab_stages=self._projected_ab_stages(
                config, bm=bm, bn=bn, bk=bk, cluster_m=1
            ),
            c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
            stage_rows=self._store_lowering_stages_rows(config),
            epi_warps=self._config_epi_warps(config),
            acc_stages=self._config_acc_stages(config),
        ):
            config["tcgen05_c_stages"] = 2

    def _fix_rowvec_stage_search_config(self, config: dict[str, object]) -> None:
        """Demote a sampled ``pre_acc_wait`` whose row stages overflow SMEM.

        ``tcgen05_aux_load_placement`` is a search fragment on every
        row-vector kernel, so a sample reaches codegen with the placement on
        whatever rings it carries: the seed builders apply
        ``rowvec_aux_stage_fits`` but the projection did not, and the fp16
        2048x4096x2048 GEMM with a 32-bit row died in NVVM on the 256x256
        deep rings (bk=64 ab=6 and bk=128 ab=3, cluster_n 1 and 2: 4 dead
        configs of 326 projected) once the 4 KiB warp-private stage joined
        them. Judge the final rings (after the c-stages gate) at the depth
        the AB envelope leaves, with the measured arena model: the 2 KiB
        stage of a 128-wide tile fits next to a 192 KiB ring and a 32 KiB
        c=4 ring (256x128x64 ab=8, 256x128x128 ab=4, one-CTA 128x128x{32,
        64, 128} at ab 12/6/3: 231 596-231 716 B of 232 448), the 4 KiB
        stage of a 256-wide tile does not. A row that takes no stage keeps
        the placement, the C-input strategy renders no stage
        (``_store_lowering_stages_rows``) and non-DEFAULT layouts keep their
        own admission.
        """
        if not self.search_enabled:
            return
        facts = self.rowvec_aux_facts
        if (
            facts is None
            or not facts.rows
            or not self._store_lowering_stages_rows(config)
            or config.get(
                TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
                Tcgen05LayoutStrategy.DEFAULT.value,
            )
            != Tcgen05LayoutStrategy.DEFAULT.value
        ):
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return
        block_sizes, m_index, n_index, k_index = config_view
        bm = block_sizes[m_index]
        bn = block_sizes[n_index]
        bk = block_sizes[k_index]
        if not (isinstance(bm, int) and isinstance(bn, int) and isinstance(bk, int)):
            return
        epi_warps = self._config_epi_warps(config)
        if (
            tcgen05_rowvec_stage_smem_bytes(rows=facts.rows, bn=bn, epi_warps=epi_warps)
            == 0
        ):
            return
        cluster_m = config.get("tcgen05_cluster_m", 1)
        c_stages = config.get("tcgen05_c_stages", 2)
        if not (isinstance(cluster_m, int) and isinstance(c_stages, int)):
            return
        if not self.rowvec_aux_stage_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            ab_stages=self._projected_ab_stages(
                config, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
            ),
            c_stages=c_stages,
            epi_warps=epi_warps,
            acc_stages=self._config_acc_stages(config),
        ):
            config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
            )

    def _projected_ab_stages(
        self,
        config: dict[str, object],
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
    ) -> int:
        """The AB depth the search's final envelope leaves on ``config``.

        ``_validate_direct_entry_ab_stage_envelope`` runs after every family
        fixup and clamps a sampled depth above 3 on a DEFAULT-layout tile to
        the deepest AB-only fit (``max_ab_stages_that_fit``), so the ring
        gates that run before it judge a sampled ab=12 at the depth codegen
        will see instead of demoting a knob the clamp was about to rescue.
        """
        ab_stages = config.get("tcgen05_ab_stages", 2)
        if not isinstance(ab_stages, int) or ab_stages <= 0:
            return 2
        if ab_stages <= 3:
            return ab_stages
        fit_max = self.max_ab_stages_that_fit(bm=bm, bn=bn, bk=bk, cluster_m=cluster_m)
        return min(ab_stages, fit_max) if fit_max > 0 else 3

    @staticmethod
    def _store_lowering_stages_rows(config: dict[str, object]) -> bool:
        """Whether the store lowering stages ``pre_acc_wait`` rows for ``config``.

        The C-input strategy is excluded: with a productive C-input warp the
        epilogue renders the plain (128, 64) subtile and no row stage (the
        row-vector C-input seed compiles to its rings plus the fixed
        overhead, 165 068 B with a promoted bias row), so the stage model
        does not apply there.
        """
        placement = config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
        return placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT and not config.get(
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
        )

    @staticmethod
    def _config_epi_warps(config: dict[str, object]) -> int:
        """Epilogue warps of ``config`` (the row stages are sized per warp)."""
        epi_warps = config.get("tcgen05_num_epi_warps", 4)
        if isinstance(epi_warps, int) and epi_warps > 0:
            return epi_warps
        return 4

    @staticmethod
    def _config_acc_stages(config: dict[str, object]) -> int:
        """Accumulator stages of ``config`` (16 B of mbarriers each)."""
        acc_stages = config.get("tcgen05_acc_stages", 2)
        if isinstance(acc_stages, int) and acc_stages > 0:
            return acc_stages
        return 2

    def _is_default_layout_full_tile_config(self, config: dict[str, object]) -> bool:
        # The canonical 256x256 DEFAULT-layout role-local tile, where the C-ring
        # AB+C SMEM model is calibrated. EXPLICIT_EPI_TILE configs use a separate
        # tile/admission and are excluded; an absent layout key defaults to
        # DEFAULT.
        layout = config.get(
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
            Tcgen05LayoutStrategy.DEFAULT.value,
        )
        if layout != Tcgen05LayoutStrategy.DEFAULT.value:
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
        """Get hardware-validated maximum ab_stages for a dtype.

        The maximum practical ab_stages depends on dtype size because
        smaller dtypes fit more pipeline stages in the same SMEM budget:
        - FP8 (1 byte): 12 stages - validated on B200, fits 2x bf16
        - FP16/BF16 (2 bytes): 12 stages - the per-CTA SMEM budget decides
          (256x256x64 two-CTA fits 6, 256x64x64 two-CTA fits 9)
        - FP32 (4 bytes): 3 stages - baseline for larger dtypes

        Args:
            dtype_bytes: Size of data type in bytes

        Returns:
            Maximum ab_stages for this dtype, or 0 if invalid
        """
        if dtype_bytes <= 0:
            return 0
        if dtype_bytes == 1:  # FP8
            return 12
        if dtype_bytes == 2:  # FP16/BF16
            return 12
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
        """Compute maximum ab_stages that fits in per-CTA SMEM budget.

        Mirrors CUTLASS's ``_compute_stages``: fill SMEM with as many AB
        pipeline stages as fit the hardware budget. Uses direct calculation
        since SMEM usage scales linearly with ab_stages.

        For FP8 (1-byte operands), this enables ~2x deeper staging than
        BF16 (2-byte), which is critical for hiding K-loop TMA latency
        in compute-bound kernels.

        Args:
            bm: Block size in M dimension
            bn: Block size in N dimension
            bk: Block size in K dimension
            cluster_m: CTA cluster size (1 or 2)
            hard_cap: Optional maximum stages override. If None, uses
                dtype-specific default (12 for FP8, 6 for FP16, 3 for FP32)

        Returns:
            Maximum valid ab_stages in [1, hard_cap], or 0 if constraints
            are unknown or configuration is invalid (e.g., ab_stages=1
            doesn't fit budget).

        Example:
            >>> # FP8 256x256x64 cluster_m=2
            >>> config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=2)
            8  # FP8 fits 8 stages

            >>> # BF16 same tile (2x larger per stage)
            >>> config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=2)
            4  # BF16 fits only 4 stages
        """
        constraints = self.ab_stages_three_search_constraints
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

    def _fix_ab_stages_three_search_config(self, config: dict[str, object]) -> None:
        if self.ab_stages_three_search_constraints is None:
            return
        if not self.search_enabled:
            return
        if config.get("tcgen05_ab_stages") != 3:
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            config["tcgen05_ab_stages"] = 2
            return
        block_sizes, m_index, n_index, k_index = config_view
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if not self.ab_stages_three_fits(
            bm=cast("int", block_sizes[m_index]),
            bn=cast("int", block_sizes[n_index]),
            bk=cast("int", block_sizes[k_index]),
            cluster_m=cluster_m,
        ):
            config["tcgen05_ab_stages"] = 2

    def _fix_ab_stages_search_config(self, config: dict[str, object]) -> None:
        # Budget-aware ab=3 admission for the lifted ``for_search`` cap (see
        # ``optional_fragments`` and cute_plan.md §4.5 for the empirical narrative).
        # Mirror ``_fix_c_stages_search_config`` (fail-CLOSED, cast-based): on the
        # canonical 256x256 DEFAULT-layout role-local path, demote a directly-sampled
        # ab=3 to 2 when it does not fit the per-CTA SMEM budget. The invariant — the
        # new dimension over the bare-AB ``_fix_ab_stages_three_search_config`` gate —
        # is REAL source-C presence, keyed on the PRECISE
        # ``exact_shape_aux_kernel_detected`` (rank-2 exact-shape residual_add), NOT
        # the broad ``aux_kernel_detected`` (also True for a rowvec broadcast bias
        # that has no source-C ring): a real source-C materializes the larger
        # (128, 64) C ring, so AB(ab=3) + C overflows the cap even at c=2 and MUST
        # demote, while the plain / rowvec-bias family (no source-C ring) keeps the
        # bare-AB calibration so its ab=3 winner stays searchable. The exact-shape
        # residual cluster_m=2 full-tile candidates are already forced to ab=2 by
        # ``_fix_aux_tma_full_tile_search_config`` (runs first); this gate's source-C
        # branch is the fail-closed backstop for any exact-shape residual ab=3 that
        # projection does not claim (e.g. cluster_m=1). The EXPLICIT_EPI_TILE
        # direct-entry (TVM-FFI) seeds use a separate (128, 32) tile + own admission
        # and are out of the DEFAULT-layout scope, so their seeded ab=3/ab=6 is
        # untouched.
        if not self.search_enabled:
            return
        if config.get("tcgen05_ab_stages") != 3:
            return
        if not self._is_default_layout_full_tile_config(config):
            return
        config_view = self._matmul_config_view(config)
        if config_view is None:
            config["tcgen05_ab_stages"] = 2
            return
        block_sizes, m_index, n_index, k_index = config_view
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if self.exact_shape_aux_kernel_detected:
            # Real source-C present: require AB(ab=3) + the (128, 64) C ring to fit
            # together. ``c_stages_fits`` fails CLOSED when no SMEM budget is
            # recorded (non-B200 / CPU host), so the over-budget and no-budget
            # arms are both covered by a single ``not c_stages_fits`` check.
            c_stages = cast("int", config.get("tcgen05_c_stages", 2))
            fits = self.c_stages_fits(
                bm=cast("int", block_sizes[m_index]),
                bn=cast("int", block_sizes[n_index]),
                bk=cast("int", block_sizes[k_index]),
                cluster_m=cluster_m,
                ab_stages=3,
                c_stages=c_stages,
                has_source_c=True,
            )
        else:
            # Plain / rowvec-bias store (no source-C ring): the bare-AB gate is the
            # calibrated admission — the small no-source-C epilogue D ring rides the
            # non-AB reservation. ``ab_stages_three_fits`` returns False with no
            # budget recorded, so this also fails CLOSED.
            fits = self.ab_stages_three_fits(
                bm=cast("int", block_sizes[m_index]),
                bn=cast("int", block_sizes[n_index]),
                bk=cast("int", block_sizes[k_index]),
                cluster_m=cluster_m,
            )
        if not fits:
            config["tcgen05_ab_stages"] = 2

    def _fix_with_scheduler_search_config(self, config: dict[str, object]) -> None:
        if not (
            self.search_enabled
            and (
                self.aux_kernel_detected or self._plain_clc_persistence_search_enabled()
            )
        ):
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
        # A productive C-input warp stages the exact-shape aux ring next to
        # the AB ring; codegen rejects every ``tcgen05_ab_stages >= 3`` with
        # it (``cute_mma``), so deeper samples keep the ring depth and drop
        # the producer warp.
        if (
            type(ab_stages) is int
            and ab_stages >= 3
            and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
        ):
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0

    def _fix_aux_tma_search_config(self, config: dict[str, object]) -> None:
        if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA:
            return
        if not self._aux_tma_search_enabled():
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
            return
        # Aux-TMA is kept when the projected cluster_m=2 candidate matches either
        # the validated edge+K-tail shape or the cycle-46 full-tile shape, and
        # the strategy is the ROLE_LOCAL_WITH_SCHEDULER + c-input warp combo that
        # the aux-TMA producer warp requires.
        if not self._is_with_scheduler_c_input_config(config):
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
            return
        if not (
            self._is_validated_cluster_m2_edge_search_candidate(config)
            or self._is_validated_cluster_m2_full_tile_search_candidate(config)
        ):
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT

    def _fix_aux_tma_full_tile_search_config(self, config: dict[str, object]) -> None:
        # Cycle 88 (Workstream B): on the residual full-tile cluster_m=2
        # family (T20/T14/T25/T28 — exact-shape-gated by
        # ``_aux_tma_full_tile_search_enabled``), project every cluster_m=2
        # candidate onto the validated aux-TMA producer regime
        # (role_local_with_scheduler + scheduler/c-input warp + ab=2 +
        # aux_load_mode=tma). Clean force-config measurements show aux-TMA
        # strictly beats the same-config SIMT control on this family (T20
        # 962.6 vs 844.0 TF = +14%; T25 730.8 vs 324.7 TF = +125%) and also
        # beats the autotuner's default monolithic-ab=3 SIMT pick (T20 ~915
        # TF). Without this projection the population search either keeps the
        # monolithic-ab=3 SIMT region (T20 reliably) or randomly lands on a
        # catastrophic SIMT cluster_m=2 pick (T25 variance: 756 aux-TMA vs 286
        # SIMT), so the +2.4-14 pp aux-TMA gain is not banked deterministically.
        # This is the aux-TMA analogue of the FFI direct-entry search
        # projection (``_fix_target1_tvm_ffi_search_config``). Tightly bounded:
        # cluster_m=1 candidates are left untouched (search still explores
        # them), the edge+K-tail T12 family keeps its own
        # ``_aux_tma_edge_search_enabled`` CLC path, and the gate fires only
        # when ``exact_shape_aux_kernel_detected`` is True (residual subset
        # only — no non-residual target is perturbed).
        if not self.search_enabled:
            return
        if not self._aux_tma_full_tile_search_enabled():
            return
        if config.get("tcgen05_cluster_m") != 2:
            return
        if not self._is_validated_cluster_m2_full_tile_search_candidate(config):
            return
        # ab=2 is the aux-TMA producer's validated stage depth for this family
        # (the aux SMEM ring forces ab<=2 under the 232 KB B200 cap; the
        # cycle-86/88 measurements ran at ab=2). ``_c_input_seed_config``
        # already emits exactly this regime as a seed, so the shape envelope
        # validated above is sufficient — no extra SMEM-fit check is needed.
        config[TCGEN05_STRATEGY_CONFIG_KEY] = (
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
        config["tcgen05_ab_stages"] = 2
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
        # Workstream A Stage 2 (cycle 90): give this family the deeper C-store
        # ring (foundation for the Stage-4 store-warp split — the store warp
        # drains tile N's TMA-D from the ring while the 4 epi warps run tile
        # N+1's T2R; a 2-stage ring leaves no slack). At ab=2 the c=4 ring fits
        # (128 KB AB + 64 KB C = 192 KB < 232 KB cap; the DEFAULT epi tile is
        # (128, 64), so each C stage is 16 KB), confirmed correct on T20
        # (cycle-90 force-config: c=2 942.2 TF vs c=4 936.1 TF — perf-neutral
        # standalone, exactly the cycle-89 NCU prediction since the TMA-D store
        # is already c_pipeline-overlapped). Gated by ``c_stages_fits`` so a
        # future ab=3 candidate in this family cannot be lifted into overflow.
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return
        block_sizes, m_index, n_index, k_index = config_view
        if self.c_stages_fits(
            bm=cast("int", block_sizes[m_index]),
            bn=cast("int", block_sizes[n_index]),
            bk=cast("int", block_sizes[k_index]),
            cluster_m=2,
            ab_stages=2,
            c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
            has_source_c=True,
        ):
            config["tcgen05_c_stages"] = TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES

    @staticmethod
    def _is_with_scheduler_c_input_config(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) == 1
            and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
        )

    def _clc_persistence_search_enabled(self) -> bool:
        """CLC (hardware tile-scheduler) persistence search gate, sm100+ only.

        Two validated families: the aux-TMA edge+K-tail family (its original
        scope) and the plain full-tile cluster_m=2 family (see
        ``_plain_clc_persistence_search_enabled``).
        """
        capability = self.config_spec.target_device_capability
        if capability is None:
            return False
        if capability[0] < 10 or "flat" not in self.allowed_pid_types:
            return False
        return (
            self._aux_tma_edge_search_enabled()
            or self._plain_clc_persistence_search_enabled()
        )

    def _plain_clc_persistence_search_enabled(self) -> bool:
        """Full-tile cluster_m=2 CLC for matmuls without aux operands.

        The CLC scheduler warp replaces the static tile sweep with the
        hardware tile scheduler, which keeps CTAs fed as tiles finish at
        uneven rates. On B200 this wins across full-tile shapes (fp16
        16384^3: 890 vs 738 TFLOP/s; bf16 8192x28672x8192: 956 vs 906; fp16
        2048x128256x4096: 914 vs 878; fp16 4096^3: 1038 vs 981 — same-GPU
        probe pairs) and on the deep-AB edge family (bf16 5000^3: 883 vs 859),
        matching quack's is_dynamic_persistent=True default. The aux families
        keep their own aux-TMA/CLC regimes.
        """
        constraints = self.cluster_m2_search_constraints
        return (
            not self.aux_kernel_detected
            and not self.matmul_has_leading_passthrough
            and constraints is not None
            and not constraints.one_wave_only
        )

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
        if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) == "two":
            fit_max = self._paired_pipeline_stage_limit(config)
            if fit_max is not None and type(ab_stages) is int:
                if ab_stages <= fit_max:
                    return
                if fix_invalid and fit_max > 0:
                    config["tcgen05_ab_stages"] = fit_max
                    return
                raise InvalidConfig(
                    "tcgen05 AB pipeline and output store ring exceed "
                    f"per-CTA shared memory (maximum AB stages: {fit_max})"
                )
        if type(ab_stages) is not int or ab_stages <= 3:
            return
        if self._grouped_dynamic_deep_config_matches(config):
            return
        if self._grouped_worklist_nm_ab_config_matches(config, ab_stages):
            return
        if (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            == TCGEN05_GROUPED_MODE_WORKLIST_NM
        ):
            # The worklist scheduler's records and mailboxes are not in the
            # plain-GEMM SMEM model below, which admits 16-bit rings up to the
            # budget (source-256/BK64/AB7 fits that model at 8 but exceeds the
            # exact worklist footprint).  A worklist ring the exact footprint
            # rejects clamps to the deepest depth it admits, or is rejected;
            # a config outside the worklist shape (cluster_n=2, other stage
            # counts, ab>7) is rejected for its shape, not its footprint.
            if fix_invalid:
                config["tcgen05_ab_stages"] = next(
                    (
                        depth
                        for depth in range(ab_stages - 1, 3, -1)
                        if self._grouped_worklist_nm_ab_config_matches(config, depth)
                    ),
                    3,
                )
                return
            mismatch = self._grouped_worklist_nm_ab_config_mismatch(config, ab_stages)
            assert mismatch is not None
            raise InvalidConfig(f"tcgen05_ab_stages={ab_stages} {mismatch}")
        # After the grouped dynamic/worklist exceptions above, ab>3 is only valid
        # on the TVM-FFI direct-entry path and for its accepted (bk, ab, c) tuples
        # (bk=64 admits (ab=6, c=4)). Everything else clamps or rejects to ab=3.
        block_sizes = config.get("block_sizes")
        k_block_index = self._direct_entry_k_block_index()
        bk = (
            block_sizes[k_block_index]
            if isinstance(block_sizes, list)
            and k_block_index is not None
            and k_block_index < len(block_sizes)
            else None
        )
        c_stages = config.get("tcgen05_c_stages")
        if (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and isinstance(bk, int)
            and not isinstance(bk, bool)
            and type(c_stages) is int
            and tcgen05_direct_entry_stage_tuple_allowed(
                bk=bk, ab_stage_count=ab_stages, c_stage_count=c_stages
            )
        ):
            return
        # Operands whose AB SMEM fits the per-CTA budget can run a deeper AB
        # pipeline than the historical bk=128-tuned cap of 3: fp8 (1-byte)
        # always could, and 16-bit operands fit 4-5 stages at bk=64 (worth
        # ~5-25% on edge shapes where the K-tail already forces bk=64/128).
        # This lets Helion emit the same deeply-pipelined CtaGroup.TWO kernel
        # CUTLASS uses for compute-bound GEMMs.
        constraints = self.ab_stages_three_search_constraints
        if constraints is not None:
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
            "TVM-FFI direct-entry path, the grouped N,M worklist "
            "path within the SMEM budget, or fp8 within the SMEM budget"
        )

    def _is_validated_clc_persistence_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if not self._clc_persistence_search_enabled():
            return False
        if self._plain_clc_persistence_search_enabled():
            # Plain (no-aux) families: scheduler warp only, no C-input warp
            # (there is no source-C / aux producer to feed). Accept the
            # full-tile candidate shape or the edge family's projected
            # cluster_m=2 shape (bm/bn are forced to the 256x256 tile and bk
            # to a valid edge bk by _fix_cluster_m2_search_config).
            if not (
                self._is_validated_cluster_m2_full_tile_search_candidate(config)
                or self._is_validated_cluster_m2_edge_search_candidate(config)
            ):
                return False
            if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
                return False
            # cluster_n=2 rides the generalized 4-CTA CLC leader broadcast
            # (Quack's dynamic-persistent 2x2 topology).
            if config.get("tcgen05_cluster_n", 1) not in (1, 2):
                return False
            if (
                config.get(TCGEN05_STRATEGY_CONFIG_KEY)
                != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
                or config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) != 1
                or config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY, 0) != 0
            ):
                return False
            if self.config_spec.supports_config_key("indexing"):
                indexing = config.get("indexing")
                if (
                    not isinstance(indexing, list)
                    or indexing
                    != ["tensor_descriptor"] * self.config_spec.indexing.length
                ):
                    return False
            return True
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
        if self._is_grouped_runtime_direct_clc_config(config):
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
            # The autotuner search surface for this key is the collapsed
            # ``EnumFragment((True,))``; autotuner-generated configs always
            # set it via ``default_flat()`` mutation. ``flatten`` only hits
            # this branch on user-supplied configs that omit the key, where
            # absence means "no FFI promotion requested" — matches the
            # validation-view default and the special case at
            # ``normalize_pre_pid_type``.
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
            config["l2_groupings"] = [1] * len(self.config_spec.l2_groupings)
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

    def _is_validated_cluster_m2_full_tile_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        # Cycle 46: full-tile cluster_m=2 candidate gate for aux-TMA admission.
        # After ``_fix_cluster_m2_search_config`` projects a full-tile sample to
        # the canonical 256x256x bk shape with ``persistent_interleaved`` pid,
        # this returns True so aux-TMA stays during the search-time fixup.
        # bk validity is already enforced by ``_fix_cluster_m2_search_config``;
        # we re-check it here so a stale/unprojected config cannot slip through.
        if config.get("tcgen05_cluster_m") != 2:
            return False
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or constraints.one_wave_only
        ):
            return False
        if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
            return False
        if config.get("tcgen05_cluster_n", 1) != 1:
            return False
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return False
        block_sizes, m_index, n_index, k_index = config_view
        if block_sizes[m_index] not in (
            TCGEN05_TWO_CTA_BLOCK_M,
            # M-paired tiles: two 256-row subtiles per work tile.
            2 * TCGEN05_TWO_CTA_BLOCK_M,
        ):
            return False
        if block_sizes[n_index] != TCGEN05_TWO_CTA_BLOCK_N:
            return False
        bk = block_sizes[k_index]
        if not isinstance(bk, int) or isinstance(bk, bool):
            return False
        return self.cluster_m2_bk_is_valid(bk, constraints)

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
        config_view = self._matmul_config_view(config)
        if config_view is None:
            return
        block_sizes, m_index, n_index, k_index = config_view
        if (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            == TCGEN05_GROUPED_MODE_WORKLIST_NM
            and config.get("tcgen05_cluster_n", 1) == 1
            and block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
            and block_sizes[n_index] == 128
            and block_sizes[k_index] in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
            and config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
            in TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES
        ):
            # The logical DSL tile remains 256x128 while the N,M-oriented
            # collective resolves to a physical 128x32/64 CtaGroup.ONE MMA.
            return
        constraints = self.cluster_m2_search_constraints
        if constraints is not None and constraints.allow_edge_k_tail_family:
            # persistent_interleaved stays in the flat enum so cluster_m=2
            # edge-family samples can encode; cluster_m=1 samples from the
            # same surface must use the validated flat edge fallback.
            config["pid_type"] = "flat"
            return
        bm = block_sizes[m_index]
        if isinstance(bm, int) and not isinstance(bm, bool):
            block_sizes[m_index] = min(bm, TCGEN05_ONE_CTA_MAX_BLOCK_M)

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
        allow_cluster_m2_one_wave_tiles: bool = False,
        cluster_m2_one_wave_only: bool = False,
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
                allow_one_wave_tiles=allow_cluster_m2_one_wave_tiles,
                one_wave_only=cluster_m2_one_wave_only,
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
            self.allow_deep_direct_entry_validation(device=ab_stages_three_device)
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
        paired_search = self.paired_pipeline_search_enabled() or bool(
            self.materialized_matmul_block_ids
        )
        if paired_search:
            cluster_m_choices = tuple(dict.fromkeys((*cluster_m_choices, 2)))
        # cluster_n=2 (the Quack-canonical 4-CTA cluster: B multicast on top
        # of the 2-CTA A multicast) is searchable on every cluster_m=2
        # family — it wins big on B-heavy problems (fp16 4096x4096x32768:
        # 911 vs 824 TFLOP/s; cuBLAS's nvjet picks 2x2 clusters for the same
        # shape) and, with the generalized 4-CTA CLC broadcast, closes the
        # edge-family gap too (bf16 5000^3: 907 vs 852 TFLOP/s, matching
        # quack's dynamic-persistent 2x2 within 1%).
        cluster_n_searchable = self.cluster_m2_search_constraints is not None
        cluster_n_choices: tuple[int, ...] = (
            (1, 2) if not for_search or cluster_n_searchable else (1,)
        )
        if for_search and self.num_epi_warps_search_choices is not None:
            num_epi_warps_fragment: ConfigSpecFragment = EnumFragment(
                self.num_epi_warps_search_choices
            )
        elif not for_search and self.num_epi_warps_validation_choices is not None:
            num_epi_warps_fragment = EnumFragment(self.num_epi_warps_validation_choices)
        else:
            num_epi_warps_fragment = IntegerFragment(1, 4, 4)
        if not for_search:
            # Validation admits what the direct-entry codegen supports: bk=64
            # accepts the deep (ab=6, c=4) tuple (see
            # ``TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK``), so the validation
            # surface lifts the AB cap to 6 for structurally identified 16-bit
            # direct-entry matmuls. The actual (bk, ab, c) tuple is gated by
            # ``_validate_direct_entry_ab_stage_envelope``;
            # SEARCH stays capped at 3 (budget-aware) since the generalized seed
            # runs at ab=3 and deeper pipelines are not worth searching.
            ab_stages_max = 6 if self.deep_direct_entry_validation_enabled else 3
            # Operands whose AB SMEM fits the per-CTA budget run the dtype's
            # deep AB pipeline (12 stages for fp8 and 16-bit operands): small
            # tiles such as the 256x64x64 two-CTA GEMM fit 9 stages, which is
            # what hides the TMA latency of a 16-step K loop on one-wave
            # shapes (fp16 1024^3: cuBLAS's 64x10 ring).
            # ``_validate_direct_entry_ab_stage_envelope`` clamps the depth
            # to the actual per-CTA SMEM budget for the chosen block sizes.
            constraints = self.ab_stages_three_search_constraints
            if constraints is not None:
                ab_stages_max = max(
                    ab_stages_max,
                    self._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes),
                )
        elif self.ab_stages_three_search_constraints is not None:
            # Cycle 97: make ab=3 BUDGET-AWARE-SEARCHABLE. Where the device/dtype
            # admits ab=3 at all (the SMEM-budget constraints were recorded by
            # ``allow_ab_stages_three_search`` at bind time — B200-class optin cap,
            # bf16/fp16), lift the ``for_search`` cap to 3 so the autotuner can
            # SAMPLE ab=3 directly instead of reaching it only through the per-shape
            # FFI / gelu seeds. ``_fix_ab_stages_search_config`` then demotes any
            # sampled ab=3 that does not fit (the residual/source-C ring overflows;
            # cluster_m=1 256x256 overflows bare-AB) before codegen, so admission is
            # free but an overflowing kernel is never generated.
            # Search up to the dtype's hardware-validated stage cap (6 for
            # 16-bit, 12 for fp8): smaller K tiles (bk=64) fit deep AB
            # pipelines that win on edge shapes (bf16 5000^3: ab=5/bk=64 at
            # 948 TFLOP/s vs the ab=2/bk=128 seed's 815), and the
            # budget-aware ``_validate_direct_entry_ab_stage_envelope``
            # fix-invalid pass clamps any sampled depth to the per-CTA SMEM
            # fit for the chosen block sizes, so over-deep samples degrade to
            # the old behavior instead of overflowing.
            constraints = self.ab_stages_three_search_constraints
            assert constraints is not None
            ab_stages_max = self._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes)
        else:
            ab_stages_max = 2
        if self.pipeline_smem_facts is not None and (not for_search or paired_search):
            ab_stages_max = max(ab_stages_max, TCGEN05_SMEM_AWARE_MAX_AB_STAGES)
        if for_search:
            l2_swizzle_choices = tuple(
                v for v in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if v <= 8
            )
        else:
            l2_swizzle_choices = TCGEN05_LEGAL_L2_SWIZZLE_SIZES
        fragments: dict[str, ConfigSpecFragment] = {
            TCGEN05_CTA_GROUP_CONFIG_KEY: EnumFragment(
                TCGEN05_CTA_GROUP_CHOICES
                if not for_search or paired_search
                else ("auto",)
            ),
            "tcgen05_cluster_m": EnumFragment(cluster_m_choices),
            "tcgen05_cluster_n": EnumFragment(cluster_n_choices),
            "tcgen05_ab_stages": IntegerFragment(1, ab_stages_max, 2),
            "tcgen05_acc_stages": IntegerFragment(1, 2, 2),
            "tcgen05_c_stages": EnumFragment((2, 4)),
            "tcgen05_num_epi_warps": num_epi_warps_fragment,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: EnumFragment(l2_swizzle_choices),
        }
        if self.matmul_has_leading_passthrough:
            # The persistent walk over a batched grid (see
            # ``TCGEN05_BATCH_RASTER_CONFIG_KEY``); plain grids raster through
            # ``loop_orders`` and do not carry the knob.
            fragments[TCGEN05_BATCH_RASTER_CONFIG_KEY] = EnumFragment(
                TCGEN05_BATCH_RASTER_CHOICES
            )
        if not for_search or self.grouped_full_coverage_eligible:
            fragments[TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY] = EnumFragment(
                TCGEN05_GROUPED_FULL_COVERAGE_MODES
            )
        if (
            not for_search
            or self.grouped_row_union_eligible
            or self.grouped_row_union_cluster4_eligible
            or self.grouped_row_union_paired_clc_eligible
        ):
            fragments[GROUPED_ROW_UNION_KEY] = BooleanFragment()
        if (
            not for_search
            or self.grouped_row_union_cluster4_eligible
            or self.grouped_row_union_paired_clc_eligible
        ):
            choices = [LEGACY_SCHEDULE]
            if not for_search or self.grouped_row_union_cluster4_eligible:
                choices.append(TRANSPOSED_SCHEDULE)
            if not for_search or self.grouped_row_union_paired_clc_eligible:
                choices.append(PAIRED_CLC_SCHEDULE)
            fragments[GROUPED_ROW_UNION_SCHEDULE_KEY] = EnumFragment(tuple(choices))
        if not for_search or self.grouped_row_union_paired_clc_eligible:
            fragments[STARTUP_PREFILL_KEY] = BooleanFragment()
        if not for_search or (
            self.grouped_row_union_eligible
            and self.grouped_row_union_multi_resident_supported
        ):
            fragments[GROUPED_RESIDENT_CTAS_KEY] = EnumFragment((1, 2))
        if not for_search or self.epilogue_fanout_search_enabled:
            fragments[FANOUT_CONFIG_KEY] = EnumFragment(FANOUT_MODES)
            fragments[TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY] = EnumFragment(
                TCGEN05_C_ACQUIRE_PLACEMENTS
            )
        # The register-direct output store (``tcgen05_c_store_mode="direct"``)
        # is an exact alternative to the SMEM-staged TMA store: no C ring, no
        # epilogue named barriers, no async-proxy fence, no bulk-store drain.
        # It wins when a CTA's whole epilogue is exposed (one or two tiles per
        # CTA) and loses to the TMA store's coalescing once the store drain
        # overlaps the next tile, so it is searched rather than decided.  The
        # field is on the search surface of every tcgen05 kernel: the family
        # flags below are set by later heuristics (the fan-out and grouped
        # registrations), and a field that came and went with them left the
        # coverage carriers normalized before the flip carrying a key the flat
        # schema no longer had.  Where the knob is inert -- the fan-out
        # epilogue owns the C ring, the grouped / row-union families keep
        # their own store protocols -- only the TMA store is sampled, and
        # ``normalize_pre_pid_type`` repairs a ``direct`` that a config pairs
        # with one of those protocols.  Explicit configs may always name the
        # mode on the plain protocol.
        direct_store_inert = (
            self.epilogue_fanout_search_enabled
            or self.grouped_full_coverage_eligible
            or self.grouped_row_union_eligible
            or self.grouped_row_union_cluster4_eligible
            or self.grouped_row_union_paired_clc_eligible
        )
        fragments[TCGEN05_C_STORE_MODE_CONFIG_KEY] = EnumFragment(
            (TCGEN05_C_STORE_MODE_NORMAL, TCGEN05_C_STORE_MODE_DIRECT)
            if for_search
            else TCGEN05_C_STORE_MODES,
            search_choices=(TCGEN05_C_STORE_MODE_NORMAL,)
            if for_search and direct_store_inert
            else None,
        )
        # The register-MMA GEMM family and its warp count are searched only
        # where the plan admitted the family (``warp_mma_admitted``); every
        # tcgen05-enabled kernel validates the keys so an explicit request on
        # a non-admitted problem is refused with the reason.
        if not for_search or self.warp_mma_admitted:
            fragments[WARP_MMA_FAMILY_KEY] = EnumFragment(MATMUL_FAMILIES)
            fragments[WARP_MMA_WARPS_KEY] = EnumFragment(WARP_MMA_WARP_CHOICES)
        if len(self.materialized_matmul_block_ids) >= 2:
            fragments["tcgen05_region_ab_stages"] = ListOf(
                IntegerFragment(0, 16, 0), len(self.materialized_matmul_block_ids)
            )
            fragments["tcgen05_region_c_stages"] = ListOf(
                EnumFragment((0, 2, 4)), len(self.materialized_matmul_block_ids)
            )
        if (
            len(self.materialized_matmul_block_ids) >= 2
            or self.materialized_operand_pdl_roots is not None
        ):
            fragments["tcgen05_materialized_pdl"] = BooleanFragment()
        if (
            self.aux_kernel_detected
            or self.epilogue_fanout_search_enabled
            or not for_search
        ):
            fragments[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = EnumFragment(
                TCGEN05_AUX_LOAD_PLACEMENTS
            )
        if not for_search:
            fragments[TCGEN05_GROUPED_MODE_CONFIG_KEY] = EnumFragment(
                TCGEN05_GROUPED_MODES
            )
            fragments[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = BooleanFragment()
            fragments[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = EnumFragment(
                (
                    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
                    *(
                        choice
                        for choice in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
                        if choice != TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
                    ),
                )
            )
        direct_entry_seed_eligible = self.full_tile_direct_entry_seed_eligible()
        if direct_entry_seed_eligible or (
            not for_search and self._direct_entry_k_block_index() is not None
        ):
            # Validation exposes the two direct-entry controls for explicit
            # configs. Layout overrides already have a generic validation path
            # below; only the seed/search surface narrows them to its fixed tile.
            # The FFI arm projects to one validated seed. Keep it as the
            # default and as an explicit/seed value, while random exploration
            # samples the ordinary arm. Sampling the seed's redundant role
            # and layout payload independently would request FFI promotion
            # again and erase the sampled geometry and pipeline choices.
            tvm_ffi_launch_fragment: ConfigSpecFragment = (
                EnumFragment((True, False), search_choices=(False,))
                if for_search
                else BooleanFragment()
            )
            flat_role_fragment: ConfigSpecFragment = (
                EnumFragment((False, True), search_choices=(False,))
                if for_search
                else BooleanFragment()
            )
            fragments.update(
                {
                    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: flat_role_fragment,
                    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: tvm_ffi_launch_fragment,
                }
            )
            if direct_entry_seed_eligible:
                fragments.update(
                    {
                        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: EnumFragment(
                            (None, 128), search_choices=(None,) if for_search else None
                        ),
                        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: EnumFragment(
                            (None, 32), search_choices=(None,) if for_search else None
                        ),
                        TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: EnumFragment(
                            (None, 32), search_choices=(None,) if for_search else None
                        ),
                    }
                )
        return fragments

    @staticmethod
    def _target1_tvm_ffi_promotion_requested(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            or config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
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
        # The generalized direct-entry seed projects FFI-requesting search
        # candidates onto the validated CtaGroup.TWO envelope for ANY eligible
        # shape (returns None for ineligible shapes, in which case the
        # promotion surface is stripped back to the DEFAULT layout below).
        seed = self.full_tile_direct_entry_seed_config(
            l2_groupings=cast("list[object] | None", config.get("l2_groupings"))
        )
        if not self._target1_tvm_ffi_promotion_requested(config):
            return
        if seed is None:
            config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY] = False
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
            return
        self._clear_target1_tvm_ffi_promotion_surface(config)
        config.update(seed.config)

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
        seed_choices = _compiler_seed_values(
            self.config_spec.compiler_seed_configs,
            TCGEN05_CONSUMER_REGS_CONFIG_KEY,
            int,
            lambda value: value in TCGEN05_CONSUMER_REGS_CHOICES,
        )
        if not self._aux_tma_edge_search_enabled():
            if not any(
                choice != TCGEN05_CONSUMER_REGS_DEFAULT for choice in seed_choices
            ):
                return {}
            return {
                TCGEN05_CONSUMER_REGS_CONFIG_KEY: _enum_fragment_with_seed_values(
                    (TCGEN05_CONSUMER_REGS_DEFAULT,),
                    seed_choices,
                    search_choices=(TCGEN05_CONSUMER_REGS_DEFAULT,),
                    search_only_if_widened=True,
                )
            }
        return {
            TCGEN05_CONSUMER_REGS_CONFIG_KEY: EnumFragment(
                TCGEN05_CONSUMER_REGS_CHOICES
            )
        }

    def persistence_model_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        pid_default_models = tuple(
            dict.fromkeys(
                derive_persistence_model_from_pid_type(pid_type).value
                for pid_type in self.allowed_pid_types
            )
        )
        seed_pid_types = _compiler_seed_values(
            self.config_spec.compiler_seed_configs,
            "pid_type",
            str,
            lambda pid_type: pid_type in TCGEN05_PERSISTENCE_MODEL_PID_TYPES,
            exact_type=False,
        )
        seed_pid_default_models = tuple(
            dict.fromkeys(
                derive_persistence_model_from_pid_type(pid_type).value
                for pid_type in seed_pid_types
            )
        )

        def seed_default_model(seed: Config) -> str:
            pid_type = seed.config.get("pid_type")
            if (
                isinstance(pid_type, str)
                and pid_type in TCGEN05_PERSISTENCE_MODEL_PID_TYPES
            ):
                return derive_persistence_model_from_pid_type(pid_type).value
            return self.persistence_model_default_from_config(seed.config).value

        seed_override_models = _compiler_seed_values(
            self.config_spec.compiler_seed_configs,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
            str,
            lambda model: model in {item.value for item in Tcgen05PersistenceModel},
            exact_type=False,
            is_valid_for_seed=lambda seed, model: model != seed_default_model(seed),
        )
        if not self._clc_persistence_search_enabled():
            seed_pid_defaults_widen_domain = any(
                model not in pid_default_models for model in seed_pid_default_models
            )
            if not seed_override_models and not seed_pid_defaults_widen_domain:
                return {}
            return {
                TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: _enum_fragment_with_seed_values(
                    pid_default_models,
                    (*seed_pid_default_models, *seed_override_models),
                    search_choices=pid_default_models,
                    search_only_if_widened=True,
                )
            }
        choices = tuple(
            dict.fromkeys(
                (
                    *pid_default_models,
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
        direct_entry_seed_eligible = self.full_tile_direct_entry_seed_eligible()
        if self.aux_kernel_detected:
            strategy_choices: tuple[str, ...] = (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            )
            scheduler_warps_choices: tuple[int, ...] = (0, 1)
            c_input_warps_choices: tuple[int, ...] = (0, 1)
        elif self._plain_clc_persistence_search_enabled():
            # Plain full-tile cluster_m=2 matmuls search the scheduler-warp
            # strategy too (its CLC dynamic persistence wins on large grids;
            # see _plain_clc_persistence_search_enabled). No C-input warp:
            # there is no aux/source-C producer to host.
            strategy_choices = (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            )
            scheduler_warps_choices = (0, 1)
            c_input_warps_choices = (0,)
        else:
            strategy_choices = (Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,)
            scheduler_warps_choices = (0,)
            c_input_warps_choices = (0,)
        strategy_seed_choices = _compiler_seed_values(
            self.config_spec.compiler_seed_configs,
            TCGEN05_STRATEGY_CONFIG_KEY,
            str,
            lambda strategy: strategy in {item.value for item in Tcgen05Strategy},
            exact_type=False,
        )
        scheduler_seed_choices = _compiler_seed_values(
            self.config_spec.compiler_seed_configs,
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
            int,
            lambda scheduler_warps: scheduler_warps in (0, 1),
        )
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
            TCGEN05_STRATEGY_CONFIG_KEY: _enum_fragment_with_seed_values(
                strategy_choices,
                strategy_seed_choices,
                search_choices=strategy_choices,
                search_only_if_widened=True,
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: EnumFragment(
                layout_choices,
                search_choices=(Tcgen05LayoutStrategy.DEFAULT.value,)
                if direct_entry_seed_eligible
                else None,
            ),
            TCGEN05_WARP_SPEC_MMA_WARPS_KEY: EnumFragment((1,)),
            TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY: EnumFragment((1,)),
            TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY: EnumFragment((0,)),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: _enum_fragment_with_seed_values(
                scheduler_warps_choices,
                scheduler_seed_choices,
                search_choices=scheduler_warps_choices,
                search_only_if_widened=True,
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
            return
        message = "; ".join(errors)
        raise InvalidConfig(f"tcgen05 strategy invariants violated: {message}")

    def _clamp_l2_swizzle_size_to_shape(self, config: dict[str, object]) -> None:
        if TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY in config:
            # A grouped signature has no single N extent to clamp against.
            # Preserve the requested value so lowering can reject swizzling
            # that the static grouped scheduler does not support.
            return
        if (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            == TCGEN05_GROUPED_MODE_WORKLIST_NM
        ):
            # Runtime-direct grouped N,M builds the panel raster from each
            # group's host-visible tile count, so there is no single static
            # shape to clamp against here. ``prepare_normalization`` already
            # forces the legacy device-search scheduler to use swizzle 1.
            return
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
                        or self._grouped_worklist_nm_ab_config_matches(
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
            self._validate_cta_group_config(config, fix_invalid=fix_invalid)
            self._validate_direct_entry_ab_stage_envelope(
                config, fix_invalid=fix_invalid
            )
        else:
            for key, fragment in optional_fragments.items():
                if key not in config:
                    continue
                # Cross-shape config reuse (``Kernel.configs`` pinning, the
                # autotune cache) legitimately carries default-valued tcgen05
                # keys tuned on a tcgen05-capable bind onto binds that are not
                # (e.g. a unit-sized M); inert defaults are dropped. Values
                # that encode a real tcgen05 strategy still fail loudly — the
                # config cannot be honored and silently changing strategy
                # could mask a miscompute (see
                # ``test_batched_two_cta_partial_edge_tiles_rejected``).
                if fix_invalid or config[key] == fragment.default():
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
                strategy_fragment = strategy_validation_fragments.get(key)
                default_value = (
                    strategy_fragment.default()
                    if strategy_fragment is not None
                    # The layout-override keys have no fragment here; ``None``
                    # is their inert value.
                    else None
                )
                if fix_invalid or config[key] == default_value:
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
                # The register-direct store is an exact alternative body, not a
                # diagnostic: it needs no invalid-output opt-in.
                (TCGEN05_C_STORE_MODE_NORMAL, TCGEN05_C_STORE_MODE_DIRECT),
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
        # The split / module-helper layouts rearrange the role-local
        # TMA-store epilogue (T2R halves, store tails); the register-direct
        # store renders none of it and codegen rejects the pair.  Repaired
        # (search) configs fall back to the normal layout instead of burning
        # a compile failure on the combination; explicit configs are told.
        if (
            config.get(TCGEN05_C_STORE_MODE_CONFIG_KEY) == TCGEN05_C_STORE_MODE_DIRECT
            and config.get(
                TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY, TCGEN05_EPILOGUE_LAYOUT_NORMAL
            )
            != TCGEN05_EPILOGUE_LAYOUT_NORMAL
        ):
            if fix_invalid:
                config[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY] = (
                    TCGEN05_EPILOGUE_LAYOUT_NORMAL
                )
            else:
                raise InvalidConfig(
                    f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}="
                    f"{config[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY]!r} splits the "
                    "TMA-store epilogue; the register-direct "
                    f"{TCGEN05_C_STORE_MODE_CONFIG_KEY}="
                    f"{TCGEN05_C_STORE_MODE_DIRECT!r} renders no TMA store"
                )
        # The register-direct store is the plain role-local store protocol.
        # The fan-out epilogue owns the C ring and the grouped / row-union
        # families keep their own store protocols, so ``direct`` paired with
        # one of them in a config changes nothing of the kernel.  Repaired
        # (search) configs keep the protocol and fall back to the TMA store;
        # explicit configs are told.
        if config.get(TCGEN05_C_STORE_MODE_CONFIG_KEY) == TCGEN05_C_STORE_MODE_DIRECT:
            other_protocol = None
            if config.get(FANOUT_CONFIG_KEY, FANOUT_MODES[0]) != FANOUT_MODES[0]:
                other_protocol = f"{FANOUT_CONFIG_KEY}={config[FANOUT_CONFIG_KEY]!r}"
            elif config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) in TCGEN05_GROUPED_MODES:
                other_protocol = (
                    f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                    f"{config[TCGEN05_GROUPED_MODE_CONFIG_KEY]!r}"
                )
            elif config.get(GROUPED_ROW_UNION_KEY, False) is not False:
                other_protocol = (
                    f"{GROUPED_ROW_UNION_KEY}={config[GROUPED_ROW_UNION_KEY]!r}"
                )
            if other_protocol is not None:
                if fix_invalid:
                    config[TCGEN05_C_STORE_MODE_CONFIG_KEY] = (
                        TCGEN05_C_STORE_MODE_NORMAL
                    )
                else:
                    raise InvalidConfig(
                        f"{other_protocol} owns the output store; the register-direct "
                        f"{TCGEN05_C_STORE_MODE_CONFIG_KEY}="
                        f"{TCGEN05_C_STORE_MODE_DIRECT!r} is the plain store protocol"
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
        normal_mode: str | tuple[str, ...],
        *,
        fix_invalid: bool,
    ) -> None:
        """Validate a mode knob; ``normal_mode`` lists the exact (non-diagnostic) values."""
        exact_modes = (normal_mode,) if isinstance(normal_mode, str) else normal_mode
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
            config[key] not in exact_modes
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
        if self._materialized_paired_stage_limits(config) is not None:
            # Each launch has its own proved axes and budget. The single-matrix
            # projection below cannot choose geometry for this complete recipe.
            self._validate_cta_group_config(config, fix_invalid=True)
            return
        if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) == "two":
            if self.paired_pipeline_search_enabled():
                config["tcgen05_cluster_m"] = 2
                config[TCGEN05_STRATEGY_CONFIG_KEY] = (
                    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
                )
                config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
                    Tcgen05PersistenceModel.STATIC_PERSISTENT.value
                )
                config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = (
                    Tcgen05LayoutStrategy.DEFAULT.value
                )
                # The paired pipeline disables direct-entry controls. Repair
                # must use the search schema's canonical representation, so
                # inactive controls cannot survive only one side of flatten.
                search_fields = self.optional_fragments(for_search=True)
                for key in (
                    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
                    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
                ):
                    if key in search_fields:
                        config[key] = False
                    else:
                        config.pop(key, None)
                for key in TCGEN05_LAYOUT_OVERRIDES_KEYS:
                    config[key] = None
                for key in (
                    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
                    TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
                    TCGEN05_WARP_SPEC_STORE_WARPS_KEY,
                ):
                    config[key] = 0
                self._fix_cluster_m2_search_config(config)
                self._clamp_l2_swizzle_size_to_shape(config)
                self._validate_cta_group_config(config, fix_invalid=True)
                self._validate_direct_entry_ab_stage_envelope(config, fix_invalid=True)
                return
            config[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
        self._fix_grouped_worklist_search_config(config)
        self._fix_aux_edge_search_config(config)
        self._fix_cluster_m2_search_config(config)
        self._fix_cluster_m1_persistent_search_config(config)
        if (
            self.search_enabled
            and self.matmul_input_dtype is not None
            and self.matmul_input_dtype.itemsize <= 2
            # A 16-bit MMA fed by a cast from a non-native load (bf16 x int16)
            # takes the non-tcgen05 fallback and keeps the knob; fp8 operands
            # trip the same flag but run on tcgen05.
            and not (
                self.matmul_has_non_tcgen05_operand
                and self.matmul_input_dtype.itemsize == 2
            )
        ):
            # The tcgen05 store splice emits one store per output tile, so a
            # sampled ``epilogue_subtile`` only makes a 16-bit / fp8 candidate
            # fail at codegen (the cluster_m=2 projection already drops it; on
            # one CTA it cost 16 of the 165 compiles of the fp8 512x1024x512
            # cold autotune). fp32 keeps the knob, which there selects the
            # exact SIMT lowering.
            config.pop("epilogue_subtile", None)
        if (
            self.search_enabled
            and config.get("tcgen05_cluster_m", 1) == 1
            and config.get("tcgen05_cluster_n", 1) == 2
        ):
            # The validated N multicast needs the two-CTA M family. A
            # one-CTA sample must retain its geometry with no N multicast;
            # explicit configurations still go through the unchanged guard.
            config["tcgen05_cluster_n"] = 1
        self._fix_ab_stages_three_search_config(config)
        self._fix_target1_tvm_ffi_search_config(config)
        self._fix_aux_tma_full_tile_search_config(config)
        self._fix_with_scheduler_search_config(config)
        self._fix_aux_tma_search_config(config)
        # Cycle 97: budget-aware ab-stages admission for the lifted for_search
        # cap. Runs after the family projections (FFI / gelu / aux-TMA full-tile)
        # have set their validated stage tuple, so a directly-SAMPLED ab=3 that no
        # projection claimed — and that does not fit (residual/source-C ring
        # overflow, or cluster_m=1 256x256 bare-AB overflow) — is demoted to 2 on
        # the DEFAULT-layout full-tile path before codegen. The plain silu/gelu
        # ab=3 winner (no source-C, cluster_m=2) is preserved.
        self._fix_ab_stages_search_config(config)
        # Final c-stages admission: demote a directly-sampled (or any unclaimed)
        # deeper C ring on the canonical 256x256 DEFAULT-layout path that does
        # not fit AB+C under the B200 cap. Runs after the family fixups so their
        # validated c=4 (residual ab=2 projection above; edge/seed families,
        # which this gate's scope excludes) is set first.
        self._fix_c_stages_search_config(config)
        # A sampled ``pre_acc_wait`` stages its rows next to the final rings;
        # demote it where the stage does not fit (the seeds' gate, applied to
        # the projection).
        self._fix_rowvec_stage_search_config(config)
        # CLC admission depends on the projected cluster/pid/strategy tuple
        # above, including the scheduler/c-input warp fix-ups.
        self._fix_clc_persistence_search_config(config)
        self._validate_sched_stage_count_config(config, fix_invalid=True)
        self._validate_cta_group_config(config, fix_invalid=True)
        self._validate_direct_entry_ab_stage_envelope(config, fix_invalid=True)

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
        self._normalize_grouped_full_coverage(
            config, fix_invalid=fix_invalid, validate_schedule=True
        )
        self._normalize_grouped_row_union(
            config, fix_invalid=fix_invalid, validate_schedule=True
        )
        self._normalize_epilogue_fanout(
            config, fix_invalid=fix_invalid, validate_schedule=True
        )

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
        seeds = self.config_spec.compiler_seed_configs
        if isinstance(fragment := fields.get("tcgen05_ab_stages"), IntegerFragment):
            fields["tcgen05_ab_stages"] = _integer_fragment_with_seed_values(
                fragment,
                _compiler_seed_values(
                    seeds, "tcgen05_ab_stages", int, lambda value: value > 0
                ),
            )
        l2_key = TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
        if isinstance(fragment := fields.get(l2_key), EnumFragment):
            fields[l2_key] = _enum_fragment_with_seed_values(
                fragment.choices,
                _compiler_seed_values(
                    seeds,
                    l2_key,
                    int,
                    lambda value: value in TCGEN05_LEGAL_L2_SWIZZLE_SIZES,
                ),
                search_choices=fragment.search_choices or fragment.choices,
                original=fragment,
            )
        modes = _compiler_seed_values(
            seeds,
            TCGEN05_GROUPED_MODE_CONFIG_KEY,
            str,
            lambda value: value in TCGEN05_GROUPED_MODES,
            exact_type=False,
        )
        if any(mode in TCGEN05_GROUPED_DYNAMIC_MODES for mode in modes):
            choices = TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES
            fields[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] = (
                _enum_fragment_with_seed_values(
                    choices,
                    _compiler_seed_values(
                        seeds,
                        TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
                        int,
                        lambda value: (
                            0 <= value <= TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
                        ),
                    ),
                    search_choices=choices,
                )
            )
        if modes:
            fields[TCGEN05_GROUPED_MODE_CONFIG_KEY] = _enum_fragment_with_seed_values(
                (None,), modes, search_choices=(None,)
            )
        runtime_direct = _compiler_seed_values(
            seeds,
            TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
            bool,
            lambda value: value is True,
        )
        if runtime_direct:
            fields[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = (
                _enum_fragment_with_seed_values(
                    (False,), runtime_direct, search_choices=(False,)
                )
            )
        source_m_tiles = _compiler_seed_values(
            seeds,
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
            int,
            lambda value: (
                value in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
            ),
        )
        if source_m_tiles:
            fields[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = (
                _enum_fragment_with_seed_values(
                    (None,), source_m_tiles, search_choices=(None,)
                )
            )
        fields.update(self.strategy_autotune_fragments())
        fields.update(self.aux_load_mode_autotune_fragments())
        fields.update(self.aux_stages_autotune_fragments())
        fields.update(self.consumer_regs_autotune_fragments())
        fields.update(self.persistence_model_autotune_fragments())
        if self.config_spec.supports_config_key("pid_type"):
            fields["pid_type"] = _enum_fragment_with_seed_values(
                self.allowed_pid_types,
                _compiler_seed_values(
                    seeds,
                    "pid_type",
                    str,
                    lambda _value: True,
                    exact_type=False,
                ),
                search_choices=self.allowed_pid_types,
                search_only_if_widened=True,
            )
        if (
            self.config_spec.supports_config_key("indexing")
            and self.config_spec.indexing.length > 0
        ):
            fields["indexing"] = self.config_spec.indexing
        return fields
