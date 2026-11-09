"""Key tables and seed-widening fragment helpers of the tcgen05 search surface.

The ``CUTE_TCGEN05_*_KEYS`` tables partition every ``tcgen05_*`` config key
into tunables, grouped-family schedule fields, diagnostics and codegen
proofs; the fragment helpers merge the compiler seeds' values into a search
fragment without widening what the random search samples.  A leaf module:
nothing here reads the state object.
"""

from __future__ import annotations

from collections.abc import Hashable
import os
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
from typing import cast

from ....autotuner.config_fragment import ConfigSpecFragment
from ....autotuner.config_fragment import EnumFragment
from ....autotuner.config_fragment import IntegerFragment
from ....exc import InvalidConfig
from ..cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY
from ..cute_warp_mma_gemm import WARP_MMA_WARPS_KEY
from ..dense_aligned_n import TILE_N_KEY
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..grouped_panel_raster import GROUPED_PANEL_ALTERNATE_KEY
from ..grouped_panel_raster import GROUPED_PANEL_RATIO_KEY
from ..grouped_row_union import AB_STAGES_KEY as GROUPED_ROW_UNION_AB_STAGES_KEY
from ..grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from ..grouped_row_union import RESIDENT_CTAS_KEY as GROUPED_RESIDENT_CTAS_KEY
from ..grouped_row_union import SCHEDULE_KEY as GROUPED_ROW_UNION_SCHEDULE_KEY
from ..grouped_row_union import STARTUP_PREFILL_KEY
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..strategies import TCGEN05_BATCH_RASTER_CONFIG_KEY
from ..strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from ..strategies import TCGEN05_NUM_EPI_WARPS_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEYS
from ..tcgen05.knobs import AB_STAGES_KEY
from ..tcgen05.knobs import ACC_STAGES_KEY
from ..tcgen05.knobs import C_STAGES_KEY
from ..tcgen05.knobs import CLUSTER_M_KEY
from ..tcgen05.knobs import CLUSTER_N_KEY
from ..tcgen05.knobs import GROUPED_WORKLIST_ORIENTATION_KEY
from ..tcgen05.knobs import MATERIALIZED_PDL_KEY
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACING_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MMA_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_LONG_N_SWEEP_L2_GROUPING
from ..tcgen05_constants import TCGEN05_LONG_N_SWEEP_MIN_TILES
from ..tcgen05_constants import TCGEN05_REGION_AB_STAGES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_REGION_C_STAGES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ....autotuner.config_fragment import BlockSizeFragment
    from ....runtime.config import Config


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
    ``output_dtype_bytes`` / ``output_stores`` describe the TMA-store rings
    (one per output store) the fit estimate charges beyond that reservation;
    ``None`` keeps the AB-only estimate.
    """

    dtype_bytes: int
    per_cta_smem_budget_bytes: int
    output_dtype_bytes: int | None = None
    output_stores: int = 1


class Tcgen05GroupedWorklistSmemFacts(NamedTuple):
    group_count: int
    device_split_sizes: bool
    output_dtype_bytes: int = 2


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


# Search axes validated wider than they are searched.  Explicit configs and
# compiler seeds may use every validated value (``flat_fields`` merges seed
# values back into the search fragment); the random and mutation search never
# samples the rest:
#   - ``tcgen05_num_epi_warps``: only four epilogue warps lower correctly (1 and
#     2 produce wrong output on B200, 3 is unsafe by extension), so the search
#     is pinned unless ``restrict_num_epi_warps_search`` lifts the gate;
#     validation stays at ``IntegerFragment(1, 4, 4)`` until
#     ``restrict_num_epi_warps_validation`` narrows it.
#   - ``tcgen05_c_stages``: one stage per epilogue warp group (1) and the
#     three-stage ring are seed-only depths; the search samples the historical
#     2 and 4 (the first choice stays the default).
#   - ``tcgen05_l2_swizzle_size``: 16 and 32 are legal but never searched.
TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES: tuple[int, ...] = (4,)
TCGEN05_C_STAGES_CHOICES: tuple[int, ...] = (2, 3, 4, 1)
TCGEN05_C_STAGES_SEARCH_CHOICES: tuple[int, ...] = (2, 4)
TCGEN05_L2_SWIZZLE_SEARCH_SIZES: tuple[int, ...] = tuple(
    size for size in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if size <= 8
)
CUTE_TCGEN05_TUNABLE_KEYS: tuple[str, ...] = (
    GROUPED_ROW_UNION_KEY,
    GROUPED_ROW_UNION_SCHEDULE_KEY,
    GROUPED_ROW_UNION_AB_STAGES_KEY,
    STARTUP_PREFILL_KEY,
    GROUPED_RESIDENT_CTAS_KEY,
    FANOUT_CONFIG_KEY,
    TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
    TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY,
    TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY,
    TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY,
    TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY,
    TCGEN05_CTA_GROUP_CONFIG_KEY,
    TILE_N_KEY,
    CLUSTER_M_KEY,
    CLUSTER_N_KEY,
    AB_STAGES_KEY,
    TCGEN05_REGION_AB_STAGES_CONFIG_KEY,
    TCGEN05_REGION_C_STAGES_CONFIG_KEY,
    TCGEN05_REGION_PID_TYPES_CONFIG_KEY,
    MATERIALIZED_PDL_KEY,
    ACC_STAGES_KEY,
    C_STAGES_KEY,
    TCGEN05_SPLIT_K_CONFIG_KEY,
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
    TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY,
    TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY,
    TCGEN05_EPI_SUBTILE_N_CONFIG_KEY,
    TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY,
    TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY,
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_STORE_MODE_CONFIG_KEY,
    TCGEN05_NUM_EPI_WARPS_CONFIG_KEY,
    TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
    TCGEN05_BATCH_RASTER_CONFIG_KEY,
    TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY,
    TCGEN05_CLC_QUERY_PACING_CONFIG_KEY,
    WARP_MMA_FAMILY_KEY,
    WARP_MMA_WARPS_KEY,
    # Pipeline and launch knobs with seeds, search fragments or a default
    # the normalizer fills (the tunables formerly filed as diagnostics).
    TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
    TCGEN05_AUX_STAGES_CONFIG_KEY,
    TCGEN05_CONSUMER_REGS_CONFIG_KEY,
    TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY,
    TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
    TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY,
    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
)
# Diagnostic knobs: debug toggles that change the rendered kernel (cubin
# line info) or unlock a deliberately wrong one (``tcgen05_c_store_mode=
# skip_epilogue_store`` is the only knob that needs the invalid-output
# opt-in).  Never seeded, never searched; every other tcgen05 key is a
# tunable, a grouped-family schedule field or a codegen proof.
CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
        TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
    }
)
# Compile-only proofs: the large-BN envelope.
CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS: frozenset[str] = frozenset(
    {TCGEN05_LARGE_BN_PROOF_CONFIG_KEY}
)
# The grouped (ragged / worklist) family's schedule state: written by its
# planner, its carriers and the AOT configs, validated per key and searched
# only through the fragments the planner admits.
CUTE_TCGEN05_GROUPED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY,
        TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY,
        TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY,
        TCGEN05_GROUPED_MODE_CONFIG_KEY,
        TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
        TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY,
        TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
        TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY,
        GROUPED_WORKLIST_ORIENTATION_KEY,
        TCGEN05_GROUPED_WORKLIST_MMA_N_CONFIG_KEY,
        GROUPED_PANEL_ALTERNATE_KEY,
        GROUPED_PANEL_RATIO_KEY,
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
# Ownership key: proved schedules may lift only this CuTe-owned pid veto.
_TWO_CTA_PERSISTENT_PID_VETO = (
    "tcgen05 two-CTA launch-grid contract does not allow persistent pid types here"
)


def _tcgen05_mma_impl_allowed() -> bool:
    """Whether HELION_CUTE_MMA_IMPL admits tcgen05; read per call for tests."""
    return os.environ.get("HELION_CUTE_MMA_IMPL", "auto").strip().lower() in (
        "auto",
        "tcgen05",
    )


def _tile_in_fragments(
    fragments: Sequence[BlockSizeFragment], tile: Sequence[int]
) -> bool:
    return all(
        fragment.low <= value <= fragment.high
        for fragment, value in zip(fragments, tile, strict=True)
    )


def _stored_block_n(config: Mapping[str, object], bn: int) -> int:
    """The N extent a tile stores: the aligned-N family's physical width
    behind its power-of-two carrier ``bn``, else ``bn``."""
    width = config.get(TILE_N_KEY, 0)
    return width if type(width) is int and width > 0 else bn


def long_n_sweep_l2_grouping(
    n_tiles: int, *, short: int, long: int = TCGEN05_LONG_N_SWEEP_L2_GROUPING
) -> int:
    """The M tiles a grouped raster walks per group over ``n_tiles`` N
    tiles: ``long`` on a long N sweep (at least
    ``TCGEN05_LONG_N_SWEEP_MIN_TILES`` N tiles: the group's A panels stay
    resident in L2 while it sweeps N, so B is re-read once per group),
    ``short`` otherwise."""
    return long if n_tiles >= TCGEN05_LONG_N_SWEEP_MIN_TILES else short
