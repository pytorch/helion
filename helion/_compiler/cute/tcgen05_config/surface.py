"""The tcgen05 search surface: fragments, flat fields and search toggles.

Module functions over the ``CuteTcgen05Config`` state object that decide
which knobs are searched and validated (``optional_fragments``,
``flat_fields``, the ``*_autotune_fragments`` builders), which search arms
are enabled (``allow_*`` / ``restrict_*`` / ``*_search_enabled``) and the
flatten / default projections the autotuner's persistence model needs.
``optional_fragments`` and ``flat_fields`` insertion order is the
``ConfigGeneration.flat_spec`` order and feeds the structural fingerprint:
it is part of the surface identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ....autotuner.config_fragment import BooleanFragment
from ....autotuner.config_fragment import ConfigSpecFragment
from ....autotuner.config_fragment import EnumFragment
from ....autotuner.config_fragment import IntegerFragment
from ....autotuner.config_fragment import ListOf
from ....exc import InvalidConfig
from ...compile_environment import FixedBlockSizeSource
from .. import target_facts
from ..cute_warp_mma_gemm import MATMUL_FAMILIES
from ..cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
from ..cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY
from ..cute_warp_mma_gemm import WARP_MMA_WARP_CHOICES
from ..cute_warp_mma_gemm import WARP_MMA_WARPS_KEY
from ..dense_aligned_n import ALIGNED_N_CHOICES
from ..dense_aligned_n import TILE_N_KEY
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..epilogue_fanout import FANOUT_MODES
from ..grouped_panel_raster import GROUPED_PANEL_ALTERNATE_KEY
from ..grouped_panel_raster import GROUPED_PANEL_RATIO_CHOICES
from ..grouped_panel_raster import GROUPED_PANEL_RATIO_KEY
from ..grouped_row_union import AB_STAGES_CHOICES as GROUPED_ROW_UNION_AB_STAGES_CHOICES
from ..grouped_row_union import AB_STAGES_KEY as GROUPED_ROW_UNION_AB_STAGES_KEY
from ..grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from ..grouped_row_union import LEGACY_SCHEDULE
from ..grouped_row_union import PAIRED_CLC_SCHEDULE
from ..grouped_row_union import RESIDENT_CTAS_KEY as GROUPED_RESIDENT_CTAS_KEY
from ..grouped_row_union import SCHEDULE_KEY as GROUPED_ROW_UNION_SCHEDULE_KEY
from ..grouped_row_union import STARTUP_PREFILL_KEY
from ..grouped_row_union import TRANSPOSED_SCHEDULE
from ..grouped_row_union import physical_schedule
from ..pipeline_smem import TCGEN05_CTA_GROUP_CHOICES
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..pipeline_smem import TCGEN05_SMEM_AWARE_MAX_AB_STAGES
from ..pipeline_smem import max_pipeline_ab_stages
from ..split_k_workspace_config import workspace_search_fields
from ..strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from ..strategies import TCGEN05_BATCH_RASTER_CHOICES
from ..strategies import TCGEN05_BATCH_RASTER_CONFIG_KEY
from ..strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from ..strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from ..strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENCE_MODEL_PID_TYPES
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_CHOICES
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY
from ..strategies import TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_MMA_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY
from ..strategies import TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY
from ..strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
from ..strategies import Tcgen05LayoutStrategy
from ..strategies import Tcgen05PersistenceModel
from ..strategies import Tcgen05Strategy
from ..strategies import derive_persistence_model_from_pid_type
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENTS
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_MAX_SUBTILES
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from ..tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from ..tcgen05_constants import TCGEN05_C_STORE_MODES
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACING_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACINGS
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_DEFAULT
from ..tcgen05_constants import TCGEN05_EPI_ROUND_SUBTILES_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUP_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_DYNAMIC_MODES
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_MODES
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_OFF
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_DIRECT
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_DYNAMIC
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_STATIC
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from ..tcgen05_constants import TCGEN05_GROUPED_MODES
from ..tcgen05_constants import TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MMA_M_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_CHOICES
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_INHERIT
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SPLIT_K_SEARCH_CHOICES
from ..tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ..tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..tcgen05_constants import TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY
from ..tcgen05_schedule_facts import Tcgen05FullTileScheduleFacts
from ..tcgen05_schedule_facts import native_tma_schedule_supported
from . import facts
from . import repair
from . import seeds
from . import smem

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ....autotuner.block_id_sequence import BlockIdSequence
    from ....runtime.config import Config
    from ....runtime.config import PidTypeLiteral
    from ...compile_environment import BlockSizeInfo
    from ...compile_environment import CompileEnvironment


from .knobs import _TWO_CTA_PERSISTENT_PID_VETO
from .knobs import TCGEN05_C_STAGES_CHOICES
from .knobs import TCGEN05_C_STAGES_SEARCH_CHOICES
from .knobs import TCGEN05_L2_SWIZZLE_SEARCH_SIZES
from .knobs import TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES
from .knobs import Tcgen05AbStagesThreeSearchConstraints
from .knobs import Tcgen05ClusterM2SearchConstraints
from .knobs import _compiler_seed_values
from .knobs import _enum_fragment_with_seed_values
from .knobs import _integer_fragment_with_seed_values
from .knobs import _tile_in_fragments
from .knobs import parse_tcgen05_grouped_static_problem_signature

if TYPE_CHECKING:
    from . import CuteTcgen05Config


def cap_k_search(state: CuteTcgen05Config, block: BlockSizeInfo, maximum: int) -> None:
    """Apply the native K search ceiling, remembering a K128 cap it owns.

    MMA analysis registers once per kernel, so this runs at most once.
    """
    try:
        spec = state.config_spec.block_sizes.block_id_lookup(block.block_id)
    except KeyError:
        return
    previous_max = spec.max_size
    block.update_max_block(maximum)
    # Own the cap only when it lowered a wider K search range.
    if maximum == spec.max_size == 128 and previous_max >= 256:
        state._native_k_search_cap = (
            block.block_id,
            spec.max_size_revision,
            spec,
        )


def widen_one_cta_pipeline_k_search(
    state: CuteTcgen05Config, env: CompileEnvironment
) -> bool:
    """Replace an owned K128 cap only after the complete native proof."""
    pipeline_facts = state.one_cta_pipeline_facts
    cap = state._native_k_search_cap
    fragments = facts._matmul_block_fragments(state)
    if (
        pipeline_facts is None
        or cap is None
        or fragments is None
        or cap[0] != pipeline_facts.schedule.block_ids[2]
        or pipeline_facts.schedule.shape[2] % 256 != 0
        or isinstance(env.block_sizes[cap[0]].block_size_source, FixedBlockSizeSource)
    ):
        return False
    block = state.config_spec.block_sizes.block_id_lookup(cap[0])
    if (
        block is not cap[2]
        or block.bounded_by_block_id is not None
        or block.max_size != 128
        or block.max_size_revision != cap[1]
        or block.dim_max_size < 256
    ):
        return False
    pid = facts._preferred_persistent_pid_type(state)
    if pid is None:
        return False
    for bm in (64, 128):
        for bn in (16, 32, 64, 128, 256):
            if not _tile_in_fragments(fragments[:2], (bm, bn)):
                continue
            sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=256)
            assert sizes is not None
            config: dict[str, object] = {
                "block_sizes": sizes,
                "pid_type": pid,
                "tcgen05_aux_load_placement": "post_acc_wait",
            }
            limit = smem._one_cta_pipeline_stage_limit(state, config)
            if limit is None or limit < 2:
                continue
            old_default = block._fragment(state.config_spec).default()
            block.max_size = 256
            block.default_size = old_default
            state._native_k_widened_block = block
            return True
    return False


def widen_paired_pipeline_k_search(
    state: CuteTcgen05Config, env: CompileEnvironment
) -> bool:
    """Replace only our K128 cap for a proved physical paired FP8 tile."""
    dense_facts = state.dense_gemm_facts
    allocations = state.pipeline_smem_facts
    cap = state._native_k_search_cap
    fragments = facts._matmul_block_fragments(state)
    tile = (128, 64, 256)
    if (
        dense_facts is None
        or allocations is None
        or not dense_facts.output_column_major
        or not allocations.output_column_major
        or dense_facts.input_dtype != torch.float8_e4m3fn
        or cap is None
        or fragments is None
        or cap[0] != dense_facts.block_ids[2]
        or not _tile_in_fragments(fragments[:2], tile[:2])
        or facts._preferred_persistent_pid_type(state) is None
        or not native_tma_schedule_supported(
            Tcgen05FullTileScheduleFacts(
                dense_facts.block_ids,
                dense_facts.shape,
                dense_facts.input_dtype,
                rhs_k_major=True,
            ),
            tile,
            cluster_m=2,
            cluster_n=1,
        )
        or isinstance(env.block_sizes[cap[0]].block_size_source, FixedBlockSizeSource)
    ):
        return False
    block = state.config_spec.block_sizes.block_id_lookup(cap[0])
    if (
        block is not cap[2]
        or block.bounded_by_block_id is not None
        or block.max_size != 128
        or block.max_size_revision != cap[1]
        or block.dim_max_size < 256
        or max_pipeline_ab_stages(
            allocations, bm=128, bn=64, bk=256, c_stages=2, acc_stages=2
        )
        < 2
    ):
        return False
    old_default = block._fragment(state.config_spec).default()
    block.max_size = 256
    block.default_size = old_default
    state._native_k_widened_block = block
    return True


def paired_pipeline_search_enabled(state: CuteTcgen05Config) -> bool:
    pipeline_facts = state.pipeline_smem_facts
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    if (
        not state.search_enabled
        or pipeline_facts is None
        or extents is None
        or fragments is None
        or any(value is None or value <= 0 for value in extents)
    ):
        return False
    if pipeline_facts.input_dtype_bytes == 1:
        return (
            "persistent_blocked" in state.allowed_pid_types
            and facts._fp8_paired_pipeline_tile_supported(state, 128, 128, 64)
            and _tile_in_fragments(fragments, (128, 128, 64))
        )
    m, n, k = cast("tuple[int, int, int]", extents)
    return (
        m % 128 == 0
        and n % 64 == 0
        and k % 64 == 0
        and _tile_in_fragments(fragments, (128, 64, 64))
    )


def region_pid_type_choices(state: CuteTcgen05Config) -> tuple[str, ...] | None:
    """Pid types the GEMM launches of an ordered barrier chain may take.

    ``None`` where ``tcgen05_region_pid_types`` has no subject: a kernel
    that is not an ordered barrier chain, a fission plan (its persistent
    pid types already apply to every launch), or a chain some region of
    which lacks the complete single-output allocation facts the two-CTA
    budget is proved from.  A chain with one GEMM launch beside plain
    phases has one entry.
    """
    if (
        not state.region_pid_type_search_enabled
        or not state.materialized_matmul_block_ids
        or "flat" not in state.allowed_pid_types
        or len(state.materialized_pipeline_facts)
        != len(state.materialized_matmul_block_ids)
        or any(
            pipeline_facts is None
            for pipeline_facts in state.materialized_pipeline_facts
        )
    ):
        return None
    return TCGEN05_REGION_PID_TYPE_CHOICES


def _region_pid_types(
    state: CuteTcgen05Config, config: dict[str, object]
) -> tuple[object, ...]:
    """The pid type each GEMM launch runs under, by region position.

    A flat ordered chain with a well-formed ``tcgen05_region_pid_types``
    runs each GEMM launch under its entry (``inherit`` keeps the flat
    launch; ``_stage_config`` projects the persistent ones); everywhere
    else every launch takes the kernel's ``pid_type``.
    """
    pid_type = config.get("pid_type")
    count = len(state.materialized_matmul_block_ids)
    values = config.get(TCGEN05_REGION_PID_TYPES_CONFIG_KEY)
    if (
        pid_type == "flat"
        and region_pid_type_choices(state) is not None
        and isinstance(values, list)
        and len(values) == count
        and all(value in TCGEN05_REGION_PID_TYPE_CHOICES for value in values)
    ):
        return tuple(
            pid_type if value == TCGEN05_REGION_PID_TYPE_INHERIT else value
            for value in values
        )
    return (pid_type,) * count


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
            raise InvalidConfig(f"{name} requires a list of length {fragment.length}")
        return [
            _validate_optional_fragment_value(f"{name}[{index}]", fragment.inner, item)
            for index, item in enumerate(value)
        ]
    raise InvalidConfig(f"Unsupported optional fragment type for {name}")


def restrict_cluster_m_search(
    state: CuteTcgen05Config, choices: tuple[int, ...]
) -> None:
    assert choices, "tcgen05_cluster_m search must allow at least one value"
    if state.config_spec.has_explicit_pdl:
        choices = (1,)
    state.cluster_m_search_choices = choices
    if 2 not in choices:
        # Also disables clustered heuristic/FFI seeds after an allow request.
        state.cluster_m2_search_constraints = None


def allow_cluster_m2_search(
    state: CuteTcgen05Config,
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
    state.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
        static_k=static_k,
        max_k_tiles=max_k_tiles,
        allow_edge_k_tail_family=allow_edge_k_tail_family,
        allow_fp8_small_grid=allow_fp8_small_grid,
        allow_one_wave_tiles=allow_one_wave_tiles,
        one_wave_only=one_wave_only,
    )
    restrict_cluster_m_search(state, (1, 2))


def _aux_tma_edge_search_enabled(state: CuteTcgen05Config) -> bool:
    # The TMA aux producer's original admission: the validated Target8-style
    # double-edge + K-tail family with ``cluster_m=2``. The CLC-persistent
    # variants and edge-perf knobs remain pinned to this slice.
    constraints = state.cluster_m2_search_constraints
    return (
        state.exact_shape_aux_kernel_detected
        and state.exact_aux_tma_provable
        and constraints is not None
        and constraints.allow_edge_k_tail_family
    )


def _aux_tma_full_tile_search_enabled(state: CuteTcgen05Config) -> bool:
    # Cycle 46: also admit aux-TMA on full-tile cluster_m=2 problems
    # (T14/T20/T25/T28 residual_add family). The codegen-side gate at
    # ``cute_mma.py`` ``tcgen05_static_output_tiles`` already accepts
    # full-tile shapes; only the search-space gate was excluding them.
    # Edge-perf knobs (``_set_clc_aux_tma_edge_perf_knobs``) and
    # CLC-persistent variants stay pinned to ``_aux_tma_edge_search_enabled``
    # so this widening does not perturb the 5000³ T12 family.
    constraints = state.cluster_m2_search_constraints
    return (
        state.exact_shape_aux_kernel_detected
        and state.exact_aux_tma_provable
        and not state.matmul_has_leading_passthrough
        and constraints is not None
        and not constraints.allow_edge_k_tail_family
        and not constraints.one_wave_only
    )


def _aux_tma_search_enabled(state: CuteTcgen05Config) -> bool:
    # The TMA aux producer is admitted on either the edge+K-tail family or
    # the full-tile cluster_m=2 family. Exact-shape aux tensors use the
    # aux-TMA producer on both full and partial-output tiles; non-staged
    # aux operands remain on the direct guarded load path.
    return _aux_tma_edge_search_enabled(state) or _aux_tma_full_tile_search_enabled(
        state
    )


def _pdl_searched(state: CuteTcgen05Config) -> bool:
    """Whether this kernel's tcgen05 surface launches as a dependent by default.

    sm_90+ targets with the knob (``griddepcontrol`` exists there): the
    fragment defaults to True and the search keeps it there
    (``optional_fragments``).  Fission bundles are excluded: their stages
    carry the materialized-operand protocol.
    """
    from ..autotune.flash_surface import pdl_searched

    return pdl_searched(state.config_spec) and not state.fission_bundle


def split_k_search_choices(state: CuteTcgen05Config) -> tuple[int, ...]:
    """The ``tcgen05_split_k`` values the plain GEMM can take: splits no
    deeper than the K loop of the smallest K tile the search admits,
    for an unbatched static-shape GEMM (the pipeline reads the operands
    through TMA on the flat one-CTA path only)."""
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    if (
        not state.search_enabled
        # A tile body with an atomic, a second store or a branch would
        # run that effect once per split.
        or not state.split_k_plain_body
        or extents is None
        or fragments is None
        or state.matmul_leading_work_multiplier != 1
        or state.grouped_worklist_smem_facts is not None
        or any(value is None or value <= 0 for value in extents)
    ):
        return ()
    m, n, k = cast("tuple[int, int, int]", extents)
    if (
        state.device_sm_count > 0
        and -(-m // 128) * -(-n // 128) >= state.device_sm_count
    ):
        # Enough output tiles to fill the machine: splitting only adds
        # the partial exchange. Without an SM count (hosts without a
        # device) the knob stays searchable and so do its seeds.
        return ()
    k_tiles = -(-k // fragments[2].low)
    return tuple(
        split for split in TCGEN05_SPLIT_K_SEARCH_CHOICES if 1 < split <= k_tiles
    )


def prepare_normalization(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    repair._normalize_dense_aligned_n(state, config, fix_invalid=fix_invalid)
    repair._normalize_warp_mma_family(state, config, fix_invalid=fix_invalid)
    repair._normalize_batch_raster(state, config, fix_invalid=fix_invalid)
    repair._normalize_persistent_grid_cap(state, config, fix_invalid=fix_invalid)
    # An explicit physical schedule owns its projection.  Validate the
    # requested carrier before generic block normalization can repair it.
    repair._normalize_grouped_row_union(
        state,
        config,
        fix_invalid=fix_invalid,
        validate_schedule=physical_schedule(config) is not None,
    )
    repair._normalize_grouped_full_coverage(state, config, fix_invalid=fix_invalid)
    repair._normalize_grouped_ab_tile_k(state, config, fix_invalid=fix_invalid)
    repair._normalize_grouped_d_shift(state, config, fix_invalid=fix_invalid)
    repair._normalize_grouped_balanced_workers(state, config, fix_invalid=fix_invalid)
    repair._normalize_grouped_prefix_adaptive_n(state, config, fix_invalid=fix_invalid)
    repair._normalize_grouped_panel_adaptation(state, config, fix_invalid=fix_invalid)
    # A 2x2 cluster on an odd N tile count is repaired to the pair before
    # the fanout is judged, so the sample keeps its fanout and its pair
    # (judged again on the final tile in ``normalize_strategy``).
    repair._validate_cluster_n_pairs_n_tiles(state, config, fix_invalid=fix_invalid)
    repair._normalize_epilogue_fanout(state, config, fix_invalid=fix_invalid)
    repair._normalize_warp_register_reallocation(state, config, fix_invalid=fix_invalid)
    full_a_key = TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY
    if full_a_key in config:
        if config[full_a_key] is False:
            config.pop(full_a_key)
        elif config[full_a_key] is not True or not (
            state.search_enabled
            and (
                repair._grouped_grid_multiplier_supported(
                    {**config, "num_sm_multiplier": 2}
                )
                or (
                    state.grouped_device_worklist_supported
                    and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
                    == TCGEN05_GROUPED_MODE_WORKLIST_NM
                    and config.get("tcgen05_cluster_m", 1) == 2
                    and config.get("tcgen05_cluster_n", 1)
                    in (
                        (1, 2)
                        if config.get("tcgen05_grouped_worklist_orientation", "nm")
                        == "mn"
                        else (1,)
                    )
                    and config.get("num_sm_multiplier", 1) == 1
                    and not repair._is_grouped_clc_config(config)
                    and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False)
                    is False
                    and config.get(TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY)
                    is None
                )
            )
            and config.get(
                TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
                TCGEN05_GROUPED_FULL_COVERAGE_OFF,
            )
            == TCGEN05_GROUPED_FULL_COVERAGE_OFF
            and physical_schedule(config) is None
        ):
            if fix_invalid:
                config.pop(full_a_key)
            else:
                raise InvalidConfig(
                    f"{full_a_key}=True requires an independent device N,M worklist"
                )
    grouped_mode = config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
    if grouped_mode in TCGEN05_GROUPED_MODES and not (
        repair._grouped_grid_multiplier_supported(config)
    ):
        if fix_invalid:
            config.pop("num_sm_multiplier", None)
        else:
            raise InvalidConfig(
                "tcgen05 grouped grid multipliers require an independent "
                "single-CTA N,M worklist without a static execution signature"
            )
    grouped_clc = repair._is_grouped_clc_config(config)
    if grouped_clc and not repair._is_grouped_runtime_direct_clc_config(config):
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
    grouped_clc = repair._is_grouped_clc_config(config)
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
        or source_m_tile not in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
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
    mma_m = config.get(TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY)
    if mma_m is not None and (
        type(mma_m) is not int
        or mma_m not in (64, 128, 256)
        or grouped_mode != TCGEN05_GROUPED_MODE_WORKLIST_NM
    ):
        if fix_invalid:
            config.pop(TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY)
        else:
            raise InvalidConfig(
                f"{TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY} requires "
                f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}="
                f"{TCGEN05_GROUPED_MODE_WORKLIST_NM!r} and a physical M "
                "dimension in (64, 128, 256)"
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
    device_mn_swizzle = (
        state.grouped_device_mn_swizzle_supported
        and config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
    )
    if (
        grouped_mode == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and type(l2_swizzle_size) is int
        and l2_swizzle_size > 1
        and config.get(runtime_direct_key) is not True
        and not device_mn_swizzle
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


def allow_ab_stages_three_search(
    state: CuteTcgen05Config,
    *,
    dtype_bytes: int,
    device: torch.device,
    output_dtype_bytes: int | None = None,
    output_stores: int = 1,
) -> None:
    assert dtype_bytes > 0, "dtype_bytes must be positive"
    if facts._matmul_block_indices(state) is None:
        state.ab_stages_three_search_constraints = None
        return
    budget_bytes = target_facts.per_cta_ab_smem_budget_bytes(device)
    if budget_bytes <= 0:
        state.ab_stages_three_search_constraints = None
        return
    state.ab_stages_three_search_constraints = Tcgen05AbStagesThreeSearchConstraints(
        dtype_bytes=dtype_bytes,
        per_cta_smem_budget_bytes=budget_bytes,
        output_dtype_bytes=(
            output_dtype_bytes if output_dtype_bytes in (2, 4) else None
        ),
        output_stores=max(1, output_stores),
    )


def allow_deep_direct_entry_validation(
    state: CuteTcgen05Config, *, device: torch.device
) -> None:
    state.deep_direct_entry_validation_enabled = (
        facts._direct_entry_k_block_index(state) is not None
        and target_facts.per_cta_ab_smem_budget_bytes(device) > 0
    )


def epilogue_group_search_enabled(state: CuteTcgen05Config) -> bool:
    """Whether the epilogue subtile and warp-group knobs are searched.

    Both act on a heavy epilogue -- several output stores or an
    exact-shape auxiliary read -- whose per-subtile chain a one-tile
    per-CTA grid leaves exposed; a plain store's search keeps its
    surface (an explicit config is validated everywhere).
    """
    facts = state.rowvec_aux_facts
    return state.search_enabled and (
        state.exact_shape_aux_kernel_detected
        or (facts is not None and facts.output_count >= 2)
    )


def _clc_persistence_search_enabled(state: CuteTcgen05Config) -> bool:
    """CLC (hardware tile-scheduler) persistence search gate, sm100+ only.

    Two validated families: the aux-TMA edge+K-tail family (its original
    scope) and the plain full-tile cluster_m=2 family (see
    ``_plain_clc_persistence_search_enabled``).
    """
    capability = state.config_spec.target_device_capability
    if capability is None:
        return False
    if capability[0] < 10 or "flat" not in state.allowed_pid_types:
        return False
    return _aux_tma_edge_search_enabled(state) or _plain_clc_persistence_search_enabled(
        state
    )


def _plain_clc_persistence_search_enabled(state: CuteTcgen05Config) -> bool:
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
    constraints = state.cluster_m2_search_constraints
    return (
        not state.aux_kernel_detected
        and not state.matmul_has_leading_passthrough
        and constraints is not None
        and not constraints.one_wave_only
    )


def _offers_pdl_fragment(state: CuteTcgen05Config) -> bool:
    """Whether ``optional_fragments`` carries the ``cute_pdl`` knob."""
    return (
        state.search_enabled
        and state.config_spec.supports_config_key("cute_pdl")
        and not state.fission_bundle
    )


def implicit_default_keys_to_preserve(
    state: CuteTcgen05Config, config: dict[str, object]
) -> set[str]:
    preserve_keys: set[str] = set()
    if _offers_pdl_fragment(state):
        # The dependent-launch knob is one of this kernel's optional
        # tcgen05 fields: a normalized config carries its default like the
        # other optional knobs, so a flat transfer round-trips exactly.
        preserve_keys.add("cute_pdl")
    if not repair._is_clc_aux_tma_config(state, config):
        return preserve_keys
    preserve_keys.add("l2_groupings")
    if seeds._clc_aux_tma_matmul_k_range_index(state) is not None:
        preserve_keys.update(
            {
                "range_flattens",
                "range_multi_buffers",
                "range_warp_specializes",
            }
        )
    return preserve_keys


def prepare_override_normalization(
    state: CuteTcgen05Config,
    config: dict[str, object],
    overrides: Mapping[str, object],
) -> None:
    if "block_sizes" in overrides:
        if overrides.get(TILE_N_KEY, 0) != 0:
            raise InvalidConfig(
                "fixed block_sizes overrides cannot select a different physical N tile"
            )
        config[TILE_N_KEY] = 0
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
    if pid_type not in state.allowed_pid_types:
        return
    derived = derive_persistence_model_from_pid_type(pid_type)
    compatible = persistence_model is derived or (
        persistence_model is Tcgen05PersistenceModel.CLC_PERSISTENT
        and derived is Tcgen05PersistenceModel.STATIC_PERSISTENT
    )
    if not compatible:
        config.pop(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, None)


def persistence_model_default_from_config(
    state: CuteTcgen05Config,
    config: dict[str, object],
) -> Tcgen05PersistenceModel:
    """Derive default persistence from pid_type."""
    pid_type = config.get("pid_type", state.allowed_pid_types[0])
    if pid_type not in state.allowed_pid_types:
        pid_type = state.allowed_pid_types[0]
    return derive_persistence_model_from_pid_type(pid_type)


def flatten_missing_field_default(
    state: CuteTcgen05Config,
    key: str,
    config: dict[str, object],
) -> tuple[bool, object]:
    if key in (TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, TILE_N_KEY):
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
    repair.fix_search_config(state, projected_config)
    return True, persistence_model_default_from_config(state, projected_config).value


def restrict_num_epi_warps_search(
    state: CuteTcgen05Config, choices: tuple[int, ...]
) -> None:
    assert choices, "tcgen05_num_epi_warps search must allow at least one value"
    state.num_epi_warps_search_choices = choices


def restrict_num_epi_warps_validation(
    state: CuteTcgen05Config, choices: tuple[int, ...]
) -> None:
    assert choices, "tcgen05_num_epi_warps validation must allow at least one value"
    state.num_epi_warps_validation_choices = choices


def narrow_autotune_to_validated_configs(
    state: CuteTcgen05Config,
    *,
    allow_persistent_pid_types: bool = False,
    enable_cluster_m2_search: bool = False,
    cluster_m2_static_k: int | None = None,
    allow_cluster_m2_edge_k_tail_family: bool = False,
    allow_cluster_m2_fp8_small_grid: bool = False,
    allow_cluster_m2_one_wave_tiles: bool = False,
    cluster_m2_one_wave_only: bool = False,
    ab_stages_three_dtype_bytes: int | None = None,
    ab_stages_three_device: torch.device | None = None,
    ab_stages_three_output_dtype_bytes: int | None = None,
    ab_stages_three_output_stores: int = 1,
    reason: str | None = None,
) -> None:
    # Keep the default tcgen05 surface to combinations with runtime
    # coverage. Some unvalidated combinations fail loudly at CuTe
    # construction/launch, while diagnostic pipeline modes can compile and
    # intentionally produce wrong output.
    if allow_cluster_m2_edge_k_tail_family:
        assert enable_cluster_m2_search, (
            "cluster_m=2 edge/K-tail admission requires cluster_m=2 search"
        )
    if allow_cluster_m2_fp8_small_grid:
        assert enable_cluster_m2_search, (
            "cluster_m=2 fp8 small-grid admission requires cluster_m=2 search"
        )
    cluster_m2_static_k_int: int | None = None
    if enable_cluster_m2_search:
        assert allow_persistent_pid_types or allow_cluster_m2_edge_k_tail_family, (
            "cluster_m=2 search requires persistent pid types or the "
            "validated output-edge + K-tail admission"
        )
        if cluster_m2_static_k is None:
            raise AssertionError("cluster_m=2 search requires a static K extent")
        cluster_m2_static_k_int = cluster_m2_static_k
    if allow_cluster_m2_edge_k_tail_family and (
        TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types
    ):
        state.allowed_pid_types = (
            *state.allowed_pid_types,
            cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
        )
    if not allow_persistent_pid_types:
        state.config_spec.disallow_pid_type(
            "persistent_blocked",
            reason=_TWO_CTA_PERSISTENT_PID_VETO,
        )
        if not allow_cluster_m2_edge_k_tail_family:
            state.config_spec.disallow_pid_type(
                "persistent_interleaved",
                reason=_TWO_CTA_PERSISTENT_PID_VETO,
            )
    if enable_cluster_m2_search:
        assert cluster_m2_static_k_int is not None
        allow_cluster_m2_search(
            state,
            static_k=cluster_m2_static_k_int,
            allow_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
            allow_fp8_small_grid=allow_cluster_m2_fp8_small_grid,
            allow_one_wave_tiles=allow_cluster_m2_one_wave_tiles,
            one_wave_only=cluster_m2_one_wave_only,
        )
    else:
        restrict_cluster_m_search(state, (1,))
    restrict_num_epi_warps_search(state, TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES)
    restrict_num_epi_warps_validation(state, TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES)
    if ab_stages_three_dtype_bytes is not None:
        assert ab_stages_three_device is not None, (
            "ab_stages_three_dtype_bytes requires ab_stages_three_device "
            "so the SMEM-budget gate consults the operand's device, not "
            "the host's current CUDA device"
        )
        allow_deep_direct_entry_validation(state, device=ab_stages_three_device)
        allow_ab_stages_three_search(
            state,
            dtype_bytes=ab_stages_three_dtype_bytes,
            device=ab_stages_three_device,
            output_dtype_bytes=ab_stages_three_output_dtype_bytes,
            output_stores=ab_stages_three_output_stores,
        )
    state.config_spec.record_restriction(
        "tcgen05 search narrowed to validated configs", reason
    )


def optional_fragments(
    state: CuteTcgen05Config, *, for_search: bool = False
) -> dict[str, ConfigSpecFragment]:
    if for_search and state.cluster_m_search_choices is not None:
        cluster_m_choices = state.cluster_m_search_choices
    else:
        cluster_m_choices = (1, 2)
    paired_search = paired_pipeline_search_enabled(state) or bool(
        state.materialized_matmul_block_ids
    )
    # Clustered and two-CTA tcgen05 codegen run under a persistent launch
    # (``tcgen05_cluster_m > 1`` is refused for ``pid_type="flat"``).  On
    # a flat-only surface -- ordered barrier phases, a fission plan whose
    # shapes vetoed the persistent walks -- those coordinates, and the
    # per-region pipeline knobs whose schedule requires them, would only
    # be sampled to fail, so the search surface withholds them.
    flat_only = for_search and all(
        pid_type == "flat" for pid_type in state.allowed_pid_types
    )
    # An ordered barrier chain is flat-only, but ``tcgen05_region_pid_types``
    # runs its GEMM launches persistent, so the two-CTA coordinates and
    # the per-region ring depths are live there.
    region_chain = region_pid_type_choices(state) is not None
    two_cta_regions = (not flat_only or region_chain) and not (
        for_search and state.config_spec.has_explicit_pdl
    )
    if (
        paired_search
        or state.full_tile_added_pairing
        or state.n_edge_cluster_schedule_facts is not None
    ) and two_cta_regions:
        cluster_m_choices = tuple(dict.fromkeys((*cluster_m_choices, 2)))
    # cluster_n=2 (the Quack-canonical 4-CTA cluster: B multicast on top
    # of the 2-CTA A multicast) is searchable on every cluster_m=2
    # family — it wins big on B-heavy problems (fp16 4096x4096x32768:
    # 911 vs 824 TFLOP/s; cuBLAS's nvjet picks 2x2 clusters for the same
    # shape) and, with the generalized 4-CTA CLC broadcast, closes the
    # edge-family gap too (bf16 5000^3: 907 vs 852 TFLOP/s, matching
    # quack's dynamic-persistent 2x2 within 1%).
    cluster_n_searchable = (
        state.cluster_m2_search_constraints is not None
        or state.n_edge_cluster_schedule_facts is not None
    )
    cluster_n_choices: tuple[int, ...] = (
        (1, 2) if not for_search or cluster_n_searchable else (1,)
    )
    if for_search and state.config_spec.has_explicit_pdl:
        # Pin unrestricted defaults and N-edge cluster facts as well as
        # heuristic-restricted choices: explicit markers own one launch.
        cluster_m_choices = cluster_n_choices = (1,)
    if for_search:
        # Pinned (``TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES``) unless the
        # ``restrict_num_epi_warps_search`` seam lifts the gate.
        search_choices = state.num_epi_warps_search_choices
        num_epi_warps_fragment: ConfigSpecFragment = EnumFragment(
            TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES
            if search_choices is None
            else search_choices
        )
    elif state.num_epi_warps_validation_choices is not None:
        num_epi_warps_fragment = EnumFragment(state.num_epi_warps_validation_choices)
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
        ab_stages_max = 6 if state.deep_direct_entry_validation_enabled else 3
        # Operands whose AB SMEM fits the per-CTA budget run the dtype's
        # deep AB pipeline (12 stages for fp8 and 16-bit operands): small
        # tiles such as the 256x64x64 two-CTA GEMM fit 9 stages, which is
        # what hides the TMA latency of a 16-step K loop on one-wave
        # shapes (fp16 1024^3: cuBLAS's 64x10 ring).
        # ``_validate_direct_entry_ab_stage_envelope`` clamps the depth
        # to the actual per-CTA SMEM budget for the chosen block sizes.
        constraints = state.ab_stages_three_search_constraints
        if constraints is not None:
            ab_stages_max = max(
                ab_stages_max,
                smem._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes),
            )
    elif state.ab_stages_three_search_constraints is not None:
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
        constraints = state.ab_stages_three_search_constraints
        assert constraints is not None
        ab_stages_max = smem._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes)
    else:
        ab_stages_max = 2
    # Physical N widths with a proved aligned-N seed (computed once).
    aligned_widths = facts._dense_aligned_n_widths(state)
    if state.pipeline_smem_facts is not None and (not for_search or paired_search):
        ab_stages_max = max(ab_stages_max, TCGEN05_SMEM_AWARE_MAX_AB_STAGES)
    if aligned_widths:
        ab_stages_max = max(ab_stages_max, TCGEN05_SMEM_AWARE_MAX_AB_STAGES)
    if state.one_cta_pipeline_facts is not None and (
        not for_search or facts._preferred_persistent_pid_type(state) is not None
    ):
        ab_stages_max = max(ab_stages_max, TCGEN05_SMEM_AWARE_MAX_AB_STAGES)
    if region_chain:
        # The flat GEMM launches of an ordered chain search the depth the
        # standalone surface searches; each launch's tile is clamped to
        # the ring that surface admits for it, tightened by the one-CTA
        # model (``_validate_direct_entry_ab_stage_envelope``).
        ab_stages_max = max(ab_stages_max, smem._flat_region_ab_stage_cap(state))
    l2_swizzle_choices = (
        TCGEN05_L2_SWIZZLE_SEARCH_SIZES
        if for_search
        else TCGEN05_LEGAL_L2_SWIZZLE_SIZES
    )
    fragments: dict[str, ConfigSpecFragment] = {
        TCGEN05_CTA_GROUP_CONFIG_KEY: EnumFragment(
            TCGEN05_CTA_GROUP_CHOICES
            if not for_search
            or (paired_search and two_cta_regions)
            # The ragged edge family's whole 128-row pairs run the 2-CTA
            # MMA only when asked for explicitly (``auto`` keeps bf16
            # bm=128 + cluster_m=2 on the CtaGroup.ONE bridge, which the
            # edge epilogues refuse); let the search ask
            # (``plain_edge_pair128_admitted``).
            or (two_cta_regions and repair.plain_edge_pair128_admitted(state))
            else ("auto",)
        ),
        "tcgen05_cluster_m": EnumFragment(cluster_m_choices),
        "tcgen05_cluster_n": EnumFragment(cluster_n_choices),
        "tcgen05_ab_stages": IntegerFragment(1, ab_stages_max, 2),
        "tcgen05_acc_stages": IntegerFragment(1, 2, 2),
        # One stage per epilogue warp group is enough with two or more
        # groups (``_normalize_epilogue_groups``: a lone group keeps the
        # two-stage ring); see ``TCGEN05_C_STAGES_SEARCH_CHOICES``.
        "tcgen05_c_stages": EnumFragment(
            TCGEN05_C_STAGES_CHOICES, search_choices=TCGEN05_C_STAGES_SEARCH_CHOICES
        ),
        "tcgen05_num_epi_warps": num_epi_warps_fragment,
        TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: EnumFragment(l2_swizzle_choices),
    }
    split_k_choices = split_k_search_choices(state)
    if not for_search or split_k_choices:
        # Flat one-CTA kernels run each output tile's K loop on this
        # many CTAs and reduce the partials in the last arriver's TMEM
        # (``_normalize_split_k`` keeps the knob at 1 off the
        # flat one-CTA family). Searched only where the output tiles
        # cannot fill the machine; otherwise the knob stays off the
        # surface (explicit configs may still set it).
        fragments[TCGEN05_SPLIT_K_CONFIG_KEY] = EnumFragment(
            TCGEN05_SPLIT_K_SEARCH_CHOICES
        )
    if not for_search or state.grouped_ab_k_fusion_supported:
        fragments[TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY] = EnumFragment((None, 128))
    if not for_search or state.grouped_d_shift_supported:
        fragments[TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY] = BooleanFragment()
    if not for_search or state.grouped_balanced_workers_supported:
        fragments[TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY] = BooleanFragment()
    if not for_search or state.grouped_prefix_adaptive_n_search_enabled:
        fragments[TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY] = (
            EnumFragment((False, True), search_choices=(False,))
            if for_search and state.config_spec.has_explicit_pdl
            else BooleanFragment()
        )
    if not for_search or state.grouped_device_mn_swizzle_supported:
        fragments[GROUPED_PANEL_ALTERNATE_KEY] = EnumFragment(
            (None, *TCGEN05_LEGAL_L2_SWIZZLE_SIZES),
            search_choices=(None, 1, 2, 4, 8),
        )
        fragments[GROUPED_PANEL_RATIO_KEY] = EnumFragment(
            GROUPED_PANEL_RATIO_CHOICES,
            search_choices=(None, 1, 2, 4),
        )
    if state.matmul_has_leading_passthrough:
        # The persistent walk over a batched grid (see
        # ``TCGEN05_BATCH_RASTER_CONFIG_KEY``); plain grids raster through
        # ``loop_orders`` and do not carry the knob.
        fragments[TCGEN05_BATCH_RASTER_CONFIG_KEY] = EnumFragment(
            TCGEN05_BATCH_RASTER_CHOICES
        )
    if not state.matmul_has_leading_passthrough and (
        not for_search or 2 in cluster_n_choices
    ):
        # How a plain static-persistent ``cluster_n=2`` grid caps its
        # launch (see ``TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY``); batched
        # grids always use the resident-cluster cap and ``cluster_n=1``
        # grids are unaffected, so neither carries the knob.
        fragments[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] = EnumFragment(
            TCGEN05_PERSISTENT_GRID_CAP_CHOICES
        )
    if not for_search or state.grouped_full_coverage_eligible:
        fragments[TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY] = EnumFragment(
            TCGEN05_GROUPED_FULL_COVERAGE_MODES
        )
    if (
        not for_search
        or state.grouped_row_union_eligible
        or state.grouped_row_union_cluster4_eligible
        or state.grouped_row_union_paired_clc_eligible
    ):
        fragments[GROUPED_ROW_UNION_KEY] = BooleanFragment()
    if (
        not for_search
        or state.grouped_row_union_cluster4_eligible
        or state.grouped_row_union_paired_clc_eligible
    ):
        choices = [LEGACY_SCHEDULE]
        if not for_search or state.grouped_row_union_cluster4_eligible:
            choices.append(TRANSPOSED_SCHEDULE)
        if not for_search or state.grouped_row_union_paired_clc_eligible:
            choices.append(PAIRED_CLC_SCHEDULE)
        fragments[GROUPED_ROW_UNION_SCHEDULE_KEY] = EnumFragment(tuple(choices))
        # The physical ring depth of the selected profile (``None`` = its
        # deepest fitting ring); values a profile cannot take are repaired
        # to ``None`` by normalization.
        fragments[GROUPED_ROW_UNION_AB_STAGES_KEY] = EnumFragment(
            GROUPED_ROW_UNION_AB_STAGES_CHOICES
        )
    if not for_search or state.grouped_row_union_cluster4_eligible:
        # Subtiles per TMA-store epilogue round (``tcgen05_epilogue_rounds``);
        # searched where the transposed row-union profile is reachable, an
        # explicit value is validated on every tcgen05 TMA-store epilogue
        # (``epi_round_subtiles_refusal``).
        fragments[TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY] = EnumFragment(
            TCGEN05_EPI_ROUND_SUBTILES_CHOICES
        )
    if not for_search or state.grouped_row_union_paired_clc_eligible:
        fragments[STARTUP_PREFILL_KEY] = BooleanFragment()
    if (
        not for_search
        or state.grouped_row_union_paired_clc_eligible
        or _clc_persistence_search_enabled(state)
    ):
        # How the scheduler warp paces its CLC queries (see
        # ``TCGEN05_CLC_QUERY_PACING_CONFIG_KEY``); normalization drops the
        # knob from configs without a CLC scheduler.
        fragments[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY] = EnumFragment(
            TCGEN05_CLC_QUERY_PACINGS
        )
    if not for_search or (
        state.grouped_row_union_eligible
        and state.grouped_row_union_multi_resident_supported
    ):
        fragments[GROUPED_RESIDENT_CTAS_KEY] = EnumFragment((1, 2))
    if not for_search or state.epilogue_fanout_search_enabled:
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
        state.epilogue_fanout_search_enabled
        or state.grouped_full_coverage_eligible
        or state.grouped_row_union_eligible
        or state.grouped_row_union_cluster4_eligible
        or state.grouped_row_union_paired_clc_eligible
    )
    fragments[TCGEN05_C_STORE_MODE_CONFIG_KEY] = EnumFragment(
        (TCGEN05_C_STORE_MODE_NORMAL, TCGEN05_C_STORE_MODE_DIRECT)
        if for_search
        else TCGEN05_C_STORE_MODES,
        search_choices=(TCGEN05_C_STORE_MODE_NORMAL,)
        if for_search and direct_store_inert
        else None,
    )
    # Validation admits every aligned width; search admits the proved ones.
    if not for_search:
        fragments[TILE_N_KEY] = EnumFragment((0, *ALIGNED_N_CHOICES))
    elif aligned_widths:
        fragments[TILE_N_KEY] = EnumFragment((0, *aligned_widths))
    # The register-MMA GEMM family and its warp count are searched only
    # where the plan admitted the family (``warp_mma_admitted``); every
    # tcgen05-enabled kernel validates the keys so an explicit request on
    # a non-admitted problem is refused with the reason.
    if _offers_pdl_fragment(state):
        # Programmatic dependent launch of the tcgen05 kernel
        # (``cute_mma``'s ``tcgen05_pdl_launch``) and of the
        # register-MMA family: the next kernel in the stream launches
        # and runs its prologue while this one drains.
        # ``griddepcontrol`` needs sm_90; the CLC and
        # materialized-operand kernels place their own waits and leave
        # the knob inert, and a fission bundle's stages carry their own
        # dependent-launch protocol (the knob is dropped from the stage
        # that releases its dependents at entry), so it is not offered
        # there.  Normalization fills the default like every other
        # optional knob, so a flat transfer carries the field.  Only the
        # tcgen05 search surface owns the knob here: every other CuTe
        # kernel keeps the generic ``cute_pdl`` boolean of its own
        # branch (``ConfigSpec._flat_fields``), untouched.
        #
        # On sm_90+ the knob defaults to True and the search does not
        # flip it.  A tcgen05 kernel releases its dependents only after
        # its MMA warp's last MMA, when every CTA of the grid is resident,
        # and waits late (the TMA-load warp ahead of its first load, the
        # epilogue warps ahead of their first global read), so the knob
        # costs nothing in isolation and gains the dependent's prologue
        # and operand prefetch (measured 0.8-0.9 us per tcgen05 edge of
        # the squeeze_and_excitation_net backward chain, 47.07 -> 45.02 us
        # on its one-call graph).  The autotuner cannot rank it: its
        # per-call-flushed regime times ``[L2 flush memset; kernel]``
        # graphs, where a memset predecessor gives the kernel a full
        # dependency whatever its launch attribute says, so the two values
        # read alike and the final pick was a coin flip (campaign winners
        # carried True 5x, False 7x).  A knob the search cannot see and
        # whose chain effect is one-signed is a default, not a sample.
        # Both values stay valid: a pinned ``cute_pdl=False`` (older
        # records) still generates the knob-off code.
        pdl_searched = _pdl_searched(state)
        fragments["cute_pdl"] = EnumFragment(
            (True, False) if pdl_searched else (False, True),
            search_choices=(True,) if pdl_searched else (False,),
        )
    if not for_search or state.warp_mma_admitted:
        # Validation accepts both spellings (normalization refuses the
        # tcgen05 family with the reason where it is not admitted); the
        # search draws only from the admitted families.
        families = (
            MATMUL_FAMILIES
            if not for_search or state.tcgen05_family_admitted
            else (MATMUL_FAMILY_WARP_MMA,)
        )
        fragments[WARP_MMA_FAMILY_KEY] = EnumFragment(families)
        fragments[WARP_MMA_WARPS_KEY] = EnumFragment(WARP_MMA_WARP_CHOICES)
    region_launches = bool(state.materialized_matmul_block_ids) and two_cta_regions
    if region_launches:
        fragments["tcgen05_region_ab_stages"] = ListOf(
            IntegerFragment(0, 16, 0), len(state.materialized_matmul_block_ids)
        )
        fragments["tcgen05_region_c_stages"] = ListOf(
            EnumFragment((0, 2, 4)), len(state.materialized_matmul_block_ids)
        )
    if region_chain:
        fragments[TCGEN05_REGION_PID_TYPES_CONFIG_KEY] = ListOf(
            EnumFragment(TCGEN05_REGION_PID_TYPE_CHOICES),
            len(state.materialized_matmul_block_ids),
        )
    # The materialized-operand PDL edge belongs to fission plans; an
    # ordered chain's launches already order through ``cute_pdl``.
    if (
        region_launches and not state.region_pid_type_search_enabled
    ) or state.materialized_operand_pdl_roots is not None:
        fragments["tcgen05_materialized_pdl"] = BooleanFragment()
    # Every aux kernel's surface carries the placement key, so a supplied
    # ``post_acc_wait`` survives the flat round trip of a coverage
    # transfer (the carriers of other coverage groups are normalized
    # configs and always hold it). ``pre_acc_wait`` is sampled only where
    # some config can render it: every store chain loads an auxiliary
    # operand, or the shared fanout epilogue (searched) merges an
    # aux-free chain with one that does. An aux kernel whose first output
    # is aux-free and has no fanout surface would only sample the
    # placement into a repair, so its coordinate collapses to the
    # default rather than leaving the surface.
    if state.aux_kernel_detected or not for_search:
        pre_acc_wait_sampled = (
            state.aux_loads_on_every_store
            or (
                state.epilogue_fanout_search_enabled
                and repair.pre_acc_wait_admissible_via_fanout(state)
            )
            or not for_search
        )
        fragments[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = EnumFragment(
            TCGEN05_AUX_LOAD_PLACEMENTS
            if pre_acc_wait_sampled
            else (TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT,)
        )
    # The whole-fragment register hoist of an exact-shape auxiliary tile
    # exists only on kernels with such an operand (validation and search
    # surface alike: other kernels' configs keep their keys); its
    # per-config legality (placement, store protocol, register budget) is
    # the strict normalizer's (``aux_fragment_hoist_admissible``).
    if state.exact_shape_aux_kernel_detected:
        fragments[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = BooleanFragment()
        # The register prefetch ring of the same fragment, by subtile
        # distance (``aux_prefetch_refusal``: budget per depth).
        fragments[TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY] = IntegerFragment(
            0, TCGEN05_AUX_PREFETCH_MAX_SUBTILES, 0
        )
    if not for_search or epilogue_group_search_enabled(state):
        # The epilogue subtile width and the epilogue warp groups of the
        # role-local TMA-store epilogue: searched on kernels whose
        # epilogue is heavy (two outputs or an exact-shape auxiliary
        # read), validated everywhere. Plain enum fields: the coverage
        # group ``cute.epi_warp_groups`` owns the groups coordinate.
        fragments[TCGEN05_EPI_SUBTILE_N_CONFIG_KEY] = EnumFragment(
            TCGEN05_EPI_SUBTILE_N_CHOICES
        )
        fragments[TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY] = EnumFragment(
            TCGEN05_EPI_WARP_GROUP_CHOICES
        )
    if not for_search:
        fragments["tcgen05_grouped_worklist_orientation"] = EnumFragment(("nm", "mn"))
        fragments["tcgen05_grouped_worklist_mma_n"] = EnumFragment(
            (None, 128, 192, 224, 256)
        )
        fragments[TCGEN05_GROUPED_MODE_CONFIG_KEY] = EnumFragment(TCGEN05_GROUPED_MODES)
        fragments[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = BooleanFragment()
        fragments[TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY] = EnumFragment(
            TCGEN05_GROUPED_WORKLIST_MMA_M_CHOICES
        )
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
    direct_entry_seed_eligible = seeds.full_tile_direct_entry_seed_eligible(state)
    if direct_entry_seed_eligible or (
        not for_search and facts._direct_entry_k_block_index(state) is not None
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


def _clear_target1_tvm_ffi_promotion_surface(config: dict[str, object]) -> None:
    for key in list(config):
        if key.startswith("tcgen05_") or key == "epilogue_subtile":
            config.pop(key, None)


def aux_load_mode_autotune_fragments(
    state: CuteTcgen05Config,
) -> dict[str, ConfigSpecFragment]:
    choices = [TCGEN05_AUX_LOAD_MODE_SIMT]
    if _aux_tma_search_enabled(state):
        choices.append(TCGEN05_AUX_LOAD_MODE_TMA)
    if epi_tma_search_enabled(state):
        choices.append(TCGEN05_AUX_LOAD_MODE_EPI_TMA)
    if len(choices) == 1:
        return {}
    return {TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: EnumFragment(tuple(choices))}


def epi_tma_search_enabled(state: CuteTcgen05Config) -> bool:
    """Whether ``tcgen05_aux_load_mode="epi_tma"`` is on the search surface.

    The epilogue-issued aux ring needs an exact-shape rank-2 auxiliary
    operand of the output's dtype (``exact_shape_aux_kernel_detected``)
    on a rank-2 matmul; every per-config condition (static full tiles,
    the standard TMA-store epilogue, the SMEM budget) is judged by
    ``_normalize_epi_tma`` on the sample.
    """
    return (
        state.search_enabled
        and state.exact_shape_aux_kernel_detected
        and state.exact_aux_tma_provable
        and not state.matmul_has_leading_passthrough
    )


def aux_stages_autotune_fragments(
    state: CuteTcgen05Config,
) -> dict[str, ConfigSpecFragment]:
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
    if not _aux_tma_edge_search_enabled(state):
        return {}
    return {
        TCGEN05_AUX_STAGES_CONFIG_KEY: EnumFragment(TCGEN05_AUX_STAGE_COUNT_CHOICES)
    }


def consumer_regs_autotune_fragments(
    state: CuteTcgen05Config,
) -> dict[str, ConfigSpecFragment]:
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
        state.config_spec.compiler_seed_configs,
        TCGEN05_CONSUMER_REGS_CONFIG_KEY,
        int,
        lambda value: value in TCGEN05_CONSUMER_REGS_CHOICES,
    )
    if not _aux_tma_edge_search_enabled(state):
        if not any(choice != TCGEN05_CONSUMER_REGS_DEFAULT for choice in seed_choices):
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
        TCGEN05_CONSUMER_REGS_CONFIG_KEY: EnumFragment(TCGEN05_CONSUMER_REGS_CHOICES)
    }


def persistence_model_autotune_fragments(
    state: CuteTcgen05Config,
) -> dict[str, ConfigSpecFragment]:
    pid_default_models = tuple(
        dict.fromkeys(
            derive_persistence_model_from_pid_type(pid_type).value
            for pid_type in state.allowed_pid_types
        )
    )
    seed_pid_types = _compiler_seed_values(
        state.config_spec.compiler_seed_configs,
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
        return persistence_model_default_from_config(state, seed.config).value

    seed_override_models = _compiler_seed_values(
        state.config_spec.compiler_seed_configs,
        TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
        str,
        lambda model: model in {item.value for item in Tcgen05PersistenceModel},
        exact_type=False,
        is_valid_for_seed=lambda seed, model: model != seed_default_model(seed),
    )
    if not _clc_persistence_search_enabled(state):
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


def strategy_autotune_fragments(
    state: CuteTcgen05Config,
) -> dict[str, ConfigSpecFragment]:
    # Aux kernels are the only current trigger for scheduler/c_input warp
    # search. The surface is derived from aux_kernel_detected so repeated
    # detection or repeated fragment construction stays idempotent.
    direct_entry_seed_eligible = seeds.full_tile_direct_entry_seed_eligible(state)
    if state.aux_kernel_detected:
        strategy_choices: tuple[str, ...] = (
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
        )
        scheduler_warps_choices: tuple[int, ...] = (0, 1)
        c_input_warps_choices: tuple[int, ...] = (0, 1)
    elif _plain_clc_persistence_search_enabled(state):
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
        state.config_spec.compiler_seed_configs,
        TCGEN05_STRATEGY_CONFIG_KEY,
        str,
        lambda strategy: strategy in {item.value for item in Tcgen05Strategy},
        exact_type=False,
    )
    scheduler_seed_choices = _compiler_seed_values(
        state.config_spec.compiler_seed_configs,
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


def strategy_validation_fragments(
    state: CuteTcgen05Config,
) -> dict[str, ConfigSpecFragment]:
    fragments = strategy_autotune_fragments(state)
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


def _tcgen05_knob_set(
    state: CuteTcgen05Config, config: dict[str, object], key: str, *, fix_invalid: bool
) -> bool:
    """Whether ``key`` is set on a tcgen05-enabled surface.

    Off a tcgen05 surface a set knob is dropped (``fix_invalid``) or refused.
    """
    if key not in config:
        return False
    if not state.search_enabled:
        repair._reject_config_key(
            config,
            key,
            f"{key} is only supported for tcgen05-enabled CuTe matmul kernels",
            fix_invalid=fix_invalid,
        )
        return False
    return True


def flat_fields(
    state: CuteTcgen05Config,
) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
    fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
        "l2_groupings": state.config_spec.l2_groupings,
    }
    if (
        state.config_spec.supports_config_key("loop_orders")
        and len(state.config_spec.loop_orders) > 0
    ):
        fields["loop_orders"] = state.config_spec.loop_orders
    fields.update(optional_fragments(state, for_search=True))
    if state.config_spec.cute_split_k_workspace_available:
        fields.update(
            workspace_search_fields(state.config_spec.cute_split_k_partition_choices)
        )
    seeds = state.config_spec.compiler_seed_configs
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
                    lambda value: 0 <= value <= TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX,
                ),
                search_choices=choices,
            )
        )
    if modes:
        fields[TCGEN05_GROUPED_MODE_CONFIG_KEY] = _enum_fragment_with_seed_values(
            (None,),
            modes,
            search_choices=(None, TCGEN05_GROUPED_MODE_WORKLIST_NM)
            if state.grouped_source128_search_enabled
            else (None,),
        )
    if (
        TCGEN05_GROUPED_MODE_WORKLIST_NM in modes
        or state.grouped_row_union_cluster4_eligible
    ):
        # Searched on the worklist N,M mode (its static register protocol)
        # and where the transposed row-union profile is reachable
        # (``_row_union_register_reallocation_optional``).
        fields[TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY] = EnumFragment(
            (True, False)
        )
    if TCGEN05_GROUPED_MODE_WORKLIST_NM in modes:
        fields[TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY] = EnumFragment(
            (False, True)
        )
        mn_orientations = _compiler_seed_values(
            seeds,
            "tcgen05_grouped_worklist_orientation",
            str,
            lambda value: value == "mn",
        )
        if mn_orientations:
            fields["tcgen05_grouped_worklist_orientation"] = (
                _enum_fragment_with_seed_values(
                    ("nm",), mn_orientations, search_choices=("nm", "mn")
                )
            )
            widths = _compiler_seed_values(
                seeds,
                "tcgen05_grouped_worklist_mma_n",
                int,
                lambda value: value in (128, 192, 224, 256),
            )
            fields["tcgen05_grouped_worklist_mma_n"] = _enum_fragment_with_seed_values(
                (None,), widths, search_choices=(None, *widths)
            )
            cluster_n = fields["tcgen05_cluster_n"]
            assert isinstance(cluster_n, EnumFragment)
            fields["tcgen05_cluster_n"] = _enum_fragment_with_seed_values(
                cluster_n.choices,
                _compiler_seed_values(
                    seeds, "tcgen05_cluster_n", int, lambda value: value in (1, 2)
                ),
                search_choices=(1, 2),
                original=cluster_n,
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
        lambda value: value in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES,
    )
    if source_m_tiles:
        fields[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = (
            _enum_fragment_with_seed_values(
                (None,),
                source_m_tiles,
                search_choices=(None, TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE)
                if state.grouped_source128_search_enabled
                else (None,),
            )
        )
    mma_m_values = _compiler_seed_values(
        seeds,
        TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY,
        int,
        lambda value: value in (64, 128, 256),
    )
    if mma_m_values:
        fields[TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY] = (
            _enum_fragment_with_seed_values(
                (None,), mma_m_values, search_choices=(None,)
            )
        )
    fields.update(strategy_autotune_fragments(state))
    fields.update(aux_load_mode_autotune_fragments(state))
    fields.update(aux_stages_autotune_fragments(state))
    fields.update(consumer_regs_autotune_fragments(state))
    fields.update(persistence_model_autotune_fragments(state))
    if state.config_spec.supports_config_key("pid_type"):
        fields["pid_type"] = _enum_fragment_with_seed_values(
            state.allowed_pid_types,
            _compiler_seed_values(
                seeds,
                "pid_type",
                str,
                lambda _value: True,
                exact_type=False,
            ),
            search_choices=state.allowed_pid_types,
            search_only_if_widened=True,
        )
    if (
        state.config_spec.supports_config_key("indexing")
        and state.config_spec.indexing.length > 0
    ):
        fields["indexing"] = state.config_spec.indexing
    return fields
