"""Repair (``fix_invalid=True``) and strict validation of tcgen05 configs.

Module functions over the ``CuteTcgen05Config`` state object.
``fix_search_config`` is the ordered projection chain the autotuner's
population builder runs on every sampled flat config; ``normalize_pre_pid_type``
/ ``normalize_strategy`` are the strict validators the compile path runs
afterwards (with ``fix_invalid`` they repair in place).  The orchestrators are
explicit sequences: a step may depend on the tuple an earlier step produced,
so the order is the contract (``test_cute_tcgen05_normalize_fixed_point``
pins that the chain is a fixed point of the strict pass).  The validity
predicates and ``*_refusal`` readers next to them are the facts the
validators enforce; the ``InvalidConfig`` reason texts are pinned by tests.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ....exc import InvalidConfig
from ..cute_epilogue import fanout_pre_wait_stages_rowvec_aux
from ..cute_warp_mma_gemm import MATMUL_FAMILIES
from ..cute_warp_mma_gemm import MATMUL_FAMILY_TCGEN05
from ..cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
from ..cute_warp_mma_gemm import WARP_MMA_DEFAULT_WARPS
from ..cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY
from ..cute_warp_mma_gemm import WARP_MMA_MAX_BM
from ..cute_warp_mma_gemm import WARP_MMA_MAX_BN
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BM
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BN
from ..cute_warp_mma_gemm import WARP_MMA_WARP_CHOICES
from ..cute_warp_mma_gemm import WARP_MMA_WARPS_KEY
from ..cute_warp_mma_gemm import warp_mma_k_step
from ..cute_warp_mma_gemm import warp_mma_max_legal_warps
from ..cute_warp_mma_gemm import warp_mma_tile_reason
from ..cute_warp_mma_gemm import warp_mma_warp_layout
from ..dense_aligned_n import ALIGNED_N_CHOICES
from ..dense_aligned_n import TILE_N_KEY
from ..dense_aligned_n import carrier_n
from ..dense_aligned_n import dense_aligned_n_defaults
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..epilogue_fanout import FANOUT_MODES
from ..epilogue_fanout import schedule_supported as fanout_schedule_supported
from ..grouped_full_coverage import full_coverage_pipeline_supported
from ..grouped_full_coverage import full_coverage_smem_upper_bound
from ..grouped_panel_raster import GROUPED_PANEL_ALTERNATE_KEY
from ..grouped_panel_raster import GROUPED_PANEL_RATIO_CHOICES
from ..grouped_panel_raster import GROUPED_PANEL_RATIO_KEY
from ..grouped_row_union import AB_STAGES_KEY as GROUPED_ROW_UNION_AB_STAGES_KEY
from ..grouped_row_union import AB_STAGES_MIN
from ..grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from ..grouped_row_union import LEGACY_SCHEDULE
from ..grouped_row_union import PAIRED_CLC_SCHEDULE
from ..grouped_row_union import RESIDENT_CTAS_KEY as GROUPED_RESIDENT_CTAS_KEY
from ..grouped_row_union import SCHEDULE_KEY as GROUPED_ROW_UNION_SCHEDULE_KEY
from ..grouped_row_union import STARTUP_PREFILL_KEY
from ..grouped_row_union import ab_stages_supported as row_union_ab_stages_supported
from ..grouped_row_union import physical_schedule
from ..grouped_row_union import (
    round_subtiles_supported as row_union_round_subtiles_supported,
)
from ..grouped_row_union import schedule_by_name as row_union_schedule_by_name
from ..grouped_row_union import schedule_supported as row_union_schedule_supported
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..pipeline_smem import one_cta_pipeline_tile_modeled
from ..split_k_workspace_config import PARTITIONS_KEY
from ..split_k_workspace_config import STAGES_KEY
from ..split_k_workspace_config import WORKSPACE_KEY
from ..strategies import TCGEN05_BATCH_RASTER_CONFIG_KEY
from ..strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY
from ..strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from ..strategies import TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES
from ..strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_DEFAULT
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEYS
from ..strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
from ..strategies import Tcgen05LayoutStrategy
from ..strategies import Tcgen05PersistenceModel
from ..strategies import Tcgen05Strategy
from ..strategies import layout_overrides_from_config
from ..strategies import tcgen05_epilogue_subtile_m
from ..strategies import tcgen05_explicit_epilogue_tile_supported
from ..strategies import validate_tcgen05_strategy_invariants
from ..strategies import warp_spec_from_config
from ..tcgen05.knobs import acc_stages_knob
from ..tcgen05.knobs import epi_subtile_n_knob
from ..tcgen05.knobs import epi_warp_groups_knob
from ..tcgen05.knobs import epi_warps_knob
from ..tcgen05_constants import TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODES
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENTS
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_MAX_SUBTILES
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from ..tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_VALIDATION_CHOICES
from ..tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_TMA_ILLEGAL_TENSOR_REASON
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_BEFORE_STORE
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from ..tcgen05_constants import TCGEN05_C_STORE_MODES
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACING_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACING_TILE
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACINGS
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CONSUMER_REGS_DEFAULT
from ..tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_ROUND_SUBTILES_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUP_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_WARPS_PER_GROUP
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from ..tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_DYNAMIC_MODES
from ..tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_DENSE
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_OFF
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from ..tcgen05_constants import TCGEN05_GROUPED_MODES
from ..tcgen05_constants import TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_TWO_CTA_SOURCE_M_TILE_CHOICES
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
from ..tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from ..tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from ..tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from ..tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from ..tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_CHOICES
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_INHERIT
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES
from ..tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from ..tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from ..tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SCHED_STAGE_COUNTS
from ..tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
from ..tcgen05_constants import TCGEN05_TWO_CTA_PAIR_BLOCK_NS
from ..tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..tcgen05_constants import TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY
from ..tcgen05_constants import resolve_tcgen05_grouped_worklist_mma_profile
from ..tcgen05_constants import tcgen05_aux_fragment_bytes_per_thread
from ..tcgen05_constants import tcgen05_aux_prefetch_bytes_per_thread
from ..tcgen05_constants import tcgen05_direct_entry_stage_tuple_allowed
from ..tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from ..tcgen05_constants import tcgen05_role_launch_n_width
from ..tcgen05_constants import tcgen05_rowvec_stage_smem_bytes
from ..tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides
from ..tcgen05_schedule_facts import n_edge_cluster_geometry_supported
from ..tcgen05_schedule_facts import n_edge_cluster_schedule_defaults
from ..tcgen05_schedule_facts import n_edge_cluster_schedule_supported
from ..tcgen05_schedule_facts import native_tma_schedule_supported
from . import facts
from . import seeds
from . import smem
from . import surface

if TYPE_CHECKING:
    from ..epilogue_fanout import PairedFanoutPlan
    from .knobs import Tcgen05ClusterM2SearchConstraints


if TYPE_CHECKING:
    from . import CuteTcgen05Config


def _full_tile_schedule_config_supported(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    schedule_facts = state.full_tile_schedule_facts
    view = facts._matmul_config_view(state, config)
    if (
        schedule_facts is None
        or view is None
        or config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto") != "auto"
    ):
        return False
    blocks, mi, ni, ki = view
    tile = (blocks[mi], blocks[ni], blocks[ki])
    return native_tma_schedule_supported(
        schedule_facts,
        cast("tuple[int, int, int]", tile),
        cluster_m=cast("int", config.get("tcgen05_cluster_m", 1)),
        cluster_n=cast("int", config.get("tcgen05_cluster_n", 1)),
    )


def _fix_full_tile_schedule_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if (
        n_edge_cluster_schedule_config_supported(state, config)
        or seeds._paired_pipeline_parameters(state, config) is not None
    ):
        return
    if state.full_tile_schedule_facts is None:
        return
    added_pair = state.full_tile_added_pairing and config.get("tcgen05_cluster_m") == 2
    restored_pid = config.get("pid_type") in state.full_tile_restored_pid_types
    if not (added_pair or restored_pid):
        return
    if _full_tile_schedule_config_supported(state, config):
        if added_pair:
            config["pid_type"] = "persistent_interleaved"
        return
    # Keep unsupported tails and ownership on the pre-existing flat path.
    config["tcgen05_cluster_m"] = 1
    config["tcgen05_cluster_n"] = 1
    if restored_pid:
        config["pid_type"] = "flat"


def _drop_unproved_n_edge_cluster_n(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """A persistent N-edge sample (whole M and K tiles, a partial last N
    tile) left at ``tcgen05_cluster_n=2`` outside the proved N-multicast
    schedule takes one N cluster: codegen refuses it (``tcgen05 N-edge
    cluster_n=2 requires ...``), e.g. on a 16-deep K tile, or on a tile
    order that a projection running after the schedule's own
    (``n_edge_cluster_schedule_defaults``) grouped again. A post-pass over
    the projected sample judges the final config, whichever projection
    moved it."""
    pid_type = config.get("pid_type", "flat")
    extents = state.matmul_compile_time_static_extents
    view = facts._matmul_config_view(state, config)
    if (
        config.get("tcgen05_cluster_n") != 2
        or not (isinstance(pid_type, str) and pid_type.startswith("persistent"))
        or extents is None
        or view is None
    ):
        return
    blocks, mi, ni, ki = view
    tile = (blocks[mi], blocks[ni], blocks[ki])
    if not all(type(value) is int and value > 0 for value in (*tile, *extents)):
        return
    (m, n, k), (bm, bn, bk) = (
        cast("tuple[int, int, int]", extents),
        cast("tuple[int, int, int]", tile),
    )
    if (
        m % bm == 0
        and n % bn != 0
        and k % bk == 0
        and not n_edge_cluster_schedule_config_supported(state, config)
    ):
        config["tcgen05_cluster_n"] = 1


def n_edge_cluster_schedule_config_supported(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    schedule_facts = state.n_edge_cluster_schedule_facts
    view = facts._matmul_config_view(state, config)
    if schedule_facts is None or view is None:
        return False
    blocks, mi, ni, ki = view
    return n_edge_cluster_schedule_supported(
        schedule_facts,
        cast("tuple[int, int, int]", (blocks[mi], blocks[ni], blocks[ki])),
        config,
    )


def _fix_n_edge_cluster_schedule_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    schedule_facts = state.n_edge_cluster_schedule_facts
    view = facts._matmul_config_view(state, config)
    if schedule_facts is None or view is None:
        return
    blocks, mi, ni, ki = view
    if config.get("tcgen05_cluster_m") == config.get(
        "tcgen05_cluster_n"
    ) == 2 and n_edge_cluster_geometry_supported(
        schedule_facts,
        cast("tuple[int, int, int]", (blocks[mi], blocks[ni], blocks[ki])),
    ):
        config.update(n_edge_cluster_schedule_defaults())
        config["pid_type"] = "persistent_interleaved"
        config.pop("epilogue_subtile", None)
    elif config.get("pid_type") in state.n_edge_cluster_restored_pid_types:
        config["pid_type"] = "flat"
        config["tcgen05_cluster_m"] = config["tcgen05_cluster_n"] = 1


def _validate_cta_group_config(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) != "two":
        return
    if facts._dense_aligned_n_plan(state, config) is not None:
        _normalize_dense_aligned_n(state, config, fix_invalid=fix_invalid)
        return
    region_limits = smem._materialized_paired_stage_limits(state, config)
    if region_limits is not None:
        counts = config.get("tcgen05_region_ab_stages", [0] * len(region_limits))
        assert isinstance(counts, list)
        effective = [count or config.get("tcgen05_ab_stages", 2) for count in counts]
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
    limit = smem._paired_pipeline_stage_limit(state, config)
    if limit is not None and limit > 0:
        return
    if config.get("tcgen05_cluster_n", 1) == 2:
        pair_limit = smem._paired_pipeline_stage_limit(
            state, {**config, "tcgen05_cluster_n": 1}
        )
        if pair_limit is not None and pair_limit > 0:
            # The CTA pair is admitted; only the 2x2 multicast cluster
            # refuses (``_paired_pipeline_parameters``).  The search
            # keeps the sampled pair and drops the cluster rather than
            # leaving the family.
            if fix_invalid:
                config["tcgen05_cluster_n"] = 1
                return
            raise InvalidConfig(
                "tcgen05_cta_group='two' runs tcgen05_cluster_n=2 (the 2x2 "
                "multicast cluster) only with a row-major output and, on a "
                "batched GEMM, whole N-tile pairs on the plain raster "
                "(l2_groupings of 1 and no L2 swizzle)"
            )
    if fix_invalid:
        # A batched cluster_n=2 sample whose N tiles do not pair keeps
        # the CTA pair at cluster_n=1 (the search projection) before the
        # pair itself is given up.
        view = facts._matmul_config_view(state, config)
        if view is not None:
            _fix_batched_cluster_n2_search_config(state, config, view[0][view[2]])
            limit = smem._paired_pipeline_stage_limit(state, config)
            if limit is not None and limit > 0:
                return
        # A sampled pair on a tile the pair cannot run (an odd N or K
        # tile) joins the ragged edge family's 128-row pair where that
        # family is admitted, before the pair itself is given up; the
        # projection is kept only when the pair then has a stage budget
        # (a failed attempt must not leave the cluster behind).
        projected = copy.deepcopy(config)
        if _project_edge_pair128_search_config(state, projected):
            # The search fixer canonicalizes the pair's surface after
            # this validator runs; price the stage budget on that
            # surface (a sampled strategy or layout knob is not what the
            # pair will run with).
            _apply_paired_pipeline_surface(state, projected)
            for c_stages in dict.fromkeys((projected.get("tcgen05_c_stages"), 2)):
                if c_stages is None:
                    continue
                projected["tcgen05_c_stages"] = c_stages
                limit = smem._paired_pipeline_stage_limit(state, projected)
                if limit is not None and limit > 0:
                    config.clear()
                    config.update(projected)
                    return
        config[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
        return
    raise InvalidConfig(
        "tcgen05_cta_group='two' requires one TMA-legal half/bfloat16 MMA "
        "or proved native FP8 MMA with FP32 broadcasts, a single output, a static "
        "persistent two-CTA schedule, and supported shared-memory geometry"
    )


def _normalize_region_launches(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``tcgen05_region_pid_types`` and the clusters of an ordered chain.

    The knob names the pid type of each GEMM launch of a flat barrier
    chain and has one subject: the complete two-CTA region schedule
    (``tcgen05_cta_group="two"``, ``tcgen05_cluster_m=2``, proved
    per-region budgets) of its persistent launches.  All-``inherit`` is
    the implicit default.  A chain has no other clustered schedule --
    its flat launches refuse clusters at codegen -- so a cluster beside
    anything else is repaired to one CTA (search) or refused (explicit
    configs).
    """
    key = TCGEN05_REGION_PID_TYPES_CONFIG_KEY
    count = len(state.materialized_matmul_block_ids)
    values = config.get(key)
    well_formed = (
        isinstance(values, list)
        and len(values) == count
        and all(value in TCGEN05_REGION_PID_TYPE_CHOICES for value in values)
    )
    if values is None or (
        well_formed
        and all(
            value == TCGEN05_REGION_PID_TYPE_INHERIT
            for value in cast("list[object]", values)
        )
    ):
        config.pop(key, None)
    elif (
        well_formed
        and config.get("pid_type") == "flat"
        and surface.region_pid_type_choices(state) is not None
        and smem._materialized_paired_stage_limits(state, config) is not None
    ):
        pass
    elif fix_invalid:
        config.pop(key, None)
    else:
        raise InvalidConfig(
            f"{key}={values!r} requires one entry per GEMM launch of an "
            "ordered hl.barrier() chain, with proved per-region budgets "
            "under the two-CTA schedule (tcgen05_cta_group='two', "
            "tcgen05_cluster_m=2, tcgen05_cluster_n=1, the role-local "
            "static persistent pipeline) for its persistent launches"
        )
    if not state.region_pid_type_search_enabled:
        return
    if (
        config.get("tcgen05_cluster_m", 1) == 1
        and config.get("tcgen05_cluster_n", 1) == 1
    ) or smem._materialized_paired_stage_limits(state, config) is not None:
        return
    if fix_invalid:
        config["tcgen05_cluster_m"] = 1
        config["tcgen05_cluster_n"] = 1
        config[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
        config.pop(key, None)
        return
    raise InvalidConfig(
        "an ordered hl.barrier() chain runs clustered GEMM launches only as "
        f"the two-CTA region schedule ({key}=persistent_*, "
        "tcgen05_cta_group='two', tcgen05_cluster_m=2)"
    )


def _pin_warp_mma_inert_knobs(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """Pin every knob the register-MMA body ignores to its default so equal
    kernels share one autotune identity: the tcgen05 pipeline knobs, the
    rasterization knobs of the (replaced) Helion grid, the per-block thread /
    vector / lane-layout knobs, the indexing choice and the occupancy hint."""
    for key, fragment in surface.optional_fragments(state, for_search=True).items():
        if key in (WARP_MMA_FAMILY_KEY, WARP_MMA_WARPS_KEY, "cute_pdl"):
            # The family's own knobs, and the dependent launch it honors.
            continue
        if key in config:
            config[key] = fragment.default()
    for key in (
        # The grouped row-union carrier's coverage coordinates select the
        # tcgen05 lowering of the same dense union; the family lowers it
        # itself, so they are inert here.
        GROUPED_ROW_UNION_KEY,
        GROUPED_ROW_UNION_SCHEDULE_KEY,
        GROUPED_RESIDENT_CTAS_KEY,
        STARTUP_PREFILL_KEY,
    ):
        config.pop(key, None)
    spec = state.config_spec
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


def _normalize_warp_mma_family(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Keep the register-MMA family and the tcgen05 tiles mutually legal.

    ``cute_matmul_family="warp_mma"`` needs the admitted problem, a tile
    the family can run (16..64 rows, 8..64 columns, a K chunk of whole
    mma steps dividing K, a warp count tiling the block) and a unit batch
    block; its tcgen05-only knobs are inert and pinned to their defaults
    so equal kernels share one identity.  The tcgen05 family keeps its
    own block floors, which the admitted family's search widened.
    """
    if not state.search_enabled:
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
    if not state.tcgen05_family_admitted and family != MATMUL_FAMILY_WARP_MMA:
        # An in-register integer cast operand: TMA packets cannot carry
        # the cast to the tensor core, so the tcgen05 family (explicit or
        # the implied default) cannot run this kernel.
        if not fix_invalid:
            raise InvalidConfig(
                f"{WARP_MMA_FAMILY_KEY}={family or MATMUL_FAMILY_TCGEN05!r} "
                "cannot load this kernel's cast matmul operand (a bare "
                "``.to(bfloat16 | float16)`` of an int16 tile); the "
                f"register-MMA family {MATMUL_FAMILY_WARP_MMA!r} converts "
                "it in registers"
            )
        config[WARP_MMA_FAMILY_KEY] = MATMUL_FAMILY_WARP_MMA
        family = MATMUL_FAMILY_WARP_MMA
    view = facts._matmul_config_view(state, config)
    if family == MATMUL_FAMILY_WARP_MMA:
        problem = facts._warp_mma_problem(state)
        if problem is None or view is None:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{WARP_MMA_FAMILY_KEY}={family!r} is not admitted for this "
                    "kernel (the register-MMA family needs one latency-bound "
                    "dense GEMM with K- or M-contiguous A: static shapes, or "
                    "the proved grouped row union at a binding whose extents "
                    "the family runs)"
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
            values_are_ints = all(type(value) is int for value in (bm, bn, bk, warps))
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
                    raise InvalidConfig(f"{WARP_MMA_FAMILY_KEY}={family!r}: {reason}")
                k_step = warp_mma_k_step(dtype)
                new_bm = facts._warp_mma_largest_divisor(
                    m,
                    bm if type(bm) is int else WARP_MMA_MAX_BM,
                    low=WARP_MMA_MIN_BM,
                    high=WARP_MMA_MAX_BM,
                )
                new_bn = facts._warp_mma_largest_divisor(
                    n,
                    bn if type(bn) is int else WARP_MMA_MAX_BN,
                    low=WARP_MMA_MIN_BN,
                    high=WARP_MMA_MAX_BN,
                )
                new_bk = facts._warp_mma_largest_divisor(
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
                    requested = warps if type(warps) is int else WARP_MMA_DEFAULT_WARPS
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
                leading_index = facts._config_block_index(
                    state, state.matmul_leading_block_id
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
                if "flat" not in state.allowed_pid_types:
                    if not fix_invalid:
                        raise InvalidConfig(
                            f"{WARP_MMA_FAMILY_KEY}={family!r} requires the flat "
                            "program-id grid, which this kernel disallows"
                        )
                    config[WARP_MMA_FAMILY_KEY] = MATMUL_FAMILY_TCGEN05
                    family = MATMUL_FAMILY_TCGEN05
                else:
                    config["pid_type"] = "flat"
                    _pin_warp_mma_inert_knobs(state, config)
                    return
    # tcgen05 family (explicit or implied): the admitted register-MMA
    # search widened the M / N floors; clamp them back the way the block
    # floors always clamped an under-sized request (``BlockSizeSpec``
    # raises a value below ``min_size`` silently, so a pinned
    # ``block_sizes=[16, 16, 32]`` on a tcgen05 kernel keeps running the
    # 64-row tile it always ran).
    if view is None or not state.warp_mma_admitted:
        return
    block_sizes, m_index, n_index, _k_index = view
    for index, floor in (
        (m_index, state.tcgen05_min_search_m),
        (n_index, state.tcgen05_min_search_n),
    ):
        value = block_sizes[index]
        if type(value) is int and value < floor:
            block_sizes[index] = floor
    if fix_invalid and WARP_MMA_WARPS_KEY in config:
        config[WARP_MMA_WARPS_KEY] = WARP_MMA_DEFAULT_WARPS


def _one_wave_two_cta_tile_is_valid(
    state: CuteTcgen05Config, bm: object, bn: object
) -> bool:
    """Whole 256 x {128, 64} CtaGroup.TWO tiles of the static problem."""
    extents = state.matmul_compile_time_static_extents
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


def _one_wave_cluster_n2_is_valid(state: CuteTcgen05Config, bn: object) -> bool:
    extents = state.matmul_compile_time_static_extents
    if extents is None or type(bn) is not int or bn <= 0:
        return False
    n = extents[1]
    return (
        not state.matmul_has_leading_passthrough
        and isinstance(n, int)
        and n % bn == 0
        and (n // bn) % 2 == 0
    )


def _pair_cluster_n2_is_valid(state: CuteTcgen05Config, bm: object, bn: object) -> bool:
    """Where the deep-ring seeds place their 2x2 (4-CTA) multicast twin.

    ``tcgen05_cluster_n=2`` multicasts each A half across the two CTA
    pairs of a cluster that share an M tile and walk adjacent N tiles (B
    stays split inside each pair).  The twin seeds take the 256-row
    CtaGroup.TWO pair tile with whole N-tile pairs of an unbatched GEMM,
    the geometry the 2x2 cluster was measured on (nvjet's); the config
    layer admits the cluster more widely (``_paired_pipeline_parameters``:
    every row-major-output pair, batched GEMMs on the plain raster), and
    refuses an odd N tile count on its own (``_cluster_n2_n_tile_refusal``).
    """
    return bm == TCGEN05_TWO_CTA_BLOCK_M and _one_wave_cluster_n2_is_valid(state, bn)


def _cluster_n2_n_tile_refusal(
    state: CuteTcgen05Config, config: dict[str, object]
) -> str | None:
    """Why ``tcgen05_cluster_n=2`` cannot pair this GEMM's N tiles.

    Every 2x2 cluster family walks whole clusters of two N tiles
    (``cute_mma`` refuses an odd ``cdiv(N, block_n)`` at render, after
    the search has spent a compile on the sample).  Judged on the static
    N extent of the single matmul the config tiles; families that own
    their own geometry (grouped worklists, the dense aligned-N plan,
    materialized chains, dynamic shapes) are not judged here.
    """
    if (
        config.get("tcgen05_cluster_n", 1) != 2
        or config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is not None
        or config.get(TILE_N_KEY, 0) != 0
    ):
        return None
    extents = state.matmul_compile_time_static_extents
    view = facts._matmul_config_view(state, config)
    if extents is None or view is None:
        return None
    n = extents[1]
    blocks, _m_index, n_index, _k_index = view
    bn = blocks[n_index]
    if not isinstance(n, int) or type(bn) is not int or bn <= 0:
        return None
    tiles = -(-n // bn)
    if tiles % 2 == 0:
        return None
    return (
        f"tcgen05_cluster_n=2 pairs N tiles, but cdiv({n}, {bn}) = {tiles} "
        "tiles is odd; use tcgen05_cluster_n=1 or a block_n with an even "
        "N tile count"
    )


def _validate_cluster_n_pairs_n_tiles(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    reason = _cluster_n2_n_tile_refusal(state, config)
    if reason is None:
        return
    if fix_invalid:
        config["tcgen05_cluster_n"] = 1
        return
    raise InvalidConfig(reason)


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


def _m_pair_block_m_is_valid(state: CuteTcgen05Config) -> bool:
    """Whether block_m=512 (M-paired tiles) is admissible for this kernel.

    The codegen envelope is the plain 16-bit full-tile static family:
    M divisible by 512 with the canonical 256-wide N tile (the fix pass
    projects bn to 256 and validates bk). Aux kernels and edge families
    keep block_m=256.
    """
    if state.aux_kernel_detected or state.matmul_has_leading_passthrough:
        return False
    constraints = state.cluster_m2_search_constraints
    if (
        constraints is None
        or constraints.allow_edge_k_tail_family
        or constraints.one_wave_only
    ):
        return False
    for fact in state.config_spec.matmul_facts:
        if (
            fact.static_m is not None
            and fact.static_m % (2 * TCGEN05_TWO_CTA_BLOCK_M) == 0
            and fact.static_n is not None
            and fact.static_n % TCGEN05_TWO_CTA_BLOCK_N == 0
            and fact.lhs_dtype in (torch.float16, torch.bfloat16)
        ):
            return True
    return False


def plain_edge_pair128_admitted(state: CuteTcgen05Config) -> bool:
    """Whether the ragged edge family may run 128-row CTA pairs
    (``tcgen05_cta_group="two"`` on ``block_m=128``: each CTA owns 64 rows
    of a 128 x 128 pair tile): a 16-bit plain GEMM on the edge K-tail
    family whose 256-row pairs cannot occupy every SM once (fewer pairs
    than half the SM count; the 256x256 pair is the family's default
    projection and underfills such grids). Gates the seeds, the sampled
    ``"two"`` and the search repair
    that keeps a sampled pair on the 128-row tile.
    """
    constraints = state.cluster_m2_search_constraints
    pipeline_facts = state.pipeline_smem_facts
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    if (
        not state.search_enabled
        or constraints is None
        or not constraints.allow_edge_k_tail_family
        or pipeline_facts is None
        or pipeline_facts.input_dtype_bytes != 2
        or state.aux_kernel_detected
        or TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types
        or extents is None
        or fragments is None
        or any(value is None or value <= 0 for value in extents)
        or state.device_sm_count <= 0
    ):
        return False
    m, n, _k = cast("tuple[int, int, int]", extents)
    bm_fragment, bn_fragment, _bk_fragment = fragments
    if not (bm_fragment.low <= 128 and bn_fragment.low <= 128 <= bn_fragment.high):
        return False
    # Enough 256-row pairs to put one CTA on every SM: the plain seed and
    # the 256x256 projection cover it.
    return -(-m // 256) * -(-n // 256) < state.device_sm_count // 2


def _project_edge_pair128_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    """Project a sampled explicit CTA pair (``tcgen05_cta_group="two"``)
    on a ``bm <= 128`` tile of the ragged edge family onto its validated
    128x128 pair tile: one N tile per CTA (no N multicast), a K tile the
    family admits and an AB depth that fits the pair's SMEM. False when
    the family is not admitted (``plain_edge_pair128_admitted``) or the
    sample is a 256-row tile, which keeps the 256x256 projection.
    """
    if not plain_edge_pair128_admitted(state):
        return False
    view = facts._matmul_config_view(state, config)
    if view is None:
        return False
    block_sizes, m_index, n_index, k_index = view
    bm = block_sizes[m_index]
    if not isinstance(bm, int) or isinstance(bm, bool) or bm > 128:
        return False
    constraints = state.cluster_m2_search_constraints
    assert constraints is not None
    bk = block_sizes[k_index]
    if not (
        isinstance(bk, int)
        and not isinstance(bk, bool)
        and cluster_m2_bk_is_valid(bk, constraints)
    ):
        bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
    block_sizes[m_index] = 128
    block_sizes[n_index] = 128
    block_sizes[k_index] = bk
    config["tcgen05_cluster_m"] = 2
    config["tcgen05_cluster_n"] = 1
    config["tcgen05_acc_stages"] = 2
    config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
    fit = smem.max_ab_stages_that_fit(state, bm=128, bn=128, bk=bk, cluster_m=2)
    stages = config.get("tcgen05_ab_stages", 2)
    if fit > 0 and (not isinstance(stages, int) or stages > fit):
        config["tcgen05_ab_stages"] = fit
    config.pop("epilogue_subtile", None)
    return True


def _normalize_split_k(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Keep ``tcgen05_split_k`` at 1 unless the sampled config is a flat
    one-CTA GEMM whose K loop has at least ``split`` tiles.

    The strict pass (``fix_invalid=False``) refuses an explicit split on the
    facts it shares with the split-K pipeline's envelope in ``cute_mma``:
    the flat pid type, a body whose only side effect is the tile store
    (``split_k_plain_body``) and a K loop at least ``split`` tiles deep.
    The pipeline keeps the plan facts the config layer cannot see (the
    one-CTA TMA pipeline, the tile family, one epilogue warp group).
    """
    split = config.get(TCGEN05_SPLIT_K_CONFIG_KEY, 1)
    searched = state.search_enabled and bool(surface.split_k_search_choices(state))
    if (
        not isinstance(split, int)
        or split <= 1
        # The workspace schedule replaces the whole GEMM launch.
        or config.get("cute_split_k_workspace")
    ):
        if fix_invalid:
            # Off the search surface the knob is absent (a coverage witness
            # or pinned record must round-trip unchanged); on it, it reads 1.
            if searched:
                config[TCGEN05_SPLIT_K_CONFIG_KEY] = 1
            else:
                config.pop(TCGEN05_SPLIT_K_CONFIG_KEY, None)
        return
    if not fix_invalid:
        reason = _split_k_refusal(state, config, split)
        if reason is not None:
            raise InvalidConfig(reason)
        return
    view = facts._matmul_config_view(state, config)
    extents = state.matmul_compile_time_static_extents
    valid = (
        split in surface.split_k_search_choices(state)
        and view is not None
        and extents is not None
        and config.get("pid_type", "flat") == "flat"
        and config.get("tcgen05_cluster_m", 1) == 1
        and config.get("tcgen05_cluster_n", 1) == 1
    )
    if valid:
        assert view is not None and extents is not None
        block_sizes, m_index, _n_index, k_index = view
        bm, bk = block_sizes[m_index], block_sizes[k_index]
        k = cast("tuple[int, int, int]", extents)[2]
        valid = (
            # The tile must run the one-CTA tcgen05 MMA (a 256-row flat
            # tile falls off tcgen05 and could not split).
            bm in (64, TCGEN05_ONE_CTA_MAX_BLOCK_M)
            and isinstance(bk, int)
            and bk > 0
            and -(-k // bk) >= split
        )
    if not valid:
        if searched:
            config[TCGEN05_SPLIT_K_CONFIG_KEY] = 1
        else:
            config.pop(TCGEN05_SPLIT_K_CONFIG_KEY, None)


def _split_k_refusal(
    state: CuteTcgen05Config, config: dict[str, object], split: int
) -> str | None:
    """Why ``tcgen05_split_k=split`` cannot render, judged from the config
    keys and bind-time facts the split-K pipeline reads, or ``None``."""
    k_tiles = split
    view = facts._matmul_config_view(state, config)
    extents = state.matmul_compile_time_static_extents
    if (
        view is not None
        and extents is not None
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is None
    ):
        bk = view[0][view[3]]
        k = cast("tuple[int, int, int]", extents)[2]
        if type(bk) is int and bk > 0 and isinstance(k, int) and k > 0:
            k_tiles = -(-k // bk)
    if (
        config.get("pid_type", "flat") == "flat"
        and state.split_k_plain_body
        and k_tiles >= split
    ):
        return None
    return (
        f"{TCGEN05_SPLIT_K_CONFIG_KEY}={split} requires the flat "
        "one-CTA tcgen05 TMA pipeline over a plain rank-2 GEMM (pid_type='flat', "
        "tcgen05_cluster_m=1, tcgen05_cluster_n=1, one epilogue warp group) "
        f"with at least {split} K tiles"
    )


def _fix_split_k_workspace_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """With the workspace schedule off, its stage and partition knobs are
    inert: a sample reads their defaults so the search does not compile
    one plain kernel per inert value."""
    if config.get(WORKSPACE_KEY):
        return
    if STAGES_KEY in config:
        config[STAGES_KEY] = 2
    if PARTITIONS_KEY in config:
        config[PARTITIONS_KEY] = 1


def _fix_cluster_m2_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if not (state.search_enabled and config.get("tcgen05_cluster_m") == 2):
        return
    if n_edge_cluster_schedule_config_supported(state, config):
        return
    if state.full_tile_added_pairing and _full_tile_schedule_config_supported(
        state, config
    ):
        config["pid_type"] = "persistent_interleaved"
        config.pop("epilogue_subtile", None)
        return
    if _fanout_pair_search_config_supported(state, config):
        # A sampled CTA pair carrying the proved shared fanout keeps the
        # pair geometry its budgets admit (a 16-bit pair N tile, two C
        # stages, role-local monolithic SIMT aux loads; see
        # ``_pair_fanout_refusal``). The projections below
        # -- the canonical 256x256 tile and the aux-TMA producer regime --
        # are calibrated for the single-output residual family and would
        # drop the fanout, so a two-output kernel's pair recipe was never
        # measured by the search.
        config.pop("epilogue_subtile", None)
        return
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        config["tcgen05_cluster_m"] = 1
        return
    block_sizes, m_index, n_index, k_index = config_view
    if config.get(
        TCGEN05_CTA_GROUP_CONFIG_KEY
    ) == "two" and surface.paired_pipeline_search_enabled(state):
        # This family uses the sampled MMA geometry rather than the
        # historical automatic 256x256 cluster projection.
        block_sizes[m_index] = 256 if block_sizes[m_index] == 256 else 128
        if block_sizes[n_index] not in (64, 128, 256):
            block_sizes[n_index] = 128
        if block_sizes[k_index] not in (64, 128, 256):
            block_sizes[k_index] = 64
        # A sampled 2x2 multicast cluster stays where the projected pair
        # tile pairs whole N tiles (the config layer refuses an odd count;
        # a batched sample is projected by
        # ``_fix_batched_cluster_n2_search_config``).
        if (
            config.get("tcgen05_cluster_n", 1) == 2
            and not state.matmul_has_leading_passthrough
            and not _one_wave_cluster_n2_is_valid(state, block_sizes[n_index])
        ):
            config["tcgen05_cluster_n"] = 1
        config["pid_type"] = "persistent_blocked"
        config.pop("epilogue_subtile", None)
        return

    def is_grouped_worklist_two_cta() -> bool:
        return (
            config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
            == TCGEN05_GROUPED_MODE_WORKLIST_NM
            and (
                config.get("tcgen05_cluster_n", 1) == 1
                or (
                    config.get("tcgen05_cluster_n", 1) == 2
                    and config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
                    and state.grouped_device_worklist_supported
                    and resolve_tcgen05_grouped_worklist_mma_profile(
                        config, block_k=block_sizes[k_index]
                    )
                    is not None
                )
            )
            and block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
            and block_sizes[n_index] == 128
            and block_sizes[k_index] in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
            and config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
            in TCGEN05_GROUPED_WORKLIST_TWO_CTA_SOURCE_M_TILE_CHOICES
        )

    constraints = state.cluster_m2_search_constraints
    if is_grouped_worklist_two_cta():
        # The selected worklist source-M family owns the validated physical
        # 256x32/128/224/256 MMA profile and its K envelope independently of
        # the generic cluster-M2 policy. Compiler seeds widen the
        # otherwise-narrow pid fragment, so keep this exact family even
        # when generic constraints reject it or are absent.
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        config.pop("epilogue_subtile", None)
        return
    if constraints is None:
        config["tcgen05_cluster_m"] = 1
        return
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        config["tcgen05_cluster_m"] = 1
        return
    edge_k_tail_family = constraints.allow_edge_k_tail_family
    is_narrow_clc_aux_tma = _is_clc_aux_tma_narrow_n_request(state, config)
    if edge_k_tail_family:
        # Plain (no-aux) kernels may keep a sampled deep-AB bk (the
        # bk=64/ab=5 family, see _plain_edge_deep_ab_stages); aux kernels
        # stay projected onto their validated bk=128/256 regimes.
        sampled_bk = block_sizes[k_index]
        if not (
            not state.aux_kernel_detected
            and isinstance(sampled_bk, int)
            and not isinstance(sampled_bk, bool)
            and cluster_m2_bk_is_valid(sampled_bk, constraints)
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
    if not cluster_m2_bk_is_valid(bk, constraints):
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
        # aux-TMA regime (``_fix_aux_tma_full_tile_search_config``: the
        # TMA aux producer and its C-input warp); a sample on the SIMT aux
        # load without that warp reads its tile per subtile from global
        # memory and takes the narrow tiles like any other epilogue.
        and (
            not state.exact_shape_aux_kernel_detected
            or (
                config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT)
                == TCGEN05_AUX_LOAD_MODE_SIMT
                and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY, 0) == 0
            )
        )
        and _one_wave_two_cta_tile_is_valid(
            state, block_sizes[m_index], block_sizes[n_index]
        )
    ):
        # One-wave family: a sampled 256 x {128, 64} CtaGroup.TWO tile
        # keeps its narrow N (the 256x256 projection would leave most SMs
        # idle on these shapes).  cluster_n=2 needs an even N tile count.
        if config.get("tcgen05_cluster_n", 1) == 2 and not (
            _one_wave_cluster_n2_is_valid(state, block_sizes[n_index])
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
        edge_k_tail_family
        and config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) == "two"
        and _project_edge_pair128_search_config(state, config)
    ):
        # 128-row pair edge family: the sampled explicit pair keeps its
        # 128x128 tile instead of the 256x256 projection below.
        return
    if (
        block_sizes[m_index] == 2 * TCGEN05_TWO_CTA_BLOCK_M
        and not edge_k_tail_family
        and _m_pair_block_m_is_valid(state)
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
    _fix_batched_cluster_n2_search_config(state, config, block_sizes[n_index])
    if edge_k_tail_family:
        if (
            not state.aux_kernel_detected
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
            config["tcgen05_ab_stages"] = seeds._plain_edge_deep_ab_stages(state, bk)
            return
        # This family is pinned to measured production stage/pipeline
        # values after search projection.
        # Placement keys remain available for non-edge diagnostic/search
        # paths, but edge+K-tail candidates do not explore partially
        # mutated placement variants.
        config.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
        if state.aux_kernel_detected and facts._has_any_matmul_fact_edge_tile(
            state, config
        ):
            _set_aux_edge_cluster_m2_prefix(config)
        if _is_clc_aux_tma_config(state, config):
            seeds._set_clc_aux_tma_edge_perf_knobs(state, config)
        if is_narrow_clc_aux_tma:
            config["tcgen05_acc_stages"] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
            config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING]


def _fix_batched_cluster_n2_search_config(
    state: CuteTcgen05Config, config: dict[str, object], bn: object
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
        state.matmul_has_leading_passthrough and config.get("tcgen05_cluster_n", 1) == 2
    ):
        return
    extents = state.matmul_compile_time_static_extents
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


def _fix_grouped_worklist_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """Project worklist search neighbors onto the validated logical tile."""
    if not state.search_enabled:
        return
    if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) != TCGEN05_GROUPED_MODE_WORKLIST_NM:
        return
    source_m_tile = config.get(
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
    )
    if (
        type(source_m_tile) is not int
        or source_m_tile not in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
    ):
        return
    config_view = facts._matmul_config_view(state, config)
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
        for fact in state.config_spec.matmul_facts
        if facts._config_block_index(state, fact.k_block_id) == k_index
        and fact.static_k is not None
    }
    constraints = state.cluster_m2_search_constraints
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
    elif source_m_tile not in TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES:
        # The established compact source-32 profile also supports CtaGroup.ONE. The
        # reviewed source-224/256 profiles are CtaGroup.TWO even when a
        # pattern neighbor independently mutates the cluster fragment.
        config["tcgen05_cluster_m"] = 2


def _is_grouped_clc_config(config: dict[str, object]) -> bool:
    return (
        config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) in TCGEN05_GROUPED_MODES
        and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        == Tcgen05PersistenceModel.CLC_PERSISTENT.value
    )


def _is_grouped_runtime_direct_clc_config(config: dict[str, object]) -> bool:
    return (
        _is_grouped_clc_config(config)
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY) is True
        and TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY not in config
    )


def _normalize_grouped_full_coverage(
    state: CuteTcgen05Config,
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
        raise InvalidConfig("source64 requires the proved grouped dense-local profile")
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
        and state.grouped_full_coverage_supported
    )
    if valid and validate_schedule:
        blocks = config.get("block_sizes")
        indices = facts._matmul_block_indices(state)
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
            and source_m_tile in TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES
            and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
            and config.get(TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY, False)
            is False
            and config.get(TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY, False)
            is False
            and config.get(TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY) is None
            and not _is_grouped_clc_config(config)
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
            and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, "default") == "default"
            and all(config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
            and config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False) is False
            and config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY, False) is False
            and all(
                config.get(key, "normal") == "normal"
                for key in (
                    TCGEN05_C_STORE_MODE_CONFIG_KEY,
                    TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
                    TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
                )
            )
        )
        if valid and config.get("tcgen05_ab_stages", 2) == 4:
            worklist_facts = state.grouped_worklist_smem_facts
            constraints = state.ab_stages_three_search_constraints
            if worklist_facts is not None and constraints is not None:
                valid = full_coverage_smem_upper_bound(
                    worklist_facts.group_count,
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
    state: CuteTcgen05Config,
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
        and facts.row_union_profile_supported(state, schedule)
        and value is True
    ):
        if not fix_invalid:
            raise InvalidConfig(
                "row-union physical schedule requires its typed BF16 shared-RHS proof"
            )
        config.pop(GROUPED_ROW_UNION_SCHEDULE_KEY, None)
        schedule = LEGACY_SCHEDULE
    stages = config.get(GROUPED_ROW_UNION_AB_STAGES_KEY)
    base_profile = (
        row_union_schedule_by_name(schedule)
        if schedule != LEGACY_SCHEDULE and value is True
        else None
    )
    if stages is None:
        config.pop(GROUPED_ROW_UNION_AB_STAGES_KEY, None)
    elif base_profile is None or not row_union_ab_stages_supported(
        base_profile, stages
    ):
        if not fix_invalid:
            raise InvalidConfig(
                f"{GROUPED_ROW_UNION_AB_STAGES_KEY}={stages!r} requires a "
                "physical row-union schedule and a ring depth between "
                f"{AB_STAGES_MIN} and the profile's own"
            )
        config.pop(GROUPED_ROW_UNION_AB_STAGES_KEY, None)
    rounds = config.get(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY, 1)
    if value is True and rounds != 1:
        # A physical profile sizes its C ring from the round
        # (``with_round_subtiles``): the round must be one of the choices,
        # the profile must store through the plain tail (the transposed
        # one) and the wider ring must fit beside the physical AB ring;
        # the legacy one-CTA schedule keeps the per-subtile loop.
        if base_profile is None or not row_union_round_subtiles_supported(
            base_profile,
            rounds,
            config.get(GROUPED_ROW_UNION_AB_STAGES_KEY),
        ):
            if not fix_invalid:
                raise InvalidConfig(
                    f"{TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY}={rounds!r} needs the "
                    "transposed row-union profile and a C ring that fits "
                    "beside its AB ring"
                )
            config.pop(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY, None)
    prefill = config.get(STARTUP_PREFILL_KEY, False)
    if prefill is False:
        config.pop(STARTUP_PREFILL_KEY, None)
    elif not (
        prefill is True
        and value is True
        and schedule == PAIRED_CLC_SCHEDULE
        and state.grouped_row_union_paired_clc_supported
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
            and state.grouped_row_union_multi_resident_supported
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
        state.grouped_row_union_supported
        if schedule == LEGACY_SCHEDULE
        else facts.row_union_profile_supported(state, cast("str", schedule))
    )
    if valid and validate_schedule:
        indices = facts._matmul_block_indices(state)
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
        config.pop(GROUPED_ROW_UNION_AB_STAGES_KEY, None)
        config.pop(GROUPED_RESIDENT_CTAS_KEY, None)
    else:
        raise InvalidConfig(
            f"{GROUPED_ROW_UNION_KEY}={value!r} requires a proved pure "
            "shared-RHS BF16 Int32-offset union, fresh contiguous output, "
            "full 128x64x128 tiles and the ordinary single-CTA AB2/ACC2 "
            "role-local dense pipeline"
        )


def _fanout_pair_search_config_supported(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    """A cluster_m=2 search sample whose proved shared fanout admits its tile."""
    return (
        bool(state.epilogue_fanout_plans)
        and config.get("tcgen05_cluster_m") == 2
        and config.get(FANOUT_CONFIG_KEY) == "shared"
        and epilogue_fanout_config_supported(state, config)
    )


def epilogue_fanout_config_supported(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    return epilogue_fanout_refusal(state, config) is None


def epilogue_fanout_refusal(
    state: CuteTcgen05Config, config: dict[str, object]
) -> str | None:
    """Why ``tcgen05_epilogue_fanout='shared'`` is refused; None if admitted.

    One CTA takes any whole tile of the fanout's outputs. The CTA pair
    (``tcgen05_cluster_m=2``) is admitted by budget rather than at one
    fixed geometry (``_pair_fanout_refusal``): it must store 16-bit
    outputs of the 256-row pair tile (the codegen store protocol's fact)
    at one of the backend's pair N tiles and leave room for one AB stage
    next to both outputs' C rings in the static arena; the ring depth
    itself is judged by ``_normalize_multi_output_rings``.
    """
    if not state.epilogue_fanout_plans or not fanout_schedule_supported(config):
        return "the fanout needs the standard role-local TMA epilogue schedule"
    if state.materialized_matmul_block_ids and config.get("tcgen05_cluster_n", 1) != 1:
        # The materialized-operand pipeline's consumer-wait proof keeps the
        # paired TWO family at cluster_n=1 (materialized_fission_codegen).
        return (
            "the materialized-operand pipeline's consumer-wait proof keeps "
            "the pair at tcgen05_cluster_n=1"
        )
    blocks = config.get("block_sizes")
    if not isinstance(blocks, list):
        return "the fanout needs the kernel's block sizes"
    for plan in state.epilogue_fanout_plans:
        for block_id, extent in zip(plan.block_ids, plan.shape, strict=True):
            index = facts._config_block_index(state, block_id)
            if index is None or index >= len(blocks):
                return "the fanout needs a block size per output axis"
            block = blocks[index]
            if type(block) is not int or block <= 0 or extent % block:
                return (
                    f"the fanout stores whole tiles: {block!r} does not "
                    f"divide the output extent {extent}"
                )
        if config.get("tcgen05_cluster_m", 1) == 2:
            reason = _pair_fanout_refusal(state, config, plan, blocks)
            if reason is not None:
                return reason
        if config.get("tcgen05_aux_load_placement") == "pre_acc_wait":
            if not plan.pre_wait_aux_safe:
                return (
                    "pre_acc_wait needs the pair's auxiliary reads proved safe "
                    "ahead of the accumulator wait"
                )
            # The splice declines SMEM-staged row-vector leaves under
            # fan-out (per-store stage and barrier); decide it here so
            # the config is invalid rather than a codegen failure.
            m_index = facts._config_block_index(state, plan.block_ids[0])
            n_index = facts._config_block_index(state, plan.block_ids[1])
            assert m_index is not None and n_index is not None
            epi_warps = config.get("tcgen05_num_epi_warps", 4)
            if type(epi_warps) is not int or fanout_pre_wait_stages_rowvec_aux(
                plan.chains,
                bm=blocks[m_index],
                bn=blocks[n_index],
                epi_warp_count=epi_warps,
            ):
                return (
                    "pre_acc_wait under the shared fanout cannot stage "
                    "SMEM row-vector auxiliary leaves"
                )
    return None


def _pair_fanout_refusal(
    state: CuteTcgen05Config,
    config: dict[str, object],
    plan: PairedFanoutPlan,
    blocks: list[object],
) -> str | None:
    """Budget admission of the shared fanout on a tcgen05 CTA pair.

    The pair's epilogue stores two 16-bit outputs of each CTA's 128-row
    half of the ``TCGEN05_TWO_CTA_BLOCK_M``-row pair tile through the
    standard role-local TMA protocol at whatever N tile the pair MMA
    runs (``TCGEN05_TWO_CTA_PAIR_BLOCK_NS``), so the geometry is admitted
    when its budgets hold rather than at one validated shape: the M tile
    must be the 256-row pair tile (the fact the codegen gate
    ``epilogue_fanout.store_protocol_supported`` demands of a two-CTA
    store; a 128-row tile under ``tcgen05_cluster_m=2`` -- the clustered
    one-CTA family, or the pair MMA under ``tcgen05_cta_group='two'`` /
    the fp8 small grid -- has no such epilogue), and one AB stage must
    fit next to both outputs' C rings and the fixed allocations in the
    per-CTA SMEM arena (``default_layout_smem_fits``, which charges one
    ring per output).  The accumulator stages of a clustered fanout are
    pinned to two by ``fanout_schedule_supported`` and ``bn <= 256`` by
    the pair N tiles, so the 512 TMEM columns always hold them.  Fails
    closed without a recorded SMEM budget.
    """
    m_index = facts._config_block_index(state, plan.block_ids[0])
    n_index = facts._config_block_index(state, plan.block_ids[1])
    assert m_index is not None and n_index is not None
    bm = blocks[m_index]
    bn = blocks[n_index]
    assert type(bm) is int and type(bn) is int
    if bm != TCGEN05_TWO_CTA_BLOCK_M:
        return (
            "a clustered fanout's role-local TMA-store epilogue stores each "
            f"CTA's {TCGEN05_TWO_CTA_BLOCK_M // 2}-row half of the "
            f"{TCGEN05_TWO_CTA_BLOCK_M}-row tcgen05 pair tile; a {bm}-row "
            f"tile under tcgen05_cluster_m=2 (the clustered one-CTA family, "
            f"or the pair MMA under {TCGEN05_CTA_GROUP_CONFIG_KEY}='two') "
            "has no such epilogue"
        )
    if plan.output_dtype.itemsize != 2:
        return "the CTA-pair fanout stores 16-bit outputs"
    if bn not in TCGEN05_TWO_CTA_PAIR_BLOCK_NS:
        return (
            f"the CTA pair runs its MMA at N tiles "
            f"{sorted(TCGEN05_TWO_CTA_PAIR_BLOCK_NS)}, not {bn}"
        )
    if config.get("tcgen05_cluster_n", 1) == 2 and (plan.shape[1] // bn) % 2:
        # The N-pair multicast walks whole clusters of two N tiles; the
        # persistent scheduler would pad an odd count with a tile that does
        # not exist (``cute_mma`` refuses the render).
        return (
            f"the 2x2 cluster pairs N tiles, but {plan.shape[1]} // {bn} = "
            f"{plan.shape[1] // bn} tiles is odd"
        )
    if state.materialized_matmul_block_ids:
        # A fission plan or barrier chain budgets each GEMM launch from
        # its typed pair proof (``_materialized_paired_stage_limits``).
        return None
    view = facts._matmul_config_view(state, config)
    if view is None:
        return "the CTA-pair fanout needs the matmul's K block"
    block_sizes, _m, _n, k_index = view
    bk = block_sizes[k_index]
    if type(bk) is not int or bk <= 0:
        return "the CTA-pair fanout needs a positive K block"
    c_stages = config.get("tcgen05_c_stages", 2)
    assert type(c_stages) is int
    rowvec_facts = state.rowvec_aux_facts
    outputs = rowvec_facts.output_count if rowvec_facts is not None else 1
    if not smem.default_layout_smem_fits(
        state,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=2,
        ab_stages=1,
        c_stages=c_stages,
        stage_rows=smem._store_lowering_stages_rows(config),
        epi_warps=epi_warps_knob(config),
        acc_stages=acc_stages_knob(config),
        epi_subtile_n=epi_subtile_n_knob(config),
        epi_warp_groups=epi_warp_groups_knob(config),
    ):
        return (
            f"a {bm}x{bn}x{bk} CTA-pair tile leaves no room for one AB "
            f"stage next to the {outputs} outputs' {c_stages}-stage C rings "
            "in the per-CTA SMEM arena"
        )
    return None


def _normalize_epilogue_fanout(
    state: CuteTcgen05Config,
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
            and len(stages) == len(state.materialized_matmul_block_ids)
            and len(stages) >= 1
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
            (
                len(state.materialized_matmul_block_ids) < 2
                or state.region_pid_type_search_enabled
            )
            and state.materialized_operand_pdl_roots is None
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
            if len(state.materialized_matmul_block_ids) >= 2
            and not state.region_pid_type_search_enabled
            else ()
        ),
    )
    if validate_schedule and any(key in config for key in region_keys):
        if smem._materialized_paired_stage_limits(state, config) is None:
            if not fix_invalid:
                raise InvalidConfig(
                    "per-region pipelines and PDL require the complete typed paired materialized fanout schedule"
                )
            for key in region_keys:
                config.pop(key, None)
    if (
        validate_schedule
        and len(state.materialized_matmul_block_ids) < 2
        and config.get("tcgen05_materialized_pdl")
    ):
        supported = facts.materialized_operand_pdl_supported(state, config)
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
        and state.epilogue_fanout_plans
        and (not validate_schedule or epilogue_fanout_config_supported(state, config))
    ):
        return
    if fix_invalid:
        config.pop(FANOUT_CONFIG_KEY, None)
        # The repaired config has to be a fixed point of strict
        # normalization: ``before_store`` is legal only beside the shared
        # fanout just dropped, so it leaves with it (the autotuner's
        # neighbour canonicalization repairs once, then validates).
        if config.get(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY) == "before_store":
            config.pop(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY)
        # Likewise ``pre_acc_wait`` where only the fanout admitted it (an
        # aux-free store), and the fragment hoist that needs the
        # placement: a hoist witness whose neighbour moved to a schedule
        # the fanout refuses otherwise reaches the render worker and is
        # refused there.
        if config.get(
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
        ) == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT and not pre_acc_wait_admissible(
            state, config
        ):
            config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
            )
        if config.get(
            TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
        ) is True and not aux_fragment_hoist_admissible(state, config):
            config[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = False
    else:
        reason = (
            epilogue_fanout_refusal(state, config)
            if type(value) is str
            and value == "shared"
            and state.epilogue_fanout_plans
            and validate_schedule
            else None
        )
        raise InvalidConfig(
            f"{FANOUT_CONFIG_KEY}={value!r} requires a proved equal-layout "
            "fresh-output fanout and a static-full standard "
            "role-local TMA epilogue" + (f": {reason}" if reason else "")
        )


def _normalize_batch_raster(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``tcgen05_batch_raster`` names a walk over a batched grid only.

    Plain grids raster through ``loop_orders``; a raster carried onto a
    plain bind (config reuse across kernels) is dropped by the fix pass
    and rejected by validation.  Batched binds validate the value through
    the optional fragment.
    """
    if TCGEN05_BATCH_RASTER_CONFIG_KEY not in config:
        return
    if state.matmul_has_leading_passthrough:
        return
    if fix_invalid:
        config.pop(TCGEN05_BATCH_RASTER_CONFIG_KEY, None)
    else:
        raise InvalidConfig(
            f"{TCGEN05_BATCH_RASTER_CONFIG_KEY} requires a batched "
            "(leading passthrough) tcgen05 GEMM"
        )


def _normalize_persistent_grid_cap(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``tcgen05_persistent_grid_cap`` only matters on a plain cluster_n=2 grid.

    The knob chooses between the SM-pair cap and the resident-cluster cap
    of a static-persistent launch (``program_id``).  Both caps agree on
    ``cluster_n=1`` grids and batched grids already take the
    resident-cluster cap, so the knob is inert there: an explicit default
    is dropped (fragment validation refills it), and a non-default value
    on an inert bind is dropped by the fix pass and rejected by validation.
    """
    if TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY not in config:
        return
    value = config[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY]
    if value == TCGEN05_PERSISTENT_GRID_CAP_DEFAULT:
        config.pop(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY)
        return
    pid_type = config.get("pid_type")
    live = (
        not state.matmul_has_leading_passthrough
        and config.get("tcgen05_cluster_n", 1) == 2
        and pid_type in ("persistent_blocked", "persistent_interleaved")
        and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        != Tcgen05PersistenceModel.CLC_PERSISTENT.value
    )
    if live:
        return
    if fix_invalid:
        config.pop(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, None)
    else:
        raise InvalidConfig(
            f"{TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY}={value!r} requires a "
            "plain (non-batched) static-persistent tcgen05 GEMM with "
            "tcgen05_cluster_n=2"
        )


def _grouped_grid_multiplier_supported(config: dict[str, object]) -> bool:
    """Admit a second independent persistent worker per physical SM.

    Each worker strides by the actual grid size and owns its flattened
    tensor-map workspace row. The launch multiplier describes the grid;
    CUDA still schedules workers within the compiled resource limits.
    Cluster cooperation and fixed execution signatures retain their
    existing one-worker contract.
    """
    multiplier = config.get("num_sm_multiplier", 1)
    if type(multiplier) is int and multiplier == 1:
        return True
    return (
        type(multiplier) is int
        and multiplier == 2
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get("tcgen05_cluster_m", 1) == 1
        and config.get("tcgen05_cluster_n", 1) == 1
        and not _is_grouped_clc_config(config)
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
        and config.get(TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY) is None
        and config.get(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, 0) == 0
    )


def _grouped_static_registers_supported(config: dict[str, object]) -> bool:
    return (
        _grouped_grid_multiplier_supported({**config, "num_sm_multiplier": 2})
        and config.get(TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY) in (None, 128)
        and config.get(
            TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
            TCGEN05_GROUPED_FULL_COVERAGE_OFF,
        )
        == TCGEN05_GROUPED_FULL_COVERAGE_OFF
        and physical_schedule(config) is None
    )


def _row_union_register_reallocation_optional(config: dict[str, object]) -> bool:
    """The transposed row-union profile launches six warps at one CTA per
    SM, so every warp can hold the consumer register budget from launch
    and the ``setmaxnreg`` pair only lengthens the TMA-load warp's
    prologue (``_tcgen05_prologue_runtime``): the search may drop it.
    The paired profile owns its static register protocol instead."""
    profile = physical_schedule(config)
    return profile is not None and profile.paired_protocol is None


def _normalize_warp_register_reallocation(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY
    if key not in config:
        return
    value = config[key]
    if value is True:
        config.pop(key)
        return
    if (
        value is not False
        or not state.search_enabled
        or not (
            _grouped_static_registers_supported(config)
            or _row_union_register_reallocation_optional(config)
        )
    ):
        if fix_invalid:
            config.pop(key)
            return
        raise InvalidConfig(
            f"{key}=False requires an independent single-CTA N,M worklist "
            "with the default physical M128 register profile, or the "
            "transposed row-union profile"
        )
    if TCGEN05_CONSUMER_REGS_CONFIG_KEY in config:
        if (
            config[TCGEN05_CONSUMER_REGS_CONFIG_KEY] == TCGEN05_CONSUMER_REGS_DEFAULT
            or fix_invalid
        ):
            # The default consumer budget is inert without the pair (a
            # full config materializes it); an explicit other value asks
            # for a reallocation that is not emitted.
            config.pop(TCGEN05_CONSUMER_REGS_CONFIG_KEY)
        else:
            raise InvalidConfig(
                f"{TCGEN05_CONSUMER_REGS_CONFIG_KEY} requires warp register "
                "reallocation"
            )


def _normalize_dense_aligned_n(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    width = config.get(TILE_N_KEY, 0)
    if type(width) is int and width == 0:
        return
    if type(width) is not int or width not in ALIGNED_N_CHOICES:
        raise InvalidConfig(
            "tcgen05_tile_n requires an aligned non-power-of-two N width"
        )
    if fix_invalid:
        view = facts._matmul_config_view(state, config)
        if view is not None:
            blocks, mi, ni, _ki = view
            blocks = list(blocks)
            blocks[mi], blocks[ni] = 128, carrier_n(width)
            config["block_sizes"] = blocks
        acc_stages = config.get("tcgen05_acc_stages", 2)
        if type(acc_stages) is not int or acc_stages not in (1, 2):
            acc_stages = 2
        store_mode = config.get("tcgen05_c_store_mode", "normal")
        if store_mode not in ("normal", "direct"):
            store_mode = "normal"
        config.update(
            dense_aligned_n_defaults(
                acc_stages=acc_stages, store_mode=cast("str", store_mode)
            )
        )
        if config.get("pid_type") not in state.allowed_pid_types or config.get(
            "pid_type"
        ) not in ("persistent_interleaved", "persistent_blocked"):
            config["pid_type"] = facts._preferred_persistent_pid_type(state)
    plan = facts._dense_aligned_n_plan(state, config)
    if plan is None:
        raise InvalidConfig(
            "tcgen05_tile_n requires its proved full-M/N dense FP8 paired static schedule and unchanged tile bounds"
        )
    for key, value in dense_aligned_n_defaults(
        acc_stages=cast("int", config.get("tcgen05_acc_stages", 2)),
        store_mode=cast("str", config.get("tcgen05_c_store_mode", "normal")),
    ).items():
        config.setdefault(key, value)
    depth = config.setdefault("tcgen05_ab_stages", 2)
    if type(depth) is int and 1 <= depth <= plan.max_ab_stages:
        return
    if not fix_invalid:
        raise InvalidConfig(
            f"tcgen05_tile_n complete SMEM budget permits AB stages 1..{plan.max_ab_stages}"
        )
    config["tcgen05_ab_stages"] = (
        max(1, min(depth, plan.max_ab_stages)) if type(depth) is int else 1
    )


def _normalize_grouped_panel_adaptation(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    alternate = config.get(GROUPED_PANEL_ALTERNATE_KEY)
    ratio = config.get(GROUPED_PANEL_RATIO_KEY)
    if alternate is None and ratio is None:
        config.pop(GROUPED_PANEL_ALTERNATE_KEY, None)
        config.pop(GROUPED_PANEL_RATIO_KEY, None)
        return
    valid = (
        type(alternate) is int
        and alternate in TCGEN05_LEGAL_L2_SWIZZLE_SIZES
        and type(ratio) is int
        and ratio in GROUPED_PANEL_RATIO_CHOICES
        and state.grouped_device_mn_swizzle_supported
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
    )
    if valid:
        return
    if fix_invalid:
        config.pop(GROUPED_PANEL_ALTERNATE_KEY, None)
        config.pop(GROUPED_PANEL_RATIO_KEY, None)
        return
    raise InvalidConfig(
        "adaptive panel widths require paired alternate/positive integer "
        "M/N ratio knobs and the proved typed-offset MN worklist"
    )


def _normalize_grouped_d_shift(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY
    value = config.get(key, False)
    if value is False or value is None:
        config.pop(key, None)
        return
    view = facts._matmul_config_view(state, config)
    valid = (
        value is True
        and state.grouped_d_shift_supported
        and view is not None
        and view[0][view[1]] == 256
        and view[0][view[2]] == 128
        and view[0][view[3]] in (64, 128)
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get(TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY) is True
        and not _is_grouped_clc_config(config)
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
        and config.get(GROUPED_ROW_UNION_KEY, False) is False
        and config.get(
            TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
            TCGEN05_GROUPED_FULL_COVERAGE_OFF,
        )
        == TCGEN05_GROUPED_FULL_COVERAGE_OFF
    )
    if valid:
        return
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(
        f"{key}=True requires a proved pure fresh-output device worklist "
        "with immutable full-allocation inputs"
    )


def _normalize_grouped_balanced_workers(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY
    value = config.get(key, False)
    if value is False or value is None:
        config.pop(key, None)
        return
    view = facts._matmul_config_view(state, config)
    valid = (
        value is True
        and state.grouped_balanced_workers_supported
        and view is not None
        and view[0][view[1]] == 256
        and view[0][view[2]] == 128
        and view[0][view[3]] in (64, 128)
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
        and config.get("tcgen05_cluster_m", 1) == 2
        and config.get("tcgen05_cluster_n", 1) in (1, 2)
        and config.get("num_sm_multiplier", 1) == 1
        and not _is_grouped_clc_config(config)
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
        and config.get(GROUPED_ROW_UNION_KEY, False) is False
        and config.get(
            TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
            TCGEN05_GROUPED_FULL_COVERAGE_OFF,
        )
        == TCGEN05_GROUPED_FULL_COVERAGE_OFF
    )
    if valid:
        return
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(
        f"{key}=True requires a proved pure fresh-output device M,N worklist "
        "with complete output feature clusters and one persistent cluster "
        "per hardware slot"
    )


def _normalize_grouped_prefix_adaptive_n(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY
    value = config.get(key, False)
    if value is False:
        config.pop(key, None)
        return
    view = facts._matmul_config_view(state, config)
    profile = facts.get_grouped_prefix_adaptive_n_profile(state, config)
    panel_supported = profile is not None and (
        profile.row_tile == 128
        and config.get(GROUPED_PANEL_ALTERNATE_KEY) is None
        and config.get(GROUPED_PANEL_RATIO_KEY) is None
        or profile.row_tile == 256
        and config.get(GROUPED_PANEL_ALTERNATE_KEY) == 4
        and config.get(GROUPED_PANEL_RATIO_KEY) == 1
    )
    valid = (
        value is True
        and state.grouped_prefix_adaptive_n_supported
        and profile is not None
        and view is not None
        and view[0][view[1]] == 256
        and view[0][view[2]] == 128
        and view[0][view[3]] == 64
        and config.get(TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY) is None
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
        and config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
        == profile.row_tile
        and config.get("tcgen05_grouped_worklist_mma_n") == profile.primary.n
        and config.get("tcgen05_cluster_m", 1) == 2
        and config.get("tcgen05_cluster_n", 1) == 1
        and config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto") in ("auto", "two")
        and config.get("cute_min_blocks_per_mp", 0) == 1
        and config.get("tcgen05_ab_stages") == profile.primary.ab_stages
        and config.get("tcgen05_c_stages") == profile.primary.c_stages
        and config.get("tcgen05_acc_stages", 2) == 2
        and config.get("tcgen05_num_epi_warps", 4) == 4
        and config.get(TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY) is True
        and config.get(TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY) is True
        and type(config.get(TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY, False)) is bool
        and config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 1) == 8
        and panel_supported
        and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, "default") == "default"
        and all(config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
        and config.get(TCGEN05_C_STORE_MODE_CONFIG_KEY, TCGEN05_C_STORE_MODE_NORMAL)
        == TCGEN05_C_STORE_MODE_NORMAL
        and config.get("pid_type") in ("persistent_interleaved", "persistent_blocked")
        and config.get("num_sm_multiplier", 1) == 1
        and config.get("tcgen05_warp_spec_c_input_warps", 0) == 0
        and config.get("tcgen05_warp_spec_mma_warps", 1) == 1
        and config.get("tcgen05_warp_spec_ab_load_warps", 1) == 1
        and config.get("tcgen05_warp_spec_scheduler_warps", 0) == 0
        and config.get("tcgen05_warp_spec_epi_load_warps", 0) == 0
        and config.get("tcgen05_warp_spec_store_warps", 0) == 0
        and config.get(TCGEN05_CONSUMER_REGS_CONFIG_KEY, 256) == 256
        and not _is_grouped_clc_config(config)
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
        and config.get(GROUPED_ROW_UNION_KEY, False) is False
        and config.get(
            TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
            TCGEN05_GROUPED_FULL_COVERAGE_OFF,
        )
        == TCGEN05_GROUPED_FULL_COVERAGE_OFF
    )
    if valid:
        return
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(
        f"{key}=True requires a proved fresh immutable paired MN prefix "
        "K64 recipe with coherent source tile, panel policy and output rings"
    )


def _normalize_grouped_ab_tile_k(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
    requested = config.get(key)
    if requested is None:
        config.pop(key, None)
        return
    view = facts._matmul_config_view(state, config)
    profile = resolve_tcgen05_grouped_worklist_mma_profile(config, block_k=128)
    valid = (
        type(requested) is int
        and requested == 128
        and state.grouped_ab_k_fusion_supported
        and view is not None
        and view[0][view[1]] == TCGEN05_TWO_CTA_BLOCK_M
        and view[0][view[2]] == 128
        and view[0][view[3]] == 64
        and profile is not None
        and config.get("tcgen05_cluster_n", 1) == 1
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
        == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get(
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
        )
        != TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
        and config.get(
            TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
            TCGEN05_GROUPED_FULL_COVERAGE_OFF,
        )
        == TCGEN05_GROUPED_FULL_COVERAGE_OFF
        and config.get(GROUPED_ROW_UNION_KEY, False) is False
    )
    if not valid:
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}=128 requires a proved pure direct-load grouped K64 "
            "addmm recurrence with a zero FP32 carry and K divisible by 128"
        )
    worklist_facts = state.grouped_worklist_smem_facts
    constraints = state.ab_stages_three_search_constraints
    if worklist_facts is None or constraints is None:
        return
    assert profile is not None
    stages = config.get("tcgen05_ab_stages", 2)
    acc_stages = config.get("tcgen05_acc_stages", 2)
    c_stages = config.get("tcgen05_c_stages", 2)
    if any(
        type(value) is not int or value <= 0 for value in (stages, acc_stages, c_stages)
    ):
        return
    capacity = (
        constraints.per_cta_smem_budget_bytes
        + TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
    )
    assert type(stages) is int
    assert type(acc_stages) is int
    assert type(c_stages) is int
    immutable_tensormaps = smem._grouped_worklist_immutable_smem(state, config)
    overrides = layout_overrides_from_config(config)
    epi_tile_shape = (
        (overrides.epi_tile_m, overrides.epi_tile_n)
        if overrides.epi_tile_m is not None and overrides.epi_tile_n is not None
        else None
    )

    def footprint(depth: int) -> int:
        return tcgen05_grouped_worklist_smem_bytes(
            group_count=worklist_facts.group_count,
            device_split_sizes=worklist_facts.device_split_sizes,
            sched_stage_count=cast(
                "int", config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY, 1)
            ),
            bm=profile.mma_m,
            bn=profile.mma_n,
            bk=128,
            dtype_bytes=constraints.dtype_bytes,
            output_dtype_bytes=worklist_facts.output_dtype_bytes,
            ab_stages=depth,
            acc_stages=acc_stages,
            c_stages=c_stages,
            cluster_m=profile.cluster_m,
            orientation=str(config.get("tcgen05_grouped_worklist_orientation", "nm")),
            epi_tile_shape=epi_tile_shape,
            immutable_ab_tensormaps=immutable_tensormaps,
            immutable_d_tensormap=immutable_tensormaps,
        )

    required = footprint(stages)
    if required <= capacity:
        return
    if not fix_invalid:
        raise InvalidConfig(
            f"physical grouped K128 stages require {required} bytes of "
            f"per-CTA SMEM, exceeding {capacity} bytes"
        )
    while stages > 1 and footprint(stages) > capacity:
        stages -= 1
    if footprint(stages) > capacity:
        raise InvalidConfig("physical grouped K128 stage does not fit per-CTA SMEM")
    config["tcgen05_ab_stages"] = stages


def _uses_grouped_static_reserved_sms(config: dict[str, object]) -> bool:
    return (
        config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) in TCGEN05_GROUPED_DYNAMIC_MODES
        and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        == Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )


def _normalize_grouped_static_reserved_sms(
    state: CuteTcgen05Config,
    config: dict[str, object],
) -> None:
    reserved_sms_key = TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
    if reserved_sms_key not in config:
        return
    if not _uses_grouped_static_reserved_sms(config):
        config.pop(reserved_sms_key, None)


def _fix_c_stages_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    # Invariant: a config carrying ``tcgen05_c_stages=4`` from any source (the
    # knob is an ``EnumFragment((2, 4))``, so the search samples it
    # independently of every projection) keeps it only when ``c_stages_fits``
    # admits AB + C; otherwise it is demoted to 2, as
    # ``_fix_ab_stages_three_search_config`` demotes AB.  A 256x256
    # cluster_m=2 ab=3 + c=4 config past this gate fails in ptxas with a raw
    # ``too much shared`` error.
    #
    # Scope: the canonical full-tile 256x256 DEFAULT-layout path, which is
    # exactly where the AB+C arithmetic is calibrated (the validated
    # CtaGroup.TWO cosize shapes) and where the role-local C ring lives,
    # plus the one-CTA bm=128 tiles (``_fix_one_cta_c_stages_search_config``).
    # The narrow-N / bm=128 edge family (``_fix_aux_edge_search_config`` sets
    # its own validated c=4 at a different cosize that the analytic model
    # would mis-judge) and the EXPLICIT_EPI_TILE direct-entry seeds (separate
    # (128, 32) tile + own admission) keep their c=4 untouched.
    if not state.search_enabled:
        return
    if config.get("tcgen05_c_stages") != TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES:
        return
    if not facts._is_default_layout_full_tile_config(state, config):
        _fix_one_cta_c_stages_search_config(state, config)
        return
    # Fail CLOSED: the ``(2, 4)`` c-stages fragment is offered on every
    # device, but with no SMEM budget recorded (non-B200 / CPU host) we
    # cannot prove c=4 fits — demote rather than leave the ptxas-overflow
    # window open. ``c_stages_fits`` itself returns False when constraints
    # are absent, so a single ``not c_stages_fits`` check covers both the
    # over-budget and the no-budget arms.
    config_view = facts._matmul_config_view(state, config)
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
    if not smem.c_stages_fits(
        state,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=_projected_ab_stages(
            state, config, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
        ),
        c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
        has_source_c=state.aux_kernel_detected,
    ):
        config["tcgen05_c_stages"] = 2


def _fix_one_cta_c_stages_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
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
    config_view = facts._matmul_config_view(state, config)
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
    if state.aux_kernel_detected and facts._has_any_matmul_fact_edge_tile(
        state, config
    ):
        return
    if not smem.default_layout_smem_fits(
        state,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=1,
        ab_stages=_projected_ab_stages(state, config, bm=bm, bn=bn, bk=bk, cluster_m=1),
        c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
        stage_rows=smem._store_lowering_stages_rows(config),
        epi_warps=epi_warps_knob(config),
        acc_stages=acc_stages_knob(config),
    ):
        config["tcgen05_c_stages"] = 2


def pre_acc_wait_admissible(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    """Whether the store lowering renders ``pre_acc_wait`` for ``config``.

    Mirrors ``memory_ops``: the placement needs per-subtile auxiliary
    loads at every store it reaches -- every analyzed store chain loads
    an auxiliary operand (``aux_loads_on_every_store``), or the shared
    fanout epilogue (``tcgen05_epilogue_fanout='shared'``) merges each
    proved pair into one chain that does, with the pair's reads provably
    safe ahead of the accumulator wait (``pre_wait_aux_safe``).
    """
    if state.aux_loads_on_every_store:
        return True
    return config.get(
        FANOUT_CONFIG_KEY, "off"
    ) == "shared" and pre_acc_wait_admissible_via_fanout(state)


def pre_acc_wait_admissible_via_fanout(state: CuteTcgen05Config) -> bool:
    """Whether the shared fanout epilogue would legalize ``pre_acc_wait``.

    Every proved pair must merge into a chain with auxiliary loads whose
    reads are safe ahead of the accumulator wait.
    """
    plans = state.epilogue_fanout_plans
    return bool(plans) and all(
        plan.pre_wait_aux_safe
        and any(chain.auxiliary_tensor_loads for chain in plan.chains)
        for plan in plans
    )


def aux_fragment_hoist_admissible(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> bool:
    """Whether ``tcgen05_aux_fragment_hoist=True`` is legal for ``config``.

    The hoist reads an exact-shape auxiliary tile into registers before
    the accumulator wait, so it needs such an operand
    (``exact_shape_aux_kernel_detected``: rank-2, the output's shape and
    dtype), the ``pre_acc_wait`` placement, the SIMT aux load path and
    the normal TMA-store epilogue, and a per-thread fragment within
    ``TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD`` (a CTA pair's
    epilogue owns half of the pair tile's rows per CTA).

    The store lowering renders the hoist only on the static-full
    TMA-store epilogue; this mirrors that from the config (the plan's
    TMA-store fact also needs the output's TMA legality, and the hoisting
    store must read the exact-shape operand itself; the lowering checks
    both). The tile must divide the compile-time static extents
    (``_static_full_tiles``: the edge-tile TMA-store families and the
    predicated SIMT store take no hoist), and a clustered config must run
    the tcgen05 CTA pair (``_config_is_two_cta_mma``): the clustered
    single-CTA family (16-bit ``bm=128`` with ``tcgen05_cluster_m=2``)
    has no role-local TMA-store epilogue. A fragment whose thread spans
    several rows (an epilogue tile narrower than the 128-lane TMEM
    datapath, e.g. ``bm=64``) is legal: the lowering re-lays each
    subtile's slice onto the accumulator carrier's profile.

    On a materialized chain the hoist renders on every GEMM launch with
    an exact-shape auxiliary operand (``chain_launches``); each is judged
    as the renderer sees it (``region``).
    """
    rowvec_facts = state.rowvec_aux_facts
    if (
        not state.search_enabled
        or not state.exact_shape_aux_kernel_detected
        or rowvec_facts is None
    ):
        return False
    if region is None and state.materialized_matmul_block_ids:
        launches = facts.chain_launches(state, config, aux_only=True)
        return bool(launches) and all(
            aux_fragment_hoist_admissible(state, launch, region=launch_region)
            for launch_region, launch in launches
        )
    if (
        config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
        != TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        or config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT)
        != TCGEN05_AUX_LOAD_MODE_SIMT
        or config.get(TCGEN05_C_STORE_MODE_CONFIG_KEY, TCGEN05_C_STORE_MODE_NORMAL)
        != TCGEN05_C_STORE_MODE_NORMAL
        or config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY)
    ):
        return False
    config_view = facts._matmul_config_view(state, config, region=region)
    if config_view is None:
        return False
    block_sizes, m_index, n_index, k_index = config_view
    bm = block_sizes[m_index]
    bn = block_sizes[n_index]
    bk = block_sizes[k_index]
    cluster_m = config.get("tcgen05_cluster_m", 1)
    if not (
        isinstance(bm, int)
        and isinstance(bn, int)
        and isinstance(bk, int)
        and bm > 0
        and bn > 0
        and bk > 0
        and cluster_m in (1, 2)
    ):
        return False
    if not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk, region=region):
        return False
    if region is not None and (
        # The one-shot flat launch of a chain has no whole-fragment hoist
        # (``memory_ops`` renders it on the role-local persistent
        # schedule only).
        config.get("pid_type") == "flat"
        or _chain_launch_arena_refusal(
            state, config, region=region, bm=bm, bn=bn, bk=bk
        )
    ):
        return False
    is_two_cta = facts._config_is_two_cta_mma(state, config, bm=bm, cluster_m=cluster_m)
    if cluster_m == 2 and not is_two_cta:
        return False
    rows = bm // 2 if is_two_cta else bm
    if rows <= 0:
        return False
    groups = epi_warp_groups_knob(config)
    if groups > 1:
        # Each epilogue warp group hoists the fragments of its own
        # subtiles through the auxiliary ring, ``ceil(count / G)`` of
        # them (the long groups of a ragged split; the store lowering's
        # ring budget prices the same count), not the average share.
        count = smem.epilogue_subtile_count(state, config, region=region)
        if count is None:
            return False
        epi_tile_m, epi_tile_n = smem.epilogue_subtile(
            state,
            bm=bm,
            bn=bn,
            cluster_m=cluster_m,
            stage_rows=smem._store_lowering_stages_rows(config),
            epi_subtile_n=epi_subtile_n_knob(config),
        )
        return (
            tcgen05_aux_prefetch_bytes_per_thread(
                epi_tile_m=epi_tile_m,
                epi_tile_n=epi_tile_n,
                itemsize=rowvec_facts.output_itemsize,
                depth=-(-count // groups) - 1,
            )
            <= TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD
        )
    return (
        tcgen05_aux_fragment_bytes_per_thread(
            bm=rows,
            bn=bn,
            itemsize=rowvec_facts.output_itemsize,
            epi_warps=epi_warps_knob(config),
        )
        <= TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD
    )


def _standard_tma_epilogue_schedule_refusal(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> str | None:
    """Why ``config`` is not the standard role-local TMA-store epilogue.

    The epilogue subtile knob, the epilogue warp groups and the auxiliary
    prefetch ring render on the one schedule the store lowering's default
    subtile loop owns: the monolithic role-local strategy with four
    epilogue warps, the default layout without overrides, the normal
    TMA store, register (SIMT) auxiliary reads, static full tiles of a
    rank-2 row-major output and -- clustered -- the 256-row CtaGroup.TWO
    pair. Every other family (grouped, store or C-input warps, the
    diagnostic layouts) keeps its own epilogue.  A materialized chain's
    GEMM launches run this schedule each (``region``: the launch judged,
    as ``region_launch_config`` projects it); judged on the whole-kernel
    config of a chain (``region`` None) the knobs are refused, as the
    callers that have no launch in hand expect.
    """
    expected: dict[str, object] = {
        TCGEN05_STRATEGY_CONFIG_KEY: Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
        TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: Tcgen05LayoutStrategy.DEFAULT.value,
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY: "normal",
        TCGEN05_C_STORE_MODE_CONFIG_KEY: TCGEN05_C_STORE_MODE_NORMAL,
        "tcgen05_num_epi_warps": TCGEN05_EPI_WARPS_PER_GROUP,
        TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 0,
        TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: 0,
        TCGEN05_WARP_SPEC_STORE_WARPS_KEY: 0,
    }
    for key, default in expected.items():
        value = config.get(key, default)
        if type(value) is not type(default) or value != default:
            return f"requires {key}={default!r}"
    if (
        config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT)
        != TCGEN05_AUX_LOAD_MODE_SIMT
    ):
        return "requires the register (simt) auxiliary load mode"
    if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is not None:
        return "is not available for grouped matmuls"
    if config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY):
        return "requires the launch-axis role coordinates"
    if config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY):
        return "is not available with the diagnostic invalid output"
    if any(config.get(key) is not None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS):
        return "requires no explicit layout overrides"
    if (
        config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        == Tcgen05PersistenceModel.CLC_PERSISTENT.value
    ):
        return "is not validated with CLC persistence"
    if region is None and state.materialized_matmul_block_ids:
        return "is judged per GEMM launch on a materialized fission chain"
    if state.matmul_has_leading_passthrough:
        return "requires a rank-2 matmul"
    pipeline_facts = facts._launch_pipeline_facts(state, region)
    if pipeline_facts is not None and pipeline_facts.output_column_major:
        return "requires a row-major output"
    cluster_m = config.get("tcgen05_cluster_m", 1)
    if type(cluster_m) is not int or cluster_m not in (1, 2):
        return "requires tcgen05_cluster_m in (1, 2)"
    view = facts._matmul_config_view(state, config, region=region)
    if view is None:
        return "requires the tcgen05 matmul block view"
    blocks, mi, ni, ki = view
    bm, bn, bk = blocks[mi], blocks[ni], blocks[ki]
    if not all(type(value) is int and value > 0 for value in (bm, bn, bk)):
        return "requires integer tcgen05 block sizes"
    bm, bn, bk = cast("int", bm), cast("int", bn), cast("int", bk)
    if not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk, region=region):
        return "requires static full output tiles"
    if cluster_m == 2 and not (
        facts._config_is_two_cta_mma(state, config, bm=bm, cluster_m=2)
        and bm == TCGEN05_TWO_CTA_BLOCK_M
    ):
        return (
            f"requires the {TCGEN05_TWO_CTA_BLOCK_M}-row CtaGroup.TWO pair "
            "tile when tcgen05_cluster_m=2"
        )
    if region is not None:
        if config.get("pid_type") == "flat":
            # The one-shot flat launch of a chain (a GEMM left
            # ``inherit``) is not the role-local persistent schedule the
            # subtile loop rewrites render on (``memory_ops`` refuses
            # them there).
            return (
                "renders on the persistent GEMM launches of a materialized "
                "chain (a launch left flat keeps the per-subtile register reads)"
            )
        return _chain_launch_arena_refusal(
            state, config, region=region, bm=bm, bn=bn, bk=bk
        )
    return None


def _chain_launch_arena_refusal(
    state: CuteTcgen05Config,
    launch: dict[str, object],
    *,
    region: int,
    bm: int,
    bn: int,
    bk: int,
) -> str | None:
    """Why a chain's GEMM launch cannot price its epilogue rings.

    The launch's AB ring is bounded by its region's allocation proof
    (``_materialized_paired_stage_limits`` for a persistent two-CTA
    launch, ``_flat_region_ab_stage_limits`` for a flat one): the rings
    the epilogue knobs add are charged there, so a launch the proofs do
    not cover (no facts; a flat launch outside the one-CTA model's tiles)
    cannot take them.  Fails closed.
    """
    launch_facts = facts.region_launch_facts(state, region)
    if launch_facts is None:
        return "needs the GEMM launch's allocation proof"
    if launch.get("tcgen05_cluster_m", 1) == 1:
        c_stages = launch.get("tcgen05_c_stages", 2)
        acc_stages = launch.get("tcgen05_acc_stages", 2)
        if not (
            type(c_stages) is int
            and type(acc_stages) is int
            and one_cta_pipeline_tile_modeled(
                bm=bm, bn=bn, bk=bk, c_stages=c_stages, acc_stages=acc_stages
            )
        ):
            return (
                "needs the one-CTA allocation model of the chain's flat "
                "launch for its epilogue rings"
            )
    return None


def epi_subtile_refusal(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> str | None:
    """Why ``tcgen05_epi_subtile_n`` is refused for ``config``; None if admitted.

    The configured subtile is ``(per-CTA M, n)`` with ``n`` as the TMA
    store box width (``tcgen05_explicit_epilogue_tile_supported``), on
    16-bit outputs of the standard role-local TMA-store epilogue.  On a
    materialized chain every GEMM launch renders it (``chain_launches``).
    """
    value = config.get(TCGEN05_EPI_SUBTILE_N_CONFIG_KEY)
    if value is None:
        return None
    if type(value) is not int or value not in TCGEN05_EPI_SUBTILE_N_CHOICES:
        return f"must be one of {TCGEN05_EPI_SUBTILE_N_CHOICES!r}"
    if region is None and state.materialized_matmul_block_ids:
        return facts._chain_launch_refusal(
            state,
            config,
            lambda launch, launch_region: epi_subtile_refusal(
                state, launch, region=launch_region
            ),
            aux_only=False,
        )
    reason = _standard_tma_epilogue_schedule_refusal(state, config, region=region)
    if reason is not None:
        return f"the configured epilogue subtile {reason}"
    rowvec_facts = state.rowvec_aux_facts
    if rowvec_facts is None or rowvec_facts.output_itemsize != 2:
        return "the configured epilogue subtile needs 16-bit outputs"
    view = facts._matmul_config_view(state, config, region=region)
    assert view is not None
    blocks, mi, ni, _ki = view
    bm = cast("int", blocks[mi])
    bn = cast("int", blocks[ni])
    is_two_cta = config.get("tcgen05_cluster_m", 1) == 2
    epi_tile_m = tcgen05_epilogue_subtile_m(bm=bm, is_two_cta=is_two_cta)
    if not tcgen05_explicit_epilogue_tile_supported(
        is_two_cta=is_two_cta,
        bm=bm,
        bn=bn,
        tile_shape=(epi_tile_m, value, value),
    ):
        return (
            f"a ({epi_tile_m}, {value}) epilogue subtile does not partition "
            f"the {bm}x{bn} tile into whole TMA store boxes"
        )
    return None


def epi_round_subtiles_refusal(config: dict[str, object]) -> str | None:
    """Why ``tcgen05_epi_round_subtiles`` is refused for ``config``; None if
    admitted.  The round must be a choice; outside the grouped row-union
    profiles (whose normalizer sizes the C ring from it) it must divide
    ``tcgen05_c_stages``, the subtile buffers the bulk-group pipeline is
    cut into rounds of.  The store lowering refuses the remaining shapes
    (fan-out, auxiliary chains, store warps, paired tails) when it
    renders (``tcgen05_epilogue_rounds``)."""
    value = config.get(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY, 1)
    if type(value) is not int or value not in TCGEN05_EPI_ROUND_SUBTILES_CHOICES:
        return f"must be one of {TCGEN05_EPI_ROUND_SUBTILES_CHOICES!r}"
    if value == 1 or config.get(GROUPED_ROW_UNION_KEY, False) is True:
        return None
    c_stages = config.get("tcgen05_c_stages", 2)
    if type(c_stages) is not int or c_stages % value != 0:
        return f"must divide tcgen05_c_stages={c_stages!r}"
    return None


def epi_warp_groups_refusal(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> str | None:
    """Why ``tcgen05_epi_warp_groups`` is refused for ``config``; None if admitted.

    Mirrors the store lowering's gate (``memory_ops``, which fails closed
    on the same facts): the standard role-local TMA-store epilogue with
    the pre-loop or before-store C acquire, the accumulator wait inside
    the subtile loop (the groups move it ahead of their own loop), no
    staged row-vector operands (the row stages are sized per warp), the
    shared fanout on a kernel with several output stores (each group
    must walk both outputs' subtiles together) and a known subtile count.
    On a materialized chain every GEMM launch renders the groups
    (``chain_launches``).
    """
    groups = config.get(TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1)
    if type(groups) is int and groups == 1:
        return None
    if type(groups) is not int or groups not in TCGEN05_EPI_WARP_GROUP_CHOICES:
        return f"must be one of {TCGEN05_EPI_WARP_GROUP_CHOICES!r}"
    if region is None and state.materialized_matmul_block_ids:
        return facts._chain_launch_refusal(
            state,
            config,
            lambda launch, launch_region: epi_warp_groups_refusal(
                state, launch, region=launch_region
            ),
            aux_only=False,
        )
    reason = _standard_tma_epilogue_schedule_refusal(state, config, region=region)
    if reason is not None:
        return f"epilogue warp groups {reason}"
    if config.get(
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY, TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP
    ) not in (
        TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
        TCGEN05_C_ACQUIRE_PLACEMENT_BEFORE_STORE,
    ):
        return (
            "epilogue warp groups require the pre_loop or before_store C "
            "acquire placement"
        )
    if (
        config.get(
            TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
            TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
        )
        != TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
    ):
        return (
            "epilogue warp groups wait for the accumulator ahead of their own "
            f"subtile loop; they require {TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP!r}"
        )
    rowvec_facts = state.rowvec_aux_facts
    if (
        rowvec_facts is not None
        and rowvec_facts.rows
        and config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
        == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
    ):
        return (
            "epilogue warp groups do not stage row-vector operands "
            f"({TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r} with a row-vector "
            "auxiliary)"
        )
    if (
        rowvec_facts is not None
        and rowvec_facts.output_count >= 2
        and config.get(FANOUT_CONFIG_KEY, "off") != "shared"
    ):
        return (
            "epilogue warp groups on a kernel with several output stores need "
            f"the shared fanout epilogue ({FANOUT_CONFIG_KEY}='shared')"
        )
    if smem.epilogue_subtile_count(state, config, region=region) is None:
        return "epilogue warp groups need a known epilogue subtile count"
    view = facts._matmul_config_view(state, config, region=region)
    assert view is not None
    blocks, _mi, ni, _ki = view
    bn = cast("int", blocks[ni])
    if bn < tcgen05_role_launch_n_width(groups):
        # The role launch's N width (a power of two covering the ``4G +
        # 2`` warps) has to divide the N tile.
        return (
            f"{groups} epilogue warp groups launch "
            f"{tcgen05_role_launch_n_width(groups)} warp rows, more than the "
            f"{bn}-wide N tile has columns"
        )
    return None


def aux_prefetch_refusal(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> str | None:
    """Why ``tcgen05_aux_prefetch_subtiles`` is refused; None if admitted.

    The ring needs the exact-shape auxiliary operand and the
    ``pre_acc_wait`` placement the hoist needs, the standard role-local
    TMA-store epilogue, no whole-fragment hoist (which supersedes it),
    and a depth within ``aux_prefetch_depth_limit``.  On a materialized
    chain it renders on every GEMM launch with an exact-shape auxiliary
    operand (``chain_launches``).
    """
    depth = config.get(TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY, 0)
    if type(depth) is int and depth == 0:
        return None
    if type(depth) is not int or depth < 0 or depth > TCGEN05_AUX_PREFETCH_MAX_SUBTILES:
        return f"must be an integer in [0, {TCGEN05_AUX_PREFETCH_MAX_SUBTILES}]"
    if not state.exact_shape_aux_kernel_detected:
        return (
            "the auxiliary prefetch ring needs an exact-shape rank-2 "
            "auxiliary operand of the output's dtype in the epilogue chain"
        )
    if region is None and state.materialized_matmul_block_ids:
        return facts._chain_launch_refusal(
            state,
            config,
            lambda launch, launch_region: aux_prefetch_refusal(
                state, launch, region=launch_region
            ),
            aux_only=True,
        )
    if (
        config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
        != TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
    ):
        return (
            "the auxiliary prefetch ring needs "
            f"{TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}"
        )
    reason = _standard_tma_epilogue_schedule_refusal(state, config, region=region)
    if reason is not None:
        return f"the auxiliary prefetch ring {reason}"
    if config.get(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY) is True:
        return (
            "the auxiliary prefetch ring is superseded by the whole-fragment "
            f"hoist ({TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY}=True)"
        )
    limit = smem.aux_prefetch_depth_limit(state, config, region=region)
    if limit is None:
        return "the auxiliary prefetch ring needs a known epilogue subtile"
    if depth > limit:
        return (
            f"a {depth}-deep auxiliary prefetch ring holds more than "
            f"{TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD} bytes per "
            f"epilogue thread at this subtile, or reaches the group's last "
            f"subtile (deepest fitting depth {limit})"
        )
    return None


def _normalize_epilogue_groups(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Judge the epilogue subtile, warp groups and auxiliary prefetch ring.

    In order of dependence: the subtile (the groups' subtile count and
    the ring's budget read it), the groups (one C stage per group is
    legal only with two or more), the hoist at the final group count
    (its budget is the per-group share), then the ring (superseded by an
    admitted hoist, clamped to the deepest fitting depth in search).
    Explicit configs are refused with the reason.
    """
    if not state.search_enabled:
        return
    reason = epi_subtile_refusal(state, config)
    if reason is not None:
        if fix_invalid:
            config[TCGEN05_EPI_SUBTILE_N_CONFIG_KEY] = None
        else:
            raise InvalidConfig(
                f"{TCGEN05_EPI_SUBTILE_N_CONFIG_KEY}="
                f"{config.get(TCGEN05_EPI_SUBTILE_N_CONFIG_KEY)!r}: {reason}"
            )
    reason = epi_round_subtiles_refusal(config)
    if reason is not None:
        if fix_invalid:
            config.pop(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY, None)
        else:
            raise InvalidConfig(
                f"{TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY}="
                f"{config.get(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY)!r}: {reason}"
            )
    elif config.get(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY) == 1:
        config.pop(TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY, None)
    reason = epi_warp_groups_refusal(state, config)
    if reason is not None:
        if fix_invalid:
            config[TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY] = 1
        else:
            raise InvalidConfig(
                f"{TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY}="
                f"{config.get(TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY)!r}: {reason}"
            )
    groups = epi_warp_groups_knob(config)
    if config.get("tcgen05_c_stages", 2) == 1 and groups == 1:
        if fix_invalid:
            config["tcgen05_c_stages"] = 2
        else:
            raise InvalidConfig(
                "tcgen05_c_stages=1 needs two or more epilogue warp groups "
                f"({TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY} >= 2): each group owns "
                "one stage of the C ring"
            )
    if config.get(
        TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
    ) is True and not aux_fragment_hoist_admissible(state, config):
        if fix_invalid:
            config[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = False
        else:
            raise InvalidConfig(
                f"{TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY}=True holds more than "
                f"{TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD} bytes of "
                "auxiliary tile per epilogue thread at this group count"
            )
    reason = aux_prefetch_refusal(state, config)
    if reason is not None:
        if not fix_invalid:
            raise InvalidConfig(
                f"{TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY}="
                f"{config.get(TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY)!r}: "
                f"{reason}"
            )
        # A ring the budget alone refuses keeps its deepest fitting depth;
        # one the schedule refuses collapses to the per-subtile read.
        probe = dict(config)
        probe[TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY] = 1
        limit = smem.aux_prefetch_depth_limit(state, config)
        config[TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY] = (
            limit
            if limit is not None
            and (limit == 0 or aux_prefetch_refusal(state, probe) is None)
            else 0
        )


def _fix_multi_output_ring_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """Shallow a multi-output sample's rings to the whole static arena."""
    _normalize_multi_output_rings(state, config, fix_invalid=True)


def _normalize_multi_output_rings(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Judge a multi-output config's rings against the whole static arena.

    The single-output gates above calibrate AB + one C ring; a kernel
    with several output stores allocates one C ring per output
    (``Tcgen05RowvecAuxFacts.output_count``), so a config they admit can
    still overflow (128x128x64 fp16 with two outputs: ab=6 + c=4 is
    192 KiB + 64 KiB; 128x128x128 ab=3 + c=4 is 192 KiB + 64 KiB too) and
    die in the NVVM backend. Judge the final rings with
    ``default_layout_smem_fits`` (which charges every ring): a sample
    (``fix_invalid``) takes the deepest AB ring that fits, at the sampled
    C depth first and at the shallow C ring otherwise; an explicit config
    is refused with the overflow. Without a recorded SMEM budget nothing
    is proved and the config is left alone.
    """
    rowvec_facts = state.rowvec_aux_facts
    if (
        not state.search_enabled
        or state.ab_stages_three_search_constraints is None
        or rowvec_facts is None
        or rowvec_facts.output_count < 2
    ):
        return
    if config.get(
        TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
        Tcgen05LayoutStrategy.DEFAULT.value,
    ) != Tcgen05LayoutStrategy.DEFAULT.value or config.get(
        TCGEN05_GROUPED_MODE_CONFIG_KEY
    ):
        return
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        return
    block_sizes, m_index, n_index, k_index = config_view
    bm = block_sizes[m_index]
    bn = block_sizes[n_index]
    bk = block_sizes[k_index]
    cluster_m = config.get("tcgen05_cluster_m", 1)
    ab_stages = config.get("tcgen05_ab_stages", 2)
    c_stages = config.get("tcgen05_c_stages", 2)
    if not all(
        isinstance(value, int) and value > 0
        for value in (bm, bn, bk, cluster_m, ab_stages, c_stages)
    ) or cluster_m not in (1, 2):
        return

    def fits(ab: int, c: int) -> bool:
        return smem._config_arena_fits(
            state,
            config,
            bm=cast("int", bm),
            bn=cast("int", bn),
            bk=cast("int", bk),
            cluster_m=cast("int", cluster_m),
            ab_stages=ab,
            c_stages=c,
            epi_subtile_n=epi_subtile_n_knob(config),
        )

    if fits(cast("int", ab_stages), cast("int", c_stages)):
        return
    if not fix_invalid:
        raise InvalidConfig(
            f"tcgen05_ab_stages={ab_stages} with tcgen05_c_stages={c_stages} "
            f"overflows the per-CTA SMEM budget at a {bm}x{bn}x{bk} tile: the "
            f"kernel's {rowvec_facts.output_count} output stores keep one C ring each"
        )
    for c in dict.fromkeys((cast("int", c_stages), 2)):
        for ab in range(cast("int", ab_stages), 0, -1):
            if fits(ab, c):
                config["tcgen05_ab_stages"] = ab
                config["tcgen05_c_stages"] = c
                return
    # The epilogue warp groups' rings (``G x c_stages`` stages per output)
    # fit at no AB depth: one stage per group, then one group.
    if epi_warp_groups_knob(config) > 1:
        for groups, c in ((epi_warp_groups_knob(config), 1), (1, 2)):
            config[TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY] = groups
            for ab in range(cast("int", ab_stages), 0, -1):
                if fits(ab, c):
                    config["tcgen05_ab_stages"] = ab
                    config["tcgen05_c_stages"] = c
                    return


def _fix_rowvec_stage_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
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
    if not state.search_enabled:
        return
    rowvec_facts = state.rowvec_aux_facts
    if (
        rowvec_facts is None
        or not rowvec_facts.rows
        or not smem._store_lowering_stages_rows(config)
        or config.get(
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
            Tcgen05LayoutStrategy.DEFAULT.value,
        )
        != Tcgen05LayoutStrategy.DEFAULT.value
    ):
        return
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        return
    block_sizes, m_index, n_index, k_index = config_view
    bm = block_sizes[m_index]
    bn = block_sizes[n_index]
    bk = block_sizes[k_index]
    if not (isinstance(bm, int) and isinstance(bn, int) and isinstance(bk, int)):
        return
    epi_warps = epi_warps_knob(config)
    if (
        tcgen05_rowvec_stage_smem_bytes(
            rows=rowvec_facts.rows, bn=bn, epi_warps=epi_warps
        )
        == 0
    ):
        return
    cluster_m = config.get("tcgen05_cluster_m", 1)
    c_stages = config.get("tcgen05_c_stages", 2)
    if not (isinstance(cluster_m, int) and isinstance(c_stages, int)):
        return
    if not smem.rowvec_aux_stage_fits(
        state,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=_projected_ab_stages(
            state, config, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
        ),
        c_stages=c_stages,
        epi_warps=epi_warps,
        acc_stages=acc_stages_knob(config),
    ):
        config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
            TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
        )


def _projected_ab_stages(
    state: CuteTcgen05Config,
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
    fit_max = smem.max_ab_stages_that_fit(
        state, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
    )
    return min(ab_stages, fit_max) if fit_max > 0 else 3


def _fix_ab_stages_three_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if state.ab_stages_three_search_constraints is None:
        return
    if not state.search_enabled:
        return
    if config.get("tcgen05_ab_stages") != 3:
        return
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        config["tcgen05_ab_stages"] = 2
        return
    block_sizes, m_index, n_index, k_index = config_view
    cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
    if not smem.ab_stages_three_fits(
        state,
        bm=cast("int", block_sizes[m_index]),
        bn=cast("int", block_sizes[n_index]),
        bk=cast("int", block_sizes[k_index]),
        cluster_m=cluster_m,
    ):
        config["tcgen05_ab_stages"] = 2


def _fix_ab_stages_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
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
    if not state.search_enabled:
        return
    if config.get("tcgen05_ab_stages") != 3:
        return
    if not facts._is_default_layout_full_tile_config(state, config):
        return
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        config["tcgen05_ab_stages"] = 2
        return
    block_sizes, m_index, n_index, k_index = config_view
    cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
    if state.exact_shape_aux_kernel_detected:
        # Real source-C present: require AB(ab=3) + the (128, 64) C ring to fit
        # together. ``c_stages_fits`` fails CLOSED when no SMEM budget is
        # recorded (non-B200 / CPU host), so the over-budget and no-budget
        # arms are both covered by a single ``not c_stages_fits`` check.
        c_stages = cast("int", config.get("tcgen05_c_stages", 2))
        fits = smem.c_stages_fits(
            state,
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
        fits = smem.ab_stages_three_fits(
            state,
            bm=cast("int", block_sizes[m_index]),
            bn=cast("int", block_sizes[n_index]),
            bk=cast("int", block_sizes[k_index]),
            cluster_m=cluster_m,
        )
    if not fits:
        config["tcgen05_ab_stages"] = 2


def _fix_with_scheduler_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if not (
        state.search_enabled
        and (
            state.aux_kernel_detected
            or surface._plain_clc_persistence_search_enabled(state)
        )
    ):
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


def _fix_aux_tma_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA:
        return
    if not surface._aux_tma_search_enabled(state):
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
        return
    # Aux-TMA is kept when the projected cluster_m=2 candidate matches either
    # the validated edge+K-tail shape or the validated full-tile shape
    # (``_is_validated_cluster_m2_*_search_candidate``), and the strategy is
    # the ROLE_LOCAL_WITH_SCHEDULER + c-input warp combination the aux-TMA
    # producer warp requires.
    if not _is_with_scheduler_c_input_config(config):
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
        return
    if not (
        _is_validated_cluster_m2_edge_search_candidate(state, config)
        or _is_validated_cluster_m2_full_tile_search_candidate(state, config)
    ):
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT


def _fix_aux_tma_full_tile_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
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
    if not state.search_enabled:
        return
    if not surface._aux_tma_full_tile_search_enabled(state):
        return
    if config.get("tcgen05_cluster_m") != 2:
        return
    if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_EPI_TMA:
        # The epilogue-issued ring keeps its sampled depth: the SMEM
        # model (``_normalize_epi_tma``) judges it instead of this ab=2
        # producer-warp projection.
        return
    if not _is_validated_cluster_m2_full_tile_search_candidate(state, config):
        return
    if _fanout_pair_search_config_supported(state, config):
        # A CTA pair carrying the proved shared fanout is its own regime
        # (role-local monolithic SIMT aux loads, two C stages); the search
        # measures it as sampled, as ``_fix_cluster_m2_search_config`` does.
        return
    # ab=2 is the aux-TMA producer's validated stage depth for this family
    # (the aux SMEM ring forces ab<=2 under the per-CTA capacity).
    # ``_c_input_seed_config`` emits exactly this regime as a seed, so the
    # shape envelope validated above is sufficient: no extra SMEM-fit check.
    config[TCGEN05_STRATEGY_CONFIG_KEY] = (
        Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
    )
    config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
    config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
    config["tcgen05_ab_stages"] = 2
    config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
    # This family runs the deeper C-store ring: the store warp drains tile
    # N's TMA-D from the ring while the epilogue warps run tile N+1's T2R,
    # which a 2-stage ring cannot overlap.  At ab=2 the c=4 ring fits (128 KB
    # AB + 64 KB C with the (128, 64) DEFAULT epilogue tile, 16 KB per C
    # stage); ``c_stages_fits`` gates it so an ab=3 candidate of this family
    # is never lifted past the budget.
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        return
    block_sizes, m_index, n_index, k_index = config_view
    if smem.c_stages_fits(
        state,
        bm=cast("int", block_sizes[m_index]),
        bn=cast("int", block_sizes[n_index]),
        bk=cast("int", block_sizes[k_index]),
        cluster_m=2,
        ab_stages=2,
        c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
        has_source_c=True,
    ):
        config["tcgen05_c_stages"] = TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES


def _is_with_scheduler_c_input_config(config: dict[str, object]) -> bool:
    return (
        config.get(TCGEN05_STRATEGY_CONFIG_KEY)
        == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) == 1
        and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
    )


def _is_clc_aux_tma_request(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    return (
        config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
        and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        == Tcgen05PersistenceModel.CLC_PERSISTENT.value
        and surface._clc_persistence_search_enabled(state)
    )


def _is_clc_aux_tma_config(state: CuteTcgen05Config, config: dict[str, object]) -> bool:
    return _is_clc_aux_tma_request(
        state, config
    ) and _is_validated_clc_persistence_search_candidate(state, config)


def _is_clc_aux_tma_narrow_n_request(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    block_sizes = config.get("block_sizes")
    if not (
        isinstance(block_sizes, list)
        and len(block_sizes) >= 3
        and block_sizes[1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        and _is_clc_aux_tma_request(state, config)
    ):
        return False
    projected_config = dict(config)
    projected_block_sizes = list(block_sizes)
    projected_config["block_sizes"] = projected_block_sizes
    projected_config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
    projected_block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
    projected_block_sizes[2] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
    if (
        state.aux_kernel_detected
        and facts._has_any_matmul_fact_edge_tile(state, projected_config)
        and projected_config.get(TCGEN05_STRATEGY_CONFIG_KEY)
        == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
    ):
        projected_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        projected_config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
    return _is_validated_clc_persistence_search_candidate(state, projected_config)


def _validate_direct_entry_ab_stage_envelope(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if facts._dense_aligned_n_plan(state, config) is not None:
        _normalize_dense_aligned_n(state, config, fix_invalid=fix_invalid)
        return
    ab_stages = config.get("tcgen05_ab_stages")
    one_cta_limit = smem._one_cta_pipeline_stage_limit(state, config)
    if one_cta_limit is not None and type(ab_stages) is int:
        if 0 < ab_stages <= one_cta_limit:
            return
        if fix_invalid and one_cta_limit > 0:
            config["tcgen05_ab_stages"] = one_cta_limit
            return
        raise InvalidConfig(
            "one-CTA FP8 pipeline and output store ring exceed per-CTA "
            f"shared memory (maximum AB stages: {one_cta_limit})"
        )
    if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) == "two":
        fit_max = smem._paired_pipeline_stage_limit(state, config)
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
    flat_limits = (
        smem._flat_region_ab_stage_limits(state, config)
        if config.get("pid_type") == "flat"
        else None
    )
    if flat_limits is not None:
        # The flat GEMM launches of an ordered chain that keep the
        # kernel's ring share the knob; the shallowest admission among
        # them bounds it.  Where none reads it (every GEMM launch is
        # persistent or carries its own region entry) the knob is dead
        # for the chain's launches and any value is harmless.  A tile
        # that fits no ring at all falls through to the standalone
        # surface's handling below.
        readers = facts._shared_ring_flat_launches(state, config)
        if not readers:
            return
        limit = min(flat_limits[region] for region in readers)
        if limit > 0:
            if ab_stages <= limit:
                return
            if fix_invalid:
                config["tcgen05_ab_stages"] = limit
                return
            raise InvalidConfig(
                "the flat GEMM launches of the ordered hl.barrier() chain "
                f"exceed per-CTA shared memory (maximum AB stages: {limit})"
            )
    if smem._grouped_dynamic_deep_config_matches(config):
        return
    if smem._grouped_worklist_nm_ab_config_matches(state, config, ab_stages):
        return
    if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == TCGEN05_GROUPED_MODE_WORKLIST_NM:
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
                    if smem._grouped_worklist_nm_ab_config_matches(state, config, depth)
                ),
                3,
            )
            return
        mismatch = smem._grouped_worklist_nm_ab_config_mismatch(
            state, config, ab_stages
        )
        assert mismatch is not None
        raise InvalidConfig(f"tcgen05_ab_stages={ab_stages} {mismatch}")
    # After the grouped dynamic/worklist exceptions above, ab>3 is only valid
    # on the TVM-FFI direct-entry path and for its accepted (bk, ab, c) tuples
    # (bk=64 admits (ab=6, c=2) and (ab=6, c=4)). Everything else clamps or
    # rejects to ab=3.
    block_sizes = config.get("block_sizes")
    k_block_index = facts._direct_entry_k_block_index(state)
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
    constraints = state.ab_stages_three_search_constraints
    if constraints is not None:
        config_view = facts._matmul_config_view(state, config)
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if config_view is not None:
            block_sizes, m_index, n_index, k_index = config_view
            fit_max = smem.max_ab_stages_that_fit(
                state,
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
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    if not surface._clc_persistence_search_enabled(state):
        return False
    if surface._plain_clc_persistence_search_enabled(state):
        # Plain (no-aux) families: scheduler warp only, no C-input warp
        # (there is no source-C / aux producer to feed). Accept the
        # full-tile candidate shape or the edge family's projected
        # cluster_m=2 shape (bm/bn are forced to the 256x256 tile and bk
        # to a valid edge bk by _fix_cluster_m2_search_config).
        if not (
            _is_validated_cluster_m2_full_tile_search_candidate(state, config)
            or _is_validated_cluster_m2_edge_search_candidate(state, config)
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
        if state.config_spec.supports_config_key("indexing"):
            indexing = config.get("indexing")
            if (
                not isinstance(indexing, list)
                or indexing != ["tensor_descriptor"] * state.config_spec.indexing.length
            ):
                return False
        return True
    if not _is_validated_cluster_m2_edge_search_candidate(state, config):
        return False
    if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
        return False
    if config.get("tcgen05_cluster_n", 1) != 1:
        return False
    if not _is_with_scheduler_c_input_config(config):
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
        if not facts._has_any_matmul_fact_n_edge_for_block_n(
            state, TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        ):
            return False
        if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA:
            return False
        if block_sizes[2] not in (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
        ):
            return False
    elif block_sizes[2] != TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K:
        return False
    if state.config_spec.supports_config_key("indexing"):
        indexing = config.get("indexing")
        if (
            not isinstance(indexing, list)
            or indexing != ["tensor_descriptor"] * state.config_spec.indexing.length
        ):
            return False
    return True


def _fix_clc_persistence_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if (
        config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        != Tcgen05PersistenceModel.CLC_PERSISTENT.value
    ):
        return
    if _is_grouped_runtime_direct_clc_config(config):
        return
    if _is_validated_clc_persistence_search_candidate(state, config):
        return
    config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
        surface.persistence_model_default_from_config(state, config).value
    )


def _config_uses_clc_scheduler(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    """Whether ``config`` runs a scheduler warp that issues CLC queries:
    the explicit CLC persistence model, the grouped runtime-direct CLC
    worklist, or the paired row-union profile (its codegen values set the
    CLC model)."""
    if (
        config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        == Tcgen05PersistenceModel.CLC_PERSISTENT.value
    ):
        return True
    if _is_grouped_runtime_direct_clc_config(config):
        return True
    return (
        config.get(GROUPED_ROW_UNION_KEY) is True
        and config.get(GROUPED_ROW_UNION_SCHEDULE_KEY) == PAIRED_CLC_SCHEDULE
    )


def _validate_clc_query_pacing_config(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    _validate_enum_config(
        state,
        config,
        TCGEN05_CLC_QUERY_PACING_CONFIG_KEY,
        TCGEN05_CLC_QUERY_PACINGS,
        fix_invalid=fix_invalid,
    )
    if TCGEN05_CLC_QUERY_PACING_CONFIG_KEY not in config:
        return
    # The default pacing is implied: a normalized config never carries it,
    # so configs that differ only in whether a seed, a flattened search
    # point or a coverage witness spelled it out read alike (the same
    # convention as the startup prefill's default).  The non-default
    # pacing stays only where a CLC scheduler reads it; elsewhere it is
    # a contradiction the search repairs and a user config may not make.
    if config[TCGEN05_CLC_QUERY_PACING_CONFIG_KEY] == TCGEN05_CLC_QUERY_PACING_TILE:
        config.pop(TCGEN05_CLC_QUERY_PACING_CONFIG_KEY, None)
    elif not _config_uses_clc_scheduler(state, config):
        if fix_invalid:
            config.pop(TCGEN05_CLC_QUERY_PACING_CONFIG_KEY, None)
        else:
            raise InvalidConfig(
                f"{TCGEN05_CLC_QUERY_PACING_CONFIG_KEY} requires a kernel whose "
                "scheduler warp issues CLC queries ("
                f"{TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY}="
                f"{Tcgen05PersistenceModel.CLC_PERSISTENT.value!r} or the "
                f"{PAIRED_CLC_SCHEDULE!r} row-union profile)"
            )


def _validate_sched_stage_count_config(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    _validate_int_enum_config(
        state,
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
    )
    # The single-CTA N,M grouped worklist publishes per-stage mailboxes,
    # so its TMA producer can fetch work tiles ahead of the epilogue.
    is_grouped_worklist_nm_one_cta = (
        config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and config.get("tcgen05_cluster_m", 1) == 1
        and config.get("tcgen05_cluster_n", 1) == 1
        and config.get(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False) is False
    )
    if (
        TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY in config
        and config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY) != 1
        and not is_grouped_worklist_nm_one_cta
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


def _fix_aux_edge_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if n_edge_cluster_schedule_config_supported(state, config):
        return
    if not (state.search_enabled and state.aux_kernel_detected):
        return
    if not facts._has_any_matmul_fact_edge_tile(state, config):
        return

    if _is_validated_cluster_m2_edge_search_candidate(state, config):
        _set_aux_edge_cluster_m2_prefix(config)
        return

    _set_aux_edge_monolithic_prefix(config)
    config["tcgen05_acc_stages"] = 2
    config["tcgen05_c_stages"] = 4
    config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
    if isinstance(config.get("l2_groupings"), list):
        config["l2_groupings"] = [1] * len(state.config_spec.l2_groupings)
    if isinstance(config.get("indexing"), list):
        indexing = cast("list[object]", config["indexing"])
        for i in range(len(indexing)):
            indexing[i] = "pointer"
    block_sizes = config.get("block_sizes")
    if not isinstance(block_sizes, list):
        return
    for fact in state.config_spec.matmul_facts:
        if fact.static_m is not None and fact.m_block_id is not None:
            try:
                m_idx = state.config_spec.block_sizes.block_id_to_index(fact.m_block_id)
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


def _set_aux_edge_monolithic_prefix(config: dict[str, object]) -> None:
    config[TCGEN05_STRATEGY_CONFIG_KEY] = Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
    config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
    config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
    if config.get("tcgen05_ab_stages") == 1:
        config["tcgen05_ab_stages"] = 2
    config.pop("epilogue_subtile", None)


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
        _set_aux_edge_monolithic_prefix(config)
        return
    if config.get("tcgen05_ab_stages") == 1:
        config["tcgen05_ab_stages"] = 2
    config.pop("epilogue_subtile", None)


def _is_validated_cluster_m2_edge_search_candidate(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    if config.get("tcgen05_cluster_m") != 2:
        return False
    constraints = state.cluster_m2_search_constraints
    if constraints is None:
        return False
    return constraints.allow_edge_k_tail_family


def _is_validated_cluster_m2_full_tile_search_candidate(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    # Cycle 46: full-tile cluster_m=2 candidate gate for aux-TMA admission.
    # After ``_fix_cluster_m2_search_config`` projects a full-tile sample to
    # the canonical 256x256x bk shape with ``persistent_interleaved`` pid,
    # this returns True so aux-TMA stays during the search-time fixup.
    # bk validity is already enforced by ``_fix_cluster_m2_search_config``;
    # we re-check it here so a stale/unprojected config cannot slip through.
    if config.get("tcgen05_cluster_m") != 2:
        return False
    constraints = state.cluster_m2_search_constraints
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
    config_view = facts._matmul_config_view(state, config)
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
    return cluster_m2_bk_is_valid(bk, constraints)


def _fix_cluster_m1_persistent_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if not (
        state.search_enabled
        and config.get("tcgen05_cluster_m", 1) == 1
        and config.get("pid_type") in {"persistent_blocked", "persistent_interleaved"}
    ):
        return
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        return
    block_sizes, m_index, n_index, k_index = config_view
    if (
        config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == TCGEN05_GROUPED_MODE_WORKLIST_NM
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
    constraints = state.cluster_m2_search_constraints
    if constraints is not None and constraints.allow_edge_k_tail_family:
        # persistent_interleaved stays in the flat enum so cluster_m=2
        # edge-family samples can encode; cluster_m=1 samples from the
        # same surface must use the validated flat edge fallback.
        config["pid_type"] = "flat"
        return
    bm = block_sizes[m_index]
    if isinstance(bm, int) and not isinstance(bm, bool):
        block_sizes[m_index] = bm = min(bm, TCGEN05_ONE_CTA_MAX_BLOCK_M)
    bn, bk = block_sizes[n_index], block_sizes[k_index]
    if (
        state.output_edge_persistent_admitted
        and isinstance(bm, int)
        and isinstance(bn, int)
        and isinstance(bk, int)
        and not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
    ):
        # The output-edge admission's persistent schedules belong to its
        # CtaGroup.TWO pairs; one-CTA persistent kernels run whole tiles
        # only (the host multi-tile guard), so partial tiles go flat.
        config["pid_type"] = "flat"


def _fix_target1_tvm_ffi_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    # The generalized direct-entry seed projects FFI-requesting search
    # candidates onto the validated CtaGroup.TWO envelope for ANY eligible
    # shape (returns None for ineligible shapes, in which case the
    # promotion surface is stripped back to the DEFAULT layout below).
    seed = seeds.full_tile_direct_entry_seed_config(
        state, l2_groupings=cast("list[object] | None", config.get("l2_groupings"))
    )
    if not surface._target1_tvm_ffi_promotion_requested(config):
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
    surface._clear_target1_tvm_ffi_promotion_surface(config)
    config.update(seed.config)


def epi_tma_refusal(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> str | None:
    """Why ``config`` cannot take the epilogue-issued aux ring, or ``None``.

    Mirrors the codegen gate in ``cute_mma`` (which fails closed on the
    same facts) so a sample is repaired and an explicit config refused
    before a compile: an exact-shape auxiliary operand, static full tiles
    of the config's blocks, the default-layout normal TMA-store epilogue,
    no producer warp, the CtaGroup.TWO pair for a clustered config (not
    its ``bm=128`` tile), a rank-2 row-major output and the plain K loop.
    On a materialized chain the ring renders on every GEMM launch with an
    exact-shape auxiliary operand (``chain_launches``), each judged as
    ``region_launch_config`` projects it; its AB ring is priced beside
    the aux rings by the region budgets.
    """
    if not state.exact_shape_aux_kernel_detected:
        return (
            "requires an exact-shape rank-2 auxiliary operand of the output's "
            "dtype in the epilogue chain"
        )
    if not state.exact_aux_tma_provable:
        return TCGEN05_AUX_TMA_ILLEGAL_TENSOR_REASON
    if state.matmul_has_leading_passthrough:
        return "requires a rank-2 matmul"
    if region is None and state.materialized_matmul_block_ids:
        return facts._chain_launch_refusal(
            state,
            config,
            lambda launch, launch_region: epi_tma_refusal(
                state, launch, region=launch_region
            ),
            aux_only=True,
        )
    view = facts._matmul_config_view(state, config, region=region)
    if view is None:
        return "requires the tcgen05 matmul block view"
    blocks, mi, ni, ki = view
    bm, bn, bk = blocks[mi], blocks[ni], blocks[ki]
    if not all(type(value) is int and value > 0 for value in (bm, bn, bk)):
        return "requires integer tcgen05 block sizes"
    bm, bn, bk = cast("int", bm), cast("int", bn), cast("int", bk)
    if not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk, region=region):
        return "requires static full output tiles"
    if config.get(
        TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, Tcgen05LayoutStrategy.DEFAULT.value
    ) != Tcgen05LayoutStrategy.DEFAULT.value or any(
        config.get(key) is not None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS
    ):
        return "requires the default epilogue layout"
    if (
        config.get(TCGEN05_C_STORE_MODE_CONFIG_KEY, TCGEN05_C_STORE_MODE_NORMAL)
        != TCGEN05_C_STORE_MODE_NORMAL
        or config.get(TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY, "normal") != "normal"
    ):
        return "requires the normal TMA-store epilogue"
    if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is not None:
        return "is not available for grouped matmuls"
    if (
        config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        == Tcgen05PersistenceModel.CLC_PERSISTENT.value
    ):
        return "is not validated with CLC persistence"
    if any(
        config.get(key, 0) != 0
        for key in (
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
            TCGEN05_WARP_SPEC_STORE_WARPS_KEY,
        )
    ):
        return (
            "replaces the aux producer warp: requires "
            "tcgen05_warp_spec_c_input_warps=0 and tcgen05_warp_spec_store_warps=0"
        )
    cluster_m = config.get("tcgen05_cluster_m", 1)
    if cluster_m == 2:
        if not facts._config_is_two_cta_mma(state, config, bm=bm, cluster_m=2):
            return "requires the CtaGroup.TWO pair when tcgen05_cluster_m=2"
        if bm != TCGEN05_TWO_CTA_BLOCK_M:
            return "is not validated on the bm=128 CtaGroup.TWO epilogue tile"
    pipeline_facts = facts._launch_pipeline_facts(state, region)
    if pipeline_facts is not None and pipeline_facts.output_column_major:
        return "requires a row-major output"
    if (
        config.get(FANOUT_CONFIG_KEY, "off") not in ("off", "shared")
        or config.get("cute_materialized_operand_schedule", "off") != "off"
        or config.get("cute_materialized_schedule", "off") != "off"
        or config.get("cute_split_k_workspace")
        or config.get("cute_split_k_schedule", "legacy") != "legacy"
    ):
        return "requires the plain K loop of one native GEMM"
    if region is not None:
        return _chain_launch_arena_refusal(
            state, config, region=region, bm=bm, bn=bn, bk=bk
        )
    return None


def _normalize_epi_tma(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``tcgen05_aux_load_mode="epi_tma"`` renders only inside its envelope.

    A sample outside it (``fix_invalid``) falls back to the register
    (SIMT) aux reads; one whose AB ring does not fit beside the aux rings
    keeps the mode at the deepest fitting ring depth; one carrying a
    C-input or store warp beside the ring keeps the ring and drops the
    warp (the ring replaces the producer warp). An explicit config is
    refused with the reason.

    Runs from ``normalize_pre_pid_type``, ahead of ``fix_search_config``:
    the warp has to be dropped here, before the aux-TMA full-tile
    projection (``_fix_aux_tma_full_tile_search_config``) would see a
    repaired ``simt`` sample and project it onto the producer-warp
    regime.
    """
    if (
        state.materialized_matmul_block_ids
        and config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
    ):
        # The producer-warp ring needs the scheduler / C-input warps,
        # which no launch of a materialized chain runs; the chain's
        # staged aux reads are the epilogue-issued ring.
        if fix_invalid:
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
            return
        raise InvalidConfig(
            f"{TCGEN05_AUX_LOAD_MODE_CONFIG_KEY}={TCGEN05_AUX_LOAD_MODE_TMA!r} "
            "(the producer-warp auxiliary ring) is not available on a "
            f"materialized fission chain; use {TCGEN05_AUX_LOAD_MODE_EPI_TMA!r}"
        )
    if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_EPI_TMA:
        return
    if fix_invalid:
        for key in (
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
            TCGEN05_WARP_SPEC_STORE_WARPS_KEY,
        ):
            if config.get(key, 0) != 0:
                config[key] = 0
    reason = epi_tma_refusal(state, config)
    if reason is None:
        limit = smem.epi_tma_ab_stage_limit(state, config)
        ab_stages = config.get("tcgen05_ab_stages", 2)
        if limit is None or (type(ab_stages) is int and 0 < ab_stages <= limit):
            return
        if limit > 0 and fix_invalid:
            config["tcgen05_ab_stages"] = limit
            return
        reason = (
            f"needs tcgen05_ab_stages={ab_stages} beside its auxiliary rings, "
            f"but at most {limit} AB stages fit the per-CTA SMEM budget"
        )
    if fix_invalid:
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
        return
    raise InvalidConfig(
        f"{TCGEN05_AUX_LOAD_MODE_CONFIG_KEY}={TCGEN05_AUX_LOAD_MODE_EPI_TMA!r} {reason}"
    )


def _normalize_grouped_static_signature_raster(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """A static grouped execution signature walks the plain raster.

    The static grouped scheduler has no L2 swizzle or L2 grouping decode;
    the ``cute_mma`` pipeline reads both knobs from the config, so they
    are judged here: a sample (``fix_invalid``) is reset to the plain
    raster, an explicit config is refused.
    """
    if config.get(TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY) is None:
        return
    if config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 1) != 1:
        if not fix_invalid:
            raise InvalidConfig(
                f"{TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY} requires "
                "tcgen05_l2_swizzle_size=1"
            )
        config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
    groupings = config.get("l2_groupings")
    if isinstance(groupings, list) and any(g != 1 for g in groupings):
        if not fix_invalid:
            raise InvalidConfig(
                f"{TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY} requires "
                "l2_groupings entries to all equal 1"
            )
        config["l2_groupings"] = [1] * len(groupings)


def _normalize_aux_producer_ab_stages(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """A productive aux producer warp keeps ``tcgen05_ab_stages`` at 2.

    The exact-shape auxiliary ring the producer warp (the C-input warp, or
    the store warp under ``tcgen05_aux_load_mode="tma"``) stages next to
    the AB ring overshoots the per-CTA SMEM cap from three AB stages up.
    Judged from the config keys and the bind-time exact-shape auxiliary
    fact the ``cute_mma`` gate reads: a sample (``fix_invalid``) drops the
    producer warp (``_fix_with_scheduler_search_config``'s rule), an
    explicit config is refused.  The pipeline keeps the gate for the
    descriptors the fact does not count (an auxiliary operand of another
    dtype).
    """
    ab_stages = config.get("tcgen05_ab_stages")
    c_input_warp = config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY, 0) == 1
    store_warp_aux_tma = (
        config.get(TCGEN05_WARP_SPEC_STORE_WARPS_KEY, 0) == 1
        and config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
    )
    if not (
        state.exact_shape_aux_kernel_detected
        and type(ab_stages) is int
        and ab_stages >= 3
        and (c_input_warp or store_warp_aux_tma)
    ):
        return
    if fix_invalid:
        config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        if store_warp_aux_tma:
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
        return
    raise InvalidConfig(
        "tcgen05 ``tcgen05_ab_stages=3`` is incompatible "
        "with a productive aux producer warp "
        "(``tcgen05_warp_spec_c_input_warps=1``, or the "
        "cycle-94 store-warp merge "
        "``tcgen05_warp_spec_store_warps=1`` + "
        "``tcgen05_aux_load_mode=tma``, + "
        "non-empty aux tensors from the epilogue chain): "
        "the aux SMEM ring + AB pipeline together "
        "overshoot the 232 KB B200 SMEM cap at every "
        "validated tile shape (the canonical "
        "``(bm=bn=256, bk=128, cluster_m=2)`` shape uses "
        "263 KB vs 232 KB cap). Drop to "
        "``tcgen05_ab_stages=2`` for residual epilogues "
        "with an aux producer warp, or drop the aux "
        "producer warp to keep ``tcgen05_ab_stages=3``. See "
        "``cute_plan.md`` §1.3 / §7.5.3.2."
    )


def validate_strategy_invariants(
    state: CuteTcgen05Config,
    config: dict[str, object],
    *,
    fix_invalid: bool,
) -> None:
    strategy = Tcgen05Strategy(config[TCGEN05_STRATEGY_CONFIG_KEY])
    persistence_model = Tcgen05PersistenceModel(
        config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
    )
    layout_strategy = Tcgen05LayoutStrategy(config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY])
    warp_spec = warp_spec_from_config(config)
    layout_overrides = layout_overrides_from_config(config)

    cluster_m_raw = config.get("tcgen05_cluster_m", 1)
    cluster_m = int(cluster_m_raw) if isinstance(cluster_m_raw, int) else 1
    cluster_n_raw = config.get("tcgen05_cluster_n", 1)
    cluster_n = int(cluster_n_raw) if isinstance(cluster_n_raw, int) else 1
    capability = state.config_spec.target_device_capability
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
                config[key] = surface.strategy_field_default(key, pid_type=pid_type)
        return
    message = "; ".join(errors)
    raise InvalidConfig(f"tcgen05 strategy invariants violated: {message}")


def _clamp_l2_swizzle_size_to_shape(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY in config:
        # A grouped signature has no single N extent to clamp against.
        # Preserve the requested value so lowering can reject swizzling
        # that the static grouped scheduler does not support.
        return
    if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == TCGEN05_GROUPED_MODE_WORKLIST_NM:
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
    indices = facts._matmul_block_indices(state)
    if indices is not None:
        n_axes = [(indices[1], state.config_spec.block_sizes[indices[1]].size_hint)]
    elif (
        state.region_pid_type_search_enabled
        and state.materialized_matmul_block_ids
        and any(
            pid_type != "flat" for pid_type in surface._region_pid_types(state, config)
        )
    ):
        # The persistent GEMM launches of an ordered chain raster under
        # the shared swizzle as the standalone persistent pipeline does;
        # the narrowest launch bounds it.  A flat launch has no tile
        # scheduler to swizzle, so a chain whose GEMM launches are all
        # flat keeps swizzle 1, as does a fission plan.
        n_axes = []
        for axes, shape in zip(
            state.materialized_matmul_block_ids,
            state.materialized_matmul_shapes,
            strict=True,
        ):
            n_block_index = facts._config_block_index(state, axes[1])
            if n_block_index is None:
                config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
                return
            n_axes.append((n_block_index, shape[1]))
    else:
        config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
        return
    block_sizes = config.get("block_sizes")
    if not isinstance(block_sizes, list):
        return
    cluster_n_raw = config.get("tcgen05_cluster_n", 1)
    cluster_n = (
        cluster_n_raw
        if isinstance(cluster_n_raw, int) and not isinstance(cluster_n_raw, bool)
        else 1
    )
    cluster_n = max(cluster_n, 1)
    ncluster_n: int | None = None
    for n_block_index, n_hint in n_axes:
        if n_block_index >= len(block_sizes):
            return
        bn = block_sizes[n_block_index]
        if not isinstance(bn, int) or isinstance(bn, bool) or bn <= 0:
            return
        if n_hint <= 0:
            return
        clusters = max(((n_hint + bn - 1) // bn) // cluster_n, 1)
        ncluster_n = clusters if ncluster_n is None else min(ncluster_n, clusters)
    assert ncluster_n is not None
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
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    reserved_sms_key = TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
    if reserved_sms_key in config and not state.search_enabled:
        if fix_invalid:
            config.pop(reserved_sms_key, None)
        else:
            raise InvalidConfig(
                f"{reserved_sms_key} is only supported for tcgen05-enabled "
                "CuTe matmul kernels"
            )
    optional_fragments = surface.optional_fragments(state)
    optional_search_fragments = surface.optional_fragments(state, for_search=True)
    if state.search_enabled:
        for key, fragment in optional_fragments.items():
            if key in config:
                if key == "tcgen05_ab_stages" and (
                    smem._grouped_dynamic_deep_config_matches(config)
                    or smem._grouped_worklist_nm_ab_config_matches(
                        state,
                        config,
                        config[key],
                    )
                ):
                    config[key] = int(cast("Any", config[key]))
                else:
                    config[key] = surface._validate_optional_fragment_value(
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
        _normalize_grouped_ab_tile_k(state, config, fix_invalid=fix_invalid)
        _normalize_grouped_d_shift(state, config, fix_invalid=fix_invalid)
        _normalize_grouped_balanced_workers(state, config, fix_invalid=fix_invalid)
        _normalize_grouped_prefix_adaptive_n(state, config, fix_invalid=fix_invalid)
        _normalize_grouped_panel_adaptation(state, config, fix_invalid=fix_invalid)
        _clamp_l2_swizzle_size_to_shape(state, config)
        _validate_cta_group_config(state, config, fix_invalid=fix_invalid)
        _normalize_region_launches(state, config, fix_invalid=fix_invalid)
        _validate_direct_entry_ab_stage_envelope(state, config, fix_invalid=fix_invalid)
    else:
        _normalize_region_launches(state, config, fix_invalid=fix_invalid)
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
            # ``test_batched_two_cta_partial_edge_tiles_projected_onto_one_cta``).
            if fix_invalid or config[key] == fragment.default():
                config.pop(key, None)
            else:
                raise InvalidConfig(
                    f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
                )

    strategy_validation_fragments = surface.strategy_validation_fragments(state)
    if not state.search_enabled:
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

    _validate_enum_config(
        state,
        config,
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
        TCGEN05_C_ACQUIRE_PLACEMENTS,
        fix_invalid=fix_invalid,
    )
    _validate_enum_config(
        state,
        config,
        TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
        TCGEN05_ACC_WAIT_PLACEMENTS,
        fix_invalid=fix_invalid,
    )
    _validate_enum_config(
        state,
        config,
        TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
        TCGEN05_AUX_LOAD_PLACEMENTS,
        fix_invalid=fix_invalid,
    )
    aux_load_placement = config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
    if (
        aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        and not state.aux_kernel_detected
    ):
        # No auxiliary loads anywhere: the placement is never offered to
        # this kernel's search (``optional_fragments``), so an explicit or
        # recorded ``pre_acc_wait`` is a config error under both
        # normalization modes, not a sample to repair.
        raise InvalidConfig(
            f"invalid {TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
            f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}: the kernel has "
            "no per-subtile auxiliary loads to place"
        )
    if (
        aux_load_placement == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        and not pre_acc_wait_admissible(state, config)
    ):
        # Some store chain has no per-subtile auxiliary loads and no
        # shared fanout merges it with one that has: the store lowering
        # refuses the placement at that store. The search offers the
        # placement here (the fanout can admit it), so a sampled one is
        # repaired to the default; an explicit one is refused here rather
        # than after a compile.
        if fix_invalid:
            config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
            )
            aux_load_placement = TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
        else:
            raise InvalidConfig(
                f"invalid {TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
                f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}: a store of "
                "the kernel has no per-subtile auxiliary loads to place "
                "(every store chain must load an auxiliary operand, or "
                f"{FANOUT_CONFIG_KEY}='shared' must merge the chains)"
            )
    hoist = config.get(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY, False)
    if hoist is not False and hoist is not True:
        if fix_invalid:
            config[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = False
        else:
            raise InvalidConfig(
                f"{TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY} must be a boolean, "
                f"got {hoist!r}"
            )
    elif hoist and not aux_fragment_hoist_admissible(state, config):
        if fix_invalid:
            config[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = False
        else:
            raise InvalidConfig(
                f"{TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY}=True requires an "
                "exact-shape auxiliary operand, "
                f"{TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY}="
                f"{TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT!r}, the SIMT aux "
                "load mode, the normal TMA store and a tile fragment of at "
                f"most {TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD} "
                "bytes per epilogue thread (per epilogue warp group)"
            )
    _normalize_epilogue_groups(state, config, fix_invalid=fix_invalid)
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
    _validate_enum_config(
        state,
        config,
        TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
        TCGEN05_AUX_LOAD_MODES,
        fix_invalid=fix_invalid,
    )
    _validate_int_enum_config(
        state,
        config,
        TCGEN05_AUX_STAGES_CONFIG_KEY,
        TCGEN05_AUX_STAGE_COUNT_VALIDATION_CHOICES,
        fix_invalid=fix_invalid,
    )
    # Only the epilogue-issued ring prices the deeper stage counts; the
    # producer-warp ring keeps its validated depths.
    aux_stages = config.get(TCGEN05_AUX_STAGES_CONFIG_KEY)
    if (
        type(aux_stages) is int
        and aux_stages not in TCGEN05_AUX_STAGE_COUNT_CHOICES
        and config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
        != TCGEN05_AUX_LOAD_MODE_EPI_TMA
    ):
        if fix_invalid:
            config[TCGEN05_AUX_STAGES_CONFIG_KEY] = max(TCGEN05_AUX_STAGE_COUNT_CHOICES)
        else:
            raise InvalidConfig(
                f"{TCGEN05_AUX_STAGES_CONFIG_KEY}={aux_stages} is only "
                f"available with {TCGEN05_AUX_LOAD_MODE_CONFIG_KEY}="
                f"{TCGEN05_AUX_LOAD_MODE_EPI_TMA!r}; the producer-warp ring "
                f"takes {TCGEN05_AUX_STAGE_COUNT_CHOICES!r}"
            )
    _normalize_epi_tma(state, config, fix_invalid=fix_invalid)
    _validate_int_enum_config(
        state,
        config,
        TCGEN05_CONSUMER_REGS_CONFIG_KEY,
        TCGEN05_CONSUMER_REGS_CHOICES,
        fix_invalid=fix_invalid,
    )
    _validate_bool_config(
        state,
        config,
        TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
        fix_invalid=fix_invalid,
    )

    _validate_diagnostic_mode(
        state,
        config,
        TCGEN05_C_STORE_MODE_CONFIG_KEY,
        TCGEN05_C_STORE_MODES,
        # The register-direct store is an exact alternative body, not a
        # diagnostic: it needs no invalid-output opt-in.
        (TCGEN05_C_STORE_MODE_NORMAL, TCGEN05_C_STORE_MODE_DIRECT),
        fix_invalid=fix_invalid,
    )

    _validate_bool_config(
        state, config, TCGEN05_CUBIN_LINEINFO_CONFIG_KEY, fix_invalid=fix_invalid
    )
    _validate_bool_config(
        state, config, TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY, fix_invalid=fix_invalid
    )
    _validate_bool_config(
        state, config, TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, fix_invalid=fix_invalid
    )
    _validate_bool_config(
        state,
        config,
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
        fix_invalid=fix_invalid,
    )
    if config.get(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY) is True:
        proof_envelope_matches = (
            tuple(cast("list[int]", config.get("block_sizes", [])))
            == TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
            and config.get("tcgen05_cluster_m", 1) == TCGEN05_LARGE_BN_PROOF_CLUSTER_M
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
    _validate_enum_config(
        state,
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
            config[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY] = TCGEN05_EPILOGUE_LAYOUT_NORMAL
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
                config[TCGEN05_C_STORE_MODE_CONFIG_KEY] = TCGEN05_C_STORE_MODE_NORMAL
            else:
                raise InvalidConfig(
                    f"{other_protocol} owns the output store; the register-direct "
                    f"{TCGEN05_C_STORE_MODE_CONFIG_KEY}="
                    f"{TCGEN05_C_STORE_MODE_DIRECT!r} is the plain store protocol"
                )
    _validate_enum_config(
        state,
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
    _validate_sched_stage_count_config(state, config, fix_invalid=fix_invalid)
    _validate_clc_query_pacing_config(state, config, fix_invalid=fix_invalid)


def _reject_config_key(
    config: dict[str, object], key: str, message: str, *, fix_invalid: bool
) -> None:
    """Drop ``key`` under ``fix_invalid``; otherwise refuse the config."""
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(message)


def _validate_enum_config(
    state: CuteTcgen05Config,
    config: dict[str, object],
    key: str,
    choices: tuple[str, ...],
    *,
    fix_invalid: bool,
) -> None:
    if not surface._tcgen05_knob_set(state, config, key, fix_invalid=fix_invalid):
        return
    if config[key] not in choices:
        _reject_config_key(
            config,
            key,
            f"{key} must be one of {choices!r}, got {config[key]!r}",
            fix_invalid=fix_invalid,
        )


def _validate_int_enum_config(
    state: CuteTcgen05Config,
    config: dict[str, object],
    key: str,
    choices: tuple[int, ...],
    *,
    fix_invalid: bool,
) -> None:
    if not surface._tcgen05_knob_set(state, config, key, fix_invalid=fix_invalid):
        return
    value = config[key]
    # ``bool`` is an ``int`` subclass, but it is not a valid stage count.
    if type(value) is not int or value not in choices:
        _reject_config_key(
            config,
            key,
            f"{key} must be one of {choices!r}, got {value!r}",
            fix_invalid=fix_invalid,
        )


def _validate_bool_config(
    state: CuteTcgen05Config,
    config: dict[str, object],
    key: str,
    *,
    fix_invalid: bool,
) -> None:
    if not surface._tcgen05_knob_set(state, config, key, fix_invalid=fix_invalid):
        return
    if not isinstance(config[key], bool):
        _reject_config_key(
            config, key, f"{key} must be a boolean", fix_invalid=fix_invalid
        )


def _validate_diagnostic_mode(
    state: CuteTcgen05Config,
    config: dict[str, object],
    key: str,
    modes: tuple[str, ...],
    exact_modes: tuple[str, ...],
    *,
    fix_invalid: bool,
) -> None:
    """Validate a mode knob; ``exact_modes`` lists the non-diagnostic values."""
    if not surface._tcgen05_knob_set(state, config, key, fix_invalid=fix_invalid):
        return
    if config[key] not in modes:
        _reject_config_key(
            config,
            key,
            f"{key} must be one of {modes!r}, got {config[key]!r}",
            fix_invalid=fix_invalid,
        )
        return
    if (
        config[key] not in exact_modes
        and config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY) is not True
    ):
        _reject_config_key(
            config,
            key,
            f"{key}={config[key]!r} changes output correctness; set "
            f"{TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY}=True "
            "only for diagnostic invalid-output runs",
            fix_invalid=fix_invalid,
        )


def fix_search_config(state: CuteTcgen05Config, config: dict[str, object]) -> None:
    _fix_search_config_body(state, config)
    _drop_unproved_n_edge_cluster_n(state, config)
    # These repairs read the repaired pid type, cluster and tile, so they
    # run last, on every path of the body (its early returns included:
    # the one-CTA pipeline, the dense-N tile, the materialized chains
    # and the explicit pair): the persistent grid cap is inert once a
    # projection drops the cluster to one N tile, and the split-K knobs
    # apply only on the flat one-CTA family.
    _fix_persistent_grid_cap_search_config(state, config)
    _normalize_split_k(state, config, fix_invalid=True)
    _fix_split_k_workspace_search_config(state, config)
    # The L2 swizzle was clamped in ``normalize_pre_pid_type`` against the
    # sampled tile; the projections above may have widened the N tile or
    # dropped the N cluster (the canonical 256x256 pair, the explicit
    # pair's 128-wide tile), leaving a swizzle wider than the projected
    # N tile-cluster count.  Strict normalization clamps it again on
    # entry, so the repaired config has to carry the clamped value to be
    # a fixed point of ``normalize(fix_invalid=False)``.
    _clamp_l2_swizzle_size_to_shape(state, config)


def _apply_paired_pipeline_surface(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """The canonical knobs of a sampled explicit CTA pair: the role-local
    static-persistent pipeline on the default layout with the
    direct-entry controls off (in the search schema's representation, so
    inactive controls cannot survive only one side of flatten) and no
    scheduler / C-input / store warps."""
    config["tcgen05_cluster_m"] = 2
    config[TCGEN05_STRATEGY_CONFIG_KEY] = Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
    config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
        Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )
    config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = Tcgen05LayoutStrategy.DEFAULT.value
    search_fields = surface.optional_fragments(state, for_search=True)
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


def _fix_search_config_body(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    if config.get(TILE_N_KEY, 0) != 0:
        _normalize_dense_aligned_n(state, config, fix_invalid=True)
        config.pop("epilogue_subtile", None)
        return
    _fix_n_edge_cluster_schedule_search_config(state, config)
    _fix_full_tile_schedule_search_config(state, config)
    if seeds._one_cta_pipeline_parameters(state, config) is not None:
        config.pop("epilogue_subtile", None)
        _validate_direct_entry_ab_stage_envelope(state, config, fix_invalid=True)
        return
    if smem._materialized_paired_stage_limits(state, config) is not None:
        # Each launch has its own proved axes and budget. The single-matrix
        # projection below cannot choose geometry for this complete recipe.
        _validate_cta_group_config(state, config, fix_invalid=True)
        return
    if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) == "two":
        explicit_pair = surface.paired_pipeline_search_enabled(state)
        if not explicit_pair and plain_edge_pair128_admitted(state):
            # The ragged edge family's explicit pair is the 128-row tile
            # only; a sampled "two" on a wider tile is reset to ``auto``
            # and repaired by the body as before (its 256-row pair is the
            # family's own ``auto`` encoding).
            view = facts._matmul_config_view(state, config)
            explicit_pair = view is not None and view[0][view[1]] == 128
        if explicit_pair:
            _apply_paired_pipeline_surface(state, config)
            _fix_cluster_m2_search_config(state, config)
            _clamp_l2_swizzle_size_to_shape(state, config)
            _validate_cta_group_config(state, config, fix_invalid=True)
            _validate_direct_entry_ab_stage_envelope(state, config, fix_invalid=True)
            return
        config[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
    _fix_grouped_worklist_search_config(state, config)
    _fix_aux_edge_search_config(state, config)
    _fix_cluster_m2_search_config(state, config)
    _fix_cluster_m1_persistent_search_config(state, config)
    if (
        state.search_enabled
        and state.matmul_input_dtype is not None
        and state.matmul_input_dtype.itemsize <= 2
        # A 16-bit MMA fed by a cast from a non-native load (bf16 x int16)
        # takes the non-tcgen05 fallback and keeps the knob; fp8 operands
        # trip the same flag but run on tcgen05.
        and not (
            state.matmul_has_non_tcgen05_operand
            and state.matmul_input_dtype.itemsize == 2
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
        state.search_enabled
        and config.get("tcgen05_cluster_m", 1) == 1
        and config.get("tcgen05_cluster_n", 1) == 2
    ):
        # The validated N multicast needs the two-CTA M family. A
        # one-CTA sample must retain its geometry with no N multicast;
        # explicit configurations still go through the unchanged guard.
        config["tcgen05_cluster_n"] = 1
    _fix_ab_stages_three_search_config(state, config)
    _fix_target1_tvm_ffi_search_config(state, config)
    _fix_aux_tma_full_tile_search_config(state, config)
    _fix_with_scheduler_search_config(state, config)
    _fix_aux_tma_search_config(state, config)
    # Cycle 97: budget-aware ab-stages admission for the lifted for_search
    # cap. Runs after the family projections (FFI / gelu / aux-TMA full-tile)
    # have set their validated stage tuple, so a directly-SAMPLED ab=3 that no
    # projection claimed — and that does not fit (residual/source-C ring
    # overflow, or cluster_m=1 256x256 bare-AB overflow) — is demoted to 2 on
    # the DEFAULT-layout full-tile path before codegen. The plain silu/gelu
    # ab=3 winner (no source-C, cluster_m=2) is preserved.
    _fix_ab_stages_search_config(state, config)
    # Final c-stages admission: demote a directly-sampled (or any unclaimed)
    # deeper C ring on the canonical 256x256 DEFAULT-layout path that does
    # not fit AB+C under the B200 cap. Runs after the family fixups so their
    # validated c=4 (residual ab=2 projection above; edge/seed families,
    # which this gate's scope excludes) is set first.
    _fix_c_stages_search_config(state, config)
    # A sampled ``pre_acc_wait`` stages its rows next to the final rings;
    # demote it where the stage does not fit (the seeds' gate, applied to
    # the projection).
    _fix_rowvec_stage_search_config(state, config)
    # The epilogue subtile, warp groups and auxiliary prefetch ring are
    # re-judged on the projected geometry ahead of the ring pricing they
    # feed (a (128, 32) subtile halves each ring; a second group doubles
    # the stages).
    _normalize_epilogue_groups(state, config, fix_invalid=True)
    # A kernel with several output stores keeps one C ring per output;
    # judge the final rings against the whole arena and shallow them
    # where the extra rings overflow.
    _fix_multi_output_ring_search_config(state, config)
    # CLC admission depends on the projected cluster/pid/strategy tuple
    # above, including the scheduler/c-input warp fix-ups.
    _fix_clc_persistence_search_config(state, config)
    _validate_sched_stage_count_config(state, config, fix_invalid=True)
    _validate_clc_query_pacing_config(state, config, fix_invalid=True)
    _validate_cta_group_config(state, config, fix_invalid=True)
    _validate_direct_entry_ab_stage_envelope(state, config, fix_invalid=True)


def _clustered_launch_loop_order_entries(
    state: CuteTcgen05Config, config: dict[str, object]
) -> list[int] | None:
    """``loop_orders`` entry of each clustered GEMM launch of a chain.

    ``LoopOrderSpec`` entries exist only for tile loops over two or more
    block ids (a 1-D phase owns none), so a launch's entry is never its
    root position: it is the spec over exactly the launch's (m, n) grid
    block ids.  ``None`` when some clustered launch has no unique such
    entry; the caller then binds the whole config's raster.
    """
    specs = state.config_spec.loop_orders
    entries: list[int] = []
    for (m_block_id, n_block_id, _k_block_id), pid_type in zip(
        state.materialized_matmul_block_ids,
        surface._region_pid_types(state, config),
        strict=True,
    ):
        if pid_type == "flat":
            continue
        matches = [
            index
            for index, spec in enumerate(specs)
            if set(spec.block_ids) == {m_block_id, n_block_id}
        ]
        if len(matches) != 1:
            return None
        entries.append(matches[0])
    return entries


def _fix_persistent_grid_cap_search_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    """Return the grid cap to its default where the projected tuple makes it inert.

    The projections above may reset ``tcgen05_cluster_n`` to 1 (one-CTA
    samples, odd N tile counts) or leave a non-persistent walk; the knob
    then has no effect, and two samples differing only in it would be the
    same kernel twice. Keep the search schema's canonical representation.
    """
    if TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY not in config:
        return
    live = (
        config.get("tcgen05_cluster_n", 1) == 2
        and config.get("pid_type") in ("persistent_blocked", "persistent_interleaved")
        and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        != Tcgen05PersistenceModel.CLC_PERSISTENT.value
    )
    if live:
        return
    if TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY in surface.optional_fragments(
        state, for_search=True
    ):
        config[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] = (
            TCGEN05_PERSISTENT_GRID_CAP_DEFAULT
        )
    else:
        config.pop(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, None)


def normalize_strategy(
    state: CuteTcgen05Config, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if not state.search_enabled:
        return
    default_loop_orders = [
        spec._fill_missing() for spec in state.config_spec.loop_orders
    ]
    loop_orders = config.get("loop_orders", default_loop_orders)
    cluster_shape = (
        config.get("tcgen05_cluster_m", 1),
        config.get("tcgen05_cluster_n", 1),
    )
    # The clustered scheduler binds physical M/N dimensions to work-tile
    # coordinates 0/1. Reordering is safe for a single CTA, but clustered
    # scheduling must become block-ID-aware before those coordinates move.
    # Only the clustered launches are bound: an ordered chain's flat
    # launches (its SIMT phases, a GEMM left ``inherit``) keep their
    # raster.  Each clustered launch's entry is found by its (m, n) block
    # ids; a chain whose clustered launches do not all own such an entry
    # falls back to the whole-config rule below.
    if (
        cluster_shape != (1, 1)
        and state.region_pid_type_search_enabled
        and isinstance(loop_orders, list)
        and len(loop_orders) == len(default_loop_orders)
        and (entries := _clustered_launch_loop_order_entries(state, config)) is not None
    ):
        if any(loop_orders[entry] != default_loop_orders[entry] for entry in entries):
            if fix_invalid:
                repaired = list(loop_orders)
                for entry in entries:
                    repaired[entry] = default_loop_orders[entry]
                config["loop_orders"] = repaired
            else:
                raise InvalidConfig(
                    "non-default loop_orders of a clustered GEMM launch "
                    "require tcgen05 cluster shape (1, 1)"
                )
    elif cluster_shape != (1, 1) and loop_orders != default_loop_orders:
        if fix_invalid:
            config["loop_orders"] = default_loop_orders
        else:
            raise InvalidConfig(
                "non-default loop_orders require tcgen05 cluster shape (1, 1)"
            )
    pid_type_for_default = config.get("pid_type")
    strategy_validation_fragments = surface.strategy_validation_fragments(state)
    for key, fragment in strategy_validation_fragments.items():
        if key in config:
            config[key] = surface._validate_optional_fragment_value(
                key, fragment, config[key]
            )
        else:
            config[key] = surface.strategy_field_default(
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
    validate_strategy_invariants(state, config, fix_invalid=fix_invalid)
    if fix_invalid:
        # Strategy validation can reset scheduler/c-input fields for
        # inconsistent user configs. Revalidate aux-TMA after that reset so
        # TMA aux loads do not outlive their producer warp. CLC does not
        # need a matching second pass because the reset path never produces
        # a CLC persistence model.
        _fix_aux_tma_search_config(state, config)
    # The producer warp's AB depth is judged on the final strategy tuple:
    # the invariants above may have reset the warp fields.
    _normalize_aux_producer_ab_stages(state, config, fix_invalid=fix_invalid)
    _normalize_grouped_static_signature_raster(state, config, fix_invalid=fix_invalid)
    _normalize_grouped_static_reserved_sms(state, config)
    _normalize_grouped_full_coverage(
        state, config, fix_invalid=fix_invalid, validate_schedule=True
    )
    _normalize_grouped_row_union(
        state, config, fix_invalid=fix_invalid, validate_schedule=True
    )
    # Every 2x2 cluster family walks whole pairs of N tiles; judged here
    # on the final tile (the search projections above may have moved
    # block_n), ahead of the fanout so a repaired cluster keeps its
    # fanout.
    _validate_cluster_n_pairs_n_tiles(state, config, fix_invalid=fix_invalid)
    _normalize_epilogue_fanout(
        state, config, fix_invalid=fix_invalid, validate_schedule=True
    )
    # The epilogue warp groups of a two-output kernel ride on the shared
    # fanout; re-judged after the fanout's own admission and ahead of the
    # ring pricing (a dropped second group halves the C rings, a lone
    # group's one-stage ring deepens to two).
    _normalize_epilogue_groups(state, config, fix_invalid=fix_invalid)
    # A kernel with several output stores keeps one C ring per output;
    # the search's fix pass shallowed its samples above, so this refuses
    # only an explicit config whose rings provably overflow the arena.
    _normalize_multi_output_rings(state, config, fix_invalid=fix_invalid)
    if (
        fix_invalid
        and config.get(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY) is True
        and not aux_fragment_hoist_admissible(state, config)
    ):
        # ``normalize_pre_pid_type`` judged the hoist on the sampled tile
        # and recipe; the search's geometry projections (the canonical
        # 256x256 pair of ``_fix_cluster_m2_search_config``, the aux-TMA
        # full-tile regime) ran after it, so the hoist's fragment budget,
        # placement and load mode are re-judged on the final config for
        # every kernel: the repaired config has to be a fixed point of
        # strict normalization (the autotuner compiled such neighbours of
        # the 256x128 hoist witness and lost them to the strict refusal).
        config[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = False
    if fix_invalid:
        # ``_validate_cluster_n_pairs_n_tiles`` above may have dropped a
        # sampled N cluster whose tile count is odd; the persistent grid
        # cap ``fix_search_config`` judged live on that cluster is inert
        # on the final tuple and strict normalization refuses it, so it
        # is re-judged here (same fixed-point rule as the hoist above).
        _fix_persistent_grid_cap_search_config(state, config)
    # ``fix_search_config`` judged a sampled split on the repaired tuple;
    # the strict pass judges an explicit one here, on the same facts.
    _normalize_split_k(state, config, fix_invalid=fix_invalid)
