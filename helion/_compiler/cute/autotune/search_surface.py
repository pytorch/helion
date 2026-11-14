"""The CuTe backend's ``BackendSearchSurface``: keys, fields, normalization, seeds.

``CuteSearchSurface`` is what ``CuteBackend.config_spec_extension()`` returns:
the stateless object ``ConfigSpec`` calls at the fixed positions of its flat
field table and its ``normalize`` pipeline (see
``helion/autotuner/search_surface.py`` for the phase order).  The functions
below are the former CuTe methods of ``ConfigSpec``, taking the spec they used
to be bound to; the flash and mma_dag families live in ``flash_surface`` and
``mma_dag_surface``, the key tables in ``config_keys``."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast

import torch

from ...._compat import target_device_capability as get_target_device_capability
from ...._utils import prev_power_of_2
from ....autotuner.compiler_refinement import compiler_refinement_policy
from ....autotuner.config_fragment import BooleanFragment
from ....autotuner.config_fragment import ConfigSpecFragment
from ....autotuner.config_fragment import EnumFragment
from ....autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from ....autotuner.config_spec import ReductionLoopSpec
from ....autotuner.config_spec import _as_sequence
from ....autotuner.search_surface import BackendSearchSurface
from ....exc import InvalidConfig
from ..block_scaled_config import block_scaled_choices
from ..block_scaled_config import normalize_block_scaled_config
from ..cute_flash import FLASH_KV_STAGE_KEY
from ..cute_flash import FLASH_PIPELINE_FAMILY_KEY
from ..cute_flash_bwd_row_mma import FLASH_BWD_FA4_FAMILY
from ..cute_flash_bwd_row_mma import FLASH_BWD_FAMILIES
from ..cute_flash_bwd_row_mma import FLASH_BWD_FAMILY_KEY
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_KV_CHOICES
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_KV_KEY
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_Q_CHUNK_CHOICES
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_Q_CHUNK_KEY
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_Q_SPLIT_CHOICES
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_Q_SPLIT_KEY
from ..cute_flash_bwd_row_mma import flash_bwd_row_search_grid
from ..cute_mma_dag import MMA_DAG_AUX_KEY
from ..cute_mma_dag import MMA_DAG_CLUSTER_KEY
from ..cute_mma_dag import MMA_DAG_DERIVED_KEY
from ..cute_mma_dag import MMA_DAG_EARLY_TMEM_RETIREMENT_KEY
from ..cute_mma_dag import MMA_DAG_EPILOGUE_KEY
from ..cute_mma_dag import MMA_DAG_EXCHANGE_KEY
from ..cute_mma_dag import MMA_DAG_KEY
from ..cute_mma_dag import MMA_DAG_KEYS
from ..cute_mma_dag import MMA_DAG_LOADS_KEY
from ..cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from ..cute_mma_dag import MMA_DAG_PERSISTENT_KEY
from ..cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_KEY
from ..cute_mma_dag import MMA_DAG_QUERY_TMA_KEY
from ..cute_mma_dag import MMA_DAG_SLEEP_WAITS_KEY
from ..cute_mma_dag import MMA_DAG_SPLIT_KEY
from ..cute_mma_dag import MMA_DAG_STAGES_KEY
from ..cute_mma_dag import MMA_DAG_STORE_KEY
from ..cute_mma_dag import MMA_DAG_TCGEN05
from ..cute_mma_dag import MMA_DAG_V_LAG_KEY
from ..cute_mma_dag import MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY
from ..cute_mma_dag import MMA_DAG_VALUE_METADATA_SKIP_KEY
from ..cute_mma_dag import MMA_DAG_WARPS_KEY
from ..cute_register_subblocks import COLSIDE_KEY as RSB_COLSIDE_KEY
from ..cute_register_subblocks import KEY as RSB_KEY
from ..cute_register_subblocks import KEYS as RSB_KEYS
from ..cute_register_subblocks import MASK_KEY as RSB_MASK_KEY
from ..cute_register_subblocks import PREFETCH_KEY as RSB_PREFETCH_KEY
from ..cute_register_subblocks import ROLL_KEY as RSB_ROLL_KEY
from ..cute_register_subblocks import WARPS_KEY as RSB_WARPS_KEY
from ..cutedsl_compat import fixed_l2_evict_last_store_policy_supported
from ..cutedsl_compat import readonly_packet_load_supported
from ..direct_affine_plan import direct_affine_schedule_choices
from ..fragment_mma_dot import FRAGMENT_MMA_K_THREADS
from ..fragment_mma_dot import FRAGMENT_MMA_KEY
from ..fragment_mma_dot import FRAGMENT_MMA_ROW_THREADS
from ..pdl_release import CUTE_PDL_CAPTURE_RELEASE_CHOICES
from ..pdl_release import CUTE_PDL_RELEASE_CHOICES
from ..pdl_release import CUTE_PDL_RELEASE_DEFAULT
from ..pdl_release import CUTE_PDL_SPREAD_KEY
from ..ring_mma_dot import RING_DEFAULT_SPLIT
from ..ring_mma_dot import RING_DEFAULT_STAGES
from ..ring_mma_dot import RING_MMA_KEY
from ..ring_mma_dot import RING_SPLIT_CHOICES
from ..ring_mma_dot import RING_SPLIT_KEY
from ..ring_mma_dot import RING_STAGE_CHOICES
from ..ring_mma_dot import RING_STAGES_KEY
from ..ring_mma_dot import ring_mma_geometry_reason
from ..ring_mma_dot import ring_mma_smem_budget
from ..selection.radix_selection_config import ALGORITHM_KEY as RADIX_ALGORITHM_KEY
from ..selection.radix_selection_config import ALGORITHMS as RADIX_ALGORITHMS
from ..selection.radix_selection_config import BOOLEAN_KEYS as RADIX_BOOLEAN_KEYS
from ..selection.radix_selection_config import CONFIG_KEYS as RADIX_CONFIG_KEYS
from ..selection.radix_selection_config import (
    GEOMETRY_CHOICES as RADIX_GEOMETRY_CHOICES,
)
from ..selection.radix_selection_config import HISTOGRAM_KEY as RADIX_HISTOGRAM_KEY
from ..selection.radix_selection_config import HISTOGRAM_MODES as RADIX_HISTOGRAM_MODES
from ..selection.radix_selection_config import (
    PIPELINE_CHOICES as RADIX_PIPELINE_CHOICES,
)
from ..selection.radix_selection_config import normalize_radix_selection
from ..split_k_cluster import validate_cluster_config
from ..split_k_cluster_config import FINALIZER_KEY as SPLIT_K_FINALIZER_KEY
from ..split_k_cluster_config import FINALIZER_WARPS
from ..split_k_cluster_config import SCHEDULE_KEY as SPLIT_K_SCHEDULE_KEY
from ..split_k_cluster_config import SCHEDULES as SPLIT_K_SCHEDULES
from ..split_k_cluster_config import normalize_cluster_finalizer
from ..split_k_cluster_config import normalize_cluster_schedule
from ..split_k_workspace_config import (
    ATOMIC_EPILOGUE_KEY as SPLIT_K_ATOMIC_EPILOGUE_KEY,
)
from ..split_k_workspace_config import PARTITIONS_KEY as SPLIT_K_PARTITIONS_KEY
from ..split_k_workspace_config import STAGES_KEY as SPLIT_K_STAGES_KEY
from ..split_k_workspace_config import WORKSPACE_KEY as SPLIT_K_WORKSPACE_KEY
from ..split_k_workspace_config import normalize_workspace_config
from ..split_k_workspace_config import workspace_search_fields
from ..tcgen05_config import CuteTcgen05Config
from ..tcgen05_config import repair
from ..tcgen05_config import seeds
from ..tcgen05_config import surface
from ..tcgen05_flat_grouped_config import CONVERTER_WARPS as GROUPED_RNA_WARPS
from ..tcgen05_flat_grouped_config import K_KEY as GROUPED_RNA_K_KEY
from ..tcgen05_flat_grouped_config import PREFIX_SCAN_KEY as GROUPED_PREFIX_SCAN_KEY
from ..tcgen05_flat_grouped_config import PREFIX_SCANS as GROUPED_PREFIX_SCANS
from ..tcgen05_flat_grouped_config import (
    RESIDENT_CTAS_KEY as GROUPED_RNA_RESIDENT_CTAS_KEY,
)
from ..tcgen05_flat_grouped_config import STAGES_KEY as GROUPED_RNA_STAGES_KEY
from ..tcgen05_flat_grouped_config import WARPS_KEY as GROUPED_RNA_WARPS_KEY
from ..tcgen05_flat_grouped_config import normalize_grouped_rna_config
from ..tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from ..tcgen05_grouped_descriptors import DYNAMIC
from ..tcgen05_grouped_descriptors import WRAPPED
from ..tcgen05_tma_rn import AUTO as CUTE_MMA_F32_AUTO
from ..tcgen05_tma_rn import CONVERSION_KEY as CUTE_MMA_F32_CONVERSION_KEY
from ..tcgen05_tma_rn import TMA_RN as CUTE_MMA_F32_TMA_RN
from ..tcgen05_tma_rn import WARP_RAW as CUTE_MMA_F32_WARP_RAW
from ..thread_budget import CUTE_REGISTER_TILE_MAX_ELEMENTS
from .config_generation_ext import CuteConfigGenerationExtension
from .config_keys import CUTE_AFFINE_SCAN_SCHEDULE_KEY
from .config_keys import CUTE_CHUNK_PREPARE_SCHEDULE_KEY
from .config_keys import CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY
from .config_keys import CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY
from .config_keys import CUTE_COMPILER_PIPELINE_CHOICES
from .config_keys import CUTE_COMPILER_PIPELINE_KEY
from .config_keys import CUTE_CONFIG_KEYS
from .config_keys import CUTE_DEBUG_KEYS
from .config_keys import CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY
from .config_keys import CUTE_GDN_RECURRENCE_MMA_M_KEY
from .config_keys import CUTE_GDN_RECURRENCE_STAGES_KEY
from .config_keys import CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY
from .config_keys import CUTE_IMPLICIT_DEFAULT_KEYS
from .config_keys import CUTE_TOPK_CHOICES
from .config_keys import VALID_CUTE_CHUNK_PREPARE_SCHEDULES
from .config_keys import VALID_CUTE_CHUNK_RECURRENCE_REGISTER_CAPS
from .config_keys import _cute_chunk_recurrence_config_is_safe
from .flash_surface import _cute_flash_autotune_fragments
from .flash_surface import _normalize_cute_flash
from .flash_surface import _normalize_cute_flash_bwd
from .flash_surface import _normalize_cute_flash_gated
from .flash_surface import _pre_normalize_cute_flash_block_sizes
from .flash_surface import _resolve_cute_flash_config
from .flash_surface import cute_flash_bwd_seed_configs
from .flash_surface import cute_flash_gated_seed_configs
from .flash_surface import cute_flash_seed_configs
from .flash_surface import fingerprint_extra
from .flash_surface import pdl_searched
from .flash_surface import prepare_config_generation_overrides
from .flash_surface import prepare_override_normalization
from .mma_dag_surface import _normalize_cute_mma_dag
from .mma_dag_surface import _normalize_cute_register_subblocks
from .mma_dag_surface import normalize_value_metadata_options

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ....autotuner.block_id_sequence import BlockIdSequence
    from ....autotuner.config_generation import ConfigGeneration
    from ....autotuner.config_spec import ConfigSpec
    from ....autotuner.config_spec import MatmulFact
    from ....runtime.config import Config


# How a fp32 quotient by a loop-invariant divisor is lowered
# (``_normalize_cute_exact_div_form``); ``guard`` was retired (a per-element
# branch whose code shape changed ptxas's contraction of the surrounding fp32
# chains, so it was not bit identical to ``plain``).
CUTE_EXACT_DIV_FORMS: tuple[str, ...] = ("redo", "plain")
_CUTE_EXACT_DIV_FORM_RETIRED: frozenset[str] = frozenset({"guard"})

# Schedule knobs of the CuTe register tile (``RegisterTileOptions``); dropped
# from a config that forms no register tile.
_CUTE_REGISTER_TILE_KNOBS: tuple[str, ...] = (
    "cute_register_tile_warp_fold",
    "cute_register_tile_store_first",
)

# A proved complete MXFP8 root is replaced by one native scaled contraction.
# These coordinates describe the displaced SIMT/collective lowering and cannot
# change that contraction. Keep the native geometry and every user tunable.
_CUTE_MXFP8_ROOT_LOWERING_KEYS = frozenset(
    {
        "block_sizes",
        "num_threads",
        "loop_orders",
        "l2_groupings",
        "flatten_loops",
        "indexing",
        "load_eviction_policies",
        "load_cache_modifiers",
        "store_cache_modifiers",
        "cute_vector_widths",
        "cute_lane_layouts",
        "cute_reduction_reloads",
        "cute_collective_mma",
        "cute_collective_static_layouts",
        "cute_collective_compute",
        "cute_collective_native_seeded",
        "cute_collective_tmem_seed",
        "cute_collective_tmem_a",
        "cute_collective_stages",
        "cute_collective_copy",
        "cute_collective_recipe",
        "cute_collective_operand_packets",
        "cute_collective_epilogue",
        "cute_gathered_mma_n",
        "cute_gathered_mma_stages",
        "cute_register_chain",
        "cute_simt_mma",
        "cute_simt_mma_layout",
        FRAGMENT_MMA_KEY,
        RING_MMA_KEY,
        RING_STAGES_KEY,
        RING_SPLIT_KEY,
        "cute_pdl",
        "cute_pdl_capture_overlap",
        "cute_pdl_release",
        "cute_pdl_spread",
        "cute_min_blocks_per_mp",
        "cute_tiles_per_block",
        "epilogue_subtile",
    }
)


_COMPILER_PIPELINE_FRAGMENT = EnumFragment(choices=CUTE_COMPILER_PIPELINE_CHOICES)


def _fragment_threads(config: Mapping[str, object]) -> int:
    return cast("int", config.get("cute_fragment_threads", 128))


def _active_warp_scan_roots(
    config_spec: ConfigSpec, config: Mapping[str, object]
) -> frozenset[int]:
    from .heuristics.cute_fragment_warp_scan import spec_active_warp_scan_roots

    return spec_active_warp_scan_roots(config_spec, config)


class _FragmentOption(NamedTuple):
    """An independent computed-fragment mode: what admits it, its requirement and choices.

    ``admitted`` reads the spec facts (roots, availability, thread floors) and
    the config's prerequisite keys; any set key in ``excludes`` disables the
    mode; ``rejection`` formats the ``InvalidConfig`` text.
    """

    admitted: Callable[[ConfigSpec, Mapping[str, object]], object]
    requirement: str
    choices: tuple[bool | int | str, ...] = (False, True)
    excludes: tuple[str, ...] = ("cute_collective_mma", "cute_register_chain")
    rejection: str = "{key}={value!r} requires {requirement}"


# The rejection of the modes that name the key alone.
_KEY_ONLY_REJECTION = "{key} requires {requirement}"

# The fragment modes normalized by ``_normalize_cute_fragment_option`` alone,
# keyed by config key: what admits the mode, the requirement named by the
# rejection and the typed choices (the first is the default).
_FRAGMENT_OPTIONS: dict[str, _FragmentOption] = {
    "cute_fragment_published_scalars": _FragmentOption(
        lambda spec, config: spec.cute_fragment_published_scalar_root_ids,
        "a complete fragment with published scalar candidates",
    ),
    "cute_fragment_skip_zero_atomics": _FragmentOption(
        lambda spec, config: spec.cute_fragment_skip_zero_atomics_root_ids,
        "a proved unused or dead-zero-result relaxed CTA-local Int32 atomic add",
    ),
    "cute_fragment_private_scalar_loops": _FragmentOption(
        lambda spec, config: spec.cute_fragment_private_scalar_loop_root_ids,
        "a proved read-only scalar finalizer loop",
    ),
    "cute_fragment_producer_cache": _FragmentOption(
        lambda spec, config: spec.cute_fragment_producer_cache_root_ids,
        "repeated pure fragment producers",
    ),
    "cute_fragment_topk_network": _FragmentOption(
        lambda spec, config: spec.cute_fragment_topk_network_root_ids,
        "a supported complete fragment top-k",
    ),
    "cute_fragment_atomic_aggregation": _FragmentOption(
        lambda spec, config: spec.cute_fragment_atomic_aggregation_root_ids,
        "proved CTA-private Int32 relaxed unused atomic updates",
    ),
    "cute_fragment_atomic_consumer_fusion": _FragmentOption(
        lambda spec, config: spec.cute_fragment_atomic_consumer_fusion_root_ids,
        "a proved same-coordinate local Int32 atomic consumer region",
    ),
    "cute_fragment_pure_producer_regions": _FragmentOption(
        lambda spec, config: spec.cute_fragment_pure_producer_regions_root_ids,
        "a proved same-owner pure producer publication region",
    ),
    "cute_fragment_integer_atomic_epochs": _FragmentOption(
        lambda spec, config: spec.cute_fragment_integer_atomic_epochs_root_ids,
        "proved CTA-private Int32 relaxed unused loop updates",
    ),
    "cute_fragment_packet_loads": _FragmentOption(
        lambda spec, config: spec.cute_fragment_packet_load_root_ids,
        "a complete fragment root with readonly Float32, Float16, BFloat16, Int8 or Uint8 host loads",
    ),
    "cute_fragment_padded_rows": _FragmentOption(
        lambda spec, config: spec.cute_fragment_padded_rows_root_ids,
        "a complete fragment root staging a tile whose rows are a multiple of 128 bytes",
    ),
    "cute_fragment_unrolled_loops": _FragmentOption(
        lambda spec, config: spec.cute_fragment_unrolled_loops_root_ids,
        "a complete fragment root",
    ),
    "cute_fragment_vector_io": _FragmentOption(
        lambda spec, config: spec.cute_fragment_vector_io_root_ids,
        "a complete fragment root moving a 16-byte-packetizable host window",
    ),
    "cute_fragment_batched_loads": _FragmentOption(
        lambda spec, config: spec.cute_fragment_batched_loads_root_ids,
        "a complete fragment root with two or more host loads",
    ),
    "cute_fragment_slab_loads": _FragmentOption(
        lambda spec, config: spec.cute_fragment_slab_loads_root_ids,
        "a complete fragment root with repeated static windows of one tensor",
    ),
    "cute_fragment_register_loads": _FragmentOption(
        lambda spec, config: spec.cute_fragment_register_load_root_ids,
        "a supported lane-private fragment load",
    ),
    "cute_fragment_reduction": _FragmentOption(
        lambda spec, config: spec.cute_fragment_reduction_root_ids,
        "a supported computed scalar reduction",
        ("serial", "warp", "cta", "cta_unroll"),
    ),
    "cute_fp32_dot": _FragmentOption(
        lambda spec, config: spec.cute_fp32_dot_root_ids,
        "an fp32 hl.dot in a computed fragment under tf32 dot precision",
        ("simt", "tf32_mma"),
    ),
    "cute_fragment_half_mma": _FragmentOption(
        lambda spec, config: spec.cute_fragment_half_mma_root_ids,
        "an FP16/BF16 hl.dot with FP32 accumulation in a computed fragment on SM80+",
    ),
    "cute_fragment_scan": _FragmentOption(
        lambda spec, config: spec.cute_fragment_scan_root_ids,
        "a supported computed additive scan",
        ("serial", "cooperative"),
    ),
    "cute_fragment_cta_topk": _FragmentOption(
        lambda spec, config: spec.cute_fragment_cta_topk_root_ids,
        "a supported complete Int64 value-only selection",
    ),
    "cute_fragment_split_k": _FragmentOption(
        lambda spec, config: spec.cute_fragment_split_k_root_ids,
        "a complete fragment root with a tf32-eligible fp32 hl.dot",
    ),
    "cute_fragment_mma_rows": _FragmentOption(
        lambda spec, config: spec.cute_fragment_mma_rows_root_ids,
        "a complete fragment root with a tf32-eligible fp32 hl.dot",
    ),
    "cute_fragment_warp_scan": _FragmentOption(
        _active_warp_scan_roots, "a supported bounded warp-prefix scan"
    ),
    "cute_fragment_local_atomic_registers": _FragmentOption(
        lambda spec, config: (
            spec.cute_fragment_local_atomic_registers_root_ids
            and _fragment_threads(config)
            >= spec.cute_fragment_local_atomic_register_min_threads
        ),
        "proved same-coordinate local Int32 atomic return consumers",
    ),
    "cute_integer_loop_reduction": _FragmentOption(
        lambda spec, config: spec.cute_integer_loop_reduction_available,
        "a pure single integer min/max loop recurrence",
        excludes=(),
        rejection=_KEY_ONLY_REJECTION,
    ),
    "cute_fragment_bounded_gather": _FragmentOption(
        lambda spec, config: spec.cute_fragment_bounded_gather_root_ids,
        "a proved bounded last-axis gather",
        rejection=_KEY_ONLY_REJECTION,
    ),
    "cute_fragment_warp_producer_regions": _FragmentOption(
        lambda spec, config: (
            spec.cute_fragment_warp_producer_regions_root_ids
            and config.get("cute_fragment_warp_scan") is True
            and config.get("cute_fragment_reduction") == "warp"
        ),
        "a proved scalar-frontier warp producer prefix",
        excludes=(
            "cute_collective_mma",
            "cute_register_chain",
            "cute_fragment_register_snapshots",
            "cute_fragment_register_loads",
            "cute_fragment_register_producers",
            "cute_fragment_warp_results",
            "cute_fragment_scan_exports",
            "reduction_loops",
        ),
        rejection=_KEY_ONLY_REJECTION,
    ),
    "cute_fragment_scan_exports": _FragmentOption(
        lambda spec, config: (
            spec.cute_fragment_scan_exports_root_ids
            and config.get("cute_fragment_warp_scan") is True
        ),
        "shared producers and a proved terminal warp-scan interval",
        excludes=(
            "cute_collective_mma",
            "cute_register_chain",
            "cute_fragment_register_snapshots",
            "cute_fragment_register_loads",
            "cute_fragment_register_producers",
            "cute_fragment_warp_results",
            "cute_fragment_pure_producer_regions",
        ),
        rejection=_KEY_ONLY_REJECTION,
    ),
    "cute_fragment_register_producers": _FragmentOption(
        lambda spec, config: (
            spec.cute_fragment_register_producer_root_ids
            and config.get("cute_fragment_register_snapshots") is True
        ),
        "bounded readonly snapshots and repeated pure producers",
    ),
    "cute_fragment_register_snapshots": _FragmentOption(
        lambda spec, config: (
            spec.cute_fragment_register_snapshots_root_ids
            and _fragment_threads(config)
            >= spec.cute_fragment_register_snapshot_min_threads
            and (
                not spec.cute_fragment_register_snapshot_while_root_ids
                or config.get("cute_fragment_bounded_gather", False) is True
            )
        ),
        "proved bounded readonly rank-one snapshots with exact integer reductions",
    ),
    "cute_fragment_warp_results": _FragmentOption(
        lambda spec, config: (
            spec.cute_fragment_warp_result_root_ids
            and _fragment_threads(config) >= spec.cute_fragment_warp_result_min_threads
        ),
        "a supported warp-owned reduction/atomic value chain",
    ),
    "cute_materialized_operand_schedule": _FragmentOption(
        lambda spec, config: spec.cute_materialized_operand_schedule_available,
        "a proved packed byte operand",
        ("off", "warp_narrow4"),
        excludes=(),
    ),
}


def _normalize_cute_fragment_option(
    config_spec: ConfigSpec,
    config: dict[str, object],
    key: str,
    option: _FragmentOption,
    *,
    fix_invalid: bool,
) -> None:
    """Normalize an independent mode without weakening its typed default.

    The default (``choices[0]``) is dropped; another choice stays when the
    option admits it under this config and none of its excluded keys is set;
    anything else is repaired away or rejected.
    """
    default = option.choices[0]
    value = config.get(key, default)
    if type(value) is type(default):
        if value == default:
            config.pop(key, None)
            return
        if (
            value in option.choices
            and option.admitted(config_spec, config)
            and not any(config.get(other) for other in option.excludes)
        ):
            return
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(
        option.rejection.format(key=key, value=value, requirement=option.requirement)
    )


def _normalize_fragment_option(
    config_spec: ConfigSpec, config: dict[str, object], key: str, *, fix_invalid: bool
) -> None:
    """Normalize one ``_FRAGMENT_OPTIONS`` mode."""
    _normalize_cute_fragment_option(
        config_spec, config, key, _FRAGMENT_OPTIONS[key], fix_invalid=fix_invalid
    )


def enable_cute_chunk_recurrence_search(
    config_spec: ConfigSpec, *, preferred_partitions: int
) -> None:
    """Expose the exact BT16 recurrence schedule as CuTe search knobs."""

    if preferred_partitions not in (2, 4):
        raise ValueError(
            f"unsupported chunk-recurrence DV partition count: {preferred_partitions}"
        )
    alternate = 4 if preferred_partitions == 2 else 2
    config_spec.cute_chunk_recurrence_dv_partitions = EnumFragment(
        choices=(preferred_partitions, alternate)
    )
    config_spec.cute_chunk_recurrence_register_cap = EnumFragment(
        choices=VALID_CUTE_CHUNK_RECURRENCE_REGISTER_CAPS
    )


def enable_cute_gdn_recurrence_search(
    config_spec: ConfigSpec,
    *,
    value_block_id: int,
    stage_choices_by_tile: Mapping[tuple[int, int], Sequence[int]],
    epilogue_warp_choices: Sequence[int],
    mma_m_choices_by_tile: Mapping[tuple[int, int], Sequence[int]],
    token_group_choices_by_tile: Mapping[tuple[int, int, int], Sequence[int]],
) -> None:
    """Expose the gdn recurrence TMA ring depth, epilogue warp count,
    tcgen05 M and token group count.

    ``mma_m_choices_by_tile`` maps every (dstate tile, epilogue warps)
    pair the planner lowers (the candidate tiles first, then the wider
    admitted tiles the ``block_sizes`` search reaches for a
    non-power-of-two dstate) to its legal MMA heights, preferred first;
    ``stage_choices_by_tile`` maps every (tile, M) pair to its legal ring
    depths and ``token_group_choices_by_tile`` every (tile, warps, M)
    triple to its pipelined token groups.  Each search domain is the
    union of its map's values so each per-tile seed is legal as written,
    and :meth:`normalize` re-validates the values against the selected
    tile: a defaulted value follows the tile, an explicit value the tile
    cannot take is rejected.
    """

    mma_m_choices: list[int] = []
    for choices in mma_m_choices_by_tile.values():
        for mma_m in choices:
            if mma_m not in mma_m_choices:
                mma_m_choices.append(mma_m)
    stage_choices: list[int] = []
    for choices in stage_choices_by_tile.values():
        for stages in choices:
            if stages not in stage_choices:
                stage_choices.append(stages)
    group_choices: list[int] = []
    for choices in token_group_choices_by_tile.values():
        for groups in choices:
            if groups not in group_choices:
                group_choices.append(groups)
    if (
        not stage_choices
        or not epilogue_warp_choices
        or not mma_m_choices
        or not group_choices
    ):
        raise ValueError("gdn recurrence search requires at least one choice")
    config_spec.cute_gdn_recurrence_value_block_id = value_block_id
    config_spec.cute_gdn_recurrence_mma_m_choices_by_tile = {
        tile: tuple(choices) for tile, choices in mma_m_choices_by_tile.items()
    }
    # The full 128-row tile leads the domain: it is legal for every
    # tile, so a config that leaves the knob out keeps the full MMA.
    config_spec.cute_gdn_recurrence_mma_m = EnumFragment(
        choices=tuple(sorted(mma_m_choices, reverse=True))
    )
    config_spec.cute_gdn_recurrence_stage_choices_by_tile = {
        tile: tuple(choices) for tile, choices in stage_choices_by_tile.items()
    }
    config_spec.cute_gdn_recurrence_stages = EnumFragment(choices=tuple(stage_choices))
    config_spec.cute_gdn_recurrence_epilogue_warps = EnumFragment(
        choices=tuple(epilogue_warp_choices)
    )
    config_spec.cute_gdn_recurrence_token_group_choices_by_tile = {
        tile: tuple(choices) for tile, choices in token_group_choices_by_tile.items()
    }
    config_spec.cute_gdn_recurrence_token_groups = EnumFragment(
        choices=tuple(group_choices)
    )


def _cute_gdn_recurrence_tile_of(
    config_spec: ConfigSpec, config: dict[str, object]
) -> int | None:
    """The dstate tile ``config`` selects, or ``None`` when the gdn knobs
    are off or ``config`` carries no dstate tile."""

    block_id = config_spec.cute_gdn_recurrence_value_block_id
    if config_spec.cute_gdn_recurrence_stages is None or block_id is None:
        return None
    block_sizes = config.get("block_sizes")
    if not isinstance(block_sizes, list):
        return None
    return config_spec.block_sizes.config_get(cast("list[int]", block_sizes), block_id)


def _cute_gdn_recurrence_mma_m_choices_for(
    config_spec: ConfigSpec, config: dict[str, object]
) -> tuple[int, ...] | None:
    """Legal gdn tcgen05 M values for the dstate tile and epilogue warp
    count ``config`` selects; ``None`` when the knob is off or the pair
    is one the planner declines (the kernel then takes the SIMT lowering
    and the knob is inert)."""

    block_size = _cute_gdn_recurrence_tile_of(config_spec, config)
    warps = config.get(CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY)
    if block_size is None or type(warps) is not int:
        return None
    return config_spec.cute_gdn_recurrence_mma_m_choices_by_tile.get(
        (block_size, warps)
    )


def _cute_gdn_recurrence_stage_choices_for(
    config_spec: ConfigSpec, config: dict[str, object]
) -> tuple[int, ...] | None:
    """Legal gdn ring depths for the dstate tile and tcgen05 M ``config``
    selects; ``None`` when the knob is off or the pair is one the planner
    declines.  Every pair the planner lowers has an entry, so a depth
    accepted here never fails at codegen."""

    block_size = _cute_gdn_recurrence_tile_of(config_spec, config)
    mma_m = config.get(CUTE_GDN_RECURRENCE_MMA_M_KEY)
    if block_size is None or type(mma_m) is not int:
        return None
    return config_spec.cute_gdn_recurrence_stage_choices_by_tile.get(
        (block_size, mma_m)
    )


def _cute_gdn_recurrence_token_group_choices_for(
    config_spec: ConfigSpec, config: dict[str, object]
) -> tuple[int, ...] | None:
    """Legal gdn token groups for the dstate tile, epilogue warp count and
    tcgen05 M ``config`` selects; ``None`` when the knob is off or the
    triple is one the planner declines."""

    block_size = _cute_gdn_recurrence_tile_of(config_spec, config)
    warps = config.get(CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY)
    mma_m = config.get(CUTE_GDN_RECURRENCE_MMA_M_KEY)
    if block_size is None or type(warps) is not int or type(mma_m) is not int:
        return None
    return config_spec.cute_gdn_recurrence_token_group_choices_by_tile.get(
        (block_size, warps, mma_m)
    )


def enable_cute_chunk_prepare_schedule_search(
    config_spec: ConfigSpec, *, preferred_schedule: str
) -> None:
    """Expose the exact BT16 prepare shared-memory schedule knob."""

    if not config_spec.supports_config_key(CUTE_CHUNK_PREPARE_SCHEDULE_KEY):
        raise InvalidConfig(
            f"{CUTE_CHUNK_PREPARE_SCHEDULE_KEY} is not supported by backend "
            f"{config_spec.backend_name!r}"
        )
    if preferred_schedule not in VALID_CUTE_CHUNK_PREPARE_SCHEDULES:
        raise ValueError(f"unsupported chunk-prepare schedule: {preferred_schedule!r}")
    config_spec.cute_chunk_prepare_schedule = EnumFragment(
        choices=(
            preferred_schedule,
            *(
                schedule
                for schedule in VALID_CUTE_CHUNK_PREPARE_SCHEDULES
                if schedule != preferred_schedule
            ),
        )
    )


def enable_cute_affine_scan_search(
    config_spec: ConfigSpec, *, step_count: int | None
) -> None:
    """Expose measured direct-affine schedules for a compatible CuTe scan."""

    if config_spec.backend_name != "cute":
        raise InvalidConfig("direct affine scan configuration requires CuTe")
    config_spec.cute_affine_scan_schedule = EnumFragment(
        choices=direct_affine_schedule_choices(step_count)
    )


def enable_cute_topk_search(
    config_spec: ConfigSpec, selection_dtype: torch.dtype | None = None
) -> None:
    """Expose the independent row, lane, and vector geometry of top-k."""
    config_spec.cute_topk_search_enabled = True
    if selection_dtype == torch.float32:
        # A full FP32 value plus an index cannot fit any 32-bit key.
        # The 16-bit packed encoders are not legal for this representation.
        config_spec.cute_topk_choices = {
            **CUTE_TOPK_CHOICES,
            "cute_topk_key_dtype": ("int64",),
            "cute_topk_key_encoder": ("dsl",),
            "cute_topk_defer_value_gathers": (False,),
        }
    # The root emitter supplies its own tiling. Ordinary block sizes do
    # not change its code; retain valid defaults for shared compiler
    # bookkeeping without searching duplicate generated kernels.
    for spec in config_spec.block_sizes:
        target = spec._fragment(config_spec).default_val
        spec.autotuner_min = target
        spec.max_size = target


def enable_cute_async_load_pipeline(config_spec: ConfigSpec) -> None:
    """Expose the narrow CuTe async state-load search dimensions."""
    if config_spec.backend_name != "cute":
        raise InvalidConfig("async state-load pipelining requires CuTe")
    config_spec.cute_async_load_pipeline_enabled = True


def enable_cute_bf16x2_recurrence(config_spec: ConfigSpec) -> None:
    """Expose native packed-BF16 recurrence lowering to the tuner."""
    if config_spec.backend_name != "cute":
        raise InvalidConfig("packed BF16 recurrence lowering requires CuTe")
    config_spec.cute_bf16x2_recurrence_enabled = True


def enable_cute_proven_bounds(config_spec: ConfigSpec) -> None:
    """Expose proof-driven CuTe bounds cleanup to the tuner."""
    if config_spec.backend_name != "cute":
        raise InvalidConfig("proven bounds cleanup requires CuTe")
    config_spec.cute_proven_bounds_enabled = True


def enable_cute_packet_prefetch(config_spec: ConfigSpec) -> None:
    """Expose independent packet staging to the pointwise tuner."""
    if config_spec.backend_name != "cute":
        raise InvalidConfig("packet prefetch requires CuTe")
    config_spec.cute_packet_prefetch_enabled = True


def _cute_config_forms_one_vector_lane(
    config_spec: ConfigSpec,
    config: dict[str, object],
    rl_spec: ReductionLoopSpec,
    available: int,
) -> bool:
    """Whether ``config`` spells out one vector per thread for the
    persistent reduction ``rl_spec``: an explicit thread count that fits
    the remaining budget and a vector width whose product is the extent
    (``PersistentReductionStrategy`` keeps a V-wide fragment per thread
    exactly then).  That lane is the requested geometry (a pointwise
    kernel's stride-1 slice, rope's half head dim), not a thread count the
    budget shrank, so it stays persistent beside wide tile axes.
    """
    nt_list = cast("list[int]", config.get("num_threads", []) or [])
    vec_list = cast("list[int]", config.get("cute_vector_widths", []) or [])
    requested = config_spec.num_threads.config_get(nt_list, rl_spec.block_id, 0)
    if (
        not isinstance(requested, int)
        or requested <= 0
        or requested > available
        or rl_spec.size_hint % requested
    ):
        return False
    if rl_spec.block_id not in config_spec.cute_vector_widths.valid_block_ids():
        return False
    vec = config_spec.cute_vector_widths.config_get(vec_list, rl_spec.block_id, 1)
    return isinstance(vec, int) and vec > 1 and rl_spec.size_hint // requested == vec


def _cute_config_forms_register_tile(
    config_spec: ConfigSpec,
    config: dict[str, object],
    rl_spec: ReductionLoopSpec,
    available: int,
) -> bool:
    """Whether ``config`` spells out a per-thread register tile for the
    persistent reduction ``rl_spec`` (see
    ``DeviceGridState.nest_reduction_lane_outside_vector_tiles``).

    The reduction block must admit the register tile (a static extent and
    a body the two-pass schedule can lower,
    ``cute_register_tile_reduction_blocks``); the reduction thread count
    must be explicit, divide the extent and fit the remaining budget; every
    lane-looped tile block must hold exactly one vector per thread
    (elements per thread == its ``cute_vector_widths`` entry > 1), at least
    one such block must exist, and the unrolled per-thread element count
    stays within ``CUTE_REGISTER_TILE_MAX_ELEMENTS``.  Anything else keeps
    the looped reduction a shrunk thread count always meant.
    """
    if rl_spec.block_id not in config_spec.cute_register_tile_reduction_blocks:
        return False
    nt_list = cast("list[int]", config.get("num_threads", []) or [])
    bs_list = cast("list[int]", config.get("block_sizes", []) or [])
    vec_list = cast("list[int]", config.get("cute_vector_widths", []) or [])
    requested = config_spec.num_threads.config_get(nt_list, rl_spec.block_id, 0)
    if (
        not isinstance(requested, int)
        or requested <= 0
        or requested > available
        or rl_spec.size_hint % requested
    ):
        return False
    unrolled = rl_spec.size_hint // requested
    reduction_block_ids = config_spec.reduction_block_ids | {
        spec.block_id for spec in config_spec.reduction_loops
    }
    vector_block_ids = config_spec.cute_vector_widths.valid_block_ids()
    vector_blocks = 0
    for nt_spec in config_spec.num_threads:
        block_id = nt_spec.block_id
        if block_id in reduction_block_ids:
            continue
        nt = config_spec.num_threads.config_get(nt_list, block_id, 0)
        bs = config_spec.block_sizes.config_get(bs_list, block_id, 1)
        if not isinstance(nt, int) or not isinstance(bs, int) or nt <= 0 or nt >= bs:
            # No lane loop on this block.
            continue
        vec = (
            config_spec.cute_vector_widths.config_get(vec_list, block_id, 1)
            if block_id in vector_block_ids
            else 1
        )
        if bs % nt or not isinstance(vec, int) or vec <= 1 or bs // nt != vec:
            return False
        unrolled *= vec
        vector_blocks += 1
    return vector_blocks > 0 and unrolled <= CUTE_REGISTER_TILE_MAX_ELEMENTS


def _normalize_cute_pointwise_pid_type(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_pointwise_pid_type"
    value = config.get(key, "inherit")
    if type(value) is str and value == "inherit":
        config.pop(key, None)
        return
    if (
        type(value) is str
        and value == "flat"
        and config_spec.cute_pointwise_region_block_ids
    ):
        return
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(f"{key}={value!r} requires a proved pointwise region")


def _normalize_cute_fragment_cta_topk(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_fragment_cta_topk"
    _normalize_fragment_option(config_spec, config, key, fix_invalid=fix_invalid)
    if config.get(key) and config.get("cute_fragment_topk_network") is not True:
        if fix_invalid:
            config.pop(key, None)
        else:
            raise InvalidConfig(
                "CTA selection requires cute_fragment_topk_network=True"
            )


def _normalize_cute_fragment_split_k(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    # The split rides the tf32 mma phase: without it the knob is inert.
    if config.get("cute_fragment_split_k") and (
        config.get("cute_fp32_dot", "simt") != "tf32_mma"
    ):
        if not fix_invalid:
            raise InvalidConfig(
                "cute_fragment_split_k=True requires cute_fp32_dot='tf32_mma'"
            )
        config.pop("cute_fragment_split_k", None)
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_split_k", fix_invalid=fix_invalid
    )


def _normalize_cute_fragment_mma_rows(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    # The row walk is a form of the tf32 mma phase: inert without it.
    if config.get("cute_fragment_mma_rows") and (
        config.get("cute_fp32_dot", "simt") != "tf32_mma"
    ):
        if not fix_invalid:
            raise InvalidConfig(
                "cute_fragment_mma_rows=True requires cute_fp32_dot='tf32_mma'"
            )
        config.pop("cute_fragment_mma_rows", None)
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_mma_rows", fix_invalid=fix_invalid
    )


def _normalize_cute_fragment_threads(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    from .heuristics.cute_fragment_threads import THREADS

    _normalize_cute_fragment_option(
        config_spec,
        config,
        "cute_fragment_threads",
        _FragmentOption(
            lambda spec, _: spec.cute_fragment_thread_root_ids,
            "a supported computed fragment root",
            (128, *(count for count in THREADS if count != 128)),
        ),
        fix_invalid=fix_invalid,
    )


def _normalize_cute_materialized_schedule(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    _normalize_cute_fragment_option(
        config_spec,
        config,
        "cute_materialized_schedule",
        _FragmentOption(
            lambda spec, _: spec.cute_materialized_schedule_available,
            "a proved compact row-resident pair",
            config_spec.cute_materialized_schedule_choices,
            excludes=(),
        ),
        fix_invalid=fix_invalid,
    )


def _normalize_cute_host_paired_sum(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_host_paired_sum"
    value = config.get(key, "off")
    if not config_spec.cute_host_paired_sum_available:
        if value != "off" and not fix_invalid:
            raise InvalidConfig(
                "host sums require a typed independent pair, terminal sum/cast "
                "or a 1-D sum/mean after the launch"
            )
        config.pop(key, None)
        return
    if value not in ("off", "mapped", "narrow"):
        if not fix_invalid:
            raise InvalidConfig(f"unsupported paired host sum layout: {value!r}")
        value = "off"
    config[key] = value


def _normalize_cute_reduction_sequence(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_reduction_sequence"
    if not config_spec.cute_sequence_reduction_blocks:
        if key in config and not fix_invalid:
            raise InvalidConfig(
                "resident sequences require a device-tile reduction chain"
            )
        config.pop(key, None)
        return
    value = config.setdefault(key, "scalar")
    if value not in ("scalar", "reload", "resident", "bounded", "bounded_layout"):
        if fix_invalid:
            config[key] = "scalar"
        else:
            raise InvalidConfig(f"unsupported CuTe reduction sequence: {value!r}")


def _normalize_cute_reduction_local_tree(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_reduction_local_tree"
    value = config.get(key, False)
    if value is False:
        config.pop(key, None)
        return
    if (
        value is True
        and config_spec.cute_resident_reduction_blocks
        and config.get("cute_reduction_schedule") in ("resident", "pipelined")
    ):
        return
    if fix_invalid:
        config.pop(key, None)
    else:
        raise InvalidConfig(
            "cute_reduction_local_tree requires a resident FP32 sum schedule"
        )


def _normalize_cute_reduction_row_output(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    row_key = "cute_reduction_row_schedule"
    pack_key = "cute_reduction_pack_output"
    row = config.get(row_key, "batched")
    resident = bool(config_spec.cute_resident_reduction_blocks) and config.get(
        "cute_reduction_schedule"
    ) in ("resident", "pipelined")
    if row == "batched":
        config.pop(row_key, None)
    elif (
        type(row) is str
        and row in ("serial", "serial_deferred")
        and resident
        and (
            row != "serial_deferred"
            or (
                config.get("cute_reduction_schedule") == "pipelined"
                and config.get("cute_reduction_pipeline_depth", 2) == 2
            )
        )
    ):
        pass
    elif fix_invalid:
        config.pop(row_key, None)
    else:
        raise InvalidConfig(
            f"{row_key} requires a resident schedule; deferred retirement requires a two-slot pipeline"
        )
    pack = config.get(pack_key, False)
    if pack is False:
        config.pop(pack_key, None)
    elif (
        pack is True
        and resident
        and config_spec.target_device_capability is not None
        and config_spec.target_device_capability[0] == 10
    ):
        pass
    elif fix_invalid:
        config.pop(pack_key, None)
    else:
        raise InvalidConfig(
            f"{pack_key} requires a Boolean and a resident FP32 output product on SM100"
        )


def _normalize_cute_reduction_schedule(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_reduction_schedule"
    group_key = "cute_reduction_group_rows"
    depth_key = "cute_reduction_pipeline_depth"
    if not config_spec.cute_resident_reduction_blocks:
        if any(k in config for k in (key, group_key, depth_key)) and not fix_invalid:
            raise InvalidConfig(
                "resident CuTe reductions require static serial-row sums"
            )
        config.pop(key, None)
        config.pop(group_key, None)
        config.pop(depth_key, None)
        return
    value = config.setdefault(key, "scalar")
    if value not in ("scalar", "resident", "pipelined"):
        if fix_invalid:
            config[key] = "scalar"
        else:
            raise InvalidConfig(f"unsupported CuTe reduction schedule: {value!r}")
    group = config.setdefault(group_key, 1)
    if type(group) is not int or group not in (1, 2, 3, 4):
        if fix_invalid:
            config[group_key] = 1
        else:
            raise InvalidConfig(f"unsupported CuTe reduction group size: {group!r}")
    if config[key] == "scalar":
        config[group_key] = 1
    # An omitted depth must not add a key to old normalized configs or
    # generated-source headers. The search fragment still defaults to 2.
    depth = config.get(depth_key, 2)
    if type(depth) is not int or depth not in (2, 4):
        if fix_invalid:
            config[depth_key] = 2
        else:
            raise InvalidConfig(f"unsupported CuTe reduction pipeline depth: {depth!r}")
    if config[key] != "pipelined" and config.get(depth_key, 2) != 2:
        if fix_invalid:
            config[depth_key] = 2
        else:
            raise InvalidConfig(
                "CuTe reduction pipeline depth 4 requires the pipelined schedule"
            )


def _normalize_cute_store_policy(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_store_policy"
    if key not in config:
        return
    value = config[key]
    if value not in ("default", "streaming", "l2_evict_last"):
        if not fix_invalid:
            raise InvalidConfig(
                f"{key} must be one of ('default', 'streaming', 'l2_evict_last'), "
                f"got {value!r}"
            )
        config[key] = "default"
    if (
        config[key] == "l2_evict_last"
        and not config_spec.cute_l2_evict_last_store_policy_supported
    ):
        config[key] = "default"


def _normalize_cute_exact_div_form(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_exact_div_form`` picks how an fp32 division by a
    loop-invariant divisor is lowered on CuTe (both forms give the IEEE
    quotient bit for bit): ``redo`` speculates the Markstein sequence
    over the loop and re-runs the loop on a hard case, ``plain`` keeps
    ``div.rn`` per element.  A recorded config carrying the retired
    ``guard`` form is repaired to ``plain`` (fail closed: the plain
    division is what every form must equal), as is any unknown value
    under ``fix_invalid``.  A kernel without such a division has nothing
    for the knob to choose; it is dropped.  The gate
    (``has_hoistable_fp32_divisions``) is a graph-level necessary
    condition: a kernel whose division the AST pass then proves
    loop-carried at every loop keeps the knob, with both forms rendering
    identically."""
    key = "cute_exact_div_form"
    if key not in config:
        return
    value = config[key]
    if value in _CUTE_EXACT_DIV_FORM_RETIRED:
        config[key] = value = "plain"
    if value not in CUTE_EXACT_DIV_FORMS:
        if not fix_invalid:
            raise InvalidConfig(
                f"{key} must be one of {CUTE_EXACT_DIV_FORMS}, got {value!r}"
            )
        config[key] = "plain"
    if not config_spec.cute_exact_division_sites or config_spec.matmul_facts:
        config.pop(key, None)


def _normalize_cute_async_load_pipeline(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    keys = (
        "cute_async_load_stages",
        "cute_async_load_lookahead",
        "cute_async_load_group_rows",
        "cute_async_load_cache",
        "cute_async_store_policy",
    )
    if not config_spec.cute_async_load_pipeline_enabled:
        supplied = [key for key in keys if key in config]
        if supplied and not fix_invalid:
            raise InvalidConfig(
                "CuTe async state-load knobs require a compatible in-place "
                "vector state update"
            )
        for key in supplied:
            config.pop(key, None)
        return

    defaults: dict[str, object] = {
        "cute_async_load_stages": 0,
        "cute_async_load_lookahead": 4,
        "cute_async_load_group_rows": 2,
        "cute_async_load_cache": "cg",
        "cute_async_store_policy": "default",
    }
    choices: dict[str, tuple[object, ...]] = {
        "cute_async_load_stages": (0, 3, 4, 5),
        "cute_async_load_lookahead": (2, 3, 4),
        "cute_async_load_group_rows": (2, 4),
        "cute_async_load_cache": ("cg", "ca"),
        "cute_async_store_policy": ("default", "l2_evict_last"),
    }
    integer_keys = frozenset(
        {
            "cute_async_load_stages",
            "cute_async_load_lookahead",
            "cute_async_load_group_rows",
        }
    )
    lookahead_was_supplied = "cute_async_load_lookahead" in config
    for key in keys:
        value = config.setdefault(key, defaults[key])
        if value not in choices[key] or (
            key in integer_keys and type(value) is not int
        ):
            if fix_invalid:
                config[key] = defaults[key]
            else:
                raise InvalidConfig(
                    f"{key} must be one of {choices[key]!r}, got {value!r}"
                )
    if (
        config["cute_async_store_policy"] == "l2_evict_last"
        and not config_spec.cute_l2_evict_last_store_policy_supported
    ):
        # Old pinned configs remain loadable on other targets/toolchains,
        # where the transform has always failed closed to the default path.
        config["cute_async_store_policy"] = "default"
    stages = cast("int", config["cute_async_load_stages"])
    if stages > 0 and not lookahead_was_supplied:
        config["cute_async_load_lookahead"] = min(
            cast("int", defaults["cute_async_load_lookahead"]), stages - 1
        )
    lookahead = cast("int", config["cute_async_load_lookahead"])
    if stages == 0:
        for key in keys[1:]:
            config[key] = defaults[key]
    elif lookahead >= stages:
        if fix_invalid:
            config["cute_async_load_lookahead"] = stages - 1
        else:
            raise InvalidConfig(
                "cute_async_load_lookahead must be smaller than cute_async_load_stages"
            )


def _normalize_cute_signed_bitfield_bf16(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_signed_bitfield_bf16"
    value = config.get(key, False)
    if type(value) is bool and (
        not value or config_spec.cute_signed_bitfield_bf16_available
    ):
        if not value:
            config.pop(key, None)
        return
    if fix_invalid:
        config.pop(key, None)
        return
    raise InvalidConfig(
        f"{key}={value!r} requires a Boolean and a signed-byte BF16 candidate on sm_100a"
    )


def _normalize_cute_bf16x2_recurrence(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_bf16x2_recurrence"
    if not config_spec.cute_bf16x2_recurrence_enabled:
        if key in config and not fix_invalid:
            raise InvalidConfig(
                "cute_bf16x2_recurrence requires a compatible BF16 recurrence"
            )
        config.pop(key, None)
        return
    value = config.setdefault(key, False)
    if not isinstance(value, bool):
        if fix_invalid:
            config[key] = False
        else:
            raise InvalidConfig(f"{key} must be a boolean, got {value!r}")
    elif config.get("cute_async_load_stages", 0) == 0:
        config[key] = False


def _normalize_cute_vector_reductions(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    for key in (
        "cute_independent_reduction",
        "cute_cluster_warp_online",
        "cute_register_row",
        "cute_register_row_tensor_copy",
        "cute_scalar_epilogue",
        "cute_scalar_epilogue_keep_inputs",
        "cute_scalar_epilogue_owner_fold",
        "cute_warp_fp32_redux",
        "cute_register_tile_warp_fold",
        "cute_register_tile_store_first",
        "cute_replicated_reduction",
        "cute_vector_packet_unroll",
    ):
        if key in config and type(config[key]) is not bool:
            if fix_invalid:
                config[key] = False
            else:
                raise InvalidConfig(f"{key} must be a boolean")
    key = "cute_scalar_epilogue_owner_fold"
    if config.get(key) and not config.get("cute_scalar_epilogue"):
        if fix_invalid:
            config[key] = False
        else:
            raise InvalidConfig(f"{key} requires cute_scalar_epilogue")
    key = "cute_register_row_tensor_copy"
    if config.get(key) and not config.get("cute_register_row"):
        if fix_invalid:
            config[key] = False
        else:
            raise InvalidConfig(f"{key} requires cute_register_row")


def _normalize_cute_warp_aggregated_atomics(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_warp_aggregated_atomics"
    if not config_spec.cute_warp_aggregated_atomics_enabled:
        if key in config and not fix_invalid:
            raise InvalidConfig(
                "cute_warp_aggregated_atomics requires relaxed Int32 tickets"
            )
        config.pop(key, None)
        return
    value = config.setdefault(key, False)
    if type(value) is not bool:
        if fix_invalid:
            config[key] = False
        else:
            raise InvalidConfig(f"{key} must be a boolean, got {value!r}")


def _normalize_cute_pdl(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_pdl`` is a boolean (programmatic dependent launch);
    ``cute_pdl_release`` names where the kernel releases its dependents
    (``auto`` unless given: ``entry`` or ``none`` by the grid, resolved at
    codegen so a flat configuration, which always carries the value, and
    a dict without the key render alike) and only acts when ``cute_pdl``
    is on; ``cute_pdl_spread`` (a boolean) pads the kernel's shared
    memory to one wave's density and rides on ``cute_pdl`` the same way.
    With the knob off a non-default placement stays in the config
    (inert, like every other knob the kernel does not read) so a flat
    configuration survives the unflatten/flatten round trip the search
    relies on; the default placement is dropped there, so the default
    config and every knob-off config without a placement read alike.
    Config identity is therefore not canonical for this key: a knob-off
    config with ``none`` and one without the key, or a knob-on config
    with ``auto`` and one without the key, hash and compare as different
    configs while generating the same code."""
    pdl = config.get("cute_pdl", False)
    if type(pdl) is not bool:
        if not fix_invalid:
            raise InvalidConfig("cute_pdl must be a boolean")
        pdl = False
    if "cute_pdl" in config or pdl:
        config["cute_pdl"] = pdl
    release = config.get("cute_pdl_release", CUTE_PDL_RELEASE_DEFAULT)
    if release not in CUTE_PDL_RELEASE_CHOICES:
        if not fix_invalid:
            raise InvalidConfig(
                f"cute_pdl_release must be one of {CUTE_PDL_RELEASE_CHOICES!r}, "
                f"got {release!r}"
            )
        release = CUTE_PDL_RELEASE_DEFAULT
    if not pdl and release == CUTE_PDL_RELEASE_DEFAULT:
        config.pop("cute_pdl_release", None)
    elif "cute_pdl_release" in config:
        config["cute_pdl_release"] = release
    spread = config.get(CUTE_PDL_SPREAD_KEY, False)
    if type(spread) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{CUTE_PDL_SPREAD_KEY} must be a boolean")
        spread = False
    if not pdl and spread is False:
        config.pop(CUTE_PDL_SPREAD_KEY, None)
    elif CUTE_PDL_SPREAD_KEY in config:
        config[CUTE_PDL_SPREAD_KEY] = spread
    if config_spec.has_explicit_pdl:
        # Explicit markers own the wait/release; canonicalize all automatic
        # policies to one identity so search does not retime identical code.
        config["cute_pdl"] = False
        config.pop("cute_pdl_release", None)
        config.pop(CUTE_PDL_SPREAD_KEY, None)


def _normalize_cute_pdl_capture_overlap(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_pdl_capture_overlap"
    value = config.get(key, False)
    if type(value) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{key} must be a boolean")
        value = False
    eligible = (
        config_spec.cute_pdl_capture_overlap_available
        and not config_spec.has_explicit_pdl
        and config.get("cute_pdl") is True
        and not config.get(CUTE_PDL_SPREAD_KEY)
        and config.get("cute_pdl_release", CUTE_PDL_RELEASE_DEFAULT)
        in CUTE_PDL_CAPTURE_RELEASE_CHOICES
        and config.get(MMA_DAG_KEY) == MMA_DAG_TCGEN05
        and config.get(MMA_DAG_PASS_WARPS_KEY) == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
    )
    if value and not eligible and not fix_invalid:
        raise InvalidConfig(
            f"{key}=True requires a supported paired attention kernel, "
            "cute_pdl=True, cute_pdl_spread=False and automatic, entry or loader-tail release"
        )
    if config.get("cute_pdl_release") == "loader_tail" and not (eligible and value):
        if not fix_invalid:
            raise InvalidConfig(
                "cute_pdl_release='loader_tail' requires eligible "
                "cute_pdl_capture_overlap=True"
            )
        if config.get("cute_pdl") is True:
            config["cute_pdl_release"] = CUTE_PDL_RELEASE_DEFAULT
        else:
            config.pop("cute_pdl_release", None)
    if not eligible:
        # Unsupported families keep their prior config/search identity; an
        # explicit unused default must not leak into strict downstream plans.
        config.pop(key, None)
        return
    if key in config or value:
        config[key] = value


def _normalize_cute_tiles_per_block(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_tiles_per_block`` packs that many one-element grid tiles per
    CTA as thread sub-groups: a flat pid type, no CTA cluster split, no L2
    grouping (its pid decode has no pack term), and a packable root-grid
    axis whose static tile count it divides."""
    from ..tile_packing import TILES_PER_BLOCK_CHOICES
    from ..tile_packing import l2_grouped

    pack = config.get("cute_tiles_per_block", 1)
    if type(pack) is not int or pack not in TILES_PER_BLOCK_CHOICES:
        if not fix_invalid:
            raise InvalidConfig(
                f"cute_tiles_per_block must be one of {TILES_PER_BLOCK_CHOICES}"
            )
        pack = 1
    if pack > 1 and not (
        config.get("pid_type", "flat") == "flat"
        and config.get("cute_cluster_n", 1) == 1
        and not l2_grouped(config)
        and any(
            count % pack == 0 for count in config_spec.cute_tile_pack_counts.values()
        )
    ):
        if not fix_invalid:
            raise InvalidConfig(
                "cute_tiles_per_block needs a flat pid type, no cluster split, "
                "no L2 grouping and a one-element grid axis whose tile count "
                "it divides"
            )
        pack = 1
    if "cute_tiles_per_block" in config or pack != 1:
        config["cute_tiles_per_block"] = pack


def _normalize_cute_wide_packet_hints(
    config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_wide_packet_hints`` is a boolean, absent meaning False: the
    L2 policies of ``load_eviction_policies`` reach 32-byte packets
    through the SM100 ``_32b`` helpers only when it is set (and only on a
    toolchain with the 256-bit hinted load, which codegen checks the way
    it does for ``nc_l2_first``).  Not in the flat search layout: the knob
    is default-off until its effect is measured."""
    key = "cute_wide_packet_hints"
    if key not in config:
        return
    value = config[key]
    if type(value) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{key} must be a boolean")
        value = False
    config[key] = value


def cute_fragment_mma_fact(config_spec: ConfigSpec) -> MatmulFact | None:
    """The contraction ``cute_dot_fragment_mma`` may lower: one dot with a
    static N of 1 (a GEMV), fp16 or bf16 operands of one dtype and tunable
    M and K blocks (``cute/fragment_mma_dot.py``)."""
    for fact in config_spec.matmul_facts:
        if (
            fact.static_n == 1
            and fact.lhs_dtype == fact.rhs_dtype
            and fact.lhs_dtype in (torch.float16, torch.bfloat16)
            and fact.m_block_id is not None
            and fact.k_block_id is not None
        ):
            return fact
    return None


def _cute_fragment_mma_searchable(config_spec: ConfigSpec) -> bool:
    """Whether the fragment-MMA knob is offered to the search: a GEMV dot
    on a device with ``mma.sync`` (compute capability 8.0 and up)."""
    capability = config_spec.target_device_capability
    return (
        config_spec.supports_config_key(FRAGMENT_MMA_KEY)
        and cute_fragment_mma_fact(config_spec) is not None
        and capability is not None
        and capability >= (8, 0)
    )


def _cute_dot_fragment_mma_reason(
    config_spec: ConfigSpec, config: dict[str, object]
) -> str | None:
    """Why ``config`` cannot run the GEMV dot on MMA fragments (None: it can).

    The rewrite permutes the lanes of one warp into the ``m16n8k16``
    fragment layout, so it needs exactly eight threads on the dot's M
    block and four on its K block, and 16- or 32-byte K packets.
    """
    fact = cute_fragment_mma_fact(config_spec)
    if fact is None or fact.m_block_id is None or fact.k_block_id is None:
        return "needs a dot with fp16/bf16 operands and a static N of 1"
    capability = config_spec.target_device_capability
    if capability is None or capability < (8, 0):
        return "needs mma.sync (compute capability >= 8.0)"
    threads = _as_sequence(config.get("num_threads"))
    m_threads = config_spec.num_threads.config_get(threads, fact.m_block_id, 0)
    k_threads = config_spec.num_threads.config_get(threads, fact.k_block_id, 0)
    if (m_threads, k_threads) != (FRAGMENT_MMA_ROW_THREADS, FRAGMENT_MMA_K_THREADS):
        return (
            f"needs num_threads of {FRAGMENT_MMA_ROW_THREADS} on the dot's M "
            f"block and {FRAGMENT_MMA_K_THREADS} on its K block"
        )
    widths = _as_sequence(config.get("cute_vector_widths"))
    k_width = config_spec.cute_vector_widths.config_get(widths, fact.k_block_id, 1)
    if k_width not in (8, 16):
        return "needs 16- or 32-byte K packets (cute_vector_widths 8 or 16)"
    return None


def _normalize_cute_dot_fragment_mma(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_dot_fragment_mma`` is a boolean; True only where the dot's
    thread layout fits the fragment rewrite (``_cute_dot_fragment_mma_reason``)."""
    if FRAGMENT_MMA_KEY not in config:
        return
    value = config[FRAGMENT_MMA_KEY]
    if type(value) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{FRAGMENT_MMA_KEY} must be a boolean")
        config[FRAGMENT_MMA_KEY] = False
        return
    if not value:
        return
    if not _cute_fragment_mma_searchable(config_spec):
        # No dot the rewrite could take: the knob is invalid for the kernel.
        if not fix_invalid:
            raise InvalidConfig(
                f"{FRAGMENT_MMA_KEY}=True "
                f"{_cute_dot_fragment_mma_reason(config_spec, config)}"
            )
        config[FRAGMENT_MMA_KEY] = False
        return
    if _cute_dot_fragment_mma_reason(config_spec, config) is not None:
        # The thread layout cannot take the fragments: the knob is inert
        # for this config (like ``cute_lane_unroll`` without a lane loop).
        config[FRAGMENT_MMA_KEY] = False


def _cute_dot_ring_searchable(config_spec: ConfigSpec) -> bool:
    """Whether the ring-MMA knobs (``cute_dot_ring_mma`` and its stage and
    split knobs) are offered to the search: the GEMV dot of
    ``cute_fragment_mma_fact`` with a static K on a device with
    ``mma.sync`` and ``cp.async`` (compute capability 8.0 and up)."""
    fact = cute_fragment_mma_fact(config_spec)
    return (
        config_spec.supports_config_key(RING_MMA_KEY)
        and fact is not None
        and fact.static_k is not None
        and _cute_fragment_mma_searchable(config_spec)
    )


def _cute_dot_ring_mma_reason(
    config_spec: ConfigSpec, config: dict[str, object]
) -> str | None:
    """Why ``config`` cannot run the GEMV dot through the cp.async ring
    (None: it can): the rows must be a rolled lane loop (one thread on the
    dot's M block), the K threads whole warps, and the geometry
    ``ring_mma_geometry_reason`` accepts (``cute/ring_mma_dot.py``)."""
    fact = cute_fragment_mma_fact(config_spec)
    if (
        fact is None
        or fact.m_block_id is None
        or fact.k_block_id is None
        or fact.static_k is None
    ):
        return "needs a dot with fp16/bf16 operands, a static N of 1 and a static K"
    capability = config_spec.target_device_capability
    if capability is None or capability < (8, 0):
        return "needs mma.sync and cp.async (compute capability >= 8.0)"
    threads = _as_sequence(config.get("num_threads"))
    m_threads = config_spec.num_threads.config_get(threads, fact.m_block_id, 0)
    k_threads = config_spec.num_threads.config_get(threads, fact.k_block_id, 0)
    if m_threads != 1:
        return "needs one thread on the dot's M block (the rows are a rolled lane loop)"
    sizes = _as_sequence(config.get("block_sizes"))
    rows = config_spec.block_sizes.config_get(sizes, fact.m_block_id, None)
    chunk = config_spec.block_sizes.config_get(sizes, fact.k_block_id, None)
    if not isinstance(rows, int) or not isinstance(chunk, int):
        return "needs static M and K blocks"
    widths = _as_sequence(config.get("cute_vector_widths"))
    k_width = config_spec.cute_vector_widths.config_get(widths, fact.k_block_id, 1)
    stages = config.get(RING_STAGES_KEY, RING_DEFAULT_STAGES)
    split = config.get(RING_SPLIT_KEY, RING_DEFAULT_SPLIT)
    return ring_mma_geometry_reason(
        rows=rows,
        chunk=chunk,
        threads=int(cast("int", k_threads)),
        stages=cast("int", stages),
        split=cast("str", split),
        elem_bytes=fact.lhs_dtype.itemsize,
        k_width=int(cast("int", k_width)),
        smem_budget=ring_mma_smem_budget(config_spec.device),
    )


def _normalize_cute_dot_ring_mma(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_dot_ring_mma`` is a boolean; True only where the dot's thread
    layout and geometry fit the ring (``_cute_dot_ring_mma_reason``); the
    stage and split knobs are repaired to a fitting value when the ring is
    on and pinned to their defaults (inert) when it is off; under the ring
    ``cute_pdl_release`` is pinned to its default (the ring's own
    placement, after the last MMA, overrides it)."""
    if not any(
        key in config for key in (RING_MMA_KEY, RING_STAGES_KEY, RING_SPLIT_KEY)
    ):
        return
    value = config.get(RING_MMA_KEY, False)
    if type(value) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{RING_MMA_KEY} must be a boolean")
        value = False
    stages = config.get(RING_STAGES_KEY, RING_DEFAULT_STAGES)
    if type(stages) is not int or stages not in RING_STAGE_CHOICES:
        if not fix_invalid:
            raise InvalidConfig(
                f"{RING_STAGES_KEY} must be one of {RING_STAGE_CHOICES}"
            )
        stages = RING_DEFAULT_STAGES
    split = config.get(RING_SPLIT_KEY, RING_DEFAULT_SPLIT)
    if split not in RING_SPLIT_CHOICES:
        if not fix_invalid:
            raise InvalidConfig(f"{RING_SPLIT_KEY} must be one of {RING_SPLIT_CHOICES}")
        split = RING_DEFAULT_SPLIT
    if value and not _cute_dot_ring_searchable(config_spec):
        if not fix_invalid:
            raise InvalidConfig(
                f"{RING_MMA_KEY}=True {_cute_dot_ring_mma_reason(config_spec, config)}"
            )
        value = False
    if value:
        # The first fitting (split, stages) pair, the requested one first
        # (fewer stages when the ring would not fit in shared memory, the
        # other split when the warps cannot divide the requested one);
        # a layout no pair fits makes the knob inert (like
        # ``cute_lane_unroll`` without a lane loop).
        fits = False
        for candidate_split in (split, *(s for s in RING_SPLIT_CHOICES if s != split)):
            for candidate_stages in (
                stages,
                *sorted((s for s in RING_STAGE_CHOICES if s < stages), reverse=True),
            ):
                config[RING_STAGES_KEY] = candidate_stages
                config[RING_SPLIT_KEY] = candidate_split
                if _cute_dot_ring_mma_reason(config_spec, config) is None:
                    fits = True
                    break
            if fits:
                break
        if not fits:
            value = False
    config[RING_MMA_KEY] = value
    if not value:
        config[RING_STAGES_KEY] = RING_DEFAULT_STAGES
        config[RING_SPLIT_KEY] = RING_DEFAULT_SPLIT
    elif "cute_pdl_release" in config:
        # The ring releases its dependents after its last MMA whatever the
        # placement says (``cute/ring_mma_dot.py``): the key is inert.
        config["cute_pdl_release"] = CUTE_PDL_RELEASE_DEFAULT


def _normalize_cute_vloop_sink(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """``cute_vloop_sink`` is a boolean; ``cute_lane_unroll`` applies to
    the lane loops of a sunk vector nest or, without sinking, to the grid
    lane loops the config forms (``cute/unroll_lane_loads.py``), and is 1
    when the config has neither."""
    sink = config.get("cute_vloop_sink", False)
    if type(sink) is not bool:
        if not fix_invalid:
            raise InvalidConfig("cute_vloop_sink must be a boolean")
        sink = False
    if "cute_vloop_sink" in config or sink:
        config["cute_vloop_sink"] = sink
    unroll = config.get("cute_lane_unroll", 1)
    if type(unroll) is not int or unroll not in _CUTE_LANE_UNROLL_CHOICES:
        if not fix_invalid:
            raise InvalidConfig(
                f"cute_lane_unroll must be one of {_CUTE_LANE_UNROLL_CHOICES}"
            )
        unroll = 1
    if not sink and not _cute_config_forms_lane_loop(config_spec, config):
        unroll = 1
    if "cute_lane_unroll" in config or unroll != 1:
        config["cute_lane_unroll"] = unroll


def _cute_config_forms_lane_loop(
    config_spec: ConfigSpec, config: dict[str, object]
) -> bool:
    """Whether some tile axis (of the grid or of a device loop) gets a
    per-thread lane loop: an explicit thread count below its static block
    size that divides it (the condition ``PerThreadNDTileStrategy``
    allocates a lane variable on)."""
    # Runs before the sequences are normalized: a single-block kernel may
    # still spell them as scalars.
    nt_list = _as_sequence(config.get("num_threads"))
    bs_list = _as_sequence(config.get("block_sizes"))
    for block_id in config_spec.num_threads.valid_block_ids():
        nt = config_spec.num_threads.config_get(nt_list, block_id, 0)
        bs = config_spec.block_sizes.config_get(bs_list, block_id, 1)
        if isinstance(nt, int) and isinstance(bs, int) and 0 < nt < bs and bs % nt == 0:
            return True
    return False


def _normalize_cute_register_chain(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_register_chain"
    if key not in config:
        return
    value = config[key]
    supported = (
        bool(config_spec.matmul_facts)
        and config_spec.target_device_capability is not None
        and config_spec.target_device_capability[0] >= 8
    )
    if type(value) is not bool or (value and not supported):
        if fix_invalid:
            config[key] = False
        else:
            raise InvalidConfig(
                "cute_register_chain requires a boolean, matrix contractions, "
                "and CUDA compute capability >= 8.0"
            )


def _normalize_cute_simt_mma(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    from ..simt_gemm import SIMT_MMA_LAYOUTS

    key = "cute_simt_mma"
    if key in config:
        value = config[key]
        if type(value) is not bool or (value and not config_spec.cute_simt_mma_enabled):
            if fix_invalid:
                config[key] = False
            else:
                raise InvalidConfig(
                    "cute_simt_mma requires a boolean and an fp32 matrix contraction"
                )
    # The layout is inert while the GEMM is off (a flat field keeps its
    # default there, and the strict coverage round trip must stay the
    # identity); only an unknown value is invalid.
    layout_key = "cute_simt_mma_layout"
    if layout_key not in config:
        return
    if config[layout_key] not in SIMT_MMA_LAYOUTS:
        if fix_invalid:
            config[layout_key] = SIMT_MMA_LAYOUTS[0]
        else:
            raise InvalidConfig(
                f"cute_simt_mma_layout must be one of {SIMT_MMA_LAYOUTS}"
            )


def _normalize_cute_proven_bounds(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_proven_bounds"
    if not config_spec.cute_proven_bounds_enabled:
        if key in config and not fix_invalid:
            raise InvalidConfig(
                "cute_proven_bounds requires exact CuTe launch and tensor facts"
            )
        config.pop(key, None)
        return
    value = config.setdefault(key, False)
    if not isinstance(value, bool):
        if fix_invalid:
            config[key] = False
        else:
            raise InvalidConfig(f"{key} must be a boolean, got {value!r}")


def _normalize_cute_affine_scan(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    fragment = config_spec.cute_affine_scan_schedule
    if fragment is None:
        if CUTE_AFFINE_SCAN_SCHEDULE_KEY in config and not fix_invalid:
            raise InvalidConfig(
                "CuTe affine scan schedule requires a compatible affine scan"
            )
        config.pop(CUTE_AFFINE_SCAN_SCHEDULE_KEY, None)
        return

    value = config.setdefault(CUTE_AFFINE_SCAN_SCHEDULE_KEY, fragment.default())
    if type(value) is not str or value not in fragment.choices:
        if fix_invalid:
            config[CUTE_AFFINE_SCAN_SCHEDULE_KEY] = fragment.default()
        else:
            raise InvalidConfig(
                f"{CUTE_AFFINE_SCAN_SCHEDULE_KEY} must be one of "
                f"{fragment.choices!r}, got {value!r}"
            )


def _normalize_cute_rng_packet(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_rng_packet"
    value = config.get(key, False)
    if not config_spec.cute_rng_packet_enabled or type(value) is not bool:
        if key in config and not fix_invalid:
            raise InvalidConfig(
                "cute_rng_packet requires the philox4 stream and a boolean"
            )
        config.pop(key, None)
        return
    config.setdefault(key, False)


def _normalize_cute_packet_prefetch(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    key = "cute_packet_prefetch"
    value = config.get(key, 0)
    if type(value) is not int or value not in (0, 2, 4, 8):
        if not fix_invalid:
            raise InvalidConfig(f"{key} must be 0, 2, 4, or 8, got {value!r}")
        config.pop(key, None)
        return
    if value == 0:
        config.pop(key, None)
        return
    if (
        not config_spec.cute_packet_prefetch_enabled
        or not config_spec.cute_proven_bounds_enabled
        or config.get("cute_proven_bounds") is not True
    ):
        if not fix_invalid:
            raise InvalidConfig(
                "cute_packet_prefetch requires pointwise facts and cute_proven_bounds"
            )
        config.pop(key, None)


def _normalize_cute_topk(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    for key, choices in config_spec.cute_topk_choices.items():
        if not config_spec.cute_topk_search_enabled:
            if key in config and not fix_invalid:
                raise InvalidConfig(f"{key} requires a compatible top-k root")
            config.pop(key, None)
            continue
        value = config.setdefault(key, choices[0])
        if type(value) is not type(choices[0]) or value not in choices:
            if fix_invalid:
                config[key] = choices[0]
            else:
                raise InvalidConfig(f"{key} must be one of {choices!r}, got {value!r}")
    if config_spec.cute_topk_search_enabled:
        lanes = cast("int", config["cute_topk_lanes_per_row"])
        rows = cast("int", config["cute_topk_rows_per_block"])
        if lanes * rows > 1024:
            if fix_invalid:
                config["cute_topk_rows_per_block"] = 1024 // lanes
            else:
                raise InvalidConfig(
                    "cute_topk_lanes_per_row * cute_topk_rows_per_block "
                    "must not exceed 1024 threads"
                )

    key = "cute_topk_coarse_keys"
    value = config.get(key, False)
    if value is False:
        config.pop(key, None)
    elif not (
        value is True
        and config_spec.cute_topk_search_enabled
        and config_spec.cute_topk_coarse_keys_available
        and config.get("cute_topk_selection_layout") == "distributed"
        and config.get("cute_topk_key_dtype") == "int64"
    ):
        if fix_invalid:
            config.pop(key, None)
        else:
            raise InvalidConfig(
                f"{key} requires a complete FP32 distributed Int64-key selection"
            )

    key = "cute_topk_key_recovery"
    value = config.get(key, "direct")
    if value == "direct":
        config.pop(key, None)
    elif value != "packed" or config.get("cute_topk_coarse_keys") is not True:
        if fix_invalid:
            config.pop(key, None)
        else:
            raise InvalidConfig(
                f"{key} requires an enabled coarse-key selection and direct/packed mode"
            )


def _normalize_cute_mxfp8_root_lowering(
    config_spec: ConfigSpec, config: dict[str, object]
) -> None:
    """Collapse source-only choices for the proved complete native root.

    This proof covers every operation in the root. The native planner
    either replaces it completely or rejects compilation; it never falls
    back to the displaced lowering when cute_scaled_mma is selected.
    Scalar configurations retain their entire independent search space.
    """
    if (
        not config_spec.cute_mxfp8_mma_available
        or config.get("cute_scaled_mma") is not True
        or config_spec.target_device_capability is None
        or config_spec.target_device_capability[0] != 10
        or config.get("pid_type", config_spec.allowed_pid_types[0]) != "flat"
    ):
        return
    key = config_spec.cute_mxfp8_partition_key
    if key is not None and (type(config.get(key)) is not int or config.get(key) != 1):
        raise InvalidConfig(f"native complete MXFP8 reduction requires {key}=1")
    fields = config_spec._flat_fields()
    for key in _CUTE_MXFP8_ROOT_LOWERING_KEYS.intersection(config):
        if key in config_spec.user_defined_tunables:
            continue
        field = fields.get(key)
        if field is None:
            config.pop(key)
        else:
            config[key] = field._flat_config(
                config_spec, lambda fragment: fragment.default()
            )


def _drop_inert_workspace_knobs(config: Config) -> None:
    """The split-K workspace schedule is a seeded search alternative; the
    public default is the plain kernel, without the schedule's inert
    stage and partition knobs (flat search still represents them), so a
    GEMM's default does not depend on whether its operands admit the
    schedule's proof."""
    if config.config.get(SPLIT_K_WORKSPACE_KEY, False) is False:
        for key in (
            SPLIT_K_WORKSPACE_KEY,
            SPLIT_K_STAGES_KEY,
            SPLIT_K_PARTITIONS_KEY,
            SPLIT_K_ATOMIC_EPILOGUE_KEY,
        ):
            config.config.pop(key, None)


def normalize_entry(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    _normalize_cute_mxfp8_root_lowering(config_spec, config)
    if CUTE_COMPILER_PIPELINE_KEY in config:
        _normalize_enum_fragment(
            config,
            CUTE_COMPILER_PIPELINE_KEY,
            _COMPILER_PIPELINE_FRAGMENT,
            fix_invalid=fix_invalid,
        )

    if config.get("cute_fragment_producer_cache") is True and (
        config.get("cute_fragment_register_loads") is True
        or config.get("cute_fragment_warp_results") is True
    ):
        raise InvalidConfig(
            "shared producer caching cannot consume lane-private fragment storage"
        )
    if (
        config.get("cute_fragment_warp_results") is True
        and config.get("cute_fragment_register_loads") is True
    ):
        # These modes assign different physical owners to the same logical
        # element. Even repair-mode search must reject, not silently pick
        # one storage policy and benchmark a different configuration.
        raise InvalidConfig(
            "cute_fragment_warp_results and cute_fragment_register_loads "
            "have incompatible physical ownership"
        )


def _require_family_enabled(
    config_spec: ConfigSpec,
    config: dict[str, object],
    keys: Sequence[str],
    enabled: bool,
    reason: str,
    *,
    fix_invalid: bool,
) -> None:
    """Drop (repair) or reject a family's keys when its emitter is not admitted."""
    for key in keys:
        if key in config and not enabled and config_spec.supports_config_key(key):
            if fix_invalid:
                config.pop(key)
            else:
                raise InvalidConfig(f"{key} is available only for {reason}")


def normalize_family_gates(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    normalize_value_metadata_options(config_spec, config, fix_invalid=fix_invalid)
    for keys, enabled, reason in (
        (
            (CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,),
            config_spec.cute_chunk_recurrence_dv_partitions is not None,
            "matched BT16 chunk-recurrence kernels",
        ),
        (
            (CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,),
            config_spec.cute_chunk_recurrence_register_cap is not None,
            "matched BT16 chunk-recurrence kernels",
        ),
        (
            (CUTE_GDN_RECURRENCE_STAGES_KEY,),
            config_spec.cute_gdn_recurrence_stages is not None,
            "matched gated-delta-rule chunk-recurrence kernels",
        ),
        (
            (CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,),
            config_spec.cute_gdn_recurrence_epilogue_warps is not None,
            "matched gated-delta-rule chunk-recurrence kernels",
        ),
        (
            (CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,),
            config_spec.cute_gdn_recurrence_token_groups is not None,
            "matched gated-delta-rule chunk-recurrence kernels",
        ),
        (
            (CUTE_GDN_RECURRENCE_MMA_M_KEY,),
            config_spec.cute_gdn_recurrence_mma_m is not None,
            "matched gated-delta-rule chunk-recurrence kernels",
        ),
        (
            RSB_KEYS,
            config_spec.cute_register_subblocks is not None,
            "roots the register sub-block family admits",
        ),
        (
            MMA_DAG_KEYS,
            config_spec.cute_mma_dag is not None,
            "matched warp-MMA state recurrence kernels",
        ),
        (
            (CUTE_CHUNK_PREPARE_SCHEDULE_KEY,),
            config_spec.cute_chunk_prepare_schedule is not None,
            "matched BT16 chunk-prepare kernels",
        ),
    ):
        _require_family_enabled(
            config_spec, config, keys, enabled, reason, fix_invalid=fix_invalid
        )

    if (
        FLASH_KV_STAGE_KEY in config
        and not config_spec.cute_flash_search_enabled
        and not config_spec.cute_flash_gated_search_enabled
        and config_spec.supports_config_key(FLASH_KV_STAGE_KEY)
    ):
        # The KV ring depth has no consumer on the scalar path. A config
        # pinned from a flash autotune must still run when the surface is
        # off (another sequence length, ``HELION_CUTE_FLASH=0``, a GPU
        # without tcgen05), so the key is dropped rather than rejected;
        # the flash surfaces validate their own values.
        config.pop(FLASH_KV_STAGE_KEY)
    from ..cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY

    if (
        FLASH_GATE_WARPGROUPS_KEY in config
        and not config_spec.cute_flash_gated_search_enabled
    ):
        # Same rule for the gate warpgroup count: only the gated body reads it.
        config.pop(FLASH_GATE_WARPGROUPS_KEY)
    if (
        "cute_flash_gated_row_warps" in config
        and not config_spec.cute_flash_gated_search_enabled
    ):
        # ... and for the gated row programs' warp count.
        config.pop("cute_flash_gated_row_warps")


def normalize_knobs(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    normalize_cluster_schedule(
        config,
        available=config_spec.cute_split_k_cluster_facts is not None,
        fix_invalid=fix_invalid,
    )
    validate_cluster_config(config_spec, config, fix_invalid=fix_invalid)
    normalize_cluster_finalizer(
        config,
        available=config_spec.cute_split_k_cluster_facts is not None,
        fix_invalid=fix_invalid,
    )
    normalize_workspace_config(
        config,
        available=config_spec.cute_split_k_workspace_available,
        fix_invalid=fix_invalid,
        partition_choices=config_spec.cute_split_k_partition_choices,
        atomic_epilogue_available=config_spec.cute_split_k_atomic_epilogue_available,
    )
    normalize_grouped_rna_config(
        config,
        available_k=config_spec.cute_grouped_rna_k_choices,
        rn_two_stage_ctas=config_spec.cute_grouped_rn_two_stage_ctas,
        block_prefix_recipes=config_spec.cute_grouped_block_prefix_recipes,
        warp_raw_available=config_spec.cute_grouped_warp_tf32_available,
        wrapped_descriptors_available=config_spec.cute_grouped_wrapped_descriptors_available,
        fix_invalid=fix_invalid,
    )
    surface.prepare_normalization(config_spec.tcgen05, config, fix_invalid=fix_invalid)
    _normalize_cute_reduction_schedule(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_reduction_local_tree(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_reduction_row_output(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_reduction_sequence(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_host_paired_sum(config_spec, config, fix_invalid=fix_invalid)
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_scan", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_reduction", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fp32_dot", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_half_mma", fix_invalid=fix_invalid
    )
    _normalize_cute_fragment_threads(config_spec, config, fix_invalid=fix_invalid)
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_producer_cache", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_warp_scan", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_topk_network", fix_invalid=fix_invalid
    )
    _normalize_cute_fragment_cta_topk(config_spec, config, fix_invalid=fix_invalid)
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_atomic_aggregation", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_fragment_integer_atomic_epochs",
        fix_invalid=fix_invalid,
    )
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_fragment_local_atomic_registers",
        fix_invalid=fix_invalid,
    )
    _normalize_fragment_option(
        config_spec, config, "cute_integer_loop_reduction", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_register_snapshots", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_fragment_atomic_consumer_fusion",
        fix_invalid=fix_invalid,
    )
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_fragment_pure_producer_regions",
        fix_invalid=fix_invalid,
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_packet_loads", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_padded_rows", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_unrolled_loops", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_vector_io", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_batched_loads", fix_invalid=fix_invalid
    )
    _normalize_cute_fragment_split_k(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_fragment_mma_rows(config_spec, config, fix_invalid=fix_invalid)
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_slab_loads", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_register_loads", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_warp_results", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_fragment_private_scalar_loops",
        fix_invalid=fix_invalid,
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_published_scalars", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_skip_zero_atomics", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_bounded_gather", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_register_producers", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec, config, "cute_fragment_scan_exports", fix_invalid=fix_invalid
    )
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_fragment_warp_producer_regions",
        fix_invalid=fix_invalid,
    )
    _normalize_cute_materialized_schedule(config_spec, config, fix_invalid=fix_invalid)
    _normalize_fragment_option(
        config_spec,
        config,
        "cute_materialized_operand_schedule",
        fix_invalid=fix_invalid,
    )
    _normalize_cute_pointwise_pid_type(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_async_load_pipeline(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_store_policy(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_exact_div_form(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_bf16x2_recurrence(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_signed_bitfield_bf16(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_proven_bounds(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_affine_scan(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_topk(config_spec, config, fix_invalid=fix_invalid)
    normalize_radix_selection(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_rng_packet(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_vector_reductions(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_warp_aggregated_atomics(
        config_spec, config, fix_invalid=fix_invalid
    )
    _normalize_cute_vloop_sink(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_wide_packet_hints(config, fix_invalid=fix_invalid)
    _normalize_cute_pdl(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_dot_fragment_mma(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_dot_ring_mma(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_tiles_per_block(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_packet_prefetch(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_register_chain(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_simt_mma(config_spec, config, fix_invalid=fix_invalid)
    if config_spec.matmul_facts:
        value = config.setdefault("cute_collective_mma", False)
        if not isinstance(value, bool):
            if fix_invalid:
                config["cute_collective_mma"] = False
            else:
                raise InvalidConfig("cute_collective_mma must be a boolean")
        for key in (
            "cute_collective_native_seeded",
            "cute_collective_tmem_seed",
            "cute_collective_tmem_a",
        ):
            seeded = config.setdefault(key, False)
            if not isinstance(seeded, bool):
                if fix_invalid:
                    config[key] = False
                else:
                    raise InvalidConfig(f"{key} must be a boolean")
            if seeded is True and (
                config_spec.target_device_capability is None
                or config_spec.target_device_capability[0] != 10
            ):
                if fix_invalid:
                    config[key] = False
                else:
                    raise InvalidConfig(f"{key} requires SM100-family hardware")
        packet_key = "cute_collective_operand_packets"
        if not isinstance(config.get(packet_key, False), bool):
            if fix_invalid:
                config[packet_key] = False
            else:
                raise InvalidConfig(f"{packet_key} must be a boolean")
        copy = config.setdefault("cute_collective_copy", "scalar")
        if copy not in ("scalar", "async", "async_cached"):
            if fix_invalid:
                config["cute_collective_copy"] = "scalar"
            else:
                raise InvalidConfig(
                    "cute_collective_copy must be scalar, async, or async_cached"
                )
        for key in ("cute_collective_recipe", "cute_collective_epilogue"):
            recipe = config.get(key, "scalar")
            if recipe not in ("scalar", "vector", "vector_unrolled"):
                if fix_invalid:
                    config[key] = "scalar"
                else:
                    raise InvalidConfig(
                        f"{key} must be scalar, vector, or vector_unrolled"
                    )
        compute = config.setdefault("cute_collective_compute", "warp")
        if compute not in ("warp", "tcgen05", "tma_gather"):
            if fix_invalid:
                config["cute_collective_compute"] = "warp"
            else:
                raise InvalidConfig(
                    "cute_collective_compute must be warp, tcgen05, or tma_gather"
                )
        if compute in ("tcgen05", "tma_gather") and (
            config_spec.target_device_capability is None
            or config_spec.target_device_capability[0] != 10
        ):
            if fix_invalid:
                config["cute_collective_compute"] = "warp"
            else:
                raise InvalidConfig(
                    "native collective compute requires SM100-family hardware"
                )
        for key, choices, default in (
            ("cute_collective_stages", (1, 2, 3, 4), 1),
            ("cute_gathered_mma_n", (128, 256, 512), 256),
            ("cute_gathered_mma_stages", (2, 3, 4), 3),
        ):
            if key in config and (
                type(config[key]) is not int or config[key] not in choices
            ):
                if fix_invalid:
                    config[key] = default
                else:
                    raise InvalidConfig(f"{key} must be one of {choices}")
        if compute == "tma_gather":
            n = config.setdefault("cute_gathered_mma_n", 256)
            stages = config.setdefault("cute_gathered_mma_stages", 3)
            if n == 512 and stages != 2:
                if fix_invalid:
                    config["cute_gathered_mma_stages"] = 2
                else:
                    raise InvalidConfig("512-column gathered MMA requires two stages")
    elif any(
        key in config
        for key in (
            "cute_collective_mma",
            "cute_collective_copy",
            "cute_collective_recipe",
            "cute_collective_operand_packets",
            "cute_collective_epilogue",
            "cute_collective_stages",
            "cute_collective_compute",
            "cute_collective_native_seeded",
            "cute_collective_tmem_seed",
            "cute_collective_tmem_a",
            "cute_gathered_mma_n",
            "cute_gathered_mma_stages",
        )
    ):
        if not fix_invalid:
            raise InvalidConfig("cute_collective_mma requires a matrix contraction")
        config.pop("cute_collective_mma", None)
        config.pop("cute_collective_copy", None)
        config.pop("cute_collective_recipe", None)
        config.pop("cute_collective_operand_packets", None)
        config.pop("cute_collective_epilogue", None)
        config.pop("cute_collective_stages", None)
        config.pop("cute_collective_compute", None)
        config.pop("cute_collective_native_seeded", None)
        config.pop("cute_collective_tmem_seed", None)
        config.pop("cute_collective_tmem_a", None)
        config.pop("cute_gathered_mma_n", None)
        config.pop("cute_gathered_mma_stages", None)

    key = "cute_collective_static_layouts"
    value = config.get(key, False)
    if value is False:
        config.pop(key, None)
    elif (
        value is True
        and config_spec.matmul_facts
        and config.get("cute_collective_mma")
        and config.get("cute_collective_compute", "warp") == "warp"
    ):
        pass
    elif fix_invalid:
        config.pop(key, None)
    else:
        raise InvalidConfig(
            "cute_collective_static_layouts requires warp collective MMA"
        )


def normalize_before_sequences(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if fix_invalid:
        _pre_normalize_cute_flash_block_sizes(config_spec, config)


def normalize_after_sequences(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    # A persistent reduction with exactly one vector fragment per
    # thread has only one physical assignment, so blocked/strided are
    # identical.  Canonicalize the inactive choice to avoid duplicate
    # autotuner candidates.  Looped reductions retain both layouts.
    lane_layouts = config.get("cute_lane_layouts")
    reduction_loops = cast("list[int | None]", config.get("reduction_loops", []) or [])
    if isinstance(lane_layouts, list):
        canonical_layouts = list(lane_layouts)
        for index, layout_spec in enumerate(config_spec.cute_lane_layouts):
            block_id = layout_spec.block_id
            if (
                block_id in config_spec.reduction_block_ids
                and config_spec.reduction_loops.config_get(
                    reduction_loops, block_id, None
                )
                is None
            ):
                canonical_layouts[index] = "blocked"
        config["cute_lane_layouts"] = canonical_layouts


def normalize_reduction_threads(
    config_spec: ConfigSpec,
    config: dict[str, object],
    *,
    fix_invalid: bool,
    register_tiles: bool,
) -> None:
    register_tile = False
    if config_spec.max_reduction_threads is not None and config_spec.reduction_loops:
        nt_list = cast("list[int]", config.get("num_threads", []) or [])
        bs_list = cast("list[int]", config.get("block_sizes", []) or [])
        # ``num_threads`` also carries per-rolled-rdim slots (the
        # reduction's own thread count), plus static persistent-rdim slots
        # that have no ``reduction_loops`` entry.  Only NON-reduction tile
        # axes consume the budget the reduction competes for.  Resolve all
        # values by block id because the two sequences need not share an
        # order.
        reduction_block_ids = config_spec.reduction_block_ids | {
            spec.block_id for spec in config_spec.reduction_loops
        }
        other_threads = 1
        for nt_spec in config_spec.num_threads:
            if nt_spec.block_id in reduction_block_ids:
                continue
            nt = config_spec.num_threads.config_get(nt_list, nt_spec.block_id, 0)
            if not isinstance(nt, int) or nt <= 0:
                bs = config_spec.block_sizes.config_get(bs_list, nt_spec.block_id, 1)
                nt = bs if isinstance(bs, int) and bs > 1 else 1
            if nt > 1:
                other_threads *= nt
        available = max(1, config_spec.max_reduction_threads // other_threads)
        reduction_loops = config.get("reduction_loops", [])
        if isinstance(reduction_loops, list):
            new_loops = list(reduction_loops)
            changed = False
            for i, spec in enumerate(config_spec.reduction_loops):
                if i >= len(new_loops):
                    break
                if spec.persistent_reason is not None:
                    continue
                if new_loops[i] is None and spec.size_hint > available:
                    # When other_threads consumes the entire CuTe 1024 thread
                    # budget there is no thread budget left for the reduction
                    # axis. A chunk of 1 is invalid (LoopedReductionStrategy
                    # requires block_size > 1) and a persistent reduction
                    # could not place a single reduction thread.  Reject
                    # the config so the autotuner skips it.
                    if available < 2:
                        raise InvalidConfig(
                            f"cute backend: reduction axis {i} has no thread "
                            f"budget left (non-reduction axes use "
                            f"{other_threads} of {config_spec.max_reduction_threads} "
                            f"threads)."
                        )
                    forms_register_tile = (
                        register_tiles
                        and _cute_config_forms_register_tile(
                            config_spec, config, spec, available
                        )
                    )
                    if forms_register_tile or _cute_config_forms_one_vector_lane(
                        config_spec, config, spec, available
                    ):
                        # Only the register tile honours the schedule
                        # knobs; a one-vector lane that is not one keeps
                        # its lane loop but drops them below.
                        register_tile = register_tile or forms_register_tile
                        continue
                    chunk = min(spec.size_hint, available)
                    if config_spec.max_reduction_loop is not None:
                        chunk = min(chunk, config_spec.max_reduction_loop)
                    new_loops[i] = chunk
                    changed = True
                elif (
                    new_loops[i] is None
                    and register_tiles
                    and isinstance(
                        requested := config_spec.num_threads.config_get(
                            nt_list, spec.block_id, 0
                        ),
                        int,
                    )
                    and 0 < requested < spec.size_hint
                    and _cute_config_forms_register_tile(
                        config_spec, config, spec, available
                    )
                ):
                    # An explicit reduction thread count below the extent
                    # within the budget (a lane loop of >= 2 rows per
                    # thread): the register tile forms here too.
                    register_tile = True
            if changed:
                config["reduction_loops"] = new_loops
    if not register_tile:
        # The register-tile schedule knobs mean nothing without the tile
        # (a one-vector persistent lane over an admitted reduction block);
        # dropping them keeps the proven schedule and a byte-identical
        # render, also on the ``register_tiles=False`` retry.
        for key in _CUTE_REGISTER_TILE_KNOBS:
            if config.get(key):
                config[key] = False


def normalize_after_defaults(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    normalize_block_scaled_config(
        config,
        available=config_spec.cute_scaled_mma_available,
        capability=config_spec.target_device_capability,
        fix_invalid=fix_invalid,
        mxfp8=config_spec.cute_mxfp8_mma_available,
        fused_scales_available=config_spec.cute_mxfp8_fused_scales_available,
        sf_tma_available=config_spec.cute_mxfp8_sf_tma_available,
        tma_output_available=config_spec.cute_mxfp8_tma_output_available,
    )


def _normalize_enum_fragment(
    config: dict[str, object],
    key: str,
    fragment: EnumFragment,
    *,
    fix_invalid: bool,
    guard: Literal["any", "int", "int_or_none"] = "any",
) -> object:
    """Default ``key`` from ``fragment`` and validate (or repair) its value.

    ``guard`` names the type check in front of the membership test: ``"int"``
    rejects non-integers, ``"int_or_none"`` admits None beside integers and
    ``"any"`` only tests membership.
    """
    value = config.setdefault(key, fragment.default())
    if guard == "int":
        invalid = type(value) is not int or value not in fragment.choices
    elif guard == "int_or_none":
        invalid = (
            value is not None and type(value) is not int
        ) or value not in fragment.choices
    else:
        invalid = value not in fragment.choices
    if invalid:
        if fix_invalid:
            config[key] = fragment.default()
        else:
            raise InvalidConfig(
                f"{key} must be one of {fragment.choices!r}, got {value!r}"
            )
    return config[key]


def normalize_recurrence_families(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    recurrence_dv_fragment = config_spec.cute_chunk_recurrence_dv_partitions
    if recurrence_dv_fragment is not None:
        _normalize_enum_fragment(
            config,
            CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
            recurrence_dv_fragment,
            fix_invalid=fix_invalid,
            guard="int",
        )
    recurrence_register_cap_fragment = config_spec.cute_chunk_recurrence_register_cap
    if recurrence_register_cap_fragment is not None:
        _normalize_enum_fragment(
            config,
            CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,
            recurrence_register_cap_fragment,
            fix_invalid=fix_invalid,
            guard="int_or_none",
        )
        if recurrence_dv_fragment is not None and not (
            _cute_chunk_recurrence_config_is_safe(
                config[CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY],
                config[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY],
            )
        ):
            if fix_invalid:
                config[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] = None
            else:
                raise InvalidConfig(
                    f"{CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY} must be None for "
                    f"{CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY}=2 because the "
                    "TMEM schedule dynamically reallocates registers"
                )
    gdn_stages_supplied = CUTE_GDN_RECURRENCE_STAGES_KEY in config
    gdn_groups_supplied = CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY in config
    gdn_mma_m_supplied = CUTE_GDN_RECURRENCE_MMA_M_KEY in config
    for gdn_key, gdn_fragment in (
        (CUTE_GDN_RECURRENCE_STAGES_KEY, config_spec.cute_gdn_recurrence_stages),
        (
            CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
            config_spec.cute_gdn_recurrence_epilogue_warps,
        ),
        (
            CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
            config_spec.cute_gdn_recurrence_token_groups,
        ),
        (CUTE_GDN_RECURRENCE_MMA_M_KEY, config_spec.cute_gdn_recurrence_mma_m),
    ):
        if gdn_fragment is not None:
            _normalize_enum_fragment(
                config, gdn_key, gdn_fragment, fix_invalid=fix_invalid, guard="int"
            )
    # The tcgen05 M follows the tile and warps, the ring depth the tile
    # and M, the token groups all three: validate in that order.
    gdn_tile_mma_m_choices = _cute_gdn_recurrence_mma_m_choices_for(config_spec, config)
    if gdn_tile_mma_m_choices is not None:
        gdn_mma_m = config[CUTE_GDN_RECURRENCE_MMA_M_KEY]
        if gdn_mma_m not in gdn_tile_mma_m_choices:
            # A defaulted MMA height follows the selected tile and warps;
            # only an explicit height they cannot take is an error.
            if fix_invalid or not gdn_mma_m_supplied:
                config[CUTE_GDN_RECURRENCE_MMA_M_KEY] = gdn_tile_mma_m_choices[0]
            else:
                raise InvalidConfig(
                    f"{CUTE_GDN_RECURRENCE_MMA_M_KEY} must be one of "
                    f"{gdn_tile_mma_m_choices!r} for the selected dstate "
                    f"tile and epilogue warps, got {gdn_mma_m!r}"
                )
    gdn_tile_stage_choices = _cute_gdn_recurrence_stage_choices_for(config_spec, config)
    if gdn_tile_stage_choices is not None:
        gdn_stages = config[CUTE_GDN_RECURRENCE_STAGES_KEY]
        if gdn_stages not in gdn_tile_stage_choices:
            # A depth left to the default follows the selected tile; only
            # an explicit depth the tile cannot hold is an error.
            if fix_invalid or not gdn_stages_supplied:
                config[CUTE_GDN_RECURRENCE_STAGES_KEY] = gdn_tile_stage_choices[0]
            else:
                raise InvalidConfig(
                    f"{CUTE_GDN_RECURRENCE_STAGES_KEY} must be one of "
                    f"{gdn_tile_stage_choices!r} for the selected dstate "
                    f"tile, got {gdn_stages!r}"
                )
    gdn_tile_group_choices = _cute_gdn_recurrence_token_group_choices_for(
        config_spec, config
    )
    if gdn_tile_group_choices is not None:
        gdn_groups = config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY]
        if gdn_groups not in gdn_tile_group_choices:
            # A defaulted group count follows the selected tile and warps;
            # only an explicit count they cannot split is an error.
            if fix_invalid or not gdn_groups_supplied:
                config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] = gdn_tile_group_choices[0]
            else:
                raise InvalidConfig(
                    f"{CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY} must be one of "
                    f"{gdn_tile_group_choices!r} for the selected dstate "
                    f"tile and epilogue warps, got {gdn_groups!r}"
                )
    if config_spec.cute_mma_dag is not None:
        _normalize_cute_mma_dag(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_pdl_capture_overlap(config_spec, config, fix_invalid=fix_invalid)
    if config_spec.cute_register_subblocks is not None:
        _normalize_cute_register_subblocks(config_spec, config, fix_invalid=fix_invalid)
    prepare_schedule_fragment = config_spec.cute_chunk_prepare_schedule
    if prepare_schedule_fragment is not None:
        _normalize_enum_fragment(
            config,
            CUTE_CHUNK_PREPARE_SCHEDULE_KEY,
            prepare_schedule_fragment,
            fix_invalid=fix_invalid,
        )


def normalize_pre_pid_type(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    repair.normalize_pre_pid_type(
        config_spec.tcgen05,
        config,
        fix_invalid=fix_invalid,
    )


def normalize_strategy(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if fix_invalid:
        repair.fix_search_config(config_spec.tcgen05, config)

    repair.normalize_strategy(
        config_spec.tcgen05,
        config,
        fix_invalid=fix_invalid,
    )
    repair.validate_cp_async_a_config(
        config_spec.tcgen05, config, fix_invalid=fix_invalid
    )
    _normalize_cute_flash(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_flash_gated(config_spec, config, fix_invalid=fix_invalid)
    _normalize_cute_flash_bwd(config_spec, config)

    if config_spec.multi_phase and config.get("num_sm_multiplier", 1) != 1:
        # The grid barrier between the phases waits for every CTA, and the
        # CuTe launcher has no cooperative launch to reject a grid larger
        # than the GPU holds at once: such a launch would hang.
        if fix_invalid:
            config.pop("num_sm_multiplier", None)
        else:
            raise InvalidConfig(
                "CuTe multi-phase kernels (hl.barrier() or a cross-loop "
                "dependency) require num_sm_multiplier=1: their grid "
                "barrier needs every CTA resident at once"
            )


def normalize_finalize(
    config_spec: ConfigSpec,
    config: dict[str, object],
    *,
    fix_invalid: bool,
    provided_keys: set[str],
) -> None:
    preserve_keys = surface.implicit_default_keys_to_preserve(
        config_spec.tcgen05, config
    )
    if config_spec.allowed_pid_types[0] != "flat":
        # ``Config.pid_type`` falls back to ``flat`` for a missing key,
        # so a persistent choice is not an implicit default.
        preserve_keys = {*preserve_keys, "pid_type"}
    if config.get(MMA_DAG_PERSISTENT_KEY):
        # The persistent dag launch's program ids (set by its knob).
        preserve_keys = {*preserve_keys, "pid_type", "num_sm_multiplier"}
    for key in CUTE_IMPLICIT_DEFAULT_KEYS - provided_keys - preserve_keys:
        config.pop(key, None)

    validate_cluster_config(config_spec, config, fix_invalid=fix_invalid)
    normalize_cluster_finalizer(
        config,
        available=config_spec.cute_split_k_cluster_facts is not None,
        fix_invalid=fix_invalid,
    )


def pdl_fields(
    config_spec: ConfigSpec,
    fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment],
) -> None:
    """Add the generic dependent-launch knobs: ``cute_pdl`` (a launch
    attribute plus one wait ahead of the body's first global access,
    searched where ``griddepcontrol`` exists, sm_90 and up) and
    ``cute_pdl_release``.

    Where a ``cute_pdl`` kernel releases its own dependents: ``auto``
    (the default) resolves at codegen to ``entry`` while the launch grid
    fits in one wave of resident CTAs and to ``last_wave`` past it,
    ``entry`` right after the wait, so the dependents launch while this
    kernel runs (their CTAs sit resident at their own wait), ``exit``
    after the body's last statement, ``none`` leaves the implicit trigger
    at grid completion, ``last_wave`` releases at entry in the last wave's
    CTAs only (plain entry for a one-wave grid).  Inert when ``cute_pdl``
    is off (the default value is then normalized away).  The search
    leaves the placement at ``auto``: a kernel whose back-to-back calls
    are separated by a plain launch (the zero fill of an output) cannot
    observe its own placement in isolation, so a searched placement is a
    coin flip there, while the kernel's dependent in the real call
    sequence pays for a non-releasing pick (mamba2_ssd backward: dqkg
    alone reads 3.66 us under every placement; the sequence reads 11.2 us
    per call with dqkg at ``none`` and 10.3 at ``entry``).  The explicit
    values stay valid for pins, seeds and users. Capture overlap and its
    ``loader_tail`` release require an explicit opt-in: their storage proof
    assumes captured topology and parameter addresses remain immutable, which
    ordinary autotuning cannot establish for the application's future graphs.
    """
    if not config_spec.supports_config_key("cute_pdl"):
        return
    searched = pdl_searched(config_spec)
    fields["cute_pdl"] = EnumFragment(
        (False, True), search_choices=(False, True) if searched else (False,)
    )
    if config_spec.cute_pdl_capture_overlap_available:
        fields["cute_pdl_capture_overlap"] = EnumFragment(
            (False, True), search_choices=(False,)
        )
    fields["cute_pdl_release"] = EnumFragment(
        CUTE_PDL_RELEASE_CHOICES,
        search_choices=(CUTE_PDL_RELEASE_DEFAULT,),
    )
    # Pad the kernel's shared memory to one wave's density so an early
    # launch cannot pack its CTAs onto the SMs its primary left idle
    # (``pdl_release.spread_pad_bytes``).  Searched with the knob: the
    # pad is a tie in the kernel's isolated timing (the dispatcher
    # spreads a fresh grid the same way) and the final selection prefers
    # it among ties (``CuteBackend.finalist_preference``), while a kernel
    # the pad slows (less L1 behind a larger carveout) keeps the knob
    # without it.
    fields[CUTE_PDL_SPREAD_KEY] = EnumFragment(
        (False, True), search_choices=(False, True) if searched else (False,)
    )


def cute_flat_fields(
    config_spec: ConfigSpec, flash_pipeline_family: str | None
) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
    fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
        "block_sizes": config_spec.block_sizes,
    }

    if config_spec.cute_signed_bitfield_bf16_available:
        fields["cute_signed_bitfield_bf16"] = BooleanFragment()
    if config_spec.cute_host_paired_sum_available:
        fields["cute_host_paired_sum"] = EnumFragment(
            choices=("off", "mapped", "narrow")
        )
    if config_spec.cute_materialized_schedule_search_enabled:
        fields["cute_materialized_schedule"] = EnumFragment(
            choices=config_spec.cute_materialized_schedule_choices,
            search_choices=("off",) if config_spec.has_explicit_pdl else None,
        )
    if config_spec.cute_materialized_operand_schedule_search_enabled:
        fields["cute_materialized_operand_schedule"] = EnumFragment(
            choices=("off", "warp_narrow4"),
            search_choices=("off",) if config_spec.has_explicit_pdl else None,
        )
    if config_spec.cute_fragment_private_scalar_loops_search_enabled:
        fields["cute_fragment_private_scalar_loops"] = BooleanFragment()
    if config_spec.cute_fragment_producer_cache_search_enabled:
        fields["cute_fragment_producer_cache"] = BooleanFragment()
    if config_spec.cute_fragment_warp_results_search_enabled:
        fields["cute_fragment_warp_results"] = BooleanFragment()
    if config_spec.cute_topk_coarse_keys_search_enabled:
        fields["cute_topk_coarse_keys"] = BooleanFragment()
    if config_spec.cute_topk_key_recovery_search_enabled:
        fields["cute_topk_key_recovery"] = EnumFragment(choices=("direct", "packed"))
    if config_spec.cute_fragment_atomic_aggregation_search_enabled:
        fields["cute_fragment_atomic_aggregation"] = BooleanFragment()
    if config_spec.cute_fragment_integer_atomic_epochs_search_enabled:
        fields["cute_fragment_integer_atomic_epochs"] = BooleanFragment()
    if config_spec.cute_fragment_packet_loads_search_enabled:
        fields["cute_fragment_packet_loads"] = BooleanFragment()
    if config_spec.cute_fragment_padded_rows_search_enabled:
        fields["cute_fragment_padded_rows"] = BooleanFragment()
    if config_spec.cute_fragment_unrolled_loops_search_enabled:
        fields["cute_fragment_unrolled_loops"] = BooleanFragment()
    # Fragment MMA and tile transport booleans (one statement: this sits at a
    # size past which pyrefly drops its narrowing of the fields below).
    fields.update(
        {
            key: BooleanFragment()
            for enabled, key in (
                (
                    config_spec.cute_fragment_vector_io_search_enabled,
                    "cute_fragment_vector_io",
                ),
                (
                    config_spec.cute_fragment_batched_loads_search_enabled,
                    "cute_fragment_batched_loads",
                ),
                (
                    config_spec.cute_fragment_split_k_search_enabled,
                    "cute_fragment_split_k",
                ),
                (
                    config_spec.cute_fragment_mma_rows_search_enabled,
                    "cute_fragment_mma_rows",
                ),
                (
                    config_spec.cute_fragment_slab_loads_search_enabled,
                    "cute_fragment_slab_loads",
                ),
                (
                    config_spec.cute_fragment_half_mma_search_enabled,
                    "cute_fragment_half_mma",
                ),
            )
            if enabled
        }
    )
    if config_spec.cute_fragment_warp_scan_search_enabled:
        fields["cute_fragment_warp_scan"] = BooleanFragment()
    if config_spec.cute_fragment_topk_network_search_enabled:
        fields["cute_fragment_topk_network"] = BooleanFragment()
    if config_spec.cute_fragment_cta_topk_search_enabled:
        fields["cute_fragment_cta_topk"] = BooleanFragment()
    if config_spec.cute_fragment_local_atomic_registers_search_enabled:
        fields["cute_fragment_local_atomic_registers"] = BooleanFragment()
    if config_spec.cute_integer_loop_reduction_search_enabled:
        fields["cute_integer_loop_reduction"] = BooleanFragment()
    if config_spec.cute_fragment_register_snapshots_search_enabled:
        fields["cute_fragment_register_snapshots"] = BooleanFragment()
    if config_spec.cute_fragment_bounded_gather_search_enabled:
        fields["cute_fragment_bounded_gather"] = BooleanFragment()
    if config_spec.cute_fragment_register_producers_search_enabled:
        fields["cute_fragment_register_producers"] = BooleanFragment()
    if config_spec.cute_fragment_warp_producer_regions_search_enabled:
        fields["cute_fragment_warp_producer_regions"] = BooleanFragment()
    if config_spec.cute_fragment_scan_exports_search_enabled:
        fields["cute_fragment_scan_exports"] = BooleanFragment()
    if config_spec.cute_fragment_published_scalars_search_enabled:
        fields["cute_fragment_published_scalars"] = BooleanFragment()
    if config_spec.cute_fragment_skip_zero_atomics_search_enabled:
        fields["cute_fragment_skip_zero_atomics"] = BooleanFragment()
    if config_spec.cute_fragment_atomic_consumer_fusion_search_enabled:
        fields["cute_fragment_atomic_consumer_fusion"] = BooleanFragment()
    if config_spec.cute_fragment_pure_producer_regions_search_enabled:
        fields["cute_fragment_pure_producer_regions"] = BooleanFragment()
    if config_spec.cute_fragment_register_loads_search_enabled:
        fields["cute_fragment_register_loads"] = BooleanFragment()
    if config_spec.cute_fragment_threads_search_enabled:
        from .heuristics.cute_fragment_threads import THREADS

        fields["cute_fragment_threads"] = EnumFragment(choices=THREADS)
    if config_spec.cute_fragment_scan_search_enabled:
        fields["cute_fragment_scan"] = EnumFragment(choices=("serial", "cooperative"))
    if config_spec.cute_fragment_reduction_search_enabled:
        fields["cute_fragment_reduction"] = EnumFragment(
            choices=("serial", "warp", "cta", "cta_unroll")
        )
    if config_spec.cute_fp32_dot_search_enabled:
        fields["cute_fp32_dot"] = EnumFragment(choices=("simt", "tf32_mma"))
    if config_spec.cute_pointwise_region_block_ids and any(
        pid_type != "flat" for pid_type in config_spec.allowed_pid_types
    ):
        # The pointwise launch may go flat while the native launch
        # stays persistent; when every launch is flat already (ordered
        # barrier phases) the knob is inert and stays off the surface.
        fields["cute_pointwise_pid_type"] = EnumFragment(choices=("inherit", "flat"))
    if config_spec.cute_topk_search_enabled:
        fields.update(
            (key, EnumFragment(choices=choices))
            for key, choices in config_spec.cute_topk_choices.items()
        )
    elif config_spec.tcgen05.search_enabled:
        fields.update(surface.flat_fields(config_spec.tcgen05))
        if config_spec.cute_pointwise_region_block_ids:
            fields["num_threads"] = config_spec.num_threads
            fields["cute_vector_widths"] = config_spec.cute_vector_widths
            fields["cute_lane_layouts"] = config_spec.cute_lane_layouts
    elif config_spec.cute_flash_search_enabled:
        fields.update(
            _cute_flash_autotune_fragments(
                config_spec,
                pipeline_family_override=flash_pipeline_family,
            )
        )
    elif config_spec.cute_flash_gated_search_enabled:
        from ..cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY

        default = config_spec._cute_flash_gated_kv_stage_default
        fields[FLASH_KV_STAGE_KEY] = EnumFragment(
            choices=(
                default,
                *(
                    choice
                    for choice in config_spec._cute_flash_gated_kv_stage_choices
                    if choice != default
                ),
            )
        )
        wg_default = config_spec._cute_flash_gated_gate_warpgroup_default
        fields[FLASH_GATE_WARPGROUPS_KEY] = EnumFragment(
            choices=(
                wg_default,
                *(
                    choice
                    for choice in config_spec._cute_flash_gated_gate_warpgroup_choices
                    if choice != wg_default
                ),
            )
        )
        if config_spec._cute_flash_gated_row_warp_choices_by_tile:
            from ..cute_flash_gated_row_mma import GATED_ROW_WARPS_KEY
            from ..cute_flash_gated_row_mma import gated_row_warps_default

            row_warps = tuple(
                sorted(
                    {
                        w
                        for warps in config_spec._cute_flash_gated_row_warp_choices_by_tile.values()
                        for w in warps
                    }
                )
            )
            row_default = gated_row_warps_default(row_warps)
            fields[GATED_ROW_WARPS_KEY] = EnumFragment(
                choices=(row_default, *(w for w in row_warps if w != row_default))
            )
    elif config_spec.cute_flash_bwd_search_enabled:
        fields["cute_flash_bwd_persistent"] = EnumFragment(
            choices=(0, 1) if config_spec._cute_flash_bwd_persistent_allowed else (0,)
        )
        fields["cute_flash_bwd_two_cta"] = EnumFragment(
            choices=(0, 1) if config_spec._cute_flash_bwd_two_cta_allowed else (0,)
        )
        # The first choice is the default config's: the f32 exp2 pair
        # measured 2-5% faster on the flex walks (and causal / d64 dense),
        # the packed f16x2 pair on the dense d128 non-causal shapes.
        fields["cute_flash_bwd_exp2_f32"] = EnumFragment(
            choices=(1, 0) if config_spec._cute_flash_bwd_flex else (0, 1)
        )
        row_facts = config_spec._cute_flash_bwd_row_facts
        if (
            not config_spec._cute_flash_bwd_flex
            and row_facts is not None
            and row_facts.admitted
        ):
            # The register-MMA row programs (cute_flash_bwd_row_mma):
            # searched beside the tcgen05 family where its grid is
            # small, legal everywhere the knobs divide the shape.
            row_search = flash_bwd_row_search_grid(row_facts)
            fields[FLASH_BWD_FAMILY_KEY] = EnumFragment(
                choices=FLASH_BWD_FAMILIES,
                search_choices=(
                    FLASH_BWD_FAMILIES if row_search else (FLASH_BWD_FA4_FAMILY,)
                ),
            )
            for key, choices in (
                (FLASH_BWD_ROW_KV_KEY, FLASH_BWD_ROW_KV_CHOICES),
                (FLASH_BWD_ROW_Q_SPLIT_KEY, FLASH_BWD_ROW_Q_SPLIT_CHOICES),
                (FLASH_BWD_ROW_Q_CHUNK_KEY, FLASH_BWD_ROW_Q_CHUNK_CHOICES),
            ):
                fields[key] = EnumFragment(
                    choices=choices,
                    search_choices=choices if row_search else choices[:1],
                )
        if config_spec._cute_flash_bwd_flex:
            if config_spec._cute_flash_bwd_has_mask:
                # How the mask_mod predicate reaches the -inf select of
                # the flex backward's P pass (see flash_score_mod).
                fields["cute_flash_bwd_mask_form"] = EnumFragment(
                    choices=("bitmask", "element")
                )
            from ..cute_flash_bwd_flex import FBWD_REG_SPLITS

            # Register budget per warp role (reduce / compute / other).
            fields["cute_flash_bwd_regs"] = EnumFragment(choices=FBWD_REG_SPLITS)
            if config_spec._cute_flash_bwd_has_hook:
                # Lanes per hook + exp2 group of the P pass (live-range control).
                fields["cute_flash_bwd_hook_cols"] = EnumFragment(choices=(32, 16, 8))
                # Order the hook coordinate math after the S wait (see fbwd_pin_i32).
                fields["cute_flash_bwd_hook_pin"] = EnumFragment(choices=(1, 0))
            if config_spec._cute_flash_bwd_additive_bias:
                # Fold an additive score bias into the LSE term (see additive_bias_region).
                fields["cute_flash_bwd_bias_fold"] = EnumFragment(choices=(1, 0))
            # Block-index -> (batch-head, KV tile) walk: heaviest-first
            # inside L2-sized batch-head groups (see fbwd_tile_group_choices).
            fields["cute_flash_bwd_tile_group"] = EnumFragment(
                choices=config_spec._cute_flash_bwd_tile_groups
            )
            if config_spec._cute_flash_bwd_stats_mma_allowed:
                # Per-column LSE / delta terms through K-slice MMAs
                # instead of broadcast smem loads in the compute passes
                # (see _fit_smem_budget; not on the 2-CTA pair plan).
                fields["cute_flash_bwd_stats_mma"] = EnumFragment(choices=(0, 1))
            if config_spec._cute_flash_bwd_s_first_allowed:
                # The tile's first S MMA ahead of the dO / V waits and
                # the first dO load ahead of the stats gather (the plain
                # TMEM plans; see emit_flash_bwd_flex_device_body).
                # "auto" follows the form (on for the persistent walk,
                # off for a one-tile CTA); the search spells it out.
                fields["cute_flash_bwd_s_first"] = EnumFragment(
                    choices=("auto", 0, 1), search_choices=(0, 1)
                )
            if config_spec._cute_flash_bwd_kv_stage_allowed:
                # Two K / V ring slots on the persistent walk: the next
                # tile's K / V load while the current tile computes
                # (one slot beside the Q / dO rings otherwise; see
                # _fit_smem_budget).
                fields["cute_flash_bwd_kv_stage"] = EnumFragment(choices=(1, 2))
            from ..cute_flash import flash_flex_aux_smem_choices

            aux_choices = flash_flex_aux_smem_choices(
                config_spec._cute_flash_bwd_aux_smem_candidates
            )
            if len(aux_choices) > 1:
                # Byte budget of the hook's gathered aux tables staged in
                # shared memory (0 = global gathers; see fbwd_aux_smem_layout).
                fields["cute_flash_bwd_aux_smem"] = EnumFragment(choices=aux_choices)
            if config_spec._cute_flash_bwd_ds_tmem_allowed:
                # The dK MMA's dS^T operand from TMEM (TS mode) instead
                # of the smem dS staging (head dim 64).
                fields["cute_flash_bwd_ds_tmem"] = EnumFragment(choices=(0, 1))
            if len(config_spec._cute_flash_bwd_table_step_choices) > 1:
                # Steps per step-table buffer, i.e. per chunk of a KV
                # tile's walk (0 = automatic: the whole walk when it fits,
                # else the largest table that does); see
                # cute_flash_bwd_flex.fbwd_table_step_choices.
                fields["cute_flash_bwd_table_steps"] = EnumFragment(
                    choices=config_spec._cute_flash_bwd_table_step_choices
                )
            if config_spec._cute_flash_bwd_fast_math:
                # Elements of every 16 sent to the polynomial exp2 on the
                # FMA pipe (an approximation: fast_math kernels only).
                fields["cute_flash_bwd_exp2_emu"] = EnumFragment(choices=(0, 8, 4))
        # The fused backward bodies (the tcgen05 families and the
        # register-MMA rows) take the generic dependent launch: the
        # wait lands ahead of the body's first global access and the
        # release follows the grid rule, as in any generic kernel.
        # On the small grids the row family serves, the backward is
        # one kernel of a chain (a preprocess kernel ahead of it), and
        # the overlapped launch is worth a few hundred nanoseconds of
        # a ~6 us kernel.
        pdl_fields(config_spec, fields)
    elif config_spec.supports_config_key("num_threads"):
        fields["num_threads"] = config_spec.num_threads
        # Loop flattening is a real codegen choice on the SIMT path
        # (flattened multi-dim tiles vectorize odd-row-length
        # pointwise kernels via flat base pointers).  Without this
        # entry the flat form was unreachable: seeds carrying
        # ``flatten_loops=[True]`` silently lost the flag in the
        # flat-config round trip and were benchmarked as their
        # (much slower) N-D counterparts.
        if (
            config_spec.supports_config_key("flatten_loops")
            and len(config_spec.flatten_loops) > 0
        ):
            fields["flatten_loops"] = config_spec.flatten_loops
        # Rolled-reduction loop chunks are tunable on the SIMT path
        # (LoopedReductionStrategy lane lattices).  Without this
        # entry the chunk silently pins to its default and neither
        # the seeds nor the search can change it.
        if (
            config_spec.supports_config_key("reduction_loops")
            and len(config_spec.reduction_loops) > 0
        ):
            fields["reduction_loops"] = config_spec.reduction_loops
        # Per-load-site L1 eviction hints (streamed rows want
        # evict_first; re-read rows want evict_last until the
        # final sweep).
        if (
            config_spec.supports_config_key("load_eviction_policies")
            and config_spec.load_eviction_policies.length > 0
        ):
            fields["load_eviction_policies"] = config_spec.load_eviction_policies
        # Universal pid emission honors ``loop_orders`` and the
        # better order is shape-dependent. tcgen05 exposes the same
        # field from tcgen05_config.surface.flat_fields().
        if (
            config_spec.supports_config_key("loop_orders")
            and len(config_spec.loop_orders) > 0
        ):
            fields["loop_orders"] = config_spec.loop_orders
        # Expose ``cute_vector_widths`` per-block so the
        # autotuner can vary V in {1, 2, 4, 8} for lane-loop
        # vec loads (and for ``LoopedReductionStrategy`` rolled
        # reductions).  Without this entry, ``flatten`` strips V
        # back to the default of 1, defeating the seed
        # heuristics that try to bias toward LDG.128 lattices.
        if (
            config_spec.supports_config_key("cute_vector_widths")
            and len(config_spec.cute_vector_widths) > 0
        ):
            fields["cute_vector_widths"] = config_spec.cute_vector_widths
        # Per-block lane layout (blocked vs strided per-thread
        # element assignment) for lane loops with more than one
        # iteration; strided gives fully coalesced warp accesses.
        if (
            config_spec.supports_config_key("cute_lane_layouts")
            and len(config_spec.cute_lane_layouts) > 0
        ):
            fields["cute_lane_layouts"] = config_spec.cute_lane_layouts
        # Where multi-sweep rolled reductions keep re-read values
        # (register fragment vs gmem/L2 reload); see
        # ``CuteReductionReloadSpec``.
        if (
            config_spec.supports_config_key("cute_reduction_reloads")
            and len(config_spec.cute_reduction_reloads) > 0
        ):
            fields["cute_reduction_reloads"] = config_spec.cute_reduction_reloads
        if config_spec.cute_sequence_reduction_blocks:
            fields["cute_reduction_sequence"] = EnumFragment(
                choices=(
                    "scalar",
                    "reload",
                    "resident",
                    "bounded",
                    "bounded_layout",
                ),
                search_choices=("scalar", "reload", "resident")
                if config_spec.has_explicit_pdl
                else None,
            )
        if config_spec.cute_resident_reduction_blocks:
            fields["cute_reduction_local_tree"] = BooleanFragment()
            fields["cute_reduction_schedule"] = EnumFragment(
                choices=("scalar", "resident", "pipelined")
            )
            fields["cute_reduction_group_rows"] = EnumFragment(choices=(1, 2, 4, 3))
            fields["cute_reduction_row_schedule"] = EnumFragment(
                choices=("batched", "serial", "serial_deferred")
            )
            if (
                config_spec.target_device_capability is not None
                and config_spec.target_device_capability[0] == 10
            ):
                fields["cute_reduction_pack_output"] = BooleanFragment()
            fields["cute_reduction_pipeline_depth"] = EnumFragment(choices=(2, 4))
        # Cache policy of the full 16-byte global packet flushes:
        # ``streaming`` (``st.global.cs``, evict-first) keeps an
        # output written once from displacing re-read inputs;
        # ``l2_evict_last`` needs the validated SM103 descriptor.
        fields["cute_store_policy"] = EnumFragment(
            choices=(
                ("default", "streaming", "l2_evict_last")
                if config_spec.cute_l2_evict_last_store_policy_supported
                else ("default", "streaming")
            )
        )
        if config_spec.cute_exact_division_sites and not config_spec.matmul_facts:
            # How an fp32 quotient by a loop-invariant divisor is
            # lowered; bitwise identical forms, see
            # ``_normalize_cute_exact_div_form``.
            fields["cute_exact_div_form"] = EnumFragment(choices=CUTE_EXACT_DIV_FORMS)
        if config_spec.cute_async_load_pipeline_enabled:
            fields["cute_async_load_stages"] = EnumFragment(choices=(0, 3, 4, 5))
            fields["cute_async_load_lookahead"] = EnumFragment(choices=(4, 2, 3))
            fields["cute_async_load_group_rows"] = EnumFragment(choices=(2, 4))
            fields["cute_async_load_cache"] = EnumFragment(choices=("cg", "ca"))
            fields["cute_async_store_policy"] = EnumFragment(
                choices=(
                    ("default", "l2_evict_last")
                    if config_spec.cute_l2_evict_last_store_policy_supported
                    else ("default",)
                )
            )
        if config_spec.cute_bf16x2_recurrence_enabled:
            fields["cute_bf16x2_recurrence"] = BooleanFragment()
        # The split-K workspace proof holds for the atomic example and
        # for plain tcgen05 GEMM stores alike; the latter has no
        # collective ``matmul_facts``, so the knobs live outside them.
        if config_spec.cute_split_k_workspace_available:
            fields.update(
                workspace_search_fields(
                    config_spec.cute_split_k_partition_choices,
                    atomic_epilogue_available=config_spec.cute_split_k_atomic_epilogue_available,
                )
            )
        if config_spec.matmul_facts:
            if (
                config_spec.cute_grouped_rna_k_choices
                or config_spec.cute_grouped_warp_tf32_available
            ):
                fields[CUTE_MMA_F32_CONVERSION_KEY] = EnumFragment(
                    choices=(CUTE_MMA_F32_AUTO,)
                    + (
                        (CUTE_MMA_F32_TMA_RN,)
                        if config_spec.cute_grouped_rna_k_choices
                        else ()
                    )
                    + (
                        (CUTE_MMA_F32_WARP_RAW,)
                        if config_spec.cute_grouped_warp_tf32_available
                        else ()
                    ),
                    search_choices=(CUTE_MMA_F32_AUTO,)
                    if config_spec.has_explicit_pdl
                    else None,
                )
            if config_spec.cute_grouped_rna_k_choices:
                fields[GROUPED_RNA_WARPS_KEY] = EnumFragment(
                    choices=GROUPED_RNA_WARPS,
                    search_choices=(0,) if config_spec.has_explicit_pdl else None,
                )
                fields[GROUPED_RNA_K_KEY] = EnumFragment(
                    choices=config_spec.cute_grouped_rna_k_choices
                )
                if config_spec.cute_grouped_rn_two_stage_ctas:
                    fields[GROUPED_RNA_STAGES_KEY] = EnumFragment(choices=(0, 2))
                    fields[GROUPED_RNA_RESIDENT_CTAS_KEY] = EnumFragment(
                        choices=config_spec.cute_grouped_rn_two_stage_ctas
                    )
                if config_spec.cute_grouped_wrapped_descriptors_available:
                    fields[DESCRIPTOR_KEY] = EnumFragment(choices=(DYNAMIC, WRAPPED))
                if config_spec.cute_grouped_block_prefix_recipes:
                    fields[GROUPED_PREFIX_SCAN_KEY] = EnumFragment(
                        choices=GROUPED_PREFIX_SCANS
                    )
            if config_spec.cute_split_k_cluster_facts is not None:
                fields[SPLIT_K_SCHEDULE_KEY] = EnumFragment(
                    SPLIT_K_SCHEDULES,
                    search_choices=("legacy",)
                    if config_spec.has_explicit_pdl
                    else None,
                )
                fields[SPLIT_K_FINALIZER_KEY] = EnumFragment(FINALIZER_WARPS)
            if config_spec.cute_collective_knobs_enabled:
                fields["cute_collective_mma"] = BooleanFragment()
                fields["cute_collective_static_layouts"] = BooleanFragment()
                capability = config_spec.target_device_capability
                native_collective = capability is not None and capability[0] == 10
                gathered_collective = native_collective and any(
                    seed.config.get("cute_collective_compute") == "tma_gather"
                    for seed in config_spec.compiler_seed_configs
                )
                fields["cute_collective_compute"] = EnumFragment(
                    choices=("warp", "tcgen05", "tma_gather")
                    if native_collective
                    else ("warp",),
                    search_choices=("warp", "tcgen05", "tma_gather")
                    if gathered_collective
                    else ("warp", "tcgen05")
                    if native_collective
                    else ("warp",),
                )
                fields["cute_collective_native_seeded"] = (
                    BooleanFragment()
                    if native_collective
                    else EnumFragment(choices=(False,))
                )
                fields["cute_collective_tmem_seed"] = (
                    BooleanFragment()
                    if native_collective
                    else EnumFragment(choices=(False,))
                )
                fields["cute_collective_tmem_a"] = (
                    BooleanFragment()
                    if native_collective
                    else EnumFragment(choices=(False,))
                )
                fields["cute_collective_stages"] = EnumFragment(choices=(1, 2, 3, 4))
                fields["cute_collective_copy"] = EnumFragment(
                    choices=("scalar", "async", "async_cached")
                )
                fields["cute_collective_recipe"] = EnumFragment(
                    choices=("scalar", "vector", "vector_unrolled")
                )
                fields["cute_collective_operand_packets"] = EnumFragment(
                    choices=(False, True),
                    search_choices=(False, True)
                    if any(
                        fact.lhs_dtype == torch.float32
                        for fact in config_spec.matmul_facts
                    )
                    else (False,),
                )
                fields["cute_collective_epilogue"] = EnumFragment(
                    choices=("scalar", "vector", "vector_unrolled")
                )
                fields["cute_gathered_mma_n"] = EnumFragment(
                    choices=(256, 128, 512),
                    search_choices=(256, 128, 512) if gathered_collective else (256,),
                )
                fields["cute_gathered_mma_stages"] = EnumFragment(
                    choices=(3, 2, 4),
                    search_choices=(3, 2, 4) if gathered_collective else (3,),
                )
        if config_spec.cute_proven_bounds_enabled:
            fields["cute_proven_bounds"] = BooleanFragment()
        if config_spec.cute_warp_aggregated_atomics_enabled:
            fields["cute_warp_aggregated_atomics"] = BooleanFragment()
        if len(config_spec.cute_lane_layouts) > 0 and not config_spec.matmul_facts:
            for key in (
                "cute_independent_reduction",
                "cute_cluster_warp_online",
                "cute_register_row",
                "cute_register_row_tensor_copy",
                "cute_scalar_epilogue",
                "cute_scalar_epilogue_keep_inputs",
                "cute_scalar_epilogue_owner_fold",
                "cute_warp_fp32_redux",
                "cute_register_tile_warp_fold",
                "cute_register_tile_store_first",
                "cute_replicated_reduction",
                "cute_vector_packet_unroll",
            ):
                seeded = any(
                    seed.config.get(key) is True
                    for seed in config_spec.compiler_seed_configs
                )
                fields[key] = EnumFragment(
                    (False, True),
                    search_choices=(False, True) if seeded else (False,),
                )
        fragment_mma_searchable = _cute_fragment_mma_searchable(config_spec)
        if fragment_mma_searchable:
            # The GEMV dot on mma.sync fragments; configs whose thread
            # layout cannot take it normalize to False.
            fields[FRAGMENT_MMA_KEY] = EnumFragment((False, True))
        if _cute_dot_ring_searchable(config_spec):
            # The GEMV dot through a cp.async ring on mma.sync; the
            # normalizer repairs the stage / split knobs to a fitting
            # pair and makes the knob inert where the layout cannot
            # take the ring.
            fields[RING_MMA_KEY] = EnumFragment((False, True))
            fields[RING_STAGES_KEY] = EnumFragment(RING_STAGE_CHOICES)
            fields[RING_SPLIT_KEY] = EnumFragment(RING_SPLIT_CHOICES)
        if len(config_spec.cute_lane_layouts) > 0 and (
            not config_spec.matmul_facts or fragment_mma_searchable
        ):
            sink_seeded = any(
                seed.config.get("cute_vloop_sink") is True
                for seed in config_spec.compiler_seed_configs
            )
            unroll_seeded = any(
                cast("int", seed.config.get("cute_lane_unroll", 1)) > 1
                for seed in config_spec.compiler_seed_configs
            )
            fields["cute_vloop_sink"] = EnumFragment(
                (False, True),
                search_choices=(False, True) if sink_seeded else (False,),
            )
            fields["cute_lane_unroll"] = EnumFragment(
                _CUTE_LANE_UNROLL_CHOICES,
                search_choices=_CUTE_LANE_UNROLL_CHOICES
                if sink_seeded or unroll_seeded
                else (1,),
            )
        # A launch attribute plus one wait, live on every kernel of
        # this branch (the tcgen05 family, whose own waits and plans
        # leave it inert, builds its fields above; the fused flash
        # backward adds the same two fields to its own branch).
        pdl_fields(config_spec, fields)
        if config_spec.cute_rng_packet_enabled:
            fields["cute_rng_packet"] = BooleanFragment()
        if config_spec.cute_packet_prefetch_enabled:
            fields["cute_packet_prefetch"] = EnumFragment(choices=(0, 2, 4, 8))
        capability = config_spec.target_device_capability
        if config_spec.matmul_facts and capability is not None and capability[0] >= 8:
            chain_seeded = any(
                seed.config.get("cute_register_chain") is True
                for seed in config_spec.compiler_seed_configs
            )
            fields["cute_register_chain"] = EnumFragment(
                (False, True),
                search_choices=(False, True) if chain_seeded else (False,),
            )
        if config_spec.cute_simt_mma_enabled:
            from ..simt_gemm import SIMT_MMA_LAYOUTS

            fields["cute_simt_mma"] = EnumFragment((False, True))
            fields["cute_simt_mma_layout"] = EnumFragment(SIMT_MMA_LAYOUTS)
        # CuTe's SIMT search normally has no pid_type coordinate.  A
        # metadata-specialized compiler seed may nevertheless prove one
        # exact 3-D ``xyz`` launch safe after the earlier, deliberately
        # conservative grid-size gate rejected it.  Preserve that seed
        # through flatten/unflatten while keeping ``xyz`` out of random
        # search when it is absent from ``allowed_pid_types``.
        if config_spec.supports_config_key("pid_type") and any(
            seed.config.get("pid_type") == "xyz"
            for seed in config_spec.compiler_seed_configs
        ):
            searchable_pid_types = tuple(
                pid_type
                for pid_type in config_spec.allowed_pid_types
                if pid_type in ("flat", "xyz")
            )
            if searchable_pid_types:
                choices = tuple(dict.fromkeys((*searchable_pid_types, "xyz")))
                fields["pid_type"] = EnumFragment(
                    choices,
                    search_choices=searchable_pid_types,
                )
        # Thread-block cluster width for SIMT reduction kernels:
        # splits a whole-extent lane-looped axis across cluster
        # CTAs (register-resident slices) with a DSM cluster
        # reduce.  Ignored (cluster 1) when the config shape
        # doesn't qualify — see
        # ``cute/lane_cluster_split.py::apply_cluster_split``.
        if (
            config_spec.supports_config_key("cute_cluster_n")
            and len(config_spec.cute_lane_layouts) > 0
            and not config_spec.matmul_facts
        ):
            fields["cute_cluster_n"] = EnumFragment(choices=(1, 2, 4, 8, 16))
        # Minimum resident CTAs per SM (ptxas ``minnctapersm``):
        # caps registers per thread so more CTAs fit, trading
        # spills for occupancy.  0 leaves ptxas free.  6 is a
        # hard squeeze (at 128 threads it caps registers at 85 ->
        # 24 warps/SM); measured on B200 softmax rows, one extra
        # resident CTA beats ~10 spilled registers.
        if (
            config_spec.supports_config_key("cute_min_blocks_per_mp")
            and len(config_spec.cute_lane_layouts) > 0
            and not config_spec.matmul_facts
        ):
            fields["cute_min_blocks_per_mp"] = EnumFragment(choices=(0, 1, 2, 3, 4, 6))
        # Several one-element grid tiles per CTA (thread sub-groups):
        # tiny-tile kernels are CTA-dispatch bound.
        if (
            config_spec.supports_config_key("cute_tiles_per_block")
            and config_spec.cute_tile_pack_counts
            and not config_spec.matmul_facts
        ):
            from ..tile_packing import TILES_PER_BLOCK_CHOICES

            fields["cute_tiles_per_block"] = EnumFragment(
                choices=TILES_PER_BLOCK_CHOICES
            )
    # The fused flash bodies (forward and backward) emit their own
    # epilogues; the subtiled store split rewrites the graphs the
    # attention matchers read, so the knob leaves the search space.
    if (
        not config_spec.cute_flash_search_enabled
        and not config_spec.cute_flash_bwd_search_enabled
        and config_spec.epilogue_subtile_autotune_choices is not None
    ):
        fields["epilogue_subtile"] = EnumFragment(
            choices=config_spec.epilogue_subtile_autotune_choices
        )
    if config_spec.cute_chunk_recurrence_dv_partitions is not None:
        fields[CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY] = (
            config_spec.cute_chunk_recurrence_dv_partitions
        )
    if config_spec.cute_chunk_recurrence_register_cap is not None:
        fields[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] = (
            config_spec.cute_chunk_recurrence_register_cap
        )
    if config_spec.cute_gdn_recurrence_stages is not None:
        fields[CUTE_GDN_RECURRENCE_STAGES_KEY] = config_spec.cute_gdn_recurrence_stages
    if config_spec.cute_gdn_recurrence_epilogue_warps is not None:
        fields[CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY] = (
            config_spec.cute_gdn_recurrence_epilogue_warps
        )
    if config_spec.cute_gdn_recurrence_token_groups is not None:
        fields[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] = (
            config_spec.cute_gdn_recurrence_token_groups
        )
    if config_spec.cute_gdn_recurrence_mma_m is not None:
        fields[CUTE_GDN_RECURRENCE_MMA_M_KEY] = config_spec.cute_gdn_recurrence_mma_m
    if config_spec.cute_mma_dag is not None:
        fields[MMA_DAG_KEY] = config_spec.cute_mma_dag
        assert config_spec.cute_mma_dag_warps is not None
        assert config_spec.cute_mma_dag_stages is not None
        assert config_spec.cute_mma_dag_store is not None
        assert config_spec.cute_mma_dag_exchange is not None
        assert config_spec.cute_mma_dag_pass_warps is not None
        assert config_spec.cute_mma_dag_loads is not None
        assert config_spec.cute_mma_dag_derived is not None
        assert config_spec.cute_mma_dag_epilogue is not None
        lookahead_fragment = config_spec.cute_mma_dag_lookahead
        assert lookahead_fragment is not None
        fields[MMA_DAG_WARPS_KEY] = config_spec.cute_mma_dag_warps
        fields[MMA_DAG_STAGES_KEY] = config_spec.cute_mma_dag_stages
        fields[MMA_DAG_STORE_KEY] = config_spec.cute_mma_dag_store
        fields[MMA_DAG_EXCHANGE_KEY] = config_spec.cute_mma_dag_exchange
        fields[MMA_DAG_PASS_WARPS_KEY] = config_spec.cute_mma_dag_pass_warps
        v_lag_fragment = config_spec.cute_mma_dag_v_lag
        if v_lag_fragment is not None:
            fields[MMA_DAG_V_LAG_KEY] = v_lag_fragment
        sleep_fragment = config_spec.cute_mma_dag_sleep_waits
        if sleep_fragment is not None:
            fields[MMA_DAG_SLEEP_WAITS_KEY] = sleep_fragment
        retirement_fragment = config_spec.cute_mma_dag_early_tmem_retirement
        if retirement_fragment is not None:
            fields[MMA_DAG_EARLY_TMEM_RETIREMENT_KEY] = retirement_fragment
        query_fragment = config_spec.cute_mma_dag_query_tma
        if query_fragment is not None:
            fields[MMA_DAG_QUERY_TMA_KEY] = query_fragment
        metadata_fragment = config_spec.cute_mma_dag_value_metadata_skip
        if metadata_fragment is not None:
            fields[MMA_DAG_VALUE_METADATA_SKIP_KEY] = metadata_fragment
        overlap_fragment = config_spec.cute_mma_dag_value_metadata_read_overlap
        if overlap_fragment is not None:
            fields[MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY] = overlap_fragment
        probability_fragment = config_spec.cute_mma_dag_probability_schedule
        if probability_fragment is not None:
            fields[MMA_DAG_PROBABILITY_SCHEDULE_KEY] = probability_fragment
        fields[MMA_DAG_LOADS_KEY] = config_spec.cute_mma_dag_loads
        # (Locals: the type checker's attribute narrowing does not
        # reach this far into the function.)
        aux_fragment = config_spec.cute_mma_dag_aux
        if aux_fragment is not None:
            fields[MMA_DAG_AUX_KEY] = aux_fragment
        fields[MMA_DAG_DERIVED_KEY] = config_spec.cute_mma_dag_derived
        fields[MMA_DAG_EPILOGUE_KEY] = config_spec.cute_mma_dag_epilogue
        fields[MMA_DAG_LOOKAHEAD_KEY] = lookahead_fragment
        persistent_fragment = config_spec.cute_mma_dag_persistent
        if persistent_fragment is not None:
            fields[MMA_DAG_PERSISTENT_KEY] = persistent_fragment
        cluster_fragment = config_spec.cute_mma_dag_cluster
        if cluster_fragment is not None:
            fields[MMA_DAG_CLUSTER_KEY] = cluster_fragment
        split_fragment = config_spec.cute_mma_dag_split
        if split_fragment is not None:
            fields[MMA_DAG_SPLIT_KEY] = split_fragment
    rsb_fragment = config_spec.cute_register_subblocks
    if rsb_fragment is not None:
        fields[RSB_KEY] = rsb_fragment
        fields.update(
            (rsb_key, rsb_knob)
            for rsb_key, rsb_knob in (
                (RSB_WARPS_KEY, config_spec.cute_register_subblocks_warps),
                (RSB_PREFETCH_KEY, config_spec.cute_register_subblocks_prefetch),
                (RSB_MASK_KEY, config_spec.cute_register_subblocks_mask),
                (RSB_COLSIDE_KEY, config_spec.cute_register_subblocks_colside),
                (RSB_ROLL_KEY, config_spec.cute_register_subblocks_roll),
            )
            if rsb_knob is not None
        )
    if config_spec.cute_chunk_prepare_schedule is not None:
        fields[CUTE_CHUNK_PREPARE_SCHEDULE_KEY] = (
            config_spec.cute_chunk_prepare_schedule
        )
    if config_spec.cute_affine_scan_schedule is not None:
        fields[CUTE_AFFINE_SCAN_SCHEDULE_KEY] = config_spec.cute_affine_scan_schedule
    if (
        config_spec.cute_scaled_mma_available
        and config_spec.target_device_capability is not None
        and config_spec.target_device_capability[0] == 10
    ):
        fields.update(
            (
                key,
                EnumFragment(
                    choices=(False,)
                    if key == "cute_scaled_fused_scales"
                    and not config_spec.cute_mxfp8_fused_scales_available
                    else (False,)
                    if key in ("cute_scaled_sf_tma", "cute_scaled_sf_inplace")
                    and not config_spec.cute_mxfp8_sf_tma_available
                    else ("default",)
                    if key == "cute_scaled_epilogue_mode"
                    and not config_spec.cute_mxfp8_tma_output_available
                    else tuple(value for value in choices if value != 64)
                    if config_spec.cute_mxfp8_mma_available
                    and key == "cute_scaled_tile_k"
                    else choices
                ),
            )
            for key, choices in block_scaled_choices(
                mxfp8=config_spec.cute_mxfp8_mma_available
            ).items()
        )
    if config_spec.cute_matmul_min_blocks_search_enabled:
        fields["cute_min_blocks_per_mp"] = EnumFragment(choices=(0, 1))
    if (
        config_spec.supports_config_key("pid_type")
        and "pid_type" not in fields
        and (
            config_spec.allowed_pid_types[0] != "flat"
            or (
                config_spec.cute_ordered_phase_pid_search
                and len(config_spec.allowed_pid_types) > 1
            )
        )
    ):
        # SIMT normally omits this coordinate and defaults to flat.
        # Non-ordered barriers can disallow flat. Ordered barriers keep this
        # coordinate even when captured reductions or CTA-local atomics need
        # flat launches for the conservative reference.
        fields["pid_type"] = EnumFragment(config_spec.allowed_pid_types)
    if config_spec.cross_loop_pipeline is not None:
        fields["cross_loop_pipeline"] = config_spec.cross_loop_pipeline
    fields.update(config_spec.user_defined_tunables)
    if config_spec.cute_radix_selection_graph is not None:
        fields[RADIX_ALGORITHM_KEY] = EnumFragment(choices=RADIX_ALGORITHMS)
        fields.update(
            (key, EnumFragment(choices=choices))
            for key, choices in RADIX_GEOMETRY_CHOICES.items()
        )
        fields[RADIX_HISTOGRAM_KEY] = EnumFragment(choices=RADIX_HISTOGRAM_MODES)
        fields.update(
            (key, BooleanFragment())
            for key in RADIX_BOOLEAN_KEYS
            if key != "cute_radix_valid_domain"
            or config_spec.cute_radix_selection_graph.largest
        )
        fields.update(
            (key, EnumFragment(choices=choices))
            for key, choices in RADIX_PIPELINE_CHOICES.items()
        )
    fields[CUTE_COMPILER_PIPELINE_KEY] = _COMPILER_PIPELINE_FRAGMENT
    return fields


def flatten_missing_field_default(
    config_spec: ConfigSpec, key: str, config: dict[str, object]
) -> tuple[bool, object]:
    if key == CUTE_COMPILER_PIPELINE_KEY:
        return True, CUTE_COMPILER_PIPELINE_CHOICES[0]
    if key == RADIX_ALGORITHM_KEY:
        return True, "legacy"
    if key == RADIX_HISTOGRAM_KEY:
        return True, RADIX_HISTOGRAM_MODES[0]
    if key in RADIX_GEOMETRY_CHOICES:
        return True, RADIX_GEOMETRY_CHOICES[key][0]
    if key == "cute_materialized_schedule":
        return True, "off"
    if key == "cute_materialized_operand_schedule":
        return True, "off"
    if key == "cute_fragment_private_scalar_loops":
        return True, False
    if key == "cute_fragment_producer_cache":
        return True, False
    if key == "cute_fragment_warp_results":
        return True, False
    if key in (
        "cute_fragment_register_loads",
        "cute_fragment_half_mma",
        "cute_fragment_warp_scan",
        "cute_fragment_topk_network",
        "cute_fragment_cta_topk",
        "cute_fragment_atomic_aggregation",
        "cute_fragment_integer_atomic_epochs",
        "cute_topk_coarse_keys",
        "cute_fragment_local_atomic_registers",
        "cute_fragment_register_snapshots",
        "cute_fragment_register_producers",
        "cute_fragment_scan_exports",
        "cute_fragment_warp_producer_regions",
        "cute_fragment_bounded_gather",
        "cute_fragment_published_scalars",
        "cute_fragment_skip_zero_atomics",
        "cute_fragment_atomic_consumer_fusion",
        "cute_fragment_pure_producer_regions",
        "cute_integer_loop_reduction",
        "cute_fragment_packet_loads",
        "cute_fragment_padded_rows",
        "cute_fragment_unrolled_loops",
        "cute_fragment_vector_io",
        "cute_fragment_batched_loads",
        "cute_fragment_split_k",
        "cute_fragment_mma_rows",
        "cute_fragment_slab_loads",
    ):
        return True, False
    if key == "cute_topk_key_recovery":
        return True, "direct"
    if key == "cute_fragment_threads":
        return True, 128
    if key in ("cute_fragment_scan", "cute_fragment_reduction"):
        return True, "serial"
    if key == "cute_fp32_dot":
        return True, "simt"
    if key == "cute_signed_bitfield_bf16":
        return True, False
    if key == "cute_exact_div_form":
        return True, CUTE_EXACT_DIV_FORMS[0]
    if config_spec.cute_flash_search_enabled and key == FLASH_PIPELINE_FAMILY_KEY:
        return True, _resolve_cute_flash_config(config_spec, config).pipeline_family
    return surface.flatten_missing_field_default(config_spec.tcgen05, key, config)


def trim_reference_config(config_spec: ConfigSpec, config: Config) -> None:
    # The additive finalizer choice must not change existing seed configs.
    config.config.pop(SPLIT_K_FINALIZER_KEY, None)
    if config.get(RADIX_ALGORITHM_KEY, "legacy") == "legacy":
        # Optional radix coverage represents these defaults in flat space.
        # Keep the ordinary public default and seed payloads unchanged.
        for key in RADIX_CONFIG_KEYS:
            config.config.pop(key, None)
    if config_spec.cute_matmul_min_blocks_search_enabled:
        # This optional matmul coordinate has no old default Config key.
        # Keep the public default exact; flat search still represents zero.
        config.config.pop("cute_min_blocks_per_mp", None)
    _drop_inert_workspace_knobs(config)


def seed_configs(config_spec: ConfigSpec) -> list[Config]:
    seed_configs = seeds.autotune_seed_configs(config_spec.tcgen05)
    if config_spec.cute_flash_search_enabled:
        seed_configs.extend(cute_flash_seed_configs(config_spec))
    if config_spec.cute_flash_gated_search_enabled:
        seed_configs.extend(cute_flash_gated_seed_configs(config_spec))
    seed_configs.extend(cute_flash_bwd_seed_configs(config_spec))
    return seed_configs


def prepare_flat_config(config_spec: ConfigSpec, config: dict[str, object]) -> None:
    # None is the flat representation of these absent grouped overrides.
    # Preserve that absence before strict validation, which intentionally
    # rejects an explicitly supplied None for either option.
    for key in (
        "tcgen05_grouped_mode",
        "tcgen05_grouped_worklist_source_m_tile",
    ):
        if config.get(key) is None:
            config.pop(key, None)


# ``load_eviction_policies`` values of the CuTe loads.  "first"/"last" lower to
# ld.global L1 eviction hints (level1_eviction_priority=evict_first/evict_last)
# on the vectorized load sites.  Pays off for reload-from-gmem sweeps (keep
# re-read rows resident, evict on the final pass).  "streaming" lowers to the
# ld.global.cs cache operator (evict-first at BOTH L1 and L2): single-use
# streaming reads stop displacing useful/dirty L2 lines, which is worth ~20% on
# row reductions whose footprint is within ~2x of L2 (measured 57.3us -> 47.1us
# on cross-entropy 32768x2048 fp32 on B200 under do_bench's dirty-L2 flush;
# triton's evict_first hints L2 too, so this closes a structural gap vs the
# triton backend).  "l2_last" emits createpolicy.fractional.L2::evict_last +
# ld.global.L2::cache_hint (inline PTX) on 16-byte hoisted vec loads only --
# triton's evict_last equivalent, which keeps up to ~L2-size of a streaming
# input resident across other traffic (+1.7% on fp32 elementwise mul on B200).
# Matching explicit L1/L2 priorities preserve a packet for later row passes or
# evict it after its final use.  Both reach 16-, 8- and 4-byte packets; 32-byte
# packets (SM100+) only under the cute_wide_packet_hints knob.
CUTE_EVICTION_POLICIES: tuple[str, ...] = (
    "",
    "first",
    "last",
    "streaming",
    "l2_last",
    "l1_l2_first",
    "l1_l2_last",
    "nc_l2_first",
)


def reduction_loop_limits(
    config_spec: ConfigSpec,
    block_id: int,
    threshold: int | None,
    max_chunk: int | None,
) -> tuple[int | None, int | None]:
    """Clamp an indexed reduction's loop limits to the warp reduction."""
    if block_id in config_spec.cute_indexed_reduction_block_ids:
        # CuTe argmin/argmax uses warp_reduction (threads_in_group <= 32).
        threshold = 32 if threshold is None else min(threshold, 32)
        max_chunk = 32 if max_chunk is None else min(max_chunk, 32)
    return threshold, max_chunk


def explicit_reduction_chunk(
    config_spec: ConfigSpec, loop: ReductionLoopSpec, value: object
) -> object:
    """Repair a ``reduction_loops`` entry to the search/cache/seed budget
    before it is flattened: a chunk at or past the extent that is not the
    explicit full-extent (or exact non-power-of-two) chunk is the persistent
    ``None``; any other chunk is a power of two within ``max_chunk``."""
    if type(value) is not int or value <= 0:
        return value
    _, max_chunk = loop._limits(config_spec)
    explicit_chunk = value == loop._full_extent_chunk() or (
        loop.allow_non_power_of_two
        and 8 <= value <= loop.size_hint
        and loop.size_hint % value == 0
        and value & (value - 1) != 0
    )
    if value >= loop.size_hint and not explicit_chunk:
        return None
    if not explicit_chunk or (max_chunk is not None and value > max_chunk):
        return max(
            loop._flat_fragment(config_spec).low, prev_power_of_2(value, max_chunk)
        )
    return value


class CuteSearchSurface(BackendSearchSurface):
    """The CuTe backend's search surface (``CuteBackend.config_spec_extension``).

    Stateless: every hook takes the ``ConfigSpec`` it acts on and delegates to
    the module functions above, which are the former CuTe methods of
    ``ConfigSpec`` (``flash_surface`` / ``mma_dag_surface`` for those
    families).  The normalization phases run in the order documented on
    :class:`BackendSearchSurface`.
    """

    config_keys = CUTE_CONFIG_KEYS
    debug_keys = CUTE_DEBUG_KEYS
    implicit_default_keys = CUTE_IMPLICIT_DEFAULT_KEYS
    # Retirement and V lookbehind change the lifetime seen by the compiler
    # pipeline. Their joint choices can win when individual changes lose;
    # strict decoding keeps these coordinates active only on paired schedules.
    compiler_refinement_keys = (
        CUTE_COMPILER_PIPELINE_KEY,
        MMA_DAG_EARLY_TMEM_RETIREMENT_KEY,
        MMA_DAG_V_LAG_KEY,
    )

    def explain_unavailable_key(self, key: str, config_spec: ConfigSpec) -> str | None:
        if key == "num_threads" and not config_spec.target_device_capability:
            return "CuTe requires CUDA target"
        return None

    def eviction_policy_search_choices(
        self, config_spec: ConfigSpec, choices: Sequence[str]
    ) -> tuple[str, ...] | None:
        if "nc_l2_first" in choices and not readonly_packet_load_supported(
            config_spec.target_device_capability, torch.version.cuda
        ):
            # Read-only packets lower to the ordinary load on this target, so
            # searching them would only repeat "".  Explicit configs stay valid.
            return tuple(choice for choice in choices if choice != "nc_l2_first")
        return None

    def default_target_device_capability(self) -> tuple[int, int] | None:
        # Direct CuTe specs use the current CUDA device so validation still
        # enforces the arch gates.
        return get_target_device_capability()

    def fields(
        self, config_spec: ConfigSpec, *, flash_pipeline_family: str | None = None
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment] | None:
        return cute_flat_fields(config_spec, flash_pipeline_family)

    def normalize_entry(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_entry(config_spec, config, fix_invalid=fix_invalid)

    def normalize_family_gates(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_family_gates(config_spec, config, fix_invalid=fix_invalid)

    def normalize_knobs(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_knobs(config_spec, config, fix_invalid=fix_invalid)

    def normalize_before_sequences(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_before_sequences(config_spec, config, fix_invalid=fix_invalid)

    def normalize_after_sequences(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_after_sequences(config_spec, config, fix_invalid=fix_invalid)

    def reduction_loop_chunk_cap(
        self, config_spec: ConfigSpec, max_chunk: int, block_threshold: int
    ) -> int:
        return prev_power_of_2(max_chunk)

    def normalize_reduction_threads(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        register_tiles: bool,
    ) -> None:
        normalize_reduction_threads(
            config_spec, config, fix_invalid=fix_invalid, register_tiles=register_tiles
        )

    def normalize_after_defaults(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_after_defaults(config_spec, config, fix_invalid=fix_invalid)

    def normalize_recurrence_families(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_recurrence_families(config_spec, config, fix_invalid=fix_invalid)

    def normalize_pre_pid_type(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_pre_pid_type(config_spec, config, fix_invalid=fix_invalid)

    def normalize_strategy(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        normalize_strategy(config_spec, config, fix_invalid=fix_invalid)

    def normalize_finalize(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        provided_keys: set[str],
    ) -> None:
        normalize_finalize(
            config_spec, config, fix_invalid=fix_invalid, provided_keys=provided_keys
        )

    def seed_configs(self, config_spec: ConfigSpec) -> list[Config]:
        return seed_configs(config_spec)

    def prepare_config_generation_overrides(
        self, config_spec: ConfigSpec, overrides: Mapping[str, object] | None
    ) -> tuple[dict[str, object] | None, str | None]:
        return prepare_config_generation_overrides(config_spec, overrides)

    def flatten_missing_field_default(
        self, config_spec: ConfigSpec, key: str, config: dict[str, object]
    ) -> tuple[bool, object]:
        return flatten_missing_field_default(config_spec, key, config)

    def prepare_override_normalization(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        prepare_override_normalization(config_spec, config, overrides)

    def trim_reference_config(self, config_spec: ConfigSpec, config: Config) -> None:
        trim_reference_config(config_spec, config)

    def trim_default_config(self, config_spec: ConfigSpec, config: Config) -> None:
        _drop_inert_workspace_knobs(config)

    def overlay_promoted_default(
        self,
        config_spec: ConfigSpec,
        base: dict[str, object],
        promoted: Mapping[str, object],
    ) -> None:
        # A promoted ``pid_type`` re-derives the tcgen05 persistence model.
        surface.drop_stale_persistence_model(config_spec.tcgen05, base, promoted)

    def prepare_flat_config(
        self, config_spec: ConfigSpec, config: dict[str, object]
    ) -> None:
        prepare_flat_config(config_spec, config)

    def fingerprint_extra(self, config_spec: ConfigSpec) -> object | None:
        return {
            "target_policy": fingerprint_extra(config_spec),
            "compiler_refinement": compiler_refinement_policy(
                self.compiler_refinement_keys
            ),
        }

    def l2_evict_last_store_policy_supported(self, config_spec: ConfigSpec) -> bool:
        return fixed_l2_evict_last_store_policy_supported(
            config_spec.target_device_capability,
            torch.version.cuda,
        )

    def new_spec_state(self, config_spec: ConfigSpec) -> CuteTcgen05Config:
        return CuteTcgen05Config(config_spec)

    def pins_thread_knobs(self, config_spec: ConfigSpec, block_id: int) -> bool:
        return bool(
            config_spec.tcgen05.search_enabled
            and config_spec.cute_pointwise_region_block_ids
            and block_id not in config_spec.cute_pointwise_region_block_ids
        )

    def eviction_policies(self) -> tuple[str, ...]:
        return CUTE_EVICTION_POLICIES

    def reduction_loop_limits(
        self,
        config_spec: ConfigSpec,
        block_id: int,
        threshold: int | None,
        max_chunk: int | None,
    ) -> tuple[int | None, int | None]:
        return reduction_loop_limits(config_spec, block_id, threshold, max_chunk)

    def explicit_reduction_chunk(
        self, config_spec: ConfigSpec, loop: ReductionLoopSpec, value: object
    ) -> object:
        return explicit_reduction_chunk(config_spec, loop, value)

    def family_fragments(
        self, config_spec: ConfigSpec, family: str
    ) -> Mapping[str, ConfigSpecFragment]:
        return _cute_flash_autotune_fragments(config_spec, None, family)

    def config_generation_extension(
        self, generation: ConfigGeneration
    ) -> CuteConfigGenerationExtension:
        return CuteConfigGenerationExtension(generation)


CUTE_SEARCH_SURFACE = CuteSearchSurface()
