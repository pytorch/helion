"""The CuTe backend's config-key tables, declared once.

Every knob family declares its keys next to its planner / emitter
(``FLASH_CONFIG_KEYS``, ``MMA_DAG_KEYS``, the radix / split-K / grouped tables,
the tcgen05 tables); the SIMT knob families that used to be spelled out as
literals in ``helion/autotuner/config_spec.py`` are declared here.  The core
tables ``VALID_KEYS`` / ``BACKEND_SPECIFIC_KEYS`` and the CuTe surface's
``config_keys`` / ``debug_keys`` / ``implicit_default_keys`` are derived from
the union below, so a new knob is added here once (a ``FragmentKnob`` row of
``fragment_knobs.py`` names its key a second time; that module checks at import
that ``CUTE_FRAGMENT_KEYS`` lists every row).

``test/test_config_spec_keys_migration.py`` pins the derived sets to the literal
tables they replaced.
"""

from __future__ import annotations

from ..block_scaled_config import BLOCK_SCALED_CONFIG_KEYS
from ..compile_options import (
    CUTE_COMPILER_PIPELINE_CHOICES as CUTE_COMPILER_PIPELINE_CHOICES,
)
from ..compile_options import CUTE_COMPILER_PIPELINE_KEY
from ..cute_flash import FLASH_CONFIG_KEYS
from ..cute_mma_dag import MMA_DAG_KEYS
from ..cute_register_subblocks import KEYS as RSB_KEYS
from ..flash_knobs import FLASH_BWD_CONFIG_KEYS
from ..selection.radix_selection_config import CONFIG_KEYS as RADIX_CONFIG_KEYS
from ..split_k_cluster_config import FINALIZER_KEY as SPLIT_K_FINALIZER_KEY
from ..split_k_cluster_config import SCHEDULE_KEY as SPLIT_K_SCHEDULE_KEY
from ..split_k_workspace_config import WORKSPACE_CONFIG_KEYS
from ..tcgen05_config import CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS
from ..tcgen05_config import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from ..tcgen05_config import CUTE_TCGEN05_GROUPED_CONFIG_KEYS
from ..tcgen05_config import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from ..tcgen05_config import CUTE_TCGEN05_TUNABLE_KEYS
from ..tcgen05_flat_grouped_config import CONFIG_KEYS as GROUPED_RNA_CONFIG_KEYS

# -- recurrence / scan / top-k families (keys owned by their emitters) --------
CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY = "cute_chunk_recurrence_dv_partitions"
CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY = "cute_chunk_recurrence_register_cap"
VALID_CUTE_CHUNK_RECURRENCE_REGISTER_CAPS = (None, 72, 76, 80)
CUTE_GDN_RECURRENCE_STAGES_KEY = "cute_gdn_recurrence_stages"
CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY = "cute_gdn_recurrence_epilogue_warps"
CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY = "cute_gdn_recurrence_token_groups"
CUTE_GDN_RECURRENCE_MMA_M_KEY = "cute_gdn_recurrence_mma_m"
CUTE_CHUNK_PREPARE_SCHEDULE_KEY = "cute_chunk_prepare_schedule"
VALID_CUTE_CHUNK_PREPARE_SCHEDULES = (
    "split_alias_cpc1",
    "split_alias_cpc2",
    "split_alias_cpc3",
    "split_alias_cpc4",
    "split_alias_cpc5",
)
CUTE_AFFINE_SCAN_SCHEDULE_KEY = "cute_affine_scan_schedule"
CUTE_TOPK_CHOICES: dict[str, tuple[int | str, ...]] = {
    "cute_topk_lanes_per_row": (16, 1, 2, 4, 8, 32),
    "cute_topk_rows_per_block": (8, 1, 2, 4, 16, 32, 64, 128),
    "cute_topk_vector_width": (8, 1, 2, 4),
    "cute_topk_output_vector_width": (1, 2, 4, 8),
    "cute_topk_value_mode": ("gather", "decode"),
    "cute_topk_key_dtype": ("int32", "float32", "float32_bits", "float32_native"),
    "cute_topk_rank_mode": ("signed", "ordinal"),
    "cute_topk_selection_layout": ("replicated", "distributed"),
    "cute_topk_sort_network": ("batcher", "compact", "compact_pruned"),
    "cute_topk_key_encoder": ("dsl", "asm", "paired"),
    "cute_topk_defer_value_gathers": (False, True),
    "cute_topk_merge_schedule": ("sequential", "balanced"),
}
CUTE_TOPK_CONFIG_KEYS: frozenset[str] = frozenset(CUTE_TOPK_CHOICES)
CUTE_RECURRENCE_KEYS: tuple[str, ...] = (
    CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
    CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,
    CUTE_GDN_RECURRENCE_STAGES_KEY,
    CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
    CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
    CUTE_GDN_RECURRENCE_MMA_M_KEY,
    CUTE_CHUNK_PREPARE_SCHEDULE_KEY,
    CUTE_AFFINE_SCAN_SCHEDULE_KEY,
)


def _cute_chunk_recurrence_config_is_safe(
    dv_partitions: object, register_cap: object
) -> bool:
    """Reject register caps on the TMEM schedule's dynamic register protocol."""

    return dv_partitions != 2 or register_cap is None


# The gated (softmax-free) attention body's gate warpgroup count,
# ``cute_flash_gated.FLASH_GATE_WARPGROUPS_KEY``.  Spelled out: importing the
# gated emitter here pulls ``device_ir`` into ``config_spec``'s import and
# closes a cycle through ``helion.Config``; the migration test pins the two.
CUTE_FLASH_GATE_KEYS: tuple[str, ...] = (
    "cute_flash_gate_warpgroups",
    "cute_flash_gated_row_warps",
)

# -- SIMT knob families -------------------------------------------------------
# Per-block-id sequences (one slot per tile / reduction block).
CUTE_SEQUENCE_KEYS: tuple[str, ...] = (
    "num_threads",
    "cute_vector_widths",
    "cute_lane_layouts",
    "cute_reduction_reloads",
)
CUTE_REDUCTION_KEYS: tuple[str, ...] = (
    "cute_reduction_schedule",
    "cute_reduction_pipeline_depth",
    "cute_reduction_local_tree",
    "cute_reduction_row_schedule",
    "cute_reduction_pack_output",
    "cute_reduction_sequence",
    "cute_host_paired_sum",
    "cute_reduction_group_rows",
    "cute_materialized_schedule",
    "cute_materialized_operand_schedule",
)
# The computed-fragment emitter's modes (``cute/computed_fragment.py``): the
# ``FragmentKnob`` rows of ``fragment_knobs.py`` and the fragment knobs with a
# heuristic class of their own.  The rows cannot be read here (this module
# loads while ``config_spec`` initializes; the rows sit behind the heuristics
# package, which needs ``config_spec`` complete), so ``fragment_knobs`` checks
# at import that every row's key is listed below.
CUTE_FRAGMENT_KEYS: tuple[str, ...] = (
    "cute_fragment_scan",
    "cute_fragment_reduction",
    "cute_fp32_dot",
    "cute_fragment_half_mma",
    "cute_fragment_threads",
    "cute_fragment_register_loads",
    "cute_fragment_producer_cache",
    "cute_fragment_warp_scan",
    "cute_fragment_topk_network",
    "cute_fragment_cta_topk",
    "cute_fragment_atomic_aggregation",
    "cute_fragment_integer_atomic_epochs",
    "cute_topk_coarse_keys",
    "cute_topk_key_recovery",
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
    "cute_fragment_warp_results",
    "cute_fragment_private_scalar_loops",
    "cute_pointwise_pid_type",
)
CUTE_ASYNC_LOAD_KEYS: tuple[str, ...] = (
    "cute_async_load_stages",
    "cute_async_load_lookahead",
    "cute_async_load_group_rows",
    "cute_async_load_cache",
    "cute_async_store_policy",
)
# The warp / tcgen05 collective GEMM over ``matmul_facts``.
CUTE_COLLECTIVE_KEYS: tuple[str, ...] = (
    "cute_collective_mma",
    "cute_collective_static_layouts",
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
# Lowering choices of the SIMT path (store policy, divisions, recurrences,
# lane-loop schedule, PDL, prefetch), normalized to implicit defaults.
CUTE_MISC_SCHEDULE_KEYS: tuple[str, ...] = (
    "cute_store_policy",
    "cute_exact_div_form",
    "cute_bf16x2_recurrence",
    "cute_register_chain",
    "cute_simt_mma",
    "cute_simt_mma_layout",
    "cute_signed_bitfield_bf16",
    "cute_proven_bounds",
    "cute_rng_packet",
    "cute_independent_reduction",
    "cute_cluster_warp_online",
    "cute_register_row",
    "cute_register_row_tensor_copy",
    "cute_scalar_epilogue",
    "cute_scalar_epilogue_keep_inputs",
    "cute_scalar_epilogue_owner_fold",
    "cute_warp_fp32_redux",
    "cute_register_tile_warp_fold",
    "cute_register_tile_cooperative_fold",
    "cute_register_tile_store_first",
    "cute_replicated_reduction",
    "cute_vector_packet_unroll",
    "cute_vloop_sink",
    "cute_grid_lane_jam",
    "cute_scratch_forwarding",
    "cute_lane_unroll",
    "cute_unpadded_rows",
    "cute_bool_byte_packets",
    "cute_dot_fragment_mma",
    "cute_dot_ring_mma",
    "cute_dot_ring_stages",
    "cute_dot_ring_split",
    "cute_pdl",
    "cute_pdl_capture_overlap",
    "cute_pdl_release",
    "cute_pdl_spread",
    "cute_warp_aggregated_atomics",
    "cute_privatize_atomics",
    "cute_packet_prefetch",
    "cute_wide_packet_hints",
)
# Launch geometry knobs: never implicit defaults (a pinned config keeps them).
CUTE_LAUNCH_KEYS: tuple[str, ...] = (
    "cute_cluster_n",
    "cute_min_blocks_per_mp",
    "cute_tiles_per_block",
)

# -- derived tables ------------------------------------------------------------
# Every backend-specific key the CuTe surface normalizes or searches.
# ``cross_loop_pipeline`` is a backend-gated generic key CuTe also accepts.
CUTE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "cross_loop_pipeline",
        CUTE_COMPILER_PIPELINE_KEY,
        *CUTE_TCGEN05_TUNABLE_KEYS,
        *CUTE_TCGEN05_GROUPED_CONFIG_KEYS,
        *CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS,
        *CUTE_TCGEN05_STRATEGY_CONFIG_KEYS,
        *BLOCK_SCALED_CONFIG_KEYS,
        SPLIT_K_SCHEDULE_KEY,
        SPLIT_K_FINALIZER_KEY,
        *WORKSPACE_CONFIG_KEYS,
        *GROUPED_RNA_CONFIG_KEYS,
        *FLASH_CONFIG_KEYS,
        *CUTE_FLASH_GATE_KEYS,
        *FLASH_BWD_CONFIG_KEYS,
        *CUTE_RECURRENCE_KEYS,
        *MMA_DAG_KEYS,
        *RSB_KEYS,
        *CUTE_TOPK_CONFIG_KEYS,
        *RADIX_CONFIG_KEYS,
        *CUTE_SEQUENCE_KEYS,
        *CUTE_REDUCTION_KEYS,
        *CUTE_FRAGMENT_KEYS,
        *CUTE_ASYNC_LOAD_KEYS,
        *CUTE_COLLECTIVE_KEYS,
        *CUTE_MISC_SCHEDULE_KEYS,
        *CUTE_LAUNCH_KEYS,
    }
)
# Diagnostic toggles: accepted by ``normalize``, never searched, seeded or
# fingerprinted (no recorded campaign winner sets one).
CUTE_DEBUG_KEYS: frozenset[str] = CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
# Keys ``normalize`` fills with a default and drops again when the config did
# not provide them (the generic names are CuTe's own defaults policy; the
# never-implicit families are the per-site sequences, the launch keys, the
# recurrences, flash, mma_dag, register sub-blocks, top-k and tcgen05).
CUTE_IMPLICIT_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        *RADIX_CONFIG_KEYS,
        CUTE_COMPILER_PIPELINE_KEY,
        "loop_orders",
        "flatten_loops",
        "l2_groupings",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "load_eviction_policies",
        "indexing",
        "atomic_indexing",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        *CUTE_REDUCTION_KEYS,
        *CUTE_FRAGMENT_KEYS,
        *CUTE_ASYNC_LOAD_KEYS,
        *CUTE_MISC_SCHEDULE_KEYS,
        *CUTE_COLLECTIVE_KEYS,
        *BLOCK_SCALED_CONFIG_KEYS,
        SPLIT_K_SCHEDULE_KEY,
        SPLIT_K_FINALIZER_KEY,
        *WORKSPACE_CONFIG_KEYS,
    }
)
