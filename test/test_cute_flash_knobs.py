from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_knobs
from helion._compiler.cute import flash_seed_templates
from helion._compiler.cute.cute_flash import FlashAttentionConfig
from helion._compiler.cute.flash_knobs import FLASH_AUTOTUNE_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_AUTOTUNE_VALUE_DEPENDENCIES
from helion._compiler.cute.flash_knobs import FLASH_BWD_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_DERIVED_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_EFFECTIVE_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_KNOB_WITNESSES
from helion._compiler.cute.flash_knobs import FLASH_KNOBS
from helion._compiler.cute.flash_knobs import FLASH_LEGACY_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_LEGACY_E2E_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
from helion._compiler.cute.flash_knobs import FLASH_Q_TILE_COUNT_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

# The literal tables the registry replaced, frozen at the commit that
# introduced it.  The autotune key order is the coverage design's axis order
# and the dependency table's insertion order is its witness-row order, so
# both are pinned as sequences, not sets.  Extend these pins (at the end) when
# a knob or witness is added on purpose.
_AXIS_KEYS: tuple[str, ...] = (
    "cute_flash_s_stage",
    "cute_flash_kv_stage",
    "cute_flash_persistent",
    "cute_flash_persistent_ctas_per_sm",
    "cute_flash_recompute_tile_coords",
    "cute_flash_e2e_schedule",
    "cute_flash_masked_e2e_schedule",
    "cute_flash_e2e_offset",
    "cute_flash_e2e_offset0",
    "cute_flash_exp2_packet",
    "cute_flash_mma_interleave",
    "cute_flash_wait_hint",
    "cute_flash_stat_transport",
    "cute_flash_pipeline_family",
    "cute_flash_softmax_disc",
    "cute_flash_disc_pipe",
    "cute_flash_split_p_arrive",
    "cute_flash_p_store_rep",
    "cute_flash_s_load_rep",
    "cute_flash_precompute_qk_desc",
    "cute_flash_first_load_order",
    "cute_flash_kv_order",
    "cute_flash_epi_tma",
    "cute_flash_o_stage",
    "cute_flash_epi_stg",
    "cute_flash_epi_stg_store",
    "cute_flash_epi_stg_gmem",
    "cute_flash_rescale_threshold",
    "cute_flash_skip_rescale_stats",
    "cute_flash_rescale_chunk_cols",
    "cute_flash_softmax_regs",
    "cute_flash_corr_regs",
    "cute_flash_other_regs",
    "cute_flash_corr_tile_size",
    "cute_flash_row_epilogue_warps",
    "cute_flash_packed_reduce",
    "cute_flash_small_biased",
    "cute_flash_causal_lpt_swizzle",
    "cute_flash_causal_lpt_l2_shift",
    "cute_flash_pair_cluster",
    "cute_flash_causal_kv_order",
    "cute_flash_role_map",
    "cute_flash_role_chain",
    "cute_flash_clc_heads_per_batch",
    "cute_flash_clc_pdl",
    "cute_flash_clc_stages",
    "cute_flash_causal_loop_split",
    "cute_flash_persistent_loop",
    "cute_flash_sp_row_sum",
    "cute_flash_rowmax",
    "cute_flash_softmax_lowering",
    "cute_flash_row_sum_schedule",
    "cute_flash_softmax_setup",
    "cute_flash_epi_tma_setup",
    "cute_flash_kv_tile_n",
    "cute_flash_q_tile_m",
    "cute_flash_ws_one_pass",
    "cute_flash_row_warps",
    "cute_flash_row_tile_m",
    "cute_flash_p_chunk_arrive",
    "cute_flash_bias_form",
    "cute_flash_uneven_kv_ring",
    "cute_flash_o_over_q",
    "cute_flash_causal_mask_storeback",
    "cute_flash_row_bound_tiling",
    "cute_flash_causal_skip_dead_tile",
    "cute_flash_causal_stagger",
    "cute_flash_causal_head_pair",
    "cute_flash_early_prologue_loads",
    "cute_flash_flex_skip_absent",
    "cute_flash_flex_dead_stage_split",
    "cute_flash_flex_early_qk",
    "cute_flash_flex_tail_split",
    "cute_flash_flex_mask_form",
    "cute_flash_kv_split",
    "cute_flash_flex_aux_smem",
    "cute_flash_flex_prefetch",
    "cute_flash_flex_coalesce",
    "cute_flash_flex_rolled_chunks",
    "cute_flash_flex_item_words",
    "cute_flash_s_resident",
    "cute_flash_softmax_region",
    "cute_flash_ws_loader_warp",
    "cute_flash_ws_mask_phases",
    "cute_flash_ws_k_stage_extra",
    "cute_flash_flex_hook_fast_full",
    "cute_flash_bias_stage",
)

_VALUE_DEPENDENCIES: tuple[tuple[str, object, dict[str, object]], ...] = (
    (
        "cute_flash_q_tile_m",
        64,
        {
            "cute_flash_pipeline_family": "ws_overlap",
            "cute_flash_persistent": False,
            "cute_flash_epi_stg": True,
            "cute_flash_kv_stage": 2,
        },
    ),
    (
        "cute_flash_ws_one_pass",
        True,
        {
            "cute_flash_pipeline_family": "ws_overlap",
            "cute_flash_persistent": False,
            "cute_flash_epi_stg": True,
            "cute_flash_kv_stage": 2,
            "cute_flash_q_tile_m": 64,
        },
    ),
    (
        "cute_flash_ws_loader_warp",
        True,
        {"cute_flash_s_stage": 2, "cute_flash_small_biased": False},
    ),
    (
        "cute_flash_ws_mask_phases",
        True,
        {"cute_flash_s_stage": 2, "cute_flash_small_biased": False},
    ),
    (
        "cute_flash_ws_k_stage_extra",
        1,
        {"cute_flash_s_stage": 2, "cute_flash_small_biased": False},
    ),
    (
        "cute_flash_o_over_q",
        True,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_persistent": False,
            "cute_flash_epi_tma": True,
        },
    ),
    (
        "cute_flash_causal_mask_storeback",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_mask_form",
        "element",
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    ("cute_flash_kv_split", 2, {"cute_flash_pipeline_family": "fa4"}),
    ("cute_flash_kv_split", 4, {"cute_flash_pipeline_family": "fa4"}),
    (
        "cute_flash_flex_aux_smem",
        4096,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_aux_smem",
        16384,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_prefetch",
        32,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_prefetch",
        64,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_prefetch",
        128,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_coalesce",
        32,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_coalesce",
        64,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_coalesce",
        128,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_rolled_chunks",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_hook_fast_full",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_flex_item_words",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_s_resident",
        True,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_softmax_disc": True,
            "cute_flash_s_load_rep": 32,
        },
    ),
    (
        "cute_flash_softmax_region",
        "shared",
        {"cute_flash_pipeline_family": "fa4", "cute_flash_softmax_disc": True},
    ),
    (
        "cute_flash_row_bound_tiling",
        "segment",
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_persistent": False,
            "cute_flash_epi_tma": True,
            "cute_flash_o_over_q": False,
        },
    ),
    (
        "cute_flash_causal_skip_dead_tile",
        True,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_causal_kv_order": "descending",
        },
    ),
    (
        "cute_flash_causal_stagger",
        True,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_causal_kv_order": "descending",
            "cute_flash_kv_stage": 4,
            "cute_flash_epi_tma": True,
            "cute_flash_o_over_q": True,
        },
    ),
    (
        "cute_flash_causal_head_pair",
        True,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_causal_kv_order": "descending",
            "cute_flash_epi_tma": True,
            "cute_flash_softmax_disc": True,
        },
    ),
    (
        "cute_flash_early_prologue_loads",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_persistent": False},
    ),
    ("cute_flash_flex_skip_absent", True, {"cute_flash_pipeline_family": "fa4"}),
    (
        "cute_flash_flex_dead_stage_split",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_flex_skip_absent": True},
    ),
    (
        "cute_flash_flex_early_qk",
        True,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_persistent": True},
    ),
    (
        "cute_flash_flex_early_qk",
        False,
        {"cute_flash_pipeline_family": "fa4", "cute_flash_persistent": True},
    ),
    (
        "cute_flash_flex_tail_split",
        2,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_persistent": False,
            "cute_flash_kv_split": 1,
        },
    ),
    (
        "cute_flash_flex_tail_split",
        4,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_persistent": False,
            "cute_flash_kv_split": 1,
        },
    ),
    (
        "cute_flash_flex_tail_split",
        8,
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_persistent": False,
            "cute_flash_kv_split": 1,
        },
    ),
)

_LEGACY_STRUCTURAL_KEYS: tuple[str, ...] = (
    "cute_flash_topology",
    "cute_flash_use_2cta",
    "cute_flash_cga2_local",
    "cute_flash_clc",
    "cute_flash_local_tma_partition",
    "cute_flash_tensor_4d_tma",
)
_LEGACY_E2E_KEYS: tuple[str, ...] = (
    "cute_flash_exp2_impl",
    "cute_flash_e2e_freq",
    "cute_flash_e2e_res",
)
_DERIVED_KEYS: tuple[str, ...] = ("cute_flash_q_tile_count",)

_EFFECTIVE_FIELDS: tuple[tuple[str, str], ...] = (
    ("cute_flash_s_stage", "s_stage"),
    ("cute_flash_kv_stage", "kv_stage"),
    ("cute_flash_persistent", "persistent"),
    ("cute_flash_persistent_ctas_per_sm", "persistent_ctas_per_sm"),
    ("cute_flash_recompute_tile_coords", "recompute_tile_coords"),
    ("cute_flash_e2e_schedule", "e2e_schedule"),
    ("cute_flash_masked_e2e_schedule", "masked_e2e_schedule"),
    ("cute_flash_e2e_offset", "e2e_offset"),
    ("cute_flash_e2e_offset0", "e2e_offset0"),
    ("cute_flash_exp2_packet", "exp2_packet"),
    ("cute_flash_mma_interleave", "mma_interleave"),
    ("cute_flash_q_tile_count", "q_tile_count"),
    ("cute_flash_wait_hint", "wait_hint"),
    ("cute_flash_stat_transport", "stat_transport"),
    ("cute_flash_pipeline_family", "pipeline_family"),
    ("cute_flash_softmax_disc", "softmax_disc"),
    ("cute_flash_disc_pipe", "disc_pipe_depth"),
    ("cute_flash_split_p_arrive", "split_p_arrive"),
    ("cute_flash_p_chunk_arrive", "p_chunk_arrive"),
    ("cute_flash_p_store_rep", "p_store_repetition"),
    ("cute_flash_s_load_rep", "s_load_repetition"),
    ("cute_flash_precompute_qk_desc", "precompute_qk_desc"),
    ("cute_flash_first_load_order", "first_load_order"),
    ("cute_flash_kv_order", "kv_order"),
    ("cute_flash_epi_tma", "epi_tma"),
    ("cute_flash_o_stage", "o_stage"),
    ("cute_flash_epi_stg", "epi_stg"),
    ("cute_flash_epi_stg_store", "epi_stg_store"),
    ("cute_flash_epi_stg_gmem", "epi_stg_gmem"),
    ("cute_flash_rescale_threshold", "rescale_threshold"),
    ("cute_flash_skip_rescale_stats", "skip_rescale_stats"),
    ("cute_flash_rescale_chunk_cols", "rescale_chunk_cols"),
    ("cute_flash_softmax_regs", "softmax_regs"),
    ("cute_flash_corr_regs", "corr_regs"),
    ("cute_flash_other_regs", "other_regs"),
    ("cute_flash_corr_tile_size", "corr_tile_size"),
    ("cute_flash_row_epilogue_warps", "row_epilogue_warps"),
    ("cute_flash_packed_reduce", "packed_reduce"),
    ("cute_flash_bias_form", "bias_form"),
    ("cute_flash_bias_stage", "bias_stage"),
    ("cute_flash_small_biased", "small_biased"),
    ("cute_flash_causal_lpt_swizzle", "causal_lpt_swizzle"),
    ("cute_flash_causal_lpt_l2_shift", "causal_lpt_l2_shift"),
    ("cute_flash_pair_cluster", "pair_cluster"),
    ("cute_flash_causal_kv_order", "causal_kv_order"),
    ("cute_flash_role_map", "role_map"),
    ("cute_flash_role_chain", "role_chain"),
    ("cute_flash_clc_heads_per_batch", "clc_heads_per_batch"),
    ("cute_flash_clc_pdl", "clc_use_pdl"),
    ("cute_flash_clc_stages", "clc_stages"),
    ("cute_flash_causal_loop_split", "causal_loop_split"),
    ("cute_flash_persistent_loop", "persistent_loop"),
    ("cute_flash_sp_row_sum", "sp_row_sum"),
    ("cute_flash_rowmax", "rowmax"),
    ("cute_flash_softmax_lowering", "softmax_lowering"),
    ("cute_flash_row_sum_schedule", "row_sum_schedule"),
    ("cute_flash_softmax_setup", "softmax_setup"),
    ("cute_flash_epi_tma_setup", "epi_tma_setup"),
    ("cute_flash_kv_tile_n", "kv_tile_n"),
    ("cute_flash_q_tile_m", "q_tile_m"),
    ("cute_flash_ws_one_pass", "ws_one_pass"),
    ("cute_flash_ws_loader_warp", "ws_loader_warp"),
    ("cute_flash_ws_mask_phases", "ws_mask_phases"),
    ("cute_flash_ws_k_stage_extra", "ws_k_stage_extra"),
    ("cute_flash_row_warps", "row_warps"),
    ("cute_flash_row_tile_m", "row_tile_m"),
    ("cute_flash_uneven_kv_ring", "uneven_kv_ring"),
    ("cute_flash_o_over_q", "o_over_q"),
    ("cute_flash_causal_mask_storeback", "causal_mask_storeback"),
    ("cute_flash_row_bound_tiling", "row_bound_tiling"),
    ("cute_flash_causal_skip_dead_tile", "causal_skip_dead_tile"),
    ("cute_flash_causal_stagger", "causal_stagger"),
    ("cute_flash_causal_head_pair", "causal_head_pair"),
    ("cute_flash_early_prologue_loads", "early_prologue_loads"),
    ("cute_flash_flex_skip_absent", "flex_skip_absent"),
    ("cute_flash_flex_dead_stage_split", "flex_dead_stage_split"),
    ("cute_flash_flex_early_qk", "flex_early_qk"),
    ("cute_flash_flex_tail_split", "flex_tail_split"),
    ("cute_flash_flex_mask_form", "flex_mask_form"),
    ("cute_flash_kv_split", "kv_split"),
    ("cute_flash_flex_aux_smem", "flex_aux_smem"),
    ("cute_flash_flex_prefetch", "flex_prefetch"),
    ("cute_flash_flex_coalesce", "flex_coalesce"),
    ("cute_flash_flex_rolled_chunks", "flex_rolled_chunks"),
    ("cute_flash_flex_item_words", "flex_item_words"),
    ("cute_flash_s_resident", "s_resident"),
    ("cute_flash_softmax_region", "softmax_region"),
    ("cute_flash_flex_hook_fast_full", "flex_hook_fast_full"),
)

_BWD_KEYS: tuple[str, ...] = (
    "cute_flash_bwd_persistent",
    "cute_flash_bwd_two_cta",
    "cute_flash_bwd_exp2_f32",
    "cute_flash_bwd_mask_form",
    "cute_flash_bwd_regs",
    "cute_flash_bwd_s_first",
    "cute_flash_bwd_kv_stage",
    "cute_flash_bwd_hook_cols",
    "cute_flash_bwd_hook_pin",
    "cute_flash_bwd_bias_fold",
    "cute_flash_bwd_tile_group",
    "cute_flash_bwd_stats_mma",
    "cute_flash_bwd_aux_smem",
    "cute_flash_bwd_ds_tmem",
    "cute_flash_bwd_exp2_emu",
    "cute_flash_bwd_table_steps",
    "cute_flash_bwd_family",
    "cute_flash_bwd_row_kv",
    "cute_flash_bwd_row_q_split",
    "cute_flash_bwd_row_q_chunk",
)


def test_axis_order_is_the_registry_order() -> None:
    assert FLASH_AUTOTUNE_CONFIG_KEYS == _AXIS_KEYS
    assert tuple(knob.key for knob in FLASH_KNOBS) == _AXIS_KEYS
    assert len(set(_AXIS_KEYS)) == len(_AXIS_KEYS)


def test_value_dependencies_are_generated_in_witness_order() -> None:
    assert [
        (key, value, deps)
        for (key, value), deps in FLASH_AUTOTUNE_VALUE_DEPENDENCIES.items()
    ] == list(_VALUE_DEPENDENCIES)
    # One entry per declared value, each with its own requirement mapping.
    entries = [
        (witness.key, value)
        for witness in FLASH_KNOB_WITNESSES
        for value in witness.values
    ]
    assert entries == list(FLASH_AUTOTUNE_VALUE_DEPENDENCIES)
    first, second = (
        FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(flash_knobs.FLASH_FLEX_PREFETCH_KEY, 32)],
        FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(flash_knobs.FLASH_FLEX_PREFETCH_KEY, 64)],
    )
    assert first == second and first is not second


def test_witnesses_name_registered_knobs() -> None:
    autotune = set(FLASH_AUTOTUNE_CONFIG_KEYS)
    for witness in FLASH_KNOB_WITNESSES:
        assert witness.key in autotune, witness.key
        assert witness.values, witness.key
        assert set(witness.requires) <= autotune, witness.key
        assert witness.key not in witness.requires, witness.key


def test_key_partitions_are_unchanged() -> None:
    assert FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS == _LEGACY_STRUCTURAL_KEYS
    assert FLASH_LEGACY_E2E_CONFIG_KEYS == _LEGACY_E2E_KEYS
    assert (*_LEGACY_E2E_KEYS, *_LEGACY_STRUCTURAL_KEYS) == FLASH_LEGACY_CONFIG_KEYS
    assert FLASH_DERIVED_CONFIG_KEYS == _DERIVED_KEYS
    assert (
        *_AXIS_KEYS,
        *_LEGACY_E2E_KEYS,
        *_LEGACY_STRUCTURAL_KEYS,
        *_DERIVED_KEYS,
    ) == FLASH_CONFIG_KEYS
    assert len(set(FLASH_CONFIG_KEYS)) == len(FLASH_CONFIG_KEYS)
    assert FLASH_BWD_CONFIG_KEYS == _BWD_KEYS
    assert not set(FLASH_BWD_CONFIG_KEYS) & set(FLASH_CONFIG_KEYS)


def test_effective_values_project_every_knob_from_a_config_field() -> None:
    fields = {field.name for field in dataclasses.fields(FlashAttentionConfig)}
    expected = dict(_EFFECTIVE_FIELDS)
    assert {knob.key: knob.field for knob in FLASH_KNOBS} == {
        key: field for key, field in expected.items() if key != FLASH_Q_TILE_COUNT_KEY
    }
    for knob in FLASH_KNOBS:
        assert knob.field in fields, knob
    # The projection order is the normalized configs' and the seeds' key
    # order: pinned as the literal above, a permutation of the axes plus the
    # derived query-slot count.
    assert tuple(expected) == FLASH_EFFECTIVE_CONFIG_KEYS
    assert sorted(FLASH_EFFECTIVE_CONFIG_KEYS) == sorted(
        (*FLASH_AUTOTUNE_CONFIG_KEYS, FLASH_Q_TILE_COUNT_KEY)
    )
    config = cute_flash.resolve_flash_config(64, 32)
    values = flash_knobs.flash_effective_config_values(config)
    assert list(values) == list(expected)
    assert values == {key: getattr(config, field) for key, field in expected.items()}


def test_every_key_constant_cute_flash_binds_is_the_flash_knobs_object() -> None:
    # ``cute_flash`` imports the key constants it reads from ``flash_knobs``
    # (never a copy, never a key of its own); the knobs it does not read
    # (``FLASH_E2E_FREQ_KEY``, ...) are read from ``flash_knobs`` directly.
    bound = {
        name: value
        for name, value in vars(cute_flash).items()
        if name.startswith("FLASH_") and name.endswith("_KEY")
    }
    assert bound
    for name, value in bound.items():
        assert getattr(flash_knobs, name) is value, name


def test_schedule_choice_tables_follow_the_packet_tables() -> None:
    names = {
        flash_knobs._flash_manual_exp2_packet_schedule_name(packet)
        for packet in flash_knobs._FLASH_MANUAL_EXP2_PACKET_SCHEDULES
    }
    assert set(flash_knobs.FLASH_E2E_SCHEDULE_CHOICES) == {
        "16/4",
        "8/2",
        "16/2",
        "xu",
        *names,
    }
    assert set(flash_knobs.FLASH_MASKED_E2E_SCHEDULE_CHOICES) == {
        "inherit",
        *flash_knobs.FLASH_E2E_SCHEDULE_CHOICES,
    }
    assert set(flash_knobs.FLASH_EXP2_PACKET_CHOICES) == {
        *flash_knobs._FLASH_EXP2_PACKET_PARAMS,
        *flash_knobs._FLASH_MANUAL_EXP2_PACKET_PARAMS,
    }


def _resolve_family(config: Mapping[str, object]) -> FlashAttentionConfig:
    return cute_flash.resolve_flash_config(64, 32, config)


def test_upgrade_legacy_flash_config_folds_the_structural_keys() -> None:
    # The legacy topology (with the persistence that selects among the
    # single-tile families) becomes the compound family; the per-flag keys go.
    config: dict[str, object] = {
        "block_sizes": [1, 128, 128],
        flash_knobs.FLASH_TOPOLOGY_KEY: "ws_overlap",
        flash_knobs.FLASH_PERSISTENT_KEY: False,
        flash_knobs.FLASH_KV_STAGE_KEY: 2,
    }
    flash_knobs.upgrade_legacy_flash_config(config, _resolve_family)
    assert config[flash_knobs.FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
    assert not set(config) & set(flash_knobs.FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS)
    assert config[flash_knobs.FLASH_KV_STAGE_KEY] == 2
    paired = {
        flash_knobs.FLASH_TOPOLOGY_KEY: "fa4",
        flash_knobs.FLASH_USE_2CTA_KEY: True,
    }
    flash_knobs.upgrade_legacy_flash_config(paired, _resolve_family)
    assert paired == {flash_knobs.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"}


def test_upgrade_legacy_flash_config_leaves_named_families_and_plain_configs() -> None:
    named: dict[str, object] = {
        flash_knobs.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        flash_knobs.FLASH_TOPOLOGY_KEY: "ws_overlap",
    }
    flash_knobs.upgrade_legacy_flash_config(named, _resolve_family)
    assert named == {
        flash_knobs.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        flash_knobs.FLASH_TOPOLOGY_KEY: "ws_overlap",
    }
    plain: dict[str, object] = {flash_knobs.FLASH_KV_STAGE_KEY: 3}
    flash_knobs.upgrade_legacy_flash_config(plain, _resolve_family)
    assert plain == {flash_knobs.FLASH_KV_STAGE_KEY: 3}
    # A legacy key set to None is an absent key.
    absent: dict[str, object] = {flash_knobs.FLASH_TOPOLOGY_KEY: None}
    flash_knobs.upgrade_legacy_flash_config(absent, _resolve_family)
    assert absent == {flash_knobs.FLASH_TOPOLOGY_KEY: None}


def test_legacy_e2e_params_override_the_parsed_cadence_component_wise() -> None:
    config = {
        flash_knobs.FLASH_EXP2_IMPL_KEY: "split",
        flash_knobs.FLASH_E2E_FREQ_KEY: 16,
    }
    assert flash_knobs.flash_legacy_e2e_params(config.get, "xu", 8, 0) == (
        "split",
        16,
        0,
    )
    assert flash_knobs.flash_legacy_e2e_params({}.get, "split", 8, 2) == (
        "split",
        8,
        2,
    )


def test_resident_lowerings_own_their_cadence_without_legacy_keys() -> None:
    for lowering in ("resident_value_graph", "resident_stateful"):
        owned = cute_flash._flash_resident_softmax_overrides(lowering)
        assert owned[flash_knobs.FLASH_E2E_SCHEDULE_KEY] == "xu"
        assert not set(owned) & set(flash_knobs.FLASH_LEGACY_E2E_CONFIG_KEYS)
        assert set(owned) <= set(FLASH_AUTOTUNE_CONFIG_KEYS)
    # The legacy spelling does not reach the whole-row lowering's cadence.
    resolved = cute_flash.resolve_flash_config(
        64,
        512,
        {
            flash_knobs.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            flash_knobs.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
            flash_knobs.FLASH_EXP2_IMPL_KEY: "split",
            flash_knobs.FLASH_E2E_FREQ_KEY: 16,
            flash_knobs.FLASH_E2E_RES_KEY: 4,
        },
        is_causal=True,
        standard_causal_output=True,
        causal_resident_compatible=True,
        tmem_rowmax_compatible=True,
        target_device_capability=(10, 3),
    )
    assert resolved.softmax_lowering == "resident_stateful"
    assert (resolved.exp2_impl, resolved.e2e_freq, resolved.e2e_res) == ("xu", 8, 0)


def test_fold_flash_ws_inactive_values_folds_known_aliases_only() -> None:
    config: dict[str, object] = {
        flash_knobs.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
        flash_knobs.FLASH_EXP2_PACKET_KEY: "deg2_16x6",
        flash_knobs.FLASH_WAIT_HINT_KEY: 0,
    }
    flash_knobs.fold_flash_ws_inactive_values(config)
    assert config == {
        flash_knobs.FLASH_MASKED_E2E_SCHEDULE_KEY: "inherit",
        flash_knobs.FLASH_EXP2_PACKET_KEY: "1x1",
        flash_knobs.FLASH_WAIT_HINT_KEY: 10_000_000,
    }
    unknown: dict[str, object] = {
        flash_knobs.FLASH_MASKED_E2E_SCHEDULE_KEY: "32/16",
        flash_knobs.FLASH_EXP2_PACKET_KEY: "unknown_packet",
        flash_knobs.FLASH_WAIT_HINT_KEY: 1,
    }
    flash_knobs.fold_flash_ws_inactive_values(unknown)
    assert unknown == {
        flash_knobs.FLASH_MASKED_E2E_SCHEDULE_KEY: "32/16",
        flash_knobs.FLASH_EXP2_PACKET_KEY: "unknown_packet",
        flash_knobs.FLASH_WAIT_HINT_KEY: 1,
    }


def test_seed_templates_name_registered_knobs() -> None:
    """Every seed body pins searched knobs only (the derived query-slot count
    and the legacy keys never appear), and every source variant's key is one."""
    autotune = set(FLASH_AUTOTUNE_CONFIG_KEYS)
    assert flash_seed_templates.FLASH_SEED_TEMPLATES
    for name, template in flash_seed_templates.FLASH_SEED_TEMPLATES.items():
        assert set(template) <= autotune, (name, set(template) - autotune)
    for key, _value, body in flash_seed_templates.SEED_SOURCE_VARIANTS:
        assert key in autotune, key
        assert key not in body, key
    # The list-walk bodies state the ring the builder caps; the mid-length
    # walk alone pins its output tile count.
    for name in (
        "flex_fa4_structure",
        "flex_short_walk",
        "flex_medium_walk",
        "flex_fa4_template",
        "flex_long_walk",
        "flex_p2c_walk",
        "flex_w_walk",
        "flex_mid_walk",
    ):
        template = flash_seed_templates.FLASH_SEED_TEMPLATES[name]
        assert isinstance(template[flash_knobs.FLASH_KV_STAGE_KEY], int), name
        assert (flash_knobs.FLASH_O_STAGE_KEY in template) == (name == "flex_mid_walk")
    body = cute_flash._flash_list_walk_seed_body(
        flash_seed_templates.SEED_FLEX_FA4_TEMPLATE, kv_stage_cap=4, o_stage=1
    )
    assert body[flash_knobs.FLASH_KV_STAGE_KEY] == 4
    assert body[flash_knobs.FLASH_O_STAGE_KEY] == 1
    pinned = cute_flash._flash_list_walk_seed_body(
        flash_seed_templates.SEED_FLEX_MID_WALK, kv_stage_cap=8, o_stage=1
    )
    assert pinned[flash_knobs.FLASH_KV_STAGE_KEY] == 3
    assert pinned[flash_knobs.FLASH_O_STAGE_KEY] == 2
