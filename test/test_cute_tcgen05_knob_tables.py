"""The tcgen05 knob tables: key classification, inert search axes and the
canonical schedule.

CPU-only facts about ``tcgen05_config`` / ``tcgen05_constants``; the one bind
renders nothing.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.epilogue_fanout import fanout_schedule
from helion._compiler.cute.grouped_row_union import carrier_schedule
from helion._compiler.cute.one_cta_pipeline import one_cta_pipeline_defaults
from helion._compiler.cute.strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from helion._compiler.cute.tcgen05_config import CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS
from helion._compiler.cute.tcgen05_config import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from helion._compiler.cute.tcgen05_config import CUTE_TCGEN05_GROUPED_CONFIG_KEYS
from helion._compiler.cute.tcgen05_config import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from helion._compiler.cute.tcgen05_config import CUTE_TCGEN05_TUNABLE_KEYS
from helion._compiler.cute.tcgen05_config import TCGEN05_C_STAGES_CHOICES
from helion._compiler.cute.tcgen05_config import TCGEN05_C_STAGES_SEARCH_CHOICES
from helion._compiler.cute.tcgen05_config import TCGEN05_L2_SWIZZLE_SEARCH_SIZES
from helion._compiler.cute.tcgen05_config import TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_ROLE_LOCAL_SCHEDULE
from helion._compiler.cute.tcgen05_constants import tcgen05_role_local_schedule
from helion._compiler.cute.tcgen05_schedule_facts import (
    n_edge_cluster_schedule_defaults,
)
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import scaled_args
from helion._testing.cute_dense_tcgen05 import scaled_mm
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.exc import InvalidConfig

# The public tcgen05 Config surface: the 71 keys that existed when the
# diagnostic set was narrowed, minus the retired axes (the seven invalid-output
# pipeline-edge modes and the clustered CtaGroup.ONE bridge flag).  The
# classification may move keys between tables; the surface only shrinks by an
# explicit retirement, which ``test_retired_tcgen05_knobs_are_rejected`` pins.
_FORMER_SURFACE = frozenset(
    {
        "cute_matmul_family",
        "cute_warp_mma_warps",
        "tcgen05_ab_stages",
        "tcgen05_ab_startup_prefill",
        "tcgen05_acc_stages",
        "tcgen05_acc_wait_placement",
        "tcgen05_aux_fragment_hoist",
        "tcgen05_aux_load_mode",
        "tcgen05_aux_load_placement",
        "tcgen05_aux_prefetch_subtiles",
        "tcgen05_aux_stages",
        "tcgen05_batch_raster",
        "tcgen05_c_acquire_placement",
        "tcgen05_c_stages",
        "tcgen05_c_store_mode",
        "tcgen05_clc_query_pacing",
        "tcgen05_cluster_m",
        "tcgen05_cluster_n",
        "tcgen05_consumer_regs",
        "tcgen05_cta_group",
        "tcgen05_cubin_lineinfo",
        "tcgen05_diagnostic_invalid_output",
        "tcgen05_epi_subtile_n",
        "tcgen05_epi_round_subtiles",
        "tcgen05_epi_warp_groups",
        "tcgen05_epilogue_fanout",
        "tcgen05_epilogue_layout",
        "tcgen05_flat_role_coordinates",
        "tcgen05_grouped_ab_tile_k",
        "tcgen05_grouped_balanced_workers",
        "tcgen05_grouped_ctas_per_sm",
        "tcgen05_grouped_d_shift",
        "tcgen05_grouped_dense_row_union",
        "tcgen05_grouped_external_direct_pointers",
        "tcgen05_grouped_external_direct_strides",
        "tcgen05_grouped_full_allocation_a",
        "tcgen05_grouped_full_coverage",
        "tcgen05_grouped_l2_swizzle_alternate",
        "tcgen05_grouped_l2_swizzle_mn_ratio",
        "tcgen05_grouped_mode",
        "tcgen05_grouped_prefix_adaptive_n",
        "tcgen05_grouped_row_union_ab_stages",
        "tcgen05_grouped_row_union_schedule",
        "tcgen05_grouped_runtime_direct",
        "tcgen05_grouped_static_problem_signature",
        "tcgen05_grouped_static_reserved_sms",
        "tcgen05_grouped_worklist_mma_m",
        "tcgen05_grouped_worklist_mma_n",
        "tcgen05_grouped_worklist_orientation",
        "tcgen05_grouped_worklist_source_m_tile",
        "tcgen05_l2_swizzle_size",
        "tcgen05_large_bn_proof",
        "tcgen05_materialized_pdl",
        "tcgen05_num_epi_warps",
        "tcgen05_persistent_grid_cap",
        "tcgen05_region_ab_stages",
        "tcgen05_region_c_stages",
        "tcgen05_region_pid_types",
        "tcgen05_sched_consumer_wait_mode",
        "tcgen05_sched_stage_count",
        "tcgen05_split_k",
        "tcgen05_tile_n",
        "tcgen05_tvm_ffi_launch",
        "tcgen05_warp_register_reallocation",
    }
)


def test_diagnostic_keys_are_the_debug_toggles() -> None:
    assert sorted(CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS) == [
        "tcgen05_cubin_lineinfo",
        "tcgen05_diagnostic_invalid_output",
    ]
    assert {"tcgen05_large_bn_proof"} == CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS
    # Searched, seeded and default-selected: not diagnostics.
    assert "tcgen05_tvm_ffi_launch" in CUTE_TCGEN05_TUNABLE_KEYS
    assert "tcgen05_flat_role_coordinates" in CUTE_TCGEN05_TUNABLE_KEYS


def test_key_tables_partition_the_former_surface() -> None:
    tables = (
        frozenset(CUTE_TCGEN05_TUNABLE_KEYS),
        CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS,
        CUTE_TCGEN05_GROUPED_CONFIG_KEYS,
        CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS,
    )
    assert len(CUTE_TCGEN05_TUNABLE_KEYS) == len(tables[0])
    assert sum(len(table) for table in tables) == len(frozenset().union(*tables))
    assert frozenset().union(*tables) == _FORMER_SURFACE
    assert _FORMER_SURFACE | CUTE_TCGEN05_STRATEGY_CONFIG_KEYS <= BACKEND_SPECIFIC_KEYS


# Retired tcgen05 axes: an explicit Config that still pins one must fail
# validation naming the key, not render a kernel or silently ignore it.
_RETIRED_KNOBS = (
    # The pure-matmul role lifecycle strategy (gemm-core-design-06; no
    # recorded campaign winner used it).
    (
        {"tcgen05_strategy": "pure_matmul_role_lifecycle"},
        r"tcgen05_strategy must be one of .*pure_matmul_role_lifecycle",
    ),
    # The invalid-output pipeline-edge modes and the clustered CtaGroup.ONE
    # bridge flag (gemm-core-structure-04 / gemm-core-design-07): the keys no
    # longer exist, so an explicit Config naming one is refused as unknown.
    *(
        ({key: value}, rf"Invalid config keys \['{key}'\]")
        for key, value in (
            ("tcgen05_acc_producer_mode", "skip_umma"),
            ("tcgen05_acc_producer_advance_mode", "skip"),
            ("tcgen05_ab_producer_acquire_mode", "skip"),
            ("tcgen05_ab_initial_producer_acquire_mode", "skip_first"),
            ("tcgen05_ab_producer_advance_mode", "skip"),
            ("tcgen05_ab_consumer_wait_mode", "skip"),
            ("tcgen05_ab_consumer_phase_mode", "phase1"),
            ("tcgen05_cluster_m2_one_cta_role_local", True),
        )
    ),
)


@pytest.mark.parametrize(("knob", "message"), _RETIRED_KNOBS)
def test_retired_tcgen05_knobs_are_rejected(
    knob: dict[str, object], message: str
) -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = scaled_mm._bind_isolated(
            scaled_args(512, 512, 256, dtype=torch.float16)
        )
    config = helion.Config(block_sizes=[128, 128, 64], pid_type="flat", **knob)
    with pytest.raises(InvalidConfig, match=message):
        bound.config_spec.normalize(config)


def test_inert_search_axes_are_declared_once() -> None:
    assert TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES == (4,)
    assert TCGEN05_C_STAGES_CHOICES == (2, 3, 4, 1)
    assert TCGEN05_C_STAGES_SEARCH_CHOICES == (2, 4)
    assert set(TCGEN05_C_STAGES_SEARCH_CHOICES) < set(TCGEN05_C_STAGES_CHOICES)
    assert TCGEN05_L2_SWIZZLE_SEARCH_SIZES == (1, 2, 4, 8)
    assert set(TCGEN05_L2_SWIZZLE_SEARCH_SIZES) < set(TCGEN05_LEGAL_L2_SWIZZLE_SIZES)


@pytest.mark.parametrize("admitted", (False, True))
def test_inert_axes_are_validated_wide_and_searched_narrow(admitted: bool) -> None:
    """The epilogue warp count, the C ring depth and the L2 swizzle keep their
    explicit-config range while the search sees one value, two values and the
    sizes up to eight; admission (``narrow_autotune_to_validated_configs``) only
    restates the pinned warp count."""
    with cpu_codegen_target(), forbid_cuda_work():
        bound = scaled_mm._bind_isolated(
            scaled_args(512, 512, 256, dtype=torch.float16)
        )
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert spec.tcgen05.search_enabled
    if not admitted:
        # Before admission the seams are unset; the search is pinned anyway.
        state.num_epi_warps_search_choices = None
        state.num_epi_warps_validation_choices = None
    search = surface.optional_fragments(state, for_search=True)
    validation = surface.optional_fragments(state, for_search=False)
    assert search["tcgen05_num_epi_warps"] == EnumFragment((4,))
    assert validation["tcgen05_num_epi_warps"] == (
        EnumFragment((4,)) if admitted else IntegerFragment(1, 4, 4)
    )
    assert search["tcgen05_c_stages"] == EnumFragment(
        (2, 3, 4, 1), search_choices=(2, 4)
    )
    assert validation["tcgen05_c_stages"] == search["tcgen05_c_stages"]
    assert search["tcgen05_l2_swizzle_size"] == EnumFragment((1, 2, 4, 8))
    assert validation["tcgen05_l2_swizzle_size"] == EnumFragment((1, 2, 4, 8, 16, 32))
    # The seam still lifts the gate for an explicit experiment.
    surface.restrict_num_epi_warps_search(state, (2, 4))
    assert surface.optional_fragments(state, for_search=True)[
        "tcgen05_num_epi_warps"
    ] == (EnumFragment((2, 4)))


# The five pinned-schedule surfaces as they were spelled out before they were
# composed over ``TCGEN05_ROLE_LOCAL_SCHEDULE`` (exact key sets and values;
# ``one_cta_pipeline_defaults`` also keeps its key order, it seeds configs).
_ONE_CTA_PIPELINE_DEFAULTS = {
    "cute_matmul_family": "tcgen05",
    "tcgen05_cta_group": "auto",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_persistence_model": "static_persistent",
    "tcgen05_layout_strategy": "default",
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_aux_load_mode": "simt",
    "tcgen05_aux_load_placement": "post_acc_wait",
    "tcgen05_tvm_ffi_launch": False,
    "tcgen05_flat_role_coordinates": False,
    "tcgen05_epilogue_fanout": "off",
    "tcgen05_epilogue_layout": "normal",
    "cute_materialized_operand_schedule": "off",
    "cute_materialized_schedule": "off",
    "cute_split_k_schedule": "legacy",
}
_N_EDGE_CLUSTER_SCHEDULE_DEFAULTS = {
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_persistence_model": "static_persistent",
    "tcgen05_layout_strategy": "default",
    "tcgen05_cta_group": "auto",
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_l2_swizzle_size": 1,
    "l2_groupings": [1],
    "tcgen05_aux_load_mode": "simt",
}
_REGION_TWO_CTA_LAUNCH_PROJECTION = {
    "tcgen05_persistence_model": "static_persistent",
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_layout_strategy": "default",
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_store_mode": "normal",
    "tcgen05_acc_wait_placement": "subtile_loop",
    "tcgen05_epilogue_layout": "normal",
    "tcgen05_tvm_ffi_launch": False,
    "tcgen05_flat_role_coordinates": False,
    "tcgen05_layout_overrides_epi_tile_m": None,
    "tcgen05_layout_overrides_epi_tile_n": None,
    "tcgen05_layout_overrides_smem_swizzle_a": None,
    "tcgen05_layout_overrides_smem_swizzle_b": None,
    "tcgen05_layout_overrides_d_store_box_n": None,
}
_FANOUT_SCHEDULE = {
    "pid_type": "persistent_interleaved",
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_persistence_model": "static_persistent",
    "tcgen05_num_epi_warps": 4,
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_layout_strategy": "default",
    "tcgen05_epilogue_layout": "normal",
    "tcgen05_c_store_mode": "normal",
    "tcgen05_acc_wait_placement": "subtile_loop",
}
_ROW_UNION_CARRIER_SCHEDULE = {
    "pid_type": "persistent_interleaved",
    "num_sm_multiplier": 1,
    "tcgen05_cta_group": "auto",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 2,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_l2_swizzle_size": 1,
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_persistence_model": "static_persistent",
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_warp_spec_register_decrease": 120,
    "tcgen05_warp_spec_register_increase": 256,
    "tcgen05_layout_strategy": "default",
    "tcgen05_grouped_full_coverage": "off",
    "tcgen05_flat_role_coordinates": False,
    "tcgen05_diagnostic_invalid_output": False,
}


def _typed_items(values: dict[str, object]) -> set[tuple[str, type, object]]:
    return {(key, type(value), repr(value)) for key, value in values.items()}


def test_canonical_role_local_schedule_is_the_shared_core() -> None:
    schedule = dict(TCGEN05_ROLE_LOCAL_SCHEDULE)
    assert len(schedule) == len(TCGEN05_ROLE_LOCAL_SCHEDULE)
    assert tcgen05_role_local_schedule() == schedule
    assert list(tcgen05_role_local_schedule()) == [
        k for k, _ in TCGEN05_ROLE_LOCAL_SCHEDULE
    ]
    assert tcgen05_role_local_schedule(without=("tcgen05_num_epi_warps",)) == {
        key: value for key, value in schedule.items() if key != "tcgen05_num_epi_warps"
    }
    for defaults in (
        _ONE_CTA_PIPELINE_DEFAULTS,
        _REGION_TWO_CTA_LAUNCH_PROJECTION,
        _ROW_UNION_CARRIER_SCHEDULE,
    ):
        assert all(defaults[key] == value for key, value in schedule.items())


def test_one_cta_pipeline_defaults_match_their_literal_in_order() -> None:
    defaults = one_cta_pipeline_defaults()
    assert _typed_items(defaults) == _typed_items(_ONE_CTA_PIPELINE_DEFAULTS)
    assert list(defaults) == list(_ONE_CTA_PIPELINE_DEFAULTS)


def test_n_edge_cluster_schedule_defaults_match_their_literal() -> None:
    defaults = n_edge_cluster_schedule_defaults()
    assert _typed_items(defaults) == _typed_items(_N_EDGE_CLUSTER_SCHEDULE_DEFAULTS)
    # The AB-load and MMA warp counts are left to the strategy's defaults.
    assert "tcgen05_warp_spec_ab_load_warps" not in defaults
    assert "tcgen05_warp_spec_mma_warps" not in defaults


def test_region_two_cta_launch_projection_matches_its_literal() -> None:
    projection = dict(TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION)
    assert len(projection) == len(TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION)
    assert _typed_items(projection) == _typed_items(_REGION_TWO_CTA_LAUNCH_PROJECTION)
    # The aux load mode, the CTA group and the cluster are judged per launch.
    for key in (
        "tcgen05_aux_load_mode",
        "tcgen05_cta_group",
        "tcgen05_cluster_m",
        "tcgen05_cluster_n",
    ):
        assert key not in projection


def test_fanout_and_row_union_schedules_match_their_literals() -> None:
    assert _typed_items(fanout_schedule()) == _typed_items(_FANOUT_SCHEDULE)
    assert "tcgen05_warp_spec_epi_load_warps" not in fanout_schedule()
    assert _typed_items(carrier_schedule()) == _typed_items(_ROW_UNION_CARRIER_SCHEDULE)
