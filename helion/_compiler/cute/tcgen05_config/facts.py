"""Proof facts of the tcgen05 GEMM: what the analyses registered at bind time.

Module functions over the ``CuteTcgen05Config`` state object: the
``register_*`` entry points record the analyses' results (matmul block ids,
pipeline SMEM facts, full-tile / N-edge schedule facts, the grouped row-union
register-MMA admission) and the readers derive block indices, tiles and
launch facts from them.  Nothing here reads a config dict's tunable keys
except through ``_matmul_config_view``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ....exc import InvalidConfig
from ....runtime.config import Config
from ..cute_warp_mma_gemm import WARP_MMA_GROUPED_MAX_GROUPS
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BM
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BN
from ..cute_warp_mma_gemm import warp_mma_admission_reason
from ..dense_aligned_n import ALIGNED_N_CHOICES
from ..dense_aligned_n import TILE_N_KEY
from ..dense_aligned_n import DenseAlignedNPlan
from ..dense_aligned_n import carrier_n
from ..dense_aligned_n import dense_aligned_n_defaults
from ..dense_aligned_n import dense_aligned_n_plan
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from ..grouped_row_union import PAIRED_CLC_SCHEDULE
from ..grouped_row_union import TRANSPOSED_SCHEDULE
from ..grouped_row_union import physical_schedule
from ..grouped_row_union import schedule_supported as row_union_schedule_supported
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..pipeline_smem import Tcgen05PipelineSmemFacts
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ..strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
from ..strategies import Tcgen05LayoutStrategy
from ..strategies import Tcgen05PersistenceModel
from ..strategies import tcgen05_explicit_epilogue_tile_supported
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODES_STAGED
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_N
from ..tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_N
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_CHOICES
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_INHERIT
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..tcgen05_schedule_facts import Tcgen05FullTileScheduleFacts
from ..tcgen05_schedule_facts import n_edge_cluster_geometry_supported
from ..tcgen05_schedule_facts import native_tma_schedule_supported
from . import seeds
from . import smem
from . import surface

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from ....autotuner.config_fragment import BlockSizeFragment
    from ....runtime.config import PidTypeLiteral
    from ...compile_environment import CompileEnvironment
    from ..grouped_prefix_adaptive_n import GroupedPrefixAdaptiveNProfile


from .knobs import _TWO_CTA_PERSISTENT_PID_VETO
from .knobs import Tcgen05GroupedWorklistSmemFacts
from .knobs import _tcgen05_mma_impl_allowed
from .knobs import _tile_in_fragments

if TYPE_CHECKING:
    from . import CuteTcgen05Config


def _preferred_persistent_pid_type(state: CuteTcgen05Config) -> PidTypeLiteral | None:
    return next(
        (
            pid
            for pid in ("persistent_interleaved", "persistent_blocked")
            if pid in state.allowed_pid_types
        ),
        None,
    )


def _two_cta_restorable_persistent_pids(
    state: CuteTcgen05Config,
) -> tuple[PidTypeLiteral, ...]:
    """Persistent pids whose only veto is the CuTe two-CTA contract."""
    reasons = state.config_spec.disallowed_pid_type_reasons
    causes = state.config_spec.disallowed_pid_type_causes
    return cast(
        "tuple[PidTypeLiteral, ...]",
        tuple(
            pid
            for pid in ("persistent_blocked", "persistent_interleaved")
            if pid not in state.allowed_pid_types
            and reasons.get(pid) == _TWO_CTA_PERSISTENT_PID_VETO
            and causes.get(pid) == {_TWO_CTA_PERSISTENT_PID_VETO}
        ),
    )


def _apply_restored_pid_types(
    state: CuteTcgen05Config, restored: tuple[PidTypeLiteral, ...]
) -> None:
    state.allowed_pid_types = (*state.allowed_pid_types, *restored)
    for pid in restored:
        state.config_spec.disallowed_pid_type_reasons.pop(pid)
        state.config_spec.disallowed_pid_type_causes.pop(pid)


def _config_block_index(state: CuteTcgen05Config, block_id: int | None) -> int | None:
    if (
        block_id is None
        or block_id not in state.config_spec.block_sizes.valid_block_ids()
    ):
        return None
    return state.config_spec.block_sizes.block_id_to_index(block_id)


def register_mma_analysis(
    state: CuteTcgen05Config,
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
    assert state.matmul_block_ids is None, "tcgen05 MMA analysis registered twice"
    state.matmul_block_ids = (m_block_id, n_block_id, k_block_id)
    state.matmul_compile_time_static_extents = compile_time_static_extents
    state.matmul_input_dtype = input_dtype
    state.matmul_has_leading_passthrough = has_leading_passthrough
    state.matmul_leading_work_multiplier = max(1, leading_work_multiplier)
    state.matmul_explicit_epi_tile_compatible = explicit_epi_tile_compatible


def _matmul_block_indices(state: CuteTcgen05Config) -> tuple[int, int, int] | None:
    if state.matmul_block_ids is None:
        return None
    indices = tuple(
        _config_block_index(state, block_id) for block_id in state.matmul_block_ids
    )
    if any(index is None for index in indices):
        return None
    return cast("tuple[int, int, int]", indices)


def row_union_profile_supported(state: CuteTcgen05Config, name: str) -> bool:
    if name == TRANSPOSED_SCHEDULE:
        return state.grouped_row_union_cluster4_supported
    if name == PAIRED_CLC_SCHEDULE:
        return state.grouped_row_union_paired_clc_supported
    return False


def project_row_union_codegen(state: CuteTcgen05Config, config: Config) -> Config:
    profile = physical_schedule(config)
    if profile is None:
        return config
    indices = _matmul_block_indices(state)
    if (
        not row_union_profile_supported(state, profile.name)
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


def register_pipeline_smem_facts(
    state: CuteTcgen05Config, facts: Tcgen05PipelineSmemFacts
) -> None:
    state.pipeline_smem_facts = facts
    if facts.input_dtype_bytes == 1:
        # Allocation knowledge alone must not resurrect FP8 schedules.
        # The native TMA/schedule registrar owns their PID restrictions.
        return
    if surface.paired_pipeline_search_enabled(state):
        # The explicit instruction family owns its own 128-row geometry.
        # It need not satisfy the older 256x256 search projection.
        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            if pid_type not in state.allowed_pid_types:
                state.allowed_pid_types = (
                    *state.allowed_pid_types,
                    cast("PidTypeLiteral", pid_type),
                )


def _full_tile_schedule_tiles(
    state: CuteTcgen05Config, facts: Tcgen05FullTileScheduleFacts, *, cluster_m: int
) -> list[tuple[int, int, int]]:
    fragments = _matmul_block_fragments(state)
    if fragments is None:
        return []
    return [
        (bm, bn, bk)
        for bm in (64, 128, 256)
        for bn in (16, 32, 64, 128, 256)
        for bk in (32, 64, 128, 256)
        if native_tma_schedule_supported(
            facts, (bm, bn, bk), cluster_m=cluster_m, cluster_n=1
        )
        and _tile_in_fragments(fragments, (bm, bn, bk))
    ]


def register_full_tile_schedule_facts(
    state: CuteTcgen05Config, facts: Tcgen05FullTileScheduleFacts
) -> None:
    """Restore only CuTe-owned search restrictions with a new tile proof.

    The frontend's maximum tile can have a tail even when smaller legal
    tiles do not. Also, a quarter-wave preference does not change the
    correctness of an otherwise admitted paired-CTA instruction.
    """
    if state.full_tile_schedule_facts is not None:
        assert state.full_tile_schedule_facts == facts
        return
    if facts.block_ids != state.matmul_block_ids:
        return
    if not _full_tile_schedule_tiles(state, facts, cluster_m=1):
        return
    restored = _two_cta_restorable_persistent_pids(state)
    persistent = "persistent_interleaved" in (*state.allowed_pid_types, *restored)
    added_pairing = (
        persistent
        and state.cluster_m_search_choices is not None
        and 2 not in state.cluster_m_search_choices
        and bool(_full_tile_schedule_tiles(state, facts, cluster_m=2))
    )
    if not restored and not added_pairing:
        return
    state.full_tile_schedule_facts = facts
    state.full_tile_restored_pid_types = restored
    state.full_tile_added_pairing = added_pairing
    _apply_restored_pid_types(state, restored)


def _n_edge_cluster_schedule_tiles(
    state: CuteTcgen05Config, facts: Tcgen05FullTileScheduleFacts
) -> list[tuple[int, int, int]]:
    fragments = _matmul_block_fragments(state)
    if fragments is None:
        return []
    return [
        (256, bn, bk)
        for bn in (128, 256)
        for bk in (64, 128)
        if n_edge_cluster_geometry_supported(facts, (256, bn, bk))
        and _tile_in_fragments(fragments, (256, bn, bk))
    ]


def register_n_edge_cluster_schedule_facts(
    state: CuteTcgen05Config, facts: Tcgen05FullTileScheduleFacts
) -> None:
    """Admit one unsplit multicast family from the native graph proof."""
    if state.n_edge_cluster_schedule_facts is not None:
        assert state.n_edge_cluster_schedule_facts == facts
        return
    if facts.block_ids != state.matmul_block_ids or not (
        _n_edge_cluster_schedule_tiles(state, facts)
    ):
        return
    restored = _two_cta_restorable_persistent_pids(state)
    if "persistent_interleaved" not in (*state.allowed_pid_types, *restored):
        return
    state.n_edge_cluster_schedule_facts = facts
    state.n_edge_cluster_restored_pid_types = restored
    _apply_restored_pid_types(state, restored)


def _fp8_paired_pipeline_tile_supported(
    state: CuteTcgen05Config, bm: int, bn: int, bk: int
) -> bool:
    """Keep allocation accounting within proved native paired schedules."""
    shape = state.matmul_compile_time_static_extents
    axes = state.matmul_block_ids
    if (
        not state.matmul_operands_tma_provable
        or state.config_spec.target_device_capability not in ((10, 0), (10, 3))
        or state.materialized_matmul_block_ids
        or axes is None
        or shape is None
        or state.matmul_input_dtype != torch.float8_e4m3fn
        or any(type(extent) is not int or extent <= 0 for extent in shape)
    ):
        return False
    # K tails additionally need the RHS-major proof from native analysis.
    # Exact tiles retain the conservative fallback for already-admitted grids.
    facts = state.full_tile_schedule_facts or Tcgen05FullTileScheduleFacts(
        axes, cast("tuple[int, int, int]", shape), torch.float8_e4m3fn
    )
    return native_tma_schedule_supported(
        facts,
        (bm, bn, bk),
        cluster_m=2,
        cluster_n=1,
    )


def _shared_ring_flat_launches(
    state: CuteTcgen05Config, config: dict[str, object]
) -> list[int]:
    """Regions of a flat chain whose GEMM launch reads ``tcgen05_ab_stages``.

    A launch left flat (``inherit``) keeps the kernel's ring unless its
    ``tcgen05_region_ab_stages`` entry replaces it (``_stage_config``);
    the entries bind only with the complete two-CTA region schedule
    (``_normalize_epilogue_fanout`` drops them otherwise), and a
    persistent launch's ring is validated per region
    (``_validate_cta_group_config``).  Empty where no launch reads the
    kernel's knob.
    """
    if config.get("pid_type") != "flat":
        return []
    pid_types = surface._region_pid_types(state, config)
    counts = config.get("tcgen05_region_ab_stages")
    if (
        not isinstance(counts, list)
        or len(counts) != len(pid_types)
        or smem._materialized_paired_stage_limits(state, config) is None
    ):
        counts = [0] * len(pid_types)
    return [
        region
        for region, (pid_type, count) in enumerate(zip(pid_types, counts, strict=True))
        if pid_type == "flat" and not count
    ]


def _region_two_cta_schedule_supported(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    """The standard role-local static persistent CtaGroup.TWO schedule.

    The keys the single-GEMM paired pipeline fixes
    (``_paired_pipeline_parameters``), judged on the whole-kernel config
    of a fission plan.  An ordered barrier chain's whole-kernel config
    keeps the flat recipe of its ``inherit`` launches; its persistent
    launches receive the paired recipe by projection
    (``TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION``), so only the
    launch-independent coordinates are judged there.
    """
    if state.region_pid_type_search_enabled:
        return config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is None and not config.get(
            "tcgen05_diagnostic_invalid_output"
        )
    pid_type = config.get("pid_type")
    expected_model = (
        Tcgen05PersistenceModel.STATIC_PERSISTENT.value
        if pid_type in ("persistent_blocked", "persistent_interleaved")
        else Tcgen05PersistenceModel.NON_PERSISTENT.value
    )
    return (
        config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) is None
        and not config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY)
        and config.get(TCGEN05_STRATEGY_CONFIG_KEY, "role_local_monolithic")
        == "role_local_monolithic"
        and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, expected_model)
        == expected_model
        and config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, "default") == "default"
        and all(config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
        and all(
            config.get(key, 0) == 0
            for key in (
                TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
                TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
                TCGEN05_WARP_SPEC_STORE_WARPS_KEY,
            )
        )
        and config.get("tcgen05_num_epi_warps", 4) == 4
        # The register reads or the epilogue-issued ring, which the paired
        # pipeline renders without a producer warp; the producer-warp ring
        # (``tma``) is refused on chains (``_normalize_epi_tma``).
        and config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT)
        in (TCGEN05_AUX_LOAD_MODE_SIMT, TCGEN05_AUX_LOAD_MODE_EPI_TMA)
        and config.get("tcgen05_c_store_mode", "normal") == "normal"
        and config.get(
            TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
            TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
        )
        == TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
        and config.get("tcgen05_epilogue_layout", "normal") == "normal"
        and config.get("tcgen05_acc_stages", 2) == 2
        and not config.get("tcgen05_diagnostic_invalid_output")
    )


def _warp_mma_problem(
    state: CuteTcgen05Config,
) -> tuple[int, int, int, torch.dtype] | None:
    """``(m, n, k, dtype)`` of the admitted register-MMA GEMM, or None."""
    extents = state.matmul_compile_time_static_extents
    if extents is None or any(value is None for value in extents):
        extents = state.warp_mma_grouped_extents
    dtype = state.matmul_input_dtype
    if (
        not state.search_enabled
        or not state.warp_mma_admitted
        or extents is None
        or dtype is None
        or any(value is None or value <= 0 for value in extents)
    ):
        return None
    m, n, k = cast("tuple[int, int, int]", extents)
    return m, n, k, dtype


def register_grouped_row_union_warp_mma(
    state: CuteTcgen05Config,
    env: CompileEnvironment,
    *,
    groups: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    b_major: str,
) -> None:
    """Admit the register-MMA family to the proved grouped row-union kernel.

    The shared-RHS grouped GEMM (``out[rows of group g] = A_packed[rows]
    @ B``) is one dense GEMM over the union of the group rows, with a
    per-row membership predicate on the stores; the family lowers it like
    a plain GEMM when the first binding's extents are in its latency-bound
    regime (``detect_warp_mma_gemm`` re-proves the semantics and
    specializes the extents at codegen).  Its 16x8 tiles widen the M / N
    search floors the way the static-shape admission does.
    """
    if (
        not state.search_enabled
        or not state.grouped_row_union_supported
        or state.matmul_block_ids is None
        or state.matmul_input_dtype != dtype
        # The body's per-thread interval setup and membership chains
        # grow with the group count (``detect_warp_mma_grouped_gemm``
        # declines above the cap).
        or not 0 < groups <= WARP_MMA_GROUPED_MAX_GROUPS
    ):
        return
    reason = warp_mma_admission_reason(
        static_m=m,
        static_n=n,
        static_k=k,
        leading=1,
        dtype=dtype,
        lhs_major="row",
        rhs_major=b_major,
        num_sms=state.device_sm_count,
        lhs_strides=(k, 1),
        rhs_strides=(n, 1) if b_major == "row" else (1, k),
    )
    if reason is not None:
        return
    state.warp_mma_admitted = True
    state.warp_mma_grouped_extents = (m, n, k)
    m_block_id, n_block_id, _k_block_id = state.matmul_block_ids
    for block_id, min_size in (
        (m_block_id, WARP_MMA_MIN_BM),
        (n_block_id, WARP_MMA_MIN_BN),
    ):
        # The tcgen05 search already raised the block floor to its own
        # (``enable_cute_tcgen05_search`` ran before this heuristic);
        # lower it to the family's atom and keep the default
        # (non-autotuned) block where it was, as the static-shape
        # admission does -- tcgen05-family configs are clamped back to
        # ``tcgen05_min_search_m`` / ``_n`` by the family normalizer.
        with contextlib.suppress(KeyError):
            block_spec = state.config_spec.block_sizes.block_id_lookup(block_id)
            if min_size >= block_spec.min_size:
                continue
            if block_spec.default_size is None:
                block_spec.default_size = block_spec._fragment(
                    state.config_spec
                ).default()
            block_spec.min_size = min_size
            block_spec.autotuner_min = min(block_spec.autotuner_min, min_size)


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


def materialized_operand_pdl_supported(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    # The static persistent role-local schedules (the paired TWO family and
    # the plain one-CTA family) emit a TMA-role dependency wait before all
    # initial/refill operand reads. Alternative/fused schedules do not
    # inherit this proof or the separate producer launch.
    parameters = seeds._paired_pipeline_parameters(state, config)
    tiles = (
        parameters[:3]
        if parameters is not None
        else seeds._one_cta_persistent_parameters(state, config)
    )
    extents = state.matmul_compile_time_static_extents
    if (
        state.materialized_operand_pdl_roots is None
        or tiles is None
        or extents is None
        or any(
            extent is None or extent % tile != 0
            for extent, tile in zip(extents, tiles, strict=True)
        )
    ):
        return False
    if parameters is not None:
        limit = smem._paired_pipeline_stage_limit(state, config)
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
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> tuple[list[object], int, int, int] | None:
    """``(block_sizes, m, n, k indices)`` of the GEMM ``config`` tiles.

    The standalone surface's matmul (``matmul_block_ids``), or -- with
    ``region`` -- that GEMM launch of a materialized chain
    (``materialized_matmul_block_ids``), whose epilogue knobs are judged
    per launch (``chain_launches``).
    """
    block_sizes = config.get("block_sizes")
    indices: tuple[int, int, int] | None
    if region is not None:
        if region >= len(state.materialized_matmul_block_ids):
            return None
        region_indices = tuple(
            _config_block_index(state, axis)
            for axis in state.materialized_matmul_block_ids[region]
        )
        if any(index is None for index in region_indices):
            return None
        indices = cast("tuple[int, int, int]", region_indices)
    else:
        indices = _matmul_block_indices(state)
    if not isinstance(block_sizes, list) or indices is None:
        return None
    m_index, n_index, k_index = indices
    if max(indices) >= len(block_sizes):
        return None
    return block_sizes, m_index, n_index, k_index


def region_launch_facts(
    state: CuteTcgen05Config, region: int
) -> Tcgen05PipelineSmemFacts | None:
    """Allocation facts of GEMM launch ``region``: the single-output proof,
    else the typed pair proof (two outputs the shared fanout merges)."""
    facts = (
        state.materialized_pipeline_facts[region]
        if region < len(state.materialized_pipeline_facts)
        else None
    )
    if facts is None and region < len(state.materialized_pair_budget_facts):
        facts = state.materialized_pair_budget_facts[region]
    return facts


def region_has_exact_aux(state: CuteTcgen05Config, region: int) -> bool:
    """Whether GEMM launch ``region``'s epilogue reads an exact-shape
    auxiliary operand of the output's dtype (the staged and prefetched
    operand of the aux mechanisms); False without the single-output
    allocation proof that records it."""
    facts = (
        state.materialized_pipeline_facts[region]
        if region < len(state.materialized_pipeline_facts)
        else None
    )
    return facts is not None and facts.exact_aux_count > 0


def _configured_region_pid_types(
    state: CuteTcgen05Config, config: dict[str, object]
) -> list[object] | None:
    """A flat ordered chain's well-formed ``tcgen05_region_pid_types``."""
    values = config.get(TCGEN05_REGION_PID_TYPES_CONFIG_KEY)
    if (
        config.get("pid_type") == "flat"
        and isinstance(values, list)
        and len(values) == len(state.materialized_matmul_block_ids)
        and all(value in TCGEN05_REGION_PID_TYPE_CHOICES for value in values)
    ):
        return list(values)
    return None


def region_launch_config(
    state: CuteTcgen05Config,
    config: dict[str, object],
    region: int,
    *,
    owns_fanout_pair: bool | None = None,
) -> dict[str, object]:
    """The config GEMM launch ``region`` of a materialized chain renders under.

    ``materialized_fission_codegen._stage_config`` renders exactly this
    (the normalizer judges what the renderer sees): under a flat ordered
    chain's ``tcgen05_region_pid_types`` a persistent launch takes its
    pid type and the paired pipeline's fixed coordinates
    (``TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION``) while an ``inherit``
    launch sheds the cluster; a non-zero ``tcgen05_region_ab_stages`` /
    ``tcgen05_region_c_stages`` entry replaces the shared ring depth; a
    launch that does not own a shared-fanout pair drops the fanout and
    its placement; and a launch without an exact-shape auxiliary operand
    takes the register aux reads (no staged ring, no prefetch ring, no
    hoist).  ``owns_fanout_pair`` defaults to the pair proof over the
    launch's (m, n) axes.
    """
    launch = dict(config)
    launch.pop(TCGEN05_REGION_PID_TYPES_CONFIG_KEY, None)
    pid_types = _configured_region_pid_types(state, config)
    if pid_types is not None:
        pid_type = pid_types[region]
        if pid_type != TCGEN05_REGION_PID_TYPE_INHERIT:
            launch["pid_type"] = pid_type
            launch.update(TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION)
        else:
            launch["tcgen05_cluster_m"] = 1
            launch["tcgen05_cluster_n"] = 1
            launch[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
    for kind in ("ab", "c"):
        stages = launch.pop(f"tcgen05_region_{kind}_stages", None)
        if isinstance(stages, list) and region < len(stages):
            count = stages[region]
            if type(count) is int and count:
                launch[f"tcgen05_{kind}_stages"] = count
    if launch.get(FANOUT_CONFIG_KEY) == "shared":
        if owns_fanout_pair is None:
            axes = set(state.materialized_matmul_block_ids[region])
            owns_fanout_pair = any(
                set(pair.block_ids) <= axes for pair in state.epilogue_fanout_plans
            )
        if not owns_fanout_pair:
            launch.pop(FANOUT_CONFIG_KEY)
            launch.pop(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY, None)
            if launch.get(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY) == "before_store":
                launch.pop(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY)
    if not region_has_exact_aux(state, region):
        if launch.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) in (
            TCGEN05_AUX_LOAD_MODES_STAGED
        ):
            launch[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
        if launch.get(TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY):
            launch[TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY] = 0
        if launch.get(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY) is True:
            launch[TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY] = False
    return launch


def chain_launches(
    state: CuteTcgen05Config, config: dict[str, object], *, aux_only: bool = False
) -> list[tuple[int, dict[str, object]]]:
    """``(region, launch config)`` of every GEMM launch of a materialized
    chain (``aux_only``: those with an exact-shape auxiliary operand)."""
    return [
        (region, region_launch_config(state, config, region))
        for region in range(len(state.materialized_matmul_block_ids))
        if not aux_only or region_has_exact_aux(state, region)
    ]


def _chain_launch_refusal(
    state: CuteTcgen05Config,
    config: dict[str, object],
    judge: Callable[[dict[str, object], int], str | None],
    *,
    aux_only: bool,
) -> str | None:
    """``judge`` over the chain's GEMM launches; the first refusal, named
    by its launch, or the lack of a launch the knob could render on."""
    launches = chain_launches(state, config, aux_only=aux_only)
    if not launches:
        return (
            "needs a GEMM launch of the materialized chain with an exact-shape "
            "auxiliary operand and its single-output allocation proof"
        )
    for region, launch in launches:
        reason = judge(launch, region)
        if reason is not None:
            return f"{reason} (GEMM launch {region} of the materialized chain)"
    return None


def _launch_pipeline_facts(
    state: CuteTcgen05Config, region: int | None
) -> Tcgen05PipelineSmemFacts | None:
    """Allocation facts of the launch the epilogue knobs are judged on."""
    if region is None:
        return state.pipeline_smem_facts
    return region_launch_facts(state, region)


def _matmul_block_fragments(
    state: CuteTcgen05Config,
) -> tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment] | None:
    indices = _matmul_block_indices(state)
    if indices is None:
        return None
    return cast(
        "tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment]",
        tuple(
            cast(
                "BlockSizeFragment",
                state.config_spec.block_sizes[index]._fragment(state.config_spec),
            )
            for index in indices
        ),
    )


def _matmul_seed_block_sizes(
    state: CuteTcgen05Config, *, bm: int, bn: int, bk: int
) -> list[int] | None:
    indices = _matmul_block_indices(state)
    if indices is None:
        return None
    block_sizes = [
        cast(
            "BlockSizeFragment",
            spec._fragment(state.config_spec),
        ).default()
        for spec in state.config_spec.block_sizes
    ]
    m_index, n_index, k_index = indices
    block_sizes[m_index] = bm
    block_sizes[n_index] = bn
    block_sizes[k_index] = bk
    return block_sizes


def _direct_entry_k_block_index(state: CuteTcgen05Config) -> int | None:
    if state.matmul_has_non_tcgen05_operand or state.matmul_input_dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return None
    indices = _matmul_block_indices(state)
    return indices[2] if indices is not None else None


def _matmul_search_bounds(
    state: CuteTcgen05Config,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """Live M/N/K tile ranges; None when an outer tile bounds an axis."""
    axes = state.matmul_block_ids
    fragments = _matmul_block_fragments(state)
    if axes is None or fragments is None:
        return None
    bounds = []
    for axis, fragment in zip(axes, fragments, strict=True):
        spec = state.config_spec.block_sizes.block_id_lookup(axis)
        if spec.bounded_by_block_id is not None:
            return None
        bounds.append(
            (max(spec.min_size, fragment.low), min(spec.max_size, fragment.high))
        )
    return cast(
        "tuple[tuple[int, int], tuple[int, int], tuple[int, int]]", tuple(bounds)
    )


def _dense_aligned_n_bounds(
    state: CuteTcgen05Config,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    bounds = _matmul_search_bounds(state)
    # A pinned N block is an exact tile, not a carrier that may shrink.
    if bounds is None or bounds[1][0] == bounds[1][1]:
        return None
    return bounds


def _dense_aligned_n_plan(
    state: CuteTcgen05Config, config: dict[str, object]
) -> DenseAlignedNPlan | None:
    if config.get(TILE_N_KEY, 0) == 0:
        return None
    view = _matmul_config_view(state, config)
    bounds = _dense_aligned_n_bounds(state)
    facts = state.dense_gemm_facts
    if (
        view is None
        or bounds is None
        or facts is None
        or facts.block_ids != state.matmul_block_ids
        or facts.shape != state.matmul_compile_time_static_extents
        or facts.input_dtype != state.matmul_input_dtype
        or config.get("pid_type") not in state.allowed_pid_types
    ):
        return None
    blocks, mi, ni, ki = view
    return dense_aligned_n_plan(
        state.dense_gemm_facts,
        config,
        cast("tuple[int, int, int]", (blocks[mi], blocks[ni], blocks[ki])),
        bounds=bounds,
    )


def _dense_aligned_n_width_plans(
    state: CuteTcgen05Config, width: int
) -> Iterator[tuple[dict[str, Any], DenseAlignedNPlan]]:
    """Normal-store seed values (in seed order) proved for one physical N."""
    facts = state.dense_gemm_facts
    if (
        not state.search_enabled
        or facts is None
        # dense_aligned_n_plan needs 128-row tiles that divide M and a
        # physical width that divides N; skip widths that cannot qualify.
        or facts.shape[0] % 128
        or facts.shape[1] % width
        or not _tcgen05_mma_impl_allowed()
    ):
        return
    pid = _preferred_persistent_pid_type(state)
    if pid is None:
        return
    for bk in (64, 128):
        for acc_stages in (2, 1):
            values: dict[str, Any] = {
                **dense_aligned_n_defaults(acc_stages=acc_stages),
                TILE_N_KEY: width,
                "block_sizes": _matmul_seed_block_sizes(
                    state, bm=128, bn=carrier_n(width), bk=bk
                ),
                "pid_type": pid,
                "tcgen05_c_stages": 2,
            }
            plan = _dense_aligned_n_plan(state, values)
            if plan is not None:
                yield values, plan


def _dense_aligned_n_widths(state: CuteTcgen05Config) -> tuple[int, ...]:
    """Physical widths with at least one proved seed, in seed order."""
    return tuple(
        width
        for width in ALIGNED_N_CHOICES
        if next(_dense_aligned_n_width_plans(state, width), None) is not None
    )


def project_dense_aligned_n_codegen(state: CuteTcgen05Config, config: Config) -> Config:
    if config.get(TILE_N_KEY, 0) == 0:
        return config
    plan = _dense_aligned_n_plan(state, config.config)
    if plan is None:
        raise InvalidConfig("physical N projection lacks its typed dense plan")
    indices = _matmul_block_indices(state)
    assert indices is not None
    blocks = list(config.block_sizes)
    for index, value in zip(indices, plan.physical_tile, strict=True):
        blocks[index] = value
    return Config.from_dict(config.config | {"block_sizes": blocks})


def get_grouped_prefix_adaptive_n_profile(
    state: CuteTcgen05Config, config: Mapping[str, object]
) -> GroupedPrefixAdaptiveNProfile | None:
    source_m = config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
    if type(source_m) is not int:
        return None
    profile = state.grouped_prefix_adaptive_n_profiles.get(source_m)
    if profile is None and source_m == 128:
        # Preserve the existing single-profile interface for M128 callers.
        return state.grouped_prefix_adaptive_n_profile
    return profile


def register_grouped_worklist_smem_facts(
    state: CuteTcgen05Config,
    *,
    group_count: int,
    device_split_sizes: bool,
    output_dtype_bytes: int = 2,
) -> None:
    if group_count <= 0:
        raise ValueError("grouped worklist SMEM facts require a positive group count")
    facts = Tcgen05GroupedWorklistSmemFacts(
        group_count, device_split_sizes, output_dtype_bytes
    )
    if state.grouped_worklist_smem_facts not in (None, facts):
        raise RuntimeError(
            "conflicting grouped worklist SMEM facts were registered for one ConfigSpec"
        )
    state.grouped_worklist_smem_facts = facts


def _static_full_tiles(
    state: CuteTcgen05Config, *, bm: int, bn: int, bk: int, region: int | None = None
) -> bool:
    """Whole tiles of the compile-time static problem on every axis.

    The config-level form of ``cute_mma._mma_tiles_are_static_full``:
    False (nothing proved) without compile-time static extents.  With
    ``region`` the problem is that GEMM launch of a materialized chain
    (``materialized_matmul_shapes``).
    """
    extents: tuple[object, object, object] | None
    if region is not None:
        extents = (
            state.materialized_matmul_shapes[region]
            if region < len(state.materialized_matmul_shapes)
            else None
        )
    else:
        extents = state.matmul_compile_time_static_extents
    if extents is None:
        return False
    m, n, k = extents
    return (
        isinstance(m, int)
        and isinstance(n, int)
        and isinstance(k, int)
        and m % bm == 0
        and n % bn == 0
        and k % bk == 0
    )


def _config_is_two_cta_mma(
    state: CuteTcgen05Config, config: dict[str, object], *, bm: int, cluster_m: int
) -> bool:
    """Whether ``config`` runs the tcgen05 CTA-pair MMA.

    Mirrors ``cute_mma._tcgen05_use_2cta_instrs``: ``tcgen05_cluster_m=2``
    with the ``bm=256`` pair tile, the fp8 small-grid ``bm=128`` tile, or
    an explicit ``tcgen05_cta_group='two'``. Any other clustered config is
    the clustered single-CTA (CtaGroup.ONE) family.
    """
    if cluster_m != 2:
        return False
    if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto") == "two":
        return True
    if bm == TCGEN05_TWO_CTA_BLOCK_M:
        return True
    return (
        bm == TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
        and state.matmul_input_dtype == torch.float8_e4m3fn
    )


def _plain_narrow_subtile_facts(
    state: CuteTcgen05Config, *, bm: int, bn: int, cluster_m: int
) -> bool:
    """The epilogue facts of ``cute_mma``'s plain narrow-subtile rule.

    A plain 16-bit store on the 256-wide two-CTA tile -- one output, no
    broadcast rows, no auxiliary operand (an exact-shape one keeps the
    plan's with-source subtile) -- renders ``(128,
    TCGEN05_PLAIN_NARROW_SUBTILE_N)`` subtiles.  The plan also asks for
    static full tiles, a row-major output and a TMA-store epilogue; the
    callers that know those facts check them.
    """
    facts = state.rowvec_aux_facts
    constraints = state.ab_stages_three_search_constraints
    return (
        facts is not None
        and constraints is not None
        and facts.output_itemsize == 2
        and not facts.rows
        and facts.output_count == 1
        and not state.aux_kernel_detected
        and constraints.dtype_bytes == 2
        and cluster_m == 2
        and bn == TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_N
        and facts.exact_aux_count == 0
        and tcgen05_explicit_epilogue_tile_supported(
            is_two_cta=True,
            bm=bm,
            bn=bn,
            tile_shape=(
                128,
                TCGEN05_PLAIN_NARROW_SUBTILE_N,
                TCGEN05_PLAIN_NARROW_SUBTILE_N,
            ),
        )
    )


def _is_default_layout_full_tile_config(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
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
    config_view = _matmul_config_view(state, config)
    if config_view is None:
        return False
    block_sizes, m_index, n_index, _ = config_view
    return (
        block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
        and block_sizes[n_index] == TCGEN05_TWO_CTA_BLOCK_N
    )


def _matmul_fact_has_edge_tile(
    state: CuteTcgen05Config, config: dict[str, object], *, fact_index: int
) -> bool:
    block_sizes = config.get("block_sizes")
    if not isinstance(block_sizes, list):
        return False
    fact = state.config_spec.matmul_facts[fact_index]
    for static_size, block_id in (
        (fact.static_m, fact.m_block_id),
        (fact.static_n, fact.n_block_id),
        (fact.static_k, fact.k_block_id),
    ):
        if static_size is None or block_id is None:
            continue
        try:
            block_idx = state.config_spec.block_sizes.block_id_to_index(block_id)
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


def _has_any_matmul_fact_edge_tile(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    return any(
        _matmul_fact_has_edge_tile(state, config, fact_index=i)
        for i in range(len(state.config_spec.matmul_facts))
    )


def _has_any_matmul_fact_n_edge_for_block_n(
    state: CuteTcgen05Config, block_n: int
) -> bool:
    for fact in state.config_spec.matmul_facts:
        if fact.static_n is None or fact.n_block_id is None:
            continue
        try:
            # Presence check: skip facts whose N block id is not registered
            # in this config spec.
            state.config_spec.block_sizes.block_id_to_index(fact.n_block_id)
        except KeyError:
            continue
        if fact.static_n % block_n != 0:
            return True
    return False
