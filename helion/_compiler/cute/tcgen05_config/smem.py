"""Shared-memory pricing of tcgen05 schedules.

Module functions over the ``CuteTcgen05Config`` state object that decide
whether a tile / ring configuration fits the per-CTA budget: the AB-stage
limits of the one-CTA, paired and region pipelines, the default-layout byte
model, and the epilogue subtile and aux-prefetch depth.  The two pipeline
byte models are reached through the one pricer in
``helion/_compiler/cute/pipeline_smem.py`` (``SmemArena`` /
``max_pipeline_ab_stages``).  The per-CTA budget readers themselves
(``per_cta_smem_*``) stay on the facade class so tests can patch them there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .. import target_facts
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..epilogue_fanout import schedule_supported as fanout_schedule_supported
from ..grouped_ab_k import physical_grouped_ab_k
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..pipeline_smem import max_pipeline_ab_stages
from ..pipeline_smem import one_cta_pipeline_tile_modeled
from ..pipeline_smem import pipeline_smem_bytes
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ..strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
from ..strategies import Tcgen05LayoutStrategy
from ..strategies import Tcgen05PersistenceModel
from ..strategies import Tcgen05Strategy
from ..strategies import layout_overrides_from_config
from ..strategies import tcgen05_epilogue_subtile_m
from ..strategies import tcgen05_explicit_epilogue_tile_supported
from ..tcgen05.knobs import acc_stages_knob
from ..tcgen05.knobs import aux_load_mode_knob
from ..tcgen05.knobs import aux_stages_knob
from ..tcgen05.knobs import epi_subtile_n_knob
from ..tcgen05.knobs import epi_warp_groups_knob
from ..tcgen05.knobs import epi_warps_knob
from ..tcgen05_constants import TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODES_STAGED
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_MAX_SUBTILES
from ..tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from ..tcgen05_constants import TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_DYNAMIC_MODES
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SMEM_ROW_STAGE_CHUNK_BYTES
from ..tcgen05_constants import TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..tcgen05_constants import resolve_tcgen05_grouped_worklist_mma_profile
from ..tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from ..tcgen05_constants import tcgen05_aux_ring_smem_bytes_per_cta
from ..tcgen05_constants import tcgen05_aux_subtile_fragment_bytes_per_thread
from ..tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from ..tcgen05_constants import tcgen05_default_epilogue_tile_size
from ..tcgen05_constants import tcgen05_fixed_smem_overhead_bytes
from ..tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from ..tcgen05_constants import tcgen05_round_up_smem_bytes
from ..tcgen05_constants import tcgen05_rowvec_stage_smem_bytes
from . import facts
from . import repair
from . import seeds
from . import surface

if TYPE_CHECKING:
    import torch


from .knobs import TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES
from .knobs import TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES
from .knobs import TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH
from .knobs import TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
from .knobs import Tcgen05AbStagesThreeSearchConstraints

if TYPE_CHECKING:
    from . import CuteTcgen05Config


def _one_cta_pipeline_stage_limit(
    state: CuteTcgen05Config, config: dict[str, object]
) -> int | None:
    parameters = seeds._one_cta_pipeline_parameters(state, config)
    if parameters is None:
        return None
    facts = state.one_cta_pipeline_facts
    assert facts is not None
    bm, bn, bk, c_stages, acc_stages = parameters
    return max_pipeline_ab_stages(
        facts.allocations,
        bm=bm,
        bn=bn,
        bk=bk,
        c_stages=c_stages,
        acc_stages=acc_stages,
        cluster_m=1,
    )


def _paired_pipeline_stage_limit(
    state: CuteTcgen05Config, config: dict[str, object]
) -> int | None:
    parameters = seeds._paired_pipeline_parameters(state, config)
    if parameters is None:
        return None
    facts = state.pipeline_smem_facts
    assert facts is not None
    bm, bn, bk, c_stages, acc_stages = parameters
    return max_pipeline_ab_stages(
        facts,
        bm=bm,
        bn=bn,
        bk=bk,
        c_stages=c_stages,
        acc_stages=acc_stages,
        epi_warp_groups=epi_warp_groups_knob(config),
        epi_subtile_n=seeds._paired_pipeline_store_subtile_n(
            state, config, bm=bm, bn=bn, bk=bk
        ),
        aux_load_placement=cast(
            "str",
            config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY, "post_acc_wait"),
        ),
        store_mode=cast("str", config.get("tcgen05_c_store_mode", "normal")),
        epi_warps=cast("int", config.get("tcgen05_num_epi_warps", 4)),
        aux_load_mode=aux_load_mode_knob(config),
        aux_stages=aux_stages_knob(config),
    )


def _flat_region_ab_stage_cap(state: CuteTcgen05Config) -> int:
    """Deepest AB ring a flat GEMM launch of a chain may search.

    The dtype's hardware-validated cap (``_get_dtype_ab_stages_hard_cap``,
    the depth the standalone single-GEMM surface searches); each launch's
    tile is then clamped to its own one-CTA SMEM fit
    (``_flat_region_ab_stage_limits``).
    """
    return min(
        _get_dtype_ab_stages_hard_cap(pipeline_facts.input_dtype_bytes)
        for pipeline_facts in state.materialized_pipeline_facts
        if pipeline_facts is not None
    )


def _flat_region_ab_stage_limits(
    state: CuteTcgen05Config, config: dict[str, object]
) -> list[int] | None:
    """Deepest AB ring each GEMM launch of a chain admits as a flat launch.

    A flat (``inherit``) launch runs the one-CTA role-local pipeline on
    its region's tile.  Its ring is bounded exactly as the standalone
    single-GEMM surface bounds the same tile (``max_ab_stages_that_fit``:
    the AB ring within the per-CTA capacity less the
    ``TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES`` reservation, capped at
    the dtype's validated depth) and, where the complete one-CTA model
    applies (``one_cta_pipeline_tile_modeled``), tightened to the fit of
    the whole A and B tile beside the launch's DEFAULT epilogue subtile
    ring (``tcgen05_region_c_stages`` entry or the kernel's
    ``tcgen05_c_stages``, as ``_stage_config`` applies them) and the
    barriers, priced against the region's complete single-output
    allocation proof (``materialized_pipeline_facts``).  Rings the
    standalone surface refuses are never admitted here: at bm=128/bk=128
    the deeper 16-bit rings it refuses were measured to change the GEMM's
    rounding.  ``None`` where the chain has no such proof or the config's
    block sizes do not name every launch's tile.
    """
    if surface.region_pid_type_choices(state) is None:
        return None
    blocks = config.get("block_sizes")
    if not isinstance(blocks, list):
        return None
    acc_stages = config.get("tcgen05_acc_stages", 2)
    c_stages = config.get("tcgen05_c_stages", 2)
    if type(acc_stages) is not int or type(c_stages) is not int:
        return None
    c_counts = config.get("tcgen05_region_c_stages")
    cap = _flat_region_ab_stage_cap(state)
    limits = []
    for region, (axes, pipeline_facts) in enumerate(
        zip(
            state.materialized_matmul_block_ids,
            state.materialized_pipeline_facts,
            strict=True,
        )
    ):
        assert pipeline_facts is not None
        indices = [facts._config_block_index(state, axis) for axis in axes]
        if any(index is None or index >= len(blocks) for index in indices):
            return None
        bm, bn, bk = (blocks[cast("int", index)] for index in indices)
        if not all(type(value) is int and value > 0 for value in (bm, bn, bk)):
            return None
        c_count = c_stages
        if isinstance(c_counts, list) and region < len(c_counts):
            entry = c_counts[region]
            if type(entry) is not int:
                return None
            c_count = entry or c_stages
        ab_one = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=pipeline_facts.input_dtype_bytes,
            ab_stages=1,
            cluster_m=1,
        )
        budget = (
            pipeline_facts.capacity_bytes - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        limit = min(cap, max(budget, 0) // ab_one) if ab_one > 0 else 0
        if limit > 0 and one_cta_pipeline_tile_modeled(
            bm=bm, bn=bn, bk=bk, c_stages=c_count, acc_stages=acc_stages
        ):
            # The launch's epilogue rings (``region_launch_config``: the
            # staged aux rings of ``epi_tma``, the C stages of every
            # epilogue warp group, the configured subtile) share the
            # arena with the AB ring.
            launch = facts.region_launch_config(state, config, region)
            limit = max_pipeline_ab_stages(
                pipeline_facts,
                bm=bm,
                bn=bn,
                bk=bk,
                c_stages=c_count,
                acc_stages=acc_stages,
                cluster_m=1,
                max_stages=limit,
                aux_load_mode=aux_load_mode_knob(launch),
                aux_stages=aux_stages_knob(launch),
                epi_warp_groups=epi_warp_groups_knob(launch),
                epi_subtile_n=epi_subtile_n_knob(launch),
            )
        limits.append(limit)
    return limits


def _materialized_paired_stage_limits(
    state: CuteTcgen05Config, config: dict[str, object]
) -> list[int] | None:
    """Deepest AB ring per materialized GEMM launch, or ``None``.

    The persistent GEMM launches of a fission plan or ordered barrier
    chain run the static persistent CtaGroup.TWO schedule
    (``tcgen05_cta_group="two"``, ``tcgen05_cluster_m=2``) with whole
    128/256-row tiles; each such region's budget is its complete
    single-output allocation proof (``materialized_pipeline_facts``),
    or, for a region whose two outputs the shared fanout merges, the
    typed pair proof (``materialized_pair_budget_facts``).  A chain's
    GEMM launch left flat (``inherit``) keeps the flat one-CTA pipeline
    and its validated ring depth.  At least one launch is persistent;
    anything less declines the whole recipe.
    """
    if not state.materialized_matmul_block_ids:
        return None
    pid_types = surface._region_pid_types(state, config)
    if (
        config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) != "two"
        or config.get("tcgen05_cluster_m", 1) != 2
        or config.get("tcgen05_cluster_n", 1) != 1
        or not all(
            pid_type in ("flat", "persistent_blocked", "persistent_interleaved")
            for pid_type in pid_types
        )
        or all(pid_type == "flat" for pid_type in pid_types)
        or not facts._region_two_cta_schedule_supported(state, config)
    ):
        return None
    # A chain's launches left flat keep the one-CTA pipeline at the depth
    # their tile fits; a fission plan has no flat launch to price.
    flat_limits: list[int] | None = None
    if "flat" in pid_types:
        flat_limits = _flat_region_ab_stage_limits(state, config)
        if flat_limits is None:
            return None
    fanout = config.get(FANOUT_CONFIG_KEY, "off")
    if fanout == "shared":
        if not (
            state.epilogue_fanout_plans
            and fanout_schedule_supported(config)
            and repair.epilogue_fanout_config_supported(state, config)
        ):
            return None
    elif fanout != "off":
        return None
    blocks = config.get("block_sizes")
    if not isinstance(blocks, list):
        return None
    c_counts = config.get(
        "tcgen05_region_c_stages", [0] * len(state.materialized_matmul_block_ids)
    )
    assert isinstance(c_counts, list)
    limits = []
    for region, (axes, shape, pipeline_facts, pair_facts) in enumerate(
        zip(
            state.materialized_matmul_block_ids,
            state.materialized_matmul_shapes,
            state.materialized_pipeline_facts,
            state.materialized_pair_budget_facts,
            strict=True,
        )
    ):
        if pid_types[region] == "flat":
            assert flat_limits is not None
            limits.append(flat_limits[region])
            continue
        indices = [facts._config_block_index(state, axis) for axis in axes]
        if any(index is None or index >= len(blocks) for index in indices):
            return None
        bm, bn, bk = (blocks[cast("int", index)] for index in indices)
        if (
            bm not in (128, 256)
            or bn not in (64, 128, 256)
            or bk not in (64, 128, 256)
            or any(
                extent % tile for extent, tile in zip(shape, (bm, bn, bk), strict=True)
            )
        ):
            return None
        c_count = c_counts[region] or config.get("tcgen05_c_stages", 2)
        if c_count not in (2, 4):
            return None
        pairs = [
            pair for pair in state.epilogue_fanout_plans if pair.block_ids == axes[:2]
        ]
        if pairs and fanout == "shared":
            if (
                len(pairs) != 1
                or pairs[0].output_dtype.itemsize != 2
                or c_count != 2
                or pair_facts is None
            ):
                return None
            pipeline_facts = pair_facts
        elif pipeline_facts is None:
            return None
        assert pipeline_facts is not None
        # The launch's epilogue rings (``region_launch_config``: the
        # staged aux rings of ``epi_tma`` on a launch with an exact-shape
        # auxiliary operand, the C stages of every epilogue warp group,
        # the configured subtile) share the arena with the AB ring.
        launch = facts.region_launch_config(state, config, region)
        epilogue_rings: dict[str, Any] = {
            "aux_load_mode": aux_load_mode_knob(launch),
            "aux_stages": aux_stages_knob(launch),
            "epi_warp_groups": epi_warp_groups_knob(launch),
            "epi_subtile_n": epi_subtile_n_knob(launch),
        }
        # Equal-size full output rings have 1024B-multiple sizes, so the
        # second ring adds no alignment gap beyond the existing reservation.
        base = pipeline_smem_bytes(
            pipeline_facts,
            bm=bm,
            bn=bn,
            bk=bk,
            ab_stages=1,
            c_stages=c_count,
            acc_stages=2,
            **epilogue_rings,
        )
        ab_one = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=pipeline_facts.input_dtype_bytes,
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
                        pipeline_facts,
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        ab_stages=count,
                        c_stages=c_count,
                        acc_stages=2,
                        **epilogue_rings,
                    )
                    + extra
                    <= pipeline_facts.capacity_bytes
                ),
                default=0,
            )
        )
    return limits if limits and all(limits) else None


def ab_stages_three_fits(
    state: CuteTcgen05Config,
    *,
    bm: int,
    bn: int,
    bk: int,
    cluster_m: int,
    ab_stages: int = 3,
) -> bool:
    return _ab_stages_fit_constraints(
        constraints=state.ab_stages_three_search_constraints,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab_stages,
    )


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
    state: CuteTcgen05Config,
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
    return default_layout_smem_fits(
        state,
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
    state: CuteTcgen05Config,
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
    aux_load_mode: str = TCGEN05_AUX_LOAD_MODE_SIMT,
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_subtile_n: int | None = None,
    epi_warp_groups: int = 1,
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
    store (one output, no auxiliary operand) on the 256-wide two-CTA
    tile, otherwise CuTe's with-source rule
    at the output width, which the role-local epilogue applies whether
    or not a source-C tensor exists ((128, 64) on a 256-wide 16-bit
    tile, (128, 32) on narrower ones). ``None`` without a
    recorded SMEM budget, and without store facts when rows are staged;
    rings without facts are sized for the widest (32-bit) output the
    epilogue stores (the with-source subtile rule makes that ring at
    least as large as any narrower output's), so an unanalyzed store
    chain is judged conservatively rather than refused. A staged aux load
    mode (``TCGEN05_AUX_LOAD_MODES_STAGED``) adds one ``aux_stages``-deep
    ring of the with-source epilogue subtile per exact-shape auxiliary
    operand (``facts.exact_aux_count``). ``epi_subtile_n`` pins the C
    subtile to ``(per-CTA M, epi_subtile_n)`` (``tcgen05_epi_subtile_n``,
    the explicit-tile path) and ``epi_warp_groups`` multiplies the C
    stages (each epilogue warp group owns ``c_stages`` stages of the
    ring, ``tcgen05_epi_warp_groups``).
    """
    constraints = state.ab_stages_three_search_constraints
    facts = state.rowvec_aux_facts
    if constraints is None or cluster_m not in (1, 2):
        return None
    if facts is None and stage_rows:
        return None
    if bm <= 0 or bn <= 0 or bk <= 0 or ab_stages <= 0 or c_stages <= 0:
        return None
    if acc_stages <= 0 or epi_warps <= 0 or epi_warp_groups <= 0:
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
        stage_bytes = 0
    else:
        output_itemsize = facts.output_itemsize
        stage_bytes = (
            tcgen05_rowvec_stage_smem_bytes(rows=facts.rows, bn=bn, epi_warps=epi_warps)
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
    epi_tile_m, epi_tile_n = epilogue_subtile(
        state,
        bm=bm,
        bn=bn,
        cluster_m=cluster_m,
        stage_rows=stage_rows,
        epi_subtile_n=epi_subtile_n,
    )
    c_bytes = tcgen05_c_smem_bytes_per_cta(
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        dtype_bytes=output_itemsize,
        c_stages=c_stages * epi_warp_groups,
    )
    # One C ring per output store: two fresh outputs of one accumulator
    # (``se_block_fwd``'s ``s`` and ``out``) allocate two rings, with or
    # without the shared fanout epilogue.  The row-vector facts count the
    # analyzed stores; without them the constraints' store count (the
    # traced stores, all ``default_layout_smem_fits`` has) prices the
    # rings.
    c_bytes *= max(
        1,
        facts.output_count if facts is not None else constraints.output_stores,
    )
    # The staged aux rings take the plan's with-source subtile whatever
    # the C ring's tile is (``cute_mma``: ``tcgen05_epi_tile``).
    aux_ring_bytes = 0
    if facts is not None and aux_load_mode in TCGEN05_AUX_LOAD_MODES_STAGED:
        ring_tile_m, ring_tile_n = tcgen05_default_epilogue_tile_size(
            bm, bn, elem_width_d=out_bits, elem_width_c=out_bits
        )
        aux_ring_bytes = tcgen05_aux_ring_smem_bytes_per_cta(
            epi_tile_m=ring_tile_m,
            epi_tile_n=ring_tile_n,
            dtype_bytes=output_itemsize,
            aux_stages=aux_stages,
            ring_count=facts.exact_aux_count,
        )
    return (
        ab_bytes
        + c_bytes
        + stage_bytes
        + aux_ring_bytes
        + tcgen05_fixed_smem_overhead_bytes(
            ab_stages=ab_stages, acc_stages=acc_stages, cluster_m=cluster_m
        )
    )


def default_layout_smem_fits(
    state: CuteTcgen05Config,
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
    aux_load_mode: str = TCGEN05_AUX_LOAD_MODE_SIMT,
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_subtile_n: int | None = None,
    epi_warp_groups: int = 1,
) -> bool:
    """Whether a DEFAULT-layout tile's SMEM plan fits the opt-in capacity.

    ``default_layout_smem_bytes`` plus the allowance for the small
    allocations the model does not enumerate
    (``TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES``) against the raw
    per-CTA capacity, which the arena may fill exactly. Fails closed:
    ``False`` without a recorded SMEM budget, or without store facts
    when rows are staged.
    """
    arena_bytes = default_layout_smem_bytes(
        state,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab_stages,
        c_stages=c_stages,
        stage_rows=stage_rows,
        epi_warps=epi_warps,
        acc_stages=acc_stages,
        aux_load_mode=aux_load_mode,
        aux_stages=aux_stages,
        epi_subtile_n=epi_subtile_n,
        epi_warp_groups=epi_warp_groups,
    )
    if arena_bytes is None:
        return False
    constraints = state.ab_stages_three_search_constraints
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
    state: CuteTcgen05Config,
    *,
    bm: int,
    bn: int,
    bk: int,
    cluster_m: int,
    ab_stages: int,
    c_stages: int,
    has_source_c: bool,
) -> bool:
    # Invariant: a deeper C-store ring (``tcgen05_c_stages=4``) is admitted
    # only when AB + C fit the ``tcgen05_ab_stages=3`` SMEM envelope (same
    # dtype bytes and per-CTA budget after the non-AB reservation); otherwise
    # ``_fix_c_stages_search_config`` demotes it to 2, because AB+C beyond
    # the budget reaches ptxas as a raw ``too much shared`` error.  The C
    # bytes use the ``Tcgen05LayoutStrategy.DEFAULT`` epilogue subtile the
    # role-local codegen allocates, not the (128, 32) EXPLICIT_EPI_TILE
    # direct-entry tile: a 256x256 16-bit tile is ``(128, 64)`` with a
    # source-C (residual family) but ``(128, 32)`` without one (plain
    # matmul), since ``compute_epilogue_tile_size`` shrinks N when no C tile
    # competes for SMEM; ``has_source_c`` carries that distinction.
    constraints = state.ab_stages_three_search_constraints
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
    # bytes do not depend on cluster_m. ``elem_width`` is the OUTPUT
    # element width in bits (the operand width where the output dtype is
    # not recorded; the validated 16-bit families are uniform, an fp8 GEMM
    # stores 16-bit outputs). ``elem_width_c`` is None for no-source-C
    # (plain) kernels so the helper picks the smaller no-source-C epilogue
    # tile. Every output store owns a ring of its own (two-output
    # epilogues), so the ring bytes are charged per store.
    output_bytes = constraints.output_dtype_bytes or constraints.dtype_bytes
    elem_width = output_bytes * 8
    epi_tile_m, epi_tile_n = tcgen05_default_epilogue_tile_size(
        bm,
        bn,
        elem_width_d=elem_width,
        elem_width_c=elem_width if has_source_c else None,
    )
    c_bytes = tcgen05_c_smem_bytes_per_cta(
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        dtype_bytes=output_bytes,
        c_stages=c_stages,
    )
    c_bytes *= max(1, constraints.output_stores)
    return ab_bytes + c_bytes <= constraints.per_cta_smem_budget_bytes


def output_edge_tma_store_smem_fits(
    state: CuteTcgen05Config,
    config: dict[str, object],
    *,
    bm: int,
    bn: int,
    bk: int,
    cluster_m: int,
    ab_stages: int,
    c_stages: int,
) -> bool:
    """Whether an output-edge kernel's TMA-store C ring fits beside its AB ring.

    ``c_stages_fits`` charges the C ring against the AB-gate budget, which
    already holds back ``TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES`` for
    the C ring and barriers, so it counts the ring twice and refuses
    arenas that fit: 1000x4096x2880 bf16 at 2x1 256x256x64 ab=6 (192 KiB
    AB + 32 KiB (128, 64) C ring + 172 B of barriers = 229 548 B of the
    232 448 B opt-in) fell back to the SIMT epilogue and lost to ab=5,
    while the hand-kept TMA store at ab=6 compiled and ran 2.8% faster.
    This judges the measured arena model (``default_layout_smem_fits``)
    at the C subtile the output-edge plan renders: edge tiles never take
    the plain (128, 32) narrow subtile (it needs static full tiles), so
    the ring is charged at CuTe's with-source subtile, or at a configured
    explicit tile when that is wider. Fails closed like the model.
    """
    facts = state.rowvec_aux_facts
    if facts is None or cluster_m not in (1, 2):
        return False
    if _store_lowering_stages_rows(config) and facts.rows:
        # This route does not admit ``pre_acc_wait`` rows (the hybrid
        # store's edge tiles read them from global memory beside the full
        # tiles' staged row); the two-stage route,
        # ``TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES``, and
        # ``c_stages_fits`` still take the hybrid store for them.
        return False
    out_bits = facts.output_itemsize * 8
    _subtile_m, subtile_n = tcgen05_default_epilogue_tile_size(
        bm, bn, elem_width_d=out_bits, elem_width_c=out_bits
    )
    for configured in (
        epi_subtile_n_knob(config),
        layout_overrides_from_config(config).epi_tile_n,
    ):
        if configured is not None:
            subtile_n = max(subtile_n, configured)
    return _config_arena_fits(
        state,
        config,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab_stages,
        c_stages=c_stages,
        epi_subtile_n=subtile_n,
    )


def _config_arena_fits(
    state: CuteTcgen05Config,
    config: dict[str, object],
    *,
    bm: int,
    bn: int,
    bk: int,
    cluster_m: int,
    ab_stages: int,
    c_stages: int,
    epi_subtile_n: int | None,
) -> bool:
    """``default_layout_smem_fits`` with every other input read from ``config``."""
    return default_layout_smem_fits(
        state,
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab_stages,
        c_stages=c_stages,
        stage_rows=_store_lowering_stages_rows(config),
        epi_warps=epi_warps_knob(config),
        acc_stages=acc_stages_knob(config),
        aux_load_mode=aux_load_mode_knob(config),
        aux_stages=aux_stages_knob(config),
        epi_subtile_n=epi_subtile_n,
        epi_warp_groups=epi_warp_groups_knob(config),
    )


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
        and config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) in TCGEN05_GROUPED_DYNAMIC_MODES
        and isinstance(block_sizes, list)
        and block_sizes[:3] == [128, 64, 64]
        and config.get("pid_type") == TCGEN05_TWO_CTA_SEED_PID_TYPE
        and all(
            config.get(key, expected) == expected for key, expected in defaults.items()
        )
        and all(config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
    )


def _grouped_worklist_immutable_smem(
    state: CuteTcgen05Config, config: dict[str, object]
) -> bool:
    """Discount descriptor allocations only for a discovered pure D-shift.

    Codegen repeats the ownership, input-domain and bounded-output proofs
    before checking the resolved allocation against the physical capacity.
    """
    facts = state.grouped_worklist_smem_facts
    return (
        state.grouped_d_shift_supported
        and facts is not None
        and facts.device_split_sizes
        and config.get(TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY) is True
        and config.get(TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY) is True
    )


def _grouped_worklist_nm_ab_config_matches(
    state: CuteTcgen05Config, config: dict[str, object], ab_stages: object
) -> bool:
    return _grouped_worklist_nm_ab_config_mismatch(state, config, ab_stages) is None


def _grouped_worklist_nm_ab_config_mismatch(
    state: CuteTcgen05Config, config: dict[str, object], ab_stages: object
) -> str | None:
    """Why ``config`` at ``ab_stages`` is not an admitted N,M worklist ring.

    ``None`` when it is: the worklist shape (a 256x128 logical tile with a
    resolvable MMA profile, ``tcgen05_cluster_n=1``, two accumulator and
    C stages, four to seven AB stages) whose exact per-CTA footprint
    (``tcgen05_grouped_worklist_smem_bytes``) fits the target capacity.
    The shape and the footprint are reported apart, so a rejected explicit
    config learns which one it missed.
    """
    config_view = facts._matmul_config_view(state, config)
    if config_view is None:
        block_sizes = config.get("block_sizes")
        if (
            state.matmul_block_ids is not None
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
    logical_k = block_sizes[k_index]
    if type(logical_k) is not int:
        return TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
    block_k = physical_grouped_ab_k(config, logical_k)
    profile = resolve_tcgen05_grouped_worklist_mma_profile(
        config,
        block_k=block_k,
    )
    if not (
        type(ab_stages) is int
        and 4
        <= ab_stages
        <= (
            32
            if config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
            else 7
        )
        and profile is not None
        and config.get("tcgen05_cluster_n", 1)
        in (
            (1, 2)
            if config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
            else (1,)
        )
        and config.get("tcgen05_acc_stages", 2) == 2
        and config.get("tcgen05_c_stages", 2)
        in (
            (2, 3, 4)
            if config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn"
            else (2,)
        )
        and block_sizes[m_index] == TCGEN05_TWO_CTA_BLOCK_M
        and block_sizes[n_index] == 128
    ):
        return TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
    constraints = state.ab_stages_three_search_constraints
    if constraints is None:
        # Fixed configs can be normalized before their input device is known.
        # CuTe MMA selection applies the real target's SMEM limit at codegen.
        return None
    target_capacity_bytes = (
        constraints.per_cta_smem_budget_bytes
        + TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
    )
    smem_facts = state.grouped_worklist_smem_facts
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
    overrides = layout_overrides_from_config(config)
    epi_tile_shape = (
        (overrides.epi_tile_m, overrides.epi_tile_n)
        if overrides.epi_tile_m is not None and overrides.epi_tile_n is not None
        else None
    )
    immutable_tensormaps = _grouped_worklist_immutable_smem(state, config)
    required_bytes = tcgen05_grouped_worklist_smem_bytes(
        group_count=smem_facts.group_count,
        device_split_sizes=smem_facts.device_split_sizes,
        sched_stage_count=sched_stage_count,
        bm=physical_bm,
        bn=physical_bn,
        bk=cast("int", block_k),
        dtype_bytes=constraints.dtype_bytes,
        output_dtype_bytes=smem_facts.output_dtype_bytes,
        ab_stages=ab_stages,
        acc_stages=2,
        c_stages=cast("int", config.get("tcgen05_c_stages", 2)),
        cluster_m=profile.cluster_m,
        orientation=str(config.get("tcgen05_grouped_worklist_orientation", "nm")),
        epi_tile_shape=epi_tile_shape,
        immutable_ab_tensormaps=immutable_tensormaps,
        immutable_d_tensormap=immutable_tensormaps,
    )
    if required_bytes <= target_capacity_bytes:
        return None
    return TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH


def grouped_dynamic_stages_fit_for_target(
    state: CuteTcgen05Config,
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
    cap_bytes = target_facts.per_cta_smem_capacity_bytes(device)
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
    return ab_bytes + c_bytes + TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES <= cap_bytes


def epilogue_subtile(
    state: CuteTcgen05Config,
    *,
    bm: int,
    bn: int,
    cluster_m: int,
    stage_rows: bool,
    epi_subtile_n: int | None = None,
) -> tuple[int, int]:
    """The C subtile ``(m, n)`` the role-local TMA-store epilogue renders.

    The matmul plan's choice (``cute_mma``): a configured subtile
    (``tcgen05_epi_subtile_n``, the explicit-tile path: ``(per-CTA M,
    n)``) wins; then the plan's (128, 32) for promoted 16-bit rows that
    are staged, or for a plain 16-bit store on the 256-wide two-CTA tile
    (``tcgen05_plain_narrow_subtile``: one output store and no auxiliary
    operand -- a two-output or aux epilogue keeps CuTe's with-source
    (128, 64) subtile there, 64 KiB of C rings for two outputs; the plan
    also asks for a single static-full-tile store, which every plain
    kernel the gates see has); otherwise CuTe's with-source rule at the
    output width, sized for a 32-bit output without store facts.
    """
    constraints = state.ab_stages_three_search_constraints
    rowvec_facts = state.rowvec_aux_facts
    if epi_subtile_n is not None:
        return (
            tcgen05_epilogue_subtile_m(bm=bm, is_two_cta=cluster_m == 2),
            epi_subtile_n,
        )
    out_bits = 32 if rowvec_facts is None else rowvec_facts.output_itemsize * 8
    narrow_subtile = (
        rowvec_facts is not None
        and constraints is not None
        and rowvec_facts.output_itemsize == 2
        and (
            stage_rows
            and bool(rowvec_facts.rows)
            and all(row.promoted for row in rowvec_facts.rows)
        )
        and tcgen05_explicit_epilogue_tile_supported(
            is_two_cta=cluster_m == 2,
            bm=bm,
            bn=bn,
            tile_shape=(128, 32, 32),
        )
    ) or facts._plain_narrow_subtile_facts(state, bm=bm, bn=bn, cluster_m=cluster_m)
    if narrow_subtile:
        return 128, 32
    return tcgen05_default_epilogue_tile_size(
        bm, bn, elem_width_d=out_bits, elem_width_c=out_bits
    )


def epilogue_subtile_count(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> int | None:
    """Epilogue subtiles of one CTA tile (the subtile loop's trip count).

    ``None`` where the per-CTA rows do not partition into the plan's
    subtile (the bm=128 CtaGroup.TWO family's permuted per-CTA tile, or a
    config without a block view).  ``region`` names the GEMM launch of a
    materialized chain.
    """
    view = facts._matmul_config_view(state, config, region=region)
    if view is None:
        return None
    blocks, mi, ni, _ki = view
    bm, bn = blocks[mi], blocks[ni]
    cluster_m = config.get("tcgen05_cluster_m", 1)
    if not (
        type(bm) is int
        and type(bn) is int
        and type(cluster_m) is int
        and bm > 0
        and bn > 0
        and cluster_m in (1, 2)
    ):
        return None
    epi_tile_m, epi_tile_n = epilogue_subtile(
        state,
        bm=bm,
        bn=bn,
        cluster_m=cluster_m,
        stage_rows=_store_lowering_stages_rows(config),
        epi_subtile_n=epi_subtile_n_knob(config),
    )
    rows = (
        bm // 2
        if facts._config_is_two_cta_mma(state, config, bm=bm, cluster_m=cluster_m)
        else bm
    )
    if rows % epi_tile_m or bn % epi_tile_n:
        return None
    return (rows // epi_tile_m) * (bn // epi_tile_n)


def aux_prefetch_depth_limit(
    state: CuteTcgen05Config, config: dict[str, object], *, region: int | None = None
) -> int | None:
    """Deepest auxiliary prefetch ring the register budget admits.

    ``(d + 1)`` subtile fragments of the config's epilogue subtile per
    thread within ``TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD``,
    at most one less than the subtiles each epilogue warp group walks
    (a ring that reaches the group's last subtile is the whole-fragment
    hoist) and the knob's ceiling. ``None`` without store facts or a
    known subtile count.  On a materialized chain: the shallowest limit
    over the GEMM launches with an exact-shape auxiliary operand.
    """
    if region is None and state.materialized_matmul_block_ids:
        limits = [
            aux_prefetch_depth_limit(state, launch, region=launch_region)
            for launch_region, launch in facts.chain_launches(
                state, config, aux_only=True
            )
        ]
        if not limits or any(limit is None for limit in limits):
            return None
        return min(cast("list[int]", limits))
    rowvec_facts = state.rowvec_aux_facts
    count = epilogue_subtile_count(state, config, region=region)
    view = facts._matmul_config_view(state, config, region=region)
    if rowvec_facts is None or count is None or view is None:
        return None
    blocks, mi, ni, _ki = view
    bm, bn = cast("int", blocks[mi]), cast("int", blocks[ni])
    cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
    epi_tile_m, epi_tile_n = epilogue_subtile(
        state,
        bm=bm,
        bn=bn,
        cluster_m=cluster_m,
        stage_rows=_store_lowering_stages_rows(config),
        epi_subtile_n=epi_subtile_n_knob(config),
    )
    per_subtile = tcgen05_aux_subtile_fragment_bytes_per_thread(
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        itemsize=rowvec_facts.output_itemsize,
    )
    groups = epi_warp_groups_knob(config)
    local_count = -(-count // groups)
    return max(
        0,
        min(
            TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD // per_subtile - 1,
            local_count - 1,
            TCGEN05_AUX_PREFETCH_MAX_SUBTILES,
        ),
    )


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
    state: CuteTcgen05Config,
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
        >>> smem.max_ab_stages_that_fit(state, bm=256, bn=256, bk=64, cluster_m=2)
        8  # FP8 fits 8 stages

        >>> # BF16 same tile (2x larger per stage)
        >>> smem.max_ab_stages_that_fit(state, bm=256, bn=256, bk=64, cluster_m=2)
        4  # BF16 fits only 4 stages
    """
    constraints = state.ab_stages_three_search_constraints
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
        hard_cap = _get_dtype_ab_stages_hard_cap(constraints.dtype_bytes)

    # Return clamped value: at least 1, at most hard_cap or budget limit
    return max(1, min(max_from_budget, hard_cap))


def epi_tma_ab_stage_limit(
    state: CuteTcgen05Config, config: dict[str, object]
) -> int | None:
    """Deepest AB ring that fits beside the epilogue-issued aux rings.

    The paired pipeline's complete allocation proof where it applies
    (``_paired_pipeline_stage_limit``), otherwise the DEFAULT-layout arena
    model with the staged rings charged (``default_layout_smem_fits``).
    ``None`` where no SMEM budget is recorded (nothing is proved and the
    config is left alone, as for the other budget gates), and on a
    materialized chain, whose launches' rings the region budgets price
    beside the aux rings (``_materialized_paired_stage_limits``,
    ``_flat_region_ab_stage_limits``).
    """
    if state.materialized_matmul_block_ids:
        return None
    limit = _paired_pipeline_stage_limit(state, config)
    if limit is not None:
        return limit
    view = facts._matmul_config_view(state, config)
    if view is None or state.ab_stages_three_search_constraints is None:
        return None
    blocks, mi, ni, ki = view
    bm, bn, bk = blocks[mi], blocks[ni], blocks[ki]
    cluster_m = config.get("tcgen05_cluster_m", 1)
    ab_stages = config.get("tcgen05_ab_stages", 2)
    c_stages = config.get("tcgen05_c_stages", 2)
    if not all(
        type(value) is int and value > 0
        for value in (bm, bn, bk, cluster_m, ab_stages, c_stages)
    ) or cluster_m not in (1, 2):
        return None
    if (
        default_layout_smem_bytes(
            state,
            bm=cast("int", bm),
            bn=cast("int", bn),
            bk=cast("int", bk),
            cluster_m=cast("int", cluster_m),
            ab_stages=1,
            c_stages=cast("int", c_stages),
            stage_rows=_store_lowering_stages_rows(config),
        )
        is None
    ):
        return None
    for stages in range(cast("int", ab_stages), 0, -1):
        if default_layout_smem_fits(
            state,
            bm=cast("int", bm),
            bn=cast("int", bn),
            bk=cast("int", bk),
            cluster_m=cast("int", cluster_m),
            ab_stages=stages,
            c_stages=cast("int", c_stages),
            stage_rows=_store_lowering_stages_rows(config),
            epi_warps=epi_warps_knob(config),
            acc_stages=acc_stages_knob(config),
            aux_load_mode=TCGEN05_AUX_LOAD_MODE_EPI_TMA,
            aux_stages=aux_stages_knob(config),
            epi_subtile_n=epi_subtile_n_knob(config),
            epi_warp_groups=epi_warp_groups_knob(config),
        ):
            return stages
    return 0
