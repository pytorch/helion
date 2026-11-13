"""Seed configs of the tcgen05 GEMM families.

Module functions over the ``CuteTcgen05Config`` state object.
``autotune_seed_configs`` is the ordered list the autotuner receives after the
compiler's heuristic seeds; each family contributes its own ``*_seed_config``
builders (paired pipeline, one-CTA pipeline, small grid, split-K, warp MMA,
one-wave / multi-tile two-CTA, full-tile direct entry, CLC / aux-TMA, dense
aligned N).  Seeds never reach codegen directly: they are repaired and
normalized like any other config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ....runtime.config import Config
from ..cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
from ..cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY
from ..cute_warp_mma_gemm import WARP_MMA_SEED_TILES
from ..cute_warp_mma_gemm import WARP_MMA_WARPS_KEY
from ..cute_warp_mma_gemm import warp_mma_k_step
from ..cute_warp_mma_gemm import warp_mma_tile_reason
from ..dense_aligned_n import ALIGNED_N_CHOICES
from ..epilogue_fanout import FANOUT_CONFIG_KEY
from ..one_cta_pipeline import one_cta_pipeline_supported
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..pipeline_smem import max_pipeline_ab_stages
from ..strategies import TCGEN05_BATCH_RASTER_CONFIG_KEY
from ..strategies import TCGEN05_BATCH_RASTER_SLOWEST
from ..strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from ..strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ..strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENT_GRID_CAP_RESIDENT_CLUSTERS
from ..strategies import TCGEN05_STRATEGY_CONFIG_KEY
from ..strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from ..strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
from ..strategies import Tcgen05LayoutStrategy
from ..strategies import Tcgen05PersistenceModel
from ..strategies import Tcgen05Strategy
from ..tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ..tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from ..tcgen05_constants import TCGEN05_CLUSTER_N2_SEED_L2_GROUPING
from ..tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPE_INHERIT
from ..tcgen05_constants import TCGEN05_REGION_PID_TYPES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SMALL_GRID_DIRECT_STORE_MAX_BN
from ..tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_DEEP_AB_STAGES
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN
from ..tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER,
)
from ..tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE,
)
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
from ..tcgen05_constants import TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
from ..tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..tcgen05_constants import tcgen05_direct_entry_stage_tuple_allowed
from ..tcgen05_constants import tcgen05_rowvec_stage_smem_bytes
from ..tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides
from ..tcgen05_schedule_facts import n_edge_cluster_schedule_defaults
from . import facts
from . import repair
from . import smem
from . import surface

if TYPE_CHECKING:
    from ....autotuner.config_fragment import BlockSizeFragment


from .knobs import _tile_in_fragments

if TYPE_CHECKING:
    from . import CuteTcgen05Config


def full_tile_schedule_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    schedule_facts = state.full_tile_schedule_facts
    if schedule_facts is None:
        return []
    seeds: list[Config] = []
    for cluster_m in (1, 2):
        if (cluster_m == 1 and not state.full_tile_restored_pid_types) or (
            cluster_m == 2 and not state.full_tile_added_pairing
        ):
            continue
        tiles = facts._full_tile_schedule_tiles(
            state, schedule_facts, cluster_m=cluster_m
        )
        if cluster_m == 1 and tiles:
            # Existing small-grid seeds cover narrow one-wave tiles. Supply
            # the widest exact tile for multi-wave problems as well.
            widest = max(bm * bn for bm, bn, _bk in tiles)
            tiles = [tile for tile in tiles if tile[0] * tile[1] == widest]
        for bm, bn, bk in tiles:
            if bk not in (64, 128):
                continue
            maximum = smem.max_ab_stages_that_fit(
                state, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
            )
            maximum = min(maximum, (schedule_facts.shape[2] + bk - 1) // bk)
            for stages in dict.fromkeys((min(2, maximum), maximum)):
                if stages <= 0:
                    continue
                values: dict[str, Any] = {
                    "block_sizes": facts._matmul_seed_block_sizes(
                        state, bm=bm, bn=bn, bk=bk
                    ),
                    "pid_type": "persistent_interleaved",
                    "tcgen05_cluster_m": cluster_m,
                    "tcgen05_cluster_n": 1,
                    "tcgen05_ab_stages": stages,
                    "tcgen05_acc_stages": 2,
                    "tcgen05_c_stages": 2,
                }
                if state.aux_kernel_detected and smem.rowvec_aux_stage_fits(
                    state,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    cluster_m=cluster_m,
                    ab_stages=stages,
                    c_stages=2,
                ):
                    values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                        TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                    )
                seeds.append(Config(**values))
    return seeds


def n_edge_cluster_schedule_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    schedule_facts = state.n_edge_cluster_schedule_facts
    if schedule_facts is None:
        return []
    return [
        Config.from_dict(
            {
                **n_edge_cluster_schedule_defaults(),
                "block_sizes": facts._matmul_seed_block_sizes(
                    state, bm=bm, bn=bn, bk=bk
                ),
                "pid_type": "persistent_interleaved",
                "tcgen05_cluster_m": 2,
                "tcgen05_cluster_n": 2,
                "tcgen05_ab_stages": 2,
                "tcgen05_acc_stages": 2,
                "tcgen05_c_stages": 2,
            }
        )
        for bm, bn, bk in facts._n_edge_cluster_schedule_tiles(state, schedule_facts)
        if smem.c_stages_fits(
            state,
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=2,
            ab_stages=2,
            c_stages=2,
            has_source_c=True,
        )
    ]


def _one_cta_pipeline_parameters(
    state: CuteTcgen05Config, config: dict[str, object]
) -> tuple[int, int, int, int, int] | None:
    pipeline_facts = state.one_cta_pipeline_facts
    view = facts._matmul_config_view(state, config)
    if (
        pipeline_facts is None
        or view is None
        or config.get("pid_type") not in state.allowed_pid_types
        or state.materialized_matmul_block_ids
    ):
        return None
    blocks, mi, ni, ki = view
    tile = cast("tuple[int, int, int]", (blocks[mi], blocks[ni], blocks[ki]))
    if not one_cta_pipeline_supported(pipeline_facts, tile, config):
        return None
    return (
        *tile,
        cast("int", config.get("tcgen05_c_stages", 2)),
        cast("int", config.get("tcgen05_acc_stages", 2)),
    )


def one_cta_pipeline_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    pipeline_facts = state.one_cta_pipeline_facts
    fragments = facts._matmul_block_fragments(state)
    pid = facts._preferred_persistent_pid_type(state)
    if pipeline_facts is None or fragments is None or pid is None:
        return []
    seeds: list[Config] = []
    for bm in (64, 128):
        for bn in (16, 32, 64, 128, 256):
            for bk in (32, 64, 128, 256):
                if not _tile_in_fragments(fragments, (bm, bn, bk)):
                    continue
                blocks = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
                assert blocks is not None
                config = Config(
                    block_sizes=blocks,
                    pid_type=pid,
                    tcgen05_cluster_m=1,
                    tcgen05_cluster_n=1,
                    tcgen05_ab_stages=2,
                    tcgen05_acc_stages=2,
                    tcgen05_c_stages=2,
                )
                if pipeline_facts.allocations.has_auxiliary_loads:
                    config.config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                        "post_acc_wait"
                    )
                maximum = smem._one_cta_pipeline_stage_limit(state, config.config)
                if maximum is None:
                    continue
                maximum = min(maximum, pipeline_facts.schedule.shape[2] // bk)
                # Existing seeds already cover shallower generic rings.
                if maximum <= smem._get_dtype_ab_stages_hard_cap(1) and bk != 256:
                    continue
                config.config["tcgen05_ab_stages"] = maximum
                seeds.append(config)
    return seeds


def _paired_pipeline_parameters(
    state: CuteTcgen05Config, config: dict[str, object]
) -> tuple[int, int, int, int, int] | None:
    """Exact allocation family supported by the smaller SMEM reservation."""
    pipeline_facts = state.pipeline_smem_facts
    view = facts._matmul_config_view(state, config)
    if (
        pipeline_facts is None
        or view is None
        or config.get(TCGEN05_CTA_GROUP_CONFIG_KEY) != "two"
        or config.get("tcgen05_cluster_m", 1) != 2
        # cluster_n=2 is the 2x2 cluster of CTA pairs (A multicast across
        # the N neighbours); the SMEM model is per CTA either way. The
        # column-major-output family keeps its priced cluster_n=1 schedule.
        or config.get("tcgen05_cluster_n", 1)
        not in ((1,) if pipeline_facts.output_column_major else (1, 2))
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
    if state.matmul_has_leading_passthrough and config.get("tcgen05_cluster_n", 1) == 2:
        # The batched cluster_n=2 kernel pairs N tiles on the scheduler's
        # swapped dim 1 and keeps the plain raster (``cute_mma`` rejects
        # the rest); mirror it so the config is invalid, not a codegen
        # failure (``_fix_batched_cluster_n2_search_config`` repairs it).
        extents = state.matmul_compile_time_static_extents
        n = extents[1] if extents is not None else None
        groupings = config.get("l2_groupings", [])
        if (
            not isinstance(n, int)
            or n % bn != 0
            or (n // bn) % 2 != 0
            or config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, 1) != 1
            or (
                isinstance(groupings, list)
                and any(grouping != 1 for grouping in groupings)
            )
        ):
            return None
    if pipeline_facts.output_column_major:
        bounds = facts._matmul_search_bounds(state)
        if bounds is None or any(
            not low <= value <= high
            for value, (low, high) in zip((bm, bn, bk), bounds, strict=True)
        ):
            return None
    if pipeline_facts.input_dtype_bytes == 1 and (
        not facts._fp8_paired_pipeline_tile_supported(state, bm, bn, bk)
        or config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY)
        or config.get(TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY, 0) != 0
        or config.get("tcgen05_num_epi_warps", 4) != 4
        or config.get("tcgen05_aux_load_mode", "simt") != "simt"
        or config.get("tcgen05_c_store_mode", "normal") not in ("normal", "direct")
        or config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY, "post_acc_wait")
        not in ("post_acc_wait", "pre_acc_wait")
        or config.get(FANOUT_CONFIG_KEY, "off") != "off"
        or config.get("cute_materialized_operand_schedule", "off") != "off"
        or config.get("cute_materialized_schedule", "off") != "off"
        or config.get("cute_split_k_workspace")
        or config.get("cute_split_k_schedule", "legacy") != "legacy"
    ):
        return None
    return cast("tuple[int, int, int, int, int]", (bm, bn, bk, c_stages, acc_stages))


def _region_two_cta_tile(
    m: int,
    n: int,
    fragments: tuple[BlockSizeFragment, BlockSizeFragment],
    *,
    pairs_per_wave: int,
    rows: int | None = None,
) -> tuple[int, int] | None:
    """The CTA-pair tile of one region from its output geometry alone.

    Whole tiles only.  A tile fills the machine when its last wave of
    CTA pairs is at least three quarters full; the widest such tile wins
    (operand reuse, fewer per-tile prologues and epilogues), the 128-row
    instruction (its 64-row per-CTA epilogue) ahead of the 256-row one
    among equals.  Where no tile fills the machine the fullest one wins:
    a narrow long-K output lands on the small tiles the standalone
    surface measured best (1024x512x4096: 128x64), wide outputs on the
    wide ones (4096x512x1024 and 4096x1024x512: 128x256).  ``rows``
    restricts the choice to one MMA M tile (the epilogue mechanisms'
    256-row pair).
    """
    best: tuple[tuple[float, int, int], tuple[int, int]] | None = None
    for bm, bn in (
        (128, 256),
        (256, 128),
        (128, 128),
        (128, 64),
        (256, 64),
        (256, 256),
    ):
        if (
            (rows is not None and bm != rows)
            or m % bm
            or n % bn
            or not fragments[0].low <= bm <= fragments[0].high
            or not fragments[1].low <= bn <= fragments[1].high
        ):
            continue
        pairs = (m // bm) * (n // bn)
        waves = -(-pairs // pairs_per_wave)
        fill = pairs / (waves * pairs_per_wave)
        score = (min(fill, 0.75), bm * bn, int(bm == 128))
        if best is None or score > best[0]:
            best = (score, (bm, bn))
    return None if best is None else best[1]


def region_two_cta_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    """Two-CTA persistent GEMM launches for an ordered barrier chain.

    Per K tile (64 and 128): every GEMM launch takes its geometry's
    CTA-pair tile (``_region_two_cta_tile``) and the deepest AB ring its
    complete budget fits, under ``persistent_blocked`` (the standalone
    paired seeds' pid type); the SIMT launches keep their defaults and
    the chain stays ``flat``.  Auxiliary epilogues load ahead of the
    accumulator wait where every store admits it.  Beside the
    all-persistent seed, one seed per GEMM launch leaves that launch
    ``inherit`` (the flat one-CTA pipeline at the shared ring depth):
    an epilogue-bound short-K GEMM measures better flat, and the
    initial population then holds that mix directly.
    """
    if surface.region_pid_type_choices(state) is None:
        return []
    spec = state.config_spec
    fragments = [
        cast("BlockSizeFragment", item._fragment(spec)) for item in spec.block_sizes
    ]
    pairs_per_wave = max(1, (state.device_sm_count or 148) // 2)
    placement = (
        TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        if state.aux_loads_on_every_store
        else TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
    )
    aux_regions = any(
        pipeline_facts is not None and pipeline_facts.has_auxiliary_loads
        for pipeline_facts in state.materialized_pipeline_facts
    )
    seeds: list[Config] = []
    for bk in (64, 128):
        blocks = [fragment.default() for fragment in fragments]
        ab_stages: list[int] = []
        for axes, shape, pipeline_facts in zip(
            state.materialized_matmul_block_ids,
            state.materialized_matmul_shapes,
            state.materialized_pipeline_facts,
            strict=True,
        ):
            assert pipeline_facts is not None
            m, n, k = shape
            indices = [spec.block_sizes.block_id_to_index(axis) for axis in axes]
            k_fragment = fragments[indices[2]]
            if k % bk or not k_fragment.low <= bk <= k_fragment.high:
                break
            tile = _region_two_cta_tile(
                m,
                n,
                (fragments[indices[0]], fragments[indices[1]]),
                pairs_per_wave=pairs_per_wave,
            )
            if tile is None:
                break
            bm, bn = tile
            depth = min(
                k // bk,
                max_pipeline_ab_stages(
                    pipeline_facts,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    c_stages=2,
                    acc_stages=2,
                    aux_load_placement=placement,
                ),
            )
            if depth <= 0:
                break
            for index, value in zip(indices, (bm, bn, bk), strict=True):
                blocks[index] = value
            ab_stages.append(depth)
        else:
            count = len(ab_stages)
            mixes: list[int | None] = [None]
            if count >= 2:
                mixes.extend(range(count))
            for flat_region in mixes:
                pid_types = ["persistent_blocked"] * count
                stages = list(ab_stages)
                if flat_region is not None:
                    pid_types[flat_region] = TCGEN05_REGION_PID_TYPE_INHERIT
                    stages[flat_region] = 0
                values: dict[str, Any] = {
                    "block_sizes": list(blocks),
                    "pid_type": "flat",
                    TCGEN05_REGION_PID_TYPES_CONFIG_KEY: pid_types,
                    TCGEN05_CTA_GROUP_CONFIG_KEY: "two",
                    "tcgen05_cluster_m": 2,
                    "tcgen05_cluster_n": 1,
                    "tcgen05_ab_stages": 2,
                    "tcgen05_region_ab_stages": stages,
                    "tcgen05_acc_stages": 2,
                    "tcgen05_c_stages": 2,
                    "tcgen05_num_epi_warps": 4,
                    TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                        Tcgen05PersistenceModel.NON_PERSISTENT.value
                    ),
                    TCGEN05_STRATEGY_CONFIG_KEY: (
                        Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
                    ),
                    TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                        Tcgen05LayoutStrategy.DEFAULT.value
                    ),
                }
                if aux_regions:
                    values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = placement
                seeds.append(Config(**values))
                if flat_region is None:
                    seeds.extend(
                        _region_two_cta_aux_mechanism_seeds(
                            state,
                            values,
                            bk=bk,
                            fragments=fragments,
                            pairs_per_wave=pairs_per_wave,
                        )
                    )
    return seeds


def _region_two_cta_aux_mechanism_seeds(
    state: CuteTcgen05Config,
    base: dict[str, Any],
    *,
    bk: int,
    fragments: list[BlockSizeFragment],
    pairs_per_wave: int,
) -> list[Config]:
    """Twins of an all-persistent chain seed that prefetch the auxiliary operand.

    The epilogue of a GEMM launch with an exact-shape auxiliary operand
    (a relu-backward mask, a residual) reads that operand per subtile
    through registers, after the accumulator wait exposes each read's
    latency: on the squeeze_and_excitation_net backward chain the mask
    read was 2.85 us of a 9.7 us launch (flushed), the GEMM alone 6.85.
    The mechanisms that hide it -- the register prefetch ring
    (``tcgen05_aux_prefetch_subtiles``), the whole-fragment hoist
    (``tcgen05_aux_fragment_hoist``) and the epilogue-issued aux ring
    (``tcgen05_aux_load_mode="epi_tma"``) -- render on the 256-row
    CtaGroup.TWO pair tile (the 128-row pair's 64-row per-CTA epilogue
    is outside their validated envelope), so each twin moves the chain's
    auxiliary launches onto their 256-row tile at the deepest AB ring
    the region budget fits beside the launch's rings and adds one
    mechanism; the other launches keep the base seed's tiles.  Twins the
    normalizer would refuse (no 256-row tile, a fragment over the
    register budget, an aux operand the TMA cannot address) are left
    out.
    """
    spec = state.config_spec
    aux_regions = [
        region
        for region in range(len(state.materialized_matmul_block_ids))
        if facts.region_has_exact_aux(state, region)
    ]
    # The mechanisms' own admission starts from the kernel-level detector
    # (``exact_shape_aux_kernel_detected``: one compatible store value
    # whose exact-shape operands match the output); a chain it declines
    # (two GEMM launches with their own aux operands) takes no twin.
    if not aux_regions or not state.exact_shape_aux_kernel_detected:
        return []
    blocks = list(cast("list[int]", base["block_sizes"]))
    tiles: dict[int, tuple[int, int]] = {}
    for region in aux_regions:
        axes = state.materialized_matmul_block_ids[region]
        m, n, _k = state.materialized_matmul_shapes[region]
        indices = [spec.block_sizes.block_id_to_index(axis) for axis in axes]
        tile = _region_two_cta_tile(
            m,
            n,
            (fragments[indices[0]], fragments[indices[1]]),
            pairs_per_wave=pairs_per_wave,
            rows=TCGEN05_TWO_CTA_BLOCK_M,
        )
        if tile is None:
            return []
        tiles[region] = tile
        blocks[indices[0]], blocks[indices[1]] = tile

    def depths(aux_load_mode: str, placement: str) -> list[int] | None:
        stages = list(cast("list[int]", base["tcgen05_region_ab_stages"]))
        for region, (bm, bn) in tiles.items():
            pipeline_facts = state.materialized_pipeline_facts[region]
            assert pipeline_facts is not None
            depth = min(
                state.materialized_matmul_shapes[region][2] // bk,
                max_pipeline_ab_stages(
                    pipeline_facts,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    c_stages=2,
                    acc_stages=2,
                    aux_load_placement=placement,
                    aux_load_mode=aux_load_mode,
                ),
            )
            if depth <= 0:
                return None
            stages[region] = depth
        return stages

    seeds: list[Config] = []
    pre_acc_wait = TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
    placement = cast(
        "str",
        base.get(
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT,
        ),
    )
    # The register ring and the hoist read ahead of the accumulator wait.
    simt_stages = depths(TCGEN05_AUX_LOAD_MODE_SIMT, pre_acc_wait)
    if simt_stages is not None and state.aux_loads_on_every_store:
        twin: dict[str, Any] = {
            **base,
            "block_sizes": list(blocks),
            "tcgen05_region_ab_stages": simt_stages,
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY: pre_acc_wait,
        }
        limit = smem.aux_prefetch_depth_limit(state, twin)
        if limit:
            ring: dict[str, Any] = {
                **twin,
                TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY: limit,
            }
            seeds.append(Config(**ring))
        hoisted: dict[str, Any] = {
            **twin,
            TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY: True,
        }
        if repair.aux_fragment_hoist_admissible(state, hoisted):
            seeds.append(Config(**hoisted))
        # The hoist alone also renders on the 128-row pair's 64-row
        # per-CTA fragment (the hoist is not a subtile-loop rewrite), so
        # the base seed's own tiles take it where the fragment fits the
        # register budget.
        hoisted = {
            **base,
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY: pre_acc_wait,
            TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY: True,
        }
        if hoisted["block_sizes"] != blocks and repair.aux_fragment_hoist_admissible(
            state, hoisted
        ):
            seeds.append(Config(**hoisted))
    # The epilogue-issued ring stages the operand through SMEM, at the
    # deepest AB ring that fits beside its rings; either placement.
    epi_tma_stages = depths(TCGEN05_AUX_LOAD_MODE_EPI_TMA, placement)
    if epi_tma_stages is not None and state.exact_aux_tma_provable:
        staged: dict[str, Any] = {
            **base,
            "block_sizes": list(blocks),
            "tcgen05_region_ab_stages": epi_tma_stages,
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: TCGEN05_AUX_LOAD_MODE_EPI_TMA,
        }
        if repair.epi_tma_refusal(state, staged) is None:
            seeds.append(Config(**staged))
    return seeds


def _paired_pipeline_seed_configs(
    state: CuteTcgen05Config, *, acc_stages: int = 2
) -> list[Config]:
    if not surface.paired_pipeline_search_enabled(state):
        return []
    pipeline_facts = state.pipeline_smem_facts
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    assert pipeline_facts is not None and extents is not None and fragments is not None
    m, n, k = cast("tuple[int, int, int]", extents)
    pid = (
        "persistent_blocked"
        if "persistent_blocked" in state.allowed_pid_types
        else facts._preferred_persistent_pid_type(state)
    )
    if pid is None:
        return []
    seeds: list[Config] = []
    for bm in (128, 256):
        for bn in (64, 128, 256):
            for bk in (
                (64, 128, 256) if pipeline_facts.input_dtype_bytes == 1 else (64, 128)
            ):
                values = (bm, bn, bk)
                if any(
                    (pipeline_facts.input_dtype_bytes != 1 and extent % value != 0)
                    or not fragment.low <= value <= fragment.high
                    for extent, value, fragment in zip(
                        (m, n, k), values, fragments, strict=True
                    )
                ):
                    continue
                if pipeline_facts.input_dtype_bytes == 1 and not (
                    facts._fp8_paired_pipeline_tile_supported(state, bm, bn, bk)
                ):
                    continue
                block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
                assert block_sizes is not None
                placements = (
                    ("post_acc_wait", "pre_acc_wait")
                    if pipeline_facts.has_auxiliary_loads
                    else ("post_acc_wait",)
                )
                # The epilogue-issued aux ring (``epi_tma``) is a second
                # aux mode on the 256-row pair tiles that stage an
                # exact-shape operand: seeded at the deepest AB ring the
                # model fits beside the rings, with both placements.
                aux_modes: tuple[str, ...] = (TCGEN05_AUX_LOAD_MODE_SIMT,)
                if (
                    pipeline_facts.exact_aux_count > 0
                    and state.exact_aux_tma_provable
                    and bm == TCGEN05_TWO_CTA_BLOCK_M
                    and pipeline_facts.input_dtype_bytes == 2
                ):
                    aux_modes = (
                        TCGEN05_AUX_LOAD_MODE_SIMT,
                        TCGEN05_AUX_LOAD_MODE_EPI_TMA,
                    )
                for placement in placements:
                    for aux_mode in aux_modes:
                        maximum = max_pipeline_ab_stages(
                            pipeline_facts,
                            bm=bm,
                            bn=bn,
                            bk=bk,
                            c_stages=2,
                            acc_stages=acc_stages,
                            aux_load_placement=placement,
                            aux_load_mode=aux_mode,
                        )
                        # Include shallow and deepest fitting rings without
                        # changing the C/accumulator allocation or store
                        # protocol; the staged ring takes the deepest only.
                        depths = (
                            (maximum,)
                            if aux_mode == TCGEN05_AUX_LOAD_MODE_EPI_TMA
                            else (min(2, maximum), maximum)
                        )
                        # nvjet's 2x2 clusters stage the same pair tiles
                        # 64 deep in K (``128x256_64x6_2x2_2cta``): the
                        # deepest bk=64 SIMT ring gets a 2x2 twin where
                        # the pair tile is inside the 4-CTA envelope
                        # (256 rows, whole N-tile pairs, full static
                        # tiles).  The twin's render is the pair's with
                        # the cluster lines changed, so the ring depth
                        # carries over.
                        # The epilogue-issued aux ring takes the same
                        # twin: its deepest bk=64 ring beside the aux
                        # rings is the shape nvjet's addmm runs
                        # (``128x256_64x6_2x2``), four stages here.
                        cluster_ns = (1,)
                        if (
                            bk == 64
                            and aux_mode
                            in (
                                TCGEN05_AUX_LOAD_MODE_SIMT,
                                TCGEN05_AUX_LOAD_MODE_EPI_TMA,
                            )
                            and all(
                                extent % value == 0
                                for extent, value in zip((m, n, k), values, strict=True)
                            )
                            and repair._pair_cluster_n2_is_valid(state, bm, bn)
                        ):
                            cluster_ns = (1, 2)
                        for stages in dict.fromkeys(depths):
                            if stages <= 0:
                                continue
                            for cluster_n in cluster_ns:
                                if cluster_n == 2 and stages != maximum:
                                    continue
                                seed = Config(
                                    block_sizes=block_sizes,
                                    pid_type=pid,
                                    tcgen05_cta_group="two",
                                    tcgen05_cluster_m=2,
                                    tcgen05_cluster_n=cluster_n,
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
                                if cluster_n == 2:
                                    # The 2x2 twin keeps its CTAs persistent
                                    # over the device's co-resident clusters
                                    # (``tcgen05_persistent_grid_cap``): the
                                    # pair cap would launch a second hardware
                                    # wave of one-tile CTAs (fp16
                                    # 4096x1024x1024 on the 256x128 pair tile:
                                    # 14.43 us vs 11.90 capped).
                                    seed.config[
                                        TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY
                                    ] = TCGEN05_PERSISTENT_GRID_CAP_RESIDENT_CLUSTERS
                                if pipeline_facts.has_auxiliary_loads:
                                    seed.config[
                                        TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
                                    ] = placement
                                if aux_mode != TCGEN05_AUX_LOAD_MODE_SIMT:
                                    seed.config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = (
                                        aux_mode
                                    )
                                seeds.append(seed)
                                if (
                                    cluster_n == 2
                                    and len(state.config_spec.l2_groupings) == 1
                                ):
                                    # The same twin on a grouped raster: the
                                    # co-resident 2x2 wave then sweeps a band
                                    # of pair rows across the cluster-N
                                    # columns while those A rows are still
                                    # in L2, instead of re-streaming A once
                                    # per cluster-N column (bf16
                                    # 16384x1024x4096: 0.91 -> 0.99 of
                                    # cuBLAS, whose 2x2 nvjet tile it now
                                    # matches).
                                    seeds.append(
                                        Config.from_dict(
                                            {
                                                **seed.config,
                                                "l2_groupings": [
                                                    TCGEN05_CLUSTER_N2_SEED_L2_GROUPING
                                                ],
                                            }
                                        )
                                    )
    return seeds


def _small_grid_seed_configs(state: CuteTcgen05Config) -> list[Config]:
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
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    if (
        not state.search_enabled
        or state.device_sm_count <= 0
        or extents is None
        or fragments is None
        or any(value is None or value <= 0 for value in extents)
        or "persistent_interleaved" not in state.allowed_pid_types
    ):
        return []
    m, n, k = cast("tuple[int, int, int]", extents)
    bm_fragment, bn_fragment, bk_fragment = fragments
    # A leading passthrough (batch) axis multiplies the tile count: the
    # batched 4 x 64x128x128 GEMM has four 128x128 tiles and wants the
    # same 64x{16,32,64} spread (cuBLAS runs it as 64 CTAs of 64x8).
    batch = state.matmul_leading_work_multiplier
    standard_tiles = batch * -(-m // 128) * -(-n // 128)
    if standard_tiles >= state.device_sm_count:
        return []
    bk = bk_fragment.high
    while bk >= bk_fragment.low and (k % bk or bk % 16):
        bk //= 2
    if bk < bk_fragment.low or bk < 16:
        return []
    rowvec_aux_only = (
        state.aux_kernel_detected and not state.exact_shape_aux_kernel_detected
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
                and batch * (m // bm) * (n // bn) > state.device_sm_count
            ):
                continue
            block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
            assert block_sizes is not None
            fit = smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
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
            if rowvec_aux_only and smem.rowvec_aux_stage_fits(
                state,
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


def _split_k_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    """Flat one-CTA split-K seeds for problems whose output tiles cannot
    fill the machine (``tcgen05_split_k``, ``split_k_exchange.py``).

    A GEMM with fewer output tiles than SMs leaves SMs idle however the
    tiles are cut; splitting the K loop across CTAs is the cuBLAS answer
    (its heuristic runs 512x512x65536 as 128x128 tiles x split-K plus a
    reduction kernel). Per admitted tile the seeds take the two smallest
    splits that bring the grid to at least one wave while every split
    keeps at least four K tiles, and the ring depth the SMEM budget
    admits for the tile.
    """
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    if (
        not state.search_enabled
        or extents is None
        or fragments is None
        or any(value is None or value <= 0 for value in extents)
        or "flat" not in state.allowed_pid_types
        or state.matmul_leading_work_multiplier != 1
        or state.grouped_worklist_smem_facts is not None
    ):
        return []
    m, n, k = cast("tuple[int, int, int]", extents)
    bm_fragment, bn_fragment, bk_fragment = fragments
    # Without an SM count (hosts without a device) the knob is searched
    # and seeded as on a small machine.
    sm_count = state.device_sm_count
    if sm_count > 0 and -(-m // 128) * -(-n // 128) >= sm_count:
        return []
    choices = surface.split_k_search_choices(state)
    seeds: list[Config] = []
    for bm, bn in ((128, 128), (128, 256), (64, 128), (64, 64), (64, 32)):
        if not (
            bm_fragment.low <= bm <= bm_fragment.high
            and bn_fragment.low <= bn <= bn_fragment.high
        ):
            continue
        tiles = -(-m // bm) * -(-n // bn)
        if sm_count > 0 and tiles >= sm_count:
            continue
        for bk in (128, 64):
            if not bk_fragment.low <= bk <= bk_fragment.high:
                continue
            k_tiles = -(-k // bk)
            splits = [
                split
                for split in choices
                if k_tiles // split >= 4 and tiles * split >= sm_count
            ][:2]
            if not splits:
                # No split reaches one wave with four tiles per split:
                # the deepest split that keeps four tiles.
                splits = [split for split in choices if k_tiles // split >= 4][-1:]
            fit = smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
            block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
            assert block_sizes is not None
            for split in splits:
                stages = max(1, min(k_tiles // split, fit if fit > 0 else 2))
                seed_config: dict[str, Any] = {
                    "block_sizes": block_sizes,
                    "pid_type": "flat",
                    "tcgen05_cta_group": "auto",
                    "tcgen05_cluster_m": 1,
                    "tcgen05_cluster_n": 1,
                    "tcgen05_ab_stages": stages,
                    "tcgen05_c_stages": 2,
                    "tcgen05_acc_stages": 1,
                    TCGEN05_SPLIT_K_CONFIG_KEY: split,
                    "tcgen05_persistence_model": "non_persistent",
                    "tcgen05_strategy": "role_local_monolithic",
                    "tcgen05_layout_strategy": "default",
                    **dict.fromkeys(TCGEN05_LAYOUT_OVERRIDES_KEYS),
                }
                if bn <= TCGEN05_SMALL_GRID_DIRECT_STORE_MAX_BN:
                    seed_config[TCGEN05_C_STORE_MODE_CONFIG_KEY] = (
                        TCGEN05_C_STORE_MODE_DIRECT
                    )
                seeds.append(Config(**seed_config))
    return seeds


def _warp_mma_seed_bk(
    state: CuteTcgen05Config, k: int, dtype: torch.dtype
) -> int | None:
    """The widest K chunk the K fragment admits that divides K."""
    fragments = facts._matmul_block_fragments(state)
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


def _warp_mma_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    """Register-MMA tiles for the admitted latency-bound GEMMs.

    One-warp 16x8 tiles (the Helion-Triton fp8 256^3 winner's structure:
    2.24 us vs cuBLAS 2.62) and the four-warp 64x8 / 32x32 and two-warp
    32x16 tiles that won the fp16 256^3 + bias and (4, 64, 128, 128)
    probes (2.64 vs 2.88, 2.19 vs 2.66); the timer decides against the
    tcgen05 small-grid seeds.
    """
    problem = facts._warp_mma_problem(state)
    fragments = facts._matmul_block_fragments(state)
    if problem is None or fragments is None:
        return []
    m, n, k, dtype = problem
    bk = _warp_mma_seed_bk(state, k, dtype)
    if bk is None:
        return []
    bm_fragment, bn_fragment, _bk_fragment = fragments
    leading_index = facts._config_block_index(state, state.matmul_leading_block_id)
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
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
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


def _batched_multi_tile_seed_configs(state: CuteTcgen05Config) -> list[Config]:
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
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    dtype = state.matmul_input_dtype
    constraints = state.cluster_m2_search_constraints
    if (
        not state.search_enabled
        or not state.matmul_has_leading_passthrough
        or state.aux_kernel_detected
        or state.device_sm_count <= 0
        or extents is None
        or fragments is None
        or dtype not in (torch.float16, torch.bfloat16)
        or constraints is None
        or constraints.one_wave_only
        or any(value is None or value <= 0 for value in extents)
        or TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types
    ):
        return []
    m, n, k = cast("tuple[int, int, int]", extents)
    bm_fragment, bn_fragment, bk_fragment = fragments
    batch = state.matmul_leading_work_multiplier
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
        or not repair.cluster_m2_bk_is_valid(bk, constraints)
        or batch * (m // bm) * (n // bn) * 2 < state.device_sm_count
    ):
        return []
    ab_stages = 6
    if not smem.ab_stages_three_fits(
        state, bm=bm, bn=bn, bk=bk, cluster_m=2, ab_stages=ab_stages
    ):
        return []
    block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
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


def _one_wave_seed_configs(state: CuteTcgen05Config) -> list[Config]:
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
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    dtype = state.matmul_input_dtype
    if (
        not state.search_enabled
        or state.device_sm_count <= 0
        or extents is None
        or fragments is None
        or dtype is None
        or any(value is None or value <= 0 for value in extents)
        or (state.aux_kernel_detected and state.exact_shape_aux_kernel_detected)
        or "persistent_interleaved" not in state.allowed_pid_types
    ):
        return []
    m, n, k = cast("tuple[int, int, int]", extents)
    bm_fragment, bn_fragment, bk_fragment = fragments
    batch = state.matmul_leading_work_multiplier
    sm_count = state.device_sm_count
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
        state.aux_kernel_detected and not state.exact_shape_aux_kernel_detected
    )
    seeds: list[Config] = []

    def emit(bm: int, bn: int, cluster_m: int, cluster_n: int) -> None:
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        assert block_sizes is not None
        fit = smem.max_ab_stages_that_fit(
            state, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
        )
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
        if rowvec_aux_only and smem.rowvec_aux_stage_fits(
            state,
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
    constraints = state.cluster_m2_search_constraints
    if (
        constraints is not None
        and constraints.allow_one_wave_tiles
        and repair.cluster_m2_bk_is_valid(bk, constraints)
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
            if repair._one_wave_cluster_n2_is_valid(state, two_cta[1]):
                emit(TCGEN05_TWO_CTA_BLOCK_M, two_cta[1], 2, 2)
    return seeds


def _multi_tile_two_cta_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    """Persistent two-CTA seeds for GEMMs whose 256x256 grid is one tile per CTA.

    When the 256x256 CtaGroup.TWO grid holds at most one tile per SM, the
    generated kernel runs a single tile per CTA: its prologue and its
    whole epilogue sit on the critical path with nothing to overlap.
    Quack's autotuner picks the 256x128 two-CTA tile in a 2x2 cluster for
    that shape and runs its 128 pair tiles PERSISTENTLY on the device's
    co-resident cluster count, so the epilogue of one tile hides behind
    the mainloop of the next and the exposed tail is half a tile.  Seed
    that geometry: the narrower one-wave N tiles whose CTA count exceeds
    the SM count, in a 2x2 cluster capped at the resident clusters
    (``tcgen05_persistent_grid_cap=resident_clusters``) and as plain CTA
    pairs (already capped at one pair per SM pair), with the deepest AB
    ring that fits.  The search's timer separates these families only
    with the seeds in the initial population.
    """
    extents = state.matmul_compile_time_static_extents
    fragments = facts._matmul_block_fragments(state)
    dtype = state.matmul_input_dtype
    constraints = state.cluster_m2_search_constraints
    if (
        not state.search_enabled
        or state.device_sm_count <= 0
        or extents is None
        or fragments is None
        or dtype is None
        or constraints is None
        or not constraints.allow_one_wave_tiles
        or state.matmul_has_leading_passthrough
        or any(value is None or value <= 0 for value in extents)
        or "persistent_interleaved" not in state.allowed_pid_types
    ):
        return []
    m, n, k = cast("tuple[int, int, int]", extents)
    bm_fragment, bn_fragment, bk_fragment = fragments
    sm_count = state.device_sm_count
    bm = TCGEN05_TWO_CTA_BLOCK_M
    if (
        m % bm
        or not (bm_fragment.low <= bm <= bm_fragment.high)
        or n % TCGEN05_TWO_CTA_BLOCK_N
        or (m // bm) * (n // TCGEN05_TWO_CTA_BLOCK_N) * 2 > sm_count
    ):
        return []
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    bk_choices = (64, 128) if dtype_bytes >= 2 else (128, 64)
    bk = next(
        (
            value
            for value in bk_choices
            if k % value == 0
            and bk_fragment.low <= value <= bk_fragment.high
            and repair.cluster_m2_bk_is_valid(value, constraints)
        ),
        None,
    )
    if bk is None:
        return []
    rowvec_aux_only = (
        state.aux_kernel_detected and not state.exact_shape_aux_kernel_detected
    )
    seeds: list[Config] = []
    for bn in TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS:
        if (
            n % bn
            or not (bn_fragment.low <= bn <= bn_fragment.high)
            or (m // bm) * (n // bn) * 2 <= sm_count
        ):
            continue
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        assert block_sizes is not None
        fit = smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=2)
        stages = max(1, min(k // bk, fit if fit > 0 else 2))
        for cluster_n in (2, 1):
            if cluster_n == 2 and not repair._one_wave_cluster_n2_is_valid(state, bn):
                continue
            seed_config: dict[str, Any] = {
                "block_sizes": block_sizes,
                "pid_type": "persistent_interleaved",
                "tcgen05_cta_group": "auto",
                "tcgen05_cluster_m": 2,
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
            if cluster_n == 2:
                seed_config[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] = (
                    TCGEN05_PERSISTENT_GRID_CAP_RESIDENT_CLUSTERS
                )
            if rowvec_aux_only and smem.rowvec_aux_stage_fits(
                state,
                bm=bm,
                bn=bn,
                bk=bk,
                cluster_m=2,
                ab_stages=stages,
                c_stages=2,
            ):
                seed_config[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                    TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                )
            if state.epilogue_fanout_plans and repair.epilogue_fanout_config_supported(
                state, seed_config
            ):
                # Two outputs of one tile share the TMEM read and the
                # bias load; the two-CTA fan-out is validated for exactly
                # this 256x128 16-bit tile.
                seed_config[FANOUT_CONFIG_KEY] = "shared"
            seeds.append(Config(**seed_config))
        # The narrowest multi-tile N keeps the seed count small; the
        # search widens or narrows the tile from there.
        break
    return seeds


def _one_cta_persistent_parameters(
    state: CuteTcgen05Config, config: dict[str, object]
) -> tuple[int, int, int] | None:
    """Plain one-CTA static-persistent role-local family (cluster 1x1).

    The same role-local ``while`` builder that emits the two-CTA TMA-role
    dependency wait serves this family, so its TMA warp can wait on a
    programmatic dependency before its first operand read.
    """
    view = facts._matmul_config_view(state, config)
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


def cluster_m2_seed_bk_choices(state: CuteTcgen05Config) -> tuple[int, ...]:
    """Keep the largest legal K seed and earlier/default K geometries.

    Search-range extensions preserve ``default_size`` so non-autotuned
    configs stay stable. When a proved dense pipeline widened this K
    block, its former native K128 ceiling, which may differ from that
    default, is kept too. Preserve both geometries in the initial
    population without replacing the expanded-high seed.
    """
    constraints = state.cluster_m2_search_constraints
    fragments = facts._matmul_block_fragments(state)
    indices = facts._matmul_block_indices(state)
    if constraints is None or fragments is None or indices is None:
        return ()
    fragment = fragments[2]
    if constraints.allow_edge_k_tail_family:
        bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
        return (
            (bk,)
            if fragment.low <= bk <= fragment.high
            and repair.cluster_m2_bk_is_valid(bk, constraints)
            else ()
        )
    bk = fragment.high
    while bk >= fragment.low:
        if repair.cluster_m2_bk_is_valid(bk, constraints):
            break
        bk //= 2
    else:
        return ()
    block = state.config_spec.block_sizes[indices[2]]
    choices = [bk]
    candidates = [block.default_size]
    if state._native_k_widened_block is block:
        # The native K128 ceiling that preceded the dense-pipeline widening.
        candidates.insert(0, 128)
    for candidate in candidates:
        if (
            candidate is not None
            and block.bounded_by_block_id is None
            and candidate not in choices
            and fragment.low <= candidate <= fragment.high
            and candidate > 0
            and candidate & (candidate - 1) == 0
            and repair.cluster_m2_bk_is_valid(candidate, constraints)
        ):
            choices.append(candidate)
    return tuple(choices)


def full_tile_direct_entry_seed_bk(state: CuteTcgen05Config) -> int | None:
    """Largest valid full-tile direct-entry K tile for the live shape.

    Mirrors the full-tile branch of the heuristic bk selection: the highest
    power-of-two ``bk`` within the K block-size fragment that divides
    ``static_k`` within the ``max_k_tiles`` cap.
    """
    constraints = state.cluster_m2_search_constraints
    fragments = facts._matmul_block_fragments(state)
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
        if repair.cluster_m2_bk_is_valid(bk, constraints):
            return bk
        bk //= 2
    return None


def full_tile_direct_entry_seed_eligible(state: CuteTcgen05Config) -> bool:
    """Structural eligibility for the generalized TVM-FFI direct-entry seed.

    The direct-entry codegen + runtime validator build their A/B/D TMA
    descriptors from the runtime tensor shapes, but the seed runs whole
    256x256 tiles only (an output edge's partial tiles take the pair search
    instead); the other constraints are structural: a full-tile (NOT
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
    if state.matmul_has_non_tcgen05_operand:
        return False
    # The flat-role launch path hard-requires the TMA A/B pipeline and the
    # TMA store epilogue, which codegen disables when an operand's or the
    # output destination's alignment is unprovable.
    if not state.matmul_operands_tma_provable:
        return False
    constraints = state.cluster_m2_search_constraints
    if (
        constraints is None
        or constraints.allow_edge_k_tail_family
        or constraints.one_wave_only
    ):
        return False
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return False
    # The direct-entry TMA descriptors, SMEM layout, and epilogue tile are
    # dtype-general for any 16-bit operand (the byte math keys on
    # ``dtype_bytes``, and bf16/fp16 are both 2 bytes), so admit bf16 and
    # fp16 with matching operand dtypes. fp32 stays excluded (no tcgen05
    # fp32 SMEM-staged MMA path).
    if state.matmul_input_dtype not in (torch.bfloat16, torch.float16):
        return False
    if state.matmul_explicit_epi_tile_compatible is not True:
        return False
    fragments = facts._matmul_block_fragments(state)
    if fragments is None:
        return False
    bm_fragment, bn_fragment, _ = fragments
    if not (bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high):
        return False
    if not (bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high):
        return False
    bk = full_tile_direct_entry_seed_bk(state)
    if bk is None:
        return False
    # The flat-role coordinates and the FFI launch validator assume whole
    # 256x256 tiles; output-edge shapes reach CtaGroup.TWO search too.
    if not facts._static_full_tiles(
        state, bm=TCGEN05_TWO_CTA_BLOCK_M, bn=TCGEN05_TWO_CTA_BLOCK_N, bk=bk
    ):
        return False
    if not tcgen05_direct_entry_stage_tuple_allowed(
        bk=bk, ab_stage_count=3, c_stage_count=2
    ):
        return False
    return smem.ab_stages_three_fits(
        state,
        bm=TCGEN05_TWO_CTA_BLOCK_M,
        bn=TCGEN05_TWO_CTA_BLOCK_N,
        bk=bk,
        cluster_m=2,
    )


def full_tile_direct_entry_seed_config(
    state: CuteTcgen05Config, *, l2_groupings: list[object] | None = None
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
    if not full_tile_direct_entry_seed_eligible(state):
        return None
    bk = full_tile_direct_entry_seed_bk(state)
    if bk is None:
        return None
    block_sizes = facts._matmul_seed_block_sizes(
        state,
        bm=TCGEN05_TWO_CTA_BLOCK_M,
        bn=TCGEN05_TWO_CTA_BLOCK_N,
        bk=bk,
    )
    if block_sizes is None:
        return None
    # A materializer or another root owns its own grouping coordinate.
    # Preserve those values when projecting an existing configuration.
    l2_groupings = (
        [item._fill_missing() for item in state.config_spec.l2_groupings]
        if l2_groupings is None
        else list(l2_groupings)
    )
    assert state.matmul_block_ids is not None
    l2_groupings[
        state.config_spec.l2_groupings.block_id_to_index(state.matmul_block_ids[0])
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
    if state.config_spec.indexing.length in (3, 4):
        seed["indexing"] = ["tensor_descriptor"] * state.config_spec.indexing.length
    return Config(**seed)


def _c_input_seed_config(state: CuteTcgen05Config) -> Config | None:
    if not state.aux_kernel_detected:
        return None
    constraints = state.cluster_m2_search_constraints
    if constraints is None or constraints.one_wave_only:
        return None
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return None
    if state.matmul_has_leading_passthrough:
        return None
    fragments = facts._matmul_block_fragments(state)
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
            and repair.cluster_m2_bk_is_valid(bk, constraints)
        ):
            return None
    else:
        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if repair.cluster_m2_bk_is_valid(bk, constraints):
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
        TCGEN05_STRATEGY_CONFIG_KEY: (Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value),
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
        ] * state.config_spec.indexing.length
    else:
        seed_config["l2_groupings"] = [1]
        if state.config_spec.indexing.length == 3:
            seed_config["indexing"] = [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ]
    return Config(**seed_config)


def _plain_clc_seed_config(
    state: CuteTcgen05Config, *, bk: int | None = None
) -> Config | None:
    """Autotune seed for the plain full-tile cluster_m=2 CLC family.

    Mirrors ``_c_input_seed_config``'s full-tile branch but for kernels
    without aux operands: ROLE_LOCAL_WITH_SCHEDULER + a scheduler warp
    driving CLC dynamic persistence, no C-input warp. Besides giving the
    search a head start, this seed widens the strategy/scheduler-warps/
    persistence fragments via the compiler-seed mechanism so neighboring
    configs stay explorable.
    """
    if not surface._plain_clc_persistence_search_enabled(state):
        return None
    if not surface._clc_persistence_search_enabled(state):
        return None
    constraints = state.cluster_m2_search_constraints
    if constraints is None or constraints.one_wave_only:
        return None
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return None
    fragments = facts._matmul_block_fragments(state)
    if fragments is None:
        return None
    bm_fragment, bn_fragment, bk_fragment = fragments
    if not (
        bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
        and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
    ):
        return None
    choices = cluster_m2_seed_bk_choices(state)
    if not choices or (bk is not None and bk not in choices):
        return None
    if bk is None:
        bk = choices[0]
    ab_stages = (
        3
        if smem.ab_stages_three_fits(
            state,
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
        TCGEN05_STRATEGY_CONFIG_KEY: (Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value),
        TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
        ),
        TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 1,
    }
    if state.config_spec.indexing.length == 3:
        seed_config["indexing"] = ["tensor_descriptor"] * 3
    return Config(**seed_config)


def _plain_m_pair_seed_config(state: CuteTcgen05Config) -> Config | None:
    """Autotune seed for block_m=512 M-paired tiles (nvjet's B-reuse).

    Two 256-row CtaGroup.TWO subtiles share each K stage's B buffer,
    halving B's SMEM/L2/DRAM traffic. Measured fp16 16384^3: 986-990
    TFLOP/s vs the block_m=256 CLC winner's 908 (co-timed vs cuBLAS's
    971-981). The doubled A staging fits ab=2 at bk=128 under the B200
    SMEM cap. Rides the CLC WITH_SCHEDULER shape when enabled, else the
    static persistent default.
    """
    if not repair._m_pair_block_m_is_valid(state):
        return None
    constraints = state.cluster_m2_search_constraints
    if constraints is None or constraints.one_wave_only:
        return None
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return None
    fragments = facts._matmul_block_fragments(state)
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
    if not repair.cluster_m2_bk_is_valid(bk, constraints):
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
    if surface._plain_clc_persistence_search_enabled(state) and (
        surface._clc_persistence_search_enabled(state)
    ):
        seed_config[TCGEN05_STRATEGY_CONFIG_KEY] = (
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )
        seed_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
    if state.config_spec.indexing.length == 3:
        seed_config["indexing"] = ["tensor_descriptor"] * 3
    return Config(**seed_config)


def _m_pair_deep_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    """nvjet's 64-wide deep ring on the M-paired (block_m=512) tile.

    The bk=128 M-pair seed stages 96 KiB per K step, so it fits two AB
    stages bare and only one next to the TMA-store C ring; a search
    rarely follows a seed that shallow.  Halving the K step halves the
    staging: three AB stages (3 x 48 KiB) plus a four-deep (128, 32) C
    ring fit the B200 budget, and that ring is what wins where the
    512x256 pair grid is the one-wave tile of the problem (fp16
    2048x4096x2048 gradient GEMMs: 34.5 us per GEMM on the 2x2 super-tile
    vs 35.3-36.2 for the 256x256 pairs, 36.9 for the bk=128 M-pair seed,
    36.4 on the 2x1 pair, 35.1 with a two-deep C ring).  Seeded on the
    2x2 super-tile and the 2x1 pair in the static-persistent monolithic
    schedule: the deepest AB ring that fits next to a two-deep C ring,
    and the four-deep C ring where it fits as well.  Nothing here reads
    the problem's extents beyond what ``_m_pair_block_m_is_valid`` and
    the K-step validity already require.
    """
    if not repair._m_pair_block_m_is_valid(state):
        return []
    constraints = state.cluster_m2_search_constraints
    if constraints is None or constraints.one_wave_only:
        return []
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return []
    fragments = facts._matmul_block_fragments(state)
    if fragments is None:
        return []
    bm_fragment, bn_fragment, bk_fragment = fragments
    pair_bm = 2 * TCGEN05_TWO_CTA_BLOCK_M
    bn = TCGEN05_TWO_CTA_BLOCK_N
    bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
    if not (
        bm_fragment.low <= pair_bm <= bm_fragment.high
        and bn_fragment.low <= bn <= bn_fragment.high
        and bk_fragment.low <= bk <= bk_fragment.high
    ):
        return []
    if not repair.cluster_m2_bk_is_valid(bk, constraints):
        return []

    def fits(ab_stages: int, c_stages: int) -> bool:
        return smem.c_stages_fits(
            state,
            bm=pair_bm,
            bn=bn,
            bk=bk,
            cluster_m=2,
            ab_stages=ab_stages,
            c_stages=c_stages,
            has_source_c=False,
        )

    ab_stages = smem.max_ab_stages_that_fit(
        state, bm=pair_bm, bn=bn, bk=bk, cluster_m=2
    )
    while ab_stages > 1 and not fits(ab_stages, 2):
        ab_stages -= 1
    if ab_stages < 2:
        return []
    c_stages = 4 if fits(ab_stages, 4) else 2
    seeds: list[Config] = []
    for cluster_n in (2, 1):
        seed_config: dict[str, Any] = {
            "block_sizes": [pair_bm, bn, bk],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "l2_groupings": [1],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": cluster_n,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": c_stages,
        }
        if state.config_spec.indexing.length == 3:
            seed_config["indexing"] = ["tensor_descriptor"] * 3
        seeds.append(Config(**seed_config))
    return seeds


def _plain_edge_deep_ab_stages(state: CuteTcgen05Config, bk: int) -> int:
    """Deepest AB pipeline that leaves room for the TMA-store epilogue's
    C ring on the 256x256 cluster_m=2 tile.

    Overflowing the C ring silently demotes the kernel to the much slower
    all-SIMT store (ab=6 fits bare AB at bk=64 but not AB + C, and
    measures far below ab=5 + TMA store), so walk down from the bare-AB
    maximum until AB + C fits.
    """
    ab_stages = smem.max_ab_stages_that_fit(
        state,
        bm=TCGEN05_TWO_CTA_BLOCK_M,
        bn=TCGEN05_TWO_CTA_BLOCK_N,
        bk=bk,
        cluster_m=2,
    )
    while ab_stages > TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES and not smem.c_stages_fits(
        state,
        bm=TCGEN05_TWO_CTA_BLOCK_M,
        bn=TCGEN05_TWO_CTA_BLOCK_N,
        bk=bk,
        cluster_m=2,
        ab_stages=ab_stages,
        c_stages=2,
        has_source_c=False,
    ):
        ab_stages -= 1
    return ab_stages


def _plain_edge_seed_config(state: CuteTcgen05Config) -> Config | None:
    """Autotune seed for the plain (no-aux) edge+K-tail cluster_m=2 family.

    The deep-AB edge regime: bk=64 halves per-stage AB SMEM so a 16-bit
    kernel fits a 5-stage AB pipeline (bf16 5000^3: 948 TFLOP/s vs 815 at
    the bk=128/ab=2 projection). Partial stripes and the K tail stay
    clamped TMA boxes; the TMA-store epilogue covers fringe output tiles
    via descriptor clamping. Seeding also widens the ab-stages fragment
    via the compiler-seed mechanism so nearby depths stay explorable.
    """
    if state.aux_kernel_detected or state.matmul_has_leading_passthrough:
        return None
    constraints = state.cluster_m2_search_constraints
    if constraints is None or not constraints.allow_edge_k_tail_family:
        return None
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return None
    fragments = facts._matmul_block_fragments(state)
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
        and repair.cluster_m2_bk_is_valid(bk, constraints)
    ):
        bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
        if not (
            bk_fragment.low <= bk <= bk_fragment.high
            and repair.cluster_m2_bk_is_valid(bk, constraints)
        ):
            return None
    ab_stages = _plain_edge_deep_ab_stages(state, bk)
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
    if surface._clc_persistence_search_enabled(state):
        # CLC dynamic persistence also wins on the deep-AB edge family
        # (bf16 5000^3: 883 vs 859 TFLOP/s static, same-GPU pair).
        seed_config[TCGEN05_STRATEGY_CONFIG_KEY] = (
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )
        seed_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
    if state.config_spec.indexing.length == 3:
        seed_config["indexing"] = ["tensor_descriptor"] * 3
    return Config(**seed_config)


def _plain_edge_pair128_seed_configs(
    state: CuteTcgen05Config, edge_seed: Config
) -> list[Config]:
    """128-row CTA-pair twins of the plain edge seed
    (``plain_edge_pair128_admitted``). bf16 1000^3 runs the pair at 7.5 us
    against the 256-row edge seed's 11.3 and the flat one-CTA winner's 8.4
    (burst timer; cuBLAS, a 2x1-cluster 64x112 kernel, 6.5 us burst and
    7.4 us sustained).
    """
    if not repair.plain_edge_pair128_admitted(state):
        return []
    fragments = facts._matmul_block_fragments(state)
    assert fragments is not None
    _bm_fragment, _bn_fragment, bk_fragment = fragments
    seeds: list[Config] = []
    constraints = state.cluster_m2_search_constraints
    assert constraints is not None
    for bk, stages in ((64, 5), (128, 6)):
        if not bk_fragment.low <= bk <= bk_fragment.high:
            continue
        # The edge family's K-tile rule (a bk the search repair would
        # rewrite is no seed) and a ring depth that fits the pair.
        if not repair.cluster_m2_bk_is_valid(bk, constraints):
            continue
        fit = smem.max_ab_stages_that_fit(state, bm=128, bn=128, bk=bk, cluster_m=2)
        if fit <= 0:
            continue
        seed_config: dict[str, Any] = dict(edge_seed.config)
        seed_config.update(
            {
                "block_sizes": facts._matmul_seed_block_sizes(
                    state, bm=128, bn=128, bk=bk
                ),
                TCGEN05_CTA_GROUP_CONFIG_KEY: "two",
                "tcgen05_cluster_m": 2,
                "tcgen05_cluster_n": 1,
                "tcgen05_ab_stages": min(stages, fit),
                "tcgen05_acc_stages": 2,
                "tcgen05_c_stages": 2,
            }
        )
        for key in (
            TCGEN05_STRATEGY_CONFIG_KEY,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
        ):
            seed_config.pop(key, None)
        seeds.append(Config(**seed_config))
    return seeds


def _plain_cluster_n2_seed_config(
    state: CuteTcgen05Config, *, deep: bool = False, bk: int | None = None
) -> Config | None:
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
        state.aux_kernel_detected and not state.exact_shape_aux_kernel_detected
    )
    if (
        state.aux_kernel_detected and not rowvec_aux_only
    ) or state.matmul_has_leading_passthrough:
        return None
    constraints = state.cluster_m2_search_constraints
    if (
        constraints is None
        or constraints.allow_edge_k_tail_family
        or constraints.one_wave_only
    ):
        return None
    if TCGEN05_TWO_CTA_SEED_PID_TYPE not in state.allowed_pid_types:
        return None
    fragments = facts._matmul_block_fragments(state)
    if fragments is None:
        return None
    bm_fragment, bn_fragment, bk_fragment = fragments
    if not (
        bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
        and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
    ):
        return None
    if deep:
        assert bk is None, "deep seeds own their K tile"
        bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
        ab_stages = _two_cta_deep_ab_stages(state, bk)
        if not (
            bk_fragment.low <= bk <= bk_fragment.high
            and repair.cluster_m2_bk_is_valid(bk, constraints)
            and smem.ab_stages_three_fits(
                state,
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=bk,
                cluster_m=2,
                ab_stages=ab_stages,
            )
        ):
            return None
    else:
        choices = cluster_m2_seed_bk_choices(state)
        if not choices or (bk is not None and bk not in choices):
            return None
        if bk is None:
            bk = choices[0]
        # ab=3 only where the SMEM-budget gate admits it (B200-class
        # optin); sub-B200 devices keep the known-good ab=2 envelope.
        ab_stages = 3 if state.ab_stages_three_search_constraints is not None else 2
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
    if rowvec_aux_only and smem.rowvec_aux_stage_fits(
        state,
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
    if state.config_spec.indexing.length == 3:
        seed_config["indexing"] = ["tensor_descriptor"] * 3
    return Config(**seed_config)


def _two_cta_deep_ab_stages(state: CuteTcgen05Config, bk: int) -> int:
    """AB depth of the deep (bk=64) 256x256 two-CTA seed.

    nvjet's 64x6 ring for 16-bit operands; fp8 operands halve the
    per-stage SMEM, so the ring goes as deep as the budget admits (12 at
    256x256x64, nvjet's fp8 kernels run 128x6 = the same bytes in flight).
    Row-vector aux kernels whose seed stages the row hand it one AB
    stage where that is what admits the stage
    (``_deep_ab_stages_with_rowvec_stage``).
    """
    constraints = state.ab_stages_three_search_constraints
    if constraints is None or constraints.dtype_bytes != 1:
        ab_stages = TCGEN05_TWO_CTA_DEEP_AB_STAGES
    else:
        fit = smem.max_ab_stages_that_fit(
            state,
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            cluster_m=2,
        )
        ab_stages = max(TCGEN05_TWO_CTA_DEEP_AB_STAGES, fit)
    return _deep_ab_stages_with_rowvec_stage(state, ab_stages, bk=bk)


def _deep_ab_stages_with_rowvec_stage(
    state: CuteTcgen05Config, ab_stages: int, *, bk: int
) -> int:
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
    facts = state.rowvec_aux_facts
    if (
        facts is None
        or not state.aux_kernel_detected
        or state.exact_shape_aux_kernel_detected
        or ab_stages <= 1
        or tcgen05_rowvec_stage_smem_bytes(
            rows=facts.rows, bn=TCGEN05_TWO_CTA_BLOCK_N, epi_warps=4
        )
        == 0
    ):
        return ab_stages

    def fits(depth: int) -> bool:
        return smem.rowvec_aux_stage_fits(
            state,
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


def _aux_tma_seed_config(
    state: CuteTcgen05Config, c_input_seed: Config
) -> Config | None:
    if not surface._aux_tma_search_enabled(state):
        return None
    seed_config: dict[str, Any] = dict(c_input_seed.config)
    seed_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
    return Config(**seed_config)


def _clc_persistence_seed_config(
    state: CuteTcgen05Config, base_seed: Config
) -> Config | None:
    if not surface._clc_persistence_search_enabled(state):
        return None
    seed_config: dict[str, Any] = dict(base_seed.config)
    seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
        Tcgen05PersistenceModel.CLC_PERSISTENT.value
    )
    return Config(**seed_config)


def _set_clc_aux_tma_edge_perf_knobs(
    state: CuteTcgen05Config, config: dict[str, object]
) -> None:
    config["tcgen05_acc_stages"] = TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES
    config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING]
    range_knobs = _clc_aux_tma_edge_range_knobs(state)
    if range_knobs is not None:
        (
            config["range_flattens"],
            config["range_multi_buffers"],
            config["range_warp_specializes"],
        ) = range_knobs


def _clc_aux_tma_wide_n_seed_config(
    state: CuteTcgen05Config, clc_aux_tma_seed: Config
) -> Config:
    seed_config: dict[str, Any] = dict(clc_aux_tma_seed.config)
    _set_clc_aux_tma_edge_perf_knobs(state, seed_config)
    return Config(**seed_config)


def _clc_aux_tma_edge_range_knobs(
    state: CuteTcgen05Config,
) -> tuple[list[bool | None], list[bool | None], list[bool | None]] | None:
    k_range_index = _clc_aux_tma_matmul_k_range_index(state)
    if k_range_index is None:
        return None
    range_flattens: list[bool | None] = [None for _ in state.config_spec.range_flattens]
    range_multi_buffers: list[bool | None] = [
        None for _ in state.config_spec.range_multi_buffers
    ]
    range_warp_specializes: list[bool | None] = [
        None for _ in state.config_spec.range_warp_specialize
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


def _clc_aux_tma_matmul_k_range_index(state: CuteTcgen05Config) -> int | None:
    k_range_indices: set[int] = set()
    range_flattens_ids = state.config_spec.range_flattens.valid_block_ids()
    range_multi_buffers_ids = state.config_spec.range_multi_buffers.valid_block_ids()
    range_warp_specialize_ids = (
        state.config_spec.range_warp_specialize.valid_block_ids()
    )
    for fact in state.config_spec.matmul_facts:
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
        range_index = state.config_spec.range_flattens.block_id_to_index(k_block_id)
        if range_index != state.config_spec.range_multi_buffers.block_id_to_index(
            k_block_id
        ):
            return None
        if range_index != state.config_spec.range_warp_specialize.block_id_to_index(
            k_block_id
        ):
            return None
        k_range_indices.add(range_index)
    if len(k_range_indices) != 1:
        return None
    return next(iter(k_range_indices))


def _clc_aux_tma_narrow_n_seed_config(
    state: CuteTcgen05Config, clc_aux_tma_seed: Config
) -> Config | None:
    if not facts._has_any_matmul_fact_n_edge_for_block_n(
        state, TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
    ):
        return None
    fragments = facts._matmul_block_fragments(state)
    if fragments is None:
        return None
    bn_fragment = fragments[1]
    if not (
        bn_fragment.low
        <= TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        <= bn_fragment.high
    ):
        return None
    constraints = state.cluster_m2_search_constraints
    if (
        constraints is None
        or constraints.one_wave_only
        or not repair.cluster_m2_bk_is_valid(
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
    seed_config["tcgen05_acc_stages"] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
    seed_config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING]
    return Config(**seed_config)


def autotune_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    seeds = _paired_pipeline_seed_configs(state)
    seeds.extend(_small_grid_seed_configs(state))
    seeds.extend(_split_k_seed_configs(state))
    seeds.extend(_warp_mma_seed_configs(state))
    seeds.extend(_one_wave_seed_configs(state))
    seeds.extend(_multi_tile_two_cta_seed_configs(state))
    seeds.extend(_batched_multi_tile_seed_configs(state))
    plain_clc_seed = _plain_clc_seed_config(state)
    if plain_clc_seed is not None:
        seeds.append(plain_clc_seed)
    plain_edge_seed = _plain_edge_seed_config(state)
    if plain_edge_seed is not None:
        seeds.append(plain_edge_seed)
        seeds.extend(_plain_edge_pair128_seed_configs(state, plain_edge_seed))
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
        constraints = state.cluster_m2_search_constraints
        if (
            constraints is not None
            and repair.cluster_m2_bk_is_valid(64, constraints)
            and smem.ab_stages_three_fits(
                state,
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
    plain_cluster_n2_seed = _plain_cluster_n2_seed_config(state)
    if plain_cluster_n2_seed is not None:
        seeds.append(plain_cluster_n2_seed)
        deep_cluster_n2_seed = _plain_cluster_n2_seed_config(state, deep=True)
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
            clc_n2_seed_config["l2_groupings"] = [TCGEN05_CLUSTER_N2_SEED_L2_GROUPING]
            seeds.append(Config(**clc_n2_seed_config))
    # An expanded K ceiling must not evict earlier/default geometries'
    # pipeline/aux placement. Rebuild each family at each K tile so its
    # dependent shared-memory decisions remain valid.
    for bk in cluster_m2_seed_bk_choices(state)[1:]:
        alternate_n2 = _plain_cluster_n2_seed_config(state, bk=bk)
        if alternate_n2 is not None:
            seeds.append(alternate_n2)
        alternate_clc = _plain_clc_seed_config(state, bk=bk)
        if alternate_clc is not None:
            seeds.append(alternate_clc)
            if alternate_n2 is not None:
                clc_n2 = dict(alternate_clc.config)
                clc_n2["tcgen05_cluster_n"] = 2
                clc_n2["l2_groupings"] = [TCGEN05_CLUSTER_N2_SEED_L2_GROUPING]
                seeds.append(Config.from_dict(clc_n2))
    plain_m_pair_seed = _plain_m_pair_seed_config(state)
    if plain_m_pair_seed is not None:
        seeds.append(plain_m_pair_seed)
        # 2x2 super-tile: A multicast across the cluster-N pairs on top
        # of the SMEM-shared B within each M-paired tile. Wins on
        # K-dominated shapes (fp16 4096x4096x32768: ratio 1.016 vs
        # cuBLAS, from 0.919 at block_m=256/cluster_n=1).
        m_pair_cn2_seed_config: dict[str, Any] = dict(plain_m_pair_seed.config)
        m_pair_cn2_seed_config["tcgen05_cluster_n"] = 2
        seeds.append(Config(**m_pair_cn2_seed_config))
        # The same pair tile on nvjet's 64-wide deep ring (2x2 and 2x1).
        seeds.extend(_m_pair_deep_seed_configs(state))
    c_input_seed = _c_input_seed_config(state)
    if c_input_seed is not None:
        seeds.append(c_input_seed)
        clc_c_input_seed = _clc_persistence_seed_config(state, c_input_seed)
        if clc_c_input_seed is not None:
            seeds.append(clc_c_input_seed)
        aux_tma_seed = _aux_tma_seed_config(state, c_input_seed)
        if aux_tma_seed is not None:
            seeds.append(aux_tma_seed)
            clc_aux_tma_seed = _clc_persistence_seed_config(state, aux_tma_seed)
            if clc_aux_tma_seed is not None:
                clc_aux_tma_seed = _clc_aux_tma_wide_n_seed_config(
                    state, clc_aux_tma_seed
                )
                seeds.append(clc_aux_tma_seed)
                clc_aux_tma_narrow_n_seed = _clc_aux_tma_narrow_n_seed_config(
                    state, clc_aux_tma_seed
                )
                if clc_aux_tma_narrow_n_seed is not None:
                    seeds.append(clc_aux_tma_narrow_n_seed)
    if surface._pdl_searched(state):
        # Every seed launches as a programmatic dependent where the knob
        # is searched.  The catalogue times a kernel back to back with
        # its neighbours, where the dependent launch is worth 1.5% (two
        # 2048x4096x2048 GEMMs: 72.7 vs 73.8 us) to 5% (8192x1024x768:
        # 15.6 vs 16.4 us), and every tcgen05 winner of the sweeps
        # carries it.  A seed without it is screened out of the final
        # selection by its own handicap before its schedule is compared
        # with the search's best, so the seeds carry the knob (which the
        # tcgen05 fragment also pins True: ``optional_fragments``), and
        # no seed list grows a twin.
        seeds = [Config.from_dict({**seed.config, "cute_pdl": True}) for seed in seeds]
    return seeds


def dense_aligned_n_seed_configs(state: CuteTcgen05Config) -> list[Config]:
    seeds: list[Config] = [
        Config.from_dict(values | {"tcgen05_ab_stages": depth})
        for width in ALIGNED_N_CHOICES
        for values, plan in facts._dense_aligned_n_width_plans(state, width)
        for depth in dict.fromkeys((min(2, plan.max_ab_stages), plan.max_ab_stages))
    ]
    # Preserve the ranked normal-store prefix exactly. Direct stores use
    # the same conservative allocation bound and append only after their
    # complete graph/geometry proof succeeds.
    for seed in tuple(seeds):
        values = seed.config | {"tcgen05_c_store_mode": "direct"}
        if facts._dense_aligned_n_plan(state, values) is not None:
            seeds.append(Config.from_dict(values))
    return seeds
