"""Autotune-off default tile for dense tcgen05 GEMMs.

Inductor's native-matmul path runs Helion kernels with
``autotune_effort="none"``, so the default config is what such a GEMM runs.
Among the compiler's proved tcgen05 seeds (plus the one-CTA 128x256 tile on
whole or N-edge tiles -- 128x128 too for a single row of M tiles --, the
256-row pair over an M edge, one N tile or a wide N, and the 16-bit dense
aligned-N pairs at their physical widths) this heuristic promotes the one
whose tile minimizes wave-quantized MMA time; whole 256x256-tile shapes keep
the FFI direct-entry pair unless a candidate is clearly cheaper, with its
promoted bias rows staged (``pre_acc_wait``) where the heuristic promotes
that seed. A whole-tile pair stages its exact-shape operands through the
epilogue-issued TMA ring (``_epi_tma_aux_values``), and a grid with fewer
128x128 tiles than SMs takes the 128-row pair seed with the fewest waves
(``_small_grid_pair_default``). Plain GEMMs only the edge + K-tail family
runs (a 16-bit K tail on whole M and N tiles, or a K tail with ragged M and
N) take that family's CLC pair seed. A small grid no pair serves splits K on
a one-CTA 64x64 tile (``_split_k_default``). A transcendental epilogue runs two
epilogue warp groups (``_transcendental_epilogue_values``). A kernel with an
atomic keeps the flat base default: codegen refuses atomics beside the pairs'
shared tiles and the persistent CTAs' warp roles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ....._utils import cdiv
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...cp_async_operand import TCGEN05_CP_ASYNC_A_BLOCK_K
from ...cp_async_operand import TCGEN05_CP_ASYNC_A_TILE_M
from ...dense_aligned_n import TILE_N_KEY
from ...strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ...strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ...strategies import Tcgen05LayoutStrategy
from ...tcgen05_config import facts
from ...tcgen05_config import knobs
from ...tcgen05_config import repair
from ...tcgen05_config import seeds
from ...tcgen05_config import smem
from ...tcgen05_config import surface
from ...tcgen05_config.knobs import long_n_sweep_l2_grouping
from ...tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from ...tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ...tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from ...tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_EPI_TMA_AUX_STAGE_COUNT_CHOICES
from ...tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_LONG_N_SWEEP_L2_GROUPING
from ...tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_SPLIT_K_MIN_SLICE_K_TILES
from ...tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ...tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ...tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ...tcgen05_constants import TCGEN05_TWO_CTA_NARROW_BLOCK_N
from ...tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING

if TYPE_CHECKING:
    import torch

    from .....autotuner.config_spec import ConfigSpec
    from .....runtime.config import PidTypeLiteral
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR
    from ...radix_store import RadixStoreLayout
    from ...tcgen05_config import CuteTcgen05Config

# Per-SM MMA efficiency of a tile relative to the CtaGroup.TWO 256x256 tile,
# whose pair shares one B tile across both CTAs; narrower tiles move more
# operand bytes per MMA.  Calibrated on B200 bf16 inductor GEMMs
# (16384x384x768: 256x128 pair 0.89 vs 256x256 pair 0.68 of cuBLAS;
# 25216x768x3072: one-CTA 128x256 0.98 vs 256x256 pair 0.90).  Tiles below a
# 128x128 per-SM footprint are not candidates of the wave-quantized default:
# their operand traffic, not wave quantization, decides their speed (the
# small-grid default, ``_small_grid_pair_default``, takes the 128-row pair on
# grids that leave SMs idle).
_TILE_EFFICIENCY = {
    (2, 256, 256): 1.0,
    (2, 256, 128): 0.92,
    (1, 128, 256): 0.92,
    (1, 128, 128): 0.85,
}


def _aligned_n_efficiency(cluster_m: int, bm: int, width: int) -> float | None:
    """Per-SM MMA efficiency of a 16-bit dense aligned-N pair tile at its
    physical ``width`` (``tcgen05_tile_n``), interpolated between the 256x128
    and 256x256 pairs: bf16 1000x2560x2048 runs one wave of 64 256x160 pairs
    at 0.93 of cuBLAS where one wave of 40 256x256 pairs runs 0.72;
    4096x2560x2560: 256x160 pairs 0.96 against the 256x256 direct-entry
    default's 0.85. Narrower widths are not candidates of the wave-quantized
    default (16384x960x1024: the 96-wide pair runs 40-42 us against 38 for
    the 256x256 one), nor is the FP8 family's 128-row pair (64 rows per CTA:
    below the per-SM floor; only the small-grid default takes it)."""
    if (cluster_m, bm) != (2, TCGEN05_TWO_CTA_BLOCK_M) or width < 128:
        return None
    low = _TILE_EFFICIENCY[(2, TCGEN05_TWO_CTA_BLOCK_M, 128)]
    high = _TILE_EFFICIENCY[(2, TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N)]
    return low + (high - low) * (width - 128) / (TCGEN05_TWO_CTA_BLOCK_N - 128)


# Tiles per CTA (waves) from which a bias GEMM's default stages a promoted
# row on an epilogue-bound pair (``pre_acc_wait`` beside the hybrid TMA
# store) or moves a row it cannot stage to ``post_acc_wait``'s hybrid TMA
# store, over the SIMT store; an output edge's partial-output store, where
# its rings hold the row stages, applies at any tile count (see ``_pick``).
_TMA_STORE_MIN_WAVES = 16
# K tiles per output tile up to which the epilogue bounds a pair GEMM, so its
# default stages promoted bias rows (``pre_acc_wait``; see ``_pick``).
_EPILOGUE_BOUND_MAX_K_TILES = 2
# An output-edge auxiliary GEMM's default grouping on sweeps shorter than
# the long-N-sweep grouping's (``TCGEN05_LONG_N_SWEEP_*``; see ``_pick``).
_SHORT_N_SWEEP_L2_GROUPING = 4
# The grouping of a long N sweep whose epilogue stages exact-shape operands
# through the epilogue-issued ring (see ``_pick``).
_STAGED_AUX_LONG_N_SWEEP_L2_GROUPING = 8
# AB stages the epilogue-issued auxiliary TMA ring must leave a pair's tile
# for the default to stage its exact-shape operands (``_epi_tma_aux_values``).
_EPI_TMA_MIN_AB_STAGES = 4
# K tiles from which the one-CTA N-edge candidates take an exact-shape
# auxiliary epilogue (see ``_one_cta_candidate``).
_ONE_CTA_N_EDGE_EXACT_AUX_MIN_K_TILES = 8
# K blocks from which a narrow batched GEMM's default runs the persistent loop
# at four AB stages (``_narrow_batched_default``).
_NARROW_BATCHED_PERSISTENT_MIN_K_TILES = 8
# The split-K default's tile edge, the AB stages it keeps at most and the
# CTAs per SM it launches at most (see ``_split_k_default``).
_SPLIT_K_TILE = 64
_SPLIT_K_AB_STAGES = 4
_SPLIT_K_MAX_CTAS_PER_SM = 3
# The shortest K loop (in k-blocks) the split-K default splits; see
# ``_split_k_default``.
_SPLIT_K_MIN_K_TILES = 32
# A candidate replaces the FFI default only below this fraction of its cost.
_FFI_REPLACEMENT_MARGIN = 0.9


def _promotable_seed_view(
    state: CuteTcgen05Config, values: dict[str, Any]
) -> tuple[list[object], int, int, int] | None:
    """The matmul block view of a seed the default may promote: one N CTA,
    the static persistence model, the monolithic role-local strategy and an
    allowed pid type; ``None`` for any other seed."""
    if (
        values.get("tcgen05_cluster_n", 1) != 1
        or values.get("tcgen05_persistence_model", "static_persistent")
        != "static_persistent"
        or values.get("tcgen05_strategy", "role_local_monolithic")
        != "role_local_monolithic"
        or values.get("pid_type") not in state.allowed_pid_types
    ):
        return None
    return facts._matmul_config_view(state, values)


def _batched_radix_store(device_ir: DeviceIR) -> tuple[RadixStoreLayout, int] | None:
    """The batched mixed-radix layout the MMA store analysis proved for the
    kernel's store (a head merge fused into a bmm's store,
    ``radix_store``), with the destination's element size."""
    # ``radix_store`` imports ``helion.language``, which imports this module.
    from ...radix_store import proven_radix_store_layout

    for graph in device_ir.graphs:
        for node in graph.graph.nodes:
            layout = proven_radix_store_layout(node)
            if layout is not None and layout.batch:
                tensor = cast("torch.fx.Node", node.args[0]).meta["val"]
                return layout, tensor.element_size()
    return None


def _one_cta_config(
    block_sizes: list[int],
    *,
    pid_type: PidTypeLiteral,
    ab_stages: int,
    split_k: int | None = None,
) -> Config:
    """A one-CTA tcgen05 config: ``block_sizes`` at ``ab_stages`` AB stages,
    two accumulator and two C stages, and ``split_k`` K slices if given."""
    values: dict[str, Any] = {
        "block_sizes": block_sizes,
        "pid_type": pid_type,
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": ab_stages,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
    }
    if split_k is not None:
        values[TCGEN05_SPLIT_K_CONFIG_KEY] = split_k
    return Config(**values)


def _rendered_values(spec: ConfigSpec, values: dict[str, Any]) -> dict[str, Any]:
    """``values`` as ``default_config()`` renders them: promoted onto the
    base default and repaired by the search fixups."""
    projected = spec.promoted_default_values(values)
    spec.normalize(projected, _fix_invalid=True)
    return projected


def _promoted_values(seed: Config) -> dict[str, Any]:
    """A promoted seed's values; PDL stays a target decision of the base
    default."""
    values = dict(seed.config)
    values.pop("cute_pdl", None)
    return values


class CuteTcgen05DefaultTileHeuristic(AutotunerHeuristic):
    name = "cute_tcgen05_default_tile"
    backend = "cute"
    # Between the two-CTA cluster search and the grouped and thread-local
    # epilogue families: their promoted defaults, later in the order, keep
    # the kernels they claim.
    priority = 24
    promote_seed_to_default = True
    # The tile efficiencies and the FFI margin are B200 measurements.
    PROMOTE_TARGETS = (("cuda", "sm100"), ("cuda", "sm103"))
    CACHE_SPECIALIZATION_FACTS = frozenset({"device_num_sm"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._pick(env, device_ir) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        return cls._pick(env, device_ir)

    @staticmethod
    def _direct_entry_staged_row(state: CuteTcgen05Config) -> Config | None:
        """The FFI direct-entry default with its promoted bias rows staged.

        Whole 256x256 tiles keep the direct-entry pair; under ``post_acc_wait``
        its epilogue loads the bias row from global memory on every subtile,
        which leaves the latency on each tile's epilogue and on the last
        tile's unoverlapped tail. ``pre_acc_wait`` stages the promoted row
        once per tile at the seed's own rings where the arena holds the stage.
        Inductor corpus, bf16 + bias on B200, of cuBLAS: 8192x2304x768 (3.9
        tiles per pair) 0.945 -> 0.983, 16384x2304x768 0.979 -> 1.000,
        32768x1536x768 (10.4 per pair) 0.980 -> 0.982.
        """
        seed = seeds.full_tile_direct_entry_seed_config(state)
        if seed is None or not repair.direct_entry_row_stage_fits(state):
            return None
        values: dict[str, Any] = dict(seed.config)
        values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
            TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
        )
        return Config(**values)

    @staticmethod
    def _edge_k_tail_default(state: CuteTcgen05Config) -> Config | None:
        """The edge + K-tail family's plain pair seed (CLC-persistent where
        the CLC search is enabled, else static persistent): a K tail on whole
        tiles, and a K tail with ragged M and N, run nowhere else but the
        flat one-CTA kernel, whose default 128x16 tile is a tenth of
        cuBLAS. bf16 2048x2560x8008 (fp32 out): 0.92 of cuBLAS on the
        family's 256x256x64 ab=5 seed, 0.08 on the flat default."""
        seed = seeds._plain_edge_seed_config(state)
        if seed is None:
            return None
        return Config(**_promoted_values(seed))

    @staticmethod
    def _cp_async_a_default(env: CompileEnvironment) -> Config | None:
        """The flat one-CTA tile whose epilogue warps copy an unaligned A with
        ``cp.async`` (``cute_mma``'s ``tcgen05_cp_async_a``), at the deepest
        ring that fits.

        Two 128-row subtiles share each B stage (a 256-row tile) while the
        256-row tiles still occupy three quarters of the SMs: every M tile
        streams all of B, so 30522x128x32768 with ``x.T`` as A read 1.9 GB of
        B beside the 2 GB A on 239 128-row tiles and half that on 120 paired
        ones (0.93 -> 1.00 of the ATen GEMM). The flat SIMT base default ran
        0.24, the staged (transposing) copy of A 0.54.
        """
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        extents = state.matmul_compile_time_static_extents
        if extents is None or not all(isinstance(v, int) and v > 0 for v in extents):
            return None
        m, n, k = cast("tuple[int, int, int]", extents)
        bk = TCGEN05_CP_ASYNC_A_BLOCK_K
        bn = 256 if n > 128 else 128
        paired_tiles = cdiv(m, 2 * TCGEN05_CP_ASYNC_A_TILE_M) * cdiv(n, bn)
        bm = (
            2 * TCGEN05_CP_ASYNC_A_TILE_M
            if 4 * paired_tiles >= 3 * state.device_sm_count
            and repair.cp_async_a_paired_rows_allowed(state)
            else TCGEN05_CP_ASYNC_A_TILE_M
        )
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        stages = smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
        if block_sizes is None or stages < 2 or "flat" not in spec.allowed_pid_types:
            return None
        return _one_cta_config(
            block_sizes, pid_type="flat", ab_stages=min(stages, k // bk)
        )

    @staticmethod
    def _epi_tma_aux_values(
        env: CompileEnvironment, values: dict[str, Any], *, k: int
    ) -> dict[str, Any] | None:
        """A whole-tile pair's exact-shape auxiliary operands staged through
        the epilogue-issued TMA ring (``epi_tma``), on the widest N tile
        whose AB ring keeps ``_EPI_TMA_MIN_AB_STAGES`` beside it, or ``None``
        where none does.

        Register (SIMT) loads leave each exact-shape operand's global latency
        on every epilogue subtile. B200, of the ATen GEMM, SIMT -> ``epi_tma``:
        8192x1024x1024 + residual 0.83 -> 1.05 (256x256, ab 6 -> 5);
        32768x256x256 + three operands 0.50 -> 0.90 on 256x128 (ab 6), where
        the 256x256 tile keeps 3 AB stages beside the ring (0.59).

        Where the AB ring still holds a whole tile's K beside a deeper ring,
        the ring takes the depth: such a short-K GEMM is bound by its
        epilogue, and stages past the tile's k-blocks only prefetch the next
        tile (32768x256x256 + three operands: four k-blocks, ab 6 -> 4 beside
        a four-stage ring, 0.92 -> 0.97; 8192x1024x1024 + three operands
        keeps two: ab 4 -> 3 beside three costs 0.99 -> 0.96).
        """
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        view = facts._matmul_config_view(state, values)
        assert view is not None
        blocks, m_index, n_index, k_index = view
        bm, bn, bk = (
            cast("int", blocks[m_index]),
            cast("int", blocks[n_index]),
            cast("int", blocks[k_index]),
        )
        promoted = spec.promoted_default_values(values)

        def keeps_ring(
            normalized: dict[str, Any], block_sizes: list[object], min_ab_stages: int
        ) -> bool:
            """Whether the search fixups (``_normalize_epi_tma``) kept the
            ring and the tile, with ``min_ab_stages`` AB stages beside it."""
            return (
                normalized.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_EPI_TMA
                and normalized["block_sizes"] == block_sizes
                and cast("int", normalized["tcgen05_ab_stages"]) >= min_ab_stages
            )

        # The narrower pair tile where the 256x256 one keeps too few AB
        # stages beside the ring.
        for tile_n in (
            (bn, TCGEN05_TWO_CTA_NARROW_BLOCK_N)
            if bn == TCGEN05_TWO_CTA_BLOCK_N
            else (bn,)
        ):
            block_sizes = list(blocks)
            block_sizes[n_index] = tile_n
            projected = {
                **promoted,
                "block_sizes": block_sizes,
                "tcgen05_ab_stages": smem.max_ab_stages_that_fit(
                    state, bm=bm, bn=tile_n, bk=bk, cluster_m=2
                ),
                TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: TCGEN05_AUX_LOAD_MODE_EPI_TMA,
            }
            # The search fixups judge the ring (``_normalize_epi_tma``) and
            # shorten the AB ring beside it.
            spec.normalize(projected, _fix_invalid=True)
            if not keeps_ring(projected, block_sizes, _EPI_TMA_MIN_AB_STAGES):
                continue
            chosen = {
                **values,
                "block_sizes": block_sizes,
                TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: TCGEN05_AUX_LOAD_MODE_EPI_TMA,
                "tcgen05_ab_stages": projected["tcgen05_ab_stages"],
                "tcgen05_c_stages": projected["tcgen05_c_stages"],
            }
            # The AB ring only shortens as the ring deepens: deeper rings are
            # tried where the default one leaves a whole tile's K in flight.
            deeper_depths = (
                sorted(
                    (
                        depth
                        for depth in TCGEN05_EPI_TMA_AUX_STAGE_COUNT_CHOICES
                        if depth > TCGEN05_AUX_STAGE_COUNT_DEFAULT
                    ),
                    reverse=True,
                )
                if cast("int", projected["tcgen05_ab_stages"]) >= k // bk
                else []
            )
            for depth in deeper_depths:
                deeper = {**projected, TCGEN05_AUX_STAGES_CONFIG_KEY: depth}
                spec.normalize(deeper, _fix_invalid=True)
                # (normalization resets the depth only off ``epi_tma``)
                if keeps_ring(
                    deeper, block_sizes, max(k // bk, _EPI_TMA_MIN_AB_STAGES)
                ):
                    chosen[TCGEN05_AUX_STAGES_CONFIG_KEY] = depth
                    chosen["tcgen05_ab_stages"] = deeper["tcgen05_ab_stages"]
                    chosen["tcgen05_c_stages"] = deeper["tcgen05_c_stages"]
                    break
            return chosen
        return None

    @staticmethod
    def _small_grid_pair_default(
        env: CompileEnvironment, m: int, n: int, k: int, batch: int
    ) -> Config | None:
        """The 128-row CtaGroup.TWO pair seed (64 rows per CTA, persistent)
        that runs in the fewest waves and, among those, fills the most SMs,
        at its seed's AB ring capped at the K loop: a GEMM with fewer
        128x128 tiles than SMs leaves most of them idle on one-CTA tiles, and
        the flat base default ran a 64x16 SIMT kernel (B200, of the ATen
        GEMM: 128x9216x4096 + bias 0.19, the 128x128 pair at its seed's 13 AB
        stages 1.15; where no pair fits one wave, 128x9600x4096 137 -> 29 us
        and 1152x1152x1024 37 -> 12 us on the 128x128 pair's two, against
        ATen's 27 and 10.8 us).

        The pairs are the paired-pipeline seeds (and, on the ragged edge
        family, its 128x128 edge pairs): the allocation proof validates
        them, not the ``cluster_m=2`` search projection, so the default takes
        them where the cluster search stays at one CTA, as the autotuner's
        initial population does. On ties the bias row loaded before the
        accumulator wait (``pre_acc_wait``) wins: one tile per CTA reads it
        once (128x9216x4096 + bias: 24.1 -> 23.2 us)."""
        state = env.config_spec._cute_tcgen05_config
        best: tuple[tuple[int, int, int, bool], Config] | None = None
        for seed in seeds.autotune_seed_configs(state):
            values = seed.config
            view = _promotable_seed_view(state, values)
            if (
                view is None
                or values.get("tcgen05_cluster_m", 1) != 2
                or values.get("pid_type") == "flat"
            ):
                continue
            blocks, m_index, n_index, k_index = view
            bm, bn, bk = blocks[m_index], blocks[n_index], blocks[k_index]
            if (
                bm != 128
                or bk != 64
                or not isinstance(bn, int)
                # the CtaGroup.TWO pair, not the clustered one-CTA family
                or not facts._config_is_two_cta_mma(state, values, bm=bm, cluster_m=2)
                # the persistent pair's validated K-tile envelope (the
                # runtime counts a K tail's tile)
                or cdiv(k, bk) > TCGEN05_TWO_CTA_MAX_K_TILES
            ):
                continue
            ctas = 2 * batch * cdiv(m, 128) * cdiv(n, bn)
            key = (
                -cdiv(ctas, state.device_sm_count),
                ctas,
                cast("int", values.get("tcgen05_ab_stages", 2)),
                values.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT,
            )
            if best is None or key > best[0]:
                best = (key, seed)
        if best is None:
            return None
        values = _promoted_values(best[1])
        # The K loop caps the seed's AB ring, never deepening it.
        values["tcgen05_ab_stages"] = min(
            cast("int", values.get("tcgen05_ab_stages", 2)), max(2, cdiv(k, 64))
        )
        return Config(**values)

    @staticmethod
    def _split_k_default(
        env: CompileEnvironment, m: int, n: int, k: int
    ) -> Config | None:
        """A small grid no 128-row pair serves (a 64-row M, a K tail the
        pair families do not take, a K loop past the pairs' envelope): the
        flat one-CTA 64x64x64 tile at four AB stages (as many as K holds, at
        least two), splitting K (``tcgen05_split_k``) over a K loop of at
        least ``_SPLIT_K_MIN_K_TILES`` k-blocks into the most slices of at
        least four k-blocks whose CTAs three per SM hold; unsplit over a
        shorter loop, where no split qualifies, or where the tiles cannot
        split (``split_k_tiles_supported``: whole tiles only without the
        flat-edge TMA path). A kernel off the split-K search surface
        (``split_k_search_choices``) keeps the base default. Each axis takes
        the 64 nearest its block-size range (a narrower N its width, a pinned
        K block its depth), and the slices are counted on that tile.

        A split pays the exchange: the partial slabs, the last arriver's
        reduction and the workspace counters' memset, which a CUDA graph
        replays with every launch (~1.6 us). B200, Inductor corpus, of the
        ATen GEMM, CUDA graphs with L2 flushed (4 rounds x 4 placements),
        unsplit -> split: 16 k-blocks lose, 128x4096x1000 0.96 -> 0.77 (two
        slices), 512x1280x1000 + mask 0.90 -> 0.64 (two), 64x4096x1000 + two
        operands 0.80 -> 0.66 (four), 128x512x1000 FP32 out 0.83 -> 0.71
        (four), 32x1280x1000 0.64 -> 0.57 (four), 128x1000x768 (12) 0.73 ->
        0.56 (two); long loops gain, 1000x32x2880 (45 k-blocks) 0.55 -> 0.78
        (eight), 64x4096x4096 (64) 0.59 -> 0.69 (four), 256x512x197952 0.11
        -> 0.62 (eight). The base default ran a 64x16 SIMT tile (0.01-0.35).
        Outside a CUDA graph a split launch also costs ~40-80 us more host
        time: its exchange workspace keeps it off the fast relaunch
        (``_cute_build_fast_relaunch`` refuses launcher-owned tensors)."""
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        choices = surface.split_k_search_choices(state)
        fragments = facts._matmul_block_fragments(state)
        # Off the split-K search surface (a body past one plain store, a
        # leading passthrough axis: ``split_k_search_choices``) the kernel
        # keeps the base default.
        if not choices or fragments is None:
            return None
        # The slices are counted on the tile that runs: 64 on each axis,
        # within its block-size range (a narrow N tiles at its width).
        bm, bn, bk = (
            min(max(_SPLIT_K_TILE, fragment.low), fragment.high)
            for fragment in fragments
        )
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        k_tiles = cdiv(k, bk)
        ab_stages = max(2, min(_SPLIT_K_AB_STAGES, k_tiles))
        if (
            bm != _SPLIT_K_TILE
            or block_sizes is None
            or "flat" not in spec.allowed_pid_types
            or smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
            < ab_stages
        ):
            return None
        # A short K loop, or ragged tiles off the flat-edge TMA path
        # (``split_k_tiles_supported``), runs the tile unsplit.
        tiles = cdiv(m, bm) * cdiv(n, bn)
        splittable = (
            k_tiles >= _SPLIT_K_MIN_K_TILES
            and surface.split_k_tiles_supported(state, bm=bm, bn=bn, bk=bk)
        )
        split = max(
            (
                choice
                for choice in choices
                if splittable
                and tiles * choice <= _SPLIT_K_MAX_CTAS_PER_SM * state.device_sm_count
                and k_tiles >= TCGEN05_SPLIT_K_MIN_SLICE_K_TILES * choice
            ),
            default=1,
        )
        return _one_cta_config(
            block_sizes, pid_type="flat", ab_stages=ab_stages, split_k=split
        )

    @staticmethod
    def _narrow_batched_default(
        env: CompileEnvironment,
        m: int,
        n: int,
        k: int,
        radix: tuple[RadixStoreLayout, int] | None = None,
    ) -> Config | None:
        """The one-CTA tile of a batched GEMM whose per-batch matrices are
        narrower than 128 rows or columns (attention's ``[B * H, S, 64]``
        bmms), or whose store merges heads (``radix``: the proved layout and
        the element size): 128 rows (64 where M is not whole in them) by the
        widest of 256, 128, 64 or 32 columns dividing N (a head merge's: the
        widest tile partitioning its layout whose store boxes sit inside its
        runs, ``RadixStoreLayout.one_cta_tile``), the deepest of 64-, 32- or
        16-deep k-blocks dividing K; flat with two AB stages, the persistent
        loop with four from eight k-blocks on.

        The base default ran a 64x16 tile; B200, Inductor corpus, of the ATen
        GEMM (many batches of one or two tiles each): 1024 x 256x64x256
        + scale 1.09 (persistent: 1.08), 3072 x 128x64x128 1.35 (1.08),
        3072 x 64x128x128 1.40 (1.19); eight k-blocks, 64x256 tiles:
        384 x 64x512x512 flat 1.00 (four AB stages 0.93, two CTAs fit an SM
        at two), persistent 1.03, 192 x 64x512x512 1.07. A head merge takes
        it on small grids too, where the 128-row pairs render no radix store
        subtile; a plain narrow GEMM's small grid keeps the pairs' deep rings
        (32 x 128x64x4096: 18.8 us against 31.4 us here)."""
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        bk = next((depth for depth in (64, 32, 16) if k % depth == 0), None)
        if radix is not None:
            layout, element_bytes = radix
            tile = layout.one_cta_tile(element_bytes)
        else:
            bm = next((rows for rows in (128, 64) if m % rows == 0), None)
            bn = next((cols for cols in (256, 128, 64, 32) if n % cols == 0), None)
            tile = None if bm is None or bn is None else (bm, bn)
        if tile is None or bk is None:
            return None
        bm, bn = tile
        persistent = (
            k // bk >= _NARROW_BATCHED_PERSISTENT_MIN_K_TILES
            and "persistent_blocked" in spec.allowed_pid_types
        )
        stages = 4 if persistent else 2
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        if (
            block_sizes is None
            or (not persistent and "flat" not in spec.allowed_pid_types)
            or smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
            < stages
        ):
            return None
        return _one_cta_config(
            block_sizes,
            pid_type="persistent_blocked" if persistent else "flat",
            ab_stages=stages,
        )

    @staticmethod
    def _persistent_tile_values(
        env: CompileEnvironment,
        k: int,
        *,
        bm: int,
        bn: int,
        cluster_m: int,
        bk: int = 64,
        min_ab_stages: int = 1,
    ) -> dict[str, Any] | None:
        """A persistent ``bm x bn x bk`` tile at the deepest AB ring that
        fits, capped at the K loop's k-blocks but not below
        ``min_ab_stages``, or ``None`` where the search does not run it."""
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        if "persistent_interleaved" not in spec.allowed_pid_types or (
            state.cluster_m_search_choices is not None
            and cluster_m not in state.cluster_m_search_choices
        ):
            return None
        fragments = facts._matmul_block_fragments(state)
        if fragments is not None and not knobs._tile_in_fragments(
            fragments, (bm, bn, bk)
        ):
            # outside the block sizes the kernel registered
            return None
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        stages = smem.max_ab_stages_that_fit(
            state, bm=bm, bn=bn, bk=bk, cluster_m=cluster_m
        )
        if block_sizes is None or stages < 2:
            return None
        return {
            "block_sizes": block_sizes,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": cluster_m,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": max(min_ab_stages, min(stages, k // bk)),
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
        }

    @classmethod
    def _pair_m_edge_candidate(
        cls, env: CompileEnvironment, m: int, n: int, k: int
    ) -> list[Config]:
        """The persistent CtaGroup.TWO pair over an M edge, at the deepest
        ring that fits: ``256 x N`` for one whole N tile (N = 128 or 256),
        ``256 x 256`` for a wider N over rows that one-CTA 128-row tiles do
        not cover either. The proved-schedule seeds all ask for whole tiles,
        so such a GEMM fell to the flat 128x16 base default (B200, +
        residual: 30000x128x4096 0.19 of the ATen GEMM, the 256x128 pair
        0.86; 30000x256x4096 616 -> 122 us on the 256x256 pair, and
        1000x4096x4096 0.09 -> 0.93)."""
        if m % TCGEN05_TWO_CTA_BLOCK_M == 0 or k % 64:
            return []
        if n in (TCGEN05_TWO_CTA_NARROW_BLOCK_N, TCGEN05_TWO_CTA_BLOCK_N):
            bn = n
        elif n % TCGEN05_TWO_CTA_BLOCK_N == 0 and m % 128:
            # A wider N over rows that one-CTA 128-row tiles do not cover
            # either (where M is whole in 128-row tiles, the one-CTA 128x256
            # candidate's whole tiles cover it).
            bn = TCGEN05_TWO_CTA_BLOCK_N
        else:
            return []
        state = env.config_spec._cute_tcgen05_config
        bk = 64
        if bn == TCGEN05_TWO_CTA_BLOCK_N and surface._aux_tma_full_tile_search_enabled(
            state
        ):
            # The 256x256 pair runs an exact-shape operand on its TMA producer
            # warp, beside whose rings the SMEM arena fits the AB ring
            # (``aux_producer_ring_fits``, ``_fix_aux_tma_full_tile_search_config``):
            # the shallowest k-blocks within the pair's K-tile cap keep the
            # most loads in flight (1000x4096x4096 + residual, of the ATen
            # GEMM: bk=128 0.79, bk=64 0.91, bk=32 0.93); register loads keep
            # 64-deep ones.
            bk = next(
                (
                    depth
                    for depth in (32, 64, 128)
                    if k % depth == 0 and k // depth <= TCGEN05_TWO_CTA_MAX_K_TILES
                ),
                64,
            )
        # Two AB stages at least: over a single k-block the second stage
        # loads the pair's next tile while the MMA reads this one (the
        # one-CTA candidate keeps the cap its defaults were measured with).
        values = cls._persistent_tile_values(
            env,
            k,
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=bn,
            cluster_m=2,
            bk=bk,
            min_ab_stages=2,
        )
        if bk != 64 and (
            # a registered K block size range without the shallower tile
            values is None
            # the projection left register loads (no AB ring fits beside the
            # operands' rings)
            or _rendered_values(env.config_spec, values).get(
                TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
            )
            != TCGEN05_AUX_LOAD_MODE_TMA
        ):
            # 64-deep k-blocks
            values = cls._persistent_tile_values(
                env, k, bm=TCGEN05_TWO_CTA_BLOCK_M, bn=bn, cluster_m=2, min_ab_stages=2
            )
        return [] if values is None else [Config(**values)]

    @classmethod
    def _one_cta_candidates(
        cls, env: CompileEnvironment, m: int, n: int, k: int
    ) -> list[Config]:
        """The one-CTA 128x256 persistent tile, and for a single row of M
        tiles the 128x128 one.

        The proved-schedule seeds carry it only when they had to restore the
        persistent pid types; a 128-row tile still quantizes M=25216 into
        whole waves where the 256-row pair tile leaves a half-empty last wave
        (0.98 vs 0.90 of cuBLAS).  The persistent one-CTA runtime runs whole
        tiles, and N-edge tiles where every tile takes the partial-output TMA
        store (``smem.one_cta_n_edge_partial_output``, at the
        deepest ring that leaves room for it): 25216x576x192 + bias, whose
        whole-tile seeds are 128x64, runs 0.99 of cuBLAS there against 0.94
        (cuBLAS: a 192x128 one-CTA tile).  With one row of M tiles (a decode
        GEMM streaming B: M = 128, as the one-CTA tiles need whole M tiles,
        so a decode step with M below 128 keeps its flat default) the MMA
        width buys nothing and the N tiles' wave quantization decides:
        128x50400x4096 + bias takes 394 128x128 tiles (2.66 waves of 148,
        0.99 of the ATen GEMM at ab=6) over 197 128x256 ones (1.33 waves,
        0.83).
        """
        return [
            candidate
            for bn in ((256, 128) if m <= 128 else (256,))
            if (candidate := cls._one_cta_candidate(env, k, bn=bn)) is not None
        ]

    @staticmethod
    def _one_cta_candidate(
        env: CompileEnvironment, k: int, *, bn: int
    ) -> Config | None:
        """The persistent one-CTA ``128 x bn`` tile at the deepest ring that
        fits (``_one_cta_candidates``), or ``None`` where it does not apply."""
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        bm, bk = 128, 64
        values = CuteTcgen05DefaultTileHeuristic._persistent_tile_values(
            env, k, bm=bm, bn=bn, cluster_m=1
        )
        if values is None:
            return None
        if facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk):
            return Config(**values)
        if (
            state.exact_shape_aux_kernel_detected
            and k // bk < _ONE_CTA_N_EDGE_EXACT_AUX_MIN_K_TILES
        ):
            # An exact-shape operand (a residual) read behind the full-tile
            # guard bounds the edge tiles' epilogue on few K tiles (measured
            # on the 256-wide tile; the 128-wide one keeps the same gate):
            # bf16 25216x576xK + residual, persistent 128x256 against the
            # flat 128x16 base default, K=256 88.0 vs 77.8 us, K=512 94.1 vs
            # 137.2, K=1024 100.3 vs 260.1, K=2048 126.8 vs 583.6.
            return None
        if state.aux_kernel_detected:
            values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
            )
        if not smem.one_cta_n_edge_tiles(state, values, bm=bm, bn=bn, bk=bk):
            # M-edge, K-tail and double-edge shapes never qualify; skip the
            # projection below.
            return None
        # ``default_config()`` runs the search fixups over the promoted
        # values: an auxiliary edge epilogue's C ring goes to 4 stages, and
        # the AB ring then shortens to the deepest that holds the
        # partial-output store (``_fix_one_cta_n_edge_rings_search_config``; bf16
        # 25216x576x768 + bias: ab=3). Promote the projected rings, where
        # the sample stays persistent on that store.
        projected = _rendered_values(spec, values)
        if projected.get(
            "pid_type"
        ) != "persistent_interleaved" or not smem.one_cta_n_edge_partial_output(
            state, projected, bm=bm, bn=bn, bk=bk
        ):
            return None
        values["tcgen05_ab_stages"] = projected["tcgen05_ab_stages"]
        values["tcgen05_c_stages"] = projected["tcgen05_c_stages"]
        return Config(**values)

    @classmethod
    def _pick(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config | None:
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        # Only kernels whose pipeline the compiler proved: TMA operands, and
        # the paired-pipeline allocation facts or, for the edge + K-tail
        # family, a plain dense GEMM body (``_edge_k_tail_default``).
        if not spec.tcgen05.search_enabled or state.device_sm_count <= 0:
            return None
        if not spec.tcgen05.matmul_operands_tma_provable:
            return cls._cp_async_a_default(env) if state.cp_async_a_provable else None
        if state.kernel_has_atomics:
            # Every tile below is a pair or a persistent CTA owning several
            # tiles, where codegen refuses atomics beside the warp roles
            # (``reject_atomics_beside_warp_roles``): an atomic kernel keeps
            # the flat base default (the ``cp.async`` tile above is flat, one
            # tile per CTA), as the FFI direct-entry seed does
            # (``full_tile_direct_entry_seed_eligible``).
            return None
        extents = state.matmul_compile_time_static_extents
        if extents is None or not all(isinstance(v, int) and v > 0 for v in extents):
            return None
        m, n, k = cast("tuple[int, int, int]", extents)
        batch = state.matmul_leading_work_multiplier
        small_grid = batch * cdiv(m, 128) * cdiv(n, 128) < state.device_sm_count
        # A batch of one still proves its batch mode (``((1, 0),)``), and its
        # default takes the narrow one-CTA tiles as for any batch.
        radix = _batched_radix_store(device_ir)
        if radix is not None or (batch > 1 and min(m, n) < 128 and not small_grid):
            # A head merge fused into a batched store renders from a one-CTA
            # tile's store boxes (the 128-row pairs' epilogue subtiles do
            # not; the direct-entry 256x256 pair, when its seed takes a
            # whole-tile shape before this heuristic, does), and per-batch
            # matrices narrower than 128 rows or columns are narrower than
            # every large-grid candidate below.
            return cls._narrow_batched_default(env, m, n, k, radix)
        if small_grid:
            # Fewer 128x128 tiles than SMs: the 128-row pair with the fewest
            # waves, then the most CTAs (``_small_grid_pair_default``); where
            # no pair serves, the one-CTA tile splitting K.
            pair = cls._small_grid_pair_default(env, m, n, k, batch)
            return pair if pair is not None else cls._split_k_default(env, m, n, k)
        constraints = state.cluster_m2_search_constraints
        if (
            constraints is not None
            and constraints.allow_edge_k_tail_family
            and state.dense_gemm_facts is not None
            and state.dense_gemm_facts.plain_epilogue
        ):
            # The edge + K-tail family proves its own schedule (no paired-
            # pipeline facts); the dense GEMM facts prove a plain body beside
            # the warp roles (no atomic, no side store, no scale epilogue),
            # the only one measured on it.
            return cls._edge_k_tail_default(state)
        if state.pipeline_smem_facts is None:
            return None
        # Whole 256x256 tiles already default to the FFI direct-entry pair
        # tile; replace it only where its wave quantization is clearly worse
        # (bf16 8192x768x3072: 96 pair tiles are 1.3 waves of 74 pairs).
        ffi_cost = None
        if seeds.full_tile_direct_entry_seed_eligible(state):
            ffi_cost = (
                cdiv(
                    batch * cdiv(m, 256) * cdiv(n, 256),
                    max(1, state.device_sm_count // 2),
                )
                * 128
                * 256
                * _FFI_REPLACEMENT_MARGIN
            )
        best: tuple[tuple[float, int, int], Config, int] | None = None
        best_tile = (0, 0, 0)
        one_cta = cls._one_cta_candidates(env, m, n, k)
        for seed in [
            *seeds.autotune_seed_configs(state),
            *one_cta,
            *cls._pair_m_edge_candidate(env, m, n, k),
            *seeds.dense_aligned_n_seed_configs(state),
        ]:
            values: dict[str, Any] = seed.config
            view = _promotable_seed_view(state, values)
            if view is None:
                continue
            blocks, mi, ni, ki = view
            bm, bn, bk = (cast("int", blocks[i]) for i in (mi, ni, ki))
            cluster_m = cast("int", values.get("tcgen05_cluster_m", 1))
            stages = cast("int", values.get("tcgen05_ab_stages", 2))
            # an aligned-N seed stores its physical width behind the carrier
            width = cast("int", values.get(TILE_N_KEY) or 0)
            if width:
                bn = width
                efficiency = _aligned_n_efficiency(cluster_m, bm, width)
            else:
                efficiency = _TILE_EFFICIENCY.get((cluster_m, bm, bn))
            if (
                efficiency is None
                or k % bk
                or (
                    state.cluster_m_search_choices is not None
                    and cluster_m not in state.cluster_m_search_choices
                )
            ):
                continue
            persistent = values.get("pid_type") in (
                "persistent_blocked",
                "persistent_interleaved",
            )
            # The runtime admits persistent tcgen05 bodies only on validated
            # schedules: one-CTA on static full tiles or N-edge tiles on the
            # partial-output TMA store, the role-local pair within its K-tile
            # cap (longer K is the split-K families' shape).
            if persistent and (
                (
                    cluster_m == 1
                    and not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
                    # N-edge tiles only on the validated one-CTA candidate
                    and not any(seed is candidate for candidate in one_cta)
                )
                or (cluster_m == 2 and k // bk > TCGEN05_TWO_CTA_MAX_K_TILES)
            ):
                continue
            slots = max(1, state.device_sm_count // cluster_m)
            waves = cdiv(batch * cdiv(m, bm) * cdiv(n, bn), slots)
            cost = waves * (bm // cluster_m) * bn / efficiency
            # Deeper AB rings and the 64-deep K tile (nvjet's staging) on ties.
            key = (cost, -stages, bk)
            if best is None or key < best[0]:
                best = (key, seed, waves)
                best_tile = (bm, bn, bk)
        if best is None or (ffi_cost is not None and best[0][0] >= ffi_cost):
            return None if ffi_cost is None else cls._direct_entry_staged_row(state)
        values = _promoted_values(best[1])
        if ffi_cost is not None:
            # The base default is the FFI direct-entry config; normalization
            # (``_fix_target1_tvm_ffi_search_config``) restores it unless the
            # promoted values turn its controls off.
            values[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY] = False
            values[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY] = False
            values[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = (
                Tcgen05LayoutStrategy.DEFAULT.value
            )
            for key in TCGEN05_LAYOUT_OVERRIDES_KEYS:
                values[key] = None
        pipeline_facts = state.pipeline_smem_facts
        assert pipeline_facts is not None
        many_tiles = best[2] >= _TMA_STORE_MIN_WAVES
        bm, bn, bk = best_tile
        # An output-edge epilogue with register auxiliary loads stores every
        # tile through TMA after the accumulator wait (``post_acc_wait``: the
        # partial-output store, see ``cute_mma``), where the rings hold its
        # row stages; any other placement keeps the fringe tiles, or every
        # tile, on the SIMT store (one 16-bit element per thread and
        # instruction). Inductor corpus on B200, + bias, of the ATen GEMM,
        # 256x256 pairs, SIMT store -> partial-output store: 32768x576x192
        # (6 tiles per pair) 0.50 -> 1.02, 25216x2304x768 (13) 0.69 -> 1.02,
        # 4096x30000x128 (26, the staged row) 0.79 -> 0.90, 32768x30528x128
        # (52, the staged row on the hybrid store) 0.93 -> 1.03.
        partial_output = False
        if (
            state.aux_kernel_detected
            and not values.get(TILE_N_KEY)
            and not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
        ):
            post: dict[str, Any] = {
                **values,
                TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY: (
                    TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
                ),
            }
            if smem.simt_aux_partial_output_fits(
                state,
                post,
                bm=bm,
                bn=bn,
                bk=bk,
                cluster_m=cast("int", values.get("tcgen05_cluster_m", 1)),
                ab_stages=cast("int", values.get("tcgen05_ab_stages", 2)),
                c_stages=cast("int", values.get("tcgen05_c_stages", 2)),
            ):
                values = post
                partial_output = True
        staged_row = False
        if (
            not partial_output
            and many_tiles
            and k // bk <= _EPILOGUE_BOUND_MAX_K_TILES
            and values.get("tcgen05_cluster_m", 1) == 2
            and not values.get(TILE_N_KEY)
            # a column-major output reads its rows per subtile, unstaged
            and repair.promoted_row_epilogue(state)
        ):
            # With at most two K tiles per output tile the epilogue bounds the
            # pair, and loading a bias row from global memory per subtile
            # leaves its latency on every subtile; ``pre_acc_wait`` stages the
            # promoted row once per tile (the partial-output check above
            # skips static full tiles, so they reach this too). Beside a row
            # stage the hybrid TMA store is admitted through ``c_stages_fits``
            # alone (``output_edge_tma_store_smem_fits`` refuses row stages),
            # which costs one AB stage: the deepest ring it admits that also
            # fits the stage. Inductor corpus on B200, 256x256x64 pairs + bf16
            # bias, post ab6 -> pre ab5 (measured before the partial-output
            # store took the edge tiles): 4096x30000x128 (26 tiles per pair)
            # 136 -> 98 us, 32768x30528x128 746 -> 596 us; static full
            # 256x128 tiles, post ab8 -> pre ab7: promoted bias 65536x1152x128
            # 44.3 -> 42.0 us; at K=768 the shallower ring costs 0-4%, and
            # below 16 tiles per pair the SIMT store wins (32768x576x128: 36.9
            # us SIMT, 51.3 staged beside the TMA store).
            c_stages = cast("int", values.get("tcgen05_c_stages", 2))
            for depth in range(cast("int", values.get("tcgen05_ab_stages", 2)), 1, -1):
                fits = {"bm": bm, "bn": bn, "bk": bk, "cluster_m": 2}
                if smem.c_stages_fits(
                    state, **fits, ab_stages=depth, c_stages=c_stages, has_source_c=True
                ) and smem.rowvec_aux_stage_fits(
                    state, **fits, ab_stages=depth, c_stages=c_stages
                ):
                    values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                        TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
                    )
                    values["tcgen05_ab_stages"] = depth
                    staged_row = True
                    break
        if (
            many_tiles
            and not partial_output
            and not staged_row
            and values.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
            == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            and pipeline_facts.rows
            and all(row.itemsize == 2 for row in pipeline_facts.rows)
            and not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
        ):
            # On an output edge a ``pre_acc_wait`` bias row the partial-output
            # store does not fit (five or more unpromoted 16-bit rows: 512 B a
            # row beside the 256x256x64 pair's six stages) keeps every tile on
            # the predicated SIMT store (``output_edge_tma_store_smem_fits``
            # refuses the hybrid TMA store beside row stages); ``post_acc_wait``
            # stores the full tiles through TMA. The TMA store wins from 16
            # tiles per CTA (pair) on: ``acc + r0..r4[cols]`` 16384x30528x768
            # 3284 -> 2539 us; 256x256x64 ab6 pairs + bf16 bias, of cuBLAS,
            # SIMT -> full-tile TMA store: 104 tiles per pair
            # (16384x30528x768) 0.63 -> 0.92, 26 (4096x30000x128) 0.41 ->
            # 0.52; 13 (25216x2304x768) 0.67 -> 0.61, 6 (32768x576x192) 0.50
            # -> 0.29.
            values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
            )
        waves = best[2]
        staged_aux = False
        if (
            surface.epi_tma_search_enabled(state)
            # A math-bound epilogue gains nothing from the staged loads and
            # loses an AB stage to their ring (GELU backward's erf and exp,
            # 16384x4096x1024: 0.33 of the ATen GEMM on SIMT loads, 0.29
            # staged).
            and not pipeline_facts.epilogue_transcendental
            and values.get("tcgen05_cluster_m", 1) == 2
            and not values.get(TILE_N_KEY)
            and facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
            and (staged := cls._epi_tma_aux_values(env, values, k=k)) is not None
        ):
            # The ring may narrow the tile: the grouping below reads its tiles.
            values = staged
            staged_aux = True
            view = facts._matmul_config_view(state, values)
            assert view is not None
            bn = cast("int", view[0][view[2]])
            waves = cdiv(
                batch * cdiv(m, bm) * cdiv(n, bn), max(1, state.device_sm_count // 2)
            )
        if state.aux_kernel_detected and facts._has_any_matmul_fact_edge_tile(
            state, values
        ):
            # An output-edge auxiliary GEMM walks its tiles in groups of M
            # tiles: 32 on long N sweeps (at least 32 N tiles: a group's A,
            # 12.6 MB at K=768, stays resident in L2 while the group sweeps N,
            # so B is re-read once per 32 M tiles), 4 on short ones (a wave
            # then covers every N tile of a few M tiles); a group as tall as
            # the M tiles decodes the plain raster. Inductor corpus on B200, +
            # bias, of the ATen GEMM, plain / grouped (4) / grouped (8) /
            # grouped (32): 16384x30528x768 0.96 / - / 0.94 / 0.98,
            # 32768x30528x768 0.86 / - / 0.94 / 0.96; 25216x2304x768 (9 N
            # tiles) 1.02 / 1.10 / 1.09 / 1.06, 32768x576x192 (3) 1.02 / 1.07
            # / 1.06 / -. Not measured past K=768, where a group's A panel
            # outgrows L2 (67 MB at K=4096).
            assert state.matmul_block_ids is not None
            groupings: list[object] = list(
                values.get("l2_groupings") or [1] * len(spec.l2_groupings)
            )
            grouping = long_n_sweep_l2_grouping(
                cdiv(n, bn), short=_SHORT_N_SWEEP_L2_GROUPING
            )
            groupings[
                spec.l2_groupings.block_id_to_index(state.matmul_block_ids[0])
            ] = grouping if cdiv(m, bm) > grouping else 1
            values["l2_groupings"] = groupings
        elif (
            waves > 1
            and len(spec.l2_groupings) == 1
            and values.get("l2_groupings", [1])
            in ([1], [TCGEN05_TWO_CTA_SEED_L2_GROUPING])
        ):
            # The proved-schedule seeds walk the grid ungrouped or by the
            # seed grouping; a multi-wave default needs the grouped raster so
            # concurrent tiles share their A/B panels in L2 (bf16
            # 10240x2560x4096 on 256x128 pair tiles: 0.80 -> 1.05 of cuBLAS),
            # and a long N sweep groups 32 M tiles
            # (``long_n_sweep_l2_grouping``; plain 16384x30528x768, which
            # takes this branch: 0.92 grouped by 4, 0.96 by 32; the direct
            # entry's 8192x32000x1024 + bias: 0.95 by 8, 1.00 by 32): the
            # group's A panels stay resident in L2. Staged exact-shape operands stream through
            # L2 beside them, so such a sweep groups 8 (eleven residuals on
            # 256x128 pairs, grouped by 32 / 8: 16384x4096x4096 0.74 / 0.85,
            # 4096x16384x4096 0.80 / 0.83, 4096^3 0.85 / 0.86; three, 4096^3:
            # 1.01 / 1.00).
            values["l2_groupings"] = [
                long_n_sweep_l2_grouping(
                    cdiv(n, bn),
                    short=TCGEN05_TWO_CTA_SEED_L2_GROUPING,
                    long=_STAGED_AUX_LONG_N_SWEEP_L2_GROUPING
                    if staged_aux
                    else TCGEN05_LONG_N_SWEEP_L2_GROUPING,
                )
            ]
        if pipeline_facts.epilogue_transcendental:
            values = cls._transcendental_epilogue_values(env, values)
        return Config(**values)

    @staticmethod
    def _transcendental_epilogue_values(
        env: CompileEnvironment, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Two epilogue warp groups for an epilogue with a transcendental op
        (``epilogue_transcendental``: erf, exp, tanh, ...), at the deepest AB
        ring that fits beside their C rings, where the search fixups keep
        them (the standard static-full TMA-store epilogue).

        One warp per scheduler issues the exact erf/exp polynomials one
        dependent step at a time, so such an epilogue outlasts the tile's
        MMA; a second group overlaps two subtiles' math. B200, GELU backward
        (erf, exp), 16384x4096x1024 on the 256x256 pair, of the ATen GEMM:
        one group 0.33, two 1.02 (ab 6 -> 5; ab 4 1.03, three groups at ab 4
        0.75, four at ab 3 0.64)."""
        spec = env.config_spec
        two = {**values, TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY: 2}
        projected = _rendered_values(spec, two)
        if projected.get(TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY) != 2:
            return values
        two["tcgen05_ab_stages"] = projected["tcgen05_ab_stages"]
        two["tcgen05_c_stages"] = projected["tcgen05_c_stages"]
        return two
