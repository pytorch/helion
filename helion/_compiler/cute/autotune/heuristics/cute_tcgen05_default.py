"""Autotune-off default tile for dense tcgen05 GEMMs.

Inductor's native-matmul path runs Helion kernels with
``autotune_effort="none"``, so the default config is what such a GEMM runs.
Among the compiler's proved tcgen05 seeds (plus the one-CTA 128x256 tile on
whole or N-edge tiles -- 128x128 too for a single row of M tiles -- and the
16-bit dense aligned-N pairs at their physical widths) this heuristic promotes the one whose tile minimizes wave-quantized
MMA time; whole 256x256-tile shapes keep the FFI direct-entry pair unless a
candidate is clearly cheaper, with its promoted bias rows staged
(``pre_acc_wait``) where the heuristic promotes that seed. Plain GEMMs only
the edge + K-tail family runs (a 16-bit K tail on whole M and N tiles, or a K
tail with ragged M and N) take that family's CLC pair seed.
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
from ...tcgen05_config import repair
from ...tcgen05_config import seeds
from ...tcgen05_config import smem
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ...tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_LONG_N_SWEEP_L2_GROUPING
from ...tcgen05_constants import TCGEN05_LONG_N_SWEEP_MIN_TILES
from ...tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ...tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ...tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ...tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR
    from ...tcgen05_config import CuteTcgen05Config

# Per-SM MMA efficiency of a tile relative to the CtaGroup.TWO 256x256 tile,
# whose pair shares one B tile across both CTAs; narrower tiles move more
# operand bytes per MMA.  Calibrated on B200 bf16 inductor GEMMs
# (16384x384x768: 256x128 pair 0.89 vs 256x256 pair 0.68 of cuBLAS;
# 25216x768x3072: one-CTA 128x256 0.98 vs 256x256 pair 0.90).  Tiles below a
# 128x128 per-SM footprint are not default candidates: their operand traffic,
# not wave quantization, decides their speed.
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
    default's 0.85. Narrower widths are not default candidates (16384x960x1024:
    the 96-wide pair runs 40-42 us against 38 for the 256x256 one), nor is
    the FP8 family's 128-row pair (64 rows per CTA: below the per-SM floor)."""
    if (cluster_m, bm) != (2, TCGEN05_TWO_CTA_BLOCK_M) or width < 128:
        return None
    low = _TILE_EFFICIENCY[(2, TCGEN05_TWO_CTA_BLOCK_M, 128)]
    high = _TILE_EFFICIENCY[(2, TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N)]
    return low + (high - low) * (width - 128) / (TCGEN05_TWO_CTA_BLOCK_N - 128)


# Tiles per CTA (waves) from which an output-edge bias GEMM's default takes
# the hybrid TMA store (``post_acc_wait``, or a staged ``pre_acc_wait`` row on
# epilogue-bound pairs) over the SIMT store (see ``_pick``).
_TMA_STORE_MIN_WAVES = 16
# K tiles per output tile up to which the epilogue bounds a pair GEMM, so its
# default stages promoted bias rows (``pre_acc_wait``; see ``_pick``).
_EPILOGUE_BOUND_MAX_K_TILES = 2
# An output-edge auxiliary GEMM's default grouping on sweeps shorter than
# the long-N-sweep grouping's (``TCGEN05_LONG_N_SWEEP_*``; see ``_pick``).
_SHORT_N_SWEEP_L2_GROUPING = 4
# K tiles from which the one-CTA N-edge candidates take an exact-shape
# auxiliary epilogue (see ``_one_cta_candidate``).
_ONE_CTA_N_EDGE_EXACT_AUX_MIN_K_TILES = 8
# A candidate replaces the FFI default only below this fraction of its cost.
_FFI_REPLACEMENT_MARGIN = 0.9


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
        return cls._pick(env) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        return cls._pick(env)

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
        values: dict[str, Any] = dict(seed.config)
        # PDL stays a target decision of the base default.
        values.pop("cute_pdl", None)
        return Config(**values)

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
        return Config(
            block_sizes=block_sizes,
            pid_type="flat",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=min(stages, k // bk),
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
        )

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
        fragments = facts._matmul_block_fragments(state)
        if (
            "persistent_interleaved" not in spec.allowed_pid_types
            or (
                state.cluster_m_search_choices is not None
                and 1 not in state.cluster_m_search_choices
            )
            # a registered block size (``hl.register_block_size``) bounds the
            # tile as it bounds the seeds
            or fragments is None
            or not all(
                fragment.low <= value <= fragment.high
                for fragment, value in zip(fragments, (bm, bn, bk), strict=True)
            )
        ):
            return None
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        stages = smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
        if block_sizes is None or stages < 2:
            return None
        deepest = min(stages, k // bk)
        values: dict[str, Any] = {
            "block_sizes": block_sizes,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": deepest,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
        }
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
        projected = {**spec._base_default_config().config, **values}
        spec.normalize(projected, _fix_invalid=True)
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
    def _pick(cls, env: CompileEnvironment) -> Config | None:
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        # Only kernels whose pipeline the compiler proved: TMA operands, and
        # the paired-pipeline allocation facts or, for the edge + K-tail
        # family, a plain dense GEMM body (``_edge_k_tail_default``).
        if not spec.tcgen05.search_enabled or state.device_sm_count <= 0:
            return None
        if not spec.tcgen05.matmul_operands_tma_provable:
            return cls._cp_async_a_default(env) if state.cp_async_a_provable else None
        extents = state.matmul_compile_time_static_extents
        if extents is None or not all(isinstance(v, int) and v > 0 for v in extents):
            return None
        m, n, k = cast("tuple[int, int, int]", extents)
        batch = state.matmul_leading_work_multiplier
        if batch * cdiv(m, 128) * cdiv(n, 128) < state.device_sm_count:
            # Fewer 128x128 tiles than SMs: per-CTA latency, not wave
            # quantization, decides the tile (the small-grid seeds' regime).
            return None
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
            *seeds.dense_aligned_n_seed_configs(state),
        ]:
            values: dict[str, Any] = seed.config
            if (
                values.get("tcgen05_cluster_n", 1) != 1
                or values.get("tcgen05_persistence_model", "static_persistent")
                != "static_persistent"
                or values.get("tcgen05_strategy", "role_local_monolithic")
                != "role_local_monolithic"
                or values.get("pid_type") not in spec.allowed_pid_types
            ):
                continue
            view = facts._matmul_config_view(state, values)
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
        values = dict(best[1].config)
        # PDL stays a target decision of the base default.
        values.pop("cute_pdl", None)
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
            # promoted row once per tile. Beside a row stage the hybrid TMA
            # store is admitted through ``c_stages_fits`` alone
            # (``output_edge_tma_store_smem_fits`` refuses row stages, see
            # below), which costs one AB stage: the deepest ring it admits
            # that also fits the stage. Inductor corpus on B200, 256x256x64
            # pairs + bf16 bias, post ab6 -> pre ab5: 4096x30000x128 (26
            # tiles per pair) 136 -> 98 us, 32768x30528x128 746 -> 596 us; at
            # K=768 the shallower ring costs 0-4%, and below 16 tiles per pair
            # the SIMT store wins (32768x576x128: 36.9 us SIMT, 51.3 staged
            # beside the TMA store).
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
            # On an output edge a ``pre_acc_wait`` bias row at the seed's deep
            # ring keeps every tile on the predicated SIMT store
            # (``output_edge_tma_store_smem_fits`` refuses the hybrid TMA
            # store beside row stages, and ``c_stages_fits`` does not admit it
            # at that ring); ``post_acc_wait`` stores every tile through TMA
            # where the arena holds the row stage (the partial-output store,
            # ``cute_mma``), else the full tiles. Measured against the full-
            # tile split, before the partial-output store took the fringe
            # tiles too, the TMA store won from 16 tiles per CTA (pair) on and
            # the SIMT store below. Inductor corpus, 256x256x64 ab6 pairs +
            # bf16 bias, of cuBLAS, SIMT -> full-tile TMA store:
            # 104 tiles per pair (16384x30528x768) 0.63 -> 0.92, 26
            # (4096x30000x128) 0.41 -> 0.52; 13 (25216x2304x768) 0.67 -> 0.61,
            # 6 (32768x576x192) 0.50 -> 0.29, 4 (2048x8008x2560) 1.06 -> 0.86.
            values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
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
            grouping = (
                TCGEN05_LONG_N_SWEEP_L2_GROUPING
                if cdiv(n, bn) >= TCGEN05_LONG_N_SWEEP_MIN_TILES
                else _SHORT_N_SWEEP_L2_GROUPING
            )
            groupings[
                spec.l2_groupings.block_id_to_index(state.matmul_block_ids[0])
            ] = grouping if cdiv(m, bm) > grouping else 1
            values["l2_groupings"] = groupings
        elif (
            best[2] > 1
            and len(spec.l2_groupings) == 1
            and values.get("l2_groupings", [1]) == [1]
        ):
            # The proved-schedule seeds walk the grid ungrouped; a multi-wave
            # default needs the grouped raster so concurrent tiles share their
            # A/B panels in L2 (bf16 10240x2560x4096 on 256x128 pair tiles:
            # 0.80 -> 1.05 of cuBLAS), and a long N sweep groups 32 M tiles
            # (bf16 8192x32000x1024 + bias on 256x256 pairs: 0.95 grouped by
            # 8, 1.00 by 32).
            values["l2_groupings"] = [
                TCGEN05_LONG_N_SWEEP_L2_GROUPING
                if cdiv(n, bn) >= TCGEN05_LONG_N_SWEEP_MIN_TILES
                else TCGEN05_TWO_CTA_SEED_L2_GROUPING
            ]
        return Config(**values)
