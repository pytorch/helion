"""Autotune-off default tile for dense tcgen05 GEMMs.

Inductor's native-matmul path runs Helion kernels with
``autotune_effort="none"``, so the default config is what such a GEMM runs.
Among the compiler's proved tcgen05 seeds (plus the one-CTA 128x256 tile on
whole tiles, and the 16-bit dense aligned-N pairs at their physical widths)
this heuristic promotes the one whose tile minimizes wave-quantized MMA
time; whole 256x256-tile shapes keep the FFI direct-entry
default unless a candidate is clearly cheaper. Plain GEMMs only the edge +
K-tail family runs (a 16-bit K tail on whole M and N tiles, or a K tail with
ragged M and N) take that family's CLC pair seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ....._utils import cdiv
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...dense_aligned_n import TILE_N_KEY
from ...strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ...strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ...strategies import Tcgen05LayoutStrategy
from ...tcgen05_config import facts
from ...tcgen05_config import seeds
from ...tcgen05_config import smem
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
from ...tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
from ...tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
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
# the hybrid TMA store (``post_acc_wait``) over the SIMT store (see ``_pick``).
_TMA_STORE_MIN_WAVES = 16
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
    def _one_cta_candidates(
        env: CompileEnvironment, m: int, n: int, k: int
    ) -> list[Config]:
        """The one-CTA 128x256 persistent tile on whole tiles.

        The proved-schedule seeds carry it only when they had to restore the
        persistent pid types; a 128-row tile still quantizes M=25216 into
        whole waves where the 256-row pair tile leaves a half-empty last wave
        (0.98 vs 0.90 of cuBLAS).  The persistent one-CTA runtime is validated
        for static full tiles only.
        """
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        bm, bn, bk = 128, 256, 64
        fragments = facts._matmul_block_fragments(state)
        if (
            not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
            # a registered block size (``hl.register_block_size``) bounds the
            # tile as it bounds the seeds
            or fragments is None
            or not all(
                fragment.low <= value <= fragment.high
                for fragment, value in zip(fragments, (bm, bn, bk), strict=True)
            )
            or "persistent_interleaved" not in spec.allowed_pid_types
            or (
                state.cluster_m_search_choices is not None
                and 1 not in state.cluster_m_search_choices
            )
        ):
            return []
        block_sizes = facts._matmul_seed_block_sizes(state, bm=bm, bn=bn, bk=bk)
        stages = smem.max_ab_stages_that_fit(state, bm=bm, bn=bn, bk=bk, cluster_m=1)
        if block_sizes is None or stages < 2:
            return []
        return [
            Config(
                block_sizes=block_sizes,
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=1,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=min(stages, k // bk),
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
            )
        ]

    @classmethod
    def _pick(cls, env: CompileEnvironment) -> Config | None:
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        # Only kernels whose pipeline the compiler proved: TMA operands, and
        # the paired-pipeline allocation facts or, for the edge + K-tail
        # family, a plain dense GEMM body (``_edge_k_tail_default``).
        if (
            not spec.tcgen05.search_enabled
            or state.device_sm_count <= 0
            or not spec.tcgen05.matmul_operands_tma_provable
        ):
            return None
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
        for seed in [
            *seeds.autotune_seed_configs(state),
            *cls._one_cta_candidates(env, m, n, k),
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
            # schedules: one-CTA on static full tiles, the role-local pair
            # within its K-tile cap (longer K is the split-K families' shape).
            if persistent and (
                (
                    cluster_m == 1
                    and not facts._static_full_tiles(state, bm=bm, bn=bn, bk=bk)
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
            return None
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
        if (
            best[2] >= _TMA_STORE_MIN_WAVES
            and values.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY)
            == TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT
            and pipeline_facts.rows
            and all(row.itemsize == 2 for row in pipeline_facts.rows)
            and not facts._static_full_tiles(
                state, bm=best_tile[0], bn=best_tile[1], bk=best_tile[2]
            )
        ):
            # On an output edge a ``pre_acc_wait`` bias row keeps every tile
            # on the predicated SIMT store (``output_edge_tma_store_smem_fits``
            # refuses the hybrid full-tile TMA store beside row stages);
            # ``post_acc_wait`` stores the full tiles through TMA. The TMA
            # store wins from 16 tiles per CTA (pair) on, the SIMT store
            # below. Inductor corpus, 256x256x64 ab6 pairs + bf16 bias, of
            # cuBLAS, SIMT -> TMA store: 104 tiles per pair
            # (16384x30528x768) 0.63 -> 0.92, 26 (4096x30000x128) 0.41 ->
            # 0.52; 13 (25216x2304x768) 0.67 -> 0.61, 6 (32768x576x192)
            # 0.50 -> 0.29, 4 (2048x8008x2560) 1.06 -> 0.86.
            values[TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY] = (
                TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT
            )
        if (
            best[2] > 1
            and len(spec.l2_groupings) == 1
            and values.get("l2_groupings", [1]) == [1]
        ):
            # The proved-schedule seeds walk the grid ungrouped; a multi-wave
            # default needs the grouped raster so concurrent tiles share their
            # A/B panels in L2 (bf16 10240x2560x4096 on 256x128 pair tiles:
            # 0.80 -> 1.05 of cuBLAS).
            values["l2_groupings"] = [TCGEN05_TWO_CTA_SEED_L2_GROUPING]
        return Config(**values)
