"""Autotune-off default tile for dense tcgen05 GEMMs.

Inductor's native-matmul path runs Helion kernels with
``autotune_effort="none"``, so the default config is what such a GEMM runs.
Among the compiler's proved tcgen05 seeds (plus the one-CTA 128x256 tile on
whole tiles) this heuristic promotes the one whose tile minimizes
wave-quantized MMA time; whole 256x256-tile shapes keep the FFI direct-entry
default unless a candidate is clearly cheaper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ..._utils import cdiv
from ...runtime.config import Config
from ..cute.strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from ..cute.strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..cute.strategies import Tcgen05LayoutStrategy
from ..cute.tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

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


# A candidate replaces the FFI default only below this fraction of its cost.
_FFI_REPLACEMENT_MARGIN = 0.9


class CuteTcgen05DefaultTileHeuristic(AutotunerHeuristic):
    name = "cute_tcgen05_default_tile"
    backend = "cute"
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
        if (
            not state._static_full_tiles(bm=bm, bn=bn, bk=bk)
            or "persistent_interleaved" not in spec.allowed_pid_types
            or (
                state.cluster_m_search_choices is not None
                and 1 not in state.cluster_m_search_choices
            )
        ):
            return []
        block_sizes = state._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
        stages = state.max_ab_stages_that_fit(bm=bm, bn=bn, bk=bk, cluster_m=1)
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
        # Only kernels whose dense pipeline the compiler proved (TMA operands,
        # allocation facts): anything else beside the warp roles (a side
        # store, an unprovable operand) refuses the persistent schedules.
        if (
            not spec.cute_tcgen05_search_enabled
            or state.device_sm_count <= 0
            or state.pipeline_smem_facts is None
            or not spec.cute_tcgen05_matmul_operands_tma_provable
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
        # Whole 256x256 tiles already default to the FFI direct-entry pair
        # tile; replace it only where its wave quantization is clearly worse
        # (bf16 8192x768x3072: 96 pair tiles are 1.3 waves of 74 pairs).
        ffi_cost = None
        if state.full_tile_direct_entry_seed_eligible():
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
        for seed in [
            *state.autotune_seed_configs(),
            *cls._one_cta_candidates(env, m, n, k),
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
            view = state._matmul_config_view(values)
            if view is None:
                continue
            blocks, mi, ni, ki = view
            bm, bn, bk = (cast("int", blocks[i]) for i in (mi, ni, ki))
            cluster_m = cast("int", values.get("tcgen05_cluster_m", 1))
            stages = cast("int", values.get("tcgen05_ab_stages", 2))
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
                (cluster_m == 1 and not state._static_full_tiles(bm=bm, bn=bn, bk=bk))
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
