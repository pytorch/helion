from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ...runtime.config import Config
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..cute.tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_fragment import BlockSizeFragment
    from ...autotuner.config_spec import BlockSizeSpec
    from ...autotuner.config_spec import ReductionLoopSpec
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _reduction_kernel_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    spec = env.config_spec
    # Single non-reduction tile + single reduction dim.
    if len(spec.block_sizes) != 1 or len(spec.reduction_loops) != 1:
        return False
    # No matmul facts (this seeds reduction kernels, not GEMMs).
    if spec.matmul_facts:
        return False
    bs_spec = cast("BlockSizeSpec", spec.block_sizes[0])
    # M-axis must accept block_size=1.
    return max(bs_spec.min_size, bs_spec.autotuner_min) <= 1


class CuteReductionTileHeuristic(AutotunerHeuristic):
    """Seed config for canonical reduction kernels (RMS norm, softmax, etc.).

    Seeds the "narrow chunk" config: bs=1, nt=1, reduction_loops=[None] for
    N<=max_threads (single-pass persistent reduction) or
    reduction_loops=[max_threads] for N>max_threads (one element per
    thread per iter, no lane loop). This config keeps the M-axis at one
    row per block so the reduction recruits all available threads, and
    the two-pass load fusion (helion/_compiler/cute/fuse_two_pass_loads.py)
    eliminates the redundant gmem reload of x in the post-reduction sweep.
    """

    name = "cute_reduction_tile"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return _reduction_kernel_eligible(env, device_ir)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        rl_spec = cast("ReductionLoopSpec", spec.reduction_loops[0])
        max_threads = spec.max_reduction_threads or 1024
        size_hint = rl_spec.size_hint
        if size_hint <= max_threads:
            # Persistent reduction (no roll). The normalize step will keep
            # reduction_loops[0]=None when the M-axis allows it.
            reduction_loops: list[int | None] = [None]
        else:
            reduction_loops = [max_threads]
        return Config(
            block_sizes=[1],
            num_threads=[1],
            reduction_loops=reduction_loops,
        )


class CuteReductionWideChunkHeuristic(AutotunerHeuristic):
    """Companion seed: chunk = max(max_threads, size_hint/2) so the inner
    reduction loop has very few outer iterations and lane_extent absorbs
    the bulk of the work.

    For large N this lattice (1-2 outer iters, large lane_extent) tends to
    schedule better than the narrow-chunk lattice (many outer iters,
    lane_extent=1) on B200 — the SASS is dominated by per-iter scheduling
    bubbles in the narrow case, while the wide case lets the compiler
    overlap the load/compute/store traffic across the unrolled lane
    iterations. Only applies when size_hint > max_threads (otherwise the
    narrow heuristic already gives the same lattice).
    """

    name = "cute_reduction_wide_chunk"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _reduction_kernel_eligible(env, device_ir):
            return False
        spec = env.config_spec
        rl_spec = cast("ReductionLoopSpec", spec.reduction_loops[0])
        max_threads = spec.max_reduction_threads or 1024
        return rl_spec.size_hint > 2 * max_threads

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        rl_spec = cast("ReductionLoopSpec", spec.reduction_loops[0])
        max_threads = spec.max_reduction_threads or 1024
        size_hint = rl_spec.size_hint
        # Halve the size_hint until it fits in the reduction_loop chunk
        # spec's [low, high] range; the autotuner explores power-of-2
        # chunks so we keep this seed PoT.
        chunk = size_hint // 2
        if chunk < max_threads:
            chunk = max_threads
        return Config(
            block_sizes=[1],
            num_threads=[1],
            reduction_loops=[chunk],
        )


class CuteTcgen05ClusterM2Heuristic(AutotunerHeuristic):
    name = "cute_tcgen05_cluster_m2"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        constraints = spec._tcgen05_cluster_m2_search_constraints
        if constraints is None:
            return False
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in spec.allowed_pid_types:
            return False
        if len(spec.block_sizes) != 3:
            return False

        bm_fragment = cast("BlockSizeFragment", spec.block_sizes[0]._fragment(spec))
        bn_fragment = cast("BlockSizeFragment", spec.block_sizes[1]._fragment(spec))
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        m_tile_reachable = (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            # The edge+K-tail surface keeps the flat M search capped below 256,
            # then normalization projects cluster_m=2 candidates to 256.
            or (edge_k_tail_family and bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M)
        )
        n_tile_reachable = (
            bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
            or (edge_k_tail_family and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N)
        )
        return m_tile_reachable and n_tile_reachable and cls._select_bk(env) is not None

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        spec = env.config_spec
        bk = cls._select_bk(env)
        if bk is None:
            raise AssertionError(f"{cls.name} get_seed_config called while ineligible")

        edge_k_tail_family = (
            spec._tcgen05_cluster_m2_search_constraints is not None
            and spec._tcgen05_cluster_m2_search_constraints.allow_edge_k_tail_family
        )
        seed: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            # Matches the validated tcgen05 search restriction.
            "tcgen05_num_epi_warps": 4,
        }
        if edge_k_tail_family:
            seed.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
        else:
            seed["l2_groupings"] = [TCGEN05_TWO_CTA_SEED_L2_GROUPING]
            # When the SMEM-budget gate admits ``ab=3`` for this seed tile
            # shape, seed the canonical 4096^3 fast config family directly so
            # it reaches the autotuner's initial population without depending
            # on a search-stage mutation.
            if spec._tcgen05_ab_stages_three_fits(
                bm=TCGEN05_TWO_CTA_BLOCK_M,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=bk,
                cluster_m=2,
            ):
                seed["tcgen05_ab_stages"] = 3
        if spec.indexing.length == 3:
            # Pure matmul has exactly the A/B/C indexing slots. Fused epilogues
            # add more memory ops, so leave those seeds to the spec default
            # rather than constructing a partial list.
            seed["indexing"] = [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ]
        elif edge_k_tail_family:
            seed["indexing"] = ["tensor_descriptor"] * spec.indexing.length
        return Config(**seed)

    @staticmethod
    def _select_bk(env: CompileEnvironment) -> int | None:
        spec = env.config_spec
        constraints = spec._tcgen05_cluster_m2_search_constraints
        if constraints is None or len(spec.block_sizes) != 3:
            return None
        bk_fragment = cast("BlockSizeFragment", spec.block_sizes[2]._fragment(spec))
        if constraints.allow_edge_k_tail_family:
            if (
                bk_fragment.low
                <= TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
                <= bk_fragment.high
                and spec._tcgen05_cluster_m2_bk_is_valid(
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                    constraints,
                )
            ):
                return TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
            return None
        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if spec._tcgen05_cluster_m2_bk_is_valid(bk, constraints):
                return bk
            bk //= 2
        return None
