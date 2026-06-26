from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ...runtime.config import Config
from ..cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..cute.strategies import Tcgen05PersistenceModel
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..cute.tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides
from .common import is_canonical_row_reduction
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_fragment import BlockSizeFragment
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import ReductionLoopSpec
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _cute_seed_vec_width(
    env: CompileEnvironment,
    rl_spec: ReductionLoopSpec,
    max_threads: int,
    size_hint: int,
    device_ir: DeviceIR,
) -> int:
    """Pick a default vector width for the cute_vector_widths seed.

    Returns 1 (scalar) when the kernel has no plausible LDG.128 win, or
    the dtype is not supported by the vector load helper. For supported
    dtypes, prefer 4 (fp32) / 8 (fp16/bf16) when the reduction is wide
    enough that a vec-load actually halves the number of inner-loop
    iters.  For fp16/bf16 the V=8 seed is sampled even when the picked
    tile doesn't exactly match the seed size_hint — the per-block
    strategy at construction time validates ``EPT % V == 0`` and falls
    back to V=1 if the chosen lattice can't fit V=8, so over-seeding is
    safe and lets the hill-climber discover the V>1 lattices that lift
    softmax-style reductions from scalar LDG to LDG.64/LDG.128.
    """
    spec = env.config_spec
    if not spec.cute_vector_widths.valid_block_ids():
        return 1
    if size_hint < max_threads:
        # Reduction extent barely fits in one wide chunk; vec wouldn't
        # remove enough loop iters to matter.
        return 1
    # Find the dtype of the reduction-source tensor by walking nodes
    # that have a fake-tensor value matching the reduction extent.
    dtype: torch.dtype | None = None
    rdim_size = rl_spec.size_hint
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                last = val.shape[-1]
                if isinstance(last, int) and last == rdim_size:
                    dtype = val.dtype
                    break
        if dtype is not None:
            break
    if dtype is torch.float32:
        return 4
    if dtype in (torch.float16, torch.bfloat16):
        return 8
    return 1


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
        return is_canonical_row_reduction(env)

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
        seed: dict[str, Any] = {
            "block_sizes": [1],
            "num_threads": [1],
            "reduction_loops": reduction_loops,
        }
        vec = _cute_seed_vec_width(env, rl_spec, max_threads, size_hint, device_ir)
        if vec > 1:
            seed["cute_vector_widths"] = [vec]
        return Config(**seed)


def _cute_tile_seed_vec_width_for_dtype(dtype: torch.dtype | None) -> int:
    """V seed for ``CuteNDTileStrategy`` lane-loop vec on a given dtype.

    Returns 4 for fp32 (LDG.128 = 16 bytes), 4 for fp16/bf16 (LDG.64,
    8 bytes per thread per outer iter).  Note: V=8 for fp16/bf16 IS now
    supported via a 2x V=4 split (see
    ``_cute_register_tile_unroll_vec_hoist_split2``), but the autotuner
    seed stays at 4 because the split emits two LDG.64s rather than a
    single LDG.128 — empirically the per-element bookkeeping in the 8-
    iter constexpr V-loop and the doubled fuser cache offset the
    extra bytes-per-load.  The split path stays available as a
    reachable point in the autotuner's V search space for shapes where
    it does win.
    """
    if dtype is torch.float32:
        return 4
    if dtype in (torch.float16, torch.bfloat16):
        return 4
    return 1


def _cute_tile_inner_block_dtype(
    env: CompileEnvironment, device_ir: DeviceIR, block_id: int
) -> torch.dtype | None:
    """Walk the device graphs to find the dtype of any tensor whose
    LAST dim corresponds to ``block_id``'s tile.  Used to seed the per-
    block vec width when the kernel has no rolled reduction (e.g.
    softmax_two_pass — its two ``hl.tile`` loops over the reduction
    axis don't go through the ``ReductionLoopSpec`` path)."""
    bs = env.block_sizes[block_id]
    block_numel = bs.numel
    try:
        block_numel_int = int(block_numel)
    except (TypeError, ValueError):
        return None
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                last = val.shape[-1]
                if isinstance(last, int) and last == block_numel_int:
                    return val.dtype
    return None


class CuteTileVecHeuristic(AutotunerHeuristic):
    """Seed config for canonical tile kernels (softmax_two_pass etc.)
    that drive their own explicit tile loop over the reduction axis.

    Seeds the "wide reduction + per-thread vec" config: block_size=
    [1, R] on the M and reduction axes (1 row per grid block), num_threads
    sized so each thread owns V contiguous elements (lane_extent = V),
    cute_vector_widths=[1, V].  This lets the strategy hoist a single
    LDG.64 / LDG.128 per outer-tile iter, lifting the kernel from scalar
    LDG bandwidth.

    The heuristic fires when:
    * The kernel has exactly 2 tile blocks (one outer row tile + one
      inner reduction tile), no matmul facts, no rolled reductions.
    * The inner tile has a stride-1 fp16/bf16/fp32 source tensor.
    """

    name = "cute_tile_vec"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        if spec.matmul_facts:
            return False
        # Two-tile pattern: outer row + inner reduction (no rolled
        # reductions registered — those use the
        # CuteReductionTileHeuristic seed instead).
        if len(spec.block_sizes) != 2 or spec.reduction_loops:
            return False
        # The inner tile block must have a vec slot registered
        # (added by ``register_rollable_reductions`` for cute tile blocks).
        inner_block_id = (
            cast("Any", spec.block_sizes[1]).block_ids[0]
            if hasattr(cast("Any", spec.block_sizes[1]), "block_ids")
            else None
        )
        if inner_block_id is None:
            return False
        if inner_block_id not in spec.cute_vector_widths.valid_block_ids():
            return False
        # Need a recognisable dtype to seed V.
        dtype = _cute_tile_inner_block_dtype(env, device_ir, inner_block_id)
        return _cute_tile_seed_vec_width_for_dtype(dtype) > 1

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        inner_block_id = (
            cast("Any", spec.block_sizes[1]).block_ids[0]
            if hasattr(cast("Any", spec.block_sizes[1]), "block_ids")
            else None
        )
        if inner_block_id is None:
            return None
        dtype = _cute_tile_inner_block_dtype(env, device_ir, inner_block_id)
        vec = _cute_tile_seed_vec_width_for_dtype(dtype)
        if vec <= 1:
            return None
        # Pick a reasonable inner block_size: prefer 1024 (matches
        # SM100 warp + L2 hit cadence) when reachable, else cap at the
        # inner tile's fragment.high.  ``num_threads`` is sized so the
        # lane_extent equals V (one vec load per thread per outer iter).
        bn_fragment = cast("Any", spec.block_sizes[1])._fragment(spec)
        bn_high = getattr(bn_fragment, "high", None)
        block_n: int
        if isinstance(bn_high, int):
            block_n = 1024 if bn_high >= 1024 else max(bn_high, vec)
        else:
            block_n = 1024
        # Threads = block_n // V so each thread owns V contiguous elts.
        nt_n = max(1, block_n // vec)
        seed: dict[str, Any] = {
            "block_sizes": [1, block_n],
            "num_threads": [0, nt_n],
            "cute_vector_widths": [1, vec],
        }
        try:
            return Config(**seed)
        except Exception:
            return None


class CuteTileVecWarpReduceHeuristic(AutotunerHeuristic):
    """Sibling seed for ``CuteTileVecHeuristic`` favouring a warp-sized
    thread block over the wider 1024-thread lattice.

    For tile kernels that reduce across the inner axis per outer-tile
    iter (softmax_two_pass, RMS norm, etc.), the cross-thread combine
    is the dominant cost. With ``num_threads <= 32`` the reduction
    strategy lowers to ``cute.arch.warp_reduction`` (one warp-shuffle,
    no shared memory, no CTA-wide barrier); with ``num_threads > 32``
    it lowers to ``_cute_grouped_reduce_shared_two_stage`` which costs
    two ``sync_threads`` per reduction. For wide reduction axes the
    larger number of outer iters is more than offset by the per-iter
    savings.

    Seeds ``block_sizes=[1, V * 32]``, ``num_threads=[0, 32]``,
    ``cute_vector_widths=[1, V]`` so each thread owns V contiguous
    elements (one vec load per outer iter) and the cross-thread
    reduction stays inside a single warp. Applies only when the
    reduction extent is large enough to amortise the launch cost of
    many CTAs (each row is a CTA) and the dtype admits a vec >= 2.
    """

    name = "cute_tile_vec_warp_reduce"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        if spec.matmul_facts:
            return False
        if len(spec.block_sizes) != 2 or spec.reduction_loops:
            return False
        inner_block_id = (
            cast("Any", spec.block_sizes[1]).block_ids[0]
            if hasattr(cast("Any", spec.block_sizes[1]), "block_ids")
            else None
        )
        if inner_block_id is None:
            return False
        if inner_block_id not in spec.cute_vector_widths.valid_block_ids():
            return False
        dtype = _cute_tile_inner_block_dtype(env, device_ir, inner_block_id)
        vec = _cute_tile_seed_vec_width_for_dtype(dtype)
        if vec <= 1:
            return False
        # Only worth it when the reduction extent is wide enough that
        # the warp-only reduction's many outer iters still amortise
        # against the launch cost (and an autotuner-picked warp lattice
        # can hold the row).  We use the inner block fragment's high
        # bound as a proxy for the reduction extent.
        bn_fragment = cast("Any", spec.block_sizes[1])._fragment(spec)
        bn_high = getattr(bn_fragment, "high", None)
        return isinstance(bn_high, int) and bn_high >= vec * 32

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        inner_block_id = (
            cast("Any", spec.block_sizes[1]).block_ids[0]
            if hasattr(cast("Any", spec.block_sizes[1]), "block_ids")
            else None
        )
        if inner_block_id is None:
            return None
        dtype = _cute_tile_inner_block_dtype(env, device_ir, inner_block_id)
        vec = _cute_tile_seed_vec_width_for_dtype(dtype)
        if vec <= 1:
            return None
        # Each thread owns V contiguous elements; one warp = 32 threads
        # = 32 * V elements per outer-tile iter. Cap by the fragment's
        # high bound so the seed is reachable for short reduction axes.
        bn_fragment = cast("Any", spec.block_sizes[1])._fragment(spec)
        bn_high = getattr(bn_fragment, "high", None)
        block_n = vec * 32
        if isinstance(bn_high, int) and bn_high < block_n:
            return None
        seed: dict[str, Any] = {
            "block_sizes": [1, block_n],
            "num_threads": [0, 32],
            "cute_vector_widths": [1, vec],
        }
        try:
            return Config(**seed)
        except Exception:
            return None


class CuteTileVecWarpPerRowHeuristic(AutotunerHeuristic):
    """P15: warp-per-row layout for softmax-shaped tile kernels.

    Sibling seed for ``CuteTileVecWarpReduceHeuristic`` that puts MORE
    than one row per CTA — each warp owns one row. The warp-per-row
    plan in ``layout_propagation.py`` swaps the thread-axis assignment
    so:

    * N (reduction axis) lands on ``thread_idx[0]`` (32 contiguous
      threads = one warp per row)
    * M (outer grid row axis) lands on ``thread_idx[1]`` (warp index =
      row index)

    The strided reduction dispatcher then picks the direct
    ``cute.arch.warp_reduction_*`` path with ``threads_in_group=32``
    (group_span == 32, one warp per group), avoiding the
    cross-warp shared-memory two-stage reduce that would dominate when
    rows are spread across threads.

    Seeds ``block_sizes=[2, V * 32]``, ``num_threads=[0, 32]``,
    ``cute_vector_widths=[1, V]`` so each row stays inside a single
    warp and 2 rows fit in one CTA (giving 64 threads = 2 warps per
    CTA -> higher occupancy on softmax-shaped reductions).

    Eligible whenever ``CuteTileVecWarpReduceHeuristic`` is eligible
    and the outer tile fragment admits M >= 2.
    """

    name = "cute_tile_vec_warp_per_row"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        if spec.matmul_facts:
            return False
        if len(spec.block_sizes) != 2 or spec.reduction_loops:
            return False
        inner_block_id = (
            cast("Any", spec.block_sizes[1]).block_ids[0]
            if hasattr(cast("Any", spec.block_sizes[1]), "block_ids")
            else None
        )
        if inner_block_id is None:
            return False
        if inner_block_id not in spec.cute_vector_widths.valid_block_ids():
            return False
        dtype = _cute_tile_inner_block_dtype(env, device_ir, inner_block_id)
        vec = _cute_tile_seed_vec_width_for_dtype(dtype)
        if vec <= 1:
            return False
        bn_fragment = cast("Any", spec.block_sizes[1])._fragment(spec)
        bn_high = getattr(bn_fragment, "high", None)
        if not isinstance(bn_high, int) or bn_high < vec * 32:
            return False
        # Outer (M) fragment must admit M=2 (warp-per-row launches 2
        # warps per CTA so each row is one warp).
        bm_fragment = cast("Any", spec.block_sizes[0])._fragment(spec)
        bm_low = getattr(bm_fragment, "low", 1)
        bm_high = getattr(bm_fragment, "high", None)
        if not isinstance(bm_high, int):
            return False
        return bm_low <= 2 <= bm_high

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        inner_block_id = (
            cast("Any", spec.block_sizes[1]).block_ids[0]
            if hasattr(cast("Any", spec.block_sizes[1]), "block_ids")
            else None
        )
        if inner_block_id is None:
            return None
        dtype = _cute_tile_inner_block_dtype(env, device_ir, inner_block_id)
        vec = _cute_tile_seed_vec_width_for_dtype(dtype)
        if vec <= 1:
            return None
        bn_fragment = cast("Any", spec.block_sizes[1])._fragment(spec)
        bn_high = getattr(bn_fragment, "high", None)
        block_n = vec * 32
        if isinstance(bn_high, int) and bn_high < block_n:
            return None
        seed: dict[str, Any] = {
            "block_sizes": [2, block_n],
            "num_threads": [0, 32],
            "cute_vector_widths": [1, vec],
        }
        try:
            return Config(**seed)
        except Exception:
            return None


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
        if not is_canonical_row_reduction(env):
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
        seed: dict[str, Any] = {
            "block_sizes": [1],
            "num_threads": [1],
            "reduction_loops": [chunk],
        }
        vec = _cute_seed_vec_width(env, rl_spec, max_threads, size_hint, device_ir)
        if vec > 1:
            seed["cute_vector_widths"] = [vec]
        return Config(**seed)


class CuteFlashAttentionHeuristic(AutotunerHeuristic):
    """Seed ``block_sizes=[1, 128, 128]`` for detected fp16 flash-attention.

    When ``HELION_CUTE_FLASH`` is on (the default), a dense online-softmax
    attention kernel at [tile_b=1, tile_m=128, tile_n=128], fp16, head_dim in
    {64, 128} lowers to the fused tcgen05 flash path
    (``cute_flash.codegen_attention_flash``) -- orders of magnitude faster than
    the scalar fallback. The flash detector fires at EXACTLY 128x128 tiles, so
    unless that config is in the autotuner population the fast path is never
    measured. This seed puts it in generation 0; the search still owns every
    other knob and benchmarks the seed against the rest, dropping it if the
    accuracy/compile check ever fails.
    """

    name = "cute_flash_attention"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_flash_search_enabled

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        if not spec.cute_flash_search_enabled:
            return None
        from ..cute.cute_flash import flash_attention_seed_config

        assert spec._cute_flash_head_dim is not None
        return flash_attention_seed_config(
            spec._cute_flash_head_dim,
            spec._cute_flash_num_kv,
            is_causal=spec._cute_flash_is_causal,
            has_kv_tile_pruning=spec._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=spec._cute_flash_requires_ws_overlap,
            small_biased_candidate=spec._cute_flash_small_biased_candidate,
            block_size_targets=spec._cute_flash_block_size_target_list(),
        )


class CuteFlashAttentionCausalLptHeuristic(AutotunerHeuristic):
    """Seed best-known causal hd64 LPT swizzle points for large-token rows."""

    name = "cute_flash_attention_causal_lpt"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        from ..cute.cute_flash import flash_attention_seed_config

        spec = env.config_spec
        if not spec.cute_flash_search_enabled or spec._cute_flash_head_dim is None:
            return False
        return (
            flash_attention_seed_config(
                spec._cute_flash_head_dim,
                spec._cute_flash_num_kv,
                is_causal=spec._cute_flash_is_causal,
                has_kv_tile_pruning=spec._cute_flash_has_kv_tile_pruning,
                requires_ws_overlap=spec._cute_flash_requires_ws_overlap,
                small_biased_candidate=spec._cute_flash_small_biased_candidate,
                block_size_targets=spec._cute_flash_block_size_target_list(),
                seed_kind="causal_lpt",
            )
            is not None
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not cls.is_eligible(env, device_ir):
            return None

        from ..cute.cute_flash import flash_attention_seed_config

        spec = env.config_spec
        assert spec._cute_flash_head_dim is not None
        return flash_attention_seed_config(
            spec._cute_flash_head_dim,
            spec._cute_flash_num_kv,
            is_causal=spec._cute_flash_is_causal,
            has_kv_tile_pruning=spec._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=spec._cute_flash_requires_ws_overlap,
            small_biased_candidate=spec._cute_flash_small_biased_candidate,
            block_size_targets=spec._cute_flash_block_size_target_list(),
            seed_kind="causal_lpt",
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
        full_tile_reachable = m_tile_reachable and n_tile_reachable
        # The fp8 small-grid family seeds the bm=128/bn=128 tile, which is
        # reachable on shapes (small M/N) where the bm=256 full tile is not.
        small_grid_reachable = (
            constraints.allow_fp8_small_grid
            and not edge_k_tail_family
            and cls._small_grid_tile_reachable(spec)
        )
        return (full_tile_reachable or small_grid_reachable) and cls._select_bk(
            env
        ) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        bk = cls._select_bk(env)
        if bk is None:
            raise AssertionError(f"{cls.name} get_seed_config called while ineligible")

        constraints = spec._tcgen05_cluster_m2_search_constraints
        edge_k_tail_family = (
            constraints is not None and constraints.allow_edge_k_tail_family
        )
        fp8_small_grid_family = (
            constraints is not None
            and constraints.allow_fp8_small_grid
            and not edge_k_tail_family
            and cls._small_grid_tile_reachable(spec)
        )
        if fp8_small_grid_family:
            # Seed the bm=128 small-grid tile only while it is the right tile
            # for the shape: it wins on B200 cold-L2 while its 128x128 cluster
            # grid fits in ~one wave (1.00-1.17x at <=72 clusters / <=0.97
            # waves) but loses from 80 clusters / 1.08 waves up (0.84-0.94x),
            # where the larger bm=256 full tile is the better seed. Above the
            # one-wave ceiling fall through to the full-tile seed -- but only
            # when that tile is actually reachable; otherwise (e.g. M not a
            # multiple of 256) keep the small-grid seed, which is still the only
            # validated cluster_m=2 starting point. The bm=128 search candidates
            # stay reachable regardless (the search-admission gate is unchanged);
            # this only chooses the heuristic's starting point.
            if cls._small_grid_within_one_wave(env) or not cls._full_tile_reachable(
                spec
            ):
                return cls._fp8_small_grid_seed_config(env, bk)
        # Generalized known-good CtaGroup.TWO template (the DEFAULT-layout,
        # non-FFI config family that the hand-pinned per-shape seeds shared).
        # Pinning the full perf-critical knob set — not just the tile + cluster
        # — gives the autotuner a strong, complete starting point for ANY
        # 2-CTA-eligible matmul shape instead of relying on the search to
        # rediscover num_warps/num_stages/staging from a partial seed.
        # ``tcgen05_strategy`` (ROLE_LOCAL_MONOLITHIC) is the default, so it is
        # left implicit; the search still owns every one of these knobs.
        seed: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "num_warps": 8,
            "num_stages": 4,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_acc_stages": 2,
            # Matches the validated tcgen05 search restriction.
            "tcgen05_num_epi_warps": 4,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
        }
        if edge_k_tail_family:
            seed.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
        else:
            seed["l2_groupings"] = [TCGEN05_TWO_CTA_SEED_L2_GROUPING]
            seed["tcgen05_c_stages"] = 2
            # When the SMEM-budget gate admits ``ab=3`` for this seed tile
            # shape, seed the canonical fast config family directly so it
            # reaches the autotuner's initial population without depending on a
            # search-stage mutation.
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

    @staticmethod
    def _small_grid_tile_reachable(spec: ConfigSpec) -> bool:
        """True when the fp8 small-grid 2-CTA tile (bm=128/bn=128) is in range."""
        if len(spec.block_sizes) != 3:
            return False
        bm_fragment = cast("BlockSizeFragment", spec.block_sizes[0]._fragment(spec))
        bn_fragment = cast("BlockSizeFragment", spec.block_sizes[1]._fragment(spec))
        return (
            bm_fragment.low
            <= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
            <= bm_fragment.high
            and bn_fragment.low
            <= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
            <= bn_fragment.high
        )

    @staticmethod
    def _full_tile_reachable(spec: ConfigSpec) -> bool:
        """True when the bm=256/bn=256 full-tile cluster_m=2 tile is in range."""
        if len(spec.block_sizes) != 3:
            return False
        bm_fragment = cast("BlockSizeFragment", spec.block_sizes[0]._fragment(spec))
        bn_fragment = cast("BlockSizeFragment", spec.block_sizes[1]._fragment(spec))
        return (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
        )

    @staticmethod
    def _small_grid_within_one_wave(env: CompileEnvironment) -> bool:
        """True when the bm=128 small-grid cluster grid fits in ~one wave.

        Each 128x128 cluster spans 2 CTAs, so the grid fills the device once at
        ``clusters * 2 == num_sms``; the small-grid tile is the right *seed*
        only at or below that point (B200 cold-L2: 1.00-1.17x at <=72 clusters,
        0.84-0.94x from 80 clusters up). Mirrors the ``num_sms // 2`` ceiling
        rationale; uses static M/N from the single matmul fact. A non-CUDA /
        unknown SM count (0) keeps the small-grid seed (search still owns the
        final choice), matching the wave-quantization gate's mocked-host policy.
        """
        facts = env.config_spec.matmul_facts
        if len(facts) != 1:
            return True
        fact = facts[0]
        if fact.static_m is None or fact.static_n is None:
            return True
        from ...runtime import get_num_sm

        try:
            num_sm = get_num_sm(env.device)
        except (AssertionError, NotImplementedError):
            return True
        if num_sm <= 0:
            return True
        clusters = (fact.static_m // TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M) * (
            fact.static_n // TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
        )
        return clusters <= num_sm // 2

    @classmethod
    def _fp8_small_grid_seed_config(cls, env: CompileEnvironment, bk: int) -> Config:
        """Seed the fp8 small-grid 2-CTA family (per-CTA 64xbn, bm=128/bn=128).

        Pins the small-grid tile plus the deep-prefetch pipeline the cold-L2
        sweeps found optimal on the small/wave-limited fp8 serving GEMMs:
        ``ab_stages=12`` (max A/B prefetch to hide the cold DRAM read),
        ``acc_stages=1`` and ``c_stages=2`` (lean accumulator + C ring),
        ``l2_groupings=1`` (no scheduler swizzle). Measured cold-L2 vs
        torch._scaled_mm on B200: 512x2048x4096 1.14x and 512x2048x2048 1.01x,
        both ahead of the shallower ab=8/acc=2/c=4/l2=4 seed (1.02x / 0.88x).
        The bm=256 full-tile seed underfills this regime (16 clusters), so this
        small-grid seed is the strong starting point the autotuner needs.

        ``ab_stages=12`` is the validator max and sits near the B200 SMEM optin
        budget; on a lower-SMEM Blackwell SKU it is dropped gracefully by the
        seed transfer (``seed_flat_config_pairs`` catches ``InvalidConfig``) and
        the search falls back to shallower samples, so seeding the max is safe.
        """
        spec = env.config_spec
        seed: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M,
                TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N,
                bk,
            ],
            "num_warps": 8,
            "num_stages": 4,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_acc_stages": 1,
            "tcgen05_c_stages": 2,
            "tcgen05_ab_stages": 12,
            "tcgen05_num_epi_warps": 4,
            "l2_groupings": [1],
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
        }
        if spec.indexing.length == 3:
            seed["indexing"] = [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ]
        return Config(**seed)


class CuteFp8GemmSkinnyMHeuristic(AutotunerHeuristic):
    """Seed config for skinny-M FP8 GEMM kernels.

    For small M (1-16) the optimal config is very different from a large GEMM:
    a single row per grid block (``block_sizes[0]=1``), a modest N tile, a warp
    of threads on the N axis, and a wide FP8 vector load. Seeding this anchors
    the autotuner in the valid small-tile region instead of letting the random
    population burn its budget on configs like ``block_sizes=[4096, 2048]`` or
    1024-thread launches that are structurally wrong for a 1-row problem (and
    frequently overflow shared memory).

    A/B benchmark (helion benchmarks/run.py, fp8_gemm, CuTe, 120s budget):
    on M=1 shapes the search locks onto this seed and produces a 1.8-2.0x
    faster kernel than with heuristics disabled, with ~25% fewer wasted
    compile failures.

    Seeds ``block_sizes=[1, 256]``, ``num_threads=[0, 32]``,
    ``cute_vector_widths=[4, 8]`` (the autotune-winning config for
    M=1, K=4096, N=4096).
    """

    name = "cute_fp8_gemm_skinny_m"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        # Needs to be a matmul with FP8 inputs
        if not spec.matmul_facts or len(spec.matmul_facts) != 1:
            return False
        fact = spec.matmul_facts[0]
        # Check for FP8 dtypes
        is_fp8 = fact.lhs_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ) and fact.rhs_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        if not is_fp8:
            return False
        # Check for skinny M (small batch / decode scenario):
        # M <= 16 is the skinny-M case.
        return fact.static_m is not None and fact.static_m <= 16

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        # Best config from autotune benchmarks for M=1, K=4096, N=4096
        # (1.5x speedup over baseline; the M=1 search reliably converges here).
        seed: dict[str, Any] = {
            "block_sizes": [1, 256],
            "num_threads": [0, 32],
            "cute_vector_widths": [4, 8],
        }
        try:
            return Config(**seed)
        except Exception:
            return None


class CuteTcgen05ClusterM2FfiHeuristic(CuteTcgen05ClusterM2Heuristic):
    """Generalized TVM-FFI seed for full-tile CtaGroup.TWO 16-bit GEMMs.

    The generic ``--enable-tvm-ffi`` launcher builds its A/B/D TMA descriptors
    from the runtime tensor shapes, so the fast launch path is shape-GENERAL:
    the only real constraints are structural (256x256 CTA tile, cluster_m=2, a
    bk in the direct-entry stage-tuple table, bf16/fp16 operands, the 128x32
    explicit epilogue subtile). This heuristic emits that full
    ``explicit_epi_tile`` + flat-role + ``tvm_ffi_launch`` config for ANY
    eligible shape, replacing the bank of hand-pinned per-shape seeds.

    The DEFAULT-layout sibling (``CuteTcgen05ClusterM2Heuristic``) still seeds
    the non-FFI config, so the autotuner benchmarks both and keeps whichever
    wins: full-autotune A/B measured the FFI direct entry ~7-21% faster on
    smaller / square GEMMs (1024x4096x1024, 2048^3) where launch + epilogue
    overhead dominates, and tied on large compute-bound shapes
    (8192x1024x1024, 8192x2048x2048). An FFI config that fails to compile or
    the accuracy check for a given shape is dropped by the autotuner,
    degrading gracefully to the DEFAULT seed.
    """

    name = "cute_tcgen05_cluster_m2_ffi"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec._tcgen05_full_tile_direct_entry_seed_eligible()

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        # Single source of truth lives on the ConfigSpec/CuteTcgen05Config so the
        # search-projection (``_fix_target1_tvm_ffi_search_config``) and this
        # population seed emit the identical FFI envelope.
        return env.config_spec._tcgen05_full_tile_direct_entry_seed_config()
