from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ...runtime.config import Config
from ..cute.cutedsl_compat import tcgen05_runtime_n_ptx_compatible
from ..cute.cutedsl_compat import warn_tcgen05_runtime_n_ptx_fallback
from ..cute.grouped_worklist_policy import GroupedBMajor
from ..cute.grouped_worklist_policy import GroupedWorklistHardwareIdentity
from ..cute.grouped_worklist_policy import get_grouped_worklist_target_policy
from ..cute.grouped_worklist_policy import grouped_worklist_target_identities
from ..cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..cute.strategies import TCGEN05_STRATEGY_CONFIG_KEY
from ..cute.strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from ..cute.strategies import Tcgen05PersistenceModel
from ..cute.strategies import Tcgen05Strategy
from ..cute.tcgen05_config import TCGEN05_GROUPED_DYNAMIC_AB4_STAGE
from ..cute.tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_GROUPED_MODE_DYNAMIC
from ..cute.tcgen05_constants import TCGEN05_GROUPED_MODE_STATIC
from ..cute.tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from ..cute.tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS
from ..cute.tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_GROUPED_STATIC_COMMON_K_BLOCK_PAIRS
from ..cute.tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
from ..cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
from ..cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
from ..cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
from ..cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
from ..cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..cute.tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides
from .common import dedupe_configs
from .common import is_canonical_row_reduction
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...autotuner.config_fragment import BlockSizeFragment
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import MatmulFact
    from ...autotuner.config_spec import ReductionLoopSpec
    from ..compile_environment import CompileEnvironment
    from ..cute.cute_mma import Tcgen05GroupedWorklistAnalysis
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


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
    """V seed for ``PerThreadNDTileStrategy`` lane-loop vec on a given dtype.

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
        bn_high = spec.block_sizes[1]._fragment(spec).high
        block_n = 1024 if bn_high >= 1024 else max(bn_high, vec)
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
        bn_high = spec.block_sizes[1]._fragment(spec).high
        return bn_high >= vec * 32

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
        bn_high = spec.block_sizes[1]._fragment(spec).high
        block_n = vec * 32
        if bn_high < block_n:
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
        bn_high = spec.block_sizes[1]._fragment(spec).high
        if bn_high < vec * 32:
            return False
        # Outer (M) fragment must admit M=2 (warp-per-row launches 2
        # warps per CTA so each row is one warp).
        bm_fragment = spec.block_sizes[0]._fragment(spec)
        return bm_fragment.low <= 2 <= bm_fragment.high

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
        bn_high = spec.block_sizes[1]._fragment(spec).high
        block_n = vec * 32
        if bn_high < block_n:
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


def _block_size_value_reachable(
    spec: ConfigSpec,
    block_index: int,
    value: int,
) -> bool:
    if block_index < 0 or block_index >= len(spec.block_sizes):
        return False
    fragment = spec.block_sizes[block_index]._fragment(spec)
    return fragment.low <= value <= fragment.high


def _filter_reachable_block_size_configs(
    spec: ConfigSpec,
    configs: Sequence[Config],
) -> list[Config]:
    """Keep seeds whose block sizes are reachable in the live config spec."""
    result: list[Config] = []
    for config in configs:
        block_sizes = config.config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) != len(
            spec.block_sizes
        ):
            continue
        if all(
            type(value) is int and _block_size_value_reachable(spec, index, value)
            for index, value in enumerate(block_sizes)
        ):
            result.append(config)
    return result


def _tcgen05_grouped_fact(env: CompileEnvironment) -> MatmulFact | None:
    spec = env.config_spec
    if (
        not spec.cute_tcgen05_search_enabled
        or len(spec.matmul_facts) != 1
        or len(spec.block_sizes) != 3
    ):
        return None
    fact = spec.matmul_facts[0]
    if fact.lhs_dtype is not fact.rhs_dtype or fact.lhs_dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return None
    if (
        fact.m_block_id is None
        or fact.n_block_id is None
        or fact.k_block_id is None
        or fact.static_m is None
        or fact.static_n is None
        or fact.static_k is None
    ):
        return None
    try:
        block_indices = (
            spec.block_sizes.block_id_to_index(fact.m_block_id),
            spec.block_sizes.block_id_to_index(fact.n_block_id),
            spec.block_sizes.block_id_to_index(fact.k_block_id),
        )
    except KeyError:
        return None
    if block_indices != (0, 1, 2):
        return None
    if fact.static_m % 128 != 0 or fact.static_n % 64 != 0:
        return None
    return fact


def _tcgen05_grouped_worklist_structural_fact(
    env: CompileEnvironment,
) -> MatmulFact | None:
    """Return structural grouped facts before dynamic shape hints are frozen."""
    spec = env.config_spec
    if len(spec.matmul_facts) != 1 or len(spec.block_sizes) != 3:
        return None
    fact = spec.matmul_facts[0]
    if fact.lhs_dtype is not torch.bfloat16 or fact.rhs_dtype is not torch.bfloat16:
        return None
    if (
        fact.m_block_id is None
        or (fact.k_block_id is None and fact.static_k != 0)
        or (fact.n_block_id is None and fact.static_n != 0)
    ):
        return None
    if fact.k_block_id is None:
        # A statically empty reduction erases the K block association from the
        # matmul fact, but the registered third block axis remains its source.
        # Recover it so the full grouped-worklist proof below can decide whether
        # this kernel needs input-metadata specialization.
        fact = fact._replace(k_block_id=spec.block_sizes[2].block_id)
    try:
        assert fact.m_block_id is not None
        assert fact.k_block_id is not None
        m_index = spec.block_sizes.block_id_to_index(fact.m_block_id)
        k_index = spec.block_sizes.block_id_to_index(fact.k_block_id)
        n_index = (
            None
            if fact.n_block_id is None
            else spec.block_sizes.block_id_to_index(fact.n_block_id)
        )
    except KeyError:
        return None
    return fact if (m_index, n_index, k_index) in ((0, 1, 2), (0, None, 2)) else None


def _tcgen05_grouped_worklist_fact(
    env: CompileEnvironment,
) -> MatmulFact | None:
    if not env.config_spec.cute_tcgen05_search_enabled:
        return None
    return _tcgen05_grouped_worklist_structural_fact(env)


def _tcgen05_grouped_dynamic_bk64_fact(env: CompileEnvironment) -> MatmulFact | None:
    fact = _tcgen05_grouped_fact(env)
    if fact is None or cast("int", fact.static_k) % 64 != 0:
        return None
    spec = env.config_spec
    if not (
        _block_size_value_reachable(spec, 0, 128)
        and _block_size_value_reachable(spec, 1, 64)
        and _block_size_value_reachable(spec, 2, 64)
    ):
        return None
    if not spec._tcgen05_grouped_dynamic_stages_fit_for_target(
        dtype_bytes=fact.lhs_dtype.itemsize,
        output_dtype_bytes=fact.lhs_dtype.itemsize,
        device=env.device,
        bm=128,
        bn=64,
        bk=64,
        cluster_m=1,
        ab_stages=TCGEN05_GROUPED_DYNAMIC_AB4_STAGE,
        c_stages=2,
    ):
        return None
    return fact


def _tcgen05_grouped_seed_grid_is_valid(device_ir: DeviceIR, fact: MatmulFact) -> bool:
    return (
        len(device_ir.root_ids) == 1
        and len(device_ir.grid_block_ids) == 1
        and device_ir.grid_block_ids[0] == [fact.m_block_id, fact.n_block_id]
    )


def _tcgen05_grouped_static_common_k_block_k(static_k: int) -> int | None:
    return next(
        (bk for k, bk in TCGEN05_GROUPED_STATIC_COMMON_K_BLOCK_PAIRS if static_k == k),
        None,
    )


def _tcgen05_grouped_static_common_k_fact(env: CompileEnvironment) -> MatmulFact | None:
    fact = _tcgen05_grouped_fact(env)
    if fact is None:
        return None
    bk = _tcgen05_grouped_static_common_k_block_k(cast("int", fact.static_k))
    if bk is None:
        return None
    spec = env.config_spec
    if not (
        _block_size_value_reachable(spec, 0, 128)
        and _block_size_value_reachable(spec, 1, 64)
        and _block_size_value_reachable(spec, 2, bk)
    ):
        return None
    return fact


def _tcgen05_grouped_seed_config(bk: int) -> Config:
    config = Config(
        block_sizes=[128, 64, bk],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )
    config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = TCGEN05_GROUPED_MODE_STATIC
    config.config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
        Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )
    return config


_TCGEN05_GROUPED_B200_REFERENCE_NUM_SMS = 148
_TCGEN05_GROUPED_B200_LOW_RESERVED_SMS = 32
_TCGEN05_GROUPED_B200_HIGH_RESERVED_SMS = 52
_TCGEN05_GROUPED_B200_PANEL_RESERVED_SMS = 20
_TCGEN05_GROUPED_TARGET_IDENTITIES = grouped_worklist_target_identities()
_TCGEN05_GROUPED_TARGET_CAPABILITIES = frozenset(
    identity[2] for identity in _TCGEN05_GROUPED_TARGET_IDENTITIES
)
# The bound-kernel key already carries device kind and capability. Registering
# every policy identity as a named promotion target adds the exact product name,
# so two products on the same architecture cannot reuse different ranked seeds.
# These measurements rank the initial seed population only. Other SM counts are
# scaled proportionally, and live autotuning remains authoritative.

# Intentionally mirrored by ``pretuned_kernels.grouped_gemm_deepgemm``'s
# stdlib-only reviewed-profile builder; its schema-parity test guards drift.
_TCGEN05_GROUPED_WORKLIST_CONFIG_KEYS = frozenset(
    {
        "block_sizes",
        "l2_groupings",
        "loop_orders",
        "num_stages",
        "num_warps",
        "pid_type",
        "tcgen05_cluster_m",
        "tcgen05_cluster_n",
        "tcgen05_ab_stages",
        "tcgen05_acc_stages",
        "tcgen05_c_stages",
        "tcgen05_num_epi_warps",
        TCGEN05_CONSUMER_REGS_CONFIG_KEY,
        TCGEN05_GROUPED_MODE_CONFIG_KEY,
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
        TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
        TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
        TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
        TCGEN05_STRATEGY_CONFIG_KEY,
        TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
        TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
    }
)


def tcgen05_grouped_worklist_config_values(
    source_m_tile: int,
    block_k: int,
    ab_stages: int,
    consumer_regs: int,
    *,
    runtime_direct: bool = True,
    l2_swizzle_size: int | None = None,
    reserved_sms: int | None = None,
    clc: bool = False,
) -> dict[str, object]:
    """Return the exact values used by the grouped-worklist seed builder.

    This side-effect-free seam lets checked-in, stdlib-only reviewed profiles
    verify value parity without depending on the compiler's private builder.
    """
    if clc and not runtime_direct:
        raise ValueError("grouped worklist CLC requires runtime_direct=True")
    cluster_m = (
        1 if source_m_tile == TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE else 2
    )
    values: dict[str, object] = {
        "block_sizes": [256, 128, block_k],
        "l2_groupings": [1],
        "loop_orders": [[0, 1, 2]],
        "num_stages": 7,
        "num_warps": 8,
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": cluster_m,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": ab_stages,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        TCGEN05_CONSUMER_REGS_CONFIG_KEY: consumer_regs,
        TCGEN05_GROUPED_MODE_CONFIG_KEY: TCGEN05_GROUPED_MODE_WORKLIST_NM,
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY: source_m_tile,
    }
    if runtime_direct:
        values[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = True
    if l2_swizzle_size is not None:
        values[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = l2_swizzle_size
    if reserved_sms is not None:
        values[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] = reserved_sms
    if clc:
        values.update(
            {
                TCGEN05_STRATEGY_CONFIG_KEY: (
                    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
                ),
                TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 1,
                TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value
                ),
            }
        )
    if missing_schema_keys := values.keys() - _TCGEN05_GROUPED_WORKLIST_CONFIG_KEYS:
        raise RuntimeError(
            "grouped-worklist config builder emitted keys missing from its schema: "
            f"{sorted(missing_schema_keys)!r}"
        )
    return values


def _tcgen05_grouped_worklist_config(
    source_m_tile: int,
    block_k: int,
    ab_stages: int,
    consumer_regs: int,
    *,
    runtime_direct: bool = True,
    l2_swizzle_size: int | None = None,
    reserved_sms: int | None = None,
    clc: bool = False,
) -> Config:
    return Config.from_dict(
        tcgen05_grouped_worklist_config_values(
            source_m_tile,
            block_k,
            ab_stages,
            consumer_regs,
            runtime_direct=runtime_direct,
            l2_swizzle_size=l2_swizzle_size,
            reserved_sms=reserved_sms,
            clc=clc,
        )
    )


def _tcgen05_grouped_worklist_target_seed(
    *,
    groups: int,
    n: int,
    k: int,
    b_major: GroupedBMajor,
    source_m_tile: int,
    source_tiles: int,
    num_sm: int,
    target_hardware_identity: GroupedWorklistHardwareIdentity | None,
    clc_ready: bool,
) -> Config | None:
    tuning = get_grouped_worklist_target_policy(target_hardware_identity).tuning_for(
        groups=groups,
        n=n,
        k=k,
        b_major=b_major,
        source_m_tile=source_m_tile,
        source_tiles=source_tiles,
        num_sm=num_sm,
    )
    if tuning is None or (tuning.clc and not clc_ready):
        return None
    return _tcgen05_grouped_worklist_config(
        source_m_tile,
        tuning.block_k,
        tuning.ab_stages,
        tuning.consumer_regs,
        runtime_direct=tuning.runtime_direct,
        l2_swizzle_size=tuning.l2_swizzle_size,
        reserved_sms=tuning.reserved_sms,
        clc=tuning.clc,
    )


def tcgen05_grouped_worklist_config_keys() -> frozenset[str]:
    """Return the exhaustive grouped-worklist compiler-seed schema."""
    return _TCGEN05_GROUPED_WORKLIST_CONFIG_KEYS


def _tcgen05_grouped_scaled_reserved_sms(
    num_sm: int,
    b200_reserved_sms: int,
) -> int:
    value = round(num_sm * b200_reserved_sms / _TCGEN05_GROUPED_B200_REFERENCE_NUM_SMS)
    # The reviewed B200 occupancy splits were selected on a two-SM reservation
    # lattice.  Preserve that lattice after proportional scaling; an odd rounded
    # value moves up so the seed never reserves less than the scaled reference.
    value += value % 2
    return min(
        max(value, 0),
        max(num_sm - 2, 0),
        TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX,
    )


def _tcgen05_grouped_small_m_reserved_sms(
    *,
    groups: int,
    work_clusters: int,
    num_sm: int,
) -> tuple[int, int]:
    """Rank low/high reserved-SM seeds scaled from B200 occupancy splits."""
    low = _tcgen05_grouped_scaled_reserved_sms(
        num_sm,
        _TCGEN05_GROUPED_B200_LOW_RESERVED_SMS,
    )
    high = _tcgen05_grouped_scaled_reserved_sms(
        num_sm,
        _TCGEN05_GROUPED_B200_HIGH_RESERVED_SMS,
    )
    active_high = num_sm - high
    high_first = (
        groups <= 8
        and work_clusters > num_sm
        and active_high > 0
        and work_clusters % active_high == 0
    )
    return (high, low) if high_first else (low, high)


def tcgen05_grouped_worklist_seed_configs(
    *,
    groups: int,
    packed_m: int,
    n: int,
    k: int,
    b_major: GroupedBMajor,
    source_m_tile: int,
    num_sm: int,
    target_hardware_identity: GroupedWorklistHardwareIdentity | None = None,
) -> list[Config]:
    """Build ranked, tile-compatible grouped-worklist compiler seeds."""
    if any(
        type(value) is not int or value <= 0
        for value in (groups, packed_m, n, k, num_sm)
    ):
        raise ValueError("grouped seed dimensions and num_sm must be positive integers")
    if b_major not in ("k", "n"):
        raise ValueError(f"unsupported grouped B major {b_major!r}")
    if source_m_tile not in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES:
        raise ValueError(f"unsupported grouped source M tile {source_m_tile!r}")
    if packed_m % source_m_tile != 0:
        raise ValueError("packed M extent must be divisible by source_m_tile")

    source_tiles = packed_m // source_m_tile
    logical_n_per_cluster = (
        128 if source_m_tile == TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE else 256
    )
    work_clusters = source_tiles * (
        (n + logical_n_per_cluster - 1) // logical_n_per_cluster
    )
    clc_ready = (
        source_m_tile != TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
        and n % 256 == 0
        and num_sm <= work_clusters <= TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS
    )
    target_seed = _tcgen05_grouped_worklist_target_seed(
        groups=groups,
        n=n,
        k=k,
        b_major=b_major,
        source_m_tile=source_m_tile,
        source_tiles=source_tiles,
        num_sm=num_sm,
        target_hardware_identity=target_hardware_identity,
        clc_ready=clc_ready,
    )

    if source_m_tile == TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE:
        preferred_reservation, alternate_reservation = (
            _tcgen05_grouped_small_m_reserved_sms(
                groups=groups,
                work_clusters=work_clusters,
                num_sm=num_sm,
            )
        )
        high_reservation = _tcgen05_grouped_scaled_reserved_sms(
            num_sm,
            _TCGEN05_GROUPED_B200_HIGH_RESERVED_SMS,
        )
        panel_reservation = _tcgen05_grouped_scaled_reserved_sms(
            num_sm,
            _TCGEN05_GROUPED_B200_PANEL_RESERVED_SMS,
        )
        bk128 = {
            "preferred_reserved": _tcgen05_grouped_worklist_config(
                source_m_tile,
                128,
                5,
                256,
                reserved_sms=preferred_reservation,
            ),
            "alternate_reserved": _tcgen05_grouped_worklist_config(
                source_m_tile,
                128,
                5,
                256,
                reserved_sms=alternate_reservation,
            ),
            "unreserved": _tcgen05_grouped_worklist_config(source_m_tile, 128, 5, 256),
        }
        bk64 = {
            "high_reserved": _tcgen05_grouped_worklist_config(
                source_m_tile, 64, 7, 240, reserved_sms=high_reservation
            ),
            "direct": _tcgen05_grouped_worklist_config(source_m_tile, 64, 7, 240),
            "panel4": _tcgen05_grouped_worklist_config(
                source_m_tile,
                64,
                7,
                240,
                l2_swizzle_size=4,
                reserved_sms=panel_reservation,
            ),
            "panel8": _tcgen05_grouped_worklist_config(
                source_m_tile,
                64,
                7,
                240,
                l2_swizzle_size=8,
                reserved_sms=panel_reservation,
            ),
            "mailbox": _tcgen05_grouped_worklist_config(
                source_m_tile, 64, 7, 240, runtime_direct=False
            ),
        }
        if groups >= 16:
            if n > k:
                primary = bk64["panel8"]
            elif n < k:
                primary = bk64["panel4"]
            else:
                primary = bk64["direct"]
        elif n >= 9 * k // 4:
            primary = bk128["alternate_reserved"]
        elif n >= 2 * k:
            primary = bk64["high_reserved"]
        elif k > n:
            primary = bk64["direct"]
        else:
            primary = bk128["preferred_reserved"]
        if target_seed is not None:
            primary = target_seed
        return dedupe_configs([primary, *bk128.values(), *bk64.values()])

    if source_m_tile == TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT:
        # Source-224 normally keeps the reviewed panel-8 direct config as its
        # deterministic primary; target policy may replace it only for an
        # exact measured workload.
        panel8_direct = _tcgen05_grouped_worklist_config(
            source_m_tile,
            64,
            7,
            256,
            l2_swizzle_size=8,
        )
        mailbox_by_ab_stages = {
            ab_stages: _tcgen05_grouped_worklist_config(
                source_m_tile, 64, ab_stages, 240, runtime_direct=False
            )
            for ab_stages in range(7, 3, -1)
        }
        bk128_direct = _tcgen05_grouped_worklist_config(source_m_tile, 128, 3, 240)
        if target_seed is not None:
            return dedupe_configs(
                [
                    target_seed,
                    panel8_direct,
                    *mailbox_by_ab_stages.values(),
                    bk128_direct,
                ]
            )
        return dedupe_configs(
            [panel8_direct, *mailbox_by_ab_stages.values(), bk128_direct]
        )

    bk128 = {
        "mailbox": _tcgen05_grouped_worklist_config(
            source_m_tile, 128, 3, 240, runtime_direct=False
        ),
        "direct": _tcgen05_grouped_worklist_config(source_m_tile, 128, 3, 240),
        "panel8": _tcgen05_grouped_worklist_config(
            source_m_tile, 128, 3, 240, l2_swizzle_size=8
        ),
    }
    bk64 = {
        "mailbox_regs224": _tcgen05_grouped_worklist_config(
            source_m_tile, 64, 6, 224, runtime_direct=False
        ),
        "mailbox_regs240": _tcgen05_grouped_worklist_config(
            source_m_tile, 64, 6, 240, runtime_direct=False
        ),
        "direct_regs240": _tcgen05_grouped_worklist_config(source_m_tile, 64, 6, 240),
        "panel16_regs224": _tcgen05_grouped_worklist_config(
            source_m_tile, 64, 6, 224, l2_swizzle_size=16
        ),
        "panel8_stage5_regs224": _tcgen05_grouped_worklist_config(
            source_m_tile, 64, 5, 224, l2_swizzle_size=8
        ),
    }
    # Runtime CLC uses fixed full-allocation TensorMaps.  In N,M orientation,
    # logical N is the physical MMA-M dimension, so the immutable descriptor
    # path requires a whole 256-row physical tile.  Keep tail-N shapes on the
    # dynamic-TensorMap runtime-direct or mailbox families.
    clc_by_consumer_regs = (
        {
            consumer_regs: _tcgen05_grouped_worklist_config(
                source_m_tile,
                64,
                6,
                consumer_regs,
                l2_swizzle_size=8,
                clc=True,
            )
            for consumer_regs in (240, 256)
        }
        if (
            n % logical_n_per_cluster == 0
            and work_clusters <= TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS
        )
        else {}
    )
    clc_configs = list(clc_by_consumer_regs.values())
    # Ranking-only splits fitted from the reviewed B200 source-256 cases:
    # 16 groups marks high fan-out, N/K=2 and 9/4 are aspect buckets, and
    # 8/24 tiles per group separate short, medium, and long expert waves.
    tiles_per_group = source_tiles // groups
    clc_ready = bool(clc_configs) and work_clusters >= num_sm
    if target_seed is not None:
        primary = target_seed
    elif groups >= 16:
        if k > n or n >= 9 * k // 4:
            primary = bk128["mailbox"]
        elif n >= 2 * k:
            primary = bk64["direct_regs240"]
        else:
            primary = bk64["mailbox_regs224"]
    elif tiles_per_group >= 8:
        if k > n and clc_ready:
            preferred_clc_consumer_regs = 256 if groups <= 4 else 240
            primary = clc_by_consumer_regs[preferred_clc_consumer_regs]
        elif n >= 9 * k // 4:
            primary = bk64["panel16_regs224"] if b_major == "n" else bk128["panel8"]
        elif n >= 2 * k:
            primary = (
                clc_by_consumer_regs[240]
                if b_major == "n" and clc_ready
                else bk64["direct_regs240"]
            )
        elif n == k:
            primary = (
                bk128["direct"]
                if tiles_per_group >= 24
                else bk64["panel8_stage5_regs224"]
            )
        else:
            primary = bk64["mailbox_regs240"]
    elif b_major == "n" and n >= 2 * k:
        primary = bk64["direct_regs240"]
    else:
        primary = bk64["mailbox_regs240"]
    return dedupe_configs([primary, *clc_configs, *bk64.values(), *bk128.values()])


def _tcgen05_grouped_worklist_source_m_tiles(
    env: CompileEnvironment,
    analysis: Tcgen05GroupedWorklistAnalysis,
) -> tuple[int, ...]:
    """Return legal source-M schedule families for one recognized input layout."""
    if analysis.input_kind == "device_split_sizes":
        # Compact A has no physical source-tile constraint. Source-32 currently
        # requires the one-CTA path, which is not valid for device split sizes.
        return (
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
            TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
        )

    facts = analysis.seed_facts
    if not env.register_grouped_worklist_compatibility_guard(
        analysis.metadata_tensor,
        group_count=facts.groups_hint,
        packed_m=facts.packed_m_hint,
    ):
        return ()
    worklist = env.runtime_value_for_tensor(analysis.metadata_tensor)
    if not isinstance(worklist, torch.Tensor):
        return ()
    from torch._subclasses import FakeTensor
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    from ..cute.grouped_worklist import (
        tcgen05_grouped_worklist_compatible_source_m_tiles,
    )

    if isinstance(worklist, FakeTensor):
        if not isinstance(worklist.constant, torch.Tensor):
            return ()
        worklist = worklist.constant
    with unset_fake_temporarily():
        rows = (
            cast("list[list[int]]", worklist.detach().cpu().tolist())
            if worklist.ndim == 2 and worklist.shape[1] == 4
            else []
        )
    compatible = tcgen05_grouped_worklist_compatible_source_m_tiles(
        rows,
        group_count=facts.groups_hint,
        packed_m=facts.packed_m_hint,
    )
    # Prefer the established two-CTA profiles over the compact source-32
    # fallback. Ordinary 224- and 256-packed inputs are distinguishable because
    # their segment capacities are incompatible with the other large tile.
    preference = {
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT: 0,
        TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE: 1,
        TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE: 2,
    }
    return tuple(sorted(compatible, key=preference.__getitem__))


_TCGEN05_GROUPED_WORKLIST_AUTOMATIC_SEED_LIMIT = 8


def _bounded_grouped_worklist_seed_families(
    families: Sequence[Sequence[Config]],
) -> list[Config]:
    """Keep every legal family represented without expanding tuning startup."""
    nonempty = [family for family in families if family]
    if not nonempty:
        return []
    ranked = [
        family[index]
        for index in range(max(map(len, nonempty)))
        for family in nonempty
        if index < len(family)
    ]
    return dedupe_configs(ranked)[:_TCGEN05_GROUPED_WORKLIST_AUTOMATIC_SEED_LIMIT]


def _tcgen05_grouped_worklist_hardware_identity(
    env: CompileEnvironment,
) -> GroupedWorklistHardwareIdentity | None:
    """Resolve the exact policy identity, failing closed on inconsistent data."""
    from ..._argument_device import _canonicalize_argument_device
    from ..._hardware import get_hardware_info

    try:
        hardware = get_hardware_info(_canonicalize_argument_device(env.device))
    except RuntimeError:
        return None
    capability = env.config_spec.target_device_capability
    if (
        capability is None
        or hardware.compute_capability != f"sm{capability[0]}{capability[1]}"
    ):
        return None
    return (
        hardware.device_kind,
        hardware.hardware_name,
        hardware.compute_capability,
    )


class CuteTcgen05GroupedWorklistHeuristic(AutotunerHeuristic):
    """Rank grouped-worklist configs for validated Blackwell products.

    The generic primary is a supported compiler default on each named target;
    measured target policies replace it only inside their workload envelopes.
    Live autotuning still receives the complete ranked seed set.
    """

    name = "cute_tcgen05_grouped_worklist"
    backend = "cute"
    promote_seed_to_default = True
    PROMOTE_NAMED_TARGETS = _TCGEN05_GROUPED_TARGET_IDENTITIES
    CACHE_SPECIALIZATION_FACTS = frozenset({"config_num_sm", "input_tensor_metadata"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        """Record grouped SMEM facts independently of seed generation.

        Grouped semantics are discovered from DeviceIR rather than a user
        annotation. Publish seed specialization even when an empty first
        binding cannot emit seeds, and input specialization when heuristics are
        disabled but correctness facts still depend on runtime metadata.
        """
        spec = env.config_spec
        fact = _tcgen05_grouped_worklist_structural_fact(env)
        if fact is None:
            return frozenset()
        from ..cute.cute_mma import analyze_tcgen05_grouped_worklist

        analysis = analyze_tcgen05_grouped_worklist(
            env,
            device_ir,
            fact,
        )
        if analysis is None:
            return frozenset()
        seed_facts = analysis.seed_facts
        if analysis.input_kind == "external_worklist":
            env.register_grouped_worklist_compatibility_guard(
                analysis.metadata_tensor,
                group_count=seed_facts.groups_hint,
                packed_m=seed_facts.packed_m_hint,
            )
        if seed_facts.groups_hint > 0:
            spec.register_cute_tcgen05_grouped_worklist_smem_facts(
                group_count=seed_facts.groups_hint,
                device_split_sizes=seed_facts.device_split_sizes,
            )
        if env.settings.disable_autotuner_heuristics:
            return frozenset({"input_tensor_metadata"})
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def _eligible_inputs(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
    ) -> tuple[MatmulFact, Tcgen05GroupedWorklistAnalysis] | None:
        spec = env.config_spec
        fact = _tcgen05_grouped_worklist_fact(env)
        if (
            fact is None
            or (fact.static_n is not None and fact.static_n % 32 != 0)
            or spec.target_device_capability is None
            or f"sm{spec.target_device_capability[0]}{spec.target_device_capability[1]}"
            not in _TCGEN05_GROUPED_TARGET_CAPABILITIES
            or not _block_size_value_reachable(spec, 0, 256)
            or not _block_size_value_reachable(spec, 1, 128)
            or not any(
                _block_size_value_reachable(spec, 2, block_k)
                for block_k in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
            )
        ):
            return None
        if not tcgen05_runtime_n_ptx_compatible():
            warn_tcgen05_runtime_n_ptx_fallback()
            return None
        from ..cute.cute_mma import analyze_tcgen05_grouped_worklist

        analysis = analyze_tcgen05_grouped_worklist(env, device_ir, fact)
        if (
            analysis is None
            or min(
                analysis.seed_facts.groups_hint,
                analysis.seed_facts.packed_m_hint,
                analysis.seed_facts.n_hint,
                analysis.seed_facts.k_hint,
            )
            <= 0
            or analysis.seed_facts.n_hint % 32 != 0
        ):
            return None
        return fact, analysis

    @classmethod
    def _seed_configs(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
    ) -> list[Config]:
        eligible = cls._eligible_inputs(env, device_ir)
        if eligible is None:
            return []
        _fact, analysis = eligible
        spec = env.config_spec
        seed_facts = analysis.seed_facts
        source_m_tiles = _tcgen05_grouped_worklist_source_m_tiles(env, analysis)
        if not source_m_tiles:
            return []
        families: list[list[Config]] = []
        hardware_identity = _tcgen05_grouped_worklist_hardware_identity(env)
        for source_m_tile in source_m_tiles:
            # ``packed_m_hint`` remains the exact compiler fact. Seed selection
            # only needs a tile-wave estimate, so round this local ranking input
            # upward. Runtime/codegen validation never consumes the rounded value.
            ranking_quantum = (
                seed_facts.groups_hint * source_m_tile
                if seed_facts.device_split_sizes
                else source_m_tile
            )
            ranking_packed_m = (
                (seed_facts.packed_m_hint + ranking_quantum - 1) // ranking_quantum
            ) * ranking_quantum
            family = tcgen05_grouped_worklist_seed_configs(
                groups=seed_facts.groups_hint,
                packed_m=ranking_packed_m,
                n=seed_facts.n_hint,
                k=seed_facts.k_hint,
                b_major=seed_facts.b_major,
                source_m_tile=source_m_tile,
                num_sm=spec.num_sm,
                target_hardware_identity=hardware_identity,
            )
            if seed_facts.device_split_sizes:
                # Device split-size kernels derive group rows on device and are
                # supported only by the scheduler-mailbox family. Convert direct
                # seeds rather than dropping useful BK/reservation variants;
                # panel swizzles and CLC have no mailbox equivalent.
                mailbox_family: list[Config] = []
                for config in family:
                    values = dict(config.config)
                    l2_swizzle_size = values.get(
                        TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
                        1,
                    )
                    if values.get(
                        TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
                    ) == Tcgen05PersistenceModel.CLC_PERSISTENT.value or (
                        type(l2_swizzle_size) is int and l2_swizzle_size > 1
                    ):
                        continue
                    values.pop(TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, None)
                    mailbox_family.append(Config.from_dict(values))
                family = dedupe_configs(mailbox_family)
            family = [
                config
                for config in family
                if seed_facts.k_hint
                % cast("list[int]", config.config["block_sizes"])[2]
                == 0
            ]
            family = _filter_reachable_block_size_configs(spec, family)
            families.append(family)
        return _bounded_grouped_worklist_seed_families(families)

    @classmethod
    def should_promote(cls, env: CompileEnvironment) -> bool:
        fact = _tcgen05_grouped_worklist_fact(env)
        return (
            tcgen05_runtime_n_ptx_compatible()
            and fact is not None
            and fact.static_k is not None
            and super().should_promote(env)
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._eligible_inputs(env, device_ir) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        configs = cls._seed_configs(env, device_ir)
        return configs[0] if configs else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        return cls._seed_configs(env, device_ir)


class CuteTcgen05GroupedStaticCommonKHeuristic(AutotunerHeuristic):
    """Seed grouped-static configs for common K tails without partial TMA."""

    name = "cute_tcgen05_grouped_static_common_k"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        from ..cute.cute_mma import tcgen05_grouped_static_seed_has_common_k_proof

        fact = _tcgen05_grouped_static_common_k_fact(env)
        return (
            fact is not None
            and _tcgen05_grouped_seed_grid_is_valid(device_ir, fact)
            and tcgen05_grouped_static_seed_has_common_k_proof(env, device_ir, fact)
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        fact = _tcgen05_grouped_static_common_k_fact(env)
        if fact is None:
            return None
        static_k = cast("int", fact.static_k)
        bk = _tcgen05_grouped_static_common_k_block_k(static_k)
        if bk is None:
            return None
        config = _tcgen05_grouped_seed_config(bk)
        if (
            static_k >= 3 * bk
            and static_k % bk == 0
            and env.config_spec._tcgen05_ab_stages_three_fits(
                bm=128,
                bn=64,
                bk=bk,
                cluster_m=1,
            )
        ):
            config.config["tcgen05_ab_stages"] = 3
        return config


class CuteTcgen05GroupedDynamicBk64Heuristic(AutotunerHeuristic):
    """Seed the proven BK64 dynamic TensorMap grouped-static config.

    This is intentionally narrow: it only fires for one tcgen05 FP16/BF16
    rank-3 RHS grouped-NT matmul whose graph proves the exact per-group
    ``k_sizes[safe_group]`` mask on both A and grouped B operands. The dynamic
    TensorMap flag stays seed-only and is not exposed as a broad random-search
    knob.
    """

    name = "cute_tcgen05_grouped_dynamic_bk64"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        from ..cute.cute_mma import tcgen05_grouped_dynamic_bk64_seed_has_exact_k_proof

        fact = _tcgen05_grouped_dynamic_bk64_fact(env)
        return (
            fact is not None
            and _tcgen05_grouped_seed_grid_is_valid(device_ir, fact)
            and tcgen05_grouped_dynamic_bk64_seed_has_exact_k_proof(
                env,
                device_ir,
                fact,
            )
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        fact = _tcgen05_grouped_dynamic_bk64_fact(env)
        if fact is None:
            return None
        config = _tcgen05_grouped_seed_config(64)
        config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = TCGEN05_GROUPED_MODE_DYNAMIC
        config.config[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] = 3
        config.config["tcgen05_ab_stages"] = TCGEN05_GROUPED_DYNAMIC_AB4_STAGE
        return config


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
        seed = flash_attention_seed_config(
            spec._cute_flash_head_dim,
            spec._cute_flash_num_kv,
            dtype=spec._cute_flash_dtype,
            is_causal=spec._cute_flash_is_causal,
            has_kv_tile_pruning=spec._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=spec._cute_flash_requires_ws_overlap,
            small_biased_candidate=spec._cute_flash_small_biased_candidate,
            standard_dense_output=spec._cute_flash_standard_dense_output,
            standard_causal_output=spec._cute_flash_standard_causal_output,
            target_device_capability=spec.target_device_capability,
            block_size_targets=spec._cute_flash_block_size_target_list(),
        )
        if seed is not None:
            # A fresh worker retry uses one setup launch plus three timed
            # launches. The median is robust while using half the launches of
            # the normal path that timed out.
            spec.compiler_seed_timeout_retry_repetitions = 3
        return seed


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
                dtype=spec._cute_flash_dtype,
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
            dtype=spec._cute_flash_dtype,
            is_causal=spec._cute_flash_is_causal,
            has_kv_tile_pruning=spec._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=spec._cute_flash_requires_ws_overlap,
            small_biased_candidate=spec._cute_flash_small_biased_candidate,
            block_size_targets=spec._cute_flash_block_size_target_list(),
            seed_kind="causal_lpt",
        )


class CuteTcgen05ThreadLocalEpilogueHeuristic(AutotunerHeuristic):
    """Seed the one-CTA tile used by tcgen05 thread-local epilogues.

    The structural region analysis only decides whether this seed is useful.
    The exhaustive, per-config ownership proof remains authoritative for
    codegen, so planting the seed does not widen the accepted kernel surface.
    """

    name = "cute_tcgen05_thread_local_epilogue"
    backend = "cute"
    promote_seed_to_default = True

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        from ..cute.cute_mma import tcgen05_fragment_epilogue_has_unique_anchor
        from ..cute.cute_mma import tcgen05_fragment_epilogue_present

        spec = env.config_spec
        return (
            spec.cute_tcgen05_search_enabled
            and TCGEN05_TWO_CTA_SEED_PID_TYPE in spec.allowed_pid_types
            and tcgen05_fragment_epilogue_present(device_ir.graphs)
            and tcgen05_fragment_epilogue_has_unique_anchor(
                device_ir.graphs,
                device_ir=device_ir,
            )
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        fragments = spec._tcgen05_matmul_block_fragments()
        if fragments is None:
            return None
        bm_fragment, bn_fragment, bk_fragment = fragments

        def select(fragment: BlockSizeFragment, choices: tuple[int, ...]) -> int | None:
            return next(
                (value for value in choices if fragment.low <= value <= fragment.high),
                None,
            )

        bm = select(bm_fragment, (128,))
        bn = select(bn_fragment, (128, 64))
        bk = select(bk_fragment, (128, 64, 32, 16))
        if bm is None or bn is None or bk is None:
            return None
        block_sizes = spec._tcgen05_matmul_seed_block_sizes(
            bm=bm,
            bn=bn,
            bk=bk,
        )
        if block_sizes is None:
            return None
        return Config(
            block_sizes=block_sizes,
            num_stages=2,
            num_warps=8,
            pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
        )


class CuteTcgen05ClusterM2Heuristic(AutotunerHeuristic):
    name = "cute_tcgen05_cluster_m2"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"device_num_sm"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        constraints = spec._tcgen05_cluster_m2_search_constraints
        if constraints is None:
            return False
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in spec.allowed_pid_types:
            return False
        fragments = spec._tcgen05_matmul_block_fragments()
        if fragments is None:
            return False
        bm_fragment, bn_fragment, _ = fragments
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
        block_sizes = spec._tcgen05_matmul_seed_block_sizes(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
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
        fragments = spec._tcgen05_matmul_block_fragments()
        if constraints is None or fragments is None:
            return None
        bk_fragment = fragments[2]
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
        fragments = spec._tcgen05_matmul_block_fragments()
        if fragments is None:
            return False
        bm_fragment, bn_fragment, _ = fragments
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
        fragments = spec._tcgen05_matmul_block_fragments()
        if fragments is None:
            return False
        bm_fragment, bn_fragment, _ = fragments
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
        block_sizes = spec._tcgen05_matmul_seed_block_sizes(
            bm=TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M,
            bn=TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N,
            bk=bk,
        )
        if block_sizes is None:
            raise AssertionError("fp8 small-grid seed requested without matmul axes")
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
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
