from __future__ import annotations

import enum
import functools
import hashlib
import itertools
import json
import logging
import math
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import cast

import torch
from torch._inductor.runtime.runtime_utils import next_power_of_2
import torch.distributed as dist

from .._compat import _regs_per_block
from .._compat import device_num_sm
from .._compat import get_triton_version
from .._compat import num_compute_units
from .._compat import supports_amd_cdna_tunables
from .._compat import supports_maxnreg
from .._compat import supports_tensor_descriptor
from .._compat import target_device_capability as get_target_device_capability
from .._compat import warps_to_threads
from .._compiler.cute.block_scaled_config import BLOCK_SCALED_CHOICES
from .._compiler.cute.block_scaled_config import BLOCK_SCALED_CONFIG_KEYS
from .._compiler.cute.block_scaled_config import normalize_block_scaled_config
from .._compiler.cute.cute_flash import FLASH_CAUSAL_LPT_SWIZZLE_KEY
from .._compiler.cute.cute_flash import FLASH_CONFIG_KEYS
from .._compiler.cute.cute_flash import FLASH_CORR_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_FREQ_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_OFFSET0_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_OFFSET_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_RES_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_SCHEDULE_KEY
from .._compiler.cute.cute_flash import FLASH_EPI_STG_GMEM_KEY
from .._compiler.cute.cute_flash import FLASH_EPI_STG_KEY
from .._compiler.cute.cute_flash import FLASH_EPI_STG_STORE_KEY
from .._compiler.cute.cute_flash import FLASH_EPI_TMA_KEY
from .._compiler.cute.cute_flash import FLASH_EXP2_IMPL_KEY
from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
from .._compiler.cute.cute_flash import FLASH_KV_STAGE_KEY
from .._compiler.cute.cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
from .._compiler.cute.cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
from .._compiler.cute.cute_flash import FLASH_MMA_INTERLEAVE_KEY
from .._compiler.cute.cute_flash import FLASH_OTHER_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_PERSISTENT_KEY
from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
from .._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_TOPOLOGY_KEY
from .._compiler.cute.cute_flash import FlashAttentionConfig
from .._compiler.cute.cute_flash import _flash_compound_exp2_packet_overrides
from .._compiler.cute.cute_flash import _flash_e2e_offset_period
from .._compiler.cute.cute_flash import _flash_e2e_schedule_default
from .._compiler.cute.cute_flash import _flash_env_get
from .._compiler.cute.cute_flash import _flash_masked_e2e_schedule_params
from .._compiler.cute.cute_flash import _flash_normalize_e2e_offset
from .._compiler.cute.cute_flash import _flash_normalize_e2e_params
from .._compiler.cute.cute_flash import _flash_parse_e2e_schedule
from .._compiler.cute.cute_flash import _flash_pipeline_family_flags
from .._compiler.cute.cute_flash import flash_effective_config_values
from .._compiler.cute.cute_flash import flash_env_fingerprint
from .._compiler.cute.cute_flash import flash_exp2_packet_is_compound
from .._compiler.cute.cute_flash import resolve_flash_config
from .._compiler.cute.cutedsl_compat import fixed_l2_evict_last_store_policy_supported
from .._compiler.cute.direct_affine_plan import direct_affine_schedule_choices
from .._compiler.cute.split_k_cluster import validate_cluster_config
from .._compiler.cute.split_k_cluster_config import (
    FINALIZER_KEY as SPLIT_K_FINALIZER_KEY,
)
from .._compiler.cute.split_k_cluster_config import FINALIZER_WARPS
from .._compiler.cute.split_k_cluster_config import SCHEDULE_KEY as SPLIT_K_SCHEDULE_KEY
from .._compiler.cute.split_k_cluster_config import SCHEDULES as SPLIT_K_SCHEDULES
from .._compiler.cute.split_k_cluster_config import normalize_cluster_finalizer
from .._compiler.cute.split_k_cluster_config import normalize_cluster_schedule
from .._compiler.cute.split_k_workspace_config import STAGES_KEY as SPLIT_K_STAGES_KEY
from .._compiler.cute.split_k_workspace_config import WORKSPACE_CONFIG_KEYS
from .._compiler.cute.split_k_workspace_config import (
    WORKSPACE_KEY as SPLIT_K_WORKSPACE_KEY,
)
from .._compiler.cute.split_k_workspace_config import normalize_workspace_config
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_TUNABLE_KEYS
from .._compiler.cute.tcgen05_config import CuteTcgen05Config
from .._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from .._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .._compiler.cute.tcgen05_flat_grouped_config import (
    CONFIG_KEYS as GROUPED_RNA_CONFIG_KEYS,
)
from .._compiler.cute.tcgen05_flat_grouped_config import (
    CONVERTER_WARPS as GROUPED_RNA_WARPS,
)
from .._compiler.cute.tcgen05_flat_grouped_config import K_KEY as GROUPED_RNA_K_KEY
from .._compiler.cute.tcgen05_flat_grouped_config import (
    PREFIX_SCAN_KEY as GROUPED_PREFIX_SCAN_KEY,
)
from .._compiler.cute.tcgen05_flat_grouped_config import (
    PREFIX_SCANS as GROUPED_PREFIX_SCANS,
)
from .._compiler.cute.tcgen05_flat_grouped_config import (
    RESIDENT_CTAS_KEY as GROUPED_RNA_RESIDENT_CTAS_KEY,
)
from .._compiler.cute.tcgen05_flat_grouped_config import (
    STAGES_KEY as GROUPED_RNA_STAGES_KEY,
)
from .._compiler.cute.tcgen05_flat_grouped_config import (
    WARPS_KEY as GROUPED_RNA_WARPS_KEY,
)
from .._compiler.cute.tcgen05_flat_grouped_config import normalize_grouped_rna_config
from .._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from .._compiler.cute.tcgen05_grouped_descriptors import DYNAMIC
from .._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from .._compiler.cute.tcgen05_tma_rn import AUTO as CUTE_MMA_F32_AUTO
from .._compiler.cute.tcgen05_tma_rn import (
    CONVERSION_KEY as CUTE_MMA_F32_CONVERSION_KEY,
)
from .._compiler.cute.tcgen05_tma_rn import TMA_RN as CUTE_MMA_F32_TMA_RN
from .._compiler.cute.tcgen05_tma_rn import WARP_RAW as CUTE_MMA_F32_WARP_RAW
from .._compiler.cute.thread_budget import CUTE_REGISTER_TILE_MAX_ELEMENTS
from .._utils import indexing_uses_tensor_descriptor
from ..exc import InvalidConfig
from ..runtime.triton.launcher import get_num_xcd
from .block_id_sequence import BlockIdSequence
from .block_id_sequence import _BlockIdItem
from .block_id_sequence import _PowerOfTwoBlockIdItem
from .compiler_coverage import CompilerCoverageGroup
from .compiler_coverage import coverage_policy
from .config_fragment import BlockSizeFragment
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import ListOf
from .config_fragment import NumThreadsFragment
from .config_fragment import NumWarpsFragment
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .config_fragment import assert_integer_power_of_two
import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import sympy

    from .._compiler.backend import Backend
    from .._compiler.cute.loop_nesting import TileLoopPath
    from .._compiler.cute.split_k_cluster import ClusterKFacts
    from .._compiler.cute.tcgen05_constants import Tcgen05RowvecAuxFacts
    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration

log = logging.getLogger(__name__)


def _live_log_restriction(feature: str, reason: str, verbose: bool) -> None:
    """Log a search-space restriction live (at INFO), best-effort.

    ``verbose`` is the owning :class:`ConfigSpec`'s
    ``log_restrictions_verbose`` flag (sourced from
    ``Settings.autotune_log_search_space_verbose``); when False this is a no-op
    so restrictions applied outside verbose autotuning stay quiet.

    Purely diagnostic: any failure here must never disrupt compilation or the
    autotuner loop.
    """
    try:
        if verbose and log.isEnabledFor(logging.INFO):
            log.info(
                "Autotuner feature restriction: %s (%s)",
                feature,
                reason,
            )
    except Exception:
        log.debug("Failed to log search-space restriction", exc_info=True)


def _record_restriction(
    store: list[tuple[str, str]],
    feature: str,
    reason: str | None,
    verbose: bool,
) -> None:
    """Record a search-space restriction and, when verbose, log it live.

    ``verbose`` is the owning :class:`ConfigSpec`'s
    ``log_restrictions_verbose`` flag.

    Purely diagnostic: any failure here must never disrupt compilation or the
    autotuner loop, so the whole body is best-effort.
    """
    if reason is None:
        return
    try:
        pair = (feature, reason)
        # De-duplicate: a shared ConfigSpec can see the same restriction applied
        # once per matmul op (enforce_dot_requirements runs per dot node), which
        # would otherwise emit duplicate summary lines. Preserve first-occurrence
        # order (the field's documented contract).
        if pair not in store:
            store.append(pair)
    except Exception:
        log.debug("Failed to record search-space restriction", exc_info=True)
    _live_log_restriction(feature, reason, verbose)


_TARGET_DEVICE_CAPABILITY_UNSET = object()


def _copy_config_structure(value: object) -> object:
    """Copy built-in config containers while preserving opaque leaf objects."""
    if type(value) is dict:
        return {key: _copy_config_structure(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_config_structure(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_config_structure(item) for item in value)
    return value


class TensorNumelConstraint(NamedTuple):
    """Tensor element count must stay within Triton's max numel limit."""

    check_fn: Callable[..., bool]
    block_indices: tuple[int, ...]
    expr_str: str


class MatmulFact(NamedTuple):
    """Shape facts recorded when matmul requirements are applied.

    ``static_m``, ``static_n``, and ``static_k`` initially contain proven
    compile-time extents. DeviceIR analysis may later fill missing values with
    representative runtime hints for whole-kernel matmul sizing.
    """

    lhs_ndim: int
    rhs_ndim: int
    m_block_id: int | None
    n_block_id: int | None
    k_block_id: int | None
    static_m: int | None
    static_n: int | None
    static_k: int | None
    lhs_dtype: torch.dtype
    rhs_dtype: torch.dtype


class DotAxisKind(enum.Enum):
    """How one M/N/K axis of a contraction maps onto the config surface.

    A GEMM has three ``TUNABLE_TILED`` axes, which is what the historical matmul gate
    assumed. A contraction written inside a chunked kernel very often has one axis the
    kernel fixes at its full extent: the author ``hl.specialize``\\d it, so the axis has
    either no block id at all or a block id that is not in ``valid_block_ids()``. That
    is not a smaller problem, only a smaller set of knobs — the seed must adjust a
    different axis or a scalar knob instead of declining.

    - ``TUNABLE_TILED`` — a real, tunable ``block_size`` entry the seed may set.
    - ``FIXED_FULL_EXTENT`` — statically known, not tunable: the tile extent along
      this axis is the extent itself and cannot be moved.
    - ``UNKNOWN`` — no static extent (dynamic / jagged); nothing can be sized.
    """

    TUNABLE_TILED = "tunable_tiled"
    FIXED_FULL_EXTENT = "fixed_full_extent"
    UNKNOWN = "unknown"


class DotAxes(NamedTuple):
    """The :class:`DotAxisKind` and the REAL per-program tile extent of one dot's
    M/N/K axes.

    ``*_extent`` is the extent the hardware sees along that axis for one program:
    the (not-yet-chosen) block size for a ``TUNABLE_TILED`` axis — recorded here as
    the axis's full static extent, an upper bound — and the fixed extent for a
    ``FIXED_FULL_EXTENT`` one. ``*_iters`` is how many times the axis is traversed
    per program (1 unless the axis is tiled and looped).
    """

    m_kind: DotAxisKind
    n_kind: DotAxisKind
    k_kind: DotAxisKind
    m_extent: int | None
    n_extent: int | None
    k_extent: int | None

    def kind(self, axis: str) -> DotAxisKind:
        return {"m": self.m_kind, "n": self.n_kind, "k": self.k_kind}[axis]

    def extent(self, axis: str) -> int | None:
        return {"m": self.m_extent, "n": self.n_extent, "k": self.k_extent}[axis]

    @property
    def tunable_axes(self) -> tuple[str, ...]:
        return tuple(
            a for a in ("m", "n", "k") if self.kind(a) is DotAxisKind.TUNABLE_TILED
        )

    @property
    def fixed_axes(self) -> tuple[str, ...]:
        return tuple(
            a for a in ("m", "n", "k") if self.kind(a) is DotAxisKind.FIXED_FULL_EXTENT
        )


class LiveTile(NamedTuple):
    """One simultaneously-live tensor tile at one graph step.

    Extends the bare ``dim_block_ids`` liveness view with the two things a resource
    estimate cannot be written without: how WIDE an element is, and WHO produced
    the tile. A dot's fp32 output is charged to tensor memory on tcgen05 and to the
    register file otherwise; a plain load is charged to the operand staging ring;
    neither substitutes for the other.

    - ``dim_block_ids`` — block id spanned per shape dim (``None`` for a static dim),
      the same representation as :class:`AccumulatorFact`.
    - ``static_dims`` — the extent at each ``None`` position, so a footprint can be
      computed without re-resolving shapes (``None`` where unresolvable).
    - ``itemsize`` — element width in bytes.
    - ``kind`` — ``"dot_out"`` | ``"load"`` | ``"carry"`` | ``"other"`` |
      ``"global"``. A global tensor handle is retained in the structural
      timeline but is not register-resident.
    - ``stageable`` — for a loop-body load, whether its index is proven to
      vary with the enclosing loop. ``None`` means uncertain and is charged
      conservatively as stageable.
    - ``promoted_lhs`` — this live value is the non-load LHS operand of a dot
      and may require Triton's power-of-two TMEM promotion scratch. This is
      independent of ``kind`` because a prior ``dot_out`` can also be an LHS.
    """

    dim_block_ids: tuple[int | None, ...]
    static_dims: tuple[int | None, ...]
    itemsize: int
    kind: str
    stageable: bool | None = None
    promoted_lhs: bool = False


class SymbolicLoopBound(NamedTuple):
    """A preserved SymPy loop extent and its candidate-dependent symbols.

    The expression remains the one produced by symbolic tracing. Symbol maps
    identify only leaves a concrete autotuner candidate can resolve; any other
    free symbol keeps evaluation explicitly unknown.
    """

    expression: sympy.Expr
    block_size_symbols: tuple[tuple[sympy.Symbol, int], ...] = ()
    tile_id_symbols: tuple[tuple[sympy.Symbol, int], ...] = ()


class LoopAxisFact(NamedTuple):
    """One tunable or fixed axis in an enclosing device loop.

    ``extent`` is the full axis length. ``bounded_by_block_id`` records an inner
    loop over exactly one outer tile (for example split-K's
    ``tile(outer.begin, outer.end)``); in that case the candidate outer block is
    the real extent, not the whole underlying axis. ``bounded_extent`` is set
    only when that outer extent is a literal; a symbolic fixed parent remains
    explicitly uncertain. ``symbolic_bound`` preserves a candidate-dependent
    loop bound instead of matching one source expression. Block-size leaves are
    replaced with the candidate values and tile IDs with the lower-median valid
    tile before evaluating it. The per-program block is deliberately not
    recorded here: it is a candidate property and must be resolved from the
    emitted ``block_sizes`` before the trip count is used.
    """

    block_id: int
    extent: int | None
    bounded_by_block_id: int | None = None
    bounded_extent: int | None = None
    symbolic_bound: SymbolicLoopBound | None = None


class PipelinedRegion(NamedTuple):
    """Memory operations belonging to one sequential device-loop body.

    Separate regions are siblings unless their loop descriptors explicitly contain
    the same ancestor axes.  Loads are conservatively considered stageable for now;
    the tuple representation keeps that uncertainty local to this fact boundary
    while allowing useful depth to be capped by the candidate-real loop length.
    """

    loop_axes: tuple[LoopAxisFact, ...]
    tiles: tuple[LiveTile, ...]


class ResidentRegion(NamedTuple):
    """Memory operations belonging to one non-loop graph."""

    tiles: tuple[LiveTile, ...]


class RootGridFact(NamedTuple):
    """One top-level device grid before a candidate block size is selected."""

    root_graph_id: int
    block_ids: tuple[int, ...]


class KernelGridFact(NamedTuple):
    """Root-grid topology shared by every specialized kernel fact.

    ``roots`` preserves the independent top-level grids in source order.
    ``graph_to_root`` assigns nested device graphs to their owning root. Concrete
    program counts are intentionally absent: they depend on the candidate block
    sizes and must be recomputed while a heuristic edits its draft.
    """

    roots: tuple[RootGridFact, ...]
    graph_to_root: tuple[tuple[int, int], ...]

    @property
    def grid_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(root.block_ids for root in self.roots)

    def root_id_for_graph(self, graph_id: int) -> int | None:
        for candidate, root_id in self.graph_to_root:
            if candidate == graph_id:
                return root_id
        return None

    def group_for_graph(self, graph_id: int) -> tuple[int, ...]:
        root_id = self.root_id_for_graph(graph_id)
        for root in self.roots:
            if root.root_graph_id == root_id:
                return root.block_ids
        return ()

    def groups_for_graphs(
        self, graph_ids: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        for graph_id in graph_ids:
            group = self.group_for_graph(graph_id)
            if group and group not in groups:
                groups.append(group)
        return tuple(groups)


class DotSite(NamedTuple):
    """Where one contraction sits in the kernel, and how much work it does.

    Attribution of a :class:`MatmulFact` (recorded at trace time, in source order) to
    a graph node is a COMPUTED pairing validated by a post-condition, never an
    assumed one: :attr:`KernelMatmulFact.attribution_complete` is False when the
    pairing could not be proven, and a consumer that needs per-dot placement must
    check it rather than trusting these fields.

    - ``graph_id`` — the device graph the dot's node lives in.
      :class:`KernelGridFact` maps that graph to its root grid; the topology is
      intentionally not copied into each site.
    - ``updates_carry`` — the dot writes into a value carried by an enclosing loop
      (``acc=`` into a loop-carried accumulator). Such a dot's accumulator is
      resident for the whole loop, which is why it gets ranking priority.
    - ``loop_axes`` — the enclosing loop axes before a candidate block size is
      selected. This is the canonical execution-count representation; consumers
      resolve it against the candidate they are evaluating.
    - ``exact_loop_trips`` — an exact dynamic execution count proven independently
      of the loop-bound expression. This is used for work estimation only; it does
      not claim that the same loop is a useful software-pipeline opportunity.
    - ``max_loop_trips`` — an optional conservative upper bound used only by
      safety-oriented consumers such as pipeline-depth capping. It must not be used
      as candidate work.
    - ``rank_reduction_scaled_accumulator_batch_block_id`` — the leading batch
      block id of a rank-3 ``baddbmm`` whose ``acc=`` input is the loop-carried
      accumulator rescaled by a row reduction derived from an earlier dot.
    """

    graph_id: int
    updates_carry: bool
    loop_axes: tuple[LoopAxisFact, ...] = ()
    exact_loop_trips: int | None = None
    max_loop_trips: int | None = None
    rank_reduction_scaled_accumulator_batch_block_id: int | None = None


class ResolvedMatmulFact(NamedTuple):
    """One :class:`MatmulFact` interpreted in its graph and config context.

    ``MatmulFact`` remains the trace-local description of the dot. This fact binds
    it to the axis roles and execution site that consumers previously recovered
    through parallel kernel-wide arrays.
    """

    fact: MatmulFact
    axes: DotAxes
    site: DotSite


class KernelMatmulFact(NamedTuple):
    """Every resolved matmul plus the shared facts needed to configure the kernel.

    Built for any kernel with at least one contraction, so the single-matmul and
    multi-matmul front ends read the same description of the workload and only their
    POLICY differs. Kernel-name- and role-free by construction: every field is a
    measured property of the contraction structure. Generic root-grid topology lives
    in :class:`KernelGridFact` and is not duplicated here.

    - ``matmuls`` — the contextual facts, one per dot.
    - ``knob_users`` — for each tunable block id, the ``(dot_index, axis)`` pairs
      whose axis maps onto it, i.e. which dots COMPETE for that knob.
    - ``sequential_loop_trips`` — product of the trip counts of the kernel's
      sequential (non-grid) loops. For a chunked recurrence this is the number of
      chunks, so ``sequential_loop_trips * fixed_k`` recovers the logical contraction
      length even though each dot only sees one chunk.
    - ``live_dot_outputs`` — the live set at the step holding the most simultaneously
      live dot-output tiles, including ancestor loop graphs.
    - ``live_promoted_lhs`` — the peak transformed-LHS set after adding ancestor
      loop graphs. Direct loads are excluded; values retain their original
      ``kind`` so a dot output reused as another dot's LHS is counted in both roles.
    - ``live_tile_steps`` — the live tile set at EVERY step of the kernel's graphs, deduped.
      A register estimate resolves each step under the candidate block sizes and selects
      the peak by bytes.
    - ``pipelined_regions`` — the loads/stores and enclosing loop axes of each LOOP
      BODY. The axes let a candidate cap useful pipeline depth by the iterations it
      actually executes. The SMEM operand ring is charged over every conservatively
      stageable load, not only the dot's own A and B: a sum within a region times its
      useful stage count, and a max across regions (separate loops run one after the
      other).
    - ``resident_regions`` — the loads/stores of the NON-loop graphs. These are charged
      ONCE, with no stage multiplier: a load outside a loop is not multi-buffered, and
      charging it per stage over-states shared memory badly enough to shrink a tile to the
      dot minimum for a phantom overflow.
    - ``attribution_complete`` — post-condition flag described above.
    """

    matmuls: tuple[ResolvedMatmulFact, ...]
    knob_users: tuple[tuple[int, tuple[tuple[int, str], ...]], ...]
    sequential_loop_trips: int
    live_dot_outputs: tuple[LiveTile, ...]
    live_promoted_lhs: tuple[LiveTile, ...]
    live_tile_steps: tuple[tuple[LiveTile, ...], ...]
    pipelined_regions: tuple[PipelinedRegion, ...]
    resident_regions: tuple[ResidentRegion, ...]
    attribution_complete: bool

    def users_of(self, block_id: int) -> tuple[tuple[int, str], ...]:
        for bid, users in self.knob_users:
            if bid == block_id:
                return users
        return ()


class ReductionCategory(enum.Enum):
    """How a reduction axis maps onto the program grid — one category per reduction.

    - ``FULL_SLICE`` — the whole axis is reduced within one program (``x[m, :]``); not on the grid.
    - ``FULL_GRID`` — a full-extent axis on the grid, block == extent (fully resident per program).
    - ``GRID_TILE`` — a grid axis reduced over but NOT full-extent: the grid parallelizes the
      reduction across programs, so the whole-axis size is not a per-program extent.
    - ``USER_TILE`` — a tunable inner sequential ``hl.tile`` over the reduction axis.
    - ``FIXED_TILE`` — an inner sequential ``hl.tile`` whose explicit fixed block differs from
      the full iteration extent.
    - ``DECLINED`` — no static extent (e.g. jagged / data-dependent); recorded but never sized.
    """

    FULL_SLICE = "full_slice"
    FULL_GRID = "full_grid"
    GRID_TILE = "grid_tile"
    USER_TILE = "user_tile"
    FIXED_TILE = "fixed_tile"
    DECLINED = "declined"


# Categories for which the seed has a statically modelable per-program reduction width.
# FIXED_TILE is already fixed rather than chosen by the seed. GRID_TILE (grid-parallelized
# partial) stays a grid row and DECLINED (no static extent) falls back to the default.
SIZED_REDUCTION_CATEGORIES = frozenset(
    {
        ReductionCategory.FULL_SLICE,
        ReductionCategory.FULL_GRID,
        ReductionCategory.USER_TILE,
        ReductionCategory.FIXED_TILE,
    }
)
# Categories that occupy the full reduction extent within one program.
FULL_EXTENT_CATEGORIES = frozenset(
    {ReductionCategory.FULL_SLICE, ReductionCategory.FULL_GRID}
)


class ReductionDescriptor(NamedTuple):
    """One reduction OCCURRENCE: a (``graph_id``, ``block_id``) reduction on the ORIGINAL
    (pre-roll) device graphs. Stage 1 emits a list of these; the allocator consumes them.

    A reduction axis may occur in more than one original graph (e.g. a kernel that reduces the
    same axis in two separate passes) — each occurrence is its own descriptor, so sequential
    passes over one axis are NOT collapsed.

    ``graph_id`` is read off the ORIGINAL graphs only (rolled ``ReductionLoopGraphInfo``
    subgraphs excluded), so it is invariant to the autotuner flipping a
    ``reduction_loops`` knob.

    Fields:
    - ``category``: the :class:`ReductionCategory`.
    - ``block_id`` / ``graph_id``: the reduction axis and original graph.
    - ``size_hint`` / ``input_load_itemsize``: logical extent and the HBM-load element width
      feeding it.
    - ``fixed_tile_size_hint``: fixed per-iteration width for ``FIXED_TILE``; ``None`` otherwise.
    - ``row_reread`` / ``reread_eviction_index``: per-reduction memory-op signals.
    """

    category: ReductionCategory
    block_id: int
    graph_id: int
    size_hint: int
    input_load_itemsize: int = 0
    row_reread: bool = False
    reread_eviction_index: int | None = None
    fixed_tile_size_hint: int | None = None


class ReductionKernelFact(NamedTuple):
    """The per-kernel reduction product: reduction descriptors, every tunable off-grid
    non-reduction loop, parallel grid axes, the complete live-tile timeline, and axes whose
    contiguous memory access makes them coalescing-sensitive.

    Built by ``build_reduction_kernel_fact``. ``reductions`` may be empty (a kernel with only
    GRID_TILE / DECLINED reductions, or none) — the seed then declines.
    """

    reductions: tuple[ReductionDescriptor, ...]
    non_reduction_loop_block_ids: tuple[int, ...] = ()
    grid_axis_block_ids: tuple[int, ...] = ()
    live_tile_steps: tuple[tuple[LiveTile, ...], ...] = ()
    coalescing_sensitive_block_ids: tuple[int, ...] = ()


class MatmulWithReductionEpilogueFact(NamedTuple):
    """A fused matmul + reduction-over-output-axis epilogue, recorded when a ``MatmulFact`` and
    a register-resident epilogue reduction co-occur in one kernel (e.g.
    ``matmul_rms_norm``: ``acc = x @ y`` then a reduction over N on the carried ``[M_BLOCK,
    N]`` accumulator, then write-back). A COMPOSED fact: it holds the matmul fact plus the few
    derived fields the seed keys on. ``TritonMatmulReductionEpilogueHeuristic`` branches on it.

    - ``matmul``: the composed matmul sub-fact.
    - ``n_extent``: the specialized output width N (= the epilogue reduction's ``size_hint``); N
      is ``hl.specialize``'d (never tiled), so both the ``[M_BLOCK, N]`` accumulator and the
      ``[K_BLOCK, N]`` operand tile scale with N — the resident-footprint signal the
      footprint-aware tile chooser keys on.
    - ``m_block_id`` / ``k_block_id``: the grid M tile and the K tile the seed sizes
      (there is no ``n_block_id`` — N is specialized, not a block_size).
    """

    matmul: MatmulFact
    n_extent: int
    m_block_id: int | None
    k_block_id: int | None


class MemoryOpFact(NamedTuple):
    """Metadata linking one ``Config.indexing`` slot to its graph memory op, one entry per
    load/store in graph-traversal order (so ``memory_op_facts[i]`` describes ``config.indexing[i]``).
    Lets heuristics reason about *which* load/store a slot is, not a bare positional index.

    The reduction-fact builders consume the enrichment fields below (all reduction-AGNOSTIC — no
    notion of which axis is "the" reduction; the builders index them by the reduction's ``block_id``).
    """

    indexing_index: int  # slot in Config.indexing (== position in this list)
    kind: str  # "load" | "store"
    eviction_index: int | None  # slot in Config.load_eviction_policies, else None
    tensor_name: str | None  # host buffer name being accessed, e.g. "x", "weight"
    dtype: torch.dtype | None  # element dtype of the accessed tensor
    ndim: int  # rank of the accessed tensor
    num_reuses: int  # downstream FX consumers of the load (0 for stores)
    matmul_operand: str | None  # matmul/dot operand: "lhs" | "rhs" | None
    # --- reduction-fact enrichment (reduction-agnostic; () / False / None for stores) ---
    # device_ir.graphs index this op lives in (scopes per-graph counts).
    graph_id: int = -1
    # per-axis count of reductions this load FEEDS: ((reduction_axis_block_id, count), ...).
    reductions_fed: tuple[tuple[int, int], ...] = ()
    # stores the load's value reaches WITHOUT passing through a reduction, each keyed by the
    # store's full subscript-axis tuple (store[tile_m, tile_n] -> (id_m, id_n)); the forward
    # walk cuts at both reductions and stores. Empty == value never bypasses a reduction.
    stores_fed: tuple[tuple[int | None, ...], ...] = ()
    # block-id per non-bare-int subscript, from the accessed tensor's SHAPE dims (``None`` where
    # unresolvable). Shape-resolved fallback for the gates below (the plain-slice case,
    # e.g. ``out[tile_m, :]``).
    indexed_block_ids: tuple[int | None, ...] = ()
    # inner-dim extent for a rank>=2 op (a reduction-width signal; gates use subscript_block_ids).
    inner_extent: int | None = None
    # AXIS the op's INDEX subscripts address (block-id per non-bare-int position, from the tile/offset
    # subscript so it is reduction-AGNOSTIC; ``None`` for a plain slice). The faithful axis key for
    # the full_width_output / input_load_itemsize gates.
    subscript_block_ids: tuple[int | None, ...] = ()
    # Element stride of the accessed tensor along each subscript position, aligned 1:1 with
    # ``subscript_block_ids`` (from ``.stride()``). A stride-1 position is the contiguous (coalescing)
    # axis — the last subscript for a row-major tensor, a different one for a transposed/strided view.
    subscript_strides: tuple[int, ...] = ()
    # GATHER stride per subscript position: the multiplier on the tile index in the op's
    # ADDRESS expression (``x[tile]`` → 1, ``x[2*tile.index]`` → 2). Independent of
    # ``subscript_strides``, the accessed tensor's LAYOUT stride -- two ops can index the
    # same row-major tensor while one reads it stride-2 and the other stride-1. A stride-k
    # gather wastes ``1 - 1/k`` of every 32B sector, so this is the coalescing signal;
    # 1 means coalesced or unknown. See ``indexing_strategy.subscript_index_scale``.
    subscript_index_scales: tuple[int, ...] = ()
    # Size-hinted extent at each subscript position. With ``subscript_affine_block_ids``
    # this gives the tile ONE op materializes: the product of extents at non-tiled
    # positions (a tiled position contributes its block size). Distinct from
    # ``accessed_numel``, a per-TENSOR count that also grows for an oversized tensor.
    subscript_extents: tuple[int, ...] = ()
    # Block-ids resolved THROUGH an affine (scaled/offset) subscript.
    # ``subscript_block_ids`` reads ``meta["tile_with_offset"]``, which the lowering sets
    # for ``tile_index + const`` but not ``tile_index * const``, so a scaled subscript
    # loses its axis there. Separate field so existing consumers are unaffected.
    subscript_affine_block_ids: tuple[int | None, ...] = ()
    # DISTINCT HBM elements the op's accessed tensor touches: product of its size-hinted shape dims
    # over NON-broadcast dims (``stride != 0``); a stride-0 dim contributes factor 1 (``0`` if no
    # resolvable fake tensor). A FULL-EXTENT op has ``accessed_numel`` == the problem numel; a
    # BROADCAST operand has a STRICTLY SMALLER count — whether it is a small tensor (``bias[N]``,
    # ``[M,1]``, ``[1,N]``) OR a full-SIZE ``.expand()``/``broadcast_tensors`` view with a stride-0
    # dim. The faithful signal for per-element HBM traffic at ANY rank/stride, unlike a bare ``ndim``
    # check (a full-rank ``[M,1]`` broadcast passes ndim) or a shape-only product (a stride-0 expand
    # passes shape).
    accessed_numel: int = 0


class AccumulatorFact(NamedTuple):
    """One loop-carried tensor accumulator in a reduction loop, recorded at compile time.
    Reduction-AGNOSTIC (like ``MemoryOpFact``): ``dim_block_ids`` is the per-dim block-id
    provenance (``None`` for a static dim), and ``itemsize`` is the element size.
    """

    dim_block_ids: tuple[int | None, ...]
    itemsize: int


class PointwiseElementwiseFact(NamedTuple):
    """Workload facts for a PURE elementwise/pointwise kernel — DEFINED by the absence of any
    reduction/matmul/accumulator fact (the disjointness rule): if one of those fired, the kernel
    belongs to that family and this fact is never built. Bandwidth-bound; the compiler defaults it to
    ``block_size=32`` (which starves HBM), so the seed sizes a saturating tile from these fields
    (derived from the walker ``MemoryOpFact`` list + block-size specs, plus one graph walk).

    - ``total_numel``: product of the tiled block dims' ``size_hint``s (the problem element count,
      M*N); the occupancy / grid-saturation input.
    - ``slab_numel``: the untiled inner slab, in ELEMENTS, that full-extent ops drag per tiled element
      = ``sum(accessed_numel // total_numel)`` over those ops (flat kernel: 1 per op; rope:
      heads*head_dim). A BROADCAST operand (``bias[N]``, ``[M,1]``, stride-0) has
      ``accessed_numel < total_numel`` → amortized → excluded; an OVERSIZED operand still touches the
      full problem → counted (hence ``>=``).
    - ``storage_itemsize`` / ``compute_itemsize``: the STORAGE (HBM) and widest COMPUTE (fp32) byte
      widths that scale slab_numel into the two budgets — ``slab_numel * storage`` = bandwidth traffic,
      ``slab_numel * compute`` = register cap (compute reads the promoted dtype; a memory op knows only
      storage). NOTE: the register cap is a COARSE proxy (blind to compute temporaries), but benign —
      pointwise is memory-bound; its only jobs are relaxing the floor for a heavy slab and capping the
      transpose-conflict tile. (Mixed-dtype storage uses the max width — a minor seed approximation.)
    - ``contig_block_ids``: TILED block-ids that are the stride-1 axis of some full-extent op (from
      ``subscript_strides``, no graph walk). Row-major → the last dim (seed unchanged); transposed →
      a different dim; two+ entries = a load-vs-store CONFLICT → the seed emits a BALANCED tile.
    - ``sfu_ops``: count of transcendental (SFU) ops. SFU ops are latency-bound on a distinct unit, so
      a transcendental-heavy tile wants more warps while an all-FMA tile of the same op count does not
      — so SFU count (not total op count) drives the num_warps ramp.
    - ``gather_stride``: the widest GATHER stride any full-extent op applies to a tiled axis in
      its ADDRESS expression (``x[tile]`` → 1, ``x[2*tile.index]`` → 2). A stride-k gather
      touches k 32B sectors per useful sector, so the same useful bytes cost ~k× the requests —
      the coalescing input to the byte budget and the num_warps ramp. NOT the layout stride,
      which cannot distinguish two ops indexing one row-major tensor at different strides.
      1 means coalesced, or an address form the walk does not recognize.
    - ``max_op_slab_numel``: the LARGEST single op's untiled fan-out, vs ``slab_numel``'s SUM
      of the same quantity. Bytes ADD across ops (so the sum sizes the byte/register budgets)
      but LANES do not (separate vector instructions over the same threads), so the max is what
      bounds a num_warps starvation cap.
    """

    total_numel: int
    slab_numel: int
    storage_itemsize: int
    compute_itemsize: int
    contig_block_ids: tuple[int, ...] = ()
    sfu_ops: int = 0
    gather_stride: int = 1
    max_op_slab_numel: int = 1


def shrink_block_sizes_for_numel_constraints(
    constraints: list[TensorNumelConstraint],
    block_sizes: list[int],
    min_sizes: list[int],
) -> None:
    """Shrink *block_sizes* in-place so every *constraint* is satisfied.

    Halves the largest involved block size first for balanced tiles.
    Fixed-point loop handles cross-constraint interactions.
    """
    prev = list(block_sizes)
    while True:
        for constraint in constraints:
            while not constraint.check_fn(
                *(block_sizes[i] for i in constraint.block_indices)
            ):
                best_idx: int | None = None
                best_val = -1
                for i in constraint.block_indices:
                    can_halve = block_sizes[i] // 2 >= min_sizes[i]
                    if can_halve and block_sizes[i] > best_val:
                        best_val = block_sizes[i]
                        best_idx = i
                if best_idx is None:
                    log.warning(
                        "tensor numel constraint unsatisfiable at minimum "
                        "block sizes: %s",
                        constraint.expr_str,
                    )
                    break
                block_sizes[best_idx] //= 2
        if block_sizes == prev:
            break
        prev = list(block_sizes)


DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1
VALID_CROSS_LOOP_PIPELINES = ("barrier", "static", "dynamic")
CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY = "cute_chunk_recurrence_dv_partitions"
CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY = "cute_chunk_recurrence_register_cap"
VALID_CUTE_CHUNK_RECURRENCE_REGISTER_CAPS = (None, 72, 76, 80)
CUTE_GDN_RECURRENCE_STAGES_KEY = "cute_gdn_recurrence_stages"
CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY = "cute_gdn_recurrence_epilogue_warps"
CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY = "cute_gdn_recurrence_token_groups"
CUTE_GDN_RECURRENCE_MMA_M_KEY = "cute_gdn_recurrence_mma_m"
CUTE_CHUNK_PREPARE_SCHEDULE_KEY = "cute_chunk_prepare_schedule"
VALID_CUTE_CHUNK_PREPARE_SCHEDULES = (
    "split_alias_cpc1",
    "split_alias_cpc2",
    "split_alias_cpc3",
    "split_alias_cpc4",
    "split_alias_cpc5",
)
CUTE_AFFINE_SCAN_SCHEDULE_KEY = "cute_affine_scan_schedule"
CUTE_TOPK_CHOICES: dict[str, tuple[int | str, ...]] = {
    "cute_topk_lanes_per_row": (16, 1, 2, 4, 8, 32),
    "cute_topk_rows_per_block": (8, 1, 2, 4, 16, 32, 64, 128),
    "cute_topk_vector_width": (8, 1, 2, 4),
    "cute_topk_output_vector_width": (1, 2, 4, 8),
    "cute_topk_value_mode": ("gather", "decode"),
    "cute_topk_key_dtype": ("int32", "float32", "float32_bits", "float32_native"),
    "cute_topk_rank_mode": ("signed", "ordinal"),
    "cute_topk_selection_layout": ("replicated", "distributed"),
    "cute_topk_sort_network": ("batcher", "compact", "compact_pruned"),
    "cute_topk_key_encoder": ("dsl", "asm", "paired"),
    "cute_topk_defer_value_gathers": (False, True),
    "cute_topk_merge_schedule": ("sequential", "balanced"),
}
CUTE_TOPK_CONFIG_KEYS: frozenset[str] = frozenset(CUTE_TOPK_CHOICES)


def _cute_chunk_recurrence_config_is_safe(
    dv_partitions: object, register_cap: object
) -> bool:
    """Reject register caps on the TMEM schedule's dynamic register protocol."""

    return dv_partitions != 2 or register_cap is None


# Upper bound (power of two) that a matmul tile dimension's block size may reach
# even when the dimension itself is smaller. Applied only to dimensions that
# feed an hl.dot (see enforce_dot_requirements), so the autotuner can mask-
# overshoot a small matmul dimension up to a hardware-friendly tile (e.g. an M
# tile matching the native MMA shape). We restrict this to matmuls because such
# kernels are memory/MMA-bound on the small dimension -- the masked-off rows/cols
# are effectively free and the larger, hardware-aligned tile runs faster -- while
# for elementwise/reduction kernels a larger-than-dimension tile is pure waste.
SMALL_DIM_BLOCK_SIZE_OVERSHOOT = 64

# Base backend tunable keys (public)
_BASE_BACKEND_TUNABLE_KEYS: frozenset[str] = frozenset(
    {
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "num_ctas",
        "occupancy",
        "pallas_worklist_grouping",
        "pallas_loop_type",
        "pallas_emit_pipeline_group_size",
        "pallas_use_low_level_scheduler",
        "pallas_fold_dot_lhs_cast",
        "pallas_pre_broadcast",
        *CUTE_TCGEN05_TUNABLE_KEYS,
    }
)
_BACKEND_DIAGNOSTIC_CONFIG_KEYS = CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS


def _get_backend_tunable_keys() -> frozenset[str]:
    """Get all backend tunable keys, including FB-private ones if available."""
    try:
        from ..fb.mtia_tunables import MTIA_TUNABLES  # pyrefly: ignore [missing-import]

        return _BASE_BACKEND_TUNABLE_KEYS | frozenset(MTIA_TUNABLES)
    except ImportError:
        return _BASE_BACKEND_TUNABLE_KEYS


BACKEND_TUNABLE_KEYS: frozenset[str] = _get_backend_tunable_keys()
_BACKEND_STRATEGY_CONFIG_KEYS = CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
# All config keys whose support depends on the backend.  The base Backend
# class rejects these by default; each backend subclass opts in selectively.
BACKEND_SPECIFIC_KEYS: frozenset[str] = (
    BACKEND_TUNABLE_KEYS
    | _BACKEND_DIAGNOSTIC_CONFIG_KEYS
    | _BACKEND_STRATEGY_CONFIG_KEYS
    | BLOCK_SCALED_CONFIG_KEYS
    | {SPLIT_K_SCHEDULE_KEY, SPLIT_K_FINALIZER_KEY}
    | WORKSPACE_CONFIG_KEYS
    | GROUPED_RNA_CONFIG_KEYS
    | frozenset(FLASH_CONFIG_KEYS)
    | {
        "cross_loop_pipeline",
        "cute_flash_gate_warpgroups",
        "cute_flash_bwd_persistent",
        "cute_flash_bwd_two_cta",
        "cute_flash_bwd_exp2_f32",
        CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
        CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,
        CUTE_GDN_RECURRENCE_STAGES_KEY,
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
        CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
        CUTE_GDN_RECURRENCE_MMA_M_KEY,
        CUTE_CHUNK_PREPARE_SCHEDULE_KEY,
        CUTE_AFFINE_SCAN_SCHEDULE_KEY,
        *CUTE_TOPK_CONFIG_KEYS,
        "num_threads",
        "cute_vector_widths",
        "cute_lane_layouts",
        "cute_reduction_reloads",
        "cute_reduction_schedule",
        "cute_reduction_pipeline_depth",
        "cute_reduction_local_tree",
        "cute_reduction_row_schedule",
        "cute_reduction_pack_output",
        "cute_reduction_sequence",
        "cute_host_paired_sum",
        "cute_reduction_group_rows",
        "cute_materialized_schedule",
        "cute_materialized_operand_schedule",
        "cute_fragment_scan",
        "cute_fragment_reduction",
        "cute_fragment_threads",
        "cute_fragment_register_loads",
        "cute_fragment_warp_results",
        "cute_pointwise_pid_type",
        "cute_async_load_stages",
        "cute_async_load_lookahead",
        "cute_async_load_group_rows",
        "cute_async_load_cache",
        "cute_async_store_policy",
        "cute_bf16x2_recurrence",
        "cute_register_chain",
        "cute_signed_bitfield_bf16",
        "cute_collective_mma",
        "cute_collective_static_layouts",
        "cute_collective_copy",
        "cute_collective_recipe",
        "cute_collective_operand_packets",
        "cute_collective_epilogue",
        "cute_collective_stages",
        "cute_collective_compute",
        "cute_collective_native_seeded",
        "cute_collective_tmem_seed",
        "cute_collective_tmem_a",
        "cute_gathered_mma_n",
        "cute_gathered_mma_stages",
        "cute_proven_bounds",
        "cute_rng_packet",
        "cute_independent_reduction",
        "cute_replicated_reduction",
        "cute_vector_packet_unroll",
        "cute_vloop_sink",
        "cute_lane_unroll",
        "cute_pdl",
        "cute_packet_prefetch",
        "cute_cluster_n",
        "cute_min_blocks_per_mp",
        "load_cache_modifiers",
        "store_cache_modifiers",
        "host_tensor_descriptors",
        "pallas_loop_type",
        "pallas_emit_pipeline_group_size",
        "pallas_use_low_level_scheduler",
        "pallas_fold_dot_lhs_cast",
        "pallas_load_buffer_count",
        "pallas_indirect_access_mode",
        "pallas_pre_broadcast",
        "xcd_remap",
    }
)
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "num_threads",
        "loop_orders",
        "l2_groupings",
        "reduction_loops",
        "flatten_loops",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "cross_loop_pipeline",
        CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
        CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,
        CUTE_GDN_RECURRENCE_STAGES_KEY,
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
        CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
        CUTE_GDN_RECURRENCE_MMA_M_KEY,
        CUTE_CHUNK_PREPARE_SCHEDULE_KEY,
        CUTE_AFFINE_SCAN_SCHEDULE_KEY,
        *CUTE_TOPK_CONFIG_KEYS,
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        "indexing",
        "atomic_indexing",
        "load_eviction_policies",
        "load_cache_modifiers",
        "store_cache_modifiers",
        "host_tensor_descriptors",
        "pallas_loop_type",
        "pallas_emit_pipeline_group_size",
        "pallas_use_low_level_scheduler",
        "pallas_fold_dot_lhs_cast",
        "pallas_load_buffer_count",
        "pallas_indirect_access_mode",
        "pallas_pre_broadcast",
        "pallas_internal_scratch",
        "cute_vector_widths",
        "cute_lane_layouts",
        "cute_reduction_reloads",
        "cute_reduction_schedule",
        "cute_reduction_pipeline_depth",
        "cute_reduction_local_tree",
        "cute_reduction_row_schedule",
        "cute_reduction_pack_output",
        "cute_reduction_sequence",
        "cute_host_paired_sum",
        "cute_reduction_group_rows",
        "cute_materialized_schedule",
        "cute_materialized_operand_schedule",
        "cute_fragment_scan",
        "cute_fragment_reduction",
        "cute_fragment_threads",
        "cute_fragment_register_loads",
        "cute_fragment_warp_results",
        "cute_pointwise_pid_type",
        "cute_async_load_stages",
        "cute_async_load_lookahead",
        "cute_async_load_group_rows",
        "cute_async_load_cache",
        "cute_async_store_policy",
        "cute_bf16x2_recurrence",
        "cute_register_chain",
        "cute_signed_bitfield_bf16",
        "cute_collective_mma",
        "cute_collective_static_layouts",
        "cute_collective_copy",
        "cute_collective_recipe",
        "cute_collective_operand_packets",
        "cute_collective_epilogue",
        "cute_collective_stages",
        "cute_collective_compute",
        "cute_collective_native_seeded",
        "cute_collective_tmem_seed",
        "cute_collective_tmem_a",
        "cute_gathered_mma_n",
        "cute_gathered_mma_stages",
        "cute_proven_bounds",
        "cute_rng_packet",
        "cute_independent_reduction",
        "cute_replicated_reduction",
        "cute_vector_packet_unroll",
        "cute_vloop_sink",
        "cute_lane_unroll",
        "cute_pdl",
        "cute_packet_prefetch",
        "cute_cluster_n",
        "cute_min_blocks_per_mp",
        *BACKEND_TUNABLE_KEYS,
        "advanced_controls_file",
        "epilogue_subtile",
        "xcd_remap",
        *_BACKEND_DIAGNOSTIC_CONFIG_KEYS,
        *_BACKEND_STRATEGY_CONFIG_KEYS,
        *FLASH_CONFIG_KEYS,
        "cute_flash_bwd_persistent",
        "cute_flash_bwd_two_cta",
        "cute_flash_bwd_exp2_f32",
        "cute_flash_gate_warpgroups",
        *BLOCK_SCALED_CONFIG_KEYS,
        SPLIT_K_SCHEDULE_KEY,
        SPLIT_K_FINALIZER_KEY,
        *WORKSPACE_CONFIG_KEYS,
        *GROUPED_RNA_CONFIG_KEYS,
    ]
)
# Loop types the autotuner searches by default for every Pallas inner loop.
AUTOTUNED_PALLAS_LOOP_TYPES = ("emit_pipeline", "unroll", "fori_loop")
VALID_PALLAS_LOOP_TYPES = AUTOTUNED_PALLAS_LOOP_TYPES
VALID_PALLAS_WORKLIST_GROUPINGS = (0, 1, 2)
VALID_PID_TYPES = (
    "flat",
    "xyz",
    "persistent_blocked",
    "persistent_interleaved",
)
MIN_NUM_SM_MULTIPLIER = 1
MAX_NUM_SM_MULTIPLIER = 128
DEFAULT_NUM_SM_MULTIPLIER = 1
EPILOGUE_SUBTILE_EXTENDED_CHOICES = (None, 2, 4)
EPILOGUE_SUBTILE_DEFAULT_CHOICES = (None, 2)
EPILOGUE_SUBTILE_MIN_K_HINT = 1024
EPILOGUE_SUBTILE_MIN_K_HINT_EXTENDED = 16384
# None means no limit. The autotuner retains this deliberately small search
# domain, while explicit configs may select any positive integer up to the
# existing supported upper bound.
AUTOTUNED_MAXNREG = (None, 32, 64, 128, 256)
# Backward-compatible name for callers that inspect the autotuning surface.
VALID_MAXNREG = AUTOTUNED_MAXNREG
MIN_MAXNREG = 1
MAX_MAXNREG = 256
DEFAULT_MAXNREG = None
_CUTE_IMPLICIT_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        "loop_orders",
        "flatten_loops",
        "l2_groupings",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "load_eviction_policies",
        "indexing",
        "atomic_indexing",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        "cute_reduction_schedule",
        "cute_reduction_pipeline_depth",
        "cute_reduction_local_tree",
        "cute_reduction_row_schedule",
        "cute_reduction_pack_output",
        "cute_reduction_sequence",
        "cute_host_paired_sum",
        "cute_reduction_group_rows",
        "cute_materialized_schedule",
        "cute_materialized_operand_schedule",
        "cute_fragment_scan",
        "cute_fragment_reduction",
        "cute_fragment_threads",
        "cute_fragment_register_loads",
        "cute_fragment_warp_results",
        "cute_pointwise_pid_type",
        "cute_async_load_stages",
        "cute_async_load_lookahead",
        "cute_async_load_group_rows",
        "cute_async_load_cache",
        "cute_async_store_policy",
        "cute_bf16x2_recurrence",
        "cute_register_chain",
        "cute_signed_bitfield_bf16",
        "cute_collective_mma",
        "cute_collective_static_layouts",
        "cute_collective_copy",
        "cute_collective_recipe",
        "cute_collective_operand_packets",
        "cute_collective_epilogue",
        "cute_collective_stages",
        "cute_collective_compute",
        "cute_collective_native_seeded",
        "cute_collective_tmem_seed",
        "cute_collective_tmem_a",
        "cute_gathered_mma_n",
        "cute_gathered_mma_stages",
        "cute_proven_bounds",
        "cute_rng_packet",
        "cute_independent_reduction",
        "cute_replicated_reduction",
        "cute_vector_packet_unroll",
        "cute_vloop_sink",
        "cute_lane_unroll",
        "cute_pdl",
        "cute_packet_prefetch",
        *BLOCK_SCALED_CONFIG_KEYS,
        SPLIT_K_SCHEDULE_KEY,
        SPLIT_K_FINALIZER_KEY,
        *WORKSPACE_CONFIG_KEYS,
    }
)


# For tileir backend or AMD ROCM, eviction policies are not supported.
# Keep this uncached: some tests patch the AMD capability helper, and caching
# only on backend name can poison later Triton ConfigSpec construction inside
# the same worker process.
def get_valid_eviction_policies(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    if backend_name == "cute":
        # "first"/"last" lower to ld.global L1 eviction hints
        # (level1_eviction_priority=evict_first/evict_last) on the
        # vectorized load sites.  Pays off for reload-from-gmem sweeps
        # (keep re-read rows resident, evict on the final pass).
        # "streaming" lowers to the ld.global.cs cache operator
        # (evict-first at BOTH L1 and L2): single-use streaming reads
        # stop displacing useful/dirty L2 lines, which is worth ~20% on
        # row reductions whose footprint is within ~2x of L2 (measured
        # 57.3us -> 47.1us on cross-entropy 32768x2048 fp32 on B200
        # under do_bench's dirty-L2 flush; triton's evict_first hints L2
        # too, so this closes a structural gap vs the triton backend).
        # "l2_last" emits createpolicy.fractional.L2::evict_last +
        # ld.global.L2::cache_hint (inline PTX) on 16-byte hoisted vec
        # loads only — triton's evict_last equivalent, which keeps up to
        # ~L2-size of a streaming input resident across other traffic
        # (+1.7% on fp32 elementwise mul on B200).
        # Matching explicit L1/L2 priorities preserve a packet for later
        # row passes or evict it after its final use; also 16-byte loads only.
        return (
            "",
            "first",
            "last",
            "streaming",
            "l2_last",
            "l1_l2_first",
            "l1_l2_last",
        )
    return ("",)


def get_valid_load_cache_modifiers(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and supports_amd_cdna_tunables():
        return ("", ".cg")
    return ("",)


def get_valid_store_cache_modifiers(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and supports_amd_cdna_tunables():
        return ("", ".cs", ".wt")
    return ("",)


class SearchDimensionInfo(NamedTuple):
    """One tunable search dimension described from the config spec.

    ``cardinality`` is the number of distinct values (``None`` if unknown /
    unbounded); ``values`` are the explicit choices when cheaply enumerable.
    """

    name: str
    cardinality: int | None
    values: list[object] | None
    is_sequence: bool
    num_items: int


class ConfigSpec:
    def __init__(
        self,
        *,
        backend: Backend,
        user_defined_tunables: Mapping[str, ConfigSpecFragment] | None = None,
        target_device_capability: tuple[int, int]
        | object
        | None = _TARGET_DEVICE_CAPABILITY_UNSET,
        device: torch.device | None = None,
        num_sm: int | None = None,
        log_restrictions_verbose: bool = False,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.name
        self.device = device
        # When True, search-space restriction decisions are logged live (at
        # INFO) the moment they are applied, in addition to being recorded for
        # the end-of-run summary. Sourced from
        # ``Settings.autotune_log_search_space_verbose`` by the compile path
        # (see CompileEnvironment); left False otherwise so restrictions applied
        # outside verbose autotuning stay quiet.
        self.log_restrictions_verbose = log_restrictions_verbose
        self.max_reduction_threads = backend.max_reduction_threads()
        self.max_reduction_loop = backend.max_reduction_loop()
        self.reduction_loop_force_threshold = self.max_reduction_threads
        # Every reduction block, including static/non-rollable persistent
        # dimensions that have no ReductionLoopSpec.  DeviceIR fills this
        # before configs are normalized.
        self.reduction_block_ids: set[int] = set()
        self.cute_indexed_reduction_block_ids: set[int] = set()
        # Non-reduction tiles that execute simultaneously. Sibling loops and
        # separate roots reuse physical axes during CuTe launch planning.
        self.cute_tile_loop_paths: tuple[TileLoopPath, ...] = ()
        self.cute_inactive_tile_block_ids: set[int] = set()
        self.user_defined_tunables = (
            {} if user_defined_tunables is None else dict(user_defined_tunables)
        )
        # Bound kernels pass an explicit target capability. Direct CuTe specs
        # use the current CUDA device so validation still enforces arch gates.
        if target_device_capability is _TARGET_DEVICE_CAPABILITY_UNSET:
            self.target_device_capability: tuple[int, int] | None = (
                get_target_device_capability() if self.backend_name == "cute" else None
            )
        else:
            self.target_device_capability = cast(
                "tuple[int, int] | None",
                target_device_capability,
            )

        # XCD count for the *compile* device, captured once so xcd_remap's
        # support/search/normalize decisions match the device used in codegen
        # (rather than the current device).  1 disables/no-ops xcd_remap.
        self.num_xcd: int = get_num_xcd(device)
        # Persistent grid SM/CU count of the compile device (after reserved_sms);
        # used to check XCD-alignment of the persistent_interleaved grid stride.
        # Defaults to the device CU count (consistent with num_xcd) when not
        # passed explicitly by the compile path.
        self.num_sm: int = num_sm if num_sm is not None else device_num_sm(device)

        self.block_sizes: BlockIdSequence[BlockSizeSpec] = BlockIdSequence()
        self.num_threads: BlockIdSequence[NumThreadsSpec] = BlockIdSequence()
        self.loop_orders: BlockIdSequence[LoopOrderSpec] = BlockIdSequence()
        self.l2_groupings: BlockIdSequence[L2GroupingSpec] = BlockIdSequence()
        self.flatten_loops: BlockIdSequence[FlattenLoopSpec] = BlockIdSequence()
        self.reduction_loops: BlockIdSequence[ReductionLoopSpec] = BlockIdSequence()
        self.cute_vector_widths: BlockIdSequence[CuteVectorWidthSpec] = (
            BlockIdSequence()
        )
        self.cute_lane_layouts: BlockIdSequence[CuteLaneLayoutSpec] = BlockIdSequence()
        self.cute_reduction_reloads: BlockIdSequence[CuteReductionReloadSpec] = (
            BlockIdSequence()
        )
        # Device-IR facts enable this only for plausible in-place 16-bit state
        # updates. Generated-AST matching is stricter and remains authoritative.
        self.cute_resident_reduction_blocks: set[int] = set()
        self.cute_sequence_reduction_blocks: set[int] = set()
        # Persistent reduction blocks whose device IR admits the register-tile
        # lane nesting and whose extent is static (``DeviceIR`` fills this in;
        # see ``cute/register_tile_admission.py``).  Only these may keep a
        # thread count below their extent persistent.
        self.cute_register_tile_reduction_blocks: set[int] = set()
        self.cute_async_load_pipeline_enabled = False
        self.cute_bf16x2_recurrence_enabled = False
        self.cute_signed_bitfield_bf16_available = False
        self.cute_scaled_mma_available = False
        self.cute_proven_bounds_enabled = False
        self.cute_rng_packet_enabled = False
        self.cute_packet_prefetch_enabled = False
        self.range_unroll_factors: BlockIdSequence[RangeUnrollFactorSpec] = (
            BlockIdSequence()
        )
        self.range_warp_specialize: BlockIdSequence[RangeWarpSpecializeSpec] = (
            BlockIdSequence()
        )
        self.range_num_stages: BlockIdSequence[RangeNumStagesSpec] = BlockIdSequence()
        self.range_multi_buffers: BlockIdSequence[RangeMultiBufferSpec] = (
            BlockIdSequence()
        )
        self.range_flattens: BlockIdSequence[RangeFlattenSpec] = BlockIdSequence()
        self.static_ranges: BlockIdSequence[StaticRangeSpec] = BlockIdSequence()

        self.allowed_pid_types: tuple[PidTypeLiteral, ...] = tuple(VALID_PID_TYPES)
        # Why each disabled pid_type was removed, for search-space logging.
        self.disallowed_pid_type_reasons: dict[str, str] = {}
        # (feature, reason) pairs for every non-pid_type search-space restriction
        # (currently the tcgen05 narrowing applied per matmul), for search-space
        # logging. De-duplicated; ordered by first occurrence.
        self.restriction_reasons: list[tuple[str, str]] = []
        self.max_num_sm_multiplier: int = MAX_NUM_SM_MULTIPLIER
        self.grid_block_ids: list[int] = []
        # Ordered root grids and nested-graph ownership, independent of any
        # specialized matmul/reduction policy.
        self.kernel_grid_fact: KernelGridFact | None = None
        self.tensor_numel_constraints: list[TensorNumelConstraint] = []
        self.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(self.backend_name)),
            length=0,
        )
        self.load_cache_modifiers = ListOf(
            EnumFragment(choices=get_valid_load_cache_modifiers(self.backend_name)),
            length=0,
        )
        self.store_cache_modifiers = ListOf(
            EnumFragment(choices=get_valid_store_cache_modifiers(self.backend_name)),
            length=0,
        )
        self.indexing = ListOf(
            EnumFragment(choices=self.valid_indexing_types()),
            length=0,
        )
        self.atomic_indexing = ListOf(
            EnumFragment(choices=self.valid_atomic_indexing_types()),
            length=0,
        )
        self.pallas_load_buffer_count = ListOf(
            IntegerFragment(1, 2, 1),
            length=0,
        )
        self.epilogue_subtile_candidate_enabled: bool = False
        self.epilogue_subtile_autotune_choices: tuple[int | None, ...] | None = None
        self.epilogue_subtile_k_hint: int = 0
        self.has_pallas_inner_loops: bool = False
        # Set by the Pallas graph lowering when an inner-loop dot can consume
        # an f32 producer directly instead of its explicit bf16/f16 cast.
        self.pallas_fold_dot_lhs_cast_search_enabled: bool = False
        self.pallas_indirect_access_modes: tuple[str, ...] = ()
        self.pallas_indirect_dma_requires_fori: bool = False
        self.has_symbolic_or_data_dependent_bounds: bool = False
        # Populated only after DeviceIR proves that this kernel contains an
        # implicit cross-root dependency supported by the CUDA Triton backend.
        self.cross_loop_pipeline: EnumFragment | None = None
        # Enabled only when the exact five-factor BT16 recurrence carrier is
        # detected. Choice ordering makes the geometry seed the no-autotune
        # default while leaving both legal schedules in cold/full search.
        self.cute_chunk_recurrence_dv_partitions: EnumFragment | None = None
        # Enabled by the same exact recurrence matcher. The backend turns the
        # selected value into a ptxas max-register constraint; unrelated CuTe
        # kernels never see this search dimension.
        self.cute_chunk_recurrence_register_cap: EnumFragment | None = None
        # Enabled only when the gated-delta-rule chunk recurrence (gdn_fwd_h)
        # is matched. Choices are derived from the matched geometry (shared
        # memory ring budget and TMEM column slicing); first choice is default.
        self.cute_gdn_recurrence_stages: EnumFragment | None = None
        self.cute_gdn_recurrence_epilogue_warps: EnumFragment | None = None
        self.cute_gdn_recurrence_token_groups: EnumFragment | None = None
        self.cute_gdn_recurrence_mma_m: EnumFragment | None = None
        # Each fragment is the union over the admitted dstate tiles;
        # ``normalize`` re-validates a config's values against the tile its
        # ``block_sizes`` entry for ``value_block_id`` selects: the tcgen05 M
        # per (tile, epilogue warps), the ring depth per (tile, M) and the
        # token groups per (tile, epilogue warps, M).
        self.cute_gdn_recurrence_value_block_id: int | None = None
        self.cute_gdn_recurrence_mma_m_choices_by_tile: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self.cute_gdn_recurrence_stage_choices_by_tile: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self.cute_gdn_recurrence_token_group_choices_by_tile: dict[
            tuple[int, int, int], tuple[int, ...]
        ] = {}
        # Enabled only when the exact five-factor BT16 chunk-prepare carrier is
        # detected. Choice order defines the default and ranked seed order.
        self.cute_chunk_prepare_schedule: EnumFragment | None = None
        # Enabled only after a generic matcher proves a compatible affine scan.
        # The first choice is the semantic-neutral ordinary lowering.
        self.cute_affine_scan_schedule: EnumFragment | None = None
        # Enabled only when the whole root is a supported top-k load/store
        # dataflow. Its emitter owns the row geometry and vector layout.
        self.cute_topk_search_enabled = False
        self.cute_topk_choices = dict(CUTE_TOPK_CHOICES)
        self._cute_tcgen05_config = CuteTcgen05Config(self)
        self.cute_host_paired_sum_available: bool = False
        # A separately launched, proved pointwise producer can share one
        # Config with a native GEMM. Keep its ordinary SIMT layout knobs in
        # the native search schema; MMA-owned axes retain their auto layout.
        self.cute_pointwise_region_block_ids: frozenset[int] = frozenset()
        # Separately launched pointwise grids whose axes have one root owner.
        # Their search floors use the rank of that launch, independently of
        # any native MMA region sharing the complete kernel configuration.
        self.cute_pointwise_region_grid_groups: tuple[tuple[int, ...], ...] = ()
        self.cute_split_k_cluster_facts: ClusterKFacts | None = None
        self.cute_split_k_workspace_available: bool = False
        self.cute_materialized_schedule_available: bool = False
        self.cute_materialized_schedule_search_enabled: bool = False
        self.cute_row_matrix_transport_available: bool = False
        self.cute_fragment_scan_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_scan_search_enabled = False
        self.cute_fragment_reduction_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_reduction_search_enabled = False
        self.cute_fragment_thread_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_threads_search_enabled = False
        self.cute_fragment_register_load_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_register_loads_search_enabled = False
        self.cute_fragment_warp_result_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_warp_results_search_enabled = False
        self.cute_fragment_warp_result_min_threads = 32
        self.cute_materialized_operand_schedule_available: bool = False
        self.cute_materialized_operand_schedule_search_enabled: bool = False
        self.cute_grouped_rna_k_choices: tuple[int, ...] = ()
        self.cute_grouped_warp_tf32_available = False
        self.cute_grouped_rn_two_stage_ctas: tuple[int, ...] = ()
        self.cute_grouped_block_prefix_recipes: tuple[tuple[int, int, int], ...] = ()
        self.cute_grouped_block_prefix_seed_enabled = False
        self.cute_grouped_wrapped_descriptors_available = False
        # CuTe flash-attention autotune surface gating.
        # Default False so the flash knobs never appear in the search surface
        # and behavior is byte-identical to the env-only path. Set True when the
        # flash detector fires (see ``lower_to_device_ir``). The shape needed to
        # build the fragments (head_dim / num_kv) is captured at the same time.
        self.cute_flash_search_enabled: bool = False
        # Gated (softmax-free) attention surface: pins the 128x128 tile shape
        # and exposes only the K/V TMA ring depth. See ``cute_flash_gated``.
        self.cute_flash_gated_search_enabled: bool = False
        self._cute_flash_gated_block_size_targets: dict[int, int] = {}
        self._cute_flash_gated_kv_block_id: int | None = None
        self._cute_flash_gated_q_block_id: int | None = None
        self._cute_flash_gated_q_tile_choices: tuple[int, ...] = (128,)
        self._cute_flash_gated_kv_tile_choices: tuple[int, ...] = ()
        self._cute_flash_gated_kv_stage_choices: tuple[int, ...] = ()
        self._cute_flash_gated_kv_stage_default: int = 2
        self._cute_flash_gated_gate_warpgroup_choices: tuple[int, ...] = (1,)
        self._cute_flash_gated_gate_warpgroup_default: int = 1
        self._cute_flash_gated_kv_stage_choices_by_tile: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self._cute_flash_head_dim: int | None = None
        self._cute_flash_num_kv: int | None = None
        self._cute_flash_num_bh: int | None = None
        # SM count of the bound device (0 when unknown); small grids seed the
        # 64-row flash tile from it.
        self._cute_flash_device_sm_count: int = 0
        self._cute_flash_tensor_4d_heads: int | None = None
        self._cute_flash_dtype: torch.dtype = torch.float16
        self._cute_flash_is_causal: bool = False
        self._cute_flash_has_kv_tile_pruning: bool = False
        self._cute_flash_requires_ws_overlap: bool = False
        self._cute_flash_small_biased_candidate: bool = False
        self._cute_flash_standard_dense_output: bool = False
        self._cute_flash_standard_causal_output: bool = False
        self._cute_flash_output_requires_tma: bool = False
        self._cute_flash_supports_tensor_4d_tma: bool = True
        self._cute_flash_has_row_epilogue: bool = False
        self._cute_flash_plain_row_body: bool = True
        self._cute_flash_has_score_modifiers: bool = False
        self._cute_flash_block_size_targets: dict[int, int] = {}
        # Memo for ``flash_autotune_fragments``: every other input is fixed
        # ConfigSpec state, so (topology_override, pipeline_family_override) is
        # a complete key once the env fingerprint below matches. The fragments
        # builder resolves dozens of configs per call and normalization calls
        # it for every candidate, so this cache dominates coverage-design cost.
        self._cute_flash_fragments_cache: dict[
            tuple[str | None, str | None], dict[str, ConfigSpecFragment]
        ] = {}
        self._cute_flash_fragments_env_fingerprint: (
            tuple[tuple[str, str], ...] | None
        ) = None
        # Some large attention outputs require the FA4-only TMA epilogue, but
        # odd KV-tile counts and some score plans require the incompatible WS
        # flash topology. Keep that narrow fallback explicit so codegen does
        # not rediscover the unavailable flash path for a 128x128 candidate.
        self.cute_attention_generic_fallback_enabled: bool = False
        self._cute_attention_generic_fallback_block_size_targets: dict[int, int] = {}
        # CuTe flash-attention BACKWARD surface: set when the fused
        # backward-attention detector fires (see cute_flash_bwd.py). Pins the
        # (kv_tile, q_tile) block sizes to the 128x128 flash envelope.
        self.cute_flash_bwd_search_enabled: bool = False
        self._cute_flash_bwd_block_size_targets: dict[int, int] = {}
        self._cute_flash_bwd_two_cta_allowed: bool = False
        self.compiler_default_config: helion.Config | None = None
        self.compiler_seed_configs: list[helion.Config] = []
        self._compiler_coverage_groups: tuple[CompilerCoverageGroup, ...] = ()
        self._compiler_coverage_fields: tuple[tuple[str | int, ...], ...] = ()
        self.cute_matmul_min_blocks_search_enabled: bool = False
        # Compiler paths can opt their seeds into a single bounded timeout
        # retry. ``None`` leaves all benchmark behavior unchanged.
        self.compiler_seed_timeout_retry_repetitions: int | None = None
        self.autotuner_heuristics: list[str] = []
        self.matmul_facts: list[MatmulFact] = []
        # Whole-kernel composed contraction fact (built for ANY kernel with >=1
        # matmul fact); both matmul front ends read it so workload analysis is
        # independent of which heuristic runs.
        self.kernel_matmul_fact: KernelMatmulFact | None = None
        # The Stage-1 categorizing product the reduction seed + allocator consume.
        self.reduction_kernel_fact: ReductionKernelFact | None = None
        self.matmul_reduction_epilogue_facts: list[MatmulWithReductionEpilogueFact] = []
        self.accumulator_facts: list[AccumulatorFact] = []
        self.pointwise_facts: list[PointwiseElementwiseFact] = []
        self.store_indices: list[int] = []
        self.memory_op_facts: list[MemoryOpFact] = []
        # FlattenLoopSpecs dropped by the vector-model partial-access gate
        # (``update_allow_flattened``); re-registered for PURE pointwise
        # kernels on cute once device-IR analysis proves the fact.
        self.cute_reflatten_candidates: list[FlattenLoopSpec] = []
        self.backend_tunable_fragments = self.backend.tunable_fragments()
        unknown_tunables = set(self.backend_tunable_fragments) - BACKEND_TUNABLE_KEYS
        if unknown_tunables:
            raise RuntimeError(
                f"Backend {self.backend_name!r} returned unknown tunables: {sorted(unknown_tunables)!r}"
            )

    @property
    def compiler_coverage_groups(self) -> tuple[CompilerCoverageGroup, ...]:
        return self._compiler_coverage_groups

    def register_compiler_coverage_group(self, group: CompilerCoverageGroup) -> None:
        """Register ranked coverage after ordinary facts/defaults/seeds are built.

        Consumers expose and normalize their own optional scalar field. They
        must not change old field domains to make a coverage mode admissible.
        No user config, compiler default or seed is changed by registration.
        """
        for current in self._compiler_coverage_groups:
            if current.mechanism == group.mechanism or current.key == group.key:
                raise ValueError("Duplicate compiler coverage mechanism or field")
        fields = self._flat_fields()
        fragment = fields.get(group.key)
        if not isinstance(fragment, ConfigSpecFragment):
            raise ValueError(f"Unknown or non-scalar coverage field {group.key!r}")
        group.validate_field(fragment)
        group.validate_dependencies(self._compiler_coverage_groups)
        groups = (*self._compiler_coverage_groups, group)
        owned = {entry.key for entry in groups}
        fingerprint = tuple(
            (key, *value.fingerprint()) for key, value in fields.items()
        )
        if self._compiler_coverage_groups and tuple(
            row for row in self._compiler_coverage_fields if row[0] not in owned
        ) != tuple(row for row in fingerprint if row[0] not in owned):
            raise ValueError("Compiler coverage registration changed old field layout")
        self._compiler_coverage_groups = groups
        self._compiler_coverage_fields = fingerprint

    def validate_compiler_coverage_groups(self) -> None:
        """Reject stale domains before constructing an immutable initial view."""
        if not self._compiler_coverage_groups:
            return
        fields = self._flat_fields()
        fingerprint = tuple(
            (key, *value.fingerprint()) for key, value in fields.items()
        )
        if fingerprint != self._compiler_coverage_fields:
            raise ValueError(
                "Config fields changed after compiler coverage registration"
            )
        for index, group in enumerate(self._compiler_coverage_groups):
            fragment = fields[group.key]
            if not isinstance(fragment, ConfigSpecFragment):
                raise ValueError("Compiler coverage requires independent scalar fields")
            group.validate_field(fragment)
            group.validate_dependencies(self._compiler_coverage_groups[:index])

    def _should_keep_epilogue_subtile_for_autotune(self) -> bool:
        if self.epilogue_subtile_autotune_choices is None:
            return False
        return supports_tensor_descriptor()

    def fix_epilogue_subtile_store_indexing(self, config: dict[str, object]) -> None:
        """Force subtiled store indexing to tensor_descriptor for correctness."""
        if (
            not self.epilogue_subtile_candidate_enabled
            or "epilogue_subtile" not in config
        ):
            return
        indexing = config.get("indexing")
        if isinstance(indexing, list):
            for i in self.store_indices:
                indexing[i] = "tensor_descriptor"

    @staticmethod
    def _infer_epilogue_subtile_k_hint(args: Sequence[object]) -> int:
        def _as_concrete_dim(dim: object) -> int | None:
            return dim if type(dim) is int else None

        tensor_args = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim >= 2
        ]
        best = 0
        for lhs, rhs in itertools.combinations(tensor_args, 2):
            candidates: list[int] = []
            lhs_last = _as_concrete_dim(lhs.shape[-1])
            lhs_prev = _as_concrete_dim(lhs.shape[-2])
            rhs_last = _as_concrete_dim(rhs.shape[-1])
            rhs_prev = _as_concrete_dim(rhs.shape[-2])
            if lhs_last is not None and rhs_prev is not None and lhs_last == rhs_prev:
                candidates.append(lhs_last)
            if lhs_prev is not None and rhs_last is not None and lhs_prev == rhs_last:
                candidates.append(lhs_prev)
            if candidates:
                best = max(best, *candidates)
        return best

    def configure_epilogue_subtile_autotune(self, args: Sequence[object]) -> None:
        self.epilogue_subtile_k_hint = self._infer_epilogue_subtile_k_hint(args)
        arch = self.target_device_capability
        if arch is None:
            self.epilogue_subtile_autotune_choices = None
            return

        if arch >= (10, 0):
            arch_enabled = (
                self.epilogue_subtile_candidate_enabled and supports_tensor_descriptor()
            )
        else:
            arch_enabled = False

        enabled = (
            arch_enabled and self.epilogue_subtile_k_hint >= EPILOGUE_SUBTILE_MIN_K_HINT
        )
        if not enabled:
            self.epilogue_subtile_autotune_choices = None
        elif (
            arch >= (10, 0)
            and self.epilogue_subtile_k_hint >= EPILOGUE_SUBTILE_MIN_K_HINT_EXTENDED
        ):
            self.epilogue_subtile_autotune_choices = EPILOGUE_SUBTILE_EXTENDED_CHOICES
        else:
            self.epilogue_subtile_autotune_choices = EPILOGUE_SUBTILE_DEFAULT_CHOICES

    def valid_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        if not self.backend.supports_block_ptr_indexing():
            return ("pointer",)
        return ("pointer", "block_ptr")

    def valid_atomic_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        """Atomic ops only support pointer and tensor_descriptor (no block_ptr)."""
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        return ("pointer",)

    def _remove_duplicates(self) -> None:
        self.num_threads._remove_duplicates()
        self.loop_orders._remove_duplicates()
        self.l2_groupings._remove_duplicates()
        self.flatten_loops._remove_duplicates()
        self.range_unroll_factors._remove_duplicates()
        self.range_warp_specialize._remove_duplicates()
        self.range_num_stages._remove_duplicates()
        self.range_multi_buffers._remove_duplicates()
        self.range_flattens._remove_duplicates()
        self.static_ranges._remove_duplicates()

    def disallow_pid_type(
        self, pid_type: PidTypeLiteral, reason: str | None = None
    ) -> None:
        """Disallow a pid_type from being used in the config.

        ``reason`` explains why the pid_type is unavailable for this kernel; it is
        recorded (first reason wins) and surfaced by the search-space logger. When
        verbose search-space logging is enabled it is also logged live.
        """

        newly_disabled = pid_type in self.allowed_pid_types
        if newly_disabled and reason is not None:
            self.disallowed_pid_type_reasons.setdefault(pid_type, reason)
        self.allowed_pid_types = tuple(
            [x for x in self.allowed_pid_types if x != pid_type]
        )
        assert self.allowed_pid_types
        if newly_disabled and reason is not None and self.log_restrictions_verbose:
            _live_log_restriction(
                f"pid_type={pid_type!r} disabled", reason, self.log_restrictions_verbose
            )

    @property
    def cute_tcgen05_search_enabled(self) -> bool:
        return self._cute_tcgen05_config.search_enabled

    @cute_tcgen05_search_enabled.setter
    def cute_tcgen05_search_enabled(self, value: bool) -> None:
        self._cute_tcgen05_config.search_enabled = value

    @property
    def cute_tcgen05_aux_kernel_detected(self) -> bool:
        return self._cute_tcgen05_config.aux_kernel_detected

    @cute_tcgen05_aux_kernel_detected.setter
    def cute_tcgen05_aux_kernel_detected(self, value: bool) -> None:
        self._cute_tcgen05_config.aux_kernel_detected = value

    @property
    def cute_tcgen05_exact_shape_aux_kernel_detected(self) -> bool:
        return self._cute_tcgen05_config.exact_shape_aux_kernel_detected

    @cute_tcgen05_exact_shape_aux_kernel_detected.setter
    def cute_tcgen05_exact_shape_aux_kernel_detected(self, value: bool) -> None:
        self._cute_tcgen05_config.exact_shape_aux_kernel_detected = value

    @property
    def cute_tcgen05_matmul_has_non_tcgen05_operand(self) -> bool:
        return self._cute_tcgen05_config.matmul_has_non_tcgen05_operand

    @cute_tcgen05_matmul_has_non_tcgen05_operand.setter
    def cute_tcgen05_matmul_has_non_tcgen05_operand(self, value: bool) -> None:
        self._cute_tcgen05_config.matmul_has_non_tcgen05_operand = value

    @property
    def cute_tcgen05_rowvec_aux_facts(self) -> Tcgen05RowvecAuxFacts | None:
        return self._cute_tcgen05_config.rowvec_aux_facts

    @cute_tcgen05_rowvec_aux_facts.setter
    def cute_tcgen05_rowvec_aux_facts(
        self, value: Tcgen05RowvecAuxFacts | None
    ) -> None:
        self._cute_tcgen05_config.rowvec_aux_facts = value

    def _cute_flash_autotune_fragments(
        self,
        topology_override: str | None = None,
        pipeline_family_override: str | None = None,
    ) -> dict[str, ConfigSpecFragment]:
        """Memoized ``flash_autotune_fragments`` for this spec's flash surface.

        The env fingerprint covers every ``HELION_CUTE_FLASH_*`` variable the
        builder reads transitively, so patched environments (tests, scripts)
        invalidate the memo instead of seeing stale fragments. The function is
        resolved through the module attribute on every miss so tests that
        ``patch.object`` it still intercept the real computation.
        """
        # Fragment values are shared: they are immutable dataclasses that all
        # callers replace rather than mutate. The mapping is copied because
        # callers merge it into their own field dicts.
        fingerprint = flash_env_fingerprint()
        if fingerprint != self._cute_flash_fragments_env_fingerprint:
            self._cute_flash_fragments_cache.clear()
            self._cute_flash_fragments_env_fingerprint = fingerprint
        key = (topology_override, pipeline_family_override)
        cached = self._cute_flash_fragments_cache.get(key)
        if cached is None:
            from .._compiler.cute.cute_flash import flash_autotune_fragments

            assert self._cute_flash_head_dim is not None
            assert self._cute_flash_num_kv is not None
            cached = flash_autotune_fragments(
                self._cute_flash_head_dim,
                self._cute_flash_num_kv,
                num_bh=self._cute_flash_num_bh,
                tensor_4d_heads=self._cute_flash_tensor_4d_heads,
                dtype=self._cute_flash_dtype,
                is_causal=self._cute_flash_is_causal,
                has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                small_biased_candidate=self._cute_flash_small_biased_candidate,
                standard_dense_output=self._cute_flash_standard_dense_output,
                standard_causal_output=self._cute_flash_standard_causal_output,
                target_device_capability=self.target_device_capability,
                output_requires_tma=self._cute_flash_output_requires_tma,
                supports_tensor_4d_tma=self._cute_flash_supports_tensor_4d_tma,
                has_row_epilogue=self._cute_flash_has_row_epilogue,
                plain_row_body=self._cute_flash_plain_row_body,
                has_score_modifiers=self._cute_flash_has_score_modifiers,
                topology_override=topology_override,
                pipeline_family_override=pipeline_family_override,
            )
            self._cute_flash_fragments_cache[key] = cached
        return dict(cached)

    def _resolve_cute_flash_config(
        self, config: Mapping[str, object]
    ) -> FlashAttentionConfig:
        assert self._cute_flash_head_dim is not None
        assert self._cute_flash_num_kv is not None
        flash_config = config
        if self._cute_flash_requires_ws_overlap:
            flash_config = {**config, FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
        return resolve_flash_config(
            self._cute_flash_head_dim,
            self._cute_flash_num_kv,
            flash_config,
            dtype=self._cute_flash_dtype,
            num_bh=self._cute_flash_num_bh,
            is_causal=self._cute_flash_is_causal,
            has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=self._cute_flash_requires_ws_overlap,
            small_biased_candidate=self._cute_flash_small_biased_candidate,
            standard_dense_output=self._cute_flash_standard_dense_output,
            standard_causal_output=self._cute_flash_standard_causal_output,
            supports_tensor_4d_tma=self._cute_flash_supports_tensor_4d_tma,
            prefer_packed_reduce=(
                self._cute_flash_has_kv_tile_pruning
                or self._cute_flash_requires_ws_overlap
            ),
            plain_row_body=self._cute_flash_plain_row_body,
            has_row_epilogue=self._cute_flash_has_row_epilogue,
            has_score_modifiers=self._cute_flash_has_score_modifiers,
        )

    def _legalize_cute_flash_compiler_seed(
        self, seed: helion.Config | None
    ) -> helion.Config | None:
        """Project compiler-owned causal seeds onto a required TMA output path."""
        if seed is None or not self._cute_flash_output_requires_tma:
            return seed
        if self._resolve_cute_flash_config(seed.config).epi_tma:
            return seed
        if not self._cute_flash_is_causal:
            return None
        projected = helion.Config.from_dict(
            {
                **seed.config,
                FLASH_EPI_TMA_KEY: True,
                FLASH_EPI_STG_KEY: False,
                FLASH_EPI_STG_STORE_KEY: "slice",
                FLASH_EPI_STG_GMEM_KEY: "stage",
            }
        )
        effective = self._resolve_cute_flash_config(projected.config)
        return projected if effective.epi_tma and not effective.epi_stg else None

    def _legalize_cute_flash_compiler_seeds(
        self, seeds: Sequence[helion.Config]
    ) -> list[helion.Config]:
        """Legalize and deduplicate compiler seeds while preserving rank order."""
        result: list[helion.Config] = []
        seen: set[helion.Config] = set()
        for seed in seeds:
            projected = self._legalize_cute_flash_compiler_seed(seed)
            if projected is not None and projected not in seen:
                result.append(projected)
                seen.add(projected)
        return result

    def _normalize_cute_flash(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """Normalize the flash-attention autotune surface.

        Only runs when ``cute_flash_search_enabled`` is set (the flash detector
        fired). Active keys are validated against their fragments, while legacy
        structural keys are projected into the compound pipeline-family key and
        removed before the config is used as an autotune identity.
        """
        if not self.cute_flash_search_enabled:
            return
        assert self._cute_flash_head_dim is not None
        assert self._cute_flash_num_kv is not None
        # ``q_tile_count`` is a family-derived invariant and is projected back
        # to its canonical value below.  The historical MMA key accepted any
        # truthy object, so preserve that fixed-config compatibility before the
        # active Boolean fragment validates it.
        if FLASH_MMA_INTERLEAVE_KEY in config:
            config[FLASH_MMA_INTERLEAVE_KEY] = bool(config[FLASH_MMA_INTERLEAVE_KEY])
        causal_lpt = config.get(FLASH_CAUSAL_LPT_SWIZZLE_KEY)
        if self._cute_flash_is_causal and type(causal_lpt) is int:
            config[FLASH_CAUSAL_LPT_SWIZZLE_KEY] = 1
        block_size_targets = self._cute_flash_block_size_target_list()
        if fix_invalid:
            config["block_sizes"] = list(block_size_targets)
            config["pid_type"] = "flat"
            self._normalize_cute_flash_default_sequence(config, "l2_groupings", 1)
            self._normalize_cute_flash_default_sequence(config, "num_threads", 0)
            self._normalize_cute_flash_default_sequence(config, "cute_vector_widths", 1)
            self._normalize_cute_flash_default_sequence(
                config, "cute_lane_layouts", "blocked"
            )
            self._normalize_cute_flash_default_loop_orders(config)
            config.pop("epilogue_subtile", None)
        elif not self._is_cute_flash_config_envelope(config, block_size_targets):
            return

        config.update(
            _flash_compound_exp2_packet_overrides(
                self._cute_flash_head_dim,
                self._cute_flash_num_kv,
                config,
                dtype=self._cute_flash_dtype,
                is_causal=self._cute_flash_is_causal,
                has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                small_biased_candidate=self._cute_flash_small_biased_candidate,
                standard_dense_output=self._cute_flash_standard_dense_output,
                standard_causal_output=self._cute_flash_standard_causal_output,
            )
        )

        has_legacy_structural_config = any(
            config.get(key) is not None for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
        )
        legacy_effective = None
        if has_legacy_structural_config and FLASH_PIPELINE_FAMILY_KEY not in config:
            legacy_resolution_config = {
                key: config[key]
                for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
                if key in config
            }
            if FLASH_PERSISTENT_KEY in config:
                legacy_resolution_config[FLASH_PERSISTENT_KEY] = config[
                    FLASH_PERSISTENT_KEY
                ]
            legacy_effective = self._resolve_cute_flash_config(legacy_resolution_config)
        if self._cute_flash_requires_ws_overlap:
            config[FLASH_PIPELINE_FAMILY_KEY] = "ws_overlap"
            topology_override = "ws_overlap"
            pipeline_family_override = "ws_overlap"
        else:
            requested_family = _flash_pipeline_family_flags(
                config.get(FLASH_PIPELINE_FAMILY_KEY)
            )
            if requested_family is not None:
                requested_family_name = str(config[FLASH_PIPELINE_FAMILY_KEY])
                if (
                    requested_family.tensor_4d_tma
                    and not self._cute_flash_supports_tensor_4d_tma
                ):
                    effective_family = self._resolve_cute_flash_config(
                        {FLASH_PIPELINE_FAMILY_KEY: requested_family_name}
                    ).pipeline_family
                    pipeline_family_override = effective_family
                    effective_flags = _flash_pipeline_family_flags(effective_family)
                    assert effective_flags is not None
                    topology_override = effective_flags.topology
                else:
                    pipeline_family_override = requested_family_name
                    topology_override = requested_family.topology
            else:
                pipeline_family_override = (
                    legacy_effective.pipeline_family
                    if legacy_effective is not None
                    else None
                )
                valid_manual_topologies = {"fa4", "ws_overlap"}
                topology_value = config.get(FLASH_TOPOLOGY_KEY)
                topology_override = (
                    topology_value
                    if topology_value in valid_manual_topologies
                    else None
                )

        def make_fragments(
            topology: str | None,
            family: str | None,
        ) -> dict[str, ConfigSpecFragment]:
            return self._cute_flash_autotune_fragments(topology, family)

        if (
            fix_invalid
            and self._cute_flash_output_requires_tma
            and pipeline_family_override is not None
        ):
            requested_effective = self._resolve_cute_flash_config(
                {
                    FLASH_PIPELINE_FAMILY_KEY: pipeline_family_override,
                    FLASH_EPI_TMA_KEY: True,
                    FLASH_EPI_STG_KEY: False,
                }
            )
            if not requested_effective.epi_tma:
                config.pop(FLASH_PIPELINE_FAMILY_KEY, None)
                topology_override = None
                pipeline_family_override = None

        fragments = make_fragments(
            cast("str | None", topology_override), pipeline_family_override
        )
        e2e_offset_was_present = FLASH_E2E_OFFSET_KEY in config
        e2e_offset0_was_present = FLASH_E2E_OFFSET0_KEY in config
        e2e_offset_keys = (FLASH_E2E_OFFSET_KEY, FLASH_E2E_OFFSET0_KEY)
        explicit_e2e_offsets = {
            key: config[key] for key in e2e_offset_keys if key in config
        }
        for key, fragment in fragments.items():
            choices = cast("EnumFragment", fragment).choices
            if key in config:
                if config[key] not in choices:
                    if key in e2e_offset_keys:
                        # Legacy explicit e2e frequency overrides can make offsets
                        # outside the autotune fragment valid. Validate the effective
                        # cadence after the e2e keys have been normalized below.
                        pass
                    elif fix_invalid:
                        config[key] = fragment.default()
                    else:
                        raise InvalidConfig(
                            f"{key} must be one of {list(choices)!r}, "
                            f"got {config[key]!r}"
                        )
            else:
                if key not in (*e2e_offset_keys, FLASH_PIPELINE_FAMILY_KEY):
                    config[key] = fragment.default()
        if FLASH_PIPELINE_FAMILY_KEY not in config:
            family_fragment = fragments[FLASH_PIPELINE_FAMILY_KEY]
            config[FLASH_PIPELINE_FAMILY_KEY] = (
                legacy_effective.pipeline_family
                if legacy_effective is not None
                else family_fragment.default()
            )

        effective = self._resolve_cute_flash_config(config)
        if self._cute_flash_output_requires_tma and not effective.epi_tma:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{FLASH_EPI_TMA_KEY}=False is not legal for this output shape"
                )
            config[FLASH_EPI_TMA_KEY] = True
            config[FLASH_EPI_STG_KEY] = False
            effective = self._resolve_cute_flash_config(config)
            if not effective.epi_tma:
                repair_fragments = make_fragments(None, None)
                family_fragment = cast(
                    "EnumFragment", repair_fragments[FLASH_PIPELINE_FAMILY_KEY]
                )
                for family in family_fragment.choices:
                    repaired = self._resolve_cute_flash_config(
                        {
                            **config,
                            FLASH_PIPELINE_FAMILY_KEY: family,
                            FLASH_EPI_TMA_KEY: True,
                            FLASH_EPI_STG_KEY: False,
                        }
                    )
                    if repaired.epi_tma:
                        config[FLASH_PIPELINE_FAMILY_KEY] = family
                        effective = repaired
                        fragments = repair_fragments
                        break
            if not effective.epi_tma:
                raise InvalidConfig(
                    "CuTe flash output requires TMA, but no legal TMA output "
                    "schedule is available"
                )
        effective_topology = effective.topology
        config.update(flash_effective_config_values(effective))
        if effective_topology == "fa4":
            config.update(explicit_e2e_offsets)
            if effective.alternating_warpgroups:
                # Both softmax warpgroups of the alternating family process
                # the same rows: one emulation phase. Keep the resolver's pin
                # in the search identity so offsets differing only in the
                # second phase do not name the same program twice.
                config[FLASH_E2E_OFFSET0_KEY] = config[FLASH_E2E_OFFSET_KEY]
        e2e_schedule_default = _flash_e2e_schedule_default(
            effective_topology, self._cute_flash_head_dim
        )
        exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
            str(config[FLASH_E2E_SCHEDULE_KEY]), e2e_schedule_default
        )
        if FLASH_EXP2_IMPL_KEY in config:
            exp2_impl = str(config[FLASH_EXP2_IMPL_KEY])
        if FLASH_E2E_FREQ_KEY in config:
            e2e_freq = cast("int", config[FLASH_E2E_FREQ_KEY])
        if FLASH_E2E_RES_KEY in config:
            e2e_res = cast("int", config[FLASH_E2E_RES_KEY])
        _impl, e2e_freq, e2e_res, _schedule = _flash_normalize_e2e_params(
            exp2_impl,
            e2e_freq,
            e2e_res,
            e2e_schedule_default,
        )
        masked_e2e_schedule = str(config.get(FLASH_MASKED_E2E_SCHEDULE_KEY, "inherit"))
        _masked_schedule, masked_e2e_freq, masked_e2e_res = (
            _flash_masked_e2e_schedule_params(
                masked_e2e_schedule,
                e2e_schedule_default,
                e2e_freq,
                e2e_res,
            )
        )
        if not self._cute_flash_is_causal:
            masked_e2e_freq = e2e_freq
            masked_e2e_res = e2e_res
        e2e_offset_period = _flash_e2e_offset_period(
            e2e_freq,
            e2e_res,
            masked_e2e_freq,
            masked_e2e_res,
        )
        if (
            e2e_offset_period > 0
            and effective_topology == "fa4"
            and self._cute_flash_head_dim == 64
        ):
            split_default_freq = e2e_freq if e2e_res > 0 else masked_e2e_freq
            schedule_default_offset = split_default_freq // 8
        else:
            schedule_default_offset = 0
        default_offset = schedule_default_offset
        env_offset = _flash_env_get("HELION_CUTE_FLASH_E2E_OFFSET")
        if env_offset is not None:
            default_offset = int(env_offset)
            if e2e_offset_period == 0:
                default_offset = 0
            elif default_offset < 0:
                default_offset = schedule_default_offset
            else:
                default_offset %= e2e_offset_period
        if not e2e_offset_was_present:
            config[FLASH_E2E_OFFSET_KEY] = default_offset
        default_offset0 = 0
        env_offset0 = _flash_env_get("HELION_CUTE_FLASH_E2E_OFFSET0")
        if env_offset0 is not None:
            env_offset0_value = int(env_offset0)
            if e2e_offset_period == 0:
                default_offset0 = 0
            elif env_offset0_value < 0:
                default_offset0 %= e2e_offset_period
            else:
                default_offset0 = env_offset0_value % e2e_offset_period
        if not e2e_offset0_was_present:
            config[FLASH_E2E_OFFSET0_KEY] = default_offset0
        for key, default in (
            (FLASH_E2E_OFFSET_KEY, default_offset),
            (FLASH_E2E_OFFSET0_KEY, default_offset0),
        ):
            e2e_offset_value = config[key]
            if not isinstance(e2e_offset_value, int):
                if fix_invalid:
                    config[key] = default
                    e2e_offset_value = default
                else:
                    raise InvalidConfig(
                        f"{key} must be an integer, got {e2e_offset_value!r}"
                    )
            e2e_offset = e2e_offset_value
            e2e_offset_invalid = (
                e2e_offset != 0
                if e2e_offset_period == 0
                else e2e_offset < 0 or e2e_offset >= e2e_offset_period
            )
            if e2e_offset_invalid:
                if fix_invalid:
                    config[key] = _flash_normalize_e2e_offset(
                        e2e_offset, default, e2e_offset_period
                    )
                else:
                    expected = (
                        [0]
                        if e2e_offset_period == 0
                        else list(range(e2e_offset_period))
                    )
                    raise InvalidConfig(
                        f"{key} must be one of {expected!r} for "
                        f"{FLASH_E2E_SCHEDULE_KEY}={config[FLASH_E2E_SCHEDULE_KEY]!r}, "
                        f"got {e2e_offset!r}"
                    )
        self._normalize_cute_flash_register_budget(
            config,
            fragments,
            effective_topology,
            fix_invalid=fix_invalid,
        )
        for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            config.pop(key, None)

    def _normalize_cute_flash_register_budget(
        self,
        config: dict[str, object],
        fragments: Mapping[str, ConfigSpecFragment],
        effective_topology: str,
        *,
        fix_invalid: bool,
    ) -> None:
        if effective_topology != "fa4":
            return
        softmax_regs = config.get(FLASH_SOFTMAX_REGS_KEY)
        corr_regs = config.get(FLASH_CORR_REGS_KEY)
        other_regs = config.get(FLASH_OTHER_REGS_KEY)
        if (
            type(softmax_regs) is not int
            or type(corr_regs) is not int
            or type(other_regs) is not int
        ):
            return
        budget = 2 * softmax_regs + corr_regs + other_regs
        if budget <= 512:
            return
        message = (
            "FA4 register budget exceeds 512: "
            f"2 * {FLASH_SOFTMAX_REGS_KEY} ({softmax_regs}) + "
            f"{FLASH_CORR_REGS_KEY} ({corr_regs}) + "
            f"{FLASH_OTHER_REGS_KEY} ({other_regs}) = {budget}"
        )
        if not fix_invalid:
            raise InvalidConfig(message)

        def _int_choices(key: str) -> tuple[int, ...]:
            fragment = cast("EnumFragment", fragments[key])
            return tuple(value for value in fragment.choices if type(value) is int)

        current = (softmax_regs, corr_regs, other_regs)
        best: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
        for softmax_candidate in _int_choices(FLASH_SOFTMAX_REGS_KEY):
            for corr_candidate in _int_choices(FLASH_CORR_REGS_KEY):
                for other_candidate in _int_choices(FLASH_OTHER_REGS_KEY):
                    candidate = (softmax_candidate, corr_candidate, other_candidate)
                    if 2 * softmax_candidate + corr_candidate + other_candidate > 512:
                        continue
                    score = (
                        sum(a != b for a, b in zip(candidate, current, strict=True)),
                        sum(
                            abs(a - b) for a, b in zip(candidate, current, strict=True)
                        ),
                        -softmax_candidate,
                    )
                    if best is None or score < best[0]:
                        best = (score, candidate)
        if best is None:
            raise InvalidConfig(message)
        _score, (softmax_regs, corr_regs, other_regs) = best
        config[FLASH_SOFTMAX_REGS_KEY] = softmax_regs
        config[FLASH_CORR_REGS_KEY] = corr_regs
        config[FLASH_OTHER_REGS_KEY] = other_regs

    def enable_cute_flash_search(
        self,
        *,
        head_dim: int,
        num_kv: int,
        num_bh: int | None = None,
        tensor_4d_heads: int | None = None,
        dtype: torch.dtype = torch.float16,
        block_size_targets: Mapping[int, int],
        is_causal: bool = False,
        has_kv_tile_pruning: bool = False,
        requires_ws_overlap: bool = False,
        small_biased_candidate: bool = False,
        standard_dense_output: bool = False,
        standard_causal_output: bool = False,
        output_requires_tma: bool = False,
        supports_tensor_4d_tma: bool = True,
        has_row_epilogue: bool = False,
        plain_row_body: bool = True,
        has_score_modifiers: bool = False,
        device_sm_count: int = 0,
    ) -> None:
        self.cute_attention_generic_fallback_enabled = False
        self._cute_attention_generic_fallback_block_size_targets = {}
        self.cute_flash_search_enabled = True
        self._cute_flash_device_sm_count = device_sm_count
        self._cute_flash_fragments_cache.clear()
        self._cute_flash_fragments_env_fingerprint = None
        self._cute_flash_head_dim = head_dim
        self._cute_flash_num_kv = num_kv
        self._cute_flash_num_bh = num_bh
        self._cute_flash_tensor_4d_heads = tensor_4d_heads
        self._cute_flash_dtype = dtype
        self._cute_flash_is_causal = is_causal
        self._cute_flash_has_kv_tile_pruning = has_kv_tile_pruning
        self._cute_flash_requires_ws_overlap = requires_ws_overlap
        self._cute_flash_small_biased_candidate = small_biased_candidate
        self._cute_flash_standard_dense_output = standard_dense_output
        self._cute_flash_standard_causal_output = standard_causal_output
        self._cute_flash_output_requires_tma = output_requires_tma
        self._cute_flash_supports_tensor_4d_tma = supports_tensor_4d_tma
        self._cute_flash_has_row_epilogue = has_row_epilogue
        self._cute_flash_plain_row_body = plain_row_body
        self._cute_flash_has_score_modifiers = has_score_modifiers
        self._cute_flash_block_size_targets = dict(block_size_targets)
        for block_id, target in block_size_targets.items():
            spec = self.block_sizes.block_id_lookup(block_id)
            spec.autotuner_min = target
            spec.max_size = target

    def enable_cute_flash_gated_search(
        self,
        *,
        block_size_targets: Mapping[int, int],
        kv_block_id: int,
        kv_tile_choices: Sequence[int],
        kv_stage_choices: Sequence[int],
        kv_stage_default: int,
        gate_warpgroup_choices: Sequence[int] = (1,),
        gate_warpgroup_default: int = 1,
        kv_stage_choices_by_tile: Mapping[tuple[int, int], Sequence[int]] | None = None,
        q_block_id: int | None = None,
        q_tile_choices: Sequence[int] = (128,),
    ) -> None:
        """Enable the gated (softmax-free) fused attention surface.

        Restricts the query block size to the fused body's row tiles (128, or
        also 64) and the KV block size to its tile widths (the detector fires
        only there) and exposes the K/V TMA ring depth and the number of gate
        warpgroups.
        """
        self.cute_flash_gated_search_enabled = True
        self._cute_flash_gated_block_size_targets = dict(block_size_targets)
        self._cute_flash_gated_kv_block_id = kv_block_id
        self._cute_flash_gated_q_block_id = q_block_id
        self._cute_flash_gated_q_tile_choices = tuple(sorted(q_tile_choices))
        self._cute_flash_gated_kv_tile_choices = tuple(sorted(kv_tile_choices))
        self._cute_flash_gated_kv_stage_choices = tuple(kv_stage_choices)
        self._cute_flash_gated_kv_stage_default = kv_stage_default
        self._cute_flash_gated_gate_warpgroup_choices = tuple(gate_warpgroup_choices)
        self._cute_flash_gated_gate_warpgroup_default = gate_warpgroup_default
        self._cute_flash_gated_kv_stage_choices_by_tile = {
            tile: tuple(stages)
            for tile, stages in (kv_stage_choices_by_tile or {}).items()
        }
        for block_id, target in block_size_targets.items():
            spec = self.block_sizes.block_id_lookup(block_id)
            if block_id == kv_block_id:
                spec.autotuner_min = min(self._cute_flash_gated_kv_tile_choices)
                spec.max_size = max(self._cute_flash_gated_kv_tile_choices)
                if min(self._cute_flash_gated_q_tile_choices) < spec.max_size:
                    # The frontend caps a KV loop bounded by the query tile
                    # (``hl.tile(0, tile_q.end)``) at the query block size.  The
                    # fused body walks each work item's KV range independently
                    # of the tile height, so a 128-column KV tile over a 64-row
                    # query tile is legal (and the efficient shape).
                    spec.bounded_by_block_id = None
            elif block_id == q_block_id:
                spec.autotuner_min = min(self._cute_flash_gated_q_tile_choices)
                spec.max_size = max(self._cute_flash_gated_q_tile_choices)
                # The 128-row tile stays the default; 64 rows is a search choice.
                spec.default_size = target
            else:
                spec.autotuner_min = target
                spec.max_size = target

    def _cute_flash_gated_tiles(self) -> list[tuple[int, int]]:
        """Every fused (query rows, KV tile) shape with a ring depth that fits
        shared memory: tallest query tile and widest KV tile first."""
        by_tile = self._cute_flash_gated_kv_stage_choices_by_tile
        return [
            (q_tile, kv_tile)
            for q_tile in sorted(self._cute_flash_gated_q_tile_choices, reverse=True)
            for kv_tile in sorted(self._cute_flash_gated_kv_tile_choices, reverse=True)
            if not by_tile or by_tile.get((q_tile, kv_tile))
        ]

    def _cute_flash_gated_block_size_lists(self) -> list[list[int]]:
        """The block sizes of every fused tile shape (``_cute_flash_gated_tiles`` order)."""
        base = self._cute_flash_gated_block_size_target_list()
        kv_block_id = self._cute_flash_gated_kv_block_id
        assert kv_block_id is not None
        kv_index = self.block_sizes.block_id_to_index(kv_block_id)
        q_block_id = self._cute_flash_gated_q_block_id
        q_index = (
            self.block_sizes.block_id_to_index(q_block_id)
            if q_block_id is not None
            else None
        )
        result: list[list[int]] = []
        for q_tile, kv_tile in self._cute_flash_gated_tiles():
            block_sizes = list(base)
            block_sizes[kv_index] = kv_tile
            if q_index is not None:
                block_sizes[q_index] = q_tile
            result.append(block_sizes)
        return result

    def _cute_flash_gated_block_size_target_list(self) -> list[int]:
        targets: list[int | None] = [None] * len(self.block_sizes)
        for block_id, target in self._cute_flash_gated_block_size_targets.items():
            targets[self.block_sizes.block_id_to_index(block_id)] = target
        if any(target is None for target in targets):
            raise InvalidConfig(
                "CuTe gated attention search has incomplete block sizes"
            )
        return [target for target in targets if target is not None]

    def cute_flash_gated_seed_configs(self) -> list[helion.Config]:
        if not self.cute_flash_gated_search_enabled:
            return []
        from .._compiler.cute.cute_flash_gated import gated_seed_configs

        return gated_seed_configs(
            self._cute_flash_gated_block_size_lists(),
            self._cute_flash_gated_kv_stage_choices,
            self._cute_flash_gated_kv_stage_default,
            tiles=self._cute_flash_gated_tiles(),
            kv_stage_choices_by_tile=self._cute_flash_gated_kv_stage_choices_by_tile,
        )

    def _cute_flash_gated_kv_tile(self, config: dict[str, object]) -> int:
        """The KV tile width of a config on the gated surface."""
        kv_block_id = self._cute_flash_gated_kv_block_id
        assert kv_block_id is not None
        value = config.get("block_sizes")
        assert isinstance(value, (list, tuple))
        kv_tile = value[self.block_sizes.block_id_to_index(kv_block_id)]
        assert isinstance(kv_tile, int)
        return kv_tile

    def _cute_flash_gated_q_tile(self, config: dict[str, object]) -> int:
        """The query tile height of a config on the gated surface."""
        q_block_id = self._cute_flash_gated_q_block_id
        if q_block_id is None:
            return max(self._cute_flash_gated_q_tile_choices)
        value = config.get("block_sizes")
        assert isinstance(value, (list, tuple))
        q_tile = value[self.block_sizes.block_id_to_index(q_block_id)]
        assert isinstance(q_tile, int)
        return q_tile

    def _normalize_cute_flash_gated(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """Normalize the gated attention surface (see ``_normalize_cute_flash``)."""
        if not self.cute_flash_gated_search_enabled:
            return
        block_size_lists = self._cute_flash_gated_block_size_lists()
        block_size_targets = block_size_lists[0]
        if fix_invalid:
            if config.get("block_sizes") not in block_size_lists:
                config["block_sizes"] = list(block_size_targets)
            config["pid_type"] = "flat"
            self._normalize_cute_flash_default_sequence(config, "l2_groupings", 1)
            self._normalize_cute_flash_default_sequence(config, "num_threads", 0)
            self._normalize_cute_flash_default_sequence(config, "cute_vector_widths", 1)
            self._normalize_cute_flash_default_sequence(
                config, "cute_lane_layouts", "blocked"
            )
            self._normalize_cute_flash_default_loop_orders(config)
            config.pop("epilogue_subtile", None)
        elif not any(
            self._is_cute_flash_config_envelope(config, block_sizes)
            for block_sizes in block_size_lists
        ):
            return
        from .._compiler.cute.cute_flash_gated import gated_kv_stage_default

        # The legal ring depths depend on the tile shape (shared memory), so
        # validate against the config's own tile.
        kv_tile = self._cute_flash_gated_kv_tile(config)
        choices = self._cute_flash_gated_kv_stage_choices_by_tile.get(
            (self._cute_flash_gated_q_tile(config), kv_tile),
            self._cute_flash_gated_kv_stage_choices,
        )
        stage_default = (
            self._cute_flash_gated_kv_stage_default
            if self._cute_flash_gated_kv_stage_default in choices
            else gated_kv_stage_default(choices)
        )
        value = config.get(FLASH_KV_STAGE_KEY)
        if value is None:
            config[FLASH_KV_STAGE_KEY] = stage_default
        elif value not in choices:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{FLASH_KV_STAGE_KEY} must be one of {list(choices)!r} "
                    f"for block_sizes {config.get('block_sizes')!r}, got {value!r}"
                )
            config[FLASH_KV_STAGE_KEY] = stage_default
        from .._compiler.cute.cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY
        from .._compiler.cute.cute_flash_gated import gated_gate_warpgroup_choices
        from .._compiler.cute.cute_flash_gated import gated_gate_warpgroup_default

        # The legal warpgroup counts depend on the tile (its chunks must split
        # evenly among the warpgroups), so validate against the config's own.
        wg_choices = tuple(
            g
            for g in gated_gate_warpgroup_choices(
                self._cute_flash_gated_q_tile(config),
                self._cute_flash_gated_kv_tile(config),
            )
            if g in self._cute_flash_gated_gate_warpgroup_choices
        ) or (1,)
        wg_default = gated_gate_warpgroup_default(wg_choices)
        wgs = config.get(FLASH_GATE_WARPGROUPS_KEY)
        if wgs is None:
            config[FLASH_GATE_WARPGROUPS_KEY] = wg_default
        elif wgs not in wg_choices:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{FLASH_GATE_WARPGROUPS_KEY} must be one of {list(wg_choices)!r} "
                    f"for block_sizes {config.get('block_sizes')!r}, got {wgs!r}"
                )
            config[FLASH_GATE_WARPGROUPS_KEY] = wg_default

    def enable_cute_chunk_recurrence_search(self, *, preferred_partitions: int) -> None:
        """Expose the exact BT16 recurrence schedule as CuTe search knobs."""

        if preferred_partitions not in (2, 4):
            raise ValueError(
                "unsupported chunk-recurrence DV partition count: "
                f"{preferred_partitions}"
            )
        alternate = 4 if preferred_partitions == 2 else 2
        self.cute_chunk_recurrence_dv_partitions = EnumFragment(
            choices=(preferred_partitions, alternate)
        )
        self.cute_chunk_recurrence_register_cap = EnumFragment(
            choices=VALID_CUTE_CHUNK_RECURRENCE_REGISTER_CAPS
        )

    def enable_cute_gdn_recurrence_search(
        self,
        *,
        value_block_id: int,
        stage_choices_by_tile: Mapping[tuple[int, int], Sequence[int]],
        epilogue_warp_choices: Sequence[int],
        mma_m_choices_by_tile: Mapping[tuple[int, int], Sequence[int]],
        token_group_choices_by_tile: Mapping[tuple[int, int, int], Sequence[int]],
    ) -> None:
        """Expose the gdn recurrence TMA ring depth, epilogue warp count,
        tcgen05 M and token group count.

        ``mma_m_choices_by_tile`` maps every (dstate tile, epilogue warps)
        pair the planner lowers (the candidate tiles first, then the wider
        admitted tiles the ``block_sizes`` search reaches for a
        non-power-of-two dstate) to its legal MMA heights, preferred first;
        ``stage_choices_by_tile`` maps every (tile, M) pair to its legal ring
        depths and ``token_group_choices_by_tile`` every (tile, warps, M)
        triple to its pipelined token groups.  Each search domain is the
        union of its map's values so each per-tile seed is legal as written,
        and :meth:`normalize` re-validates the values against the selected
        tile: a defaulted value follows the tile, an explicit value the tile
        cannot take is rejected.
        """

        mma_m_choices: list[int] = []
        for choices in mma_m_choices_by_tile.values():
            for mma_m in choices:
                if mma_m not in mma_m_choices:
                    mma_m_choices.append(mma_m)
        stage_choices: list[int] = []
        for choices in stage_choices_by_tile.values():
            for stages in choices:
                if stages not in stage_choices:
                    stage_choices.append(stages)
        group_choices: list[int] = []
        for choices in token_group_choices_by_tile.values():
            for groups in choices:
                if groups not in group_choices:
                    group_choices.append(groups)
        if (
            not stage_choices
            or not epilogue_warp_choices
            or not mma_m_choices
            or not group_choices
        ):
            raise ValueError("gdn recurrence search requires at least one choice")
        self.cute_gdn_recurrence_value_block_id = value_block_id
        self.cute_gdn_recurrence_mma_m_choices_by_tile = {
            tile: tuple(choices) for tile, choices in mma_m_choices_by_tile.items()
        }
        # The full 128-row tile leads the domain: it is legal for every
        # tile, so a config that leaves the knob out keeps the full MMA.
        self.cute_gdn_recurrence_mma_m = EnumFragment(
            choices=tuple(sorted(mma_m_choices, reverse=True))
        )
        self.cute_gdn_recurrence_stage_choices_by_tile = {
            tile: tuple(choices) for tile, choices in stage_choices_by_tile.items()
        }
        self.cute_gdn_recurrence_stages = EnumFragment(choices=tuple(stage_choices))
        self.cute_gdn_recurrence_epilogue_warps = EnumFragment(
            choices=tuple(epilogue_warp_choices)
        )
        self.cute_gdn_recurrence_token_group_choices_by_tile = {
            tile: tuple(choices)
            for tile, choices in token_group_choices_by_tile.items()
        }
        self.cute_gdn_recurrence_token_groups = EnumFragment(
            choices=tuple(group_choices)
        )

    def _cute_gdn_recurrence_tile_of(self, config: dict[str, object]) -> int | None:
        """The dstate tile ``config`` selects, or ``None`` when the gdn knobs
        are off or ``config`` carries no dstate tile."""

        block_id = self.cute_gdn_recurrence_value_block_id
        if self.cute_gdn_recurrence_stages is None or block_id is None:
            return None
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return None
        return self.block_sizes.config_get(cast("list[int]", block_sizes), block_id)

    def _cute_gdn_recurrence_mma_m_choices_for(
        self, config: dict[str, object]
    ) -> tuple[int, ...] | None:
        """Legal gdn tcgen05 M values for the dstate tile and epilogue warp
        count ``config`` selects; ``None`` when the knob is off or the pair
        is one the planner declines (the kernel then takes the SIMT lowering
        and the knob is inert)."""

        block_size = self._cute_gdn_recurrence_tile_of(config)
        warps = config.get(CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY)
        if block_size is None or type(warps) is not int:
            return None
        return self.cute_gdn_recurrence_mma_m_choices_by_tile.get((block_size, warps))

    def _cute_gdn_recurrence_stage_choices_for(
        self, config: dict[str, object]
    ) -> tuple[int, ...] | None:
        """Legal gdn ring depths for the dstate tile and tcgen05 M ``config``
        selects; ``None`` when the knob is off or the pair is one the planner
        declines.  Every pair the planner lowers has an entry, so a depth
        accepted here never fails at codegen."""

        block_size = self._cute_gdn_recurrence_tile_of(config)
        mma_m = config.get(CUTE_GDN_RECURRENCE_MMA_M_KEY)
        if block_size is None or type(mma_m) is not int:
            return None
        return self.cute_gdn_recurrence_stage_choices_by_tile.get((block_size, mma_m))

    def _cute_gdn_recurrence_token_group_choices_for(
        self, config: dict[str, object]
    ) -> tuple[int, ...] | None:
        """Legal gdn token groups for the dstate tile, epilogue warp count and
        tcgen05 M ``config`` selects; ``None`` when the knob is off or the
        triple is one the planner declines."""

        block_size = self._cute_gdn_recurrence_tile_of(config)
        warps = config.get(CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY)
        mma_m = config.get(CUTE_GDN_RECURRENCE_MMA_M_KEY)
        if block_size is None or type(warps) is not int or type(mma_m) is not int:
            return None
        return self.cute_gdn_recurrence_token_group_choices_by_tile.get(
            (block_size, warps, mma_m)
        )

    def enable_cute_flash_bwd_search(
        self,
        *,
        block_size_targets: Mapping[int, int],
        two_cta_allowed: bool = False,
    ) -> None:
        """Enable the CuTe flash-attention BACKWARD surface.

        Pins the (kv_tile, q_tile) block sizes to the 128x128 envelope the
        fused backward emitter supports. ``two_cta_allowed`` opens the
        ``cute_flash_bwd_two_cta`` knob (cluster-of-2 tcgen05 family) when the
        problem shape supports 256-row cluster KV tiles.
        """
        self.cute_flash_bwd_search_enabled = True
        self._cute_flash_bwd_block_size_targets = dict(block_size_targets)
        self._cute_flash_bwd_two_cta_allowed = two_cta_allowed
        for block_id, target in block_size_targets.items():
            spec = self.block_sizes.block_id_lookup(block_id)
            spec.autotuner_min = target
            spec.max_size = target

    def enable_cute_attention_generic_fallback(
        self, *, block_size_targets: Mapping[int, int] | None = None
    ) -> None:
        """Use generic CuTe for an attention shape with no legal flash output."""
        self.cute_attention_generic_fallback_enabled = True
        self._cute_attention_generic_fallback_block_size_targets = dict(
            block_size_targets or {}
        )
        # These are legality bounds for the generic attention lowering, not a
        # selected winner. They make the ordinary default a known-good initial
        # candidate while leaving all larger block sizes available to tuning.
        for (
            block_id,
            target,
        ) in self._cute_attention_generic_fallback_block_size_targets.items():
            spec = self.block_sizes.block_id_lookup(block_id)
            spec.autotuner_min = max(spec.autotuner_min, target)

    def enable_cute_chunk_prepare_schedule_search(
        self, *, preferred_schedule: str
    ) -> None:
        """Expose the exact BT16 prepare shared-memory schedule knob."""

        if not self.supports_config_key(CUTE_CHUNK_PREPARE_SCHEDULE_KEY):
            raise InvalidConfig(
                f"{CUTE_CHUNK_PREPARE_SCHEDULE_KEY} is not supported by backend "
                f"{self.backend_name!r}"
            )
        if preferred_schedule not in VALID_CUTE_CHUNK_PREPARE_SCHEDULES:
            raise ValueError(
                f"unsupported chunk-prepare schedule: {preferred_schedule!r}"
            )
        self.cute_chunk_prepare_schedule = EnumFragment(
            choices=(
                preferred_schedule,
                *(
                    schedule
                    for schedule in VALID_CUTE_CHUNK_PREPARE_SCHEDULES
                    if schedule != preferred_schedule
                ),
            )
        )

    def enable_cute_affine_scan_search(self, *, step_count: int | None) -> None:
        """Expose measured direct-affine schedules for a compatible CuTe scan."""

        if self.backend_name != "cute":
            raise InvalidConfig("direct affine scan configuration requires CuTe")
        self.cute_affine_scan_schedule = EnumFragment(
            choices=direct_affine_schedule_choices(step_count)
        )

    def enable_cute_topk_search(
        self, selection_dtype: torch.dtype | None = None
    ) -> None:
        """Expose the independent row, lane, and vector geometry of top-k."""
        self.cute_topk_search_enabled = True
        if selection_dtype == torch.float32:
            # A full FP32 value plus an index cannot fit any 32-bit key.
            # The 16-bit packed encoders are not legal for this representation.
            self.cute_topk_choices = {
                **CUTE_TOPK_CHOICES,
                "cute_topk_key_dtype": ("int64",),
                "cute_topk_key_encoder": ("dsl",),
                "cute_topk_defer_value_gathers": (False,),
            }
        # The root emitter supplies its own tiling. Ordinary block sizes do
        # not change its code; retain valid defaults for shared compiler
        # bookkeeping without searching duplicate generated kernels.
        for spec in self.block_sizes:
            target = spec._fragment(self).default_val
            spec.autotuner_min = target
            spec.max_size = target

    def _pre_normalize_cute_flash_block_sizes(self, config: dict[str, object]) -> None:
        if "block_sizes" not in config:
            return
        if self.cute_flash_search_enabled:
            block_size_targets = self._cute_flash_block_size_target_list()
        elif self.cute_flash_gated_search_enabled:
            if config["block_sizes"] in self._cute_flash_gated_block_size_lists():
                return
            block_size_targets = self._cute_flash_gated_block_size_target_list()
        else:
            return
        value = config["block_sizes"]
        raw_block_sizes = [*value] if isinstance(value, (list, tuple)) else [value]
        if raw_block_sizes == block_size_targets:
            return
        config["block_sizes"] = list(block_size_targets)

    def _cute_flash_block_size_target_list(self) -> list[int]:
        targets: list[int | None] = [None] * len(self.block_sizes)
        for block_id, target in self._cute_flash_block_size_targets.items():
            targets[self.block_sizes.block_id_to_index(block_id)] = target
        if any(target is None for target in targets):
            raise InvalidConfig(
                "CuTe flash attention search has incomplete block sizes"
            )
        return [target for target in targets if target is not None]

    def _normalize_cute_flash_default_sequence(
        self,
        config: dict[str, object],
        key: str,
        default: object,
    ) -> None:
        value = config.get(key)
        if not value:
            config.pop(key, None)
            return
        if not isinstance(value, list) or any(item != default for item in value):
            config.pop(key, None)
            return
        config.pop(key, None)

    def _normalize_cute_flash_default_loop_orders(
        self, config: dict[str, object]
    ) -> None:
        value = config.get("loop_orders")
        if not value:
            config.pop("loop_orders", None)
            return
        defaults = [spec._fill_missing() for spec in self.loop_orders]
        if value != defaults:
            config.pop("loop_orders", None)
            return
        config.pop("loop_orders", None)

    def _is_cute_flash_config_envelope(
        self, config: dict[str, object], block_size_targets: list[int]
    ) -> bool:
        if config.get("block_sizes") != block_size_targets:
            return False
        if config.get("pid_type", "flat") != "flat":
            return False
        if "epilogue_subtile" in config:
            return False
        for key, default in (
            ("l2_groupings", 1),
            ("num_threads", 0),
            ("cute_vector_widths", 1),
            ("cute_lane_layouts", "blocked"),
        ):
            value = config.get(key)
            if value and (
                not isinstance(value, list) or any(item != default for item in value)
            ):
                return False
        loop_orders = config.get("loop_orders")
        if loop_orders:
            defaults = [spec._fill_missing() for spec in self.loop_orders]
            if loop_orders != defaults:
                return False
        return True

    @property
    def _tcgen05_cluster_m_search_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.cluster_m_search_choices

    @_tcgen05_cluster_m_search_choices.setter
    def _tcgen05_cluster_m_search_choices(self, value: tuple[int, ...] | None) -> None:
        self._cute_tcgen05_config.cluster_m_search_choices = value

    @property
    def _tcgen05_cluster_m2_search_constraints(
        self,
    ) -> Tcgen05ClusterM2SearchConstraints | None:
        return self._cute_tcgen05_config.cluster_m2_search_constraints

    @_tcgen05_cluster_m2_search_constraints.setter
    def _tcgen05_cluster_m2_search_constraints(
        self, value: Tcgen05ClusterM2SearchConstraints | None
    ) -> None:
        self._cute_tcgen05_config.cluster_m2_search_constraints = value

    @property
    def _tcgen05_ab_stages_three_search_constraints(
        self,
    ) -> Tcgen05AbStagesThreeSearchConstraints | None:
        return self._cute_tcgen05_config.ab_stages_three_search_constraints

    @_tcgen05_ab_stages_three_search_constraints.setter
    def _tcgen05_ab_stages_three_search_constraints(
        self, value: Tcgen05AbStagesThreeSearchConstraints | None
    ) -> None:
        self._cute_tcgen05_config.ab_stages_three_search_constraints = value

    @property
    def _tcgen05_num_epi_warps_search_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.num_epi_warps_search_choices

    @_tcgen05_num_epi_warps_search_choices.setter
    def _tcgen05_num_epi_warps_search_choices(
        self, value: tuple[int, ...] | None
    ) -> None:
        self._cute_tcgen05_config.num_epi_warps_search_choices = value

    @property
    def _tcgen05_num_epi_warps_validation_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.num_epi_warps_validation_choices

    @_tcgen05_num_epi_warps_validation_choices.setter
    def _tcgen05_num_epi_warps_validation_choices(
        self, value: tuple[int, ...] | None
    ) -> None:
        self._cute_tcgen05_config.num_epi_warps_validation_choices = value

    def _tcgen05_full_tile_direct_entry_seed_eligible(self) -> bool:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_eligible()

    def _tcgen05_full_tile_direct_entry_seed_bk(self) -> int | None:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_bk()

    def _tcgen05_full_tile_direct_entry_seed_config(self) -> helion.Config | None:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_config()

    def register_cute_tcgen05_mma_analysis(
        self,
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
        self._cute_tcgen05_config.register_mma_analysis(
            m_block_id=m_block_id,
            n_block_id=n_block_id,
            k_block_id=k_block_id,
            compile_time_static_extents=compile_time_static_extents,
            input_dtype=input_dtype,
            has_leading_passthrough=has_leading_passthrough,
            explicit_epi_tile_compatible=explicit_epi_tile_compatible,
            leading_work_multiplier=leading_work_multiplier,
        )

    def _tcgen05_matmul_block_fragments(
        self,
    ) -> tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment] | None:
        return self._cute_tcgen05_config._matmul_block_fragments()

    def _tcgen05_matmul_block_ids(self) -> tuple[int, int, int] | None:
        return self._cute_tcgen05_config.matmul_block_ids

    def _tcgen05_matmul_compile_time_static_extents(
        self,
    ) -> tuple[int | None, int | None, int | None] | None:
        return self._cute_tcgen05_config.matmul_compile_time_static_extents

    def _tcgen05_matmul_seed_block_sizes(
        self, *, bm: int, bn: int, bk: int
    ) -> list[int] | None:
        return self._cute_tcgen05_config._matmul_seed_block_sizes(
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def restrict_tcgen05_cluster_m_search(self, choices: tuple[int, ...]) -> None:
        self._cute_tcgen05_config.restrict_cluster_m_search(choices)

    def allow_tcgen05_cluster_m2_search(
        self,
        *,
        static_k: int,
        max_k_tiles: int = TCGEN05_TWO_CTA_MAX_K_TILES,
        allow_edge_k_tail_family: bool = False,
    ) -> None:
        self._cute_tcgen05_config.allow_cluster_m2_search(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
        )

    @staticmethod
    def _tcgen05_cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        return CuteTcgen05Config.cluster_m2_bk_is_valid(bk, constraints)

    def _tcgen05_c_input_seed_config(self) -> helion.Config | None:
        return self._cute_tcgen05_config._c_input_seed_config()

    def autotune_seed_configs(self) -> list[helion.Config]:
        seeds = self._cute_tcgen05_config.autotune_seed_configs()
        if self.backend_name == "cute" and self.cute_flash_search_enabled:
            from .._compiler.cute.cute_flash import flash_attention_seed_configs

            assert self._cute_flash_head_dim is not None
            flash_seeds = list(
                flash_attention_seed_configs(
                    self._cute_flash_head_dim,
                    self._cute_flash_num_kv,
                    num_bh=self._cute_flash_num_bh,
                    tensor_4d_heads=self._cute_flash_tensor_4d_heads,
                    dtype=self._cute_flash_dtype,
                    is_causal=self._cute_flash_is_causal,
                    has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                    requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                    small_biased_candidate=self._cute_flash_small_biased_candidate,
                    standard_dense_output=self._cute_flash_standard_dense_output,
                    standard_causal_output=self._cute_flash_standard_causal_output,
                    target_device_capability=self.target_device_capability,
                    supports_tensor_4d_tma=(self._cute_flash_supports_tensor_4d_tma),
                    has_row_epilogue=self._cute_flash_has_row_epilogue,
                    plain_row_body=self._cute_flash_plain_row_body,
                    has_score_modifiers=self._cute_flash_has_score_modifiers,
                    block_size_targets=self._cute_flash_block_size_target_list(),
                    device_sm_count=self._cute_flash_device_sm_count,
                )
            )
            flash_seeds = self._legalize_cute_flash_compiler_seeds(flash_seeds)
            seeds.extend(flash_seeds)
        if self.backend_name == "cute" and self.cute_flash_gated_search_enabled:
            seeds.extend(self.cute_flash_gated_seed_configs())
        return seeds

    def _fix_tcgen05_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        self._cute_tcgen05_config._fix_cluster_m2_search_config(config)

    def allow_tcgen05_ab_stages_three_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        self._cute_tcgen05_config.allow_ab_stages_three_search(
            dtype_bytes=dtype_bytes,
            device=device,
        )

    def register_cute_tcgen05_grouped_worklist_smem_facts(
        self, *, group_count: int, device_split_sizes: bool
    ) -> None:
        self._cute_tcgen05_config.register_grouped_worklist_smem_facts(
            group_count=group_count,
            device_split_sizes=device_split_sizes,
        )

    @staticmethod
    def _cute_per_cta_ab_smem_budget_bytes(device: torch.device) -> int:
        return CuteTcgen05Config.per_cta_ab_smem_budget_bytes(device)

    def _tcgen05_ab_stages_three_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
    ) -> bool:
        return self._cute_tcgen05_config.ab_stages_three_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
        )

    def _tcgen05_grouped_dynamic_stages_fit_for_target(
        self,
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
        return self._cute_tcgen05_config.grouped_dynamic_stages_fit_for_target(
            dtype_bytes=dtype_bytes,
            output_dtype_bytes=output_dtype_bytes,
            device=device,
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            ab_stages=ab_stages,
            c_stages=c_stages,
        )

    def _fix_tcgen05_ab_stages_three_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_ab_stages_three_search_config(config)

    def _fix_tcgen05_with_scheduler_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_with_scheduler_search_config(config)

    def _fix_tcgen05_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_cluster_m1_persistent_search_config(config)

    def restrict_tcgen05_num_epi_warps_search(self, choices: tuple[int, ...]) -> None:
        self._cute_tcgen05_config.restrict_num_epi_warps_search(choices)

    def restrict_tcgen05_num_epi_warps_validation(
        self, choices: tuple[int, ...]
    ) -> None:
        self._cute_tcgen05_config.restrict_num_epi_warps_validation(choices)

    def narrow_tcgen05_autotune_to_validated_configs(
        self,
        *,
        allow_persistent_pid_types: bool = False,
        allow_cluster_m2_search: bool = False,
        cluster_m2_static_k: int | None = None,
        allow_cluster_m2_edge_k_tail_family: bool = False,
        allow_cluster_m2_fp8_small_grid: bool = False,
        allow_cluster_m2_one_wave_tiles: bool = False,
        cluster_m2_one_wave_only: bool = False,
        ab_stages_three_dtype_bytes: int | None = None,
        ab_stages_three_device: torch.device | None = None,
        reason: str | None = None,
    ) -> None:
        self._cute_tcgen05_config.narrow_autotune_to_validated_configs(
            allow_persistent_pid_types=allow_persistent_pid_types,
            allow_cluster_m2_search=allow_cluster_m2_search,
            cluster_m2_static_k=cluster_m2_static_k,
            allow_cluster_m2_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
            allow_cluster_m2_fp8_small_grid=allow_cluster_m2_fp8_small_grid,
            allow_cluster_m2_one_wave_tiles=allow_cluster_m2_one_wave_tiles,
            cluster_m2_one_wave_only=cluster_m2_one_wave_only,
            ab_stages_three_dtype_bytes=ab_stages_three_dtype_bytes,
            ab_stages_three_device=ab_stages_three_device,
        )
        _record_restriction(
            self.restriction_reasons,
            "tcgen05 search narrowed to validated configs",
            reason,
            self.log_restrictions_verbose,
        )

    def supports_config_key(self, key: str) -> bool:
        if key == "host_tensor_descriptors":
            return (
                self.device is not None
                and self.device.type == "cuda"
                and self.target_device_capability is not None
                and self.target_device_capability >= (9, 0)
                and self.backend.supports_config_key(key)
            )
        if (
            key == "cross_loop_pipeline"
            and self.device is not None
            and self.device.type != "cuda"
        ):
            return False
        return self.backend.supports_config_key(key)

    def enable_cross_loop_pipeline(
        self, *, choices: tuple[str, ...] = VALID_CROSS_LOOP_PIPELINES
    ) -> None:
        """Expose the compiler-owned cross-loop execution dimension."""
        if not self.supports_config_key("cross_loop_pipeline"):
            raise InvalidConfig(
                f"cross_loop_pipeline is not supported by backend {self.backend_name!r}"
            )
        self.cross_loop_pipeline = EnumFragment(choices)

    def enable_cute_async_load_pipeline(self) -> None:
        """Expose the narrow CuTe async state-load search dimensions."""
        if self.backend_name != "cute":
            raise InvalidConfig("async state-load pipelining requires CuTe")
        self.cute_async_load_pipeline_enabled = True

    @property
    def cute_l2_evict_last_store_policy_supported(self) -> bool:
        """Whether the fixed async-store policy is valid for this target."""

        return fixed_l2_evict_last_store_policy_supported(
            self.target_device_capability,
            torch.version.cuda,
        )

    def enable_cute_bf16x2_recurrence(self) -> None:
        """Expose native packed-BF16 recurrence lowering to the tuner."""
        if self.backend_name != "cute":
            raise InvalidConfig("packed BF16 recurrence lowering requires CuTe")
        self.cute_bf16x2_recurrence_enabled = True

    def enable_cute_proven_bounds(self) -> None:
        """Expose proof-driven CuTe bounds cleanup to the tuner."""
        if self.backend_name != "cute":
            raise InvalidConfig("proven bounds cleanup requires CuTe")
        self.cute_proven_bounds_enabled = True

    def enable_cute_packet_prefetch(self) -> None:
        """Expose independent packet staging to the pointwise tuner."""
        if self.backend_name != "cute":
            raise InvalidConfig("packet prefetch requires CuTe")
        self.cute_packet_prefetch_enabled = True

    def _cute_config_forms_register_tile(
        self,
        config: dict[str, object],
        rl_spec: ReductionLoopSpec,
        available: int,
    ) -> bool:
        """Whether ``config`` spells out a per-thread register tile for the
        persistent reduction ``rl_spec`` (see
        ``DeviceGridState.nest_reduction_lane_outside_vector_tiles``).

        The reduction block must admit the register tile (a static extent and
        a body the two-pass schedule can lower,
        ``cute_register_tile_reduction_blocks``); the reduction thread count
        must be explicit, divide the extent and fit the remaining budget; every
        lane-looped tile block must hold exactly one vector per thread
        (elements per thread == its ``cute_vector_widths`` entry > 1), at least
        one such block must exist, and the unrolled per-thread element count
        stays within ``CUTE_REGISTER_TILE_MAX_ELEMENTS``.  Anything else keeps
        the looped reduction a shrunk thread count always meant.
        """
        if rl_spec.block_id not in self.cute_register_tile_reduction_blocks:
            return False
        nt_list = cast("list[int]", config.get("num_threads", []) or [])
        bs_list = cast("list[int]", config.get("block_sizes", []) or [])
        vec_list = cast("list[int]", config.get("cute_vector_widths", []) or [])
        requested = self.num_threads.config_get(nt_list, rl_spec.block_id, 0)
        if (
            not isinstance(requested, int)
            or requested <= 0
            or requested > available
            or rl_spec.size_hint % requested
        ):
            return False
        unrolled = rl_spec.size_hint // requested
        reduction_block_ids = self.reduction_block_ids | {
            spec.block_id for spec in self.reduction_loops
        }
        vector_block_ids = self.cute_vector_widths.valid_block_ids()
        vector_blocks = 0
        for nt_spec in self.num_threads:
            block_id = nt_spec.block_id
            if block_id in reduction_block_ids:
                continue
            nt = self.num_threads.config_get(nt_list, block_id, 0)
            bs = self.block_sizes.config_get(bs_list, block_id, 1)
            if (
                not isinstance(nt, int)
                or not isinstance(bs, int)
                or nt <= 0
                or nt >= bs
            ):
                # No lane loop on this block.
                continue
            vec = (
                self.cute_vector_widths.config_get(vec_list, block_id, 1)
                if block_id in vector_block_ids
                else 1
            )
            if bs % nt or not isinstance(vec, int) or vec <= 1 or bs // nt != vec:
                return False
            unrolled *= vec
            vector_blocks += 1
        return vector_blocks > 0 and unrolled <= CUTE_REGISTER_TILE_MAX_ELEMENTS

    def _normalize_cute_pointwise_pid_type(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_pointwise_pid_type"
        value = config.get(key, "inherit")
        if type(value) is str and value == "inherit":
            config.pop(key, None)
            return
        if (
            type(value) is str
            and value == "flat"
            and self.cute_pointwise_region_block_ids
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(f"{key}={value!r} requires a proved pointwise region")

    def _normalize_cute_fragment_register_loads(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_fragment_register_loads"
        value = config.get(key, False)
        if value is False:
            config.pop(key, None)
            return
        if (
            value is True
            and self.cute_fragment_register_load_root_ids
            and not config.get("cute_collective_mma")
            and not config.get("cute_register_chain")
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a supported lane-private fragment load"
        )

    def _normalize_cute_fragment_warp_results(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_fragment_warp_results"
        value = config.get(key, False)
        if value is False:
            config.pop(key, None)
            return
        if (
            value is True
            and self.cute_fragment_warp_result_root_ids
            and cast("int", config.get("cute_fragment_threads", 128))
            >= self.cute_fragment_warp_result_min_threads
            and not config.get("cute_collective_mma")
            and not config.get("cute_register_chain")
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a supported warp-owned reduction/atomic value chain"
        )

    def _normalize_cute_fragment_threads(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        from .._compiler.autotuner_heuristics.cute_fragment_threads import THREADS

        key = "cute_fragment_threads"
        value = config.get(key, 128)
        if type(value) is int and value == 128:
            config.pop(key, None)
            return
        if (
            type(value) is int
            and value in THREADS
            and self.cute_fragment_thread_root_ids
            and not config.get("cute_collective_mma")
            and not config.get("cute_register_chain")
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a supported computed fragment root"
        )

    def _normalize_cute_fragment_reduction(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_fragment_reduction"
        value = config.get(key, "serial")
        if type(value) is str and value == "serial":
            config.pop(key, None)
            return
        if (
            type(value) is str
            and value == "warp"
            and self.cute_fragment_reduction_root_ids
            and not config.get("cute_collective_mma")
            and not config.get("cute_register_chain")
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a supported computed scalar reduction"
        )

    def _normalize_cute_fragment_scan(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_fragment_scan"
        value = config.get(key, "serial")
        if type(value) is str and value == "serial":
            config.pop(key, None)
            return
        if (
            type(value) is str
            and value == "cooperative"
            and self.cute_fragment_scan_root_ids
            and not config.get("cute_collective_mma")
            and not config.get("cute_register_chain")
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a supported computed additive scan"
        )

    def _normalize_cute_materialized_operand_schedule(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_materialized_operand_schedule"
        value = config.get(key, "off")
        if type(value) is str and value == "off":
            config.pop(key, None)
            return
        if (
            type(value) is str
            and value == "warp_narrow4"
            and self.cute_materialized_operand_schedule_available
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(f"{key}={value!r} requires a proved packed byte operand")

    @property
    def cute_materialized_schedule_choices(self) -> tuple[str, ...]:
        choices = ("off", "warp_rows2")
        if self.cute_row_matrix_transport_available:
            return (*choices, "warp_rows2_matrix")
        return choices

    def _normalize_cute_materialized_schedule(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_materialized_schedule"
        value = config.get(key, "off")
        if type(value) is str and value == "off":
            config.pop(key, None)
            return
        if (
            type(value) is str
            and value in self.cute_materialized_schedule_choices[1:]
            and self.cute_materialized_schedule_available
        ):
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a proved compact row-resident pair"
        )

    def _normalize_cute_host_paired_sum(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_host_paired_sum"
        value = config.get(key, "off")
        if not self.cute_host_paired_sum_available:
            if value != "off" and not fix_invalid:
                raise InvalidConfig(
                    "host sums require a typed independent pair or terminal sum/cast"
                )
            config.pop(key, None)
            return
        if value not in ("off", "mapped", "narrow"):
            if not fix_invalid:
                raise InvalidConfig(f"unsupported paired host sum layout: {value!r}")
            value = "off"
        config[key] = value

    def _normalize_cute_reduction_sequence(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_reduction_sequence"
        if not self.cute_sequence_reduction_blocks:
            if key in config and not fix_invalid:
                raise InvalidConfig(
                    "resident sequences require a device-tile reduction chain"
                )
            config.pop(key, None)
            return
        value = config.setdefault(key, "scalar")
        if value not in ("scalar", "reload", "resident", "bounded", "bounded_layout"):
            if fix_invalid:
                config[key] = "scalar"
            else:
                raise InvalidConfig(f"unsupported CuTe reduction sequence: {value!r}")

    def _normalize_cute_reduction_local_tree(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_reduction_local_tree"
        value = config.get(key, False)
        if value is False:
            config.pop(key, None)
            return
        if (
            value is True
            and self.cute_resident_reduction_blocks
            and config.get("cute_reduction_schedule") in ("resident", "pipelined")
        ):
            return
        if fix_invalid:
            config.pop(key, None)
        else:
            raise InvalidConfig(
                "cute_reduction_local_tree requires a resident FP32 sum schedule"
            )

    def _normalize_cute_reduction_row_output(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        row_key = "cute_reduction_row_schedule"
        pack_key = "cute_reduction_pack_output"
        row = config.get(row_key, "batched")
        resident = bool(self.cute_resident_reduction_blocks) and config.get(
            "cute_reduction_schedule"
        ) in ("resident", "pipelined")
        if row == "batched":
            config.pop(row_key, None)
        elif (
            type(row) is str
            and row in ("serial", "serial_deferred")
            and resident
            and (
                row != "serial_deferred"
                or (
                    config.get("cute_reduction_schedule") == "pipelined"
                    and config.get("cute_reduction_pipeline_depth", 2) == 2
                )
            )
        ):
            pass
        elif fix_invalid:
            config.pop(row_key, None)
        else:
            raise InvalidConfig(
                f"{row_key} requires a resident schedule; deferred retirement requires a two-slot pipeline"
            )
        pack = config.get(pack_key, False)
        if pack is False:
            config.pop(pack_key, None)
        elif (
            pack is True
            and resident
            and self.target_device_capability is not None
            and self.target_device_capability[0] == 10
        ):
            pass
        elif fix_invalid:
            config.pop(pack_key, None)
        else:
            raise InvalidConfig(
                f"{pack_key} requires a Boolean and a resident FP32 output product on SM100"
            )

    def _normalize_cute_reduction_schedule(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_reduction_schedule"
        group_key = "cute_reduction_group_rows"
        depth_key = "cute_reduction_pipeline_depth"
        if not self.cute_resident_reduction_blocks:
            if (
                any(k in config for k in (key, group_key, depth_key))
                and not fix_invalid
            ):
                raise InvalidConfig(
                    "resident CuTe reductions require static serial-row sums"
                )
            config.pop(key, None)
            config.pop(group_key, None)
            config.pop(depth_key, None)
            return
        value = config.setdefault(key, "scalar")
        if value not in ("scalar", "resident", "pipelined"):
            if fix_invalid:
                config[key] = "scalar"
            else:
                raise InvalidConfig(f"unsupported CuTe reduction schedule: {value!r}")
        group = config.setdefault(group_key, 1)
        if type(group) is not int or group not in (1, 2, 3, 4):
            if fix_invalid:
                config[group_key] = 1
            else:
                raise InvalidConfig(f"unsupported CuTe reduction group size: {group!r}")
        if config[key] == "scalar":
            config[group_key] = 1
        # An omitted depth must not add a key to old normalized configs or
        # generated-source headers. The search fragment still defaults to 2.
        depth = config.get(depth_key, 2)
        if type(depth) is not int or depth not in (2, 4):
            if fix_invalid:
                config[depth_key] = 2
            else:
                raise InvalidConfig(
                    f"unsupported CuTe reduction pipeline depth: {depth!r}"
                )
        if config[key] != "pipelined" and config.get(depth_key, 2) != 2:
            if fix_invalid:
                config[depth_key] = 2
            else:
                raise InvalidConfig(
                    "CuTe reduction pipeline depth 4 requires the pipelined schedule"
                )

    def _normalize_cute_async_load_pipeline(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        keys = (
            "cute_async_load_stages",
            "cute_async_load_lookahead",
            "cute_async_load_group_rows",
            "cute_async_load_cache",
            "cute_async_store_policy",
        )
        if not self.cute_async_load_pipeline_enabled:
            supplied = [key for key in keys if key in config]
            if supplied and not fix_invalid:
                raise InvalidConfig(
                    "CuTe async state-load knobs require a compatible in-place "
                    "vector state update"
                )
            for key in supplied:
                config.pop(key, None)
            return

        defaults: dict[str, object] = {
            "cute_async_load_stages": 0,
            "cute_async_load_lookahead": 4,
            "cute_async_load_group_rows": 2,
            "cute_async_load_cache": "cg",
            "cute_async_store_policy": "default",
        }
        choices: dict[str, tuple[object, ...]] = {
            "cute_async_load_stages": (0, 3, 4, 5),
            "cute_async_load_lookahead": (2, 3, 4),
            "cute_async_load_group_rows": (2, 4),
            "cute_async_load_cache": ("cg", "ca"),
            "cute_async_store_policy": ("default", "l2_evict_last"),
        }
        integer_keys = frozenset(
            {
                "cute_async_load_stages",
                "cute_async_load_lookahead",
                "cute_async_load_group_rows",
            }
        )
        lookahead_was_supplied = "cute_async_load_lookahead" in config
        for key in keys:
            value = config.setdefault(key, defaults[key])
            if value not in choices[key] or (
                key in integer_keys and type(value) is not int
            ):
                if fix_invalid:
                    config[key] = defaults[key]
                else:
                    raise InvalidConfig(
                        f"{key} must be one of {choices[key]!r}, got {value!r}"
                    )
        if (
            config["cute_async_store_policy"] == "l2_evict_last"
            and not self.cute_l2_evict_last_store_policy_supported
        ):
            # Old pinned configs remain loadable on other targets/toolchains,
            # where the transform has always failed closed to the default path.
            config["cute_async_store_policy"] = "default"
        stages = cast("int", config["cute_async_load_stages"])
        if stages > 0 and not lookahead_was_supplied:
            config["cute_async_load_lookahead"] = min(
                cast("int", defaults["cute_async_load_lookahead"]), stages - 1
            )
        lookahead = cast("int", config["cute_async_load_lookahead"])
        if stages == 0:
            for key in keys[1:]:
                config[key] = defaults[key]
        elif lookahead >= stages:
            if fix_invalid:
                config["cute_async_load_lookahead"] = stages - 1
            else:
                raise InvalidConfig(
                    "cute_async_load_lookahead must be smaller than "
                    "cute_async_load_stages"
                )

    def _normalize_cute_signed_bitfield_bf16(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_signed_bitfield_bf16"
        value = config.get(key, False)
        if type(value) is bool and (
            not value or self.cute_signed_bitfield_bf16_available
        ):
            if not value:
                config.pop(key, None)
            return
        if fix_invalid:
            config.pop(key, None)
            return
        raise InvalidConfig(
            f"{key}={value!r} requires a Boolean and a signed-byte BF16 candidate on sm_100a"
        )

    def _normalize_cute_bf16x2_recurrence(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_bf16x2_recurrence"
        if not self.cute_bf16x2_recurrence_enabled:
            if key in config and not fix_invalid:
                raise InvalidConfig(
                    "cute_bf16x2_recurrence requires a compatible BF16 recurrence"
                )
            config.pop(key, None)
            return
        value = config.setdefault(key, False)
        if not isinstance(value, bool):
            if fix_invalid:
                config[key] = False
            else:
                raise InvalidConfig(f"{key} must be a boolean, got {value!r}")
        elif config.get("cute_async_load_stages", 0) == 0:
            config[key] = False

    def _normalize_cute_vector_reductions(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        for key in (
            "cute_independent_reduction",
            "cute_replicated_reduction",
            "cute_vector_packet_unroll",
        ):
            if key in config and type(config[key]) is not bool:
                if fix_invalid:
                    config[key] = False
                else:
                    raise InvalidConfig(f"{key} must be a boolean")

    def _normalize_cute_pdl(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """``cute_pdl`` is a boolean (programmatic dependent launch)."""
        pdl = config.get("cute_pdl", False)
        if type(pdl) is not bool:
            if not fix_invalid:
                raise InvalidConfig("cute_pdl must be a boolean")
            pdl = False
        if "cute_pdl" in config or pdl:
            config["cute_pdl"] = pdl

    def _normalize_cute_vloop_sink(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """``cute_vloop_sink`` is a boolean; ``cute_lane_unroll`` applies to
        the lane loops of a sunk vector nest or, without sinking, to the grid
        lane loops the config forms (``cute/unroll_lane_loads.py``), and is 1
        when the config has neither."""
        sink = config.get("cute_vloop_sink", False)
        if type(sink) is not bool:
            if not fix_invalid:
                raise InvalidConfig("cute_vloop_sink must be a boolean")
            sink = False
        if "cute_vloop_sink" in config or sink:
            config["cute_vloop_sink"] = sink
        unroll = config.get("cute_lane_unroll", 1)
        if type(unroll) is not int or unroll not in _CUTE_LANE_UNROLL_CHOICES:
            if not fix_invalid:
                raise InvalidConfig(
                    f"cute_lane_unroll must be one of {_CUTE_LANE_UNROLL_CHOICES}"
                )
            unroll = 1
        if not sink and not self._cute_config_forms_grid_lane_loop(config):
            unroll = 1
        if "cute_lane_unroll" in config or unroll != 1:
            config["cute_lane_unroll"] = unroll

    def _cute_config_forms_grid_lane_loop(self, config: dict[str, object]) -> bool:
        """Whether some grid tile axis gets a per-thread lane loop: an explicit
        thread count below its static block size that divides it (the
        condition ``PerThreadNDTileStrategy`` allocates a lane variable on)."""
        # Runs before the sequences are normalized: a single-block kernel may
        # still spell them as scalars.
        nt_list = _as_sequence(config.get("num_threads"))
        bs_list = _as_sequence(config.get("block_sizes"))
        thread_block_ids = self.num_threads.valid_block_ids()
        for block_id in self.grid_block_ids:
            if block_id not in thread_block_ids:
                continue
            nt = self.num_threads.config_get(nt_list, block_id, 0)
            bs = self.block_sizes.config_get(bs_list, block_id, 1)
            if (
                isinstance(nt, int)
                and isinstance(bs, int)
                and 0 < nt < bs
                and bs % nt == 0
            ):
                return True
        return False

    def _normalize_cute_register_chain(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_register_chain"
        if key not in config:
            return
        value = config[key]
        supported = (
            bool(self.matmul_facts)
            and self.target_device_capability is not None
            and self.target_device_capability[0] >= 8
        )
        if type(value) is not bool or (value and not supported):
            if fix_invalid:
                config[key] = False
            else:
                raise InvalidConfig(
                    "cute_register_chain requires a boolean, matrix contractions, "
                    "and CUDA compute capability >= 8.0"
                )

    def _normalize_cute_proven_bounds(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_proven_bounds"
        if not self.cute_proven_bounds_enabled:
            if key in config and not fix_invalid:
                raise InvalidConfig(
                    "cute_proven_bounds requires exact CuTe launch and tensor facts"
                )
            config.pop(key, None)
            return
        value = config.setdefault(key, False)
        if not isinstance(value, bool):
            if fix_invalid:
                config[key] = False
            else:
                raise InvalidConfig(f"{key} must be a boolean, got {value!r}")

    def _normalize_cute_affine_scan(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        fragment = self.cute_affine_scan_schedule
        if fragment is None:
            if CUTE_AFFINE_SCAN_SCHEDULE_KEY in config and not fix_invalid:
                raise InvalidConfig(
                    "CuTe affine scan schedule requires a compatible affine scan"
                )
            config.pop(CUTE_AFFINE_SCAN_SCHEDULE_KEY, None)
            return

        value = config.setdefault(CUTE_AFFINE_SCAN_SCHEDULE_KEY, fragment.default())
        if type(value) is not str or value not in fragment.choices:
            if fix_invalid:
                config[CUTE_AFFINE_SCAN_SCHEDULE_KEY] = fragment.default()
            else:
                raise InvalidConfig(
                    f"{CUTE_AFFINE_SCAN_SCHEDULE_KEY} must be one of "
                    f"{fragment.choices!r}, got {value!r}"
                )

    def _normalize_cute_rng_packet(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_rng_packet"
        value = config.get(key, False)
        if not self.cute_rng_packet_enabled or type(value) is not bool:
            if key in config and not fix_invalid:
                raise InvalidConfig(
                    "cute_rng_packet requires the philox4 stream and a boolean"
                )
            config.pop(key, None)
            return
        config.setdefault(key, False)

    def _normalize_cute_packet_prefetch(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        key = "cute_packet_prefetch"
        value = config.get(key, 0)
        if type(value) is not int or value not in (0, 2, 4, 8):
            if not fix_invalid:
                raise InvalidConfig(f"{key} must be 0, 2, 4, or 8, got {value!r}")
            config.pop(key, None)
            return
        if value == 0:
            config.pop(key, None)
            return
        if (
            not self.cute_packet_prefetch_enabled
            or not self.cute_proven_bounds_enabled
            or config.get("cute_proven_bounds") is not True
        ):
            if not fix_invalid:
                raise InvalidConfig(
                    "cute_packet_prefetch requires pointwise facts and cute_proven_bounds"
                )
            config.pop(key, None)

    def _normalize_cute_topk(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        for key, choices in self.cute_topk_choices.items():
            if not self.cute_topk_search_enabled:
                if key in config and not fix_invalid:
                    raise InvalidConfig(f"{key} requires a compatible top-k root")
                config.pop(key, None)
                continue
            value = config.setdefault(key, choices[0])
            if type(value) is not type(choices[0]) or value not in choices:
                if fix_invalid:
                    config[key] = choices[0]
                else:
                    raise InvalidConfig(
                        f"{key} must be one of {choices!r}, got {value!r}"
                    )
        if self.cute_topk_search_enabled:
            lanes = cast("int", config["cute_topk_lanes_per_row"])
            rows = cast("int", config["cute_topk_rows_per_block"])
            if lanes * rows > 1024:
                if fix_invalid:
                    config["cute_topk_rows_per_block"] = 1024 // lanes
                else:
                    raise InvalidConfig(
                        "cute_topk_lanes_per_row * cute_topk_rows_per_block "
                        "must not exceed 1024 threads"
                    )

    def supported_config_keys(self) -> frozenset[str]:
        return frozenset(key for key in VALID_KEYS if self.supports_config_key(key))

    def _default_num_stages(self) -> int:
        return DEFAULT_NUM_STAGES

    def _num_stages_fragment(self) -> ConfigSpecFragment:
        if self.backend_name == "tileir":
            return EnumFragment(choices=tuple(range(1, 11)))
        if supports_amd_cdna_tunables():
            return IntegerFragment(1, 4, self._default_num_stages())
        if self.backend_name == "metal":
            return IntegerFragment(1, 1, 1)
        return IntegerFragment(1, 8, self._default_num_stages())

    def _tcgen05_optional_fragments(
        self, *, for_search: bool = False
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.optional_fragments(for_search=for_search)

    def _tcgen05_strategy_autotune_fragments(
        self,
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.strategy_autotune_fragments()

    def _tcgen05_strategy_validation_fragments(
        self,
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.strategy_validation_fragments()

    @staticmethod
    def _tcgen05_strategy_field_default(key: str, *, pid_type: object = None) -> object:
        return CuteTcgen05Config.strategy_field_default(key, pid_type=pid_type)

    def _validate_tcgen05_strategy_invariants_in_normalize(
        self,
        config: dict[str, object],
        *,
        _fix_invalid: bool,
    ) -> None:
        self._cute_tcgen05_config.validate_strategy_invariants(
            config,
            fix_invalid=_fix_invalid,
        )

    @staticmethod
    def _validate_optional_fragment_value(
        name: str, fragment: ConfigSpecFragment, value: object
    ) -> object:
        return CuteTcgen05Config._validate_optional_fragment_value(
            name,
            fragment,
            value,
        )

    def _clamp_tcgen05_l2_swizzle_size_to_shape(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._clamp_l2_swizzle_size_to_shape(config)

    def unsupported_config_keys(self, config: Mapping[str, object]) -> list[str]:
        return sorted(
            key
            for key in config
            if key in VALID_KEYS and not self.supports_config_key(key)
        )

    def is_supported_config(self, config: Mapping[str, object]) -> bool:
        return not self.unsupported_config_keys(config)

    def normalized_config(
        self,
        config: helion.Config | Mapping[str, object],
        *,
        _cute_register_tiles: bool = True,
    ) -> helion.Config:
        """Return a normalized copy without mutating the requested config."""
        values = config.config if isinstance(config, helion.Config) else config
        copied_values = {
            key: _copy_config_structure(value) for key, value in values.items()
        }
        normalized = helion.Config(
            **copied_values  # pyrefly: ignore[bad-argument-type]
        )
        self.normalize(normalized, _cute_register_tiles=_cute_register_tiles)
        return normalized

    def _normalize_amd_mfma(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        if (
            config.get("matrix_instr_nonkdim") != 32
            or self.backend_name != "triton"
            or torch.version.hip is None
            or not (3, 4) <= get_triton_version().release < (3, 8)
        ):
            return
        properties = torch.cuda.get_device_properties(self.device)
        arch = properties.gcnArchName  # pyrefly: ignore [missing-attribute]
        if arch.split(":")[0] != "gfx950":
            return

        block_sizes = cast("list[int]", config["block_sizes"])
        for fact in self.matmul_facts:
            n = (
                self.block_sizes.config_get(block_sizes, fact.n_block_id, fact.static_n)
                if fact.n_block_id is not None
                else fact.static_n
            )
            if n is None or n >= 16:
                continue
            # Triton 3.4-3.7 lack triton-lang/triton#10305 (fixed in 3.8):
            # MFMA32's wide-store epilogue corrupts the compiler heap for N < 16.
            # Automatic instruction selection avoids that layout.
            if fix_invalid:
                config["matrix_instr_nonkdim"] = 0
                return
            raise InvalidConfig(
                "matrix_instr_nonkdim=32 with a matmul N tile smaller than 16 "
                "can crash Triton 3.4-3.7 on gfx950; use matrix_instr_nonkdim=0 "
                "or 16, or an N tile of at least 16"
            )

    def normalize(
        self,
        config: helion.Config | dict[str, object],
        *,
        _fix_invalid: bool = False,
        _cute_register_tiles: bool = True,
    ) -> None:
        """Normalize the config to match the block_sizes and validate the config.

        Args:
            config: The config to normalize (modified in place).
            _fix_invalid: If True, silently fix invalid combinations instead of raising
                errors. Used internally during autotuning config generation.
            _cute_register_tiles: If False, a CuTe persistent reduction whose
                threads cannot cover its extent is looped even when the config
                spells out a per-thread register tile
                (``_cute_config_forms_register_tile``).  ``generate_ast`` uses
                it to regenerate a kernel whose register tile the lowering
                rejected as the config without the register tile.
        """
        if isinstance(config, helion.Config):
            self.normalize(
                config.config,
                _fix_invalid=_fix_invalid,
                _cute_register_tiles=_cute_register_tiles,
            )
            return

        if (
            self.backend_name == "cute"
            and config.get("cute_fragment_warp_results") is True
            and config.get("cute_fragment_register_loads") is True
        ):
            # These modes assign different physical owners to the same logical
            # element. Even repair-mode search must reject, not silently pick
            # one storage policy and benchmark a different configuration.
            raise InvalidConfig(
                "cute_fragment_warp_results and cute_fragment_register_loads "
                "have incompatible physical ownership"
            )

        # ``cross_loop_schedule`` was the former public name. Accept old configs
        # only at this boundary, then keep ``cross_loop_pipeline`` as the sole
        # internal key.
        if "cross_loop_schedule" in config:
            legacy_value = config.pop("cross_loop_schedule")
            legacy_pipeline = (
                {
                    "barrier": "barrier",
                    "static_pipeline": "static",
                }.get(legacy_value)
                if isinstance(legacy_value, str)
                else None
            )
            if legacy_pipeline is None:
                if not _fix_invalid:
                    raise InvalidConfig(
                        "cross_loop_schedule must be one of "
                        "('barrier', 'static_pipeline')"
                    )
            elif "cross_loop_pipeline" not in config:
                config["cross_loop_pipeline"] = legacy_pipeline
            elif config["cross_loop_pipeline"] != legacy_pipeline and not _fix_invalid:
                raise InvalidConfig(
                    "cross_loop_schedule and cross_loop_pipeline select "
                    "conflicting execution policies"
                )

        for name in (
            "block_size",
            "loop_order",
            "reduction_loop",
            "l2_grouping",
            "flatten_loop",
            "range_unroll_factor",
            "range_warp_specialize",
            "range_num_stage",
            "range_multi_buffer",
            "range_flatten",
            "static_range",
        ):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                value = config.pop(name)
                if name == "reduction_loop" and len(self.reduction_loops) > 1:
                    # Apply the same reduction_loop setting to every
                    # reduction dimension so a single scalar value works
                    # when multiple dims can be rolled.
                    config[names] = [value for _ in range(len(self.reduction_loops))]
                else:
                    config[names] = [value]

        if (
            "cross_loop_pipeline" in config
            and self.cross_loop_pipeline is None
            and self.supports_config_key("cross_loop_pipeline")
        ):
            if _fix_invalid:
                config.pop("cross_loop_pipeline", None)
            else:
                raise InvalidConfig(
                    "cross_loop_pipeline is available only for kernels "
                    "with compiler-inferred cross-loop dependencies"
                )

        if (
            CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY in config
            and self.cute_chunk_recurrence_dv_partitions is None
            and self.supports_config_key(CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY)
        ):
            if _fix_invalid:
                config.pop(CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY)
            else:
                raise InvalidConfig(
                    f"{CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY} is available only "
                    "for matched BT16 chunk-recurrence kernels"
                )

        if (
            CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY in config
            and self.cute_chunk_recurrence_register_cap is None
            and self.supports_config_key(CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY)
        ):
            if _fix_invalid:
                config.pop(CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY)
            else:
                raise InvalidConfig(
                    f"{CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY} is available only "
                    "for matched BT16 chunk-recurrence kernels"
                )

        for gdn_key, gdn_fragment in (
            (CUTE_GDN_RECURRENCE_STAGES_KEY, self.cute_gdn_recurrence_stages),
            (
                CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
                self.cute_gdn_recurrence_epilogue_warps,
            ),
            (
                CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
                self.cute_gdn_recurrence_token_groups,
            ),
            (CUTE_GDN_RECURRENCE_MMA_M_KEY, self.cute_gdn_recurrence_mma_m),
        ):
            if (
                gdn_key in config
                and gdn_fragment is None
                and self.supports_config_key(gdn_key)
            ):
                if _fix_invalid:
                    config.pop(gdn_key)
                else:
                    raise InvalidConfig(
                        f"{gdn_key} is available only for matched gated-delta-rule "
                        "chunk-recurrence kernels"
                    )

        if (
            CUTE_CHUNK_PREPARE_SCHEDULE_KEY in config
            and self.cute_chunk_prepare_schedule is None
            and self.supports_config_key(CUTE_CHUNK_PREPARE_SCHEDULE_KEY)
        ):
            if _fix_invalid:
                config.pop(CUTE_CHUNK_PREPARE_SCHEDULE_KEY)
            else:
                raise InvalidConfig(
                    f"{CUTE_CHUNK_PREPARE_SCHEDULE_KEY} is available only for "
                    "matched BT16 chunk-prepare kernels"
                )

        if (
            FLASH_KV_STAGE_KEY in config
            and not self.cute_flash_search_enabled
            and not self.cute_flash_gated_search_enabled
            and self.supports_config_key(FLASH_KV_STAGE_KEY)
        ):
            # The KV ring depth has no consumer on the scalar path. A config
            # pinned from a flash autotune must still run when the surface is
            # off (another sequence length, ``HELION_CUTE_FLASH=0``, a GPU
            # without tcgen05), so the key is dropped rather than rejected;
            # the flash surfaces validate their own values.
            config.pop(FLASH_KV_STAGE_KEY)
        from .._compiler.cute.cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY

        if (
            FLASH_GATE_WARPGROUPS_KEY in config
            and not self.cute_flash_gated_search_enabled
        ):
            # Same rule for the gate warpgroup count: only the gated body reads it.
            config.pop(FLASH_GATE_WARPGROUPS_KEY)

        if unsupported := self.unsupported_config_keys(config):
            # Separate backend-specific keys (e.g. AMD tunables, TileIR tunables)
            # from common keys (e.g. num_warps, num_stages, indexing).
            # Backend-specific keys should raise errors; common keys are
            # silently stripped so configs are portable across backends.
            backend_specific = [k for k in unsupported if k in BACKEND_SPECIFIC_KEYS]
            common = [k for k in unsupported if k not in BACKEND_SPECIFIC_KEYS]
            for key in common:
                config.pop(key, None)
            if backend_specific:
                if _fix_invalid:
                    for key in backend_specific:
                        config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"Unsupported config keys for backend {self.backend_name!r}: {backend_specific}"
                    )
        if self.backend_name == "cute":
            normalize_cluster_schedule(
                config,
                available=self.cute_split_k_cluster_facts is not None,
                fix_invalid=_fix_invalid,
            )
            validate_cluster_config(self, config, fix_invalid=_fix_invalid)
            normalize_cluster_finalizer(
                config,
                available=self.cute_split_k_cluster_facts is not None,
                fix_invalid=_fix_invalid,
            )
            normalize_workspace_config(
                config,
                available=self.cute_split_k_workspace_available,
                fix_invalid=_fix_invalid,
            )
            normalize_grouped_rna_config(
                config,
                available_k=self.cute_grouped_rna_k_choices,
                rn_two_stage_ctas=self.cute_grouped_rn_two_stage_ctas,
                block_prefix_recipes=self.cute_grouped_block_prefix_recipes,
                warp_raw_available=self.cute_grouped_warp_tf32_available,
                wrapped_descriptors_available=self.cute_grouped_wrapped_descriptors_available,
                fix_invalid=_fix_invalid,
            )
            self._cute_tcgen05_config.prepare_normalization(
                config, fix_invalid=_fix_invalid
            )
            self._normalize_cute_reduction_schedule(config, fix_invalid=_fix_invalid)
            self._normalize_cute_reduction_local_tree(config, fix_invalid=_fix_invalid)
            self._normalize_cute_reduction_row_output(config, fix_invalid=_fix_invalid)
            self._normalize_cute_reduction_sequence(config, fix_invalid=_fix_invalid)
            self._normalize_cute_host_paired_sum(config, fix_invalid=_fix_invalid)
            self._normalize_cute_fragment_scan(config, fix_invalid=_fix_invalid)
            self._normalize_cute_fragment_reduction(config, fix_invalid=_fix_invalid)
            self._normalize_cute_fragment_threads(config, fix_invalid=_fix_invalid)
            self._normalize_cute_fragment_register_loads(
                config, fix_invalid=_fix_invalid
            )
            self._normalize_cute_fragment_warp_results(config, fix_invalid=_fix_invalid)
            self._normalize_cute_materialized_schedule(config, fix_invalid=_fix_invalid)
            self._normalize_cute_materialized_operand_schedule(
                config, fix_invalid=_fix_invalid
            )
            self._normalize_cute_pointwise_pid_type(config, fix_invalid=_fix_invalid)
            self._normalize_cute_async_load_pipeline(config, fix_invalid=_fix_invalid)
            self._normalize_cute_bf16x2_recurrence(config, fix_invalid=_fix_invalid)
            self._normalize_cute_signed_bitfield_bf16(config, fix_invalid=_fix_invalid)
            self._normalize_cute_proven_bounds(config, fix_invalid=_fix_invalid)
            self._normalize_cute_affine_scan(config, fix_invalid=_fix_invalid)
            self._normalize_cute_topk(config, fix_invalid=_fix_invalid)
            self._normalize_cute_rng_packet(config, fix_invalid=_fix_invalid)
            self._normalize_cute_vector_reductions(config, fix_invalid=_fix_invalid)
            self._normalize_cute_vloop_sink(config, fix_invalid=_fix_invalid)
            self._normalize_cute_pdl(config, fix_invalid=_fix_invalid)
            self._normalize_cute_packet_prefetch(config, fix_invalid=_fix_invalid)
            self._normalize_cute_register_chain(config, fix_invalid=_fix_invalid)
            if self.matmul_facts:
                value = config.setdefault("cute_collective_mma", False)
                if not isinstance(value, bool):
                    if _fix_invalid:
                        config["cute_collective_mma"] = False
                    else:
                        raise InvalidConfig("cute_collective_mma must be a boolean")
                for key in (
                    "cute_collective_native_seeded",
                    "cute_collective_tmem_seed",
                    "cute_collective_tmem_a",
                ):
                    seeded = config.setdefault(key, False)
                    if not isinstance(seeded, bool):
                        if _fix_invalid:
                            config[key] = False
                        else:
                            raise InvalidConfig(f"{key} must be a boolean")
                    if seeded is True and (
                        self.target_device_capability is None
                        or self.target_device_capability[0] != 10
                    ):
                        if _fix_invalid:
                            config[key] = False
                        else:
                            raise InvalidConfig(f"{key} requires SM100-family hardware")
                packet_key = "cute_collective_operand_packets"
                if not isinstance(config.get(packet_key, False), bool):
                    if _fix_invalid:
                        config[packet_key] = False
                    else:
                        raise InvalidConfig(f"{packet_key} must be a boolean")
                copy = config.setdefault("cute_collective_copy", "scalar")
                if copy not in ("scalar", "async", "async_cached"):
                    if _fix_invalid:
                        config["cute_collective_copy"] = "scalar"
                    else:
                        raise InvalidConfig(
                            "cute_collective_copy must be scalar, async, or async_cached"
                        )
                for key in ("cute_collective_recipe", "cute_collective_epilogue"):
                    recipe = config.get(key, "scalar")
                    if recipe not in ("scalar", "vector", "vector_unrolled"):
                        if _fix_invalid:
                            config[key] = "scalar"
                        else:
                            raise InvalidConfig(
                                f"{key} must be scalar, vector, or vector_unrolled"
                            )
                compute = config.setdefault("cute_collective_compute", "warp")
                if compute not in ("warp", "tcgen05", "tma_gather"):
                    if _fix_invalid:
                        config["cute_collective_compute"] = "warp"
                    else:
                        raise InvalidConfig(
                            "cute_collective_compute must be warp, tcgen05, or tma_gather"
                        )
                if compute in ("tcgen05", "tma_gather") and (
                    self.target_device_capability is None
                    or self.target_device_capability[0] != 10
                ):
                    if _fix_invalid:
                        config["cute_collective_compute"] = "warp"
                    else:
                        raise InvalidConfig(
                            "native collective compute requires SM100-family hardware"
                        )
                for key, choices, default in (
                    ("cute_collective_stages", (1, 2, 3, 4), 1),
                    ("cute_gathered_mma_n", (128, 256, 512), 256),
                    ("cute_gathered_mma_stages", (2, 3, 4), 3),
                ):
                    if key in config and (
                        type(config[key]) is not int or config[key] not in choices
                    ):
                        if _fix_invalid:
                            config[key] = default
                        else:
                            raise InvalidConfig(f"{key} must be one of {choices}")
                if compute == "tma_gather":
                    n = config.setdefault("cute_gathered_mma_n", 256)
                    stages = config.setdefault("cute_gathered_mma_stages", 3)
                    if n == 512 and stages != 2:
                        if _fix_invalid:
                            config["cute_gathered_mma_stages"] = 2
                        else:
                            raise InvalidConfig(
                                "512-column gathered MMA requires two stages"
                            )
            elif any(
                key in config
                for key in (
                    "cute_collective_mma",
                    "cute_collective_copy",
                    "cute_collective_recipe",
                    "cute_collective_operand_packets",
                    "cute_collective_epilogue",
                    "cute_collective_stages",
                    "cute_collective_compute",
                    "cute_collective_native_seeded",
                    "cute_collective_tmem_seed",
                    "cute_collective_tmem_a",
                    "cute_gathered_mma_n",
                    "cute_gathered_mma_stages",
                )
            ):
                if not _fix_invalid:
                    raise InvalidConfig(
                        "cute_collective_mma requires a matrix contraction"
                    )
                config.pop("cute_collective_mma", None)
                config.pop("cute_collective_copy", None)
                config.pop("cute_collective_recipe", None)
                config.pop("cute_collective_operand_packets", None)
                config.pop("cute_collective_epilogue", None)
                config.pop("cute_collective_stages", None)
                config.pop("cute_collective_compute", None)
                config.pop("cute_collective_native_seeded", None)
                config.pop("cute_collective_tmem_seed", None)
                config.pop("cute_collective_tmem_a", None)
                config.pop("cute_gathered_mma_n", None)
                config.pop("cute_gathered_mma_stages", None)
        if self.backend_name == "cute":
            key = "cute_collective_static_layouts"
            value = config.get(key, False)
            if value is False:
                config.pop(key, None)
            elif (
                value is True
                and self.matmul_facts
                and config.get("cute_collective_mma", False)
                and config.get("cute_collective_compute", "warp") == "warp"
            ):
                pass
            elif _fix_invalid:
                config.pop(key, None)
            else:
                raise InvalidConfig(
                    "cute_collective_static_layouts requires warp collective MMA"
                )
        provided_keys = set(config)
        if _fix_invalid:
            self._pre_normalize_cute_flash_block_sizes(config)

        for name, mapping, flatten in [
            ("block_sizes", self.block_sizes, True),
            ("num_threads", self.num_threads, True),
            ("flatten_loops", self.flatten_loops, True),
            ("l2_groupings", self.l2_groupings, True),
            ("loop_orders", self.loop_orders, False),
            ("reduction_loops", self.reduction_loops, True),
            ("cute_vector_widths", self.cute_vector_widths, True),
            ("cute_lane_layouts", self.cute_lane_layouts, True),
            ("cute_reduction_reloads", self.cute_reduction_reloads, True),
            ("range_unroll_factors", self.range_unroll_factors, True),
            ("range_warp_specializes", self.range_warp_specialize, True),
            ("range_num_stages", self.range_num_stages, True),
            ("range_multi_buffers", self.range_multi_buffers, True),
            ("range_flattens", self.range_flattens, True),
            ("static_ranges", self.static_ranges, True),
        ]:
            if not self.supports_config_key(name):
                if name in config:
                    raise InvalidConfig(
                        f"{name} is not supported on backend {self.backend_name!r}"
                    )
                config.pop(name, None)
                continue
            config[name] = mapping._normalize(
                name, config.get(name, ()), flatten=flatten
            )

        if self.backend_name == "cute":
            # A persistent reduction with exactly one vector fragment per
            # thread has only one physical assignment, so blocked/strided are
            # identical.  Canonicalize the inactive choice to avoid duplicate
            # autotuner candidates.  Looped reductions retain both layouts.
            lane_layouts = config.get("cute_lane_layouts")
            reduction_loops = cast(
                "list[int | None]", config.get("reduction_loops", []) or []
            )
            if isinstance(lane_layouts, list):
                canonical_layouts = list(lane_layouts)
                for index, layout_spec in enumerate(self.cute_lane_layouts):
                    block_id = layout_spec.block_id
                    if (
                        block_id in self.reduction_block_ids
                        and self.reduction_loops.config_get(
                            reduction_loops, block_id, None
                        )
                        is None
                    ):
                        canonical_layouts[index] = "blocked"
                config["cute_lane_layouts"] = canonical_layouts

        # Clamp inner block sizes that are bounded by an outer block
        # (e.g. ``hl.tile(outer.begin, outer.end)``): at this point the
        # outer's concrete block size for this config is known, and the
        # inner extent can never exceed it.
        block_sizes_list = config.get("block_sizes")
        if isinstance(block_sizes_list, list):
            changed = False
            new_block_sizes = list(block_sizes_list)
            for i, spec in enumerate(self.block_sizes):
                bb = spec.bounded_by_block_id
                if (
                    bb is None
                    or i >= len(new_block_sizes)
                    or new_block_sizes[i] is None
                ):
                    continue
                try:
                    outer_index = self.block_sizes.block_id_to_index(bb)
                except KeyError:
                    continue
                outer_val = (
                    new_block_sizes[outer_index]
                    if outer_index < len(new_block_sizes)
                    else None
                )
                if (
                    isinstance(outer_val, int)
                    and isinstance(new_block_sizes[i], int)
                    and new_block_sizes[i] > outer_val
                ):
                    new_block_sizes[i] = outer_val
                    changed = True
            if changed:
                config["block_sizes"] = new_block_sizes
                num_threads = config.get("num_threads")
                if isinstance(num_threads, list):
                    new_num_threads = list(num_threads)
                    for i, (block_size, num_thread) in enumerate(
                        zip(new_block_sizes, new_num_threads, strict=False)
                    ):
                        if (
                            type(block_size) is not int
                            or type(num_thread) is not int
                            or num_thread <= 0
                        ):
                            continue
                        if num_thread > block_size:
                            num_thread = 1 << (max(block_size, 1).bit_length() - 1)
                        while num_thread > 1 and block_size % num_thread != 0:
                            num_thread //= 2
                        new_num_threads[i] = max(num_thread, 1)
                    config["num_threads"] = new_num_threads

        if self.supports_config_key("num_threads"):
            num_threads = cast("list[int]", config.get("num_threads", []))
            if all(value == 0 for value in num_threads):
                config.pop("num_threads", None)
        else:
            config.pop("num_threads", None)

        # Cap reduction loops at the backend's max loop chunk, while using the
        # live reduction thread threshold to decide when a persistent reduction
        # must be rolled.
        if self.max_reduction_threads is not None and self.reduction_loops:
            force_threshold = self.reduction_loop_force_threshold
            max_loop = self.max_reduction_loop
            reduction_loops = config.get("reduction_loops", [])
            if force_threshold is not None and isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    # Indexed reductions (argmin/argmax) on CuTe must keep
                    # the persistent thread count or rolled chunk within a
                    # single warp, since cute.arch.warp_reduction only
                    # supports threads_in_group<=32.
                    block_threshold = force_threshold
                    if (
                        self.backend_name == "cute"
                        and spec.block_id in self.cute_indexed_reduction_block_ids
                    ):
                        block_threshold = min(block_threshold, 32)
                    if new_loops[i] is None and spec.size_hint > block_threshold:
                        new_loops[i] = min(spec.size_hint, block_threshold)
                        changed = True
                    elif (
                        new_loops[i] is not None
                        and max_loop is not None
                        and (
                            new_loops[i] > max_loop
                            or (
                                self.backend_name == "cute"
                                and spec.block_id
                                in self.cute_indexed_reduction_block_ids
                                and new_loops[i] > 32
                            )
                        )
                    ):
                        new_loops[i] = min(
                            new_loops[i] if max_loop is None else max_loop,
                            block_threshold,
                        )
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

        # CuTe-specific: a persistent reduction whose thread count is shrunk
        # below the reduction extent by adjust_reduction_thread_count wraps
        # the kernel body in a synthetic lane loop.  Force a looped reduction
        # whenever the available reduction threads (max_reduction_threads //
        # product_of_non_reduction_thread_axes) cannot cover the full
        # reduction extent, unless the config spells out a per-thread
        # register tile over a reduction block that admits one
        # (``_cute_config_forms_register_tile``): that lane loop is the
        # requested geometry and its lane reductions are lowered by
        # ``split_lane_loop_reductions``; a body that lowering cannot place
        # is regenerated with ``_cute_register_tiles=False`` (the looped
        # reduction this branch chooses without the register tile), and an
        # unproved reordering rejects the config instead of miscompiling.
        if (
            self.backend_name == "cute"
            and self.max_reduction_threads is not None
            and self.reduction_loops
        ):
            nt_list = cast("list[int]", config.get("num_threads", []) or [])
            bs_list = cast("list[int]", config.get("block_sizes", []) or [])
            # ``num_threads`` also carries per-rolled-rdim slots (the
            # reduction's own thread count), plus static persistent-rdim slots
            # that have no ``reduction_loops`` entry.  Only NON-reduction tile
            # axes consume the budget the reduction competes for.  Resolve all
            # values by block id because the two sequences need not share an
            # order.
            reduction_block_ids = self.reduction_block_ids | {
                spec.block_id for spec in self.reduction_loops
            }
            other_threads = 1
            for nt_spec in self.num_threads:
                if nt_spec.block_id in reduction_block_ids:
                    continue
                nt = self.num_threads.config_get(nt_list, nt_spec.block_id, 0)
                if not isinstance(nt, int) or nt <= 0:
                    bs = self.block_sizes.config_get(bs_list, nt_spec.block_id, 1)
                    nt = bs if isinstance(bs, int) and bs > 1 else 1
                if nt > 1:
                    other_threads *= nt
            available = max(1, self.max_reduction_threads // other_threads)
            reduction_loops = config.get("reduction_loops", [])
            if isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    if new_loops[i] is None and spec.size_hint > available:
                        # When other_threads consumes the entire CuTe 1024 thread
                        # budget there is no thread budget left for the reduction
                        # axis. A chunk of 1 is invalid (LoopedReductionStrategy
                        # requires block_size > 1) and a persistent reduction
                        # could not place a single reduction thread.  Reject
                        # the config so the autotuner skips it.
                        if available < 2:
                            raise InvalidConfig(
                                f"cute backend: reduction axis {i} has no thread "
                                f"budget left (non-reduction axes use "
                                f"{other_threads} of {self.max_reduction_threads} "
                                f"threads)."
                            )
                        if (
                            _cute_register_tiles
                            and self._cute_config_forms_register_tile(
                                config, spec, available
                            )
                        ):
                            continue
                        chunk = min(spec.size_hint, available)
                        if self.max_reduction_loop is not None:
                            chunk = min(chunk, self.max_reduction_loop)
                        new_loops[i] = chunk
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

        # Disable range_* configs for static ranges
        static_range_block_ids = [
            block_id
            for block_id in self.static_ranges.valid_block_ids()
            if self.static_ranges.config_get(
                cast("list[bool]", config.get("static_ranges", [])),
                block_id,
            )
        ]
        if static_range_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=static_range_block_ids
                )

        for name in (
            "loop_orders",
            "l2_groupings",
            "flatten_loops",
            "reduction_loops",
            "cute_vector_widths",
            "cute_lane_layouts",
            "cute_reduction_reloads",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
            "indexing",
            "atomic_indexing",
        ):
            value = config.get(name)
            if name == "load_eviction_policies" and value == "":
                continue
            if not value:
                config.pop(name, None)

        # Remove unsupported keys before setting defaults
        for name in (
            "num_warps",
            "num_stages",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
            "indexing",
            "atomic_indexing",
            "pallas_load_buffer_count",
            "pid_type",
            "num_sm_multiplier",
            "maxnreg",
            "host_tensor_descriptors",
        ):
            if not self.supports_config_key(name):
                config.pop(name, None)

        if self.supports_config_key("num_warps"):
            config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        if self.supports_config_key("num_stages"):
            config.setdefault("num_stages", self._default_num_stages())
        if self.supports_config_key("host_tensor_descriptors"):
            value = config.setdefault("host_tensor_descriptors", False)
            if type(value) is not bool:
                if _fix_invalid:
                    config["host_tensor_descriptors"] = False
                else:
                    raise InvalidConfig("host_tensor_descriptors must be a bool")
        if self.supports_config_key("load_eviction_policies"):
            config.setdefault(
                "load_eviction_policies", self.load_eviction_policies.default()
            )
        if (
            self.supports_config_key("load_cache_modifiers")
            and self.load_cache_modifiers.length > 0
        ):
            config.setdefault(
                "load_cache_modifiers", self.load_cache_modifiers.default()
            )
        if (
            self.supports_config_key("store_cache_modifiers")
            and self.store_cache_modifiers.length > 0
        ):
            config.setdefault(
                "store_cache_modifiers", self.store_cache_modifiers.default()
            )
        if self.supports_config_key("indexing"):
            config.setdefault("indexing", self.indexing.default())
        if self.supports_config_key("atomic_indexing"):
            config.setdefault("atomic_indexing", self.atomic_indexing.default())
        if config.get("host_tensor_descriptors"):
            indexing_values = (config.get("indexing"), config.get("atomic_indexing"))
            uses_tensor_descriptor = any(
                map(indexing_uses_tensor_descriptor, indexing_values)
            )
            if not uses_tensor_descriptor:
                # Host and device materialization are identical when no memory
                # operation selected descriptor indexing. Keep one tune point.
                config["host_tensor_descriptors"] = False
        for key, fragment in self.backend_tunable_fragments.items():
            config.setdefault(key, fragment.default())
        self._normalize_amd_mfma(config, fix_invalid=_fix_invalid)
        if self.backend_name == "cute":
            normalize_block_scaled_config(
                config,
                available=self.cute_scaled_mma_available,
                capability=self.target_device_capability,
                fix_invalid=_fix_invalid,
            )
        cross_loop_pipeline_fragment = self.cross_loop_pipeline
        if cross_loop_pipeline_fragment is not None:
            cross_loop_pipeline = config.setdefault(
                "cross_loop_pipeline",
                cross_loop_pipeline_fragment.default(),
            )
            if cross_loop_pipeline not in cross_loop_pipeline_fragment.choices:
                if _fix_invalid:
                    config["cross_loop_pipeline"] = (
                        cross_loop_pipeline_fragment.default()
                    )
                else:
                    raise InvalidConfig(
                        "cross_loop_pipeline must be one of "
                        f"{cross_loop_pipeline_fragment.choices!r}, got "
                        f"{cross_loop_pipeline!r}"
                    )
        recurrence_dv_fragment = self.cute_chunk_recurrence_dv_partitions
        if recurrence_dv_fragment is not None:
            recurrence_dv_partitions = config.setdefault(
                CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
                recurrence_dv_fragment.default(),
            )
            if (
                type(recurrence_dv_partitions) is not int
                or recurrence_dv_partitions not in recurrence_dv_fragment.choices
            ):
                if _fix_invalid:
                    config[CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY] = (
                        recurrence_dv_fragment.default()
                    )
                else:
                    raise InvalidConfig(
                        f"{CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY} must be one of "
                        f"{recurrence_dv_fragment.choices!r}, got "
                        f"{recurrence_dv_partitions!r}"
                    )
        recurrence_register_cap_fragment = self.cute_chunk_recurrence_register_cap
        if recurrence_register_cap_fragment is not None:
            recurrence_register_cap = config.setdefault(
                CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,
                recurrence_register_cap_fragment.default(),
            )
            if (
                (
                    recurrence_register_cap is not None
                    and type(recurrence_register_cap) is not int
                )
                or recurrence_register_cap
                not in recurrence_register_cap_fragment.choices
            ):
                if _fix_invalid:
                    config[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] = (
                        recurrence_register_cap_fragment.default()
                    )
                else:
                    raise InvalidConfig(
                        f"{CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY} must be one of "
                        f"{recurrence_register_cap_fragment.choices!r}, got "
                        f"{recurrence_register_cap!r}"
                    )
            if recurrence_dv_fragment is not None and not (
                _cute_chunk_recurrence_config_is_safe(
                    config[CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY],
                    config[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY],
                )
            ):
                if _fix_invalid:
                    config[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] = None
                else:
                    raise InvalidConfig(
                        f"{CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY} must be None for "
                        f"{CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY}=2 because the "
                        "TMEM schedule dynamically reallocates registers"
                    )
        gdn_stages_supplied = CUTE_GDN_RECURRENCE_STAGES_KEY in config
        gdn_groups_supplied = CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY in config
        gdn_mma_m_supplied = CUTE_GDN_RECURRENCE_MMA_M_KEY in config
        for gdn_key, gdn_fragment in (
            (CUTE_GDN_RECURRENCE_STAGES_KEY, self.cute_gdn_recurrence_stages),
            (
                CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
                self.cute_gdn_recurrence_epilogue_warps,
            ),
            (
                CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
                self.cute_gdn_recurrence_token_groups,
            ),
            (CUTE_GDN_RECURRENCE_MMA_M_KEY, self.cute_gdn_recurrence_mma_m),
        ):
            if gdn_fragment is None:
                continue
            gdn_value = config.setdefault(gdn_key, gdn_fragment.default())
            if type(gdn_value) is not int or gdn_value not in gdn_fragment.choices:
                if _fix_invalid:
                    config[gdn_key] = gdn_fragment.default()
                else:
                    raise InvalidConfig(
                        f"{gdn_key} must be one of {gdn_fragment.choices!r}, got "
                        f"{gdn_value!r}"
                    )
        # The tcgen05 M follows the tile and warps, the ring depth the tile
        # and M, the token groups all three: validate in that order.
        gdn_tile_mma_m_choices = self._cute_gdn_recurrence_mma_m_choices_for(config)
        if gdn_tile_mma_m_choices is not None:
            gdn_mma_m = config[CUTE_GDN_RECURRENCE_MMA_M_KEY]
            if gdn_mma_m not in gdn_tile_mma_m_choices:
                # A defaulted MMA height follows the selected tile and warps;
                # only an explicit height they cannot take is an error.
                if _fix_invalid or not gdn_mma_m_supplied:
                    config[CUTE_GDN_RECURRENCE_MMA_M_KEY] = gdn_tile_mma_m_choices[0]
                else:
                    raise InvalidConfig(
                        f"{CUTE_GDN_RECURRENCE_MMA_M_KEY} must be one of "
                        f"{gdn_tile_mma_m_choices!r} for the selected dstate "
                        f"tile and epilogue warps, got {gdn_mma_m!r}"
                    )
        gdn_tile_stage_choices = self._cute_gdn_recurrence_stage_choices_for(config)
        if gdn_tile_stage_choices is not None:
            gdn_stages = config[CUTE_GDN_RECURRENCE_STAGES_KEY]
            if gdn_stages not in gdn_tile_stage_choices:
                # A depth left to the default follows the selected tile; only
                # an explicit depth the tile cannot hold is an error.
                if _fix_invalid or not gdn_stages_supplied:
                    config[CUTE_GDN_RECURRENCE_STAGES_KEY] = gdn_tile_stage_choices[0]
                else:
                    raise InvalidConfig(
                        f"{CUTE_GDN_RECURRENCE_STAGES_KEY} must be one of "
                        f"{gdn_tile_stage_choices!r} for the selected dstate "
                        f"tile, got {gdn_stages!r}"
                    )
        gdn_tile_group_choices = self._cute_gdn_recurrence_token_group_choices_for(
            config
        )
        if gdn_tile_group_choices is not None:
            gdn_groups = config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY]
            if gdn_groups not in gdn_tile_group_choices:
                # A defaulted group count follows the selected tile and warps;
                # only an explicit count they cannot split is an error.
                if _fix_invalid or not gdn_groups_supplied:
                    config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] = (
                        gdn_tile_group_choices[0]
                    )
                else:
                    raise InvalidConfig(
                        f"{CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY} must be one of "
                        f"{gdn_tile_group_choices!r} for the selected dstate "
                        f"tile and epilogue warps, got {gdn_groups!r}"
                    )
        prepare_schedule_fragment = self.cute_chunk_prepare_schedule
        if prepare_schedule_fragment is not None:
            prepare_schedule = config.setdefault(
                CUTE_CHUNK_PREPARE_SCHEDULE_KEY,
                prepare_schedule_fragment.default(),
            )
            if prepare_schedule not in prepare_schedule_fragment.choices:
                if _fix_invalid:
                    config[CUTE_CHUNK_PREPARE_SCHEDULE_KEY] = (
                        prepare_schedule_fragment.default()
                    )
                else:
                    raise InvalidConfig(
                        f"{CUTE_CHUNK_PREPARE_SCHEDULE_KEY} must be one of "
                        f"{prepare_schedule_fragment.choices!r}, got "
                        f"{prepare_schedule!r}"
                    )
        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_pre_pid_type(
                config,
                fix_invalid=_fix_invalid,
            )
        indirect_modes = (
            self.pallas_indirect_access_modes
            if self.supports_config_key("pallas_indirect_access_mode")
            else ()
        )
        if indirect_modes:
            mode = config.setdefault("pallas_indirect_access_mode", indirect_modes[0])
            if mode not in indirect_modes:
                if _fix_invalid:
                    config["pallas_indirect_access_mode"] = indirect_modes[0]
                else:
                    raise InvalidConfig(
                        "pallas_indirect_access_mode must be one of "
                        f"{indirect_modes!r}, got {mode!r}"
                    )
        else:
            config.pop("pallas_indirect_access_mode", None)
        if self.has_pallas_inner_loops:
            if (
                self.pallas_indirect_dma_requires_fori
                and config.get("pallas_indirect_access_mode") == "dma"
            ):
                loop_type = config.setdefault("pallas_loop_type", "fori_loop")
                if loop_type != "fori_loop":
                    if _fix_invalid:
                        config["pallas_loop_type"] = "fori_loop"
                    else:
                        raise InvalidConfig(
                            "pallas_indirect_access_mode='dma' requires "
                            "pallas_loop_type='fori_loop' for inner-loop gathers"
                        )
            elif self.has_symbolic_or_data_dependent_bounds:
                # "unroll" uses Python range() which can't handle traced bounds.
                # Between the remaining options, prefer "fori_loop": it handles
                # both DMA-aligned and unaligned inner blocks, while
                # "emit_pipeline" fails on unaligned dims.
                config.setdefault("pallas_loop_type", "fori_loop")
            else:
                config.setdefault("pallas_loop_type", VALID_PALLAS_LOOP_TYPES[0])
        use_low_level_scheduler = config.get("pallas_use_low_level_scheduler")
        if use_low_level_scheduler is not None and not isinstance(
            use_low_level_scheduler, bool
        ):
            raise InvalidConfig("pallas_use_low_level_scheduler must be a bool")
        fold_dot_lhs_cast = config.get("pallas_fold_dot_lhs_cast")
        if fold_dot_lhs_cast is not None and not isinstance(fold_dot_lhs_cast, bool):
            raise InvalidConfig("pallas_fold_dot_lhs_cast must be a bool")
        if config.get("pallas_loop_type") == "emit_pipeline":
            group_size = config.get("pallas_emit_pipeline_group_size")
            if group_size is not None and (
                type(group_size) is not int or group_size < 1
            ):
                raise InvalidConfig(
                    "pallas_emit_pipeline_group_size must be a positive integer"
                )
        else:
            config.pop("pallas_emit_pipeline_group_size", None)
        if (
            self.supports_config_key("pallas_load_buffer_count")
            and self.has_pallas_inner_loops
            and config.get("pallas_loop_type") in ("fori_loop", "unroll")
        ):
            values = config.setdefault(
                "pallas_load_buffer_count", self.pallas_load_buffer_count.default()
            )
            expected = self.pallas_load_buffer_count.length
            if (
                not isinstance(values, list)
                or len(values) != expected
                or any(
                    type(value) is not int or value not in (1, 2) for value in values
                )
            ):
                raise InvalidConfig(
                    "pallas_load_buffer_count must be a list containing one "
                    "buffer count (1 or 2) per input tensor "
                    f"(expected {expected}, got {values!r})"
                )
            if expected == 0:
                config.pop("pallas_load_buffer_count")
        else:
            config.pop("pallas_load_buffer_count", None)
        if (
            self.supports_config_key("pallas_pre_broadcast")
            and self.has_pallas_inner_loops
            and config.get("pallas_loop_type") not in ("fori_loop", "emit_pipeline")
        ):
            # The transform widens loop-carried VMEM scratch, so it only applies
            # to the streaming lowerings.  "unroll" carries values through the
            # jax.lax.fori_loop tuple and allocates no scratch to widen; pin the
            # flag off there so both settings do not autotune as distinct configs.
            config.pop("pallas_pre_broadcast", None)

        if self.supports_config_key("pid_type"):
            if "pid_type" in config:
                if config["pid_type"] not in VALID_PID_TYPES:
                    raise InvalidConfig(
                        f"Invalid value for 'pid_type': {config['pid_type']!r} must be one of {list(VALID_PID_TYPES)!r}"
                    )
            else:
                config["pid_type"] = VALID_PID_TYPES[0]

        if self.supports_config_key("xcd_remap"):
            if "xcd_remap" in config:
                if not isinstance(config["xcd_remap"], bool):
                    raise InvalidConfig(
                        f"Invalid value for 'xcd_remap': {config['xcd_remap']!r} must be a bool"
                    )
                if config["xcd_remap"]:
                    pid_type = config.get("pid_type", "flat")
                    if self.num_xcd <= 1:
                        # No-op on single-XCD devices: silently disable rather
                        # than reject (the remap is the identity at NUM_XCDS=1).
                        config["xcd_remap"] = False
                    elif pid_type not in (
                        "flat",
                        "persistent_blocked",
                        "persistent_interleaved",
                    ):
                        # xcd_remap is only defined for flat and the persistent
                        # (blocked / interleaved) PID strategies.
                        if _fix_invalid:
                            config["xcd_remap"] = False
                        else:
                            raise InvalidConfig(
                                "xcd_remap=True requires pid_type in "
                                "{'flat', 'persistent_blocked', 'persistent_interleaved'}"
                            )
                    elif pid_type == "persistent_interleaved":
                        # interleaved remaps each virtual pid, so it needs the
                        # persistent grid stride to be XCD-aligned (this can be
                        # broken by reserved_sms); otherwise a worker spans
                        # multiple XCD regions.  Silently disable (perf no-op).
                        mult = config.get("num_sm_multiplier", 1)
                        if not isinstance(mult, int) or mult < 1:
                            mult = 1
                        if (self.num_sm * mult) % self.num_xcd != 0:
                            config["xcd_remap"] = False
            else:
                config["xcd_remap"] = False
        else:
            config.pop("xcd_remap", None)

        if _fix_invalid and self.backend_name == "cute":
            self._cute_tcgen05_config.fix_search_config(config)

        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_strategy(
                config,
                fix_invalid=_fix_invalid,
            )
            self._normalize_cute_flash(config, fix_invalid=_fix_invalid)
            self._normalize_cute_flash_gated(config, fix_invalid=_fix_invalid)

        if self.supports_config_key("num_sm_multiplier"):
            # The default autotuning domain remains powers of two, while an
            # explicitly selected configuration may use an intermediate worker
            # count when occupancy has a narrow optimum.
            if "num_sm_multiplier" in config:
                val = config["num_sm_multiplier"]
                if (
                    type(val) is not int
                    or val < MIN_NUM_SM_MULTIPLIER
                    or val > MAX_NUM_SM_MULTIPLIER
                ):
                    raise InvalidConfig(
                        f"Invalid value for 'num_sm_multiplier': {val!r} must be an integer between {MIN_NUM_SM_MULTIPLIER} and {MAX_NUM_SM_MULTIPLIER}"
                    )
            else:
                config["num_sm_multiplier"] = DEFAULT_NUM_SM_MULTIPLIER

        # Only validate maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            if "maxnreg" in config:
                value = config["maxnreg"]
                if value is not None and (
                    type(value) is not int or value < MIN_MAXNREG or value > MAX_MAXNREG
                ):
                    raise InvalidConfig(
                        f"Invalid value for 'maxnreg': {value!r} must be None or an integer between {MIN_MAXNREG} and {MAX_MAXNREG}"
                    )
            else:
                config["maxnreg"] = DEFAULT_MAXNREG

            # Cap maxnreg so that maxnreg * threads_per_block doesn't exceed
            # the register file.  On sm100+ ptxas honours .maxnreg over
            # .reqntid, so an uncapped value causes "out of resource: threads"
            # at load.
            maxnreg = cast("int | None", config.get("maxnreg"))
            num_warps = config.get("num_warps", DEFAULT_NUM_WARPS)
            if maxnreg is not None and isinstance(num_warps, int):
                limit = _regs_per_block() // warps_to_threads(num_warps)
                if maxnreg > limit:
                    if _fix_invalid:
                        valid = [
                            v for v in AUTOTUNED_MAXNREG if v is not None and v <= limit
                        ]
                        if valid:
                            config["maxnreg"] = max(valid)
                        else:
                            config.pop("maxnreg", None)
                    else:
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} exceeds register budget for "
                            f"num_warps={num_warps} (max {limit})"
                        )
        else:
            # Remove maxnreg if not supported
            config.pop("maxnreg", None)

        # Handle num_sm_multiplier and maxnreg for non-persistent pid_types
        # These options only make sense for persistent kernels
        pid_type = config.get("pid_type")
        if pid_type in ("flat", "xyz"):
            # Handle num_sm_multiplier
            num_sm_multiplier = config.get(
                "num_sm_multiplier", DEFAULT_NUM_SM_MULTIPLIER
            )
            if num_sm_multiplier != DEFAULT_NUM_SM_MULTIPLIER:
                if _fix_invalid:
                    # Silently fix during autotuning config generation
                    config.pop("num_sm_multiplier", None)
                else:
                    # Raise error for user-specified invalid combinations
                    raise InvalidConfig(
                        f"num_sm_multiplier={num_sm_multiplier} can only be used with persistent "
                        f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                        f"got pid_type={pid_type!r}"
                    )
            else:
                # Remove default value from config
                config.pop("num_sm_multiplier", None)

            # Handle maxnreg - only makes sense for persistent kernels (and only on non-AMD and non-Intel GPU)
            if supports_maxnreg():
                maxnreg = config.get("maxnreg", DEFAULT_MAXNREG)
                if maxnreg != DEFAULT_MAXNREG:
                    if _fix_invalid:
                        # Silently fix during autotuning config generation
                        config.pop("maxnreg", None)
                    else:
                        # Raise error for user-specified invalid combinations
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} can only be used with persistent "
                            f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                            f"got pid_type={pid_type!r}"
                        )
                else:
                    # Remove default value from config
                    config.pop("maxnreg", None)

        if "advanced_controls_file" in config:
            value = config.get("advanced_controls_file") or ""
            if not isinstance(value, str):
                raise InvalidConfig(
                    f"advanced_controls_file must be a string path, got {value!r}"
                )
            config["advanced_controls_file"] = value

        if "epilogue_subtile" in config:
            val = config["epilogue_subtile"]
            # Normalize bool to int for backward compat
            if val is True:
                config["epilogue_subtile"] = 2
            elif not val:
                config.pop("epilogue_subtile", None)
            elif val not in EPILOGUE_SUBTILE_EXTENDED_CHOICES:
                raise InvalidConfig(
                    f"epilogue_subtile must be one of {EPILOGUE_SUBTILE_EXTENDED_CHOICES!r}, got {val!r}"
                )
            elif _fix_invalid and not self._should_keep_epilogue_subtile_for_autotune():
                config.pop("epilogue_subtile", None)
            # Epilogue subtiling is incompatible with flatten_loops because
            # FlattenedTileStrategy does not support offset_var needed by
            # the epilogue store codegen path.
            flatten_loops = config.get("flatten_loops")
            if (
                "epilogue_subtile" in config
                and isinstance(flatten_loops, list)
                and any(flatten_loops)
            ):
                if _fix_invalid:
                    config.pop("epilogue_subtile", None)
                else:
                    raise InvalidConfig(
                        "epilogue_subtile is incompatible with flatten_loops=True"
                    )

        # Set default values for grid indices when pid_type is not persistent
        if pid_type in ("flat", "xyz") and self.grid_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=self.grid_block_ids
                )

        range_warp_specializes = cast(
            "list[bool | None]", config.get("range_warp_specializes", [])
        )

        if range_warp_specializes and any(range_warp_specializes):
            # Only one range_warp_specializes is allowed, take the first one
            # Prefer warp specialize on outermost loop
            first_idx = range_warp_specializes.index(True)
            for i in range(first_idx + 1, len(range_warp_specializes)):
                range_warp_specializes[i] = None

            range_unroll_factors = cast(
                "list[int]", config.get("range_unroll_factors", [])
            )
            if range_unroll_factors and range_unroll_factors[first_idx] > 1:
                if range_unroll_factors[first_idx]:
                    range_unroll_factors[first_idx] = 0

                config["range_unroll_factors"] = range_unroll_factors

        if self.supports_config_key("range_warp_specializes"):
            config["range_warp_specializes"] = range_warp_specializes

        if self.backend_name == "cute":
            preserve_keys = self._cute_tcgen05_config.implicit_default_keys_to_preserve(
                config
            )
            for key in _CUTE_IMPLICIT_DEFAULT_KEYS - provided_keys - preserve_keys:
                config.pop(key, None)

        if self.backend_name == "cute":
            validate_cluster_config(self, config, fix_invalid=_fix_invalid)
            normalize_cluster_finalizer(
                config,
                available=self.cute_split_k_cluster_facts is not None,
                fix_invalid=_fix_invalid,
            )

        # Allow tunable parameter keys in addition to backend-supported keys.
        allowed_keys = self.supported_config_keys() | {
            *self.user_defined_tunables.keys()
        }
        if invalid_keys := ({*config} - allowed_keys):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def raise_grid_block_minimums(self) -> None:
        """Raise search floors for grid block dimensions based on problem size.

        Very small block sizes produce enormous grids that the autotuner
        wastes time exploring.  This heuristic sets a floor so the total
        number of blocks per dimension stays within a reasonable range
        derived from ``num_compute_units``.

        The raised minimum never exceeds the default block size that
        ``_fragment`` would compute, so memory and shared-memory
        constraints from non-tiled dimensions are respected.
        """
        if not self.grid_block_ids:
            return

        n_cus = num_compute_units()
        n_dims = len(self.grid_block_ids)
        max_blocks_per_dim = math.ceil((n_cus * 64) ** (1.0 / n_dims))
        independent_max_blocks: dict[int, int] = {
            block_id: math.ceil((n_cus * 64) ** (1.0 / len(grid)))
            for grid in self.cute_pointwise_region_grid_groups
            for block_id in grid
        }

        for grid_bid in self.grid_block_ids:
            try:
                spec = self.block_sizes.block_id_lookup(grid_bid)
            except KeyError:
                continue
            if spec.size_hint <= 0:
                continue
            default = spec._fragment(self).default_val
            min_block = spec.size_hint // independent_max_blocks.get(
                grid_bid, max_blocks_per_dim
            )
            min_block = min(min_block, default)
            if min_block >= 2:
                min_block = 1 << (min_block.bit_length() - 1)
                min_block = min(min_block, spec.max_size)
                spec.autotuner_min = assert_integer_power_of_two(
                    max(min_block, spec.autotuner_min)
                )

    def create_config_generation(
        self,
        *,
        overrides: Mapping[str, object] | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
        compiler_coverage_enabled: bool = True,
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        effective_overrides = dict(overrides) if overrides else None
        family_override: str | None = None
        if self.cute_flash_search_enabled and effective_overrides:
            if FLASH_PIPELINE_FAMILY_KEY not in effective_overrides:
                legacy = {
                    key: effective_overrides[key]
                    for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
                    if effective_overrides.get(key) is not None
                }
                if legacy:
                    if FLASH_PERSISTENT_KEY in effective_overrides:
                        legacy[FLASH_PERSISTENT_KEY] = effective_overrides[
                            FLASH_PERSISTENT_KEY
                        ]
                    effective_overrides[FLASH_PIPELINE_FAMILY_KEY] = (
                        self._resolve_cute_flash_config(legacy).pipeline_family
                    )

            requested_packet = effective_overrides.get(FLASH_EXP2_PACKET_KEY)
            if flash_exp2_packet_is_compound(requested_packet):
                assert self._cute_flash_head_dim is not None
                assert self._cute_flash_num_kv is not None
                packet_requirements = _flash_compound_exp2_packet_overrides(
                    self._cute_flash_head_dim,
                    self._cute_flash_num_kv,
                    {
                        key: value
                        for key, value in effective_overrides.items()
                        if key != FLASH_PIPELINE_FAMILY_KEY
                    },
                    dtype=self._cute_flash_dtype,
                    is_causal=self._cute_flash_is_causal,
                    has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                    requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                    small_biased_candidate=self._cute_flash_small_biased_candidate,
                    standard_dense_output=self._cute_flash_standard_dense_output,
                    standard_causal_output=self._cute_flash_standard_causal_output,
                )
                if not packet_requirements:
                    raise InvalidConfig(
                        f"{FLASH_EXP2_PACKET_KEY}={requested_packet!r} is not "
                        "effective for this kernel"
                    )
                for key, required_value in packet_requirements.items():
                    if (
                        key in effective_overrides
                        and effective_overrides[key] != required_value
                    ):
                        raise InvalidConfig(
                            f"{FLASH_EXP2_PACKET_KEY}={requested_packet!r} "
                            f"requires {key}={required_value!r}, got "
                            f"{effective_overrides[key]!r}"
                        )
                    effective_overrides[key] = required_value

            exact_family = effective_overrides.get(FLASH_PIPELINE_FAMILY_KEY)
            family_flags = _flash_pipeline_family_flags(exact_family)
            if family_flags is not None:
                family_override = cast("str", exact_family)
                if family_flags.use_clc_scheduler or family_flags.local_tma_partition:
                    if effective_overrides.get(FLASH_PERSISTENT_KEY) is False:
                        raise InvalidConfig(
                            f"{FLASH_PIPELINE_FAMILY_KEY}={family_override!r} "
                            f"requires {FLASH_PERSISTENT_KEY}=True"
                        )
                    effective_overrides[FLASH_PERSISTENT_KEY] = True

            if self._cute_flash_output_requires_tma:
                if effective_overrides.get(FLASH_EPI_TMA_KEY) is False:
                    raise InvalidConfig(
                        f"{FLASH_EPI_TMA_KEY}=False is not legal for this output shape"
                    )
                effective_overrides[FLASH_EPI_TMA_KEY] = True
        return ConfigGeneration(
            self,
            overrides=effective_overrides,
            _flash_pipeline_family_override=family_override,
            advanced_controls_files=advanced_controls_files,
            process_group_name=process_group_name,
            compiler_coverage_enabled=compiler_coverage_enabled,
        )

    def flatten_missing_field_default(
        self,
        key: str,
        config: dict[str, object],
    ) -> tuple[bool, object]:
        if self.backend_name == "cute":
            if key == "cute_materialized_schedule":
                return True, "off"
            if key == "cute_materialized_operand_schedule":
                return True, "off"
            if key == "cute_fragment_warp_results":
                return True, False
            if key == "cute_fragment_register_loads":
                return True, False
            if key == "cute_fragment_threads":
                return True, 128
            if key in ("cute_fragment_scan", "cute_fragment_reduction"):
                return True, "serial"
            if key == "cute_signed_bitfield_bf16":
                return True, False
            if self.cute_flash_search_enabled and key == FLASH_PIPELINE_FAMILY_KEY:
                return True, self._resolve_cute_flash_config(config).pipeline_family
            return self._cute_tcgen05_config.flatten_missing_field_default(key, config)
        return False, None

    def prepare_override_normalization(
        self,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        if self.backend_name == "cute":
            family = overrides.get(FLASH_PIPELINE_FAMILY_KEY)
            family_flags = _flash_pipeline_family_flags(family)
            if self.cute_flash_search_enabled and family_flags is not None:
                effective_family = self._resolve_cute_flash_config(
                    {FLASH_PIPELINE_FAMILY_KEY: family}
                ).pipeline_family
                if effective_family != family:
                    raise InvalidConfig(
                        f"cute_flash_pipeline_family={family!r} is not effective "
                        f"for this kernel; it normalizes to {effective_family!r}"
                    )
            if self.cute_flash_search_enabled and FLASH_EXP2_PACKET_KEY in overrides:
                assert self._cute_flash_head_dim is not None
                assert self._cute_flash_num_kv is not None
                packet_requirements = _flash_compound_exp2_packet_overrides(
                    self._cute_flash_head_dim,
                    self._cute_flash_num_kv,
                    {
                        key: value
                        for key, value in overrides.items()
                        if key != FLASH_PIPELINE_FAMILY_KEY
                    },
                    dtype=self._cute_flash_dtype,
                    is_causal=self._cute_flash_is_causal,
                    has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                    requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                    small_biased_candidate=self._cute_flash_small_biased_candidate,
                    standard_dense_output=self._cute_flash_standard_dense_output,
                    standard_causal_output=self._cute_flash_standard_causal_output,
                )
                requested_packet = overrides[FLASH_EXP2_PACKET_KEY]
                for key, required_value in packet_requirements.items():
                    if key in overrides and overrides[key] != required_value:
                        raise InvalidConfig(
                            f"cute_flash_exp2_packet={requested_packet!r} "
                            f"requires {key}={required_value!r}, got "
                            f"{overrides[key]!r}"
                        )
                if family_flags is not None:
                    effective_packet = self._resolve_cute_flash_config(
                        {
                            FLASH_PIPELINE_FAMILY_KEY: family,
                            FLASH_EXP2_PACKET_KEY: requested_packet,
                        }
                    ).exp2_packet
                    if effective_packet != requested_packet:
                        raise InvalidConfig(
                            f"cute_flash_exp2_packet={requested_packet!r} is not "
                            f"effective with cute_flash_pipeline_family={family!r}"
                        )
            if (
                self.cute_flash_search_enabled
                and family_flags is not None
                and (family_flags.use_clc_scheduler or family_flags.local_tma_partition)
            ):
                if overrides.get(FLASH_PERSISTENT_KEY) is False:
                    raise InvalidConfig(
                        f"cute_flash_pipeline_family={family!r} requires "
                        "cute_flash_persistent=True"
                    )
                config[FLASH_PERSISTENT_KEY] = True
            self._cute_tcgen05_config.prepare_override_normalization(
                config,
                overrides,
            )

    def _base_default_config(self) -> helion.Config:
        config = self.flat_config(lambda x: x.default())
        # The additive finalizer choice must not change existing seed configs.
        config.config.pop(SPLIT_K_FINALIZER_KEY, None)
        if self.cute_matmul_min_blocks_search_enabled:
            # This optional matmul coordinate has no old default Config key.
            # Keep the public default exact; flat search still represents zero.
            config.config.pop("cute_min_blocks_per_mp", None)
        self._shrink_for_numel_constraints(config)
        return config

    def autotune_reference_config(self) -> helion.Config:
        """Return the conservative fragment config used as the autotuning
        reference."""
        return self._base_default_config()

    def default_config(self) -> helion.Config:
        """Return the config used for execution when autotuning is disabled."""
        if self.compiler_default_config is None:
            return self.autotune_reference_config()
        # A promoted seed only specifies the knobs it cares about (e.g. block_sizes); layer it over
        # the full base defaults so every other key — including user register_tunable defaults — is
        # preserved rather than dropped.
        merged = dict(self.autotune_reference_config().config)
        merged.update(self.compiler_default_config.config)
        config = helion.Config.from_dict(merged)
        # Then normalize, so a promoted compiler default has the same canonical field set as the
        # ``_base_default_config`` path: without this its ``repr``/equality differs from its own
        # flatten/unflatten round-trip, which breaks callers that key on the config identity
        # (e.g. benchmark result maps).
        self.normalize(config, _fix_invalid=True)
        self._shrink_for_numel_constraints(config)
        return config

    def _shrink_for_numel_constraints(self, config: helion.Config) -> None:
        """Shrink block_sizes in *config* in-place so every tensor numel
        constraint is satisfied.
        """
        block_sizes = config.config.get("block_sizes")
        if (
            not isinstance(block_sizes, list)
            or not block_sizes
            or not self.tensor_numel_constraints
        ):
            return
        min_sizes = [
            max(self.block_sizes[i].min_size, 1) for i in range(len(block_sizes))
        ]
        shrink_block_sizes_for_numel_constraints(
            self.tensor_numel_constraints, block_sizes, min_sizes
        )

    def shrink_block_sizes_once(self, config: helion.Config) -> helion.Config | None:
        """Return a copy of *config* with its largest block size halved.

        ``None`` once every block size sits at its minimum.  Used to back off a
        config Helion chose itself after the backend compiler rejected it for a
        hardware limit Helion cannot model (scratch space, shared memory,
        register pressure).
        """
        block_sizes = config.config.get("block_sizes")
        if not isinstance(block_sizes, list) or not block_sizes:
            return None
        best_idx: int | None = None
        best_val = -1
        for i, value in enumerate(block_sizes):
            if not isinstance(value, int):
                continue
            if value // 2 >= max(self.block_sizes[i].min_size, 1) and value > best_val:
                best_val = value
                best_idx = i
        if best_idx is None:
            return None
        shrunk = [*block_sizes]
        shrunk[best_idx] //= 2
        new_config = helion.Config.from_dict({**config.config, "block_sizes": shrunk})
        self.normalize(new_config, _fix_invalid=True)
        return new_config

    def iter_search_dimensions(
        self, value_limit: int = 100
    ) -> Iterator[SearchDimensionInfo]:
        """Yield one :class:`SearchDimensionInfo` per tunable field.

        Public entry point for describing the search space. Scalar fragments
        report their own ``cardinality()``/``search_values()``; a
        ``BlockIdSequence`` reports the product of its per-item cardinalities.
        Fields whose sizing depends on ConfigSpec-level state rather than the
        fragment (e.g. ``pid_type``, ``num_stages``) are left to the caller.
        """
        from .block_id_sequence import BlockIdSequence

        for name, field in self._flat_fields().items():
            if isinstance(field, BlockIdSequence):
                yield SearchDimensionInfo(
                    name=name,
                    cardinality=field.cardinality(self),
                    values=None,
                    is_sequence=True,
                    num_items=len(field),
                )
            else:
                yield SearchDimensionInfo(
                    name=name,
                    cardinality=field.cardinality(),
                    values=field.search_values(value_limit),
                    is_sequence=False,
                    num_items=0,
                )

    def _flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        return self._flat_fields_with_flash_family()

    def _flat_fields_with_flash_family(
        self,
        _flash_pipeline_family_override: str | None = None,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        """Return {key: field} for all tunable fields in flat_config() order.

        This is the single source of truth for field ordering.
        """
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "block_sizes": self.block_sizes,
        }
        if self.backend_name == "cute":
            if self.cute_signed_bitfield_bf16_available:
                fields["cute_signed_bitfield_bf16"] = BooleanFragment()
            if self.cute_host_paired_sum_available:
                fields["cute_host_paired_sum"] = EnumFragment(
                    choices=("off", "mapped", "narrow")
                )
            if self.cute_materialized_schedule_search_enabled:
                fields["cute_materialized_schedule"] = EnumFragment(
                    choices=self.cute_materialized_schedule_choices
                )
            if self.cute_materialized_operand_schedule_search_enabled:
                fields["cute_materialized_operand_schedule"] = EnumFragment(
                    choices=("off", "warp_narrow4")
                )
            if self.cute_fragment_warp_results_search_enabled:
                fields["cute_fragment_warp_results"] = BooleanFragment()
            if self.cute_fragment_register_loads_search_enabled:
                fields["cute_fragment_register_loads"] = BooleanFragment()
            if self.cute_fragment_threads_search_enabled:
                from .._compiler.autotuner_heuristics.cute_fragment_threads import (
                    THREADS,
                )

                fields["cute_fragment_threads"] = EnumFragment(choices=THREADS)
            if self.cute_fragment_scan_search_enabled:
                fields["cute_fragment_scan"] = EnumFragment(
                    choices=("serial", "cooperative")
                )
            if self.cute_fragment_reduction_search_enabled:
                fields["cute_fragment_reduction"] = EnumFragment(
                    choices=("serial", "warp")
                )
            if self.cute_pointwise_region_block_ids:
                fields["cute_pointwise_pid_type"] = EnumFragment(
                    choices=("inherit", "flat")
                )
            if self.cute_topk_search_enabled:
                fields.update(
                    (key, EnumFragment(choices=choices))
                    for key, choices in self.cute_topk_choices.items()
                )
            elif self.cute_tcgen05_search_enabled:
                fields.update(self._cute_tcgen05_config.flat_fields())
                if self.cute_pointwise_region_block_ids:
                    fields["num_threads"] = self.num_threads
                    fields["cute_vector_widths"] = self.cute_vector_widths
                    fields["cute_lane_layouts"] = self.cute_lane_layouts
            elif self.cute_flash_search_enabled:
                fields.update(
                    self._cute_flash_autotune_fragments(
                        pipeline_family_override=_flash_pipeline_family_override,
                    )
                )
            elif self.cute_flash_gated_search_enabled:
                from .._compiler.cute.cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY

                default = self._cute_flash_gated_kv_stage_default
                fields[FLASH_KV_STAGE_KEY] = EnumFragment(
                    choices=(
                        default,
                        *(
                            choice
                            for choice in self._cute_flash_gated_kv_stage_choices
                            if choice != default
                        ),
                    )
                )
                wg_default = self._cute_flash_gated_gate_warpgroup_default
                fields[FLASH_GATE_WARPGROUPS_KEY] = EnumFragment(
                    choices=(
                        wg_default,
                        *(
                            choice
                            for choice in self._cute_flash_gated_gate_warpgroup_choices
                            if choice != wg_default
                        ),
                    )
                )
            elif self.cute_flash_bwd_search_enabled:
                fields["cute_flash_bwd_persistent"] = EnumFragment(choices=(0, 1))
                fields["cute_flash_bwd_two_cta"] = EnumFragment(
                    choices=(0, 1) if self._cute_flash_bwd_two_cta_allowed else (0,)
                )
                fields["cute_flash_bwd_exp2_f32"] = EnumFragment(choices=(0, 1))
            elif self.supports_config_key("num_threads"):
                fields["num_threads"] = self.num_threads
                # Loop flattening is a real codegen choice on the SIMT path
                # (flattened multi-dim tiles vectorize odd-row-length
                # pointwise kernels via flat base pointers).  Without this
                # entry the flat form was unreachable: seeds carrying
                # ``flatten_loops=[True]`` silently lost the flag in the
                # flat-config round trip and were benchmarked as their
                # (much slower) N-D counterparts.
                if (
                    self.supports_config_key("flatten_loops")
                    and len(self.flatten_loops) > 0
                ):
                    fields["flatten_loops"] = self.flatten_loops
                # Rolled-reduction loop chunks are tunable on the SIMT path
                # (LoopedReductionStrategy lane lattices).  Without this
                # entry the chunk silently pins to its default and neither
                # the seeds nor the search can change it.
                if (
                    self.supports_config_key("reduction_loops")
                    and len(self.reduction_loops) > 0
                ):
                    fields["reduction_loops"] = self.reduction_loops
                # Per-load-site L1 eviction hints (streamed rows want
                # evict_first; re-read rows want evict_last until the
                # final sweep).
                if (
                    self.supports_config_key("load_eviction_policies")
                    and self.load_eviction_policies.length > 0
                ):
                    fields["load_eviction_policies"] = self.load_eviction_policies
                # Universal pid emission honors ``loop_orders`` and the
                # better order is shape-dependent. tcgen05 exposes the same
                # field from CuteTcgen05Config.flat_fields().
                if (
                    self.supports_config_key("loop_orders")
                    and len(self.loop_orders) > 0
                ):
                    fields["loop_orders"] = self.loop_orders
                # Expose ``cute_vector_widths`` per-block so the
                # autotuner can vary V in {1, 2, 4, 8} for lane-loop
                # vec loads (and for ``LoopedReductionStrategy`` rolled
                # reductions).  Without this entry, ``flatten`` strips V
                # back to the default of 1, defeating the seed
                # heuristics that try to bias toward LDG.128 lattices.
                if (
                    self.supports_config_key("cute_vector_widths")
                    and len(self.cute_vector_widths) > 0
                ):
                    fields["cute_vector_widths"] = self.cute_vector_widths
                # Per-block lane layout (blocked vs strided per-thread
                # element assignment) for lane loops with more than one
                # iteration; strided gives fully coalesced warp accesses.
                if (
                    self.supports_config_key("cute_lane_layouts")
                    and len(self.cute_lane_layouts) > 0
                ):
                    fields["cute_lane_layouts"] = self.cute_lane_layouts
                # Where multi-sweep rolled reductions keep re-read values
                # (register fragment vs gmem/L2 reload); see
                # ``CuteReductionReloadSpec``.
                if (
                    self.supports_config_key("cute_reduction_reloads")
                    and len(self.cute_reduction_reloads) > 0
                ):
                    fields["cute_reduction_reloads"] = self.cute_reduction_reloads
                if self.cute_sequence_reduction_blocks:
                    fields["cute_reduction_sequence"] = EnumFragment(
                        choices=(
                            "scalar",
                            "reload",
                            "resident",
                            "bounded",
                            "bounded_layout",
                        )
                    )
                if self.cute_resident_reduction_blocks:
                    fields["cute_reduction_local_tree"] = BooleanFragment()
                    fields["cute_reduction_schedule"] = EnumFragment(
                        choices=("scalar", "resident", "pipelined")
                    )
                    fields["cute_reduction_group_rows"] = EnumFragment(
                        choices=(1, 2, 4, 3)
                    )
                    fields["cute_reduction_row_schedule"] = EnumFragment(
                        choices=("batched", "serial", "serial_deferred")
                    )
                    if (
                        self.target_device_capability is not None
                        and self.target_device_capability[0] == 10
                    ):
                        fields["cute_reduction_pack_output"] = BooleanFragment()
                    fields["cute_reduction_pipeline_depth"] = EnumFragment(
                        choices=(2, 4)
                    )
                if self.cute_async_load_pipeline_enabled:
                    fields["cute_async_load_stages"] = EnumFragment(
                        choices=(0, 3, 4, 5)
                    )
                    fields["cute_async_load_lookahead"] = EnumFragment(
                        choices=(4, 2, 3)
                    )
                    fields["cute_async_load_group_rows"] = EnumFragment(choices=(2, 4))
                    fields["cute_async_load_cache"] = EnumFragment(choices=("cg", "ca"))
                    fields["cute_async_store_policy"] = EnumFragment(
                        choices=(
                            ("default", "l2_evict_last")
                            if self.cute_l2_evict_last_store_policy_supported
                            else ("default",)
                        )
                    )
                if self.cute_bf16x2_recurrence_enabled:
                    fields["cute_bf16x2_recurrence"] = BooleanFragment()
                if self.matmul_facts:
                    if (
                        self.cute_grouped_rna_k_choices
                        or self.cute_grouped_warp_tf32_available
                    ):
                        fields[CUTE_MMA_F32_CONVERSION_KEY] = EnumFragment(
                            choices=(CUTE_MMA_F32_AUTO,)
                            + (
                                (CUTE_MMA_F32_TMA_RN,)
                                if self.cute_grouped_rna_k_choices
                                else ()
                            )
                            + (
                                (CUTE_MMA_F32_WARP_RAW,)
                                if self.cute_grouped_warp_tf32_available
                                else ()
                            )
                        )
                    if self.cute_grouped_rna_k_choices:
                        fields[GROUPED_RNA_WARPS_KEY] = EnumFragment(
                            choices=GROUPED_RNA_WARPS
                        )
                        fields[GROUPED_RNA_K_KEY] = EnumFragment(
                            choices=self.cute_grouped_rna_k_choices
                        )
                        if self.cute_grouped_rn_two_stage_ctas:
                            fields[GROUPED_RNA_STAGES_KEY] = EnumFragment(
                                choices=(0, 2)
                            )
                            fields[GROUPED_RNA_RESIDENT_CTAS_KEY] = EnumFragment(
                                choices=self.cute_grouped_rn_two_stage_ctas
                            )
                        if self.cute_grouped_wrapped_descriptors_available:
                            fields[DESCRIPTOR_KEY] = EnumFragment(
                                choices=(DYNAMIC, WRAPPED)
                            )
                        if self.cute_grouped_block_prefix_recipes:
                            fields[GROUPED_PREFIX_SCAN_KEY] = EnumFragment(
                                choices=GROUPED_PREFIX_SCANS
                            )
                    if self.cute_split_k_cluster_facts is not None:
                        fields[SPLIT_K_SCHEDULE_KEY] = EnumFragment(SPLIT_K_SCHEDULES)
                        fields[SPLIT_K_FINALIZER_KEY] = EnumFragment(FINALIZER_WARPS)
                    if self.cute_split_k_workspace_available:
                        fields[SPLIT_K_WORKSPACE_KEY] = BooleanFragment()
                        fields[SPLIT_K_STAGES_KEY] = IntegerFragment(1, 16, 2)
                    fields["cute_collective_mma"] = BooleanFragment()
                    fields["cute_collective_static_layouts"] = BooleanFragment()
                    native_collective = (
                        self.target_device_capability is not None
                        and self.target_device_capability[0] == 10
                    )
                    gathered_collective = native_collective and any(
                        seed.config.get("cute_collective_compute") == "tma_gather"
                        for seed in self.compiler_seed_configs
                    )
                    fields["cute_collective_compute"] = EnumFragment(
                        choices=("warp", "tcgen05", "tma_gather")
                        if native_collective
                        else ("warp",),
                        search_choices=("warp", "tcgen05", "tma_gather")
                        if gathered_collective
                        else ("warp", "tcgen05")
                        if native_collective
                        else ("warp",),
                    )
                    fields["cute_collective_native_seeded"] = (
                        BooleanFragment()
                        if native_collective
                        else EnumFragment(choices=(False,))
                    )
                    fields["cute_collective_tmem_seed"] = (
                        BooleanFragment()
                        if native_collective
                        else EnumFragment(choices=(False,))
                    )
                    fields["cute_collective_tmem_a"] = (
                        BooleanFragment()
                        if native_collective
                        else EnumFragment(choices=(False,))
                    )
                    fields["cute_collective_stages"] = EnumFragment(
                        choices=(1, 2, 3, 4)
                    )
                    fields["cute_collective_copy"] = EnumFragment(
                        choices=("scalar", "async", "async_cached")
                    )
                    fields["cute_collective_recipe"] = EnumFragment(
                        choices=("scalar", "vector", "vector_unrolled")
                    )
                    fields["cute_collective_operand_packets"] = EnumFragment(
                        choices=(False, True),
                        search_choices=(False, True)
                        if any(
                            fact.lhs_dtype == torch.float32
                            for fact in self.matmul_facts
                        )
                        else (False,),
                    )
                    fields["cute_collective_epilogue"] = EnumFragment(
                        choices=("scalar", "vector", "vector_unrolled")
                    )
                    fields["cute_gathered_mma_n"] = EnumFragment(
                        choices=(256, 128, 512),
                        search_choices=(256, 128, 512)
                        if gathered_collective
                        else (256,),
                    )
                    fields["cute_gathered_mma_stages"] = EnumFragment(
                        choices=(3, 2, 4),
                        search_choices=(3, 2, 4) if gathered_collective else (3,),
                    )
                if self.cute_proven_bounds_enabled:
                    fields["cute_proven_bounds"] = BooleanFragment()
                if len(self.cute_lane_layouts) > 0 and not self.matmul_facts:
                    for key in (
                        "cute_independent_reduction",
                        "cute_replicated_reduction",
                        "cute_vector_packet_unroll",
                    ):
                        seeded = any(
                            seed.config.get(key) is True
                            for seed in self.compiler_seed_configs
                        )
                        fields[key] = EnumFragment(
                            (False, True),
                            search_choices=(False, True) if seeded else (False,),
                        )
                if len(self.cute_lane_layouts) > 0 and not self.matmul_facts:
                    sink_seeded = any(
                        seed.config.get("cute_vloop_sink") is True
                        for seed in self.compiler_seed_configs
                    )
                    unroll_seeded = any(
                        cast("int", seed.config.get("cute_lane_unroll", 1)) > 1
                        for seed in self.compiler_seed_configs
                    )
                    fields["cute_vloop_sink"] = EnumFragment(
                        (False, True),
                        search_choices=(False, True) if sink_seeded else (False,),
                    )
                    fields["cute_lane_unroll"] = EnumFragment(
                        _CUTE_LANE_UNROLL_CHOICES,
                        search_choices=_CUTE_LANE_UNROLL_CHOICES
                        if sink_seeded or unroll_seeded
                        else (1,),
                    )
                if self.supports_config_key("cute_pdl"):
                    # A launch attribute plus one wait, live on every kernel
                    # of this branch (the tcgen05 and flash families, whose
                    # own waits and plans leave it inert, build their fields
                    # above) and searched where ``griddepcontrol`` exists,
                    # sm_90 and up.
                    pdl_searched = (
                        self.target_device_capability is not None
                        and self.target_device_capability >= (9, 0)
                    )
                    fields["cute_pdl"] = EnumFragment(
                        (False, True),
                        search_choices=(False, True) if pdl_searched else (False,),
                    )
                if self.cute_rng_packet_enabled:
                    fields["cute_rng_packet"] = BooleanFragment()
                if self.cute_packet_prefetch_enabled:
                    fields["cute_packet_prefetch"] = EnumFragment(choices=(0, 2, 4, 8))
                if (
                    self.matmul_facts
                    and self.target_device_capability is not None
                    and self.target_device_capability[0] >= 8
                ):
                    chain_seeded = any(
                        seed.config.get("cute_register_chain") is True
                        for seed in self.compiler_seed_configs
                    )
                    fields["cute_register_chain"] = EnumFragment(
                        (False, True),
                        search_choices=(False, True) if chain_seeded else (False,),
                    )
                # CuTe's SIMT search normally has no pid_type coordinate.  A
                # metadata-specialized compiler seed may nevertheless prove one
                # exact 3-D ``xyz`` launch safe after the earlier, deliberately
                # conservative grid-size gate rejected it.  Preserve that seed
                # through flatten/unflatten while keeping ``xyz`` out of random
                # search when it is absent from ``allowed_pid_types``.
                if self.supports_config_key("pid_type") and any(
                    seed.config.get("pid_type") == "xyz"
                    for seed in self.compiler_seed_configs
                ):
                    searchable_pid_types = tuple(
                        pid_type
                        for pid_type in self.allowed_pid_types
                        if pid_type in ("flat", "xyz")
                    )
                    if searchable_pid_types:
                        choices = tuple(dict.fromkeys((*searchable_pid_types, "xyz")))
                        fields["pid_type"] = EnumFragment(
                            choices,
                            search_choices=searchable_pid_types,
                        )
                # Thread-block cluster width for SIMT reduction kernels:
                # splits a whole-extent lane-looped axis across cluster
                # CTAs (register-resident slices) with a DSM cluster
                # reduce.  Ignored (cluster 1) when the config shape
                # doesn't qualify — see
                # ``PerThreadNDTileStrategy._maybe_apply_cute_cluster``.
                if (
                    self.supports_config_key("cute_cluster_n")
                    and len(self.cute_lane_layouts) > 0
                    and not self.matmul_facts
                ):
                    fields["cute_cluster_n"] = EnumFragment(choices=(1, 2, 4, 8, 16))
                # Minimum resident CTAs per SM (ptxas ``minnctapersm``):
                # caps registers per thread so more CTAs fit, trading
                # spills for occupancy.  0 leaves ptxas free.  6 is a
                # hard squeeze (at 128 threads it caps registers at 85 ->
                # 24 warps/SM); measured on B200 softmax rows, one extra
                # resident CTA beats ~10 spilled registers.
                if (
                    self.supports_config_key("cute_min_blocks_per_mp")
                    and len(self.cute_lane_layouts) > 0
                    and not self.matmul_facts
                ):
                    fields["cute_min_blocks_per_mp"] = EnumFragment(
                        choices=(0, 1, 2, 3, 4, 6)
                    )
            if (
                not self.cute_flash_search_enabled
                and self.epilogue_subtile_autotune_choices is not None
            ):
                fields["epilogue_subtile"] = EnumFragment(
                    choices=self.epilogue_subtile_autotune_choices
                )
            if self.cute_chunk_recurrence_dv_partitions is not None:
                fields[CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY] = (
                    self.cute_chunk_recurrence_dv_partitions
                )
            if self.cute_chunk_recurrence_register_cap is not None:
                fields[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] = (
                    self.cute_chunk_recurrence_register_cap
                )
            if self.cute_gdn_recurrence_stages is not None:
                fields[CUTE_GDN_RECURRENCE_STAGES_KEY] = self.cute_gdn_recurrence_stages
            if self.cute_gdn_recurrence_epilogue_warps is not None:
                fields[CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY] = (
                    self.cute_gdn_recurrence_epilogue_warps
                )
            if self.cute_gdn_recurrence_token_groups is not None:
                fields[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] = (
                    self.cute_gdn_recurrence_token_groups
                )
            if self.cute_gdn_recurrence_mma_m is not None:
                fields[CUTE_GDN_RECURRENCE_MMA_M_KEY] = self.cute_gdn_recurrence_mma_m
            if self.cute_chunk_prepare_schedule is not None:
                fields[CUTE_CHUNK_PREPARE_SCHEDULE_KEY] = (
                    self.cute_chunk_prepare_schedule
                )
            if self.cute_affine_scan_schedule is not None:
                fields[CUTE_AFFINE_SCAN_SCHEDULE_KEY] = self.cute_affine_scan_schedule
            if (
                self.cute_scaled_mma_available
                and self.target_device_capability is not None
                and self.target_device_capability[0] == 10
            ):
                fields.update(
                    (key, EnumFragment(choices=choices))
                    for key, choices in BLOCK_SCALED_CHOICES.items()
                )
            if self.cute_matmul_min_blocks_search_enabled:
                fields["cute_min_blocks_per_mp"] = EnumFragment(choices=(0, 1))
            if (
                self.supports_config_key("pid_type")
                and "pid_type" not in fields
                and "flat" not in self.allowed_pid_types
            ):
                # SIMT normally omits this coordinate and defaults to flat.
                # Barriers require a persistent launch in the reference and in
                # every candidate, including after a flatten/unflatten round trip.
                fields["pid_type"] = EnumFragment(self.allowed_pid_types)
            fields.update(self.user_defined_tunables)
            return fields

        # Only add sequence keys that the backend supports
        fields.update(
            {
                name: seq
                for name, seq in [
                    ("loop_orders", self.loop_orders),
                    ("flatten_loops", self.flatten_loops),
                    ("l2_groupings", self.l2_groupings),
                    ("reduction_loops", self.reduction_loops),
                    ("range_unroll_factors", self.range_unroll_factors),
                    ("range_warp_specializes", self.range_warp_specialize),
                    ("range_num_stages", self.range_num_stages),
                    ("range_multi_buffers", self.range_multi_buffers),
                    ("range_flattens", self.range_flattens),
                    ("static_ranges", self.static_ranges),
                ]
                if self.supports_config_key(name)
            }
        )

        # Scalar fields (ConfigSpecFragment)
        is_tileir = self.backend_name == "tileir"
        if is_tileir:
            # TileIR: num_warps is unused (fixed at 4), num_stages has wider range
            num_warps_fragment: ConfigSpecFragment = NumWarpsFragment(4, 4)
        elif supports_amd_cdna_tunables():
            num_warps_fragment = NumWarpsFragment(1, 16, DEFAULT_NUM_WARPS)
        else:
            num_warps_fragment = NumWarpsFragment(1, 32, DEFAULT_NUM_WARPS)
        num_stages_fragment = self._num_stages_fragment()

        if self.supports_config_key("num_warps"):
            fields["num_warps"] = num_warps_fragment
        if self.supports_config_key("num_stages"):
            fields["num_stages"] = num_stages_fragment
        if self.supports_config_key("indexing"):
            fields["indexing"] = self.indexing
        if self.supports_config_key("atomic_indexing"):
            fields["atomic_indexing"] = self.atomic_indexing
        if self.supports_config_key("host_tensor_descriptors") and (
            self.indexing.length > 0 or self.atomic_indexing.length > 0
        ):
            fields["host_tensor_descriptors"] = BooleanFragment()
        if (
            self.supports_config_key("pallas_load_buffer_count")
            and self.has_pallas_inner_loops
            and self.pallas_load_buffer_count.length > 0
        ):
            fields["pallas_load_buffer_count"] = self.pallas_load_buffer_count
        if (
            self.supports_config_key("pallas_indirect_access_mode")
            and self.pallas_indirect_access_modes
        ):
            fields["pallas_indirect_access_mode"] = EnumFragment(
                choices=self.pallas_indirect_access_modes
            )
        if self.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
        if self.cross_loop_pipeline is not None:
            fields["cross_loop_pipeline"] = self.cross_loop_pipeline
        if self.supports_config_key("xcd_remap") and self.num_xcd > 1:
            fields["xcd_remap"] = BooleanFragment()
        if self.supports_config_key("num_sm_multiplier"):
            fields["num_sm_multiplier"] = PowerOfTwoFragment(
                MIN_NUM_SM_MULTIPLIER,
                self.max_num_sm_multiplier,
                DEFAULT_NUM_SM_MULTIPLIER,
            )
        if self.supports_config_key("load_eviction_policies"):
            fields["load_eviction_policies"] = self.load_eviction_policies
        if (
            self.supports_config_key("load_cache_modifiers")
            and self.load_cache_modifiers.length > 0
        ):
            fields["load_cache_modifiers"] = self.load_cache_modifiers
        if (
            self.supports_config_key("store_cache_modifiers")
            and self.store_cache_modifiers.length > 0
        ):
            fields["store_cache_modifiers"] = self.store_cache_modifiers
        if self.supports_config_key("num_threads"):
            fields["num_threads"] = self.num_threads
        if is_tileir:
            fields["num_ctas"] = self.backend_tunable_fragments["num_ctas"]
            fields["occupancy"] = self.backend_tunable_fragments["occupancy"]
        else:
            fields.update(self.backend_tunable_fragments)
        if self.has_pallas_inner_loops:
            choices = AUTOTUNED_PALLAS_LOOP_TYPES
            if (
                self.pallas_indirect_dma_requires_fori
                and self.pallas_indirect_access_modes == ("dma",)
            ):
                choices = ("fori_loop",)
            elif self.has_symbolic_or_data_dependent_bounds:
                # Exclude "unroll" (uses Python range(), can't handle traced
                # bounds) and put "fori_loop" first: it handles both DMA-aligned
                # and unaligned inner blocks, while "emit_pipeline" fails on
                # unaligned dims.
                # TODO(thcmbs): Also exclude "emit_pipeline" when has_pallas_dma_unaligned
                # is set, to avoid wasted autotuning effort. See PR #1969 review discussion.
                choices = ("fori_loop", "emit_pipeline")
                if self.grid_block_ids:
                    # Owner hl.grid + jagged bounds may be compactable. The full
                    # detector remains authoritative, so residual mismatches are
                    # autotuner-skippable InvalidConfig candidates.
                    choices = (*choices, "unroll")
                    fields["pallas_worklist_grouping"] = EnumFragment(
                        choices=VALID_PALLAS_WORKLIST_GROUPINGS
                    )
            fields["pallas_loop_type"] = EnumFragment(choices=choices)
            if self.supports_config_key("pallas_emit_pipeline_group_size"):
                fields["pallas_emit_pipeline_group_size"] = PowerOfTwoFragment(1, 16, 1)
            if self.supports_config_key("pallas_use_low_level_scheduler"):
                fields["pallas_use_low_level_scheduler"] = BooleanFragment()
            if self.supports_config_key("pallas_pre_broadcast"):
                fields["pallas_pre_broadcast"] = BooleanFragment()
            if (
                self.supports_config_key("pallas_fold_dot_lhs_cast")
                and self.pallas_fold_dot_lhs_cast_search_enabled
            ):
                fields["pallas_fold_dot_lhs_cast"] = BooleanFragment()
        # Only include maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            fields["maxnreg"] = EnumFragment(AUTOTUNED_MAXNREG)
        if self.epilogue_subtile_autotune_choices is not None:
            fields["epilogue_subtile"] = EnumFragment(
                choices=self.epilogue_subtile_autotune_choices
            )
        # Add tunable parameters
        fields.update(self.user_defined_tunables)
        return fields

    def structural_fingerprint(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> tuple[tuple[str | int, ...], ...]:
        """Return a hashable structural description of this ConfigSpec's search space.

        Captures field names, sequence lengths, per-item block_ids lengths
        (for PermutationFragment), ListOf inner lengths, and optional ACF slot
        presence.  Two ConfigSpecs with the same fingerprint can safely exchange
        FlatConfig values.
        """
        result: list[tuple[str | int, ...]] = [
            (key, *field.fingerprint()) for key, field in self._flat_fields().items()
        ]
        acf_fragment = self._advanced_controls_file_fragment(advanced_controls_files)
        if acf_fragment is not None:
            result.append(
                (
                    "advanced_controls_file",
                    *cast("tuple[str, ...]", acf_fragment.choices),
                )
            )
        return tuple(result)

    def structural_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        """Return a hex-digest SHA-256 hash of the structural fingerprint."""
        return hashlib.sha256(
            repr(
                self.structural_fingerprint(
                    advanced_controls_files=advanced_controls_files
                )
            ).encode("utf-8")
        ).hexdigest()

    def cache_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        """Return the ConfigSpec identity used by persistent autotune caches.

        Compiler-owned defaults and seeds affect which configs are tried first,
        but are intentionally absent from :meth:`structural_fingerprint`: two
        shapes with different promoted seeds can still share the same flat
        layout during multi-shape autotuning.  Persistent cache reuse is
        different: a result selected under an old compiler seed policy must not
        bypass a newly promoted policy, so include the ordered compiler configs
        in that identity.

        Target lowering policy also affects generated code even when its
        promoted seed is unchanged, so include its stable identity.  Generic
        targets with no compiler default or seed retain the historical
        structural hash verbatim.
        """
        structural_hash = self.structural_fingerprint_hash(
            advanced_controls_files=advanced_controls_files
        )
        legacy_hash = self._cache_fingerprint_from_structural_hash(structural_hash)
        policy = coverage_policy(self.compiler_coverage_groups)
        if policy is None:
            return legacy_hash
        return hashlib.sha256(repr((legacy_hash, policy)).encode("utf-8")).hexdigest()

    def projected_cache_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        """Identity of the old layout, used only by initial warm-base decoding."""
        owned = {group.key for group in self.compiler_coverage_groups}
        fingerprint = tuple(
            row
            for row in self.structural_fingerprint(
                advanced_controls_files=advanced_controls_files
            )
            if row[0] not in owned
        )
        structural_hash = hashlib.sha256(repr(fingerprint).encode("utf-8")).hexdigest()
        return self._cache_fingerprint_from_structural_hash(structural_hash)

    def _cache_fingerprint_from_structural_hash(self, structural_hash: str) -> str:
        target_policy_identity: object | None = None
        if self.backend_name == "cute" and self.cute_flash_search_enabled:
            from .._compiler.cute.flash_policy import flash_target_policy_cache_identity

            target_policy_identity = flash_target_policy_cache_identity(
                self.target_device_capability,
                head_dim=self._cute_flash_head_dim,
                torch_dtype=str(self._cute_flash_dtype).removeprefix("torch."),
                num_kv=self._cute_flash_num_kv,
                is_causal=self._cute_flash_is_causal,
            )
        if (
            self.compiler_default_config is None
            and not self.compiler_seed_configs
            and target_policy_identity is None
        ):
            return structural_hash

        def canonical_config(config: helion.Config | None) -> str | None:
            if config is None:
                return None
            return json.dumps(
                config.config,
                sort_keys=True,
                separators=(",", ":"),
            )

        compiler_config_identity = (
            canonical_config(self.compiler_default_config),
            tuple(canonical_config(config) for config in self.compiler_seed_configs),
        )
        return hashlib.sha256(
            repr(
                (
                    structural_hash,
                    compiler_config_identity,
                    target_policy_identity,
                )
            ).encode("utf-8")
        ).hexdigest()

    def _advanced_controls_file_fragment(
        self, advanced_controls_files: list[str] | None
    ) -> EnumFragment | None:
        # Empty list means no autotuning with ACFs.
        if not advanced_controls_files:
            return None
        files = advanced_controls_files
        # When non-empty list is provided then ensure default -O3 is considered.
        if "" not in files:
            files = [*files, ""]
        return EnumFragment(tuple(files))

    def flat_key_layout(
        self,
        *,
        advanced_controls_files: list[str] | None = None,
    ) -> list[tuple[str, int, bool]]:
        """Return (key_name, num_flat_entries, is_sequence) for each field.

        is_sequence is True for BlockIdSequence keys whose list values
        are spread across individual flat slots.
        """
        fields = self._flat_fields()
        result = [(key, *field._flat_key_info()) for key, field in fields.items()]
        if self._advanced_controls_file_fragment(advanced_controls_files) is not None:
            result.append(("advanced_controls_file", 1, False))
        return result

    def _flat_key_layout_with_flash_family(
        self,
        *,
        advanced_controls_files: list[str] | None,
        flash_pipeline_family: str,
    ) -> list[tuple[str, int, bool]]:
        fields = self._flat_fields_with_flash_family(flash_pipeline_family)
        result = [(key, *field._flat_key_info()) for key, field in fields.items()]
        if self._advanced_controls_file_fragment(advanced_controls_files) is not None:
            result.append(("advanced_controls_file", 1, False))
        return result

    def flat_config(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        *,
        advanced_controls_files: list[str] | None = None,
    ) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        return self._flat_config_from_fields(
            fn,
            self._flat_fields(),
            advanced_controls_files=advanced_controls_files,
        )

    def _flat_config_with_flash_family(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        *,
        advanced_controls_files: list[str] | None,
        flash_pipeline_family: str,
    ) -> helion.Config:
        return self._flat_config_from_fields(
            fn,
            self._flat_fields_with_flash_family(flash_pipeline_family),
            advanced_controls_files=advanced_controls_files,
        )

    def _flat_config_from_fields(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        fields: Mapping[str, BlockIdSequence[Any] | ConfigSpecFragment],
        *,
        advanced_controls_files: list[str] | None,
        _fix_invalid: bool = True,
    ) -> helion.Config:
        config: dict[str, Any] = {}
        for key, field in fields.items():
            config[key] = field._flat_config(self, fn)

        # None is the flat representation of these absent grouped overrides.
        # Preserve that absence before strict validation, which intentionally
        # rejects an explicitly supplied None for either option.
        for key in (
            "tcgen05_grouped_mode",
            "tcgen05_grouped_worklist_source_m_tile",
        ):
            if config.get(key) is None:
                config.pop(key, None)

        for name in (
            "loop_orders",
            "num_threads",
            "flatten_loops",
            "reduction_loops",
            "l2_groupings",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
            "indexing",
            "atomic_indexing",
            "pallas_load_buffer_count",
        ):
            if not config.get(name):
                config.pop(name, None)
        acf_fragment = self._advanced_controls_file_fragment(advanced_controls_files)
        if acf_fragment is not None:
            config["advanced_controls_file"] = fn(acf_fragment)
        self.normalize(config, _fix_invalid=_fix_invalid)
        return helion.Config(**config)


class LoopOrderSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PermutationFragment:
        return PermutationFragment(len(self.block_ids))

    def _normalize(self, name: str, value: object) -> list[int]:
        if type(value) is not list:
            if not isinstance(value, tuple):
                raise InvalidConfig(f"{name} must be a list, got {value!r}")
            value = [*value]
        length = len(self.block_ids)
        if len(value) != length:
            raise InvalidConfig(f"{name} must be length {length}, got {len(value)}")
        if {*value} != {*range(length)}:
            raise InvalidConfig(f"{name} must be permutation, got {value!r}")
        return value

    def _fill_missing(self) -> list[int]:
        """Provide a value when not provided by the user."""
        return [*range(len(self.block_ids))]


class L2GroupingSpec(_PowerOfTwoBlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PowerOfTwoFragment:
        return PowerOfTwoFragment(1, 64, 1)

    def _fill_missing(self) -> int:
        return 1


class BlockSizeSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
        min_size: int = 1,
        max_size: int | None = None,
        bounded_by_block_id: int | None = None,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

        # TODO(shunting): it's a bit conservative since not every block is split
        # for different ranks.
        bounded_hint = size_hint
        if dist.is_initialized():
            world_size = dist.get_world_size()
            bounded_hint = bounded_hint // world_size

        bounded_hint = max(bounded_hint, 1)
        self.min_size: int = min_size
        self.autotuner_min: int = min_size
        # Largest power-of-two block that fits inside the dimension. allow_overshoot
        # may raise max_size above this for matmul dims, but the default block size
        # stays clamped to dim_max_size (see _fragment).
        self.dim_max_size: int = (
            next_power_of_2(bounded_hint) if max_size is None else max_size
        )
        self.max_size: int = self.dim_max_size
        # A search surface may pin the default (non-autotuned) block size while
        # widening the autotuner's range (see ``_fragment``).
        self.default_size: int | None = None
        # Outer block_id whose tile extent caps this block's size in normalize().
        self.bounded_by_block_id: int | None = bounded_by_block_id
        if self.max_size < self.min_size:
            self.max_size = self.min_size
        assert self.min_size <= self.max_size

    def __repr__(self) -> str:
        fields: list[str] = []
        for field, default in (
            ("block_id", None),
            ("size_hint", None),
            ("min_size", 1),
            ("max_size", next_power_of_2(self.size_hint)),
            ("bounded_by_block_id", None),
        ):
            value = getattr(self, field)
            if value != default:
                fields.append(f"{field}={value!r}")
        return f"BlockSizeSpec({', '.join(fields)})"

    def _normalize(self, name: str, value: object) -> int | None:
        result = super()._normalize(name, value)
        if isinstance(result, int) and result < self.min_size:
            result = self.min_size
        return result

    def update_min(self, value: int) -> None:
        self.min_size = assert_integer_power_of_two(max(value, self.min_size))
        if self.max_size < self.min_size:
            self.max_size = self.min_size

    def update_max(self, value: int) -> None:
        clamped = max(value, 1)
        self.max_size = assert_integer_power_of_two(min(clamped, self.max_size))

    def allow_overshoot(self, ceiling: int) -> None:
        """Raise the autotuner search ceiling above the dimension size.

        Used for matmul tile dimensions: a block larger than a small dimension
        (with the extra rows/cols masked off) can map to a more efficient MMA
        tile and run faster. Only the search ceiling grows; the default block
        size stays clamped to the dimension (see _fragment). Dimensions bounded
        by an outer tile extent are left untouched.
        """
        if self.bounded_by_block_id is not None:
            return
        self.max_size = max(self.max_size, next_power_of_2(max(ceiling, 1)))

    def update_hint(self, value: int) -> None:
        self.size_hint = value
        self.update_max(next_power_of_2(max(value, 1)))

    def _fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        total_ndim = len(base.block_sizes)
        reduction_numel = _product(
            [next_power_of_2(spec.size_hint) for spec in base.reduction_loops]
        )
        if total_ndim <= 2 and reduction_numel <= 128:
            default = 32
        elif total_ndim >= 3 and reduction_numel > 1:
            # With 3+ tiled dimensions and a non-trivial reduction/full-slice
            # dimension, the total tensor numel (default^total_ndim *
            # reduction_numel) grows quickly and can cause Triton JIT
            # compilation to hang or exceed shared memory limits.
            # Compute a default that keeps total numel <= 32768 (safe for
            # 64KB shared memory with 2-byte elements like bf16).
            target = 32768
            per_dim = int((target / reduction_numel) ** (1.0 / total_ndim))
            default = max(1, 1 << (per_dim.bit_length() - 1)) if per_dim >= 1 else 1
        elif reduction_numel <= 256:
            default = 16
        else:
            default = 1
        low = min(max(self.min_size, self.autotuner_min), self.max_size)
        # Clamp the default within the dimension so allow_overshoot only widens
        # the autotuner *search*, never the default (non-autotuned) block size.
        # Needed for matmul dims smaller than the heuristic default (e.g. M<16),
        # where the default would otherwise overshoot to a masked tile.
        default = min(default, self.dim_max_size)
        if self.default_size is not None:
            default = self.default_size
        if any(
            self.block_id in group for group in base.cute_pointwise_region_grid_groups
        ):
            # Widen independent pointwise searches without changing the old
            # effective default, which the fragment clamps to its soft floor.
            # Shared axes and hard layout/alignment minima remain unchanged.
            default = max(min(default, self.max_size), low)
            low = min(self.min_size, self.max_size)
        return BlockSizeFragment(
            low,
            self.max_size,
            default,
        )


class NumThreadsSpec(_PowerOfTwoBlockIdItem):
    def __init__(self, *, block_id: int, size_hint: int) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _normalize(self, name: str, value: object) -> int | None:
        # 0 is a valid sentinel meaning "use block_size as thread count"
        if value == 0:
            return 0
        return super()._normalize(name, value)

    def _fragment(self, base: ConfigSpec) -> NumThreadsFragment | EnumFragment:
        if (
            base.cute_tcgen05_search_enabled
            and base.cute_pointwise_region_block_ids
            and self.block_id not in base.cute_pointwise_region_block_ids
        ):
            return EnumFragment((0,))
        max_threads = min(max(self.size_hint, 1), 1024)
        default = next_power_of_2(max_threads)
        return NumThreadsFragment(default)

    def _fill_missing(self) -> int:
        return 0


class FlattenLoopSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}") from None
        return value

    def _fill_missing(self) -> bool:
        return False


class ReductionLoopSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _flat_fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        # Shared by both directions:
        # - unflatten: flat integer -> Config value via _flat_config()
        # - flatten: Config value -> flat integer via _encode_flat_value()
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(max(low, self.size_hint))
        default = min(high, 4096)
        # Cap default at the backend's max reduction loop so that
        # large reductions default to looped rather than persistent.
        if base.max_reduction_loop is not None:
            force_threshold = base.reduction_loop_force_threshold
            if force_threshold is not None and self.size_hint > force_threshold:
                default = min(default, base.max_reduction_loop)
        return BlockSizeFragment(low, high, default)

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> int | None:
        fragment = self._flat_fragment(base)
        low = fragment.low
        high = fragment.high
        value = fn(fragment)
        assert isinstance(value, int)
        if not (low <= value <= high):
            raise InvalidConfig(
                f"Invalid value for reduction loop {low} <= {value} <= {high}"
            )
        if value >= self.size_hint:
            return None  # max size becomes persistent reduction
        return value

    def _encode_flat_value(self, base: ConfigSpec, value: object) -> object:
        # Encode None ("persistent reduction") so the inverse ``_flat_config``
        # decodes it back to None. ``_flat_config`` returns None for any value
        # >= size_hint, so the encoding must also be >= size_hint: use the
        # fragment's ``high`` (always >= size_hint). The fragment *default* is
        # capped at max_reduction_loop and can fall below size_hint, which would
        # round-trip None into a slow looped config (e.g. size_hint=32000 ->
        # default 4096 -> reduction_loops=[4096]).
        if value is None:
            return self._flat_fragment(base).high
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        normalized = super()._normalize(name, value)
        # A looped chunk of 1 is degenerate: "hold the whole axis" is encoded as
        # ``None`` (persistent), not 1, and ``LoopedReductionStrategy`` rejects a
        # block size <= 1.  The autotuner search never proposes < 8 (its fragment
        # ``low`` is 8), but the reduction seed's byte budget can collapse the chunk
        # to 1 on a reduction co-resident with a wide feature.  Floor a stray 1 up to
        # that same search floor of 8; for a small extent the ``>= size_hint`` rule
        # below then collapses it to persistent ``None``, and 1 is the only
        # power-of-two chunk that can hit this (so legal chunks 2, 4, 8, ... are
        # left byte-identical).
        if isinstance(normalized, int) and normalized < 2:
            normalized = 8
        # A looped reduction whose chunk equals or exceeds the reduction
        # extent has only one iteration — it is semantically identical to a
        # persistent reduction, but the looped codegen path occasionally
        # produces subtly different results on the CuTe backend (e.g. when a
        # multi-pass kernel like layer_norm reuses the loaded inputs across
        # two reductions).  Collapsing to ``None`` here matches the
        # ``_flat_config`` behaviour and keeps the persistent/loop choice in
        # sync regardless of how the value was generated.
        if isinstance(normalized, int) and normalized >= self.size_hint:
            return None
        return normalized

    def _fill_missing(self) -> None:
        return None


_CUTE_VECTOR_WIDTH_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
_CUTE_LANE_LAYOUT_CHOICES: tuple[str, ...] = ("blocked", "strided")
_CUTE_REDUCTION_RELOAD_CHOICES: tuple[str, ...] = ("auto", "register", "gmem")
# Manual unroll of the row lane loop of a sunk vector nest (``cute_vloop_sink``).
_CUTE_LANE_UNROLL_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16)


def _as_sequence(value: object) -> list[object]:
    """A config entry as a list: scalars name a single block, ``None`` is empty."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class CuteReductionReloadSpec(_BlockIdItem):
    """Where a multi-sweep reduction keeps the values it re-reads.

    Rolled or persistent LayerNorm/RMSNorm-style kernels sweep the same input
    several times (reduce, then consume).  ``"register"`` caches the earliest
    sweep's loads in a per-thread register fragment (fastest when the
    per-thread slice is small; spills to local memory when it is not).
    ``"gmem"`` re-loads from global memory on later sweeps (the row is usually
    still resident in L2, and no registers are burned).  ``"auto"`` (default)
    keeps the legacy size heuristic in ``fuse_two_pass_loads``.
    """

    def __init__(self, *, block_id: int) -> None:
        super().__init__([block_id])

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment(choices=_CUTE_REDUCTION_RELOAD_CHOICES)

    def _normalize(self, name: str, value: object) -> str:
        if value not in _CUTE_REDUCTION_RELOAD_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {_CUTE_REDUCTION_RELOAD_CHOICES}, got {value!r}"
            )
        return str(value)

    def _fill_missing(self) -> str:
        return "auto"


class CuteLaneLayoutSpec(_BlockIdItem):
    """Per-block per-thread element assignment for CuTe lane loops.

    ``"blocked"``: thread ``t`` owns ``EPT`` contiguous elements
    (``base = offset + t*EPT + lane*V``).  Consecutive threads in a warp
    touch addresses ``EPT*elem_size`` bytes apart, so a warp's loads/stores
    of one lane iter span many cache lines (poor per-instruction
    coalescing; only acceptable when the lane loop has a single iteration,
    where both layouts coincide).

    ``"strided"``: thread ``t`` owns V-wide chunks strided by the thread
    count (``base = offset + (lane*NT + t)*V``).  Consecutive threads touch
    consecutive V-chunks, so each warp instruction is fully coalesced.
    """

    def __init__(
        self,
        *,
        block_id: int,
    ) -> None:
        super().__init__([block_id])

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        if (
            base.cute_tcgen05_search_enabled
            and base.cute_pointwise_region_block_ids
            and self.block_id not in base.cute_pointwise_region_block_ids
        ):
            return EnumFragment(("blocked",))
        return EnumFragment(choices=_CUTE_LANE_LAYOUT_CHOICES)

    def _normalize(self, name: str, value: object) -> str:
        if value not in _CUTE_LANE_LAYOUT_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {_CUTE_LANE_LAYOUT_CHOICES}, got {value!r}"
            )
        return str(value)

    def _fill_missing(self) -> str:
        return "blocked"


class CuteVectorWidthSpec(_BlockIdItem):
    """Per-reduction-block vector load width for the CuTe backend.

    V=1 disables vectorization (scalar loads). V=2/4/8 emits
    ``cute.arch.load(..., ir.VectorType.get([V], elem_dtype.mlir_type))``
    for the inner reduction load, lowering to LDG.64/LDG.128.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        if (
            base.cute_tcgen05_search_enabled
            and base.cute_pointwise_region_block_ids
            and self.block_id not in base.cute_pointwise_region_block_ids
        ):
            return EnumFragment((1,))
        return EnumFragment(choices=_CUTE_VECTOR_WIDTH_CHOICES)

    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        if value not in _CUTE_VECTOR_WIDTH_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {_CUTE_VECTOR_WIDTH_CHOICES}, got {value!r}"
            )
        return value

    def _fill_missing(self) -> int:
        return 1


class _OptionalIntSpec(_BlockIdItem):
    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        return value

    def _fill_missing(self) -> int:
        """Provide a value when not provided by the user."""
        return 0


class _OptionalBoolSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment((None, False, True))

    def _normalize(self, name: str, value: object) -> bool | None:
        if value is not None and not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean or None, got {value!r}")
        return value

    def _fill_missing(self) -> None:
        """Provide a value when not provided by the user."""
        return None


class RangeUnrollFactorSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeWarpSpecializeSpec(_OptionalBoolSpec):
    pass


class RangeNumStagesSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeMultiBufferSpec(_OptionalBoolSpec):
    pass


class RangeFlattenSpec(_OptionalBoolSpec):
    pass


class StaticRangeSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
        return value

    def _fill_missing(self) -> bool:
        """Provide a value when not provided by the user."""
        return False


def _product(seq: Sequence[int]) -> int:
    """Return the product of the elements in the sequence."""
    return functools.reduce(operator.mul, seq, 1)
