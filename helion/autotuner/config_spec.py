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
import warnings

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
from .._compat import warps_to_threads
from .._compiler.backend_registry import all_search_surfaces
from .._compiler.backend_registry import registry_generation
from .._compiler.cute.autotune.config_keys import CUTE_TOPK_CHOICES
from .._compiler.cute.flash_facts import FlashEnv
from .._compiler.cute.flash_facts import FlashSurfaceFacts
from .._utils import indexing_uses_tensor_descriptor
from .._utils import prev_power_of_2
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
    from .._compiler.cute.autotune.heuristics.cute_fragment_common import (
        FragmentRootRequirement,
    )
    from .._compiler.cute.cute_flash import FlashAttentionConfig
    from .._compiler.cute.cute_flash_bwd_row_mma import FlashBwdRowFacts
    from .._compiler.cute.cute_mma_dag import CuteMmaDagFacts
    from .._compiler.cute.cute_register_subblocks import RegisterSubblockFacts
    from .._compiler.cute.loop_nesting import TileLoopPath
    from .._compiler.cute.selection.radix_selection import RadixSelectionGraph
    from .._compiler.cute.split_k_cluster import ClusterKFacts
    from .._compiler.cute.tcgen05_config import CuteTcgen05Config
    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration
    from .search_surface import BackendSearchSurface

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
    }
)


def _get_backend_tunable_keys() -> frozenset[str]:
    """Get all backend tunable keys, including FB-private ones if available."""
    try:
        from ..fb.mtia_tunables import MTIA_TUNABLES  # pyrefly: ignore [missing-import]

        return _BASE_BACKEND_TUNABLE_KEYS | frozenset(MTIA_TUNABLES)
    except ImportError:
        return _BASE_BACKEND_TUNABLE_KEYS


BACKEND_TUNABLE_KEYS: frozenset[str] = _get_backend_tunable_keys()
# Generic keys whose support is gated per backend (the Triton host descriptor
# and cache-modifier knobs, the Pallas loop knobs, the AMD XCD remap and the
# cross-loop pipeline) rather than declared by one backend's surface.
_BACKEND_GATED_GENERIC_KEYS: frozenset[str] = frozenset(
    {
        "cross_loop_pipeline",
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
# The generic keys every backend's ``ConfigSpec`` understands.
GENERIC_KEYS: frozenset[str] = frozenset(
    {
        "block_sizes",
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
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        "reduction_fusion",
        "indexing",
        "atomic_indexing",
        "load_eviction_policies",
        "pallas_internal_scratch",
        "advanced_controls_file",
        "epilogue_subtile",
    }
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


# For tileir backend or AMD ROCM, eviction policies are not supported.
# Keep this uncached: some tests patch the AMD capability helper, and caching
# only on backend name can poison later Triton ConfigSpec construction inside
# the same worker process.
def get_valid_eviction_policies(backend: Backend) -> tuple[str, ...]:
    if backend.name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    return backend.config_spec_extension().eviction_policies()


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
        # The backend's contribution to this spec's search surface (keys,
        # fields, normalization phases, seeds); stateless, so one object per
        # backend serves every spec.
        self.search_surface: BackendSearchSurface = backend.config_spec_extension()
        self.device = device
        # ``VALID_KEYS`` split into (supported, unsupported), keyed on the
        # backend and target it was computed for; see ``_config_key_support``.
        self._config_key_support_cache: (
            tuple[tuple[object, ...], frozenset[str], frozenset[str]] | None
        ) = None
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
        # Whether fp32 matmuls stay exact (dot_precision ieee / tf32x3): the
        # SIMT GEMM knob's precondition, recorded by DeviceIR.
        self.cute_fp32_dot_exact: bool = False
        self.user_defined_tunables = (
            {} if user_defined_tunables is None else dict(user_defined_tunables)
        )
        # Bound kernels pass an explicit target capability; the backend's
        # surface picks the default a direct spec validates against.
        if target_device_capability is _TARGET_DEVICE_CAPABILITY_UNSET:
            self.target_device_capability: tuple[int, int] | None = (
                self.search_surface.default_target_device_capability()
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
        self.reduction_fusion_available: bool = False
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
        # Set by DeviceIR analysis: an fp32 quotient whose divisor is a
        # scalar or a broadcast row value (``hoist_loop_invariant_recip``'s
        # exact hoist can apply), so ``cute_exact_div_form`` is tunable.
        self.cute_exact_division_sites = False
        self.cute_scaled_mma_available = False
        self.cute_mxfp8_fused_scales_available = False
        self.cute_mxfp8_sf_tma_available = False
        self.cute_mxfp8_problem_shape: tuple[int, int, int] | None = None
        self.cute_mxfp8_output_element_size = 0
        self.cute_mxfp8_tma_output_available = False
        self.cute_mxfp8_mma_available = False
        self.cute_mxfp8_partition_key: str | None = None
        self.cute_proven_bounds_enabled = False
        self.cute_rng_packet_enabled = False
        self.cute_warp_aggregated_atomics_enabled = False
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
        # All independent vetoes, including callers without a diagnostic reason.
        # A backend may restore a restriction it owns only if no other veto exists.
        self.disallowed_pid_type_causes: dict[str, set[str | None]] = {}
        # (feature, reason) pairs for every non-pid_type search-space restriction
        # (currently the tcgen05 narrowing applied per matmul), for search-space
        # logging. De-duplicated; ordered by first occurrence.
        self.restriction_reasons: list[tuple[str, str]] = []
        self.max_num_sm_multiplier: int = MAX_NUM_SM_MULTIPLIER
        self.default_num_sm_multiplier: int = backend.default_num_sm_multiplier()
        self.grid_block_ids: list[int] = []
        # Ordered root grids and nested-graph ownership, independent of any
        # specialized matmul/reduction policy.
        self.kernel_grid_fact: KernelGridFact | None = None
        self.tensor_numel_constraints: list[TensorNumelConstraint] = []
        self.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(backend)),
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
        # Cleared by the CuTe DeviceIR pass when no contraction can reach the
        # collective marker; the ``cute_collective_*`` fragments then stay out
        # of the search surface while normalization keeps their defaults.
        self.cute_collective_knobs_enabled: bool = True
        self.has_pallas_inner_loops: bool = False
        # Set by the Pallas graph lowering when an inner-loop dot can consume
        # an f32 producer directly instead of its explicit bf16/f16 cast.
        self.pallas_fold_dot_lhs_cast_search_enabled: bool = False
        self.pallas_indirect_access_modes: tuple[str, ...] = ()
        self.pallas_indirect_dma_requires_fori: bool = False
        self.has_symbolic_or_data_dependent_bounds: bool = False
        # Populated only after DeviceIR proves that this kernel contains an
        # implicit cross-root dependency the backend can lower (CuTe offers
        # the barrier pipeline only).
        self.cross_loop_pipeline: EnumFragment | None = None
        # Whether the kernel runs several phases in one persistent launch
        # (``hl.barrier()`` or an implicit cross-root dependency).
        self.multi_phase: bool = False
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
        # Enabled only when the warp-MMA state recurrence owner
        # (``cute_mma_dag``) admits the kernel: the mode, its warp count and
        # its operand stage depth.  ``cute_mma_dag_facts`` carries the
        # geometry ``normalize`` validates a tile / warp / stage choice
        # against without the graphs.
        self.cute_mma_dag: EnumFragment | None = None
        self.cute_mma_dag_warps: EnumFragment | None = None
        self.cute_mma_dag_stages: EnumFragment | None = None
        self.cute_mma_dag_store: EnumFragment | None = None
        self.cute_mma_dag_exchange: EnumFragment | None = None
        self.cute_mma_dag_pass_warps: EnumFragment | None = None
        self.cute_mma_dag_loads: EnumFragment | None = None
        self.cute_mma_dag_aux: EnumFragment | None = None
        # How a dot operand that is a pointwise chain over loaded tiles
        # reaches its tile (``cute_mma_dag_derived``: the exchange of a
        # derived fragment, or a packet pass from the staged sources).
        self.cute_mma_dag_derived: EnumFragment | None = None
        # The tensor-memory region programs' epilogue layout
        # (``cute_mma_dag_epilogue``: one row per thread on four warps, or
        # the row's columns split over a warp pair on eight).
        self.cute_mma_dag_epilogue: EnumFragment | None = None
        # The region programs' ring-loop schedule (``cute_mma_dag_lookahead``:
        # the one dot of the loop issued one iteration ahead of its epilogue).
        self.cute_mma_dag_lookahead: EnumFragment | None = None
        # The tcgen05 mode's persistent launch (CTAs per SM); None where the
        # mode or its one-tile-per-CTA region programs leave it no subject.
        self.cute_mma_dag_persistent: EnumFragment | None = None
        # The tcgen05 region renderer's thread-block cluster over a
        # full-slice fragment axis; None where the program has no such axis.
        self.cute_mma_dag_cluster: EnumFragment | None = None
        # How the split's CTAs combine their row reductions: one cluster's
        # distributed shared memory, or the fixup grid's global workspace.
        self.cute_mma_dag_split: EnumFragment | None = None
        self.cute_mma_dag_facts: CuteMmaDagFacts | None = None
        # Enabled only when the register sub-block family
        # (``cute_register_subblocks``, cute_register_subblocks.py) admits
        # the root: the family switch and its knobs; the facts carry the
        # register estimates ``normalize`` validates a knob choice against.
        self.cute_register_subblocks: EnumFragment | None = None
        self.cute_register_subblocks_warps: EnumFragment | None = None
        self.cute_register_subblocks_prefetch: EnumFragment | None = None
        self.cute_register_subblocks_mask: EnumFragment | None = None
        self.cute_register_subblocks_colside: EnumFragment | None = None
        self.cute_register_subblocks_roll: EnumFragment | None = None
        self.cute_register_subblocks_facts: RegisterSubblockFacts | None = None
        # Enabled only when the exact five-factor BT16 chunk-prepare carrier is
        # detected. Choice order defines the default and ranked seed order.
        self.cute_chunk_prepare_schedule: EnumFragment | None = None
        # Enabled only after a generic matcher proves a compatible affine scan.
        # The first choice is the semantic-neutral ordinary lowering.
        self.cute_affine_scan_schedule: EnumFragment | None = None
        # Enabled only when the whole root is a supported top-k load/store
        # dataflow. Its emitter owns the row geometry and vector layout.
        self.cute_topk_search_enabled = False
        self.cute_topk_coarse_keys_available = False
        self.cute_topk_coarse_keys_search_enabled = False
        self.cute_topk_key_recovery_search_enabled = False
        self.cute_topk_choices = dict(CUTE_TOPK_CHOICES)
        self.cute_radix_selection_graph: RadixSelectionGraph | None = None
        # The backend's per-spec state (``BackendSearchSurface.new_spec_state``):
        # CuTe stores its tcgen05 GEMM facade here (read through ``tcgen05``);
        # every other backend stores ``None`` and never reads it.
        self._cute_tcgen05_config: CuteTcgen05Config = cast(
            "CuteTcgen05Config", self.search_surface.new_spec_state(self)
        )
        self.cute_host_paired_sum_available: bool = False
        # A separately launched, proved pointwise producer can share one
        # Config with a native GEMM. Keep its ordinary SIMT layout knobs in
        # the native search schema; MMA-owned axes retain their auto layout.
        self.cute_pointwise_region_block_ids: frozenset[int] = frozenset()
        # Root-grid axes fixed at one element per tile -> static tile count
        # (``cute_tiles_per_block`` packs several of their tiles per CTA).
        self.cute_tile_pack_counts: dict[int, int] = {}
        # Separately launched pointwise grids whose axes have one root owner.
        # Their search floors use the rank of that launch, independently of
        # any native MMA region sharing the complete kernel configuration.
        self.cute_pointwise_region_grid_groups: tuple[tuple[int, ...], ...] = ()
        self.cute_split_k_cluster_facts: ClusterKFacts | None = None
        self.cute_split_k_workspace_available: bool = False
        # ``cute_split_k_partitions`` values of a proved plain GEMM store
        # (``split_k_workspace.plain_store_partition_choices``); empty for
        # the atomic pattern, which partitions through its own tunable.
        self.cute_split_k_partition_choices: tuple[int, ...] = ()
        # ``cute_split_k_atomic_epilogue`` is offered when the proved split-K
        # reduction is an atomic add into an output kept in a dtype with a
        # packed atomic add (``split_k_workspace.atomic_epilogue_admissible``).
        self.cute_split_k_atomic_epilogue_available: bool = False
        self.cute_materialized_schedule_available: bool = False
        self.cute_materialized_schedule_search_enabled: bool = False
        self.cute_row_matrix_transport_available: bool = False
        self.cute_fragment_scan_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_scan_search_enabled = False
        self.cute_fragment_reduction_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_reduction_search_enabled = False
        self.cute_fp32_dot_root_ids: frozenset[int] = frozenset()
        self.cute_fp32_dot_search_enabled = False
        self.cute_fp32_dot_seed_worthy = False
        self.cute_fragment_thread_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_threads_search_enabled = False
        self.cute_fragment_producer_cache_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_producer_cache_search_enabled = False
        self.cute_fragment_register_load_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_warp_scan_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_warp_scan_requirements: tuple[
            FragmentRootRequirement, ...
        ] = ()
        self.cute_fragment_warp_scan_search_enabled = False
        self.cute_fragment_topk_network_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_topk_network_search_enabled = False
        self.cute_fragment_cta_topk_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_cta_topk_search_enabled = False
        self.cute_fragment_atomic_aggregation_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_integer_atomic_epochs_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_integer_atomic_epochs_search_enabled = False
        self.cute_fragment_atomic_aggregation_search_enabled = False
        self.cute_fragment_local_atomic_registers_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_local_atomic_registers_search_enabled = False
        self.cute_fragment_atomic_consumer_fusion_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_atomic_consumer_fusion_search_enabled = False
        self.cute_fragment_pure_producer_regions_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_pure_producer_regions_search_enabled = False
        self.cute_fragment_local_atomic_register_min_threads = 0
        self.cute_fragment_register_snapshots_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_register_snapshot_while_root_ids: frozenset[int] = (
            frozenset()
        )
        self.cute_fragment_register_snapshots_search_enabled = False
        self.cute_fragment_register_producer_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_register_producers_search_enabled = False
        self.cute_fragment_scan_exports_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_scan_exports_search_enabled = False
        self.cute_fragment_warp_producer_regions_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_warp_producer_regions_search_enabled = False
        self.cute_fragment_bounded_gather_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_bounded_gather_search_enabled = False
        self.cute_integer_loop_reduction_available = False
        self.cute_integer_loop_reduction_search_enabled = False
        self.cute_fragment_register_snapshot_min_threads = 0
        self.cute_fragment_packet_load_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_packet_loads_search_enabled = False
        self.cute_fragment_padded_rows_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_padded_rows_search_enabled = False
        self.cute_fragment_unrolled_loops_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_unrolled_loops_search_enabled = False
        self.cute_fragment_vector_io_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_vector_io_search_enabled = False
        self.cute_fragment_batched_loads_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_batched_loads_search_enabled = False
        self.cute_fragment_split_k_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_split_k_search_enabled = False
        self.cute_fragment_mma_rows_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_mma_rows_search_enabled = False
        self.cute_fragment_slab_loads_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_slab_loads_search_enabled = False
        self.cute_fragment_register_loads_search_enabled = False
        self.cute_fragment_warp_result_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_warp_results_search_enabled = False
        self.cute_fragment_published_scalar_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_published_scalars_search_enabled = False
        self.cute_fragment_skip_zero_atomics_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_skip_zero_atomics_search_enabled = False
        self.cute_fragment_private_scalar_loop_root_ids: frozenset[int] = frozenset()
        self.cute_fragment_private_scalar_loops_search_enabled = False
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
        # Knobs whose value picks a distinct kernel body (a pipeline family).
        # The full-effort final selection keeps the best shortlisted config
        # of every such family in contention for its steady windows
        # (``FinalSelectionConfig.family_finalists``); a surface that offers
        # several bodies names its family knobs when it is enabled.
        self.final_selection_family_keys: tuple[str, ...] = ()
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
        # Register-MMA row-program tiles of the gated surface
        # (``cute_flash_gated_row_mma``): the legal warp counts per (query
        # rows, KV tile) and the ring depths per (query rows, KV tile, warps).
        self._cute_flash_gated_row_warp_choices_by_tile: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self._cute_flash_gated_row_stage_choices: dict[
            tuple[int, int, int], tuple[int, ...]
        ] = {}
        # 16-row programs of the grid when known (the row seeds are capped).
        self._cute_flash_gated_row_seed_programs: int | None = None
        self._cute_flash_head_dim: int | None = None
        # V/O head dim when it differs from the Q/K head dim (MLA prefill).
        self._cute_flash_head_dim_v: int | None = None
        self._cute_flash_num_kv: int | None = None
        self._cute_flash_num_bh: int | None = None
        # Grouped-query divisor (``heads_q // heads_kv``; 1 for plain rows).
        self._cute_flash_kv_group: int = 1
        # SM count of the bound device (0 when unknown); small grids seed the
        # 64-row flash tile from it.
        self._cute_flash_device_sm_count: int = 0
        self._cute_flash_tensor_4d_heads: int | None = None
        self._cute_flash_dtype: torch.dtype = torch.float16
        self._cute_flash_is_causal: bool = False
        self._cute_flash_has_kv_tile_pruning: bool = False
        self._cute_flash_requires_ws_overlap: bool = False
        self._cute_flash_has_list_walk: bool = False
        self._cute_flash_small_biased_candidate: bool = False
        self._cute_flash_standard_dense_output: bool = False
        self._cute_flash_dense_lse_output: bool = False
        self._cute_flash_standard_causal_output: bool = False
        self._cute_flash_tmem_rowmax_compatible: bool = False
        self._cute_flash_causal_resident_compatible: bool = False
        self._cute_flash_output_requires_tma: bool = False
        self._cute_flash_supports_tensor_4d_tma: bool = True
        self._cute_flash_has_row_epilogue: bool = False
        self._cute_flash_plain_row_body: bool = True
        self._cute_flash_has_score_modifiers: bool = False
        self._cute_flash_score_modifier_kinds: tuple[str, ...] = ()
        # Flex-envelope facts (``score_modifier_kinds == ("flex",)``): the
        # data-dependent BlockMask list walk (a third variable-trip-count
        # category next to causal / pruned-interval plans), the captured
        # region kinds and the separate Q / KV lengths. Recorded here for the
        # resolver / seed sites that key on them; no searchable field.
        self._cute_flash_has_list_walk: bool = False
        self._cute_flash_flex_has_score_region: bool = False
        self._cute_flash_flex_has_mask_region: bool = False
        self._cute_flash_flex_trivial_walk: bool = False
        self._cute_flash_q_seq: int | None = None
        self._cute_flash_kv_seq: int | None = None
        self._cute_flash_flex_aux_smem_candidates: tuple[int, ...] = ()
        self._cute_flash_flex_prefetch_regs: tuple[int, ...] = ()
        self._cute_flash_flex_coalesce_steps: tuple[int, ...] = ()
        self._cute_flash_flex_hook_ops: int = 0
        # The kernel's ``fast_math`` setting: the polynomial exp2 packets are
        # approximations and reach the flash surface only under it. Recorded
        # by ``enable_cute_flash_search`` and passed to every flash surface
        # call this spec makes (fragments, resolver, seeds, packet overrides).
        self._cute_flash_fast_math: bool = False
        self._cute_flash_block_size_targets: dict[int, int] = {}
        # The ``HELION_CUTE_FLASH_*`` knob channel captured when the flash
        # surface was enabled (bind time); every flash builder this spec calls
        # reads it instead of the process environment.
        self._cute_flash_env: FlashEnv | None = None
        # Memo for ``flash_fragments_from_facts``: the facts and the two
        # overrides are its complete input. The builder resolves dozens of
        # configs per call and normalization calls it for every candidate,
        # so this cache dominates coverage-design cost.
        self._cute_flash_fragments_cache: dict[
            tuple[FlashSurfaceFacts, str | None, str | None],
            dict[str, ConfigSpecFragment],
        ] = {}
        # Memoized ``flash_surface._resolve_cute_flash_pins`` results; reset by
        # ``enable_cute_flash_search``.
        self._cute_flash_pin_resolutions: dict[
            tuple[object, ...], FlashAttentionConfig
        ] = {}
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
        self._cute_flash_bwd_aux_smem_candidates: tuple[int, ...] = ()
        self._cute_flash_bwd_ds_tmem_allowed: bool = False
        self._cute_flash_bwd_stats_mma_allowed: bool = True
        self._cute_flash_bwd_s_first_allowed: bool = False
        self._cute_flash_bwd_kv_stage_allowed: bool = False
        self._cute_flash_bwd_has_mask: bool = True
        self._cute_flash_bwd_has_hook: bool = True
        self._cute_flash_bwd_additive_bias: bool = True
        self._cute_flash_bwd_block_size_targets: dict[int, int] = {}
        self._cute_flash_bwd_two_cta_allowed: bool = False
        self._cute_flash_bwd_persistent_allowed: bool = True
        self._cute_flash_bwd_table_step_choices: tuple[int, ...] = (0,)
        # The 2-CTA pair kernel's automatic step table (0: no pair kernel).
        self._cute_flash_bwd_two_cta_table_steps: int = 0
        # The FlexAttention backward envelope: opens the mask-form knob.
        self._cute_flash_bwd_flex: bool = False
        self._cute_flash_bwd_tile_groups: tuple[int, ...] = (0,)
        self._cute_flash_bwd_fast_math: bool = False
        # The dense backward's register-MMA row family: the plan facts that
        # legalize its knobs (None off the dense bwd surface).
        self._cute_flash_bwd_row_facts: FlashBwdRowFacts | None = None
        self.compiler_default_config: helion.Config | None = None
        self.compiler_seed_configs: list[helion.Config] = []
        # A compiler may request complete initial coverage of its normalized
        # seeds after the unchanged nominal and declared-coverage population.
        self.compiler_seed_overflow_requested: bool = False
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
        self.disallowed_pid_type_causes.setdefault(pid_type, set()).add(reason)
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

    def record_restriction(self, feature: str, reason: str | None) -> None:
        """Record a backend's search-space restriction (``restriction_reasons``).

        ``reason`` explains why ``feature`` was narrowed for this kernel; it is
        recorded once (first occurrence wins) and surfaced by the search-space
        logger, and logged live under verbose restriction logging.
        """
        _record_restriction(
            self.restriction_reasons, feature, reason, self.log_restrictions_verbose
        )

    @property
    def tcgen05(self) -> CuteTcgen05Config:
        """The CuTe tcgen05 GEMM facade (``_cute_tcgen05_config``).

        Built in ``__init__`` by ``BackendSearchSurface.new_spec_state``: one
        ``CuteTcgen05Config`` per CuTe spec.  Every other backend's spec stores
        ``None`` in the slot and nothing in core reads it (the per-block thread
        knobs ask ``BackendSearchSurface.pins_thread_knobs``), so the return
        type describes the CuTe case only.  The setter exists for tests that
        install a stand-in.
        """
        return self._cute_tcgen05_config

    @tcgen05.setter
    def tcgen05(self, value: CuteTcgen05Config) -> None:
        self._cute_tcgen05_config = value

    @property
    def flash_facts(self) -> FlashSurfaceFacts:
        """This spec's flash surface facts, built from its ``_cute_flash_*``
        attributes (written by ``enable_cute_flash_search``; tests adjust them
        directly) and the knob channel captured when the surface was enabled.
        """
        assert self._cute_flash_head_dim is not None
        return FlashSurfaceFacts(
            head_dim=self._cute_flash_head_dim,
            num_kv=self._cute_flash_num_kv,
            num_bh=self._cute_flash_num_bh,
            tensor_4d_heads=self._cute_flash_tensor_4d_heads,
            dtype=self._cute_flash_dtype,
            block_size_targets=self._cute_flash_block_size_targets,
            is_causal=self._cute_flash_is_causal,
            has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=self._cute_flash_requires_ws_overlap,
            small_biased_candidate=self._cute_flash_small_biased_candidate,
            standard_dense_output=self._cute_flash_standard_dense_output,
            dense_lse_output=self._cute_flash_dense_lse_output,
            standard_causal_output=self._cute_flash_standard_causal_output,
            tmem_rowmax_compatible=self._cute_flash_tmem_rowmax_compatible,
            causal_resident_compatible=self._cute_flash_causal_resident_compatible,
            output_requires_tma=self._cute_flash_output_requires_tma,
            supports_tensor_4d_tma=self._cute_flash_supports_tensor_4d_tma,
            has_row_epilogue=self._cute_flash_has_row_epilogue,
            plain_row_body=self._cute_flash_plain_row_body,
            has_score_modifiers=self._cute_flash_has_score_modifiers,
            score_modifier_kinds=self._cute_flash_score_modifier_kinds,
            device_sm_count=self._cute_flash_device_sm_count,
            head_dim_v=self._cute_flash_head_dim_v,
            kv_group=self._cute_flash_kv_group,
            fast_math=self._cute_flash_fast_math,
            has_list_walk=self._cute_flash_has_list_walk,
            flex_has_score_region=self._cute_flash_flex_has_score_region,
            flex_has_mask_region=self._cute_flash_flex_has_mask_region,
            flex_trivial_walk=self._cute_flash_flex_trivial_walk,
            q_seq=self._cute_flash_q_seq,
            kv_seq=self._cute_flash_kv_seq,
            flex_aux_smem_candidates=self._cute_flash_flex_aux_smem_candidates,
            flex_prefetch_regs=self._cute_flash_flex_prefetch_regs,
            flex_coalesce_steps=self._cute_flash_flex_coalesce_steps,
            flex_hook_ops=self._cute_flash_flex_hook_ops,
            env=self._cute_flash_env,
        )

    @property
    def flash_env(self) -> FlashEnv | None:
        """The knob channel the flash surface was bound under (None: unbound)."""
        return self._cute_flash_env

    def autotune_seed_configs(self) -> list[helion.Config]:
        """The backend's ordered compiler seeds for this spec's enabled families."""
        return self.search_surface.seed_configs(self)

    def _config_key_support(self) -> tuple[frozenset[str], frozenset[str]]:
        """``VALID_KEYS`` partitioned into (supported, unsupported) keys.

        ``normalize`` asks about every key of every config it sees, and the
        answer depends only on the backend and target.  Tests reassign those
        after construction, so the partition is cached against them.
        """
        state = (
            self.backend,
            self.backend_name,
            self.device,
            self.target_device_capability,
            registry_generation(),
        )
        cached = self._config_key_support_cache
        if cached is None or cached[0] != state:
            keys = valid_keys()
            supported = frozenset(
                key for key in keys if self._supports_config_key_uncached(key)
            )
            cached = (state, supported, keys - supported)
            self._config_key_support_cache = cached
        return cached[1], cached[2]

    def supports_config_key(self, key: str) -> bool:
        if key in valid_keys():
            return key in self._config_key_support()[0]
        return self._supports_config_key_uncached(key)

    def _supports_config_key_uncached(self, key: str) -> bool:
        # Reads only the attributes in ``_config_key_support``'s state tuple;
        # a new dependency here must join that tuple or the cache goes stale.
        if key == "reduction_fusion":
            return self.backend.supports_config_key(key)
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

    @property
    def cute_l2_evict_last_store_policy_supported(self) -> bool:
        """Whether the fixed store-policy descriptor is valid for this target."""

        return self.search_surface.l2_evict_last_store_policy_supported(self)

    @property
    def cute_materialized_schedule_choices(self) -> tuple[str, ...]:
        choices = ("off", "warp_rows2")
        if self.cute_row_matrix_transport_available:
            return (*choices, "warp_rows2_matrix")
        return choices

    @property
    def cute_simt_mma_enabled(self) -> bool:
        """Whether the exact fp32 SIMT GEMM (``cute_simt_mma``) has a target.

        fp32 contractions under ``dot_precision="tf32"`` belong to the tensor
        cores; the knob exists only when the kernel's precision keeps them
        exact (``cute_fp32_dot_exact``, recorded by DeviceIR).
        """
        return self.cute_fp32_dot_exact and any(
            fact.lhs_dtype == torch.float32 and fact.rhs_dtype == torch.float32
            for fact in self.matmul_facts
        )

    def supported_config_keys(self) -> frozenset[str]:
        return self._config_key_support()[0]

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

    def unsupported_config_keys(self, config: Mapping[str, object]) -> list[str]:
        return sorted(self._config_key_support()[1].intersection(config))

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

        surface = self.search_surface
        surface.normalize_entry(self, config, fix_invalid=_fix_invalid)

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

        surface.normalize_family_gates(self, config, fix_invalid=_fix_invalid)

        if unsupported := self.unsupported_config_keys(config):
            # Separate backend-specific keys (e.g. AMD tunables, TileIR tunables)
            # from common keys (e.g. num_warps, num_stages, indexing).
            # Backend-specific keys should raise errors; common keys are
            # silently stripped so configs are portable across backends.
            specific_keys = backend_specific_keys()
            backend_specific = [k for k in unsupported if k in specific_keys]
            common = [k for k in unsupported if k not in specific_keys]
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
        surface.normalize_knobs(self, config, fix_invalid=_fix_invalid)
        provided_keys = set(config)
        surface.normalize_before_sequences(self, config, fix_invalid=_fix_invalid)

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
            raw = config.get(name, ())
            if (
                name == "static_ranges"
                and isinstance(raw, (list, tuple))
                and len(raw) > len(mapping)
            ):
                if not _fix_invalid:
                    warnings.warn(
                        "Ignoring config['static_ranges'] because the current kernel "
                        f"has {len(mapping)} static-range loops, but the config has "
                        f"{len(raw)} values",
                        RuntimeWarning,
                        stacklevel=5,
                    )
                # static_ranges has no block-id metadata, so positional
                # values may refer to different loops after specialization.
                raw = ()
            config[name] = mapping._normalize(name, raw, flatten=flatten)

        surface.normalize_after_sequences(self, config, fix_invalid=_fix_invalid)

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
            reduction_loops = config.get("reduction_loops", [])
            if isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    block_threshold, max_chunk = spec._limits(self)
                    if block_threshold is None:
                        continue
                    if new_loops[i] is None and spec.size_hint > block_threshold:
                        new_loops[i] = min(spec.size_hint, block_threshold)
                        changed = True
                    elif (
                        new_loops[i] is not None
                        and max_chunk is not None
                        and new_loops[i] > max_chunk
                    ):
                        new_loops[i] = surface.reduction_loop_chunk_cap(
                            self, max_chunk, block_threshold
                        )
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

        surface.normalize_reduction_threads(
            self, config, fix_invalid=_fix_invalid, register_tiles=_cute_register_tiles
        )

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
            "reduction_fusion",
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
        if self.supports_config_key("reduction_fusion"):
            if self.reduction_fusion_available:
                config.setdefault("reduction_fusion", False)
            if (
                "reduction_fusion" in config
                and type(config["reduction_fusion"]) is not bool
            ):
                if _fix_invalid:
                    config["reduction_fusion"] = False
                else:
                    raise InvalidConfig("reduction_fusion must be a bool")
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
        surface.normalize_after_defaults(self, config, fix_invalid=_fix_invalid)
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
        surface.normalize_recurrence_families(self, config, fix_invalid=_fix_invalid)
        surface.normalize_pre_pid_type(self, config, fix_invalid=_fix_invalid)
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
                # ``allowed_pid_types`` is order-preserving and non-empty, so
                # its head is the preferred legal choice even when ``flat``
                # has been disallowed (``hl.barrier()``, forced persistence).
                config["pid_type"] = self.allowed_pid_types[0]

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

        surface.normalize_strategy(self, config, fix_invalid=_fix_invalid)

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
                config["num_sm_multiplier"] = self.default_num_sm_multiplier

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
                "num_sm_multiplier", self.default_num_sm_multiplier
            )
            if num_sm_multiplier != self.default_num_sm_multiplier:
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
                if not self.supports_config_key(name):
                    continue
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=self.grid_block_ids
                )

        range_warp_specializes = cast(
            "list[bool | None]", config.get("range_warp_specializes", [])
        )

        if range_warp_specializes and any(range_warp_specializes):
            # Only one range_warp_specializes per loop nest, take the first one
            # Prefer warp specialize on outermost loop
            nest_starts = [0]
            if config.get("cross_loop_pipeline") == "static":
                # Static cross-loop roots are sequential loop nests in one kernel.
                grid_ids = {*self.grid_block_ids}
                nest_starts = [
                    i
                    for i, spec in enumerate(self.range_warp_specialize)
                    if i == 0 or spec.block_ids[0] in grid_ids
                ]
            nest_stops = [*nest_starts[1:], len(range_warp_specializes)]
            range_unroll_factors = cast(
                "list[int]", config.get("range_unroll_factors", [])
            )
            for start, stop in zip(nest_starts, nest_stops, strict=True):
                nest = range_warp_specializes[start:stop]
                if True not in nest:
                    continue
                first_idx = start + nest.index(True)
                for i in range(first_idx + 1, stop):
                    range_warp_specializes[i] = None
                if range_unroll_factors and range_unroll_factors[first_idx] > 1:
                    range_unroll_factors[first_idx] = 0
                    config["range_unroll_factors"] = range_unroll_factors

        if self.supports_config_key("range_warp_specializes"):
            config["range_warp_specializes"] = range_warp_specializes

        surface.normalize_finalize(
            self, config, fix_invalid=_fix_invalid, provided_keys=provided_keys
        )

        # Allow tunable parameter keys in addition to backend-supported keys.
        unknown_keys = {*config} - valid_keys()
        invalid_keys = (unknown_keys | {*self.unsupported_config_keys(config)}) - {
            *self.user_defined_tunables.keys()
        }
        if invalid_keys:
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def raise_grid_block_minimums(self) -> None:
        """Raise search floors for grid block dimensions based on problem size.

        Very small block sizes produce enormous grids that the autotuner
        wastes time exploring.  This heuristic floors each root grid so its
        program count (the product of its dimensions' block counts) stays
        within a budget derived from ``num_compute_units``.

        The budget is shared across the dimensions of one grid by water
        filling: a dimension that cannot use its even share of the budget
        (``B=4`` in a ``[B, L]`` grid) is left unfloored and hands its unused
        share to the remaining dimensions, so ``L`` may still tile down to 1
        when the whole grid fits the budget.  Each root grid (``KernelGridFact``
        roots, or the fissioned pointwise launches) owns a full budget; with
        no recorded grid topology every grid axis belongs to one grid, the
        legacy rule.

        The raised minimum never exceeds the default block size that
        ``_fragment`` would compute, so memory and shared-memory
        constraints from non-tiled dimensions are respected.
        """
        if not self.grid_block_ids:
            return

        budget = num_compute_units() * 64
        for grid in self._grid_floor_groups():
            specs: list[BlockSizeSpec] = []
            for grid_bid in grid:
                try:
                    spec = self.block_sizes.block_id_lookup(grid_bid)
                except KeyError:
                    continue
                if spec.size_hint > 0:
                    specs.append(spec)
            remaining = float(budget)
            # Smallest dimensions first: they are the ones that may not use
            # their even share, and the surplus flows to the larger ones.
            # Dimensions of one extent take the same share at once, so a
            # symmetric grid floors every axis alike (for a uniform grid this
            # is exactly the legacy per-axis rule, ceil(budget ** (1 / n))).
            ordered = sorted(specs, key=lambda s: s.size_hint)
            index = 0
            while index < len(ordered):
                size = ordered[index].size_hint
                tied = [spec for spec in ordered[index:] if spec.size_hint == size]
                share = remaining ** (1.0 / (len(ordered) - index))
                index += len(tied)
                if size <= share:
                    # The whole dimension fits in its share at block size 1.
                    remaining = max(remaining / size ** len(tied), 1.0)
                    continue
                max_blocks = math.ceil(share)
                remaining = max(remaining / max_blocks ** len(tied), 1.0)
                for spec in tied:
                    default = spec._fragment(self).default_val
                    min_block = min(size // max_blocks, default)
                    if min_block >= 2:
                        min_block = 1 << (min_block.bit_length() - 1)
                        min_block = min(min_block, spec.max_size)
                        spec.autotuner_min = assert_integer_power_of_two(
                            max(min_block, spec.autotuner_min)
                        )

    def _grid_floor_groups(self) -> list[tuple[int, ...]]:
        """Root grids for the occupancy floor, each owning a full budget."""
        independent = [
            tuple(grid) for grid in self.cute_pointwise_region_grid_groups if grid
        ]
        covered = {block_id for grid in independent for block_id in grid}
        if self.kernel_grid_fact is not None:
            roots = [
                tuple(block_id for block_id in root if block_id not in covered)
                for root in self.kernel_grid_fact.grid_groups
            ]
            known = covered | {block_id for root in roots for block_id in root}
            # Axes added after the topology was recorded (a promoted output
            # axis) share one grid with every other unassigned axis.
            rest = tuple(b for b in self.grid_block_ids if b not in known)
        else:
            roots = []
            rest = tuple(b for b in self.grid_block_ids if b not in covered)
        return [grid for grid in (*independent, *roots, rest) if grid]

    def create_config_generation(
        self,
        *,
        overrides: Mapping[str, object] | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
        compiler_coverage_enabled: bool = True,
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        effective_overrides, family_override = (
            self.search_surface.prepare_config_generation_overrides(self, overrides)
        )
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
        return self.search_surface.flatten_missing_field_default(self, key, config)

    def prepare_override_normalization(
        self,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        self.search_surface.prepare_override_normalization(self, config, overrides)

    def _base_default_config(self) -> helion.Config:
        config = self.flat_config(lambda x: x.default())
        self.search_surface.trim_reference_config(self, config)
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
        self.search_surface.trim_default_config(self, config)
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
            max(self.block_sizes[i].min_size, self.block_sizes[i].structural_pin or 1)
            for i in range(len(block_sizes))
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
            if self.block_sizes[i].structural_pin is not None:
                # Normalization would restore the pinned size: no progress.
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
        surface_fields = self.search_surface.fields(
            self, flash_pipeline_family=_flash_pipeline_family_override
        )
        if surface_fields is not None:
            return surface_fields
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "block_sizes": self.block_sizes,
        }
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
        if self.reduction_fusion_available and self.supports_config_key(
            "reduction_fusion"
        ):
            fields["reduction_fusion"] = BooleanFragment()
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
                self.default_num_sm_multiplier,
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
        full_extent_slots = tuple(
            int(item.allow_full_extent) for item in self.reduction_loops
        )
        if any(full_extent_slots):
            # Old flat full-size tokens represented persistent reductions. Do
            # not reinterpret a cached legacy token as an explicit full loop.
            # Version the capability, not the bound extent: autotune_multi must
            # accept the same kernel at power-of-two and neighboring shapes.
            result.append(
                (
                    "reduction_loop_full_extent_encoding_v2",
                    *full_extent_slots,
                )
            )
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
        result = (
            legacy_hash
            if policy is None
            else hashlib.sha256(repr((legacy_hash, policy)).encode("utf-8")).hexdigest()
        )
        if self.compiler_seed_overflow_requested and self.compiler_seed_configs:
            return hashlib.sha256(
                repr((result, "compiler_seed_overflow:v1")).encode("utf-8")
            ).hexdigest()
        return result

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
        target_policy_identity = self.search_surface.fingerprint_extra(self)
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

    def _fields_for_family(
        self, flash_pipeline_family: str | None
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        # ``_flat_fields`` stays the polymorphic entry point (``UserConfigSpec``
        # overrides it and tests patch it); a pinned family bypasses it.
        if flash_pipeline_family is None:
            return self._flat_fields()
        return self._flat_fields_with_flash_family(flash_pipeline_family)

    def flat_key_layout(
        self,
        *,
        advanced_controls_files: list[str] | None = None,
        flash_pipeline_family: str | None = None,
    ) -> list[tuple[str, int, bool]]:
        """Return (key_name, num_flat_entries, is_sequence) for each field.

        is_sequence is True for BlockIdSequence keys whose list values
        are spread across individual flat slots.  ``flash_pipeline_family``
        pins the flash family whose fields the layout describes.
        """
        fields = self._fields_for_family(flash_pipeline_family)
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
        return self.flat_key_layout(
            advanced_controls_files=advanced_controls_files,
            flash_pipeline_family=flash_pipeline_family,
        )

    def flat_config(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        *,
        advanced_controls_files: list[str] | None = None,
        flash_pipeline_family: str | None = None,
    ) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        return self._flat_config_from_fields(
            fn,
            self._fields_for_family(flash_pipeline_family),
            advanced_controls_files=advanced_controls_files,
        )

    def _flat_config_with_flash_family(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        *,
        advanced_controls_files: list[str] | None,
        flash_pipeline_family: str,
    ) -> helion.Config:
        return self.flat_config(
            fn,
            advanced_controls_files=advanced_controls_files,
            flash_pipeline_family=flash_pipeline_family,
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

        self.search_surface.prepare_flat_config(self, config)

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
        self._max_size: int = self.dim_max_size
        self._max_size_revision: int = 0
        # A search surface may pin the default (non-autotuned) block size while
        # widening the autotuner's range (see ``_fragment``).
        self.default_size: int | None = None
        # The ceiling the kernel declared (``hl.register_block_size`` bounds,
        # ``update_max`` from an MMA shape analysis), independent of the
        # dimension's extent; ``None`` when only the extent caps the search
        # (``update_hint``). Fused surfaces whose tiles overshoot the
        # dimension (ragged flash rows) check their pins against this.
        self.declared_max_size: int | None = None
        # Outer block_id whose tile extent caps this block's size in normalize().
        self.bounded_by_block_id: int | None = bounded_by_block_id
        # The one block size a fused lowering's envelope admits
        # (``pin_structurally``): the search sees a single point and a
        # provided config's other value is normalized onto it (see
        # ``_normalize``). ``None`` where no surface pinned this block.
        self.structural_pin: int | None = None
        self.structural_pin_reason: str = ""
        self._structural_pin_warned: set[int] = set()
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
        if not isinstance(result, int):
            return result
        pin = self.structural_pin
        if pin is not None and result != pin:
            # A decorator / saved config written for another backend (the
            # examples pin Triton-shaped 64x64 tiles) must not silently pick
            # the scalar lowering: the fused family is the only one the
            # surface offers, so the value is normalized onto its envelope.
            if result not in self._structural_pin_warned:
                self._structural_pin_warned.add(result)
                log.warning(
                    "%s=%d in the provided config is normalized onto %d: "
                    "%s pins this block size",
                    name,
                    result,
                    pin,
                    self.structural_pin_reason,
                )
            return pin
        if result < self.min_size:
            result = self.min_size
        return result

    def pin_structurally(self, value: int, *, reason: str) -> None:
        """Pin this block size to the one value a fused lowering admits.

        The autotuner's range collapses to ``value`` and ``_normalize`` maps
        any other provided value onto it (with a logged warning naming the
        key and ``reason``), so a config pinned by a decorator or loaded from
        a cache written for another backend still reaches the fused family.
        """
        self.autotuner_min = value
        self.max_size = value
        self.structural_pin = value
        self.structural_pin_reason = reason

    def update_min(self, value: int) -> None:
        self.min_size = assert_integer_power_of_two(max(value, self.min_size))
        if self.max_size < self.min_size:
            self.max_size = self.min_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        # Record even an equal assignment: an independent consumer or explicit
        # user bound may impose the same ceiling as an earlier compiler cap.
        self._max_size = value
        self._max_size_revision += 1

    @property
    def max_size_revision(self) -> int:
        """Mutation token for a compiler-owned ceiling that may later widen."""
        return self._max_size_revision

    def update_max(self, value: int) -> None:
        clamped = max(value, 1)
        self.max_size = assert_integer_power_of_two(min(clamped, self.max_size))
        self.declared_max_size = (
            clamped
            if self.declared_max_size is None
            else min(self.declared_max_size, clamped)
        )

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
        """Resolve the dimension's extent: it caps ``max_size`` but is not a
        declared ceiling (see ``declared_max_size``)."""
        self.size_hint = value
        self.max_size = min(self.max_size, next_power_of_2(max(value, 1)))

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
        # CuTe SIMT reductions can use any whole number of warps. Keep these
        # values available to explicit seeds even though the random search
        # fragment remains power-of-two for a compact search space.
        if type(value) is int and 32 <= value <= 1024 and value % 32 == 0:
            return value
        return super()._normalize(name, value)

    def _fragment(self, base: ConfigSpec) -> NumThreadsFragment | EnumFragment:
        if base.search_surface.pins_thread_knobs(base, self.block_id):
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
        allow_non_power_of_two: bool = False,
        allow_full_extent: bool = False,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint
        self.allow_non_power_of_two = allow_non_power_of_two
        self.allow_full_extent = allow_full_extent

    def _full_extent_chunk(self) -> int | None:
        # A padded single iteration can preserve vector packets where an exact
        # non-power-of-two chunk leaves too few elements per thread. Keep it
        # distinct from persistent None, just like an exact full-width chunk.
        if self.allow_full_extent and self.size_hint >= 8:
            return next_power_of_2(self.size_hint)
        return None

    def _limits(self, base: ConfigSpec) -> tuple[int | None, int | None]:
        """Persistent thread threshold and rolled chunk limit for this axis."""
        return base.search_surface.reduction_loop_limits(
            base,
            self.block_id,
            base.reduction_loop_force_threshold,
            base.max_reduction_loop,
        )

    def _flat_fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        # Shared by both directions:
        # - unflatten: flat integer -> Config value via _flat_config()
        # - flatten: Config value -> flat integer via _encode_flat_value()
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(max(low, self.size_hint))
        default = min(high, 4096)
        # Cap default at the backend's max reduction loop so that
        # large reductions default to looped rather than persistent.
        force_threshold, max_chunk = self._limits(base)
        if max_chunk is not None:
            if force_threshold is not None and self.size_hint > force_threshold:
                default = prev_power_of_2(default, max_chunk)
        if high == self._full_extent_chunk():
            # The full padded chunk now denotes an explicit rolled loop.
            # Reserve the next power of two for None, retaining the old default
            # whenever that default selected a persistent reduction.
            old_high = high
            high *= 2
            if default == old_high:
                default = high
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
        if value >= self.size_hint and not (
            value == self._full_extent_chunk()
            or (
                self.allow_non_power_of_two
                and value == self.size_hint
                and value & (value - 1) != 0
            )
        ):
            return None  # max size becomes persistent reduction
        return value

    def _encode_flat_value(self, base: ConfigSpec, value: object) -> object:
        # Encode None ("persistent reduction") so the inverse ``_flat_config``
        # decodes it back to None. The fragment's ``high`` is >= size_hint,
        # and strictly greater when the full extent is an explicit loop.
        # The fragment *default* is
        # capped at max_reduction_loop and can fall below size_hint, which would
        # round-trip None into a slow looped config (e.g. size_hint=32000 ->
        # default 4096 -> reduction_loops=[4096]).
        # The backend may repair search/cache/seed budgets before decoding.
        value = base.search_surface.explicit_reduction_chunk(base, self, value)
        if value is None:
            return self._flat_fragment(base).high
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        if (
            self.allow_non_power_of_two
            and type(value) is int
            and value >= 8
            and value <= self.size_hint
            and self.size_hint % value == 0
            and value & (value - 1) != 0
        ):
            # Exact non-power-of-two divisors avoid padding model widths such
            # as 3584 and 7168. They are seeded explicitly; the continuous
            # search fragment remains power-of-two for a compact search space.
            normalized = value
        else:
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
        # sync regardless of how the value was generated. CuTe can opt into
        # a distinct explicit full-loop encoding; None remains persistent.
        # This distinction does not waive multi-pass cache correctness checks.
        if (
            isinstance(normalized, int)
            and normalized >= self.size_hint
            and normalized & (normalized - 1) == 0
            and normalized != self._full_extent_chunk()
        ):
            return None
        return normalized

    def _fill_missing(self) -> None:
        return None


_CUTE_VECTOR_WIDTH_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16)
_CUTE_LANE_LAYOUT_CHOICES: tuple[str, ...] = ("blocked", "strided")
_CUTE_REDUCTION_RELOAD_CHOICES: tuple[str, ...] = (
    "auto",
    "register",
    "gmem",
    "register_prefetch",
)
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
    keeps the legacy size heuristic in ``fuse_two_pass_loads``. The optional
    ``"register_prefetch"`` mode retains register reloads and also prefetches
    one proven read-only consumer packet across a one-trip reduction.
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
        if base.search_surface.pins_thread_knobs(base, self.block_id):
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

    V=1 disables vectorization (scalar loads). V=2/4/8/16 emits
    ``cute.arch.load(..., ir.VectorType.get([V], elem_dtype.mlir_type))``
    for the inner reduction load. SM100+ also admits 32-byte vectors.
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
        if base.search_surface.pins_thread_knobs(base, self.block_id):
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
    # Set by ``lower_to_device_ir`` (``intra_loop_raw_reread_block_ids``) for
    # a loop whose region stores and then re-reads the same storage: Triton's
    # warp specialization splits such a body across partitions and the
    # re-read observes stale memory.  The slot stays, so the config shape
    # does not depend on the analysis, but its value is pinned to None.
    pinned: bool = False

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        if self.pinned:
            return EnumFragment((None,))
        return super()._fragment(base)

    def _normalize(self, name: str, value: object) -> bool | None:
        value = super()._normalize(name, value)
        return None if self.pinned else value


class RangeNumStagesSpec(_OptionalIntSpec):
    # Set by ``CompileEnvironment._pin_range_num_stages_for_aliasing``
    # (triton#8259) for a loop that reads and writes argument storage: the slot
    # stays, so the config shape does not depend on the alias analysis, but
    # its value is pinned to 0 (no per-loop ``num_stages``).
    pinned: bool = False

    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        if self.pinned:
            return IntegerFragment(0, 0, 0)
        return IntegerFragment(0, 4, 0)

    def _normalize(self, name: str, value: object) -> int:
        value = super()._normalize(name, value)
        return 0 if self.pinned else value

    def _omitted_from_legacy_configs(self) -> bool:
        # The workaround used to remove a pinned loop's slot outright.
        return self.pinned


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


# The backend-specific key tables are derived from the registered backends'
# search surfaces (``Backend.config_spec_extension``): a backend declares its
# keys once, on its surface.  They are computed on first use and recomputed
# when the registry changes (``registry_generation``), so a backend registered
# after this module imported contributes its keys too.  In-package code calls
# the accessors; the module attributes ``BACKEND_SPECIFIC_KEYS`` /
# ``VALID_KEYS`` / ``DEBUG_KEYS`` resolve to the current tables through
# ``__getattr__`` for callers that import them by name.
_KEY_TABLES: tuple[int, frozenset[str], frozenset[str], frozenset[str]] | None = None


def _key_tables() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    global _KEY_TABLES
    generation = registry_generation()
    cached = _KEY_TABLES
    if cached is None or cached[0] != generation:
        surfaces = all_search_surfaces()
        backend_specific = (
            BACKEND_TUNABLE_KEYS
            | _BACKEND_GATED_GENERIC_KEYS
            | frozenset().union(
                *(surface.config_keys | surface.debug_keys for surface in surfaces)
            )
        )
        cached = (
            generation,
            backend_specific,
            GENERIC_KEYS | backend_specific,
            frozenset().union(*(surface.debug_keys for surface in surfaces)),
        )
        _KEY_TABLES = cached
    return cached[1], cached[2], cached[3]


def backend_specific_keys() -> frozenset[str]:
    """All config keys whose support depends on the backend.  The base Backend
    class rejects these by default; each backend's search surface opts in
    (``Backend.supports_config_key``: own-surface keys, or keys no backend
    declares)."""
    return _key_tables()[0]


def valid_keys() -> frozenset[str]:
    """Every config key some registered backend accepts: the generic keys and
    :func:`backend_specific_keys`."""
    return _key_tables()[1]


def debug_keys() -> frozenset[str]:
    """Diagnostic keys: accepted by ``normalize``, never searched or
    fingerprinted."""
    return _key_tables()[2]


def __getattr__(name: str) -> frozenset[str]:
    if name == "BACKEND_SPECIFIC_KEYS":
        return backend_specific_keys()
    if name == "VALID_KEYS":
        return valid_keys()
    if name == "DEBUG_KEYS":
        return debug_keys()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
