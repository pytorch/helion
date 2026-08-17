from __future__ import annotations

import enum
import functools
import hashlib
import itertools
import logging
import math
import operator
import os
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
from .._compat import num_compute_units
from .._compat import supports_amd_cdna_tunables
from .._compat import supports_maxnreg
from .._compat import supports_tensor_descriptor
from .._compat import target_device_capability as get_target_device_capability
from .._compat import warps_to_threads
from .._compiler.cute.cute_flash import FLASH_CONFIG_KEYS
from .._compiler.cute.cute_flash import FLASH_CORR_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_FREQ_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_OFFSET0_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_OFFSET_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_RES_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_SCHEDULE_KEY
from .._compiler.cute.cute_flash import FLASH_EXP2_IMPL_KEY
from .._compiler.cute.cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
from .._compiler.cute.cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
from .._compiler.cute.cute_flash import FLASH_MMA_INTERLEAVE_KEY
from .._compiler.cute.cute_flash import FLASH_OTHER_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_PERSISTENT_KEY
from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
from .._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_TOPOLOGY_KEY
from .._compiler.cute.cute_flash import FlashAttentionConfig
from .._compiler.cute.cute_flash import _flash_causal_hd64_seed_num_kv_supported
from .._compiler.cute.cute_flash import _flash_causal_hd64_seed_offset0
from .._compiler.cute.cute_flash import _flash_causal_hd64_seed_params
from .._compiler.cute.cute_flash import _flash_e2e_offset_period
from .._compiler.cute.cute_flash import _flash_e2e_schedule_default
from .._compiler.cute.cute_flash import _flash_masked_e2e_schedule_params
from .._compiler.cute.cute_flash import _flash_normalize_e2e_offset
from .._compiler.cute.cute_flash import _flash_normalize_e2e_params
from .._compiler.cute.cute_flash import _flash_parse_e2e_schedule
from .._compiler.cute.cute_flash import _flash_pipeline_family_flags
from .._compiler.cute.cute_flash import flash_effective_config_values
from .._compiler.cute.cute_flash import resolve_flash_config
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_TUNABLE_KEYS
from .._compiler.cute.tcgen05_config import CuteTcgen05Config
from .._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from .._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ..exc import InvalidConfig
from ..runtime.triton.launcher import get_num_xcd
from .block_id_sequence import BlockIdSequence
from .block_id_sequence import _BlockIdItem
from .block_id_sequence import _PowerOfTwoBlockIdItem
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

    from .._compiler.backend import Backend
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


class TensorNumelConstraint(NamedTuple):
    """Tensor element count must stay within Triton's max numel limit."""

    check_fn: Callable[..., bool]
    block_indices: tuple[int, ...]
    expr_str: str


class MatmulFact(NamedTuple):
    """Shape facts recorded when matmul requirements are applied."""

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
    """One simultaneously-live tensor tile at a graph's peak-live step.

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
    - ``kind`` — ``"dot_out"`` | ``"load"`` | ``"carry"`` | ``"other"``.
    - ``stageable`` — for a loop-body load, whether its index is proven to
      vary with the enclosing loop. ``None`` means uncertain and is charged
      conservatively as stageable.
    """

    dim_block_ids: tuple[int | None, ...]
    static_dims: tuple[int | None, ...]
    itemsize: int
    kind: str
    stageable: bool | None = None


class LoopAxisFact(NamedTuple):
    """One tunable or fixed axis in an enclosing device loop.

    ``extent`` is the full axis length. ``bounded_by_block_id`` records an inner
    loop over exactly one outer tile (for example split-K's
    ``tile(outer.begin, outer.end)``); in that case the candidate outer block is
    the real extent, not the whole underlying axis. ``bounded_extent`` is set
    only when that outer extent is a literal; a symbolic fixed parent remains
    explicitly uncertain. The per-program block is deliberately not recorded
    here: it is a candidate property and must be resolved from the emitted
    ``block_sizes`` before the trip count is used.
    """

    block_id: int
    extent: int | None
    bounded_by_block_id: int | None = None
    bounded_extent: int | None = None


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
    - ``loop_trips`` — product of the sequential loop trip counts enclosing it, i.e.
      how many times one program executes this dot.
    - ``updates_carry`` — the dot writes into a value carried by an enclosing loop
      (``acc=`` into a loop-carried accumulator). Such a dot's accumulator is
      resident for the whole loop, which is why it gets ranking priority.
    - ``loop_axes`` — the enclosing loop axes before a candidate block size is
      selected. Consumers that need an exact execution count resolve these axes
      against the candidate instead of treating ``loop_trips`` as exact.
    - ``exact_loop_trips`` — an exact dynamic execution count proven independently
      of the loop-bound expression. This is used for work estimation only; it does
      not claim that the same loop is a useful software-pipeline opportunity.
    """

    graph_id: int
    loop_trips: int
    updates_carry: bool
    loop_axes: tuple[LoopAxisFact, ...] = ()
    exact_loop_trips: int | None = None


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
    - ``outer_grid`` — parallel programs contributed by grid axes that are no dot's
      M or N tile: the occupancy the kernel already has before any tiling choice.
    - ``sequential_loop_trips`` — product of the trip counts of the kernel's
      sequential (non-grid) loops. For a chunked recurrence this is the number of
      chunks, so ``sequential_loop_trips * fixed_k`` recovers the logical contraction
      length even though each dot only sees one chunk.
    - ``live_tiles`` — the peak simultaneously-live tile set over the whole kernel
      (:class:`LiveTile`), from the same last-use liveness sweep the reduction fact
      uses. The maximum simultaneously-live footprint, NOT the sum of every op.
    - ``live_dot_outputs`` — the live set at the step holding the most simultaneously
      live DOT OUTPUT tiles. A separate measurement from ``live_tiles`` because that
      one's peak is chosen by rank profile (the right question for a reduction's
      register working set, the wrong one for the accumulator budget: the rank peak can
      be a step where the heavy loads are live and the accumulators are not). This is
      what a sum-over-live-accumulators tensor-memory/register term is computed from.
    - ``live_tile_steps`` — the live tile set at EVERY step of the kernel's graphs, deduped.
      ``live_tiles`` above is one step chosen by RANK PROFILE, which is block-size-free and so
      the right question when block sizes are unknown; a register estimate is asked later, when
      the candidate block sizes ARE known, and should pick its peak by resolved BYTES. Keeping
      every step is what lets it. Selecting by rank instead under-counted a kernel that spills
      540 registers at one warp while over-counting one that spills none — the ordering
      inverted, so no threshold on that estimate could separate them.
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
    - ``n_dot_nodes`` — contraction sites counted spelling-independently, so a kernel
      whose dots are written as ``torch.matmul``/``@`` is not read as having none.
    - ``attribution_complete`` — post-condition flag described above.
    """

    matmuls: tuple[ResolvedMatmulFact, ...]
    knob_users: tuple[tuple[int, tuple[tuple[int, str], ...]], ...]
    outer_grid: int
    sequential_loop_trips: int
    live_tiles: tuple[LiveTile, ...]
    live_dot_outputs: tuple[LiveTile, ...]
    live_tile_steps: tuple[tuple[LiveTile, ...], ...]
    pipelined_regions: tuple[PipelinedRegion, ...]
    resident_regions: tuple[ResidentRegion, ...]
    n_dot_nodes: int
    attribution_complete: bool

    def users_of(self, block_id: int) -> tuple[tuple[int, str], ...]:
        for bid, users in self.knob_users:
            if bid == block_id:
                return users
        return ()

    def dot_work(self, index: int) -> int:
        """Dynamic work of one dot: ``m*n*k`` per execution times its executions.

        Falls back to the axis classification's per-program extent when the fact's own
        ``static_*`` is absent, which happens whenever the axis length is dynamic but the
        per-program extent is not."""
        resolved = self.matmuls[index]
        f = resolved.fact
        ax = resolved.axes
        m = f.static_m or ax.m_extent or 1
        n = f.static_n or ax.n_extent or 1
        k = f.static_k or ax.k_extent or 1
        site = resolved.site
        trips = site.exact_loop_trips or site.loop_trips
        return m * n * k * max(1, trips)


class ReductionCategory(enum.Enum):
    """How a reduction axis maps onto the program grid — one category per reduction.

    - ``FULL_SLICE`` — the whole axis is reduced within one program (``x[m, :]``); not on the grid.
    - ``FULL_GRID`` — a full-extent axis on the grid, block == extent (fully resident per program).
    - ``GRID_TILE`` — a grid axis reduced over but NOT full-extent: the grid parallelizes the
      reduction across programs, so the whole-axis size is not a per-program extent.
    - ``USER_TILE`` — an inner sequential ``hl.tile`` the user wrote over the reduction axis.
    - ``DECLINED`` — no static extent (e.g. jagged / data-dependent); recorded but never sized.
    """

    FULL_SLICE = "full_slice"
    FULL_GRID = "full_grid"
    GRID_TILE = "grid_tile"
    USER_TILE = "user_tile"
    DECLINED = "declined"


# Categories the seed sizes a per-program reduction extent for. GRID_TILE (grid-parallelized
# partial) stays a grid row and DECLINED (no static extent) falls back to the default, so neither
# is sized as a reduction.
SIZED_REDUCTION_CATEGORIES = frozenset(
    {
        ReductionCategory.FULL_SLICE,
        ReductionCategory.FULL_GRID,
        ReductionCategory.USER_TILE,
    }
)
# Categories that occupy the full reduction extent within one program.
FULL_EXTENT_CATEGORIES = frozenset(
    {ReductionCategory.FULL_SLICE, ReductionCategory.FULL_GRID}
)


class ReductionDescriptor(NamedTuple):
    """One reduction OCCURRENCE: a (``graph_id``, ``block_id``) reduction on the ORIGINAL
    (pre-roll) device graphs. Stage 1 emits a list of these; the Stage-2 allocator consumes them.

    A reduction axis may occur in more than one original graph (e.g. a kernel that reduces the
    same axis in two separate passes) — each occurrence is its own descriptor, so sequential
    passes over one axis are NOT collapsed.

    Descriptors with the same ``graph_id`` are co-resident (the compiler fused them into one graph
    -> shared resident working set). ``graph_id`` is read off the ORIGINAL graphs only (rolled
    ``ReductionLoopGraphInfo`` subgraphs excluded), so it is invariant to the autotuner flipping a
    ``reduction_loops`` knob.

    Fields:
    - ``category``: the :class:`ReductionCategory`.
    - ``block_id`` / ``graph_id``: the reduction axis + its original-graph co-residency key.
    - ``size_hint`` / ``itemsize`` / ``input_load_itemsize``: extent (element count), the
      fp32-promoted accumulator itemsize, and the HBM-load element width feeding it.
    - ``carried_2d_count``: the NUMBER of >=2-D ``[M_BLOCK, R_BLOCK]`` loop-carried accumulators
      whose last dim is this rdim (e.g. kl_div=1, jsd=2); those tiles stay resident the whole loop.
      A count (not a bool) because the carried byte cap divides the budget by it. 0 = none here.
    - ``row_reread`` / ``reread_eviction_index`` / ``num_load``: per-reduction memory-op signals.
    """

    category: ReductionCategory
    block_id: int
    graph_id: int
    size_hint: int
    itemsize: int
    input_load_itemsize: int = 0
    carried_2d_count: int = 0
    row_reread: bool = False
    reread_eviction_index: int | None = None
    num_load: int = 0


class CoResidencyGroup(NamedTuple):
    """A ``graph_id`` equivalence class of reductions whose working tiles are live at the same
    time, so ONE budget must fit them all. ``descriptor_indices`` indexes into
    ``ReductionKernelFact.reductions``.

    ``live_tiles`` is the group's resident tile set — one ``dim_block_ids`` tuple per
    register-resident tile (the block id each dim spans, ``None`` for a static/broadcast dim). It
    is the peak live set of the group's home graph, combined with the for-loop bodies the group
    drives and its If/Else branch siblings (see device_ir ``_group_live_tiles``). Each loop-carried
    accumulator is captured inline at its real shape, so the Stage-2 footprint can sum ``∏(dim
    widths)`` per actual tile. Empty when the fact is built without a live env (a bare-spec test).
    """

    graph_id: int
    descriptor_indices: tuple[int, ...]
    live_tiles: tuple[tuple[int | None, ...], ...] = ()


class ReductionKernelFact(NamedTuple):
    """The per-kernel Stage-1 product that Stage 2 consumes: the list of reduction descriptors,
    their co-residency groups (``graph_id`` classes), the non-reduction user-tiled loops (sized as
    a separate pass), and the parallel grid axes (rows with no reduction over them).

    Built by ``build_reduction_kernel_fact``. ``reductions`` may be empty (a kernel with only
    GRID_TILE / DECLINED reductions, or none) — the seed then declines.
    """

    reductions: tuple[ReductionDescriptor, ...]
    coresidency_groups: tuple[CoResidencyGroup, ...]
    non_reduction_loop_block_ids: tuple[int, ...] = ()
    grid_axis_block_ids: tuple[int, ...] = ()


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
    provenance (``None`` for a static dim), ``itemsize`` the element size. Accumulators whose last
    dim is the reduction axis feed ``ReductionDescriptor.carried_2d_count`` (a 1-D ``[M_BLOCK]``
    scalar accumulator counts as 0).
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
    | frozenset(FLASH_CONFIG_KEYS)
    | {
        "num_threads",
        "cute_vector_widths",
        "load_cache_modifiers",
        "store_cache_modifiers",
        "pallas_loop_type",
        "pallas_load_buffer_count",
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
        "pallas_loop_type",
        "pallas_load_buffer_count",
        "pallas_pre_broadcast",
        "cute_vector_widths",
        *BACKEND_TUNABLE_KEYS,
        "advanced_controls_file",
        "epilogue_subtile",
        "xcd_remap",
        *_BACKEND_DIAGNOSTIC_CONFIG_KEYS,
        *_BACKEND_STRATEGY_CONFIG_KEYS,
        *FLASH_CONFIG_KEYS,
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
# maxnreg values: None means no limit, otherwise limit to this many registers per thread
# Lower values allow higher occupancy but may hurt performance for register-heavy kernels
VALID_MAXNREG = (None, 32, 64, 128, 256)
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
    }
)


# For tileir backend or AMD ROCM, eviction policies are not supported.
# Keep this uncached: some tests patch the AMD capability helper, and caching
# only on backend name can poison later Triton ConfigSpec construction inside
# the same worker process.
def get_valid_eviction_policies(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
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
        self.cute_indexed_reduction_block_ids: set[int] = set()
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
        self.has_symbolic_or_data_dependent_bounds: bool = False
        self._cute_tcgen05_config = CuteTcgen05Config(self)
        # CuTe flash-attention autotune surface gating.
        # Default False so the flash knobs never appear in the search surface
        # and behavior is byte-identical to the env-only path. Set True when the
        # flash detector fires (see ``lower_to_device_ir``). The shape needed to
        # build the fragments (head_dim / num_kv) is captured at the same time.
        self.cute_flash_search_enabled: bool = False
        self._cute_flash_head_dim: int | None = None
        self._cute_flash_num_kv: int | None = None
        self._cute_flash_dtype: torch.dtype = torch.float16
        self._cute_flash_is_causal: bool = False
        self._cute_flash_has_kv_tile_pruning: bool = False
        self._cute_flash_requires_ws_overlap: bool = False
        self._cute_flash_small_biased_candidate: bool = False
        self._cute_flash_standard_dense_output: bool = False
        self._cute_flash_standard_causal_output: bool = False
        self._cute_flash_block_size_targets: dict[int, int] = {}
        self.compiler_default_config: helion.Config | None = None
        self.compiler_seed_configs: list[helion.Config] = []
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
        self.backend_tunable_fragments = self.backend.tunable_fragments()
        unknown_tunables = set(self.backend_tunable_fragments) - BACKEND_TUNABLE_KEYS
        if unknown_tunables:
            raise RuntimeError(
                f"Backend {self.backend_name!r} returned unknown tunables: {sorted(unknown_tunables)!r}"
            )

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
            is_causal=self._cute_flash_is_causal,
            standard_dense_output=self._cute_flash_standard_dense_output,
            prefer_packed_reduce=(
                self._cute_flash_has_kv_tile_pruning
                or self._cute_flash_requires_ws_overlap
            ),
        )

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
        from .._compiler.cute.cute_flash import flash_autotune_fragments

        block_size_targets = self._cute_flash_block_size_target_list()
        if fix_invalid:
            config["block_sizes"] = list(block_size_targets)
            config["pid_type"] = "flat"
            self._normalize_cute_flash_default_sequence(config, "l2_groupings", 1)
            self._normalize_cute_flash_default_sequence(config, "num_threads", 0)
            self._normalize_cute_flash_default_sequence(config, "cute_vector_widths", 1)
            self._normalize_cute_flash_default_loop_orders(config)
            config.pop("epilogue_subtile", None)
        elif not self._is_cute_flash_config_envelope(config, block_size_targets):
            return

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
            pipeline_family_override = (
                str(config[FLASH_PIPELINE_FAMILY_KEY])
                if requested_family is not None
                else legacy_effective.pipeline_family
                if legacy_effective is not None
                else None
            )
            if requested_family is not None:
                topology_override = requested_family.topology
            else:
                valid_manual_topologies = {"fa4", "ws_overlap"}
                topology_value = config.get(FLASH_TOPOLOGY_KEY)
                topology_override = (
                    topology_value
                    if topology_value in valid_manual_topologies
                    else None
                )
        fragments = flash_autotune_fragments(
            self._cute_flash_head_dim,
            self._cute_flash_num_kv,
            dtype=self._cute_flash_dtype,
            is_causal=self._cute_flash_is_causal,
            has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=self._cute_flash_requires_ws_overlap,
            small_biased_candidate=self._cute_flash_small_biased_candidate,
            standard_dense_output=self._cute_flash_standard_dense_output,
            standard_causal_output=self._cute_flash_standard_causal_output,
            topology_override=cast("str | None", topology_override),
            pipeline_family_override=pipeline_family_override,
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
        effective_topology = effective.topology
        config.update(flash_effective_config_values(effective))
        if effective_topology == "fa4":
            config.update(explicit_e2e_offsets)
        e2e_schedule_default = (
            "8/2"
            if (
                effective_topology == "fa4"
                and self._cute_flash_is_causal
                and self._cute_flash_head_dim == 64
                and _flash_causal_hd64_seed_num_kv_supported(self._cute_flash_num_kv)
            )
            else _flash_e2e_schedule_default(
                effective_topology, self._cute_flash_head_dim
            )
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
            if self._cute_flash_is_causal and _flash_causal_hd64_seed_num_kv_supported(
                self._cute_flash_num_kv
            ):
                schedule_default_offset = (
                    _flash_causal_hd64_seed_params(self._cute_flash_num_kv)[0]
                    % e2e_offset_period
                )
            else:
                split_default_freq = e2e_freq if e2e_res > 0 else masked_e2e_freq
                schedule_default_offset = split_default_freq // 8
        else:
            schedule_default_offset = 0
        default_offset = schedule_default_offset
        env_offset = os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET")
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
        default_offset0 = (
            _flash_causal_hd64_seed_offset0(self._cute_flash_num_kv)
            if (
                e2e_offset_period > 0
                and effective_topology == "fa4"
                and self._cute_flash_is_causal
                and self._cute_flash_head_dim == 64
                and _flash_causal_hd64_seed_num_kv_supported(self._cute_flash_num_kv)
            )
            else 0
        )
        env_offset0 = os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET0")
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
        dtype: torch.dtype = torch.float16,
        block_size_targets: Mapping[int, int],
        is_causal: bool = False,
        has_kv_tile_pruning: bool = False,
        requires_ws_overlap: bool = False,
        small_biased_candidate: bool = False,
        standard_dense_output: bool = False,
        standard_causal_output: bool = False,
    ) -> None:
        self.cute_flash_search_enabled = True
        self._cute_flash_head_dim = head_dim
        self._cute_flash_num_kv = num_kv
        self._cute_flash_dtype = dtype
        self._cute_flash_is_causal = is_causal
        self._cute_flash_has_kv_tile_pruning = has_kv_tile_pruning
        self._cute_flash_requires_ws_overlap = requires_ws_overlap
        self._cute_flash_small_biased_candidate = small_biased_candidate
        self._cute_flash_standard_dense_output = standard_dense_output
        self._cute_flash_standard_causal_output = standard_causal_output
        self._cute_flash_block_size_targets = dict(block_size_targets)
        for block_id, target in block_size_targets.items():
            spec = self.block_sizes.block_id_lookup(block_id)
            spec.autotuner_min = target
            spec.max_size = target

    def _pre_normalize_cute_flash_block_sizes(self, config: dict[str, object]) -> None:
        if not self.cute_flash_search_enabled or "block_sizes" not in config:
            return
        block_size_targets = self._cute_flash_block_size_target_list()
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
        input_dtype: torch.dtype,
        has_leading_passthrough: bool,
        explicit_epi_tile_compatible: bool,
    ) -> None:
        self._cute_tcgen05_config.register_mma_analysis(
            m_block_id=m_block_id,
            n_block_id=n_block_id,
            k_block_id=k_block_id,
            input_dtype=input_dtype,
            has_leading_passthrough=has_leading_passthrough,
            explicit_epi_tile_compatible=explicit_epi_tile_compatible,
        )

    def _tcgen05_matmul_block_fragments(
        self,
    ) -> tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment] | None:
        return self._cute_tcgen05_config._matmul_block_fragments()

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
            seeds.extend(
                flash_attention_seed_configs(
                    self._cute_flash_head_dim,
                    self._cute_flash_num_kv,
                    dtype=self._cute_flash_dtype,
                    is_causal=self._cute_flash_is_causal,
                    has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                    requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                    small_biased_candidate=self._cute_flash_small_biased_candidate,
                    standard_dense_output=self._cute_flash_standard_dense_output,
                    standard_causal_output=self._cute_flash_standard_causal_output,
                    block_size_targets=self._cute_flash_block_size_target_list(),
                )
            )
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
        return self.backend.supports_config_key(key)

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

    def normalize(
        self, config: helion.Config | dict[str, object], *, _fix_invalid: bool = False
    ) -> None:
        """Normalize the config to match the block_sizes and validate the config.

        Args:
            config: The config to normalize (modified in place).
            _fix_invalid: If True, silently fix invalid combinations instead of raising
                errors. Used internally during autotuning config generation.
        """
        if isinstance(config, helion.Config):
            self.normalize(config.config, _fix_invalid=_fix_invalid)
            return

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
            self._cute_tcgen05_config.prepare_normalization(
                config, fix_invalid=_fix_invalid
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

        # CuTe-specific: persistent reduction whose thread count is shrunk
        # below the reduction extent by adjust_reduction_thread_count would
        # wrap the kernel body in a synthetic lane loop. The lane loop
        # carries the body-level reduction's accumulator across iterations,
        # so the reduction result would only reflect the last lane iter.
        # Force a looped reduction whenever the available reduction threads
        # (max_reduction_threads // product_of_non_reduction_thread_axes)
        # cannot cover the full reduction extent.
        if (
            self.backend_name == "cute"
            and self.max_reduction_threads is not None
            and self.reduction_loops
        ):
            nt_list = cast("list[int]", config.get("num_threads", []) or [])
            bs_list = cast("list[int]", config.get("block_sizes", []) or [])
            other_threads = 1
            for i, _ in enumerate(self.num_threads):
                nt = nt_list[i] if i < len(nt_list) else 0
                if not isinstance(nt, int) or nt <= 0:
                    bs = bs_list[i] if i < len(bs_list) else 1
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
                        # would hit the synthetic-lane-loop bug described above.
                        # Reject the config so the autotuner skips it.
                        if available < 2:
                            raise InvalidConfig(
                                f"cute backend: reduction axis {i} has no thread "
                                f"budget left (non-reduction axes use "
                                f"{other_threads} of {self.max_reduction_threads} "
                                f"threads)."
                            )
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
            if not config.get(name):
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
        ):
            if not self.supports_config_key(name):
                config.pop(name, None)

        if self.supports_config_key("num_warps"):
            config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        if self.supports_config_key("num_stages"):
            config.setdefault("num_stages", self._default_num_stages())
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
        for key, fragment in self.backend_tunable_fragments.items():
            config.setdefault(key, fragment.default())
        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_pre_pid_type(
                config,
                fix_invalid=_fix_invalid,
            )
        if self.has_pallas_inner_loops:
            if self.has_symbolic_or_data_dependent_bounds:
                # "unroll" uses Python range() which can't handle traced bounds.
                # Between the remaining options, prefer "fori_loop": it handles
                # both DMA-aligned and unaligned inner blocks, while
                # "emit_pipeline" fails on unaligned dims.
                config.setdefault("pallas_loop_type", "fori_loop")
            else:
                config.setdefault("pallas_loop_type", VALID_PALLAS_LOOP_TYPES[0])
        if (
            self.supports_config_key("pallas_load_buffer_count")
            and self.has_pallas_inner_loops
            and config.get("pallas_loop_type") == "fori_loop"
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

        if self.supports_config_key("num_sm_multiplier"):
            # Validate num_sm_multiplier is a power of two in range
            if "num_sm_multiplier" in config:
                val = config["num_sm_multiplier"]
                if (
                    not isinstance(val, int)
                    or val < MIN_NUM_SM_MULTIPLIER
                    or val > MAX_NUM_SM_MULTIPLIER
                    or (val & (val - 1)) != 0  # not a power of two
                ):
                    raise InvalidConfig(
                        f"Invalid value for 'num_sm_multiplier': {val!r} must be a power of two between {MIN_NUM_SM_MULTIPLIER} and {MAX_NUM_SM_MULTIPLIER}"
                    )
            else:
                config["num_sm_multiplier"] = DEFAULT_NUM_SM_MULTIPLIER

        # Only validate maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            if "maxnreg" in config:
                if config["maxnreg"] not in VALID_MAXNREG:
                    raise InvalidConfig(
                        f"Invalid value for 'maxnreg': {config['maxnreg']!r} must be one of {list(VALID_MAXNREG)!r}"
                    )
            else:
                config["maxnreg"] = VALID_MAXNREG[0]

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
                            v for v in VALID_MAXNREG if v is not None and v <= limit
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

        # Allow tunable parameter keys in addition to backend-supported keys.
        allowed_keys = self.supported_config_keys() | {
            *self.user_defined_tunables.keys()
        }
        if invalid_keys := ({*config} - allowed_keys):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def raise_grid_block_minimums(self) -> None:
        """Raise min_size for grid block dimensions based on problem size.

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

        for grid_bid in self.grid_block_ids:
            try:
                spec = self.block_sizes.block_id_lookup(grid_bid)
            except KeyError:
                continue
            if spec.size_hint <= 0:
                continue
            default = spec._fragment(self).default_val
            min_block = spec.size_hint // max_blocks_per_dim
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
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        effective_overrides = overrides
        family_override: str | None = None
        if self.cute_flash_search_enabled and overrides:
            exact_family = overrides.get(FLASH_PIPELINE_FAMILY_KEY)
            if _flash_pipeline_family_flags(exact_family) is not None:
                family_override = cast("str", exact_family)
            elif FLASH_PIPELINE_FAMILY_KEY not in overrides:
                legacy = {
                    key: overrides[key]
                    for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
                    if overrides.get(key) is not None
                }
                if legacy:
                    if FLASH_PERSISTENT_KEY in overrides:
                        legacy[FLASH_PERSISTENT_KEY] = overrides[FLASH_PERSISTENT_KEY]
                    family_override = self._resolve_cute_flash_config(
                        legacy
                    ).pipeline_family
                    effective_overrides = {
                        **overrides,
                        FLASH_PIPELINE_FAMILY_KEY: family_override,
                    }
        return ConfigGeneration(
            self,
            overrides=effective_overrides,
            _flash_pipeline_family_override=family_override,
            advanced_controls_files=advanced_controls_files,
            process_group_name=process_group_name,
        )

    def flatten_missing_field_default(
        self,
        key: str,
        config: dict[str, object],
    ) -> tuple[bool, object]:
        if self.backend_name == "cute":
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
            self._cute_tcgen05_config.prepare_override_normalization(
                config,
                overrides,
            )

    def _base_default_config(self) -> helion.Config:
        config = self.flat_config(lambda x: x.default())
        self._shrink_for_numel_constraints(config)
        return config

    def default_config(self) -> helion.Config:
        if self.compiler_default_config is None:
            return self._base_default_config()
        # A promoted seed only specifies the knobs it cares about (e.g. block_sizes); layer it over
        # the full base defaults so every other key — including user register_tunable defaults — is
        # preserved rather than dropped.
        merged = dict(self._base_default_config().config)
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
            if self.cute_tcgen05_search_enabled:
                fields.update(self._cute_tcgen05_config.flat_fields())
            elif self.cute_flash_search_enabled:
                from .._compiler.cute.cute_flash import flash_autotune_fragments

                assert self._cute_flash_head_dim is not None
                assert self._cute_flash_num_kv is not None
                fields.update(
                    flash_autotune_fragments(
                        self._cute_flash_head_dim,
                        self._cute_flash_num_kv,
                        dtype=self._cute_flash_dtype,
                        is_causal=self._cute_flash_is_causal,
                        has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                        requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                        small_biased_candidate=(
                            self._cute_flash_small_biased_candidate
                        ),
                        standard_dense_output=self._cute_flash_standard_dense_output,
                        standard_causal_output=(
                            self._cute_flash_standard_causal_output
                        ),
                        pipeline_family_override=_flash_pipeline_family_override,
                    )
                )
            elif self.supports_config_key("num_threads"):
                fields["num_threads"] = self.num_threads
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
            if (
                not self.cute_flash_search_enabled
                and self.epilogue_subtile_autotune_choices is not None
            ):
                fields["epilogue_subtile"] = EnumFragment(
                    choices=self.epilogue_subtile_autotune_choices
                )
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
        if (
            self.supports_config_key("pallas_load_buffer_count")
            and self.has_pallas_inner_loops
            and self.pallas_load_buffer_count.length > 0
        ):
            fields["pallas_load_buffer_count"] = self.pallas_load_buffer_count
        if self.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
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
            if self.has_symbolic_or_data_dependent_bounds:
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
            if self.supports_config_key("pallas_pre_broadcast"):
                fields["pallas_pre_broadcast"] = BooleanFragment()
        # Only include maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            fields["maxnreg"] = EnumFragment(VALID_MAXNREG)
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
    ) -> helion.Config:
        config: dict[str, Any] = {}
        for key, field in fields.items():
            config[key] = field._flat_config(self, fn)

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
        self.normalize(config, _fix_invalid=True)
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

    def _fragment(self, base: ConfigSpec) -> NumThreadsFragment:
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
