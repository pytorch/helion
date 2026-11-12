"""Thread-axis execution plans for CuTe root kernels.

``plan_execution`` runs four planners over each root graph once per config.
Each one recognizes a kernel shape from the graph and its tile strategies and
records a :class:`CuTeGridExecutionPlan` on the graph, which
``TileStrategyDispatch`` and ``PerThreadNDTileStrategy`` read to order the
launch's thread axes and to skip the reduction-axis reservation:

* ``_plan_matmul_execution``: a dedicated single-``mm`` root whose M block
  owns one thread per row and whose scalar reduction block spans N in groups
  of eight.  Also records a :class:`MatmulExecutionPlan` on the ``mm`` node
  (``node_matmul_plan``) so the direct grouped-N register MMA lowering fires.
* ``_plan_warp_per_row_execution``: the softmax shape, one warp per row.
* ``_plan_register_tile_execution``: a register-tile reduction lane whose
  vectorized tile blocks take the lowest thread axes.
* ``_plan_vloop_sink_reduction_execution``: a reduction beside a vectorized
  grid tile that ``cute_vloop_sink`` sinks into it; the tile takes the
  lowest thread axes.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import memory_ops
from ...language import reduce_ops
from ..compile_environment import CompileEnvironment
from .matmul_utils import analyze_direct_grouped_n_loads

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from ..device_ir import GraphInfo
    from ..device_ir import RootGraphInfo
    from ..reduction_strategy import PersistentReductionStrategy
    from ..tile_dispatch import TileStrategyDispatch
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy
    from ..tile_strategy import TileStrategy

MATMUL_PLAN_META_KEY = "cute_matmul_plan"


class MatmulExecutionKind(enum.Enum):
    """Planner-selected CuTe execution scheme for a matmul node."""

    DIRECT_GROUPED_N = "direct_grouped_n"


@dataclass(frozen=True)
class MatmulExecutionPlan:
    """Planner-owned execution layout for a CuTe matmul node."""

    kind: MatmulExecutionKind
    m_block_id: int
    scalar_block_id: int
    bm: int
    bn: int
    bk: int
    groups_per_lane: int
    lane_extent: int


@dataclass(frozen=True)
class CuTeGridExecutionPlan:
    """Branch-scoped CuTe thread-axis policy chosen by execution planning."""

    scoped_block_ids: frozenset[int]
    block_axis_priority: dict[int, int]
    disable_reduction_axis_reservation_for: frozenset[int]

    def applies_to_any_block(self, block_ids: tuple[int, ...] | list[int]) -> bool:
        return any(block_id in self.scoped_block_ids for block_id in block_ids)

    def priority_for_block(self, block_id: int) -> int | None:
        if block_id not in self.scoped_block_ids:
            return None
        return self.block_axis_priority.get(block_id)

    def disables_reduction_axis_reservation(self, block_id: int) -> bool:
        return (
            block_id in self.scoped_block_ids
            and block_id in self.disable_reduction_axis_reservation_for
        )

    def merged_with(
        self, others: Iterable[CuTeGridExecutionPlan]
    ) -> CuTeGridExecutionPlan:
        """The union of this plan and ``others`` (the plans that apply to one
        loop): every scoped block, each block's lowest priority and every
        reservation veto."""
        scoped_block_ids = set(self.scoped_block_ids)
        block_axis_priority = dict(self.block_axis_priority)
        disable_reduction_axis_reservation_for = set(
            self.disable_reduction_axis_reservation_for
        )
        for plan in others:
            scoped_block_ids.update(plan.scoped_block_ids)
            disable_reduction_axis_reservation_for.update(
                plan.disable_reduction_axis_reservation_for
            )
            for block_id, priority in plan.block_axis_priority.items():
                previous = block_axis_priority.get(block_id)
                if previous is None or priority < previous:
                    block_axis_priority[block_id] = priority
        return CuTeGridExecutionPlan(
            scoped_block_ids=frozenset(scoped_block_ids),
            block_axis_priority=block_axis_priority,
            disable_reduction_axis_reservation_for=frozenset(
                disable_reduction_axis_reservation_for
            ),
        )


def node_matmul_plan(node: torch.fx.Node | None) -> MatmulExecutionPlan | None:
    """The planner-owned CuTe matmul execution plan recorded on an FX node."""
    if node is None:
        return None
    return node.meta.get(MATMUL_PLAN_META_KEY)


def plan_execution(
    graphs: list[GraphInfo],
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Record the execution plans of every root graph in *graphs* (in place)."""
    from .state import cute_env_state

    # Per pass: ``CuteRegeneration`` reads it after a refusal.
    env_state = cute_env_state(CompileEnvironment.current())
    env_state.vloop_sink_reduction_layout_planned = False
    for graph_info in graphs:
        _plan_matmul_execution(graph_info, tile_strategy)
        _plan_warp_per_row_execution(graph_info, tile_strategy)
        _plan_register_tile_execution(graph_info, tile_strategy)
        _plan_vloop_sink_reduction_execution(graph_info, tile_strategy)


def _plan_matmul_execution(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    if not isinstance(graph_info, RootGraphInfo):
        return
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 1:
        return
    (m_block_id,) = device_ir.grid_block_ids[0]
    env = CompileEnvironment.current()
    config = tile_strategy.strategies[0].fn.config
    m_block_size = env.block_sizes[m_block_id].from_config(config)
    if not isinstance(m_block_size, int) or m_block_size <= 0 or m_block_size % 16 != 0:
        return
    m_threads = tile_strategy.thread_extent_for_block_id(m_block_id)
    if not isinstance(m_threads, int) or m_threads != m_block_size:
        return
    # The direct grouped-N MMA path currently owns the root-kernel thread-axis
    # mapping, so only enable it for a dedicated single-mm root graph.
    matmul_nodes = [
        node
        for node in graph_info.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.mm.default
    ]
    if len(matmul_nodes) != 1:
        return
    if any(
        node.op == "call_function" and node.target is reduce_ops._reduce
        for node in graph_info.graph.nodes
    ):
        return
    (node,) = matmul_nodes
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor) or val.ndim < 2 or len(node.args) < 2:
        return
    lhs_node = node.args[0]
    rhs_node = node.args[1]
    if not isinstance(lhs_node, torch.fx.Node) or not isinstance(
        rhs_node, torch.fx.Node
    ):
        return
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return
    if lhs_val.ndim != 2 or rhs_val.ndim != 2:
        return
    if lhs_val.dtype not in (torch.float16, torch.bfloat16):
        return
    scalar_strategy = _direct_grouped_n_scalar_strategy(
        tile_strategy,
        lhs_val,
        rhs_val,
        exclude_block_id=m_block_id,
    )
    if scalar_strategy is None:
        return
    # The warp-MMA tiles give lane ``l`` of thread ``t`` column
    # ``l * threads + t``, the cyclic lanes; a row of V-wide vectors holds
    # element ``l % V`` of its chunk ``l // V`` there instead, so its stores
    # would take other columns.
    if scalar_strategy._cute_reduction_vec_width > 1:
        return
    scalar_block_id = scalar_strategy.block_index
    scalar_threads = tile_strategy.thread_extent_for_block_id(scalar_block_id)
    if scalar_threads is None or scalar_threads < 8 or scalar_threads % 8 != 0:
        return
    lane_extent = scalar_strategy._synthetic_cute_lane_extent
    n_extent = env.size_hint(rhs_val.shape[1])
    k_extent = env.size_hint(lhs_val.shape[1])
    if lhs_node.target is not memory_ops.load or rhs_node.target is not memory_ops.load:
        return
    load_plan = analyze_direct_grouped_n_loads(
        lhs_node,
        rhs_node,
        k_extent=k_extent,
        n_extent=n_extent,
    )
    if load_plan is None:
        return
    if scalar_threads * lane_extent < n_extent:
        return

    node.meta[MATMUL_PLAN_META_KEY] = MatmulExecutionPlan(
        kind=MatmulExecutionKind.DIRECT_GROUPED_N,
        m_block_id=m_block_id,
        scalar_block_id=scalar_block_id,
        bm=m_block_size,
        bn=8,
        bk=16,
        groups_per_lane=scalar_threads // 8,
        lane_extent=lane_extent,
    )
    graph_info.cute_grid_execution_plans = (
        *graph_info.cute_grid_execution_plans,
        CuTeGridExecutionPlan(
            scoped_block_ids=frozenset({m_block_id, scalar_block_id}),
            block_axis_priority={
                m_block_id: 0,
                scalar_block_id: 1,
            },
            disable_reduction_axis_reservation_for=frozenset(
                {m_block_id, scalar_block_id}
            ),
        ),
    )


def _direct_grouped_n_scalar_strategy(
    tile_strategy: TileStrategyDispatch,
    lhs_val: torch.Tensor,
    rhs_val: torch.Tensor,
    *,
    exclude_block_id: int,
) -> PersistentReductionStrategy | None:
    """The one persistent reduction whose block spans K of *lhs* and N of *rhs*."""
    from ..reduction_strategy import PersistentReductionStrategy

    env = CompileEnvironment.current()

    def hinted_equal(lhs: int | torch.SymInt, rhs: int | torch.SymInt) -> bool:
        return env.known_equal(lhs, rhs) or env.size_hint(lhs) == env.size_hint(rhs)

    candidates: list[PersistentReductionStrategy] = []
    for strategy in tile_strategy.strategies:
        if not isinstance(strategy, PersistentReductionStrategy):
            continue
        if strategy.block_index == exclude_block_id:
            continue
        size = env.block_sizes[strategy.block_index].size
        if not isinstance(size, int | torch.SymInt):
            continue
        if hinted_equal(size, lhs_val.shape[1]) and hinted_equal(
            size, rhs_val.shape[1]
        ):
            candidates.append(strategy)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _plan_warp_per_row_execution(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Detect the softmax-shaped warp-per-row layout.

    When the kernel has a single 1-D outer grid loop over rows (M-axis)
    and inner non-reduction tile passes over the same reduction axis (N),
    and the autotuner picks ``num_threads`` such that:

      * M-block (outer grid) has a thread extent >= 2 (multi-row CTAs)
      * N-tile (inner) has a thread extent >= 32 and is a multiple of 32
      * Joint thread extent (M_threads * N_threads) <= 1024

    we want each CUDA warp to own ONE row so the inner reduction can
    use ``cute.arch.warp_reduction_*`` (single warp shuffle) rather
    than the cross-warp shared-memory two-stage reduce. That requires
    swapping the thread-axis assignment so:

      * N (inner) lands on thread_idx[0] (32 contiguous threads per warp)
      * M (outer grid) lands on thread_idx[1] (warp index = row index)

    This emits a ``CuTeGridExecutionPlan`` that sets
    ``block_axis_priority`` to put N before M and disables the
    reduction-axis reservation (no reduction strategy claims a thread
    axis here — the strided reduction inside the inner tile body picks
    the warp-reduce path because ``group_span = N_threads = 32`` once
    M has moved to a higher-indexed thread axis).
    """
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    if not isinstance(graph_info, RootGraphInfo):
        return
    device_ir = HostFunction.current().device_ir
    # Only one outer grid (1 M-block).
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 1:
        return
    (m_block_id,) = device_ir.grid_block_ids[0]
    # No MMA — those use a different plan.
    matmul_nodes = [
        node
        for node in graph_info.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.mm.default
    ]
    if matmul_nodes:
        return
    # Don't conflict with a plan already attached for this graph (e.g.
    # the matmul plan above).
    if graph_info.cute_grid_execution_plans:
        return
    # No reduction strategy (rolled reduction) — that path runs the
    # CuteTileVecWarpReduceHeuristic-style single-row layout already.
    # A single-thread reduction claims no axis (``_claims_thread_axis``) but
    # keeps this gate as before, so the plan's domain is unchanged.
    if any(
        isinstance(s, ReductionStrategy) and s._reduction_thread_count() > 0
        for s in tile_strategy.strategies
    ):
        return
    # Find the M strategy (outer grid) and the N strategy (inner tile)
    # by walking ``tile_strategy.strategies``.  The M strategy owns
    # ``m_block_id``; the N strategy is any other PerThreadNDTileStrategy
    # with a different block_id and at least one thread axis.
    m_strategy = tile_strategy.block_id_to_strategy.get((m_block_id,))
    if not isinstance(m_strategy, PerThreadNDTileStrategy):
        return
    if m_strategy.thread_axes_used() != 1:
        return
    if (
        m_strategy.fn.config.config.get("cute_vloop_sink") is True
        and m_strategy.lane_context(m_block_id).vec_width > 1
    ):
        # Vector-loop sinking vectorizes the OUTER grid axis (a column sum
        # loads V contiguous columns per row): adjacent column chunks must
        # sit in adjacent threads, so the grid axis keeps thread_idx[0] and
        # the inner reduced axis stays above it.  Record that the knob shaped
        # the layout: should no V-loop be sunk after all, codegen restarts
        # with the knob off so the knob alone never changes the code.
        m_strategy.fn.cute_state.lanes.vloop_sink_shaped_code = True
        return
    m_threads = tile_strategy.thread_extent_for_block_id(m_block_id)
    if not isinstance(m_threads, int) or m_threads < 2:
        return
    n_strategies: list[PerThreadNDTileStrategy] = []
    for strategy in tile_strategy.strategies:
        if strategy is m_strategy:
            continue
        if not isinstance(strategy, PerThreadNDTileStrategy):
            continue
        if strategy.thread_axes_used() != 1:
            continue
        if m_block_id in strategy.block_ids:
            continue
        n_strategies.append(strategy)
    if not n_strategies:
        return
    # The row coordinate is emitted once outside all sibling passes. Reserving
    # a lower N axis only in some branches gives that shared row strategy two
    # different offsets; the first branch then silently determines both. A
    # scalar sibling has no N axis to reserve, so keep the ordinary row-first
    # layout unless a threaded N strategy covers every branch of this grid.
    if not tile_strategy.strategies_cover_branches(m_strategy, n_strategies):
        return
    n_block_ids = {bid for s in n_strategies for bid in s.block_ids}
    if len(n_block_ids) != 1:
        from .loop_nesting import sibling_row_loop_blocks

        sibling_blocks = sibling_row_loop_blocks(
            CompileEnvironment.current(),
            device_ir,
            m_strategy.fn.codegen.codegen_graphs,
        )
        if sibling_blocks is None or set(sibling_blocks[1]) != n_block_ids:
            return
    thread_extents = {
        tile_strategy.thread_extent_for_block_id(block_id) for block_id in n_block_ids
    }
    if len(thread_extents) != 1:
        return
    (n_threads,) = thread_extents
    if not isinstance(n_threads, int) or n_threads < 32 or n_threads % 32 != 0:
        return
    # Joint thread budget check.
    from .thread_budget import MAX_THREADS_PER_BLOCK

    if m_threads * n_threads > MAX_THREADS_PER_BLOCK:
        return
    scoped_block_ids = frozenset({m_block_id, *n_block_ids})
    graph_info.cute_grid_execution_plans = (
        *graph_info.cute_grid_execution_plans,
        CuTeGridExecutionPlan(
            scoped_block_ids=scoped_block_ids,
            block_axis_priority={
                **dict.fromkeys(n_block_ids, 0),
                m_block_id: 1,
            },
            disable_reduction_axis_reservation_for=scoped_block_ids,
        ),
    )


def _plan_register_tile_execution(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Lay the vectorized tile axes below a register-tile reduction lane.

    A persistent reduction whose scalar synthetic lane nests outside the
    one-vector tile wrappers (``_cute_register_tile_predicted``) gives every
    thread V contiguous elements of the stride-1 tile axis per lane.  With the
    reduction threads on ``thread_idx[0]`` (the CuTe default, so a warp
    shuffle can combine them) the 32 lanes of a warp hold 32 different rows,
    and every V-wide load instruction touches 32 cache lines for 16 bytes
    each: the L1/TEX wavefront rate, not DRAM, bounds the kernel.  Putting the
    vectorized tile blocks on the lowest thread axes makes consecutive lanes
    cover consecutive fragments of one row (``T * V`` contiguous elements per
    warp row), so each instruction touches a few full lines.  The reduction
    threads then sit above ``pre`` sibling coordinates and the register-tile
    finalize combines every tile element with the strided shared-memory
    column reduce (``_cute_grouped_reduce_shared_columns``).

    Records the tile-first plan of ``_record_tile_first_plan``.
    """
    from ..device_ir import RootGraphInfo
    from ..reduction_strategy import PersistentReductionStrategy
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import PerThreadFlattenedTileStrategy

    if not isinstance(graph_info, RootGraphInfo):
        return
    if graph_info.cute_grid_execution_plans:
        return
    # Counts single-thread reductions too (they claim no axis, see
    # ``ReductionStrategy._claims_thread_axis``) so the plan's domain is
    # unchanged.
    reductions = [
        strategy
        for strategy in tile_strategy.strategies
        if isinstance(strategy, ReductionStrategy)
        and strategy._reduction_thread_count() > 0
    ]
    if len(reductions) != 1:
        return
    reduction = reductions[0]
    if (
        not isinstance(reduction, PersistentReductionStrategy)
        or not reduction._cute_register_tile_predicted
    ):
        return

    def vectorized(
        strategy: PerThreadNDTileStrategy | PerThreadFlattenedTileStrategy,
        block_id: int,
    ) -> bool:
        if isinstance(strategy, PerThreadFlattenedTileStrategy):
            return strategy._lane_var is not None
        return block_id in strategy._lane_var_by_block

    _record_tile_first_plan(
        graph_info,
        _tile_thread_axis_blocks(tile_strategy, vectorized),
        reduction.block_index,
    )


class VloopSinkLayoutRefused(exc.BackendUnsupported):
    """Codegen refused the tile-first layout ``cute_vloop_sink`` laid a
    reduction out in (``_plan_vloop_sink_reduction_execution``): a lane
    combine the tile threads interleaved below the reduction's axis defeat.
    An argreduce's finalize folds consecutive lanes; a sibling-axis combine
    (``_cute_sibling_axis_reduction_expr``) refuses a cluster split, another
    block on the reduce axis, thread axes that do not tile its group, a group
    the launch does not prove, and a missing linear thread index.

    The kernel is generated again with the default layout and the knob still
    on (``CuteEnvState.vloop_sink_reduction_layout_disabled``): by
    ``generate_ast`` for the codegen graphs it builds, and by a caller that
    supplied its own graphs (materialized fission), whose execution plans
    record the layout and which therefore have to be rebuilt.
    """

    @staticmethod
    def refusal_type(planned: bool) -> type[exc.BackendUnsupported]:
        """What a lane combine raises when it cannot combine its lanes: this
        exception under the layout ``cute_vloop_sink`` planned (``planned``,
        ``vloop_sink_reduction_layout_planned``), which regenerates the
        kernel with the default layout, else ``BackendUnsupported``."""
        return VloopSinkLayoutRefused if planned else exc.BackendUnsupported


def _plan_vloop_sink_reduction_execution(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Lay a vectorized grid tile below a threaded rolled reduction it encloses.

    ``cute_vloop_sink`` interchanges a vectorized grid axis's constexpr
    V-loop into the loops of its body, so every lane of a reduction loads V
    contiguous elements of a row at once (a channels-last spatial mean:
    ``x[tile_n, :, tile_c]`` reduced over the middle dim).  With the
    reduction threads on ``thread_idx[0]`` (the CuTe default, for warp
    shuffles) the lanes of a warp sit on different rows, ``C`` elements
    apart, and every V-wide load instruction touches as many cache lines as
    the warp has reduction lanes.  As for the column sums of
    ``_plan_warp_per_row_execution``, adjacent column chunks must sit in
    adjacent threads: the tile-first plan of ``_record_tile_first_plan``
    puts the vectorized grid blocks on the lowest thread axes, and the
    reduction combines its lanes across the interleaved tile threads (the
    grouped shared-memory reduce of the looped emitter).

    Records that the knob shaped the layout, so codegen restarts with it off
    when no V-loop is sunk after all.  Keeps the default layout when the
    tile-first order would put an axis past its launch limit (128 reduction
    threads on z); a lane combine the interleaved threads defeat (the
    argreduce finalize, or a sibling-axis combine refusal) refuses the
    layout in codegen (``VloopSinkLayoutRefused``), and the kernel is
    generated again with the default layout.
    """
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction
    from ..reduction_strategy import LoopedReductionStrategy
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import PerThreadNDTileStrategy
    from .state import cute_env_state
    from .thread_budget import MAX_THREAD_BLOCK_DIMS

    if not isinstance(graph_info, RootGraphInfo):
        return
    # One grid, as for ``_plan_warp_per_row_execution``: the plan lays out the
    # strategies of every root, and would rank another root's blocks.
    if len(HostFunction.current().device_ir.grid_block_ids) != 1:
        return
    if graph_info.cute_grid_execution_plans:
        return
    env_state = cute_env_state(CompileEnvironment.current())
    if env_state.vloop_sink_reduction_layout_disabled:
        return
    reductions = [
        strategy
        for strategy in tile_strategy.strategies
        if isinstance(strategy, ReductionStrategy)
        and strategy._reduction_thread_count() > 1
    ]
    # A rolled reduction combines its lanes across the interleaved tile
    # threads in shared memory; a persistent one keeps its warp shuffles.
    if len(reductions) != 1 or not isinstance(reductions[0], LoopedReductionStrategy):
        return
    (reduction,) = reductions
    if reduction.fn.config.config.get("cute_vloop_sink") is not True:
        return
    tiles = [
        strategy
        for strategy in tile_strategy.strategies
        if not isinstance(strategy, ReductionStrategy)
    ]
    if not tiles or not all(
        isinstance(strategy, PerThreadNDTileStrategy) and not strategy.mma_mode
        for strategy in tiles
    ):
        return
    grid_strategy = tiles[0]
    assert isinstance(grid_strategy, PerThreadNDTileStrategy)

    def vectorized(
        strategy: PerThreadNDTileStrategy | PerThreadFlattenedTileStrategy,
        block_id: int,
    ) -> bool:
        return (
            strategy is grid_strategy
            and block_id in grid_strategy._lane_var_by_block
            and grid_strategy.lane_context(block_id).vec_width > 1
        )

    blocks = _tile_thread_axis_blocks(tile_strategy, vectorized)
    # Every strategy but the reduction is a per-thread tile (checked above).
    assert blocks is not None
    vector_blocks, other_blocks = blocks
    # The dispatcher orders whole strategies by the plan, and the grid
    # strategy numbers its own threaded blocks in loop order
    # (``_thread_axis_map``): the vectors take the lowest axes only when its
    # loop order reaches them before its other threaded blocks.  Otherwise
    # keep the default layout rather than put rows below them (blocks
    # [32, 64] on (32, 8) threads would launch (32, 8, 2), each warp on 32
    # images).
    threaded = [
        block_id
        for block_id in _axis_ordered_blocks(grid_strategy)
        if block_id in vector_blocks or block_id in other_blocks
    ]
    if threaded[: len(vector_blocks)] != [b for b in threaded if b in vector_blocks]:
        return
    # The plan orders the axes as listed, the reduction last: the vectors,
    # then the other threaded blocks, whose order only matters on z, where
    # the reduction sits when three axes are threaded.
    order = [*vector_blocks, *other_blocks, reduction.block_index]
    extents = [tile_strategy.thread_extent_for_block_id(b) for b in order]
    if len(order) > len(MAX_THREAD_BLOCK_DIMS) or any(
        extent is None or extent > limit
        for extent, limit in zip(extents, MAX_THREAD_BLOCK_DIMS, strict=False)
    ):
        return
    if _record_tile_first_plan(graph_info, blocks, reduction.block_index):
        grid_strategy.fn.cute_state.lanes.vloop_sink_shaped_code = True
        env_state.vloop_sink_reduction_layout_planned = True


def _tile_thread_axis_blocks(
    tile_strategy: TileStrategyDispatch,
    vectorized: Callable[
        [PerThreadNDTileStrategy | PerThreadFlattenedTileStrategy, int], bool
    ],
) -> tuple[list[int], list[int]] | None:
    """The tile blocks that take a thread axis, split into the vectorized
    ones (``vectorized(strategy, block_id)``) and the others; None when a
    strategy other than a reduction is not a per-thread tile.  A flattened
    strategy's one axis carries all of its blocks, judged by its last."""
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    vector_blocks: list[int] = []
    other_blocks: list[int] = []
    for strategy in tile_strategy.strategies:
        if isinstance(strategy, ReductionStrategy):
            continue
        if isinstance(strategy, PerThreadFlattenedTileStrategy):
            if strategy._uses_thread_axis():
                target = (
                    vector_blocks
                    if vectorized(strategy, strategy.block_ids[-1])
                    else other_blocks
                )
                target.extend(strategy.block_ids)
            continue
        if not isinstance(strategy, PerThreadNDTileStrategy):
            return None
        assert isinstance(strategy.block_size, list)
        for block_id, block_size in zip(
            strategy.block_ids, strategy.block_size, strict=True
        ):
            if strategy._uses_thread_axis_for_block(block_id, block_size):
                target = (
                    vector_blocks if vectorized(strategy, block_id) else other_blocks
                )
                target.append(block_id)
    return vector_blocks, other_blocks


def _axis_ordered_blocks(strategy: TileStrategy) -> list[int]:
    """``strategy``'s blocks in the order it numbers their thread axes: loop
    order (``_thread_axis_map``); a flattened strategy's one axis carries them
    all."""
    from ..tile_strategy import PerThreadNDTileStrategy

    if isinstance(strategy, PerThreadNDTileStrategy):
        return [strategy.block_ids[i] for i in strategy.loop_order]
    return list(strategy.block_ids)


def _record_tile_first_plan(
    graph_info: RootGraphInfo,
    blocks: tuple[list[int], list[int]] | None,
    reduction_block_id: int,
) -> bool:
    """Record a ``CuTeGridExecutionPlan`` ranking the vectorized tile blocks
    of ``blocks`` (``_tile_thread_axis_blocks``) first, the other threaded
    tile blocks next and the reduction last, with the reduction-axis
    reservation disabled for the tile blocks.  False, recording nothing,
    without a vectorized block."""
    if blocks is None or not blocks[0]:
        return False
    vector_blocks, other_blocks = blocks
    priority = {
        **dict.fromkeys(vector_blocks, 0),
        **dict.fromkeys(other_blocks, 1),
        reduction_block_id: 2,
    }
    graph_info.cute_grid_execution_plans = (
        *graph_info.cute_grid_execution_plans,
        CuTeGridExecutionPlan(
            scoped_block_ids=frozenset(priority),
            block_axis_priority=priority,
            disable_reduction_axis_reservation_for=frozenset(
                [*vector_blocks, *other_blocks]
            ),
        ),
    )
    return True
