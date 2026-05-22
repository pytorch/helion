"""Layout planning pass for the CuTe backend.

Walks the Device IR graphs and:
  1. Seeds constrained nodes (loads, stores, reductions) with preferred input
     and/or output layouts.
  2. Propagates passthrough layouts forward through unconstrained pointwise
     nodes.
  3. Propagates consumer input layouts backward so flexible producers adopt
     them.
  4. Resolves authoritative input/output layouts.
  5. Detects remaining conflicts and inserts ``_cute_layout_change`` nodes.
  6. Rejects unresolved producer-output -> consumer-input mismatches early.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import reduce_ops
from ..compile_environment import CompileEnvironment
from ..device_ir import RootGraphInfo
from .layout import CuTeGridExecutionPlan
from .layout import LayoutConstraint
from .layout import LayoutTag
from .layout import MatmulExecutionKind
from .layout import MatmulExecutionPlan
from .layout import ThreadLayout
from .layout_rules import preferred_constraint_for_node
from .matmul_utils import analyze_direct_grouped_n_loads

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch
    from .layout import SymIntLike

log = logging.getLogger(__name__)

META_KEY = "cute_layout_constraint"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_layouts(
    graphs: list[GraphInfo],
    config: Config,
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Run the full layout planning pipeline on *graphs* (mutates in place).

    This annotates every relevant node with a ``LayoutConstraint`` in
    ``node.meta["cute_layout_constraint"]`` and inserts
    ``_cute_layout_change`` nodes where needed. Any remaining mismatch on a
    producer-output -> consumer-input edge is rejected before lowering.

    Args:
        graphs: Codegen graph copies to annotate.
        config: Current autotuner configuration.
        tile_strategy: Tile strategy dispatch, used to query actual thread
            counts from strategies.
    """
    for graph_info in graphs:
        _seed_constraints(graph_info, tile_strategy)
        _plan_matmul_execution(graph_info, tile_strategy)
        _plan_warp_per_row_execution(graph_info, tile_strategy)
        _forward_propagate(graph_info)
        _backward_propagate(graph_info)
        _resolve_layouts(graph_info)
        _insert_layout_changes(graph_info)
        _validate_layout_contracts(graph_info)


# ---------------------------------------------------------------------------
# Step 1 — Seed constrained nodes
# ---------------------------------------------------------------------------


def _seed_constraints(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Attach preferred LayoutConstraints to loads, stores, reductions."""
    for node in graph_info.graph.nodes:
        constraint = preferred_constraint_for_node(node, graph_info, tile_strategy)
        if constraint is not None:
            node.meta[META_KEY] = constraint


def _plan_matmul_execution(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
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

    for node in matmul_nodes:
        constraint = node.meta.get(META_KEY)
        if (
            constraint is None
            or constraint.matmul_axes is None
            or node.op != "call_function"
            or node.target is not torch.ops.aten.mm.default
            or len(node.args) < 2
        ):
            continue
        lhs_node = node.args[0]
        rhs_node = node.args[1]
        if not isinstance(lhs_node, torch.fx.Node) or not isinstance(
            rhs_node, torch.fx.Node
        ):
            continue
        lhs_val = lhs_node.meta.get("val")
        rhs_val = rhs_node.meta.get("val")
        if not isinstance(lhs_val, torch.Tensor) or not isinstance(
            rhs_val, torch.Tensor
        ):
            continue
        if lhs_val.ndim != 2 or rhs_val.ndim != 2:
            continue
        if lhs_val.dtype not in (torch.float16, torch.bfloat16):
            continue
        scalar_block_id = _direct_grouped_n_scalar_block_id(
            tile_strategy,
            lhs_val,
            rhs_val,
            exclude_block_id=m_block_id,
        )
        if scalar_block_id is None:
            continue
        scalar_threads = tile_strategy.thread_extent_for_block_id(scalar_block_id)
        if scalar_threads is None or scalar_threads < 8 or scalar_threads % 8 != 0:
            continue
        scalar_strategy = tile_strategy.block_id_to_strategy.get((scalar_block_id,))
        if scalar_strategy is None:
            continue
        lane_extent = getattr(scalar_strategy, "_synthetic_cute_lane_extent", 1)
        if not isinstance(lane_extent, int) or lane_extent <= 0:
            lane_extent = 1
        size_hint = getattr(env, "size_hint", None)
        n_extent = (
            size_hint(rhs_val.shape[1]) if callable(size_hint) else rhs_val.shape[1]
        )
        if not isinstance(n_extent, int):
            continue
        k_extent = (
            size_hint(lhs_val.shape[1]) if callable(size_hint) else lhs_val.shape[1]
        )
        if not isinstance(k_extent, int):
            continue
        lhs_load = lhs_node if lhs_node.target is memory_ops.load else None
        rhs_load = rhs_node if rhs_node.target is memory_ops.load else None
        load_plan = (
            None
            if lhs_load is None or rhs_load is None
            else analyze_direct_grouped_n_loads(
                lhs_load,
                rhs_load,
                k_extent=k_extent,
                n_extent=n_extent,
            )
        )
        if lhs_load is None or rhs_load is None or load_plan is None:
            continue
        if scalar_threads * lane_extent < n_extent:
            continue

        constraint.matmul_plan = MatmulExecutionPlan(
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


def _plan_warp_per_row_execution(
    graph_info: GraphInfo,
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Detect the softmax-shaped warp-per-row layout.

    When the kernel has a single 1-D outer grid loop over rows (M-axis)
    and an inner non-reduction tile loop over the reduction axis (N),
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
    from ..host_function import HostFunction
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import CuteNDTileStrategy

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
    if any(
        isinstance(s, ReductionStrategy) and s.thread_axes_used() > 0
        for s in tile_strategy.strategies
    ):
        return
    # Find the M strategy (outer grid) and the N strategy (inner tile)
    # by walking ``tile_strategy.strategies``.  The M strategy owns
    # ``m_block_id``; the N strategy is any other CuteNDTileStrategy
    # with a different block_id and at least one thread axis.
    m_strategy = tile_strategy.block_id_to_strategy.get((m_block_id,))
    if not isinstance(m_strategy, CuteNDTileStrategy):
        return
    if m_strategy.thread_axes_used() != 1:
        return
    m_threads = tile_strategy.thread_extent_for_block_id(m_block_id)
    if not isinstance(m_threads, int) or m_threads < 2:
        return
    n_strategies: list[CuteNDTileStrategy] = []
    for strategy in tile_strategy.strategies:
        if strategy is m_strategy:
            continue
        if not isinstance(strategy, CuteNDTileStrategy):
            continue
        if strategy.thread_axes_used() != 1:
            continue
        if m_block_id in strategy.block_ids:
            continue
        n_strategies.append(strategy)
    if not n_strategies:
        return
    n_block_ids = {bid for s in n_strategies for bid in s.block_ids}
    if len(n_block_ids) != 1:
        # Multiple distinct inner blocks — not the simple softmax shape.
        return
    (n_block_id,) = n_block_ids
    n_threads = tile_strategy.thread_extent_for_block_id(n_block_id)
    if not isinstance(n_threads, int) or n_threads < 32 or n_threads % 32 != 0:
        return
    # Joint thread budget check.
    from .thread_budget import MAX_THREADS_PER_BLOCK

    if m_threads * n_threads > MAX_THREADS_PER_BLOCK:
        return
    graph_info.cute_grid_execution_plans = (
        *graph_info.cute_grid_execution_plans,
        CuTeGridExecutionPlan(
            scoped_block_ids=frozenset({m_block_id, n_block_id}),
            block_axis_priority={
                n_block_id: 0,
                m_block_id: 1,
            },
            disable_reduction_axis_reservation_for=frozenset({m_block_id, n_block_id}),
        ),
    )


def _direct_grouped_n_scalar_block_id(
    tile_strategy: TileStrategyDispatch,
    lhs_val: torch.Tensor,
    rhs_val: torch.Tensor,
    *,
    exclude_block_id: int,
) -> int | None:
    from ..reduction_strategy import PersistentReductionStrategy

    env = CompileEnvironment.current()

    def hinted_equal(lhs: int | torch.SymInt, rhs: int | torch.SymInt) -> bool:
        if env.known_equal(lhs, rhs):
            return True
        size_hint = getattr(env, "size_hint", None)
        if not callable(size_hint):
            return False
        hinted_lhs = size_hint(lhs)
        hinted_rhs = size_hint(rhs)
        return isinstance(hinted_lhs, int) and hinted_lhs == hinted_rhs

    candidates: list[int] = []
    for strategy in tile_strategy.strategies:
        if not isinstance(strategy, PersistentReductionStrategy):
            continue
        block_id = strategy.block_index
        if block_id == exclude_block_id:
            continue
        size = env.block_sizes[block_id].size
        if not isinstance(size, int | torch.SymInt):
            continue
        if hinted_equal(size, lhs_val.shape[1]) and hinted_equal(
            size, rhs_val.shape[1]
        ):
            candidates.append(block_id)
    if len(candidates) != 1:
        return None
    return candidates[0]


# ---------------------------------------------------------------------------
# Step 2 — Forward propagation
# ---------------------------------------------------------------------------


def _forward_propagate(graph_info: GraphInfo) -> None:
    """Passthrough tensor ops inherit layout from their first tensor input."""
    for node in graph_info.graph.nodes:
        if not _is_passthrough_layout_node(node):
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        constraint = _constraint_for_node(node)
        if constraint.preferred_output is not None:
            continue

        result = _first_input_layout_node(node)
        if result is None:
            continue
        layout, input_node = result
        # Shape-collapsing reductions (e.g. ``aten.amax``/``aten.sum``) cover
        # fewer elements than their input.  Forward inheriting the input layout
        # would describe the wrong tile, so leave the output flexible and let
        # backward propagation pick a layout that matches the reduced tile.
        if _is_reduction_target(node.target) and not _numels_match(
            _node_tile_numel(node), _node_tile_numel(input_node)
        ):
            continue
        inherited = layout.with_tag(LayoutTag.INHERITED)
        constraint.preferred_input = inherited
        constraint.preferred_output = inherited


# ---------------------------------------------------------------------------
# Step 3 — Backward propagation
# ---------------------------------------------------------------------------


def _backward_propagate(graph_info: GraphInfo) -> None:
    """If all users of a node agree on a layout, the node adopts it.

    This avoids inserting layout changes when the producer (e.g. a load)
    can cheaply produce any layout.  Nodes with semantic layout preferences
    (reductions, MMA) are not overridden — only "flexible" nodes (loads,
    pointwise) adopt backward-propagated layouts.
    """
    for node in reversed(list(graph_info.graph.nodes)):
        if not _is_output_flexible_layout_node(node):
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        constraint = _constraint_for_node(node)
        if constraint.required:
            continue  # non-negotiable

        # Don't backward-propagate through nodes with semantic preferences
        # (reductions need threads along the reduction axis).
        if (
            constraint.preferred_output is not None
            and constraint.preferred_output.tag
            in (
                LayoutTag.REDUCTION,
                LayoutTag.MMA_OPERAND_A,
                LayoutTag.MMA_OPERAND_B,
                LayoutTag.MMA_ACCUMULATOR,
            )
        ):
            continue

        user_layouts = _collect_user_layouts(node)
        if not user_layouts:
            continue

        # All users agree on the same layout?
        first = user_layouts[0]
        if all(first.is_compatible(ul) for ul in user_layouts[1:]):
            inherited = first.with_tag(LayoutTag.INHERITED)
            if _is_passthrough_layout_node(node):
                constraint.preferred_input = inherited
            constraint.preferred_output = inherited


# ---------------------------------------------------------------------------
# Step 4 — Resolve final layouts
# ---------------------------------------------------------------------------


def _resolve_layouts(graph_info: GraphInfo) -> None:
    """Resolve authoritative input/output layouts for every annotated node."""
    for node in graph_info.graph.nodes:
        constraint = node.meta.get(META_KEY)
        if constraint is None:
            continue
        if constraint.preferred_input is not None:
            constraint.input_layout = constraint.preferred_input
        if constraint.preferred_output is not None:
            constraint.output_layout = constraint.preferred_output


# ---------------------------------------------------------------------------
# Step 5 — Insert layout changes at mismatched edges
# ---------------------------------------------------------------------------


def _insert_layout_changes(graph_info: GraphInfo) -> None:
    """Where a producer and consumer disagree on layout, insert a change node."""
    from .layout_change import _cute_layout_change

    nodes = list(graph_info.graph.nodes)  # snapshot — we mutate the graph
    for node in nodes:
        producer_lc = node.meta.get(META_KEY)
        if producer_lc is None or producer_lc.output_layout is None:
            continue
        producer_layout = producer_lc.output_layout

        for user in list(node.users):
            user_lc = user.meta.get(META_KEY)
            if user_lc is None or user_lc.input_layout is None:
                continue
            consumer_layout = user_lc.input_layout

            if producer_layout.is_compatible(consumer_layout):
                continue

            # Only insert a layout change when both layouts describe the
            # same tile (same total element count).  Shape-changing ops
            # like reductions collapse dimensions, so producer and consumer
            # layouts may cover different-sized tiles — a shared-memory
            # permutation between them is meaningless.
            if not _tile_numels_match(producer_layout, consumer_layout):
                continue

            # The layout change codegen does a scalar smem round-trip (one
            # element per thread).  This only works when both layouts use the
            # same number of threads; otherwise some threads would read
            # elements that no thread wrote.
            if not _thread_counts_match(producer_layout, consumer_layout):
                continue

            # The current layout-change lowering permutes a single scalar
            # per thread — it ignores value_shape/value_stride.  Skip insertion
            # when either layout has multiple values per thread until
            # multi-value permutation is implemented.
            if not _values_are_scalar(producer_layout, consumer_layout):
                continue

            # Need a layout change between producer and consumer
            with graph_info.graph.inserting_before(user):
                change_node = graph_info.graph.call_function(
                    _cute_layout_change,
                    args=(node,),
                )
                # Propagate fake tensor metadata
                if "val" in node.meta:
                    change_node.meta["val"] = node.meta["val"]
                if "location" in node.meta:
                    change_node.meta["location"] = node.meta["location"]
                change_node.meta[META_KEY] = LayoutConstraint(
                    preferred_input=producer_layout,
                    preferred_output=consumer_layout,
                    input_layout=producer_layout,
                    output_layout=consumer_layout,
                )
                change_node.meta["cute_layout_change_src"] = producer_layout

                # Set lowering metadata so codegen can process this node.
                # This is needed because layout changes are inserted after
                # prepare_graph_lowerings() has already run.
                from ..inductor_lowering import APIFuncLowering

                APIFuncLowering.normalize_args_kwargs(_cute_layout_change, change_node)  # type: ignore[arg-type]
                change_node.meta["lowering"] = APIFuncLowering(_cute_layout_change)

                user.replace_input_with(node, change_node)
                log.debug(
                    "inserted layout change %s -> %s before %s",
                    producer_layout.tag.value,
                    consumer_layout.tag.value,
                    user.name,
                )


def _validate_layout_contracts(graph_info: GraphInfo) -> None:
    """Reject unresolved producer-output -> consumer-input mismatches."""
    for node in graph_info.graph.nodes:
        producer_lc = node.meta.get(META_KEY)
        if producer_lc is None or producer_lc.output_layout is None:
            continue
        producer_layout = producer_lc.output_layout
        for user in node.users:
            user_lc = user.meta.get(META_KEY)
            if user_lc is None or user_lc.input_layout is None:
                continue
            if user.target is reduce_ops._reduce:
                # Reduction lowering still has custom fallbacks for arbitrary
                # producer layouts, so a missed relayout here is not fatal.
                continue
            if _is_shape_reducing_user(node, user):
                # Shape-collapsing reductions (e.g. ``aten.amax``/``aten.sum``)
                # consume the producer's full tile and own the layout transition
                # to the reduced output themselves (warp/shared reduce), so a
                # producer/consumer layout difference here is expected.
                continue
            consumer_layout = user_lc.input_layout
            if (
                producer_layout.tag is LayoutTag.MMA_ACCUMULATOR
                and node.op == "call_function"
                and node.target
                in {
                    torch.ops.aten.mm.default,
                    torch.ops.aten.bmm.default,
                    torch.ops.aten.addmm.default,
                    torch.ops.aten.baddbmm.default,
                }
            ):
                # Matmul nodes may lower through a fused MMA epilogue or the
                # scalar fallback, both of which own the accumulator transition
                # directly instead of requiring an explicit relayout node.
                continue
            if producer_layout.is_compatible(consumer_layout):
                continue
            raise exc.BackendUnsupported(
                "cute",
                (
                    "unresolved CuTe layout mismatch between "
                    f"{node.name} ({producer_layout.tag.value}) and "
                    f"{user.name} ({consumer_layout.tag.value})"
                ),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_input_layout(node: torch.fx.Node) -> ThreadLayout | None:
    """Return the preferred/resolved output layout of the first input."""
    result = _first_input_layout_node(node)
    return result[0] if result is not None else None


def _first_input_layout_node(
    node: torch.fx.Node,
) -> tuple[ThreadLayout, torch.fx.Node] | None:
    """Return the first input's output layout together with that input node."""
    for inp in node.all_input_nodes:
        lc = inp.meta.get(META_KEY)
        if lc is None:
            continue
        layout = lc.output_layout or lc.preferred_output
        if layout is not None:
            return layout, inp
    return None


def _collect_user_layouts(node: torch.fx.Node) -> list[ThreadLayout]:
    """Collect preferred/resolved consumer input layouts from all users.

    A reduction user that never received a proper reduction-axis layout and
    instead picked up a *degenerate scalar* layout from its own consumer
    (covering far fewer elements than the producer's own tile) cannot legally
    describe *node*'s full tile.  Adopting such a layout would corrupt the
    producer (every thread reading a single reduced element instead of its
    slice of the full tile), so those reduction users are skipped.  A
    correctly-seeded reduction keeps a full reduction-axis input layout — at
    least as large as the producer's tile — so it is retained and continues to
    drive the producer onto the shared reduction layout.
    """
    layouts: list[ThreadLayout] = []
    for user in node.users:
        lc = user.meta.get(META_KEY)
        if lc is None:
            continue
        layout = lc.input_layout or lc.preferred_input
        if layout is None:
            continue
        if _is_reduction_target(user.target) and _reduction_layout_is_degenerate(
            node, layout
        ):
            continue
        layouts.append(layout)
    return layouts


_ATEN_REDUCTION_TARGETS = frozenset(
    {
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.sum.default,
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
        torch.ops.aten.prod.dim_int,
        torch.ops.aten.mean.dim,
        torch.ops.aten.max.dim,
        torch.ops.aten.min.dim,
    }
)


def _is_reduction_target(target: object) -> bool:
    """True if *target* is a tile reduction op (collapses one or more dims)."""
    return target is reduce_ops._reduce or target in _ATEN_REDUCTION_TARGETS


def _reduction_layout_is_degenerate(
    node: torch.fx.Node, reduction_layout: ThreadLayout
) -> bool:
    """True if a reduction user's input layout is degenerate w.r.t. *node*.

    A legitimately-seeded reduction reads the producer's full tile, so its
    input layout spans at least as many elements as the producer's own
    resolved layout.  A *degenerate* reduction (one that never got a real
    reduction-axis layout and instead inherited a scalar layout from its own
    consumer) covers strictly fewer elements than the producer's tile — that is
    the only case backward propagation must ignore.

    Comparing the two thread/value layouts (rather than the fake-tensor numels)
    keeps this provable: both layouts are built from concrete thread counts and
    block sizes, whereas a node's fake-tensor numel may be symbolic and would
    spuriously flag every reduction as degenerate.
    """
    producer_layout = _node_layout(node) or _first_input_layout(node)
    if producer_layout is None:
        return False
    return _known_lt(reduction_layout.tile_numel(), producer_layout.tile_numel())


def _node_layout(node: torch.fx.Node) -> ThreadLayout | None:
    """Return *node*'s own resolved/preferred output layout, if any."""
    lc = node.meta.get(META_KEY)
    if lc is None:
        return None
    return lc.output_layout or lc.preferred_output


def _is_shape_reducing_user(node: torch.fx.Node, user: torch.fx.Node) -> bool:
    """True if *user* is a reduction consuming a smaller tile than *node*.

    A reduction (e.g. ``aten.amax``/``aten.sum``) collapses one or more of
    *node*'s dims, so the user's tensor has fewer elements.  Such users own
    the layout transition from the producer's full tile to the reduced output
    in their own lowering, so an edge-level layout mismatch is expected.
    """
    if not _is_reduction_target(user.target):
        return False
    node_numel = _node_tile_numel(node)
    user_numel = _node_tile_numel(user)
    if node_numel is None or user_numel is None:
        return False
    if isinstance(node_numel, int) and isinstance(user_numel, int):
        return user_numel < node_numel
    # Symbolic: a reduction either preserves or shrinks the tile, so treat it
    # as reducing whenever the numels are not provably equal.
    return not _numels_match(node_numel, user_numel)


def _node_tile_numel(node: torch.fx.Node) -> SymIntLike | None:
    """Total element count of *node*'s tile, or None if unknown."""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None
    numel: SymIntLike = 1
    for dim in val.shape:
        numel = numel * dim  # type: ignore[operator]
    return numel


def _numels_match(a: SymIntLike | None, b: SymIntLike | None) -> bool:
    """Conservatively compare two (possibly symbolic) element counts.

    Returns True only when the two counts are provably equal.  When either
    side is unknown, fall back to True so callers keep their prior behaviour.
    Symbolic comparison goes through ``known_equal`` (static evaluation) so it
    never installs a guard or trips a value-range assertion.
    """
    if a is None or b is None:
        return True
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return CompileEnvironment.current().known_equal(a, b)


def _known_lt(a: SymIntLike, b: SymIntLike) -> bool:
    """True only when *a* is provably strictly less than *b*.

    Layout numels are concrete integers in practice; the symbolic branch falls
    back to ``False`` (not provably smaller -> not degenerate) so the caller
    never drops a legitimate reduction layout on an unprovable comparison.
    """
    if isinstance(a, int) and isinstance(b, int):
        return a < b
    return False


def _constraint_for_node(node: torch.fx.Node) -> LayoutConstraint:
    constraint = node.meta.get(META_KEY)
    if constraint is None:
        constraint = LayoutConstraint()
        node.meta[META_KEY] = constraint
    return constraint


def _is_passthrough_layout_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target is memory_ops.load or node.target is memory_ops.store:
        return False
    if node.target is reduce_ops._reduce:
        return False
    return not _tracing_ops.is_for_loop_target(node.target)


def _is_output_flexible_layout_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target is memory_ops.store or node.target is reduce_ops._reduce:
        return False
    return not _tracing_ops.is_for_loop_target(node.target)


def _tile_numels_match(a: ThreadLayout, b: ThreadLayout) -> bool:
    """Return True if both layouts cover the same number of tile elements.

    A layout change (shared-memory permutation) only makes sense when the
    producer and consumer operate on the same tile.  Shape-changing ops
    (e.g. reductions) produce outputs with fewer elements, so the
    producer's tile_numel differs from the consumer's.
    """
    na, nb = a.tile_numel(), b.tile_numel()
    if isinstance(na, int) and isinstance(nb, int):
        return na == nb
    # Symbolic comparison — conservative: only match when provably equal.
    try:
        return bool(na == nb)
    except (TypeError, RuntimeError):
        return False


def _values_are_scalar(a: ThreadLayout, b: ThreadLayout) -> bool:
    """Return True if both layouts have exactly one value per thread.

    Multi-value layouts require a loop over value indices to permute all
    elements; the current codegen only handles the single-scalar case.
    """
    na, nb = a.num_values(), b.num_values()
    if isinstance(na, int) and isinstance(nb, int):
        return na == 1 and nb == 1
    # Symbolic — conservatively reject.
    return False


def _thread_counts_match(a: ThreadLayout, b: ThreadLayout) -> bool:
    """Return True if both layouts use the same number of threads.

    The scalar layout-change codegen writes/reads one element per thread,
    so it requires every writing thread to have a corresponding reading
    thread (and vice-versa).
    """
    na, nb = a.num_threads(), b.num_threads()
    if isinstance(na, int) and isinstance(nb, int):
        return na == nb
    try:
        return bool(na == nb)
    except (TypeError, RuntimeError):
        return False
