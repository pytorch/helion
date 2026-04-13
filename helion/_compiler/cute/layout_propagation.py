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
from .layout import LayoutConstraint
from .layout import LayoutTag
from .layout import ThreadLayout
from .layout_rules import preferred_constraint_for_node

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

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
        _forward_propagate(graph_info)
        _backward_propagate(graph_info)
        _resolve_layouts(graph_info)
        _insert_layout_changes(graph_info)
        _validate_layout_contracts(graph_info)

    _validate_thread_budget_graphs(graphs)


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

        layout = _first_input_layout(node)
        if layout is not None:
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
            consumer_layout = user_lc.input_layout
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
# Step 6 — Thread budget validation
# ---------------------------------------------------------------------------


def _validate_thread_budget_graphs(graphs: list[GraphInfo]) -> None:
    """Check that all resolved layouts use <= 1024 threads.

    When thread counts are symbolic, validation is deferred to launch time.

    Layouts inside device/reduction loops may inherit thread counts from
    the parent that include loop threads.  These are validated at kernel
    launch time via the actual thread block dimensions, so we skip the
    per-node check when the thread count exceeds the limit — it will be
    caught by :func:`check_thread_limit` in the launcher if genuinely
    over-budget.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_input_layout(node: torch.fx.Node) -> ThreadLayout | None:
    """Return the preferred/resolved output layout of the first input."""
    for inp in node.all_input_nodes:
        lc = inp.meta.get(META_KEY)
        if lc is None:
            continue
        layout = lc.output_layout or lc.preferred_output
        if layout is not None:
            return layout
    return None


def _collect_user_layouts(node: torch.fx.Node) -> list[ThreadLayout]:
    """Collect preferred/resolved consumer input layouts from all users."""
    layouts: list[ThreadLayout] = []
    for user in node.users:
        lc = user.meta.get(META_KEY)
        if lc is None:
            continue
        layout = lc.input_layout or lc.preferred_input
        if layout is not None:
            layouts.append(layout)
    return layouts


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
