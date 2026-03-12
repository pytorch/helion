"""Layout propagation pass for the CuTe backend.

Walks the Device IR graphs and:
  1. Seeds constrained nodes (loads, stores, reductions) with preferred layouts.
  2. Propagates layouts forward through unconstrained (pointwise) nodes.
  3. Propagates layouts backward so producers adopt consumers' layouts.
  4. Detects remaining conflicts and inserts ``_cute_layout_change`` nodes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

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
    ``_cute_layout_change`` nodes where needed.

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
    """Unconstrained nodes inherit layout from their first tensor input."""
    for node in graph_info.graph.nodes:
        if node.op != "call_function":
            continue
        if META_KEY in node.meta:
            continue  # already has a constraint

        # Check if this node produces a tensor
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue

        # Try to inherit from the first input that has a layout
        layout = _first_input_layout(node)
        if layout is not None:
            node.meta[META_KEY] = LayoutConstraint(
                preferred=layout.with_tag(LayoutTag.INHERITED),
            )


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
        if node.op != "call_function":
            continue
        constraint = node.meta.get(META_KEY)
        if constraint is not None and constraint.required:
            continue  # non-negotiable

        # Don't backward-propagate through nodes with semantic preferences
        # (reductions need threads along the reduction axis).
        if constraint is not None and constraint.preferred is not None:
            if constraint.preferred.tag in (
                LayoutTag.REDUCTION,
                LayoutTag.MMA_OPERAND_A,
                LayoutTag.MMA_OPERAND_B,
                LayoutTag.MMA_ACCUMULATOR,
            ):
                continue

        user_layouts = _collect_user_layouts(node)
        if not user_layouts:
            continue

        # All users agree on the same layout?
        first = user_layouts[0]
        if all(first.is_compatible(ul) for ul in user_layouts[1:]):
            # Adopt the users' layout if we don't have a semantic constraint
            inherited = first.with_tag(LayoutTag.INHERITED)
            if constraint is None:
                node.meta[META_KEY] = LayoutConstraint(preferred=inherited)
            else:
                constraint.preferred = inherited


# ---------------------------------------------------------------------------
# Step 4 — Resolve final layouts
# ---------------------------------------------------------------------------


def _resolve_layouts(graph_info: GraphInfo) -> None:
    """Copy ``preferred`` into ``layout`` for every annotated node."""
    for node in graph_info.graph.nodes:
        constraint = node.meta.get(META_KEY)
        if constraint is not None and constraint.preferred is not None:
            constraint.layout = constraint.preferred


# ---------------------------------------------------------------------------
# Step 5 — Insert layout changes at mismatched edges
# ---------------------------------------------------------------------------


def _insert_layout_changes(graph_info: GraphInfo) -> None:
    """Where a producer and consumer disagree on layout, insert a change node."""
    from .layout_change import _cute_layout_change

    nodes = list(graph_info.graph.nodes)  # snapshot — we mutate the graph
    for node in nodes:
        producer_lc = node.meta.get(META_KEY)
        if producer_lc is None or producer_lc.layout is None:
            continue
        producer_layout = producer_lc.layout

        for user in list(node.users):
            user_lc = user.meta.get(META_KEY)
            if user_lc is None or user_lc.layout is None:
                continue
            consumer_layout = user_lc.layout

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
                    preferred=consumer_layout,
                    layout=consumer_layout,
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


# ---------------------------------------------------------------------------
# Step 6 — Thread budget validation
# ---------------------------------------------------------------------------


def _validate_thread_budget_graphs(graphs: list[GraphInfo]) -> None:
    """Check that all resolved layouts use <= 1024 threads.

    When thread counts are symbolic, validation is deferred to launch time.
    """
    from .thread_budget import check_thread_limit

    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            constraint = node.meta.get(META_KEY)
            if constraint is None or constraint.layout is None:
                continue
            nt = constraint.layout.num_threads()
            if isinstance(nt, int):
                check_thread_limit(nt, context=node.name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_input_layout(node: torch.fx.Node) -> ThreadLayout | None:
    """Return the resolved (or preferred) layout of node's first annotated input."""
    for inp in node.all_input_nodes:
        lc = inp.meta.get(META_KEY)
        if lc is not None:
            return lc.layout or lc.preferred
    return None


def _collect_user_layouts(node: torch.fx.Node) -> list[ThreadLayout]:
    """Collect resolved/preferred layouts from all users of *node*."""
    layouts: list[ThreadLayout] = []
    for user in node.users:
        lc = user.meta.get(META_KEY)
        if lc is not None:
            layout = lc.layout or lc.preferred
            if layout is not None:
                layouts.append(layout)
    return layouts


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
