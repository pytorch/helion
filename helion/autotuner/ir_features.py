"""Collect a lossless dump of Helion's device IR for the kernel-artifact dataset.

This module walks the ``torch.fx`` graphs held by :class:`DeviceIR` and emits a
single JSON object per autotune run (one per ``run_id``) describing the union of
all device-side graphs as a node-link graph. The output is plain JSON in the
node-link schema, so a consumer can rebuild a graph with no extra effort::

    import json, networkx as nx

    data = json.loads(line)  # one line of <base>.ir.jsonl
    g = nx.node_link_graph(data, edges="links")

Two edge kinds are emitted, both grounded in *live* graph nodes: ``data`` edges
from ``node.args``/``kwargs``, and ``region`` edges from explicit control-flow
call nodes (``_for_loop``/``_for_loop_step``/``_if``/``_while_loop``) whose live
argument lists map positionally onto the child graph's placeholders.

Device IR is config-independent (built once per bound kernel), so the dump is
per ``run_id`` and joins back to the ``.meta.jsonl`` record and per-config CSV
rows on ``run_id``. Host-side state (host -> root edges) is out of scope because
it lives in the host AST / origin maps, not in device IR.

Extraction is best-effort: anything that cannot be resolved is recorded as
``None`` rather than raising. ``extract_ir_graph`` does assume a well-formed
``DeviceIR`` (iterable ``graphs`` of ``GraphInfo``); the autotuner call site
(:meth:`BaseSearch._extract_ir_graph`) wraps it so a malformed/absent device IR
degrades to "no IR artifact" and never breaks autotuning.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING
from typing import TypedDict

import torch
from torch.fx.node import map_arg

from ..language._tracing_ops import _if
from ..language._tracing_ops import _while_loop
from ..language._tracing_ops import is_for_loop_target

if TYPE_CHECKING:
    from .._compiler.device_ir import DeviceIR

log = logging.getLogger(__name__)

# Cap on stringified scalar values / source locations so a stray large object
# can't bloat the JSON line.
_MAX_VALUE_LEN = 256

# Lowering class names dispatched on (kept as strings to avoid importing the
# lowering classes, which would create an autotuner -> _compiler import cycle).
# A rename upstream is caught by the reduction/pointwise extractor tests.
_REDUCTION_LOWERING = "ReductionLowering"
_POINTWISE_LOWERING = "PointwiseLowering"

# Region kinds (``GraphInfo`` subclass names) this extractor is known to handle.
# A graph whose kind is absent here is dumped anyway, but we warn (once per
# kind) so an upstream change to Helion's device IR is noticed. The
# test_known_region_kinds_match_device_ir test additionally fails loudly in CI
# if this set drifts from the concrete GraphInfo subclasses.
_KNOWN_REGION_KINDS = frozenset(
    {
        "RootGraphInfo",
        "ForLoopGraphInfo",
        "ReductionLoopGraphInfo",
        "IfGraphInfo",
        "ElseGraphInfo",
        "WhileConditionGraphInfo",
        "WhileLoopGraphInfo",
        "HelperFunctionGraphInfo",
    }
)


class ValFeatures(TypedDict):
    """Features derived from a node's ``meta['val']`` (tensor or sym scalar)."""

    dtype: str | None
    shape: list[str] | None
    concrete_shape: list[int | None] | None
    stride: list[str] | None
    concrete_stride: list[int | None] | None
    device: str | None
    value: str | None


class LoweringFeatures(TypedDict):
    """Features derived from a node's ``meta['lowering']``."""

    lowering_class: str | None
    reduction_type: str | None
    reduction_ranges: list[str] | None
    pointwise_ranges: list[str] | None
    input_dtypes: list[str] | None
    input_shapes: list[list[str]] | None


class IrNode(TypedDict):
    """A single device-IR fx node (a node-link ``nodes`` entry)."""

    id: str
    graph_id: int
    region_kind: str
    block_ids: list[int]
    op_kind: str
    target: str
    lowering_class: str | None
    dtype: str | None
    shape: list[str] | None
    concrete_shape: list[int | None] | None
    stride: list[str] | None
    concrete_stride: list[int | None] | None
    device: str | None
    value: str | None
    input_dtypes: list[str] | None
    input_shapes: list[list[str]] | None
    reduction_type: str | None
    reduction_ranges: list[str] | None
    pointwise_ranges: list[str] | None
    source_loc: str | None


class IrEdge(TypedDict):
    """A dataflow or region edge (a node-link ``links`` entry)."""

    source: str
    target: str
    edge_kind: str  # "data" | "region"
    arg_positions: list[int]
    dtype: str | None
    shape: list[str] | None


class GraphHierarchyEntry(TypedDict):
    graph_id: int
    region_kind: str
    block_ids: list[int]


class RolledReductionEntry(TypedDict):
    original_graph_id: int
    rolled_block_ids: list[int]
    used_rdim: bool
    can_be_rolled_by_caller: bool


class IrGraphMeta(TypedDict):
    """Graph-level attributes (networkx ``graph`` dict)."""

    run_id: str
    kernel_id: str
    kernel_name: str
    input_shapes: str
    num_graphs: int
    root_ids: list[int]
    graphs: list[GraphHierarchyEntry]
    rolled_reductions: list[RolledReductionEntry]


class IrGraphRecord(TypedDict):
    """One ``.ir.jsonl`` record; loadable via ``nx.node_link_graph(.., edges="links")``."""

    schema_version: int
    run_id: str
    directed: bool
    multigraph: bool
    graph: IrGraphMeta
    nodes: list[IrNode]
    links: list[IrEdge]


# Bump when the .ir.jsonl record shape changes, so consumers can detect format
# drift. 1 = initial schema.
IR_SCHEMA_VERSION = 1


@functools.cache
def _warn_unknown_region_kind(region_kind: str) -> None:
    """Warn once per process for a region kind we don't recognize.

    ``functools.cache`` gives thread-safe warn-once semantics with no
    module-level mutable state (tests call
    ``_warn_unknown_region_kind.cache_clear()``).
    """
    if region_kind not in _KNOWN_REGION_KINDS:
        log.warning(
            "ir_features: unrecognized device IR region_kind %r; dumping it as-is. "
            "Helion's device IR may have changed -- the extractor should be updated.",
            region_kind,
        )


def _truncate(text: str) -> str:
    """Cap a stringified value so one object cannot bloat the JSON line."""
    if len(text) > _MAX_VALUE_LEN:
        return text[:_MAX_VALUE_LEN] + "...<truncated>"
    return text


def _target_str(target: object) -> str:
    """Stable, address-free string for an fx node target.

    ``str()`` of a plain function includes its memory address (non-deterministic),
    so prefer the qualified name; OpOverloads have a stable ``str`` already.
    """
    if isinstance(target, str):
        return target
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name:
        module = getattr(target, "__module__", None)
        return f"{module}.{name}" if module else name
    return str(target)


def _concrete_dim(dim: object) -> int | None:
    """Best-effort concrete size for a (possibly symbolic) dim; ``None`` if unknown."""
    if isinstance(dim, int):
        return dim
    try:
        return int(dim)  # type: ignore[call-overload]
    except (TypeError, ValueError, RuntimeError):
        return None


def _val_features(val: object) -> ValFeatures:
    """Type/shape/stride features for ``node.meta['val']`` (tensor or sym scalar)."""
    out: ValFeatures = {
        "dtype": None,
        "shape": None,
        "concrete_shape": None,
        "stride": None,
        "concrete_stride": None,
        "device": None,
        "value": None,
    }
    if isinstance(val, torch.Tensor):  # FakeTensor is a torch.Tensor subclass
        out["dtype"] = str(val.dtype)
        out["shape"] = [str(d) for d in val.shape]
        out["concrete_shape"] = [_concrete_dim(d) for d in val.shape]
        stride = val.stride()
        out["stride"] = [str(s) for s in stride]
        out["concrete_stride"] = [_concrete_dim(s) for s in stride]
        out["device"] = str(val.device)
    elif val is not None:
        # Symbolic scalars (SymInt/SymFloat/SymBool) and any other non-tensor
        # value: record the type name and a length-capped stringified value.
        out["dtype"] = type(val).__name__
        out["value"] = _truncate(str(val))
    return out


def _lowering_features(node: torch.fx.Node) -> LoweringFeatures:
    """Lowering-class and reduction/pointwise/operand features for a node.

    Reads ``node.meta['lowering']`` and dispatches reduction/pointwise ranges by
    lowering class name (see ``_REDUCTION_LOWERING``/``_POINTWISE_LOWERING``).
    """
    out: LoweringFeatures = {
        "lowering_class": None,
        "reduction_type": None,
        "reduction_ranges": None,
        "pointwise_ranges": None,
        "input_dtypes": None,
        "input_shapes": None,
    }
    lowering = node.meta.get("lowering")
    if lowering is None:
        return out
    cls = type(lowering).__name__
    out["lowering_class"] = cls
    buffer = getattr(lowering, "buffer", None)
    data = getattr(buffer, "data", None)
    if cls == _REDUCTION_LOWERING:
        reduction_type = getattr(lowering, "reduction_type", None)
        out["reduction_type"] = None if reduction_type is None else str(reduction_type)
        ranges = getattr(data, "reduction_ranges", None)
        if ranges is not None:
            out["reduction_ranges"] = [str(r) for r in ranges]
    elif cls == _POINTWISE_LOWERING:
        ranges = getattr(data, "ranges", None)
        if ranges is not None:
            out["pointwise_ranges"] = [str(r) for r in ranges]
    # Operand-side tensors: only InductorLowering subclasses (those with a
    # ``buffer``) expose ``input_fake_tensors``; the call reads node.meta["val"]
    # so guard it best-effort.
    if buffer is not None:
        try:
            fakes = type(lowering).input_fake_tensors(node)  # staticmethod
        except (KeyError, AttributeError, TypeError, RuntimeError):
            fakes = None
        if fakes is not None:
            out["input_dtypes"] = [str(t.dtype) for t in fakes]
            out["input_shapes"] = [[str(d) for d in t.shape] for t in fakes]
    return out


def _input_edges(node: torch.fx.Node) -> dict[torch.fx.Node, list[int]]:
    """Map each producer node to the argument positions it feeds ``node``.

    Uses fx's own ``map_arg`` to walk ``(node.args, node.kwargs)`` (handling all
    fx container types), so a producer used in multiple operand slots keeps all
    its positions. ``arg_positions`` is the ordinal of each producer occurrence
    in that flattened node traversal (preserving multiplicity without a
    multigraph).
    """
    positions: dict[torch.fx.Node, list[int]] = {}
    counter = 0

    def visit(producer: torch.fx.Node) -> torch.fx.Node:
        nonlocal counter
        positions.setdefault(producer, []).append(counter)
        counter += 1
        return producer

    map_arg((node.args, node.kwargs), visit)
    return positions


def _region_specs(node: torch.fx.Node) -> list[tuple[int, object]]:
    """For a control-flow call node, return ``(child_graph_id, live_args_list)``.

    Each control-flow op encodes the invoked child graph id(s) and the live
    argument list that maps positionally onto that child's placeholders. Each
    branch length-guards ``node.args`` and returns ``[]`` for a malformed or
    non-control-flow node (best-effort, never ``IndexError``):

    * ``_for_loop(graph_id, begin, end, args)`` / ``_for_loop_step(..., step)``
    * ``_if(test, if_graph_id, else_graph_id, if_args, else_args)``
    * ``_while_loop(cond_graph_id, body_graph_id, args, orelse_graph_id=None)``
      -- cond/body/orelse all receive the same loop-carried ``args``.
    """
    target = node.target
    args = node.args
    # Child graph ids must be concrete ints; narrow them so a malformed node
    # degrades to [] rather than producing a wrong edge (and to satisfy typing).
    if is_for_loop_target(target):
        if len(args) >= 4 and isinstance(args[0], int):
            return [(args[0], args[3])]
        return []
    if target is _if:
        if len(args) >= 5 and isinstance(args[1], int) and isinstance(args[2], int):
            return [(args[1], args[3]), (args[2], args[4])]
        return []
    if target is _while_loop:
        if len(args) >= 3 and isinstance(args[0], int) and isinstance(args[1], int):
            specs: list[tuple[int, object]] = [(args[1], args[2]), (args[0], args[2])]
            if len(args) > 3 and isinstance(args[3], int):
                specs.append((args[3], args[2]))
            return specs
        return []
    return []


def extract_ir_graph(
    device_ir: DeviceIR,
    *,
    run_id: str,
    kernel_id: str,
    kernel_name: str,
    input_shapes: str,
) -> IrGraphRecord:
    """Return a node-link record for the union of all device IR graphs.

    The result is directly loadable with
    ``networkx.node_link_graph(result, edges="links")``. ``run_id`` is mirrored
    at the top level (the per-line join key) and inside ``graph`` (so it survives
    into the reconstructed networkx graph); ``kernel_id`` and the rest of the
    identity live only in ``graph``.

    Assumes a well-formed ``DeviceIR``; callers that cannot guarantee that should
    guard the call (the autotuner does), since a malformed IR may raise.
    """
    graphs = device_ir.graphs

    # Pass 1: assign every node a globally-unique id (fx node names are only
    # unique within a single graph) and precompute its val features once, so
    # edges reuse them instead of recomputing per producer (O(N) not O(N+E)).
    graphs_by_id = {gi.graph_id: gi for gi in graphs}
    node_id: dict[torch.fx.Node, str] = {}
    val_by_node: dict[torch.fx.Node, ValFeatures] = {}
    for graph_info in graphs:
        gid = graph_info.graph_id
        for node in graph_info.graph.nodes:
            node_id[node] = f"g{gid}:{node.name}"
            val_by_node[node] = _val_features(node.meta.get("val"))

    nodes: list[IrNode] = []
    links: list[IrEdge] = []

    for graph_info in graphs:
        gid = graph_info.graph_id
        region_kind = type(graph_info).__name__
        _warn_unknown_region_kind(region_kind)
        block_ids = list(getattr(graph_info, "block_ids", []) or [])

        for node in graph_info.graph.nodes:
            valf = val_by_node[node]
            lowf = _lowering_features(node)
            location = node.meta.get("location")
            nodes.append(
                {
                    "id": node_id[node],
                    "graph_id": gid,
                    "region_kind": region_kind,
                    "block_ids": block_ids,
                    "op_kind": node.op,
                    "target": _target_str(node.target),
                    "lowering_class": lowf["lowering_class"],
                    "dtype": valf["dtype"],
                    "shape": valf["shape"],
                    "concrete_shape": valf["concrete_shape"],
                    "stride": valf["stride"],
                    "concrete_stride": valf["concrete_stride"],
                    "device": valf["device"],
                    "value": valf["value"],
                    "input_dtypes": lowf["input_dtypes"],
                    "input_shapes": lowf["input_shapes"],
                    "reduction_type": lowf["reduction_type"],
                    "reduction_ranges": lowf["reduction_ranges"],
                    "pointwise_ranges": lowf["pointwise_ranges"],
                    "source_loc": None
                    if location is None
                    else _truncate(str(location)),
                }
            )

            # Data edges: producer -> this node, one edge per producer with the
            # argument positions it feeds.
            for producer, arg_positions in _input_edges(node).items():
                producer_val = val_by_node[producer]
                links.append(
                    {
                        "source": node_id[producer],
                        "target": node_id[node],
                        "edge_kind": "data",
                        "arg_positions": arg_positions,
                        "dtype": producer_val["dtype"],
                        "shape": producer_val["shape"],
                    }
                )

            # Region edges: from a live control-flow call node, the outer arg
            # list maps positionally onto the child graph's placeholders.
            for child_gid, live in _region_specs(node):
                child = graphs_by_id.get(child_gid)
                if child is None or not isinstance(live, (list, tuple)):
                    continue
                placeholders = list(child.graph.find_nodes(op="placeholder"))
                # Best-effort positional zip: a child may have more placeholders
                # (e.g. loop induction/state) than passed args, so don't be strict.
                for placeholder, arg in zip(placeholders, live, strict=False):
                    if (
                        isinstance(arg, torch.fx.Node)
                        and arg in node_id
                        and placeholder in node_id
                    ):
                        arg_val = val_by_node[arg]
                        links.append(
                            {
                                "source": node_id[arg],
                                "target": node_id[placeholder],
                                "edge_kind": "region",
                                "arg_positions": [],
                                "dtype": arg_val["dtype"],
                                "shape": arg_val["shape"],
                            }
                        )

    graph_meta: IrGraphMeta = {
        "run_id": run_id,
        "kernel_id": kernel_id,
        "kernel_name": kernel_name,
        "input_shapes": input_shapes,
        "num_graphs": len(graphs),
        "root_ids": list(getattr(device_ir, "root_ids", []) or []),
        # Per-graph region hierarchy (graph-level features for the GNN).
        "graphs": [
            {
                "graph_id": gi.graph_id,
                "region_kind": type(gi).__name__,
                "block_ids": list(getattr(gi, "block_ids", []) or []),
            }
            for gi in graphs
        ],
        # Reduction loops are rolled alternates of an original graph, not called
        # subgraphs; record that relationship structurally instead of as edges.
        "rolled_reductions": [
            {
                "original_graph_id": info.original_graph_id,
                "rolled_block_ids": list(info.rolled_block_ids),
                "used_rdim": bool(info.used_rdim),
                "can_be_rolled_by_caller": bool(info.can_be_rolled_by_caller),
            }
            for info in getattr(device_ir, "rolled_reductions", [])
        ],
    }
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "run_id": run_id,  # top-level per-line join key (networkx ignores it)
        "directed": True,
        "multigraph": False,
        "graph": graph_meta,
        "nodes": nodes,
        "links": links,
    }
