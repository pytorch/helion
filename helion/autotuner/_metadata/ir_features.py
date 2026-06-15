"""Lossless dump of Helion device IR for the kernel-artifact dataset.

Walks DeviceIR torch.fx graphs and emits one node-link graph per autotune run,
inlined on the .meta.jsonl record under "ir_graph". Load with
nx.node_link_graph(rec["ir_graph"], edges="edges").

Edges:
  data:    producer -> consumer from node.args/kwargs
  region:  control-flow call (_for_loop, _for_loop_step, _if, _while_loop)
           maps live args positionally to child-graph placeholders.
HelperFunctionGraphInfo is emitted but not connected, helper calls have no
region edges, so those subgraphs appear disconnected.

Graph is directed, multigraph=False. No (source,target) collision: data edges
target consuming nodes, region edges target placeholders only.

Device IR is config-independent; host -> root edges are out of scope.
Extraction is best-effort per node (unresolvable -> None). extract_ir_graph
assumes well-formed DeviceIR; caller guards malformed case.

String caps: scalar value and source_loc truncated to _MAX_VALUE_LEN.
Shape/stride strings are uncapped, small by construction.

Schema version lives on the ir_graph object::

    ir = rec.get("ir_graph")
    v = ir.get("schema_version", 0) if ir else 0
    if v != IR_SCHEMA_VERSION:
        ...
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeGuard
from typing import cast

import torch
from torch.fx.node import map_arg

from ...language._tracing_ops import _if
from ...language._tracing_ops import _while_loop
from ...language._tracing_ops import is_for_loop_target

if TYPE_CHECKING:
    from ..._compiler.device_ir import DeviceIR

__all__ = [
    "IR_SCHEMA_VERSION",
    "IrEdge",
    "IrGraphMeta",
    "IrGraphRecord",
    "IrNode",
    "extract_ir_graph",
]

# Cap scalar strings and source locations to prevent JSON line bloat.
_MAX_VALUE_LEN = 4096
_TRUNCATION_SUFFIX = "...<truncated>"

# Strip memory addresses to keep callable/partial strings deterministic across
# processes.
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")

# Kept as strings to avoid autotuner -> _compiler import cycle.
_REDUCTION_LOWERING = "ReductionLowering"
_POINTWISE_LOWERING = "PointwiseLowering"


class ValFeatures(TypedDict):
    """Features derived from a node's meta['val'] (tensor or sym scalar)."""

    dtype: str | None
    shape: list[str] | None
    concrete_shape: list[int | None] | None
    stride: list[str] | None
    concrete_stride: list[int | None] | None
    device: str | None
    value: str | None


class LoweringFeatures(TypedDict):
    """Features derived from a node's meta['lowering']."""

    lowering_class: str | None
    reduction_type: str | None
    reduction_ranges: list[str] | None
    pointwise_ranges: list[str] | None
    input_dtypes: list[str] | None
    input_shapes: list[list[str]] | None


class _NodeCore(TypedDict):
    """Node fields that come from the fx node itself (not val/lowering)."""

    id: str
    graph_id: int
    region_kind: str
    block_ids: list[int]
    op_kind: str
    target: str
    source_loc: str | None


class IrNode(_NodeCore, ValFeatures, LoweringFeatures):
    """Node-edge schema node. keys are disjoint union of the three bases."""


class IrEdge(TypedDict):
    """A dataflow or region edge (a node-edge edges entry)."""

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
    """Structural graph metadata (networkx graph dict); run identity lives in
    .meta.jsonl."""

    num_graphs: int
    root_ids: list[int]
    graphs: list[GraphHierarchyEntry]
    rolled_reductions: list[RolledReductionEntry]


class IrGraphRecord(TypedDict):
    """The per-run ir_graph value; loadable via nx.node_link_graph(.., edges="edges")."""

    schema_version: int
    directed: bool
    multigraph: bool
    graph: IrGraphMeta
    nodes: list[IrNode]
    edges: list[IrEdge]


# Bump when the ir_graph record shape changes for consumers to detect format changes
IR_SCHEMA_VERSION = 1


def _is_graph_id(value: object) -> TypeGuard[int]:
    """True for concrete int graph ids (not bool); narrows type via TypeGuard."""
    return type(value) is int


def _truncate(text: str) -> str:
    """Strictly cap stringified values to _MAX_VALUE_LEN characters to prevent JSON
    line bloat."""
    if len(text) > _MAX_VALUE_LEN:
        return text[: _MAX_VALUE_LEN - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
    return text


def _target_str(target: object) -> str:
    """Returns a deterministic target string by stripping memory addresses from callables."""
    if isinstance(target, str):
        return target
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name:
        module = getattr(target, "__module__", None)
        return f"{module}.{name}" if module else name
    return _ADDRESS_RE.sub("", str(target))


def _concrete_dim(dim: object) -> int | None:
    """Returns static ``int`` size for a dim; returns ``None`` for symbols to avoid shape
    guard side effects."""
    return dim if isinstance(dim, int) else None


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
    if isinstance(val, torch.Tensor):
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
    """Extracts node features from ``node.meta['lowering']`` by lowering class name."""
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
        # Read reduction_type from buffer data, bypassing the lowering getter assert.
        # Allows malformed lowerings to safely degrade to None.
        reduction_type = getattr(data, "reduction_type", None)
        out["reduction_type"] = None if reduction_type is None else str(reduction_type)
        ranges = getattr(data, "reduction_ranges", None)
        if ranges is not None:
            out["reduction_ranges"] = [str(r) for r in ranges]
    elif cls == _POINTWISE_LOWERING:
        ranges = getattr(data, "ranges", None)
        if ranges is not None:
            out["pointwise_ranges"] = [str(r) for r in ranges]
    # Read ``input_fake_tensors`` only if a ``buffer`` is present.
    # Guards the node.meta["val"] read best-effort.
    if buffer is not None:
        try:
            fakes = type(lowering).input_fake_tensors(node)
        except (KeyError, AttributeError, TypeError, RuntimeError):
            fakes = None
        if fakes is not None:
            out["input_dtypes"] = [str(t.dtype) for t in fakes]
            out["input_shapes"] = [[str(d) for d in t.shape] for t in fakes]
    return out


def _input_edges(node: torch.fx.Node) -> dict[torch.fx.Node, list[int]]:
    """Map producers to their ordinal argument positions in node, preserving
    multiplicity via map_arg."""
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
    """Safely extracts (child_graph_id, live_args_list) from for/if/while
    control-flow nodes."""
    target = node.target
    args = node.args
    # Graph ids must be concrete ints. A malformed or non-control-flow node
    # returns [].
    if is_for_loop_target(target):
        if len(args) >= 4 and _is_graph_id(args[0]):
            return [(args[0], args[3])]
        return []
    if target is _if:
        if len(args) >= 5 and _is_graph_id(args[1]) and _is_graph_id(args[2]):
            return [(args[1], args[3]), (args[2], args[4])]
        return []
    if target is _while_loop:
        if len(args) >= 3 and _is_graph_id(args[0]) and _is_graph_id(args[1]):
            specs: list[tuple[int, object]] = [(args[1], args[2]), (args[0], args[2])]
            if len(args) > 3 and _is_graph_id(args[3]):
                specs.append((args[3], args[2]))
            return specs
        return []
    return []


def _has_networkx_node_link() -> bool:
    """Whether networkx>=3.4 (node_link_data's ``edges=`` kwarg) is available.

    The single definition of the version requirement: ``extract_ir_graph`` uses it
    to raise a clear error, and the test suite imports it for its skip gate, so both
    agree on one capability check. A capability probe, not a version string -- the
    ``edges=`` kwarg landed in networkx 3.4.
    """
    try:
        import networkx as nx

        nx.node_link_graph({"nodes": [], "edges": []}, edges="edges")
    except (ImportError, TypeError):
        return False
    return True


def _graph_meta(device_ir: DeviceIR) -> IrGraphMeta:
    """Structural, graph-level metadata seeded onto ``G.graph`` (emitted verbatim
    by ``node_link_data``); run identity lives in the .meta.jsonl record, not here."""
    graphs = device_ir.graphs
    return {
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


def extract_ir_graph(device_ir: DeviceIR) -> IrGraphRecord:
    """Returns a networkx-loadable node-link dict of structural DeviceIR metadata.

    Builds a ``networkx.DiGraph`` and serializes it with ``node_link_data`` so the
    node-link schema is correct by construction; ``schema_version`` is re-added at
    the top.
    """
    if not _has_networkx_node_link():
        raise ImportError(
            "networkx>=3.4 is required to extract IR graphs "
            "(node_link_data needs the `edges=` kwarg); "
            "install with `pip install 'networkx>=3.4'`."
        )
    import networkx as nx

    graphs = device_ir.graphs

    # Pass 1: assign global IDs and precompute val features once (O(N), not O(N+E)).
    graphs_by_id = {gi.graph_id: gi for gi in graphs}
    node_id: dict[torch.fx.Node, str] = {}
    val_by_node: dict[torch.fx.Node, ValFeatures] = {}
    for graph_info in graphs:
        gid = graph_info.graph_id
        for node in graph_info.graph.nodes:
            node_id[node] = f"g{gid}:{node.name}"
            val_by_node[node] = _val_features(node.meta.get("val"))

    nx_graph = nx.DiGraph()
    nx_graph.graph.update(_graph_meta(device_ir))

    # Pass 2: add every node first, so add_edge never auto-creates an
    # attribute-less placeholder (region edges target child-graph placeholders
    # that are visited in a later graph_info).
    for graph_info in graphs:
        gid = graph_info.graph_id
        region_kind = type(graph_info).__name__
        block_ids = list(getattr(graph_info, "block_ids", []) or [])
        for node in graph_info.graph.nodes:
            valf = val_by_node[node]
            lowf = _lowering_features(node)
            location = node.meta.get("location")
            nx_graph.add_node(
                node_id[node],
                graph_id=gid,
                region_kind=region_kind,
                block_ids=block_ids,
                op_kind=node.op,
                target=_target_str(node.target),
                lowering_class=lowf["lowering_class"],
                dtype=valf["dtype"],
                shape=valf["shape"],
                concrete_shape=valf["concrete_shape"],
                stride=valf["stride"],
                concrete_stride=valf["concrete_stride"],
                device=valf["device"],
                value=valf["value"],
                input_dtypes=lowf["input_dtypes"],
                input_shapes=lowf["input_shapes"],
                reduction_type=lowf["reduction_type"],
                reduction_ranges=lowf["reduction_ranges"],
                pointwise_ranges=lowf["pointwise_ranges"],
                source_loc=None if location is None else _truncate(str(location)),
            )

    # Pass 3: add edges. multigraph=False is safe; data edges target consuming
    # nodes and region edges target placeholders, so no (source, target) collides.
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            # Data edges: producer -> this node, one edge per producer with the
            # argument positions it feeds.
            for producer, arg_positions in _input_edges(node).items():
                producer_val = val_by_node[producer]
                nx_graph.add_edge(
                    node_id[producer],
                    node_id[node],
                    edge_kind="data",
                    arg_positions=arg_positions,
                    dtype=producer_val["dtype"],
                    shape=producer_val["shape"],
                )

            # Region edges: from a live control-flow call node, the outer arg
            # list maps positionally onto the child graph's placeholders.
            for child_gid, live in _region_specs(node):
                child = graphs_by_id.get(child_gid)
                if child is None or not isinstance(live, (list, tuple)):
                    continue
                placeholders = list(child.graph.find_nodes(op="placeholder"))
                # A child legitimately has more placeholders (induction/state)
                # than passed args, so the positional zip is best-effort.
                for placeholder, arg in zip(placeholders, live, strict=False):
                    if (
                        isinstance(arg, torch.fx.Node)
                        and arg in node_id
                        and placeholder in node_id
                    ):
                        arg_val = val_by_node[arg]
                        nx_graph.add_edge(
                            node_id[arg],
                            node_id[placeholder],
                            edge_kind="region",
                            arg_positions=[],
                            dtype=arg_val["dtype"],
                            shape=arg_val["shape"],
                        )

    data = nx.node_link_data(nx_graph, edges="edges")
    return cast("IrGraphRecord", {"schema_version": IR_SCHEMA_VERSION, **data})
