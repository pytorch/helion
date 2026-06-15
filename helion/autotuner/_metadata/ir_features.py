"""Best-effort structural dump of Helion device IR for the kernel-artifact dataset.

Emits one networkx node-link graph per autotune run, inlined on the .meta.jsonl
record under "ir_graph"; load with ``nx.node_link_graph(rec["ir_graph"],
edges="edges")``. Edges are "data" (producer -> consumer) or "region"
(control-flow call -> child placeholder). Best-effort: anything unresolvable is
None, and a malformed DeviceIR is the caller's responsibility. Full schema and
field meanings live in .spec/CONTRACTS.md.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeGuard

import torch
from torch.fx.node import map_arg

from ...language._tracing_ops import _if
from ...language._tracing_ops import _while_loop
from ...language._tracing_ops import is_for_loop_target

if TYPE_CHECKING:
    from ..._compiler.device_ir import DeviceIR

__all__ = ["extract_ir_graph"]

# Cap stringified scalars/locations to keep JSON lines small.
_MAX_VALUE_LEN = 4096
_TRUNCATION_SUFFIX = "...<truncated>"
_TRUNCATE_KEEP = _MAX_VALUE_LEN - len(_TRUNCATION_SUFFIX)

# Strip memory addresses so callable strings are deterministic across processes.
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")

# Matched by class name (not isinstance) to avoid an autotuner -> _compiler import
# cycle; a rename is caught by test_ir_features (it filters on these strings).
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


def _is_graph_id(value: object) -> TypeGuard[int]:
    """True for concrete int graph ids (not bool); narrows type via TypeGuard."""
    return type(value) is int


def _truncate(text: str) -> str:
    """Strictly cap stringified values to _MAX_VALUE_LEN characters to prevent JSON
    line bloat."""
    if len(text) > _MAX_VALUE_LEN:
        return text[:_TRUNCATE_KEEP] + _TRUNCATION_SUFFIX
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
    """Map each producer node to its argument positions in ``node``.

    Positions are a flat ordinal over ``node.args`` then ``node.kwargs`` (FX
    insertion order, walked by ``map_arg``); positional vs keyword origin is not
    distinguished. A producer feeding multiple slots gets multiple positions."""
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
    """Extract ``(child_graph_id, live_args)`` pairs from for/if/while control-flow
    nodes. Positions below are the arg signatures in ``helion.language._tracing_ops``;
    a malformed or non-control-flow node returns ``[]`` (graph ids must be concrete
    ints). If those signatures change upstream, region edges silently drop. The live
    args stay typed ``object`` -- statically ``node.args[i]`` is ``Argument``, not a
    proven sequence, so the caller runtime-checks ``isinstance(live, (list, tuple))``."""
    target = node.target
    args = node.args
    if is_for_loop_target(target):
        # _for_loop[_step](graph_id, begin, end, args, ...): id@0, live args@3.
        if len(args) >= 4 and _is_graph_id(args[0]):
            return [(args[0], args[3])]
        return []
    if target is _if:
        # _if(test, if_graph_id, else_graph_id, if_args, else_args): ids@1,2; args@3,4.
        if len(args) >= 5 and _is_graph_id(args[1]) and _is_graph_id(args[2]):
            return [(args[1], args[3]), (args[2], args[4])]
        return []
    if target is _while_loop:
        # _while_loop(cond_graph_id, body_graph_id, args, [orelse_graph_id]):
        # ids@0,1 (+optional@3) all share the live args@2.
        if len(args) >= 3 and _is_graph_id(args[0]) and _is_graph_id(args[1]):
            specs: list[tuple[int, object]] = [(args[1], args[2]), (args[0], args[2])]
            if len(args) > 3 and _is_graph_id(args[3]):
                specs.append((args[3], args[2]))
            return specs
        return []
    return []


def _has_networkx_node_link() -> bool:
    """Whether networkx>=3.4 is installed -- the version that gave node_link_data /
    node_link_graph the ``edges=`` kwarg this dump relies on.

    The single definition of the version requirement: ``extract_ir_graph`` uses it
    to raise a clear error and the test suite imports it for its skip gate. Asserts
    the version directly via a ``.``-split (no ``packaging`` dependency) instead of
    probing the API.
    """
    try:
        import networkx as nx
    except ImportError:
        return False

    def _leading_int(part: str) -> int:
        # Leading digits only, so pre-release/dev tags parse (e.g. "4rc1" -> 4).
        match = re.match(r"\d+", part)
        return int(match.group()) if match else 0

    parts = nx.__version__.split(".")
    major = _leading_int(parts[0]) if parts else 0
    minor = _leading_int(parts[1]) if len(parts) > 1 else 0
    return (major, minor) >= (3, 4)


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


def extract_ir_graph(device_ir: DeviceIR) -> dict[str, object]:
    """Returns a networkx-loadable node-link dict of structural DeviceIR metadata.

    Builds a ``networkx.DiGraph`` and serializes it with ``node_link_data`` so the
    node-link schema is correct by construction.
    """
    if not _has_networkx_node_link():
        raise ImportError(
            "networkx>=3.4 is required to extract IR graphs "
            "(node_link_data needs the `edges=` kwarg); "
            "install with `pip install 'networkx>=3.4'`."
        )
    import networkx as nx

    graphs = device_ir.graphs

    # Precompute ids + val features once, reused by nodes and edges (O(N), not O(N+E)).
    graphs_by_id = {gi.graph_id: gi for gi in graphs}
    node_id: dict[torch.fx.Node, str] = {}
    val_by_node: dict[torch.fx.Node, ValFeatures] = {}
    for graph_info in graphs:
        gid = graph_info.graph_id
        for node in graph_info.graph.nodes:
            # node.name is unique per fx graph; the g{gid}: prefix makes ids globally
            # unique (else DiGraph would merge same-named nodes across graphs).
            node_id[node] = f"g{gid}:{node.name}"
            val_by_node[node] = _val_features(node.meta.get("val"))

    nx_graph = nx.DiGraph()
    nx_graph.graph.update(_graph_meta(device_ir))

    def add_typed_edge(
        src: str, dst: str, kind: str, positions: list[int], val: ValFeatures
    ) -> None:
        # Edges carry only dtype+shape (the source node holds the full features),
        # deliberately not ``**val``.
        nx_graph.add_edge(
            src,
            dst,
            edge_kind=kind,
            arg_positions=positions,
            dtype=val["dtype"],
            shape=val["shape"],
        )

    # Add all nodes before any edge: add_edge would otherwise auto-create
    # attribute-less placeholders for region targets in not-yet-visited child graphs.
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
                source_loc=None if location is None else _truncate(str(location)),
                **valf,
                **lowf,
            )

    # multigraph=False is safe: data edges target consumers, region edges target
    # placeholders, so (source, target) never collides.
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            for producer, arg_positions in _input_edges(node).items():
                add_typed_edge(
                    node_id[producer],
                    node_id[node],
                    "data",
                    arg_positions,
                    val_by_node[producer],
                )

            # Region edges: the call's live args map positionally onto the child's
            # placeholders.
            for child_gid, live in _region_specs(node):
                child = graphs_by_id.get(child_gid)
                if child is None or not isinstance(live, (list, tuple)):
                    continue
                placeholders = list(child.graph.find_nodes(op="placeholder"))
                # A child may have more placeholders (induction/state) than args,
                # so the positional zip is best-effort.
                for placeholder, arg in zip(placeholders, live, strict=False):
                    if (
                        isinstance(arg, torch.fx.Node)
                        and arg in node_id
                        and placeholder in node_id
                    ):
                        add_typed_edge(
                            node_id[arg],
                            node_id[placeholder],
                            "region",
                            [],
                            val_by_node[arg],
                        )

    return nx.node_link_data(nx_graph, edges="edges")
