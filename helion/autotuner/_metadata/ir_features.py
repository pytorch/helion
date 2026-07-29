"""Structural DeviceIR dump for the autotune dataset sidecar.

Saves one networkx graph per autotune run inside the .meta.jsonl file
under the "ir_graph" key. The graph structure is strictly defined by the
ValFeatures, LoweringFeatures, and IrGraphMeta.

Structural failures raise to the metadata writer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeGuard

from packaging import version
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


@dataclass
class ValFeatures:
    """Features derived from a node's meta['val'] (tensor or sym scalar)."""

    dtype: str | None = None
    shape: list[str] | None = None
    concrete_shape: list[int | None] | None = None
    stride: list[str] | None = None
    concrete_stride: list[int | None] | None = None
    device: str | None = None
    value: str | None = None


@dataclass
class LoweringFeatures:
    """Features derived from a node's meta['lowering']."""

    lowering_class: str | None = None
    reduction_type: str | None = None
    reduction_ranges: list[str] | None = None
    pointwise_ranges: list[str] | None = None
    input_dtypes: list[str] | None = None
    input_shapes: list[list[str]] | None = None


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
    num_graphs: int
    root_ids: list[int]
    graphs: list[GraphHierarchyEntry]
    rolled_reductions: list[RolledReductionEntry]


def _is_graph_id(value: object) -> TypeGuard[int]:
    """True for a concrete int graph id (not ``bool``)."""
    return type(value) is int


def _truncate(text: str) -> str:
    if len(text) > _MAX_VALUE_LEN:
        return text[:_TRUNCATE_KEEP] + _TRUNCATION_SUFFIX
    return text


def _target_str(target: object) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name:
        module = getattr(target, "__module__", None)
        return f"{module}.{name}" if module else name
    return _ADDRESS_RE.sub("", str(target))


def _concrete_dim(dim: object) -> int | None:
    """``None`` for a symbolic dim -- reading it would trigger shape-guard side effects."""
    return dim if isinstance(dim, int) else None


def _val_features(val: object) -> ValFeatures:
    if isinstance(val, torch.Tensor):
        stride = val.stride()
        return ValFeatures(
            dtype=str(val.dtype),
            shape=[str(d) for d in val.shape],
            concrete_shape=[_concrete_dim(d) for d in val.shape],
            stride=[str(s) for s in stride],
            concrete_stride=[_concrete_dim(s) for s in stride],
            device=str(val.device),
        )
    if val is not None:
        return ValFeatures(dtype=type(val).__name__, value=_truncate(str(val)))
    return ValFeatures()


def _lowering_features(node: torch.fx.Node) -> LoweringFeatures:
    lowering = node.meta.get("lowering")
    if lowering is None:
        return LoweringFeatures()
    cls = type(lowering).__name__
    out = LoweringFeatures(lowering_class=cls)
    buffer = getattr(lowering, "buffer", None)
    data = getattr(buffer, "data", None)
    if cls == _REDUCTION_LOWERING:
        # Read reduction_type from buffer data, bypassing the lowering getter assert.
        # Allows malformed lowerings to safely degrade to None.
        reduction_type = getattr(data, "reduction_type", None)
        out.reduction_type = None if reduction_type is None else str(reduction_type)
        ranges = getattr(data, "reduction_ranges", None)
        if ranges is not None:
            out.reduction_ranges = [str(r) for r in ranges]
    elif cls == _POINTWISE_LOWERING:
        ranges = getattr(data, "ranges", None)
        if ranges is not None:
            out.pointwise_ranges = [str(r) for r in ranges]
    if buffer is not None:
        # A buffer-allocating lowering exposes input_fake_tensors and a missing
        # method or non-iterable result fails loudly.
        fakes = type(lowering).input_fake_tensors(node)
        out.input_dtypes = [str(t.dtype) for t in fakes]
        out.input_shapes = [[str(d) for d in t.shape] for t in fakes]

    return out


def _input_edges(node: torch.fx.Node) -> dict[torch.fx.Node, list[int]]:
    """Map each producer node to its occurrence positions in ``node``.

    ``map_arg`` visits only ``torch.fx.Node`` leaves of ``node.args``/``node.kwargs``
    (in FX traversal order), so these are producer occurrence indexes, not true
    argument slots: constants and other non-node args do not advance the counter.
    A producer feeding multiple slots gets multiple positions."""
    positions: dict[torch.fx.Node, list[int]] = {}
    counter = 0

    def visit(producer: torch.fx.Node) -> torch.fx.Node:
        nonlocal counter
        positions.setdefault(producer, []).append(counter)
        counter += 1
        return producer

    map_arg((node.args, node.kwargs), visit)
    return positions


def _region_specs(node: torch.fx.Node) -> list[int]:
    """Return child graph ids for control-flow nodes.

    Handles for, if, and while. Returns [] for non-control-flow nodes or
    when the graph id is not a concrete int. Child inputs live on the
    child graph's node_args, not on this node.
    """
    target = node.target
    args = node.args
    if is_for_loop_target(target):
        if len(args) >= 4 and _is_graph_id(args[0]):
            return [args[0]]
        return []
    if target is _if:
        if len(args) >= 5 and _is_graph_id(args[1]) and _is_graph_id(args[2]):
            return [args[1], args[2]]
        return []
    if target is _while_loop:
        if len(args) >= 3 and _is_graph_id(args[0]) and _is_graph_id(args[1]):
            ids = [args[1], args[0]]
            if len(args) > 3 and _is_graph_id(args[3]):
                ids.append(args[3])
            return ids
        return []
    return []


def _has_networkx_node_link() -> bool:
    """Whether networkx>=3.4 is installed -- the version that gave node_link_data /
    node_link_graph the ``edges=`` kwarg this dump relies on.

    The single definition of the version requirement: ``extract_ir_graph`` uses it
    to raise a clear error and the test suite imports it for its skip gate.
    ``base_version`` drops any pre/post/dev suffix so a 3.4 pre-release still counts
    as >=3.4.
    """
    try:
        import networkx as nx
    except ImportError:
        return False

    base = version.parse(nx.__version__).base_version
    return version.parse(base) >= version.parse("3.4")


def _graph_meta(device_ir: DeviceIR) -> IrGraphMeta:
    """Structural, graph-level metadata seeded onto ``G.graph`` (emitted verbatim
    by ``node_link_data``); run identity lives in the .meta.jsonl record, not here."""
    graphs = device_ir.graphs
    return {
        "num_graphs": len(graphs),
        "root_ids": list(getattr(device_ir, "root_ids", []) or []),
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
        # deliberately not the full ``val``.
        nx_graph.add_edge(
            src,
            dst,
            edge_kind=kind,
            arg_positions=positions,
            dtype=val.dtype,
            shape=val.shape,
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
                **vars(valf),
                **vars(lowf),
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

            for child_gid in _region_specs(node):
                child = graphs_by_id.get(child_gid)
                outer_args = getattr(child, "node_args", None) if child else None
                if not child or not outer_args:
                    continue
                # node_args is 1:1 with the child's placeholders by construction the dict
                # lookups fail loud if that invariant ever breaks.
                placeholders = list(child.graph.find_nodes(op="placeholder"))
                for placeholder, arg in zip(placeholders, outer_args, strict=True):
                    add_typed_edge(
                        node_id[arg],
                        node_id[placeholder],
                        "region",
                        [],
                        val_by_node[arg],
                    )

    return nx.node_link_data(nx_graph, edges="edges")
