"""Collect lossless dump of Helion's device IR for the kernel-artifact dataset.

This module walks the ``torch.fx`` graphs held by :class:`DeviceIR` and emits a
single JSON object per autotune run (one per ``run_id``) describing the union of
all device-side graphs as a node-link graph. The output is plain JSON. The
node-link schema can be used by a consumer to rebuild a graph.

Two kinds of edge kinds are emitted, both grounded in *live* graph nodes:
``data`` edges from ``node.args``/``kwargs``, and ``region`` edges from
explicit control-flow call nodes (``_for_loop``/``_for_loop_step``/``_if``/
``_while_loop``) whose live argument lists map positionally onto the child
graph's placeholders.
Device IR is config-independent (built once per bound kernel), so the dump is
per ``run_id`` and joins back to the ``.meta.jsonl`` record and per-config
CSV rows on ``run_id``.

Extraction is best-effort: anything that cannot be resolved is recorded as
``None``. Host-side state (host -> root edges) is out of scope bacuse it lives
in the host AST / origin maps, not in device IR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from ..language._tracing_ops import _if
from ..language._tracing_ops import _while_loop
from ..language._tracing_ops import is_for_loop_target

if TYPE_CHECKING:
    from .._compiler.device_ir import DeviceIR

log = logging.getLogger(__name__)

# Region kinds (``GraphInfo`` subclass names) this extractor is known to handle.
# A graph whose kind is absent here is dumped anyway, but we warn once so that an
# upstream change to Helion's device IR is caught early rather than silently.
_KNOWN_REGION_KINDS = frozenset(
    {
        "RootGraphInfo",
        "ForLoopGraphInfo",
        "ReductionLoopGraphInfo",
        "IfGraphInfo",
        "ElseGraphInfo",
        "WhileConditionGraphInfo",
        "WhileLoopGraphInfo",
    }
)
_warned_region_kinds: set[str] = set()


def _warn_unknown_region_kind(region_kind: str) -> None:
    if (
        region_kind not in _KNOWN_REGION_KINDS
        and region_kind not in _warned_region_kinds
    ):
        _warned_region_kinds.add(region_kind)
        log.warning(
            "ir_features: unrecognized device IR region_kind %r; dumping it as-is. "
            "Helion's device IR may have changed -- the extractor should be updated.",
            region_kind,
        )


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


def _val_features(val: object) -> dict[str, object]:
    """Type/shape/stride features for ``node.meta['val']`` (tensor or sym scalar)."""
    out: dict[str, object] = {
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
    elif isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)) or (
        val is not None
    ):
        # Symbolic scalars and any other non-tensor value: record the type name
        # and a stringified value.
        out["dtype"] = type(val).__name__
        out["value"] = str(val)
    return out


def _lowering_features(node: torch.fx.Node) -> dict[str, object]:
    """Lowering-class and reduction/pointwise/operand features for a node.

    Reads ``node.meta['lowering']``. Reduction/pointwise ranges are dispatched by
    lowering class name (no import of the lowering classes, avoiding an
    autotuner -> _compiler import cycle).
    """
    out: dict[str, object] = {
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
    if cls == "ReductionLowering":
        reduction_type = getattr(lowering, "reduction_type", None)
        out["reduction_type"] = None if reduction_type is None else str(reduction_type)
        ranges = getattr(data, "reduction_ranges", None)
        if ranges is not None:
            out["reduction_ranges"] = [str(r) for r in ranges]
    elif cls == "PointwiseLowering":
        ranges = getattr(data, "ranges", None)
        if ranges is not None:
            out["pointwise_ranges"] = [str(r) for r in ranges]
    # Operand-side tensors: only InductorLowering subclasses (those with a
    # ``buffer``) expose ``input_fake_tensors``.
    if buffer is not None:
        fakes = type(lowering).input_fake_tensors(node)  # staticmethod
        out["input_dtypes"] = [str(t.dtype) for t in fakes]
        out["input_shapes"] = [[str(d) for d in t.shape] for t in fakes]
    return out


def _input_edges(node: torch.fx.Node) -> dict[torch.fx.Node, list[int]]:
    """Map each producer node to the flattened argument positions it feeds.

    Walks ``node.args`` then ``node.kwargs`` assigning an incrementing index to
    every leaf, so a producer used in multiple operand slots keeps all its
    positions (preserving multiplicity without needing a multigraph).
    """
    positions: dict[torch.fx.Node, list[int]] = {}
    counter = 0

    def walk(obj: object) -> None:
        nonlocal counter
        if isinstance(obj, torch.fx.Node):
            positions.setdefault(obj, []).append(counter)
            counter += 1
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item)
        else:
            counter += 1

    walk(node.args)
    walk(node.kwargs)
    return positions


def _region_specs(node: torch.fx.Node) -> list[tuple[int, object]]:
    """For a control-flow call node, return ``(child_graph_id, live_args_list)``.

    Each control-flow op encodes the invoked child graph id(s) and the live
    argument list that maps positionally onto that child's placeholders:

    * ``_for_loop(graph_id, begin, end, args)`` / ``_for_loop_step(..., step)``
    * ``_if(test, if_graph_id, else_graph_id, if_args, else_args)``
    * ``_while_loop(cond_graph_id, body_graph_id, args, orelse_graph_id=None)``
      -- cond/body/orelse all receive the same loop-carried ``args``.
    """
    target = node.target
    args = node.args
    if is_for_loop_target(target):
        return [(args[0], args[3])]
    if target is _if:
        return [(args[1], args[3]), (args[2], args[4])]
    if target is _while_loop:
        specs: list[tuple[int, object]] = [(args[1], args[2]), (args[0], args[2])]
        if len(args) > 3 and args[3] is not None:
            specs.append((args[3], args[2]))
        return specs
    return []


def extract_ir_graph(
    device_ir: DeviceIR,
    *,
    run_id: str,
    kernel_id: str,
    kernel_name: str,
    input_shapes: str,
) -> dict[str, object]:
    """Return a node-link JSON dict for the union of all device IR graphs.

    The result is directly loadable with
    ``networkx.node_link_graph(result, edges="links")``.
    """
    graphs = device_ir.graphs
    graphs_by_id = {gi.graph_id: gi for gi in graphs}

    # Pass 1: assign a globally-unique id to every node across all graphs
    # (fx node names are only unique within a single graph).
    node_id: dict[torch.fx.Node, str] = {}
    for graph_info in graphs:
        gid = graph_info.graph_id
        for node in graph_info.graph.nodes:
            node_id[node] = f"g{gid}:{node.name}"

    nodes: list[dict[str, object]] = []
    links: list[dict[str, object]] = []

    for graph_info in graphs:
        gid = graph_info.graph_id
        region_kind = type(graph_info).__name__
        _warn_unknown_region_kind(region_kind)
        block_ids = list(getattr(graph_info, "block_ids", []) or [])

        for node in graph_info.graph.nodes:
            valf = _val_features(node.meta.get("val"))
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
                    "source_loc": None if location is None else str(location),
                }
            )

            # Data edges: producer -> this node, one edge per producer with the
            # flattened argument positions it feeds.
            for producer, arg_positions in _input_edges(node).items():
                producer_val = _val_features(producer.meta.get("val"))
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
                        arg_val = _val_features(arg.meta.get("val"))
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

    return {
        # Top-level join keys (networkx ignores extra keys); also mirrored into
        # ``graph`` so they survive into the reconstructed networkx graph.
        "run_id": run_id,
        "kernel_id": kernel_id,
        "directed": True,
        "multigraph": False,
        "graph": {
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
            # Reduction loops are rolled alternates of an original graph, not
            # called subgraphs; record that relationship structurally.
            "rolled_reductions": [
                {
                    "original_graph_id": info.original_graph_id,
                    "rolled_block_ids": list(info.rolled_block_ids),
                    "used_rdim": bool(info.used_rdim),
                    "can_be_rolled_by_caller": bool(info.can_be_rolled_by_caller),
                }
                for info in getattr(device_ir, "rolled_reductions", [])
            ],
        },
        "nodes": nodes,
        "links": links,
    }
