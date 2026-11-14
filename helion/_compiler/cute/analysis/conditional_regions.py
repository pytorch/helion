"""Explicit capture and phi ownership for a single structured device root.

This certificate describes control flow, not attention arithmetic, memory
accesses or an executable schedule. Consumers must repeat the proof against
their final codegen graphs and separately account for effects and reductions.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ....language import _tracing_ops
from ...device_ir import ForLoopGraphInfo
from ...device_ir import IfGraphInfo
from ...device_ir import NodeArgsGraphInfo
from ...device_ir import RootGraphInfo
from ...device_ir import control_flow_child_graph_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...device_ir import GraphInfo


@dataclasses.dataclass(frozen=True)
class RegionCapture:
    placeholder: Node
    source: Node


@dataclasses.dataclass(frozen=True)
class BranchJoin:
    call: Node
    slot: int
    phi: Node
    when_true: Node
    when_false: Node


@dataclasses.dataclass(frozen=True)
class LoopCarry:
    call: Node
    slot: int
    phi: Node
    initial: Node
    placeholder: Node
    backedge: Node


@dataclasses.dataclass(frozen=True)
class UnjoinedBranchOutput:
    call: Node
    graph_id: int
    slot: int
    value: Node


@dataclasses.dataclass(frozen=True)
class OwnedRegion:
    graph: GraphInfo
    parent: Node | None
    captures: tuple[RegionCapture, ...]
    outputs: tuple[Node, ...]


@dataclasses.dataclass(frozen=True)
class ConditionalRegions:
    root: RootGraphInfo
    regions: tuple[OwnedRegion, ...]
    branches: tuple[BranchJoin, ...]
    carries: tuple[LoopCarry, ...]
    unjoined: tuple[UnjoinedBranchOutput, ...]


class _Refused(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Refused


def _nodes(value: object) -> tuple[Node, ...]:
    if not isinstance(value, (tuple, list)):
        raise _Refused
    result = []
    for node in value:
        if not isinstance(node, Node):
            raise _Refused
        result.append(node)
    return tuple(result)


def _same_metadata(left: Node, right: Node) -> bool:
    """Do not add shape guards while proving an existing capture edge."""
    lhs, rhs = left.meta.get("val"), right.meta.get("val")
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):

        def expression(value: object) -> object:
            if isinstance(value, torch.SymInt):
                return value._sympy_()
            return value

        return (
            lhs.dtype == rhs.dtype
            and tuple(map(expression, lhs.shape)) == tuple(map(expression, rhs.shape))
            and tuple(map(expression, lhs.stride()))
            == tuple(map(expression, rhs.stride()))
        )
    return isinstance(
        lhs, (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool)
    ) and type(lhs) is type(rhs)


class _RegionProof:
    def __init__(self, graphs: Sequence[GraphInfo], root_id: int) -> None:
        self.graphs = {info.graph_id: info for info in graphs}
        _require(len(self.graphs) == len(graphs))
        # Ordered phase projection preserves graph ids and replaces other
        # stages by exact output(None) roots. Only these effect-free holes may
        # lie outside this root's complete closure; active foreign graphs
        # remain a refusal rather than being silently dropped.
        for graph_id, info in tuple(self.graphs.items()):
            if graph_id == root_id or not isinstance(info, RootGraphInfo):
                continue
            nodes = tuple(info.graph.nodes)
            if (
                len(nodes) == 1
                and nodes[0].op == "output"
                and nodes[0].args == (None,)
                and not nodes[0].kwargs
            ):
                del self.graphs[graph_id]
        roots = [
            info for info in self.graphs.values() if isinstance(info, RootGraphInfo)
        ]
        _require(len(roots) == 1 and roots[0].graph_id == root_id)
        self.root = roots[0]
        self.seen: set[int] = set()
        self.regions: list[OwnedRegion] = []
        self.branches: list[BranchJoin] = []
        self.carries: list[LoopCarry] = []
        self.unjoined: list[UnjoinedBranchOutput] = []

    def region(
        self, graph_id: int, parent: Node | None, arguments: object
    ) -> OwnedRegion:
        _require(graph_id in self.graphs and graph_id not in self.seen)
        self.seen.add(graph_id)
        info = self.graphs[graph_id]
        captures = []
        if parent is None:
            _require(info is self.root)
        else:
            if not isinstance(info, NodeArgsGraphInfo):
                raise _Refused
            sources = _nodes(arguments)
            placeholders = tuple(info.graph.find_nodes(op="placeholder"))
            _require(len(placeholders) == len(sources))
            _require(tuple(info.node_args) == sources)
            for placeholder, source in zip(placeholders, sources, strict=True):
                _require(_same_metadata(placeholder, source))
                captures.append(RegionCapture(placeholder, source))
        output = tuple(info.graph.find_nodes(op="output"))
        _require(len(output) == 1 and len(output[0].args) == 1)
        outputs = () if output[0].args[0] is None else _nodes(output[0].args[0])
        owned = OwnedRegion(info, parent, tuple(captures), outputs)
        self.regions.append(owned)
        earlier: set[Node] = set()
        for node in info.graph.nodes:
            _require(all(source in earlier for source in node.all_input_nodes))
            earlier.add(node)
            children = control_flow_child_graph_ids(node)
            if not children:
                continue
            if node.target is _tracing_ops._if:
                _require(len(children) == 2 and len(node.args) == 5)
                true = self.region(children[0], node, node.args[3])
                false = self.region(children[1], node, node.args[4])
                self.branch(node, true, false)
            elif _tracing_ops.is_for_loop_target(node.target):
                _require(len(children) == 1 and len(node.args) in (4, 5))
                body = self.region(children[0], node, node.args[3])
                _require(isinstance(body.graph, ForLoopGraphInfo))
                self.loop(node, body)
            else:
                raise _Refused
        return owned

    @staticmethod
    def extracted(call: Node, count: int) -> dict[int, Node]:
        result = {}
        for node in call.users:
            _require(node.target is operator.getitem and len(node.args) == 2)
            index = node.args[1]
            _require(type(index) is int and 0 <= index < count and index not in result)
            result[cast("int", index)] = node
        _require(set(result) == set(range(count)))
        return result

    def branch(self, call: Node, true: OwnedRegion, false: OwnedRegion) -> None:
        info = true.graph
        if not isinstance(info, IfGraphInfo) or info.branches_outputs is None:
            raise _Refused
        captures: dict[str, Node] = {}
        for names, side in ((info.if_arg_names, true), (info.else_arg_names, false)):
            if names is None or len(names) != len(side.captures):
                raise _Refused
            for name, capture in zip(names, side.captures, strict=True):
                _require(name not in captures or captures[name] is capture.source)
                captures[name] = capture.source
        count = len(info.branches_outputs)
        extracted = self.extracted(call, 2 * count)
        used: list[set[int]] = [set(), set()]
        for index, pair in enumerate(info.branches_outputs):
            _require(len(pair) == 2)
            values = []
            for branch_index, (slot, region) in enumerate(
                zip(pair, (true, false), strict=True)
            ):
                if type(slot) is int:
                    _require(0 <= slot < len(region.outputs))
                    values.append(region.outputs[slot])
                    used[branch_index].add(slot)
                elif isinstance(slot, str):
                    _require(slot in captures)
                    values.append(captures[slot])
                else:
                    raise _Refused
            lhs, rhs = extracted[index + count], extracted[index]
            _require(len(lhs.users) == len(rhs.users) == 1)
            phi = next(iter(lhs.users))
            _require(phi in rhs.users and phi.target is _tracing_ops._phi)
            _require(phi.args == (lhs, rhs) and not phi.kwargs)
            _require(_same_metadata(phi, values[0]) and _same_metadata(phi, values[1]))
            self.branches.append(BranchJoin(call, index, phi, values[0], values[1]))
        # Tracing returns branch-local assignments as well as names joined in
        # the parent. Keep every such output explicit in the certificate. The
        # exact call-user check above proves that none is extracted outside
        # this branch; effect validation still owns the entire child graph.
        for slots, region in zip(used, (true, false), strict=True):
            self.unjoined.extend(
                UnjoinedBranchOutput(call, region.graph.graph_id, slot, value)
                for slot, value in enumerate(region.outputs)
                if slot not in slots
            )

    def loop(self, call: Node, body: OwnedRegion) -> None:
        extracted = self.extracted(call, len(body.outputs))
        for index, backedge in enumerate(body.outputs):
            result = extracted[index]
            _require(len(result.users) == 1)
            phi = next(iter(result.users))
            _require(phi.target is _tracing_ops._phi and len(phi.args) == 2)
            initial, current = phi.args
            if current is not result or not isinstance(initial, Node) or phi.kwargs:
                raise _Refused
            matching = [
                capture for capture in body.captures if capture.source is initial
            ]
            _require(len(matching) == 1)
            _require(_same_metadata(phi, initial) and _same_metadata(phi, backedge))
            self.carries.append(
                LoopCarry(call, index, phi, initial, matching[0].placeholder, backedge)
            )

    def prove(self) -> ConditionalRegions:
        self.region(self.root.graph_id, None, ())
        _require(self.seen == set(self.graphs))
        return ConditionalRegions(
            self.root,
            tuple(self.regions),
            tuple(self.branches),
            tuple(self.carries),
            tuple(self.unjoined),
        )


def prove_conditional_regions(
    graphs: Sequence[GraphInfo], root_id: int
) -> ConditionalRegions | None:
    """Refuse unowned regions, captures or noncanonical conditional/loop joins."""
    try:
        return _RegionProof(graphs, root_id).prove()
    except _Refused:
        return None
