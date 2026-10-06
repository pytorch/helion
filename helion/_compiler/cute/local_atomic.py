"""Conservative ownership proof for addressable CTA-private allocations."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import memory_ops
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries

if TYPE_CHECKING:
    from ..device_ir import GraphInfo


def _reachable_graphs(graphs: list[GraphInfo]) -> list[GraphInfo]:
    by_id = {info.graph_id: info for info in graphs}
    pending: list[GraphInfo] = [
        info for info in graphs if isinstance(info, RootGraphInfo)
    ]
    result = {}
    while pending:
        info = pending.pop()
        if info.graph_id in result:
            continue
        result[info.graph_id] = info
        for node in info.graph.nodes:
            if _tracing_ops.is_for_loop_target(node.target):
                pending.append(by_id[cast("int", node.args[0])])
            elif node.target is _tracing_ops._if:
                pending.extend(
                    by_id[cast("int", graph_id)] for graph_id in node.args[1:3]
                )
    return list(result.values())


def atomic_target_origins(graphs: list[GraphInfo]) -> dict[Node, Node]:
    """Resolve atomic target identities through uniform loop captures."""
    graphs = _reachable_graphs(graphs)
    captures: dict[Node, Node] = {}
    parents = control_flow_parent_entries(graphs)
    for info in graphs:
        if info.graph_id in parents:
            call, slot = parents[info.graph_id]
            captures.update(
                zip(
                    info.graph.find_nodes(op="placeholder"),
                    cast("list[Node]", call.args[slot]),
                    strict=True,
                )
            )

    def origin(node: Node) -> Node:
        while True:
            if node in captures:
                node = captures[node]
            elif node.target in (_tracing_ops._new_var, _tracing_ops._phi):
                node = cast("Node", node.args[0])
            elif (
                node.target is operator.getitem
                and isinstance(node.args[0], Node)
                and _tracing_ops.is_for_loop_target(node.args[0].target)
            ):
                call = node.args[0]
                info = next(info for info in graphs if info.graph_id == call.args[0])
                output = info.graph.find_nodes(op="output")[0]
                node = cast("list[Node]", output.args[0])[cast("int", node.args[1])]
            else:
                return node

    return {
        node: origin(cast("Node", node.args[0]))
        for info in graphs
        for node in info.graph.nodes
        if node.target is atomic_ops.atomic_add
    }


def local_atomic_allocations(graphs: list[GraphInfo]) -> frozenset[Node]:
    """Find local targets without changing ordinary host atomic ownership."""
    return frozenset(
        target
        for target in atomic_target_origins(graphs).values()
        if target.target is not _tracing_ops._host_tensor
    )


def prove_local_atomics(graphs: list[GraphInfo]) -> frozenset[Node]:
    """Allow direct allocations, uniform captures, and final nonescaping reads.

    This first bounded form does not admit view aliases, conditional phases,
    returned atomic values, or a read that precedes a later mutation. Each
    allocation lives in its root and is captured unchanged through scalar loops.
    The emitter additionally proves scalar grid/loop geometry and capacity.
    """
    graphs = _reachable_graphs(graphs)
    allocations = local_atomic_allocations(graphs)
    if not allocations:
        return allocations

    def reject(reason: str) -> NoReturn:
        raise exc.InvalidConfig(f"CTA-local atomics require {reason}")

    parents = control_flow_parent_entries(graphs)
    by_graph = {info.graph: info for info in graphs}
    captures: dict[Node, Node] = {}
    for info in graphs:
        if info.graph_id in parents:
            call, slot = parents[info.graph_id]
            captures.update(
                zip(
                    info.graph.find_nodes(op="placeholder"),
                    cast("list[Node]", call.args[slot]),
                    strict=True,
                )
            )
        if any(n.target is _tracing_ops._if for n in info.graph.nodes):
            reject("unconditional uniform phases")

    def origin(node: Node) -> Node:
        while True:
            if node in captures:
                node = captures[node]
            elif node.target in (_tracing_ops._new_var, _tracing_ops._phi):
                node = cast("Node", node.args[0])
            elif (
                node.target is operator.getitem
                and isinstance(node.args[0], Node)
                and _tracing_ops.is_for_loop_target(node.args[0].target)
            ):
                call = node.args[0]
                info = next(info for info in graphs if info.graph_id == call.args[0])
                output = info.graph.find_nodes(op="output")[0]
                node = cast("list[Node]", output.args[0])[cast("int", node.args[1])]
            else:
                return node

    for allocation in allocations:
        if allocation.target is not creation_ops.full:
            reject("a direct fresh hl.full/zeros target without view aliases")
        fake = allocation.meta.get("val")
        dimensions, initial, dtype, _device = allocation.args
        if (
            not isinstance(fake, torch.Tensor)
            or fake.ndim != 1
            or dtype not in (torch.int32, torch.float32)
            or len(cast("list[object]", dimensions)) != 1
            or not isinstance(cast("list[object]", dimensions)[0], int)
            or cast("list[int]", dimensions)[0] <= 0
            or not isinstance(initial, (int, float))
            or by_graph[allocation.graph].graph_id in parents
        ):
            reject("a constant one-dimensional root allocation of int32/float32")
        positions = {node: i for i, node in enumerate(allocation.graph.nodes)}
        updates: list[int] = []
        reads: list[int] = []
        for info in graphs:
            for alias in info.graph.nodes:
                if origin(alias) is not allocation:
                    continue
                for user in alias.users:
                    if user.target is _tracing_ops._new_var:
                        continue
                    if user.target is _tracing_ops._phi:
                        if origin(cast("Node", user.args[1])) is not allocation:
                            reject("loop outputs that retain the same allocation")
                        continue
                    if user.op == "output" and info.graph_id in parents:
                        continue
                    if _tracing_ops.is_for_loop_target(user.target):
                        if (
                            alias not in user.args[3]
                            or alias in user.args[1]
                            or alias in user.args[2]
                            or (
                                _tracing_ops._for_loop_step is user.target
                                and alias in user.args[4]
                            )
                        ):
                            reject("an unchanged allocation capture, not a loop bound")
                        continue
                    if user.target is atomic_ops.atomic_add and user.args[0] is alias:
                        if user.args[2] is alias or alias in user.args[1]:
                            reject("updates that do not read their mutable target")
                        if user.users or user.args[3] != "relaxed":
                            reject("unused relaxed atomic results")
                        parent = user
                        while parent.graph is not allocation.graph:
                            entry = parents.get(by_graph[parent.graph].graph_id)
                            if entry is None:
                                reject("one root owner")
                            parent = entry[0]
                        updates.append(positions[parent])
                        continue
                    if (
                        user.graph is allocation.graph
                        and user.target in (memory_ops.store, atomic_ops.atomic_add)
                        and user.args[2] is alias
                        and cast("Node", user.args[0]).target
                        is _tracing_ops._host_tensor
                    ):
                        reads.append(positions[user])
                        continue
                    reject("only direct final stores/flushes after local updates")
        if not updates or any(position <= max(updates) for position in reads):
            reject("initialization, then updates, then final reads")
    return allocations
