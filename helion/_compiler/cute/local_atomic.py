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
from ...language import scan_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering

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


def terminal_finalizer_inputs(
    node: Node, graphs: list[GraphInfo]
) -> tuple[frozenset[Node], frozenset[Node]]:
    """Prove a terminal root branch after local mutation, without local captures.

    Dynamic control derives from one scalar ordered global ticket. The emitter
    additionally checks its physical shared ownership and every symbolic origin.
    Neither tensor rank alone nor a single-element warp register proves uniformity.
    """

    def reject() -> NoReturn:
        raise exc.InvalidConfig(
            "CTA-local atomics require a terminal uniform scalar-ticket finalizer"
        )

    graphs = _reachable_graphs(graphs)
    by_id = {info.graph_id: info for info in graphs}
    root = next(info for info in graphs if info.graph is node.graph)
    following = list(node.graph.nodes)
    if (
        not isinstance(root, RootGraphInfo)
        or node.users
        or any(n.op != "output" for n in following[following.index(node) + 1 :])
    ):
        reject()
    info = by_id[cast("int", node.args[1])]
    if not isinstance(info, IfGraphInfo) or info.branches_outputs:
        reject()
    for graph_id in node.args[1:3]:
        branch = by_id[cast("int", graph_id)]
        for child in branch.graph.nodes:
            if child.target in (_tracing_ops._if, atomic_ops.atomic_add):
                reject()
            if _tracing_ops.is_for_loop_target(child.target):
                terminal_loop_symbols(child, graphs)
    # Follow all producer dependencies, including loop captures and lazy views.
    # Over-approximating a loop result is deliberate: no local alias is admitted.
    allocations = local_atomic_allocations(graphs)
    pending = list(cast("list[Node]", node.args[3])) + list(
        cast("list[Node]", node.args[4])
    )
    seen: set[Node] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        if source in allocations:
            reject()
        pending.extend(source.all_input_nodes)

    tickets: set[Node] = set()
    symbols: set[Node] = set()
    proved: set[Node] = set()

    def uniform(value: object) -> None:
        if isinstance(value, (int, bool)):
            return
        if not isinstance(value, Node) or value.graph is not node.graph:
            reject()
        if value in proved:
            return
        proved.add(value)
        fake = value.meta.get("val")
        if value.target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            if not isinstance(fake, (int, torch.SymInt)):
                reject()
            symbols.add(value)
            return
        if (
            not isinstance(fake, torch.Tensor)
            or fake.ndim != 0
            or fake.dtype
            not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            reject()
        if value.target is atomic_ops.atomic_add:
            target, indices, contribution, sem = value.args
            if (
                not isinstance(target, Node)
                or target.target is not _tracing_ops._host_tensor
                or fake.dtype != torch.int32
                or sem not in ("acquire", "acq_rel")
            ):
                reject()
            tickets.add(value)
            for index in cast("list[object]", indices):
                uniform(index)
            uniform(contribution)
            return
        if value.target is not _tracing_ops._new_var and not (
            isinstance(value.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in value.target.tags
            and torch.Tag.nondeterministic_seeded not in value.target.tags
            and not value.target._schema.is_mutable
        ):
            reject()
        for source in value.all_input_nodes:
            uniform(source)

    uniform(node.args[0])
    if len(tickets) != 1:
        reject()
    return frozenset(tickets), frozenset(symbols)


def local_buffer_conditional_inputs(
    node: Node, graphs: list[GraphInfo]
) -> tuple[frozenset[Node], frozenset[Node]] | None:
    """Admit a terminal uniform branch controlled by completed local reductions.

    This is separate from ordered global-ticket finalizers. Only direct, complete
    local allocations may cross this branch as read-only captures. Fresh mutable
    targets must stay wholly in one immediate arm; no allocation alias escapes.
    Scalar shape is necessary but not sufficient: the emitter snapshots the final
    predicate into one shared slot before any branch code can reuse its storage.
    """
    allocations = local_atomic_allocations(graphs)

    def dependencies(value: object) -> set[Node]:
        pending = [value] if isinstance(value, Node) else []
        seen: set[Node] = set()
        while pending:
            source = pending.pop()
            if source not in seen:
                seen.add(source)
                pending.extend(source.all_input_nodes)
        return seen

    if not dependencies(node.args[0]) & allocations:
        return None

    def reject() -> NoReturn:
        raise exc.InvalidConfig(
            "local-buffer conditionals require terminal uniform branches, "
            "complete allocation captures and uniform scalar local reductions"
        )

    by_id = {info.graph_id: info for info in _reachable_graphs(graphs)}
    root = next(info for info in by_id.values() if info.graph is node.graph)
    following = list(node.graph.nodes)
    if (
        not isinstance(root, RootGraphInfo)
        or node.users
        or any(n.op != "output" for n in following[following.index(node) + 1 :])
    ):
        reject()
    info = by_id[cast("int", node.args[1])]
    if not isinstance(info, IfGraphInfo) or info.branches_outputs:
        reject()
    targets = atomic_target_origins(graphs)
    for graph_id in node.args[1:3]:
        branch = by_id[cast("int", graph_id)]
        for child in branch.graph.nodes:
            if child.target is _tracing_ops._if or _tracing_ops.is_for_loop_target(
                child.target
            ):
                reject()
            if child.target is atomic_ops.atomic_add:
                target = targets[child]
                # Ordered/global operations and mutation through parent captures
                # remain outside this proof. Each target has one lexical arm.
                if (
                    target.graph is not branch.graph
                    or target.target is not creation_ops.full
                ):
                    reject()

    def direct(value: Node) -> Node:
        while value.target is _tracing_ops._new_var:
            value = cast("Node", value.args[0])
        return value

    reductions: set[Node] = set()
    symbols: set[Node] = set()
    proved: set[Node] = set()

    def uniform(value: object) -> None:
        if isinstance(value, (int, bool)):
            return
        if not isinstance(value, Node) or value.graph is not node.graph:
            reject()
        if value in proved:
            return
        fake = value.meta.get("val")
        if value.target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            if not isinstance(fake, (int, torch.SymInt)):
                reject()
            symbols.add(value)
            proved.add(value)
            return
        if (
            not isinstance(fake, torch.Tensor)
            or fake.ndim != 0
            or fake.dtype not in (torch.bool, torch.int32, torch.int64)
        ):
            reject()
        if value.target in (
            torch.ops.aten.sum.default,
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.amax.default,
            torch.ops.aten.amin.default,
        ):
            source = value.args[0]
            # Reduction neutralization reads the same complete allocation. It
            # changes invalid padding values, never its coordinates or lifetime.
            if (
                isinstance(source, Node)
                and source.target is _tracing_ops._mask_to
                and isinstance(source.args[1], int)
            ):
                source = source.args[0]
            if (
                not isinstance(source, Node)
                or direct(source) not in allocations
                or direct(source).graph is not node.graph
                or cast("torch.Tensor", source.meta["val"]).dtype != torch.int32
            ):
                reject()
            reductions.add(value)
        else:
            if value.target is not _tracing_ops._new_var and not (
                isinstance(value.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in value.target.tags
                and torch.Tag.nondeterministic_seeded not in value.target.tags
                and not value.target._schema.is_mutable
            ):
                reject()
            for source in value.all_input_nodes:
                uniform(source)
        proved.add(value)

    uniform(node.args[0])
    if not reductions:
        reject()
    for values in node.args[3:5]:
        for source in cast("list[Node]", values):
            if dependencies(source) & allocations and direct(source) not in allocations:
                # A scalar already proved uniform is safe to read, but a lazy
                # tensor view/pointwise capture is deliberately outside this scope.
                uniform(source)
    return frozenset(reductions), frozenset(symbols)


def terminal_loop_symbols(node: Node, graphs: list[GraphInfo]) -> frozenset[Node]:
    """Prove initialized scalar carries and a read-only, straight-line body.

    The emitter checks the returned symbolic origins and scalar loop geometry.
    Existing shared carry snapshots/copies provide uniform collective epochs.
    """

    def reject() -> NoReturn:
        raise exc.InvalidConfig(
            "local finalizer loops require initialized scalar carries and a "
            "read-only scalar body with uniform bounds"
        )

    info = next(info for info in graphs if info.graph_id == node.args[0])
    if type(info) is not ForLoopGraphInfo or len(info.block_ids) != 1:
        reject()
    symbols: set[Node] = set()

    def bound(value: object) -> None:
        if isinstance(value, int):
            return
        if not isinstance(value, Node) or not isinstance(
            value.meta.get("val"), (int, torch.SymInt)
        ):
            reject()
        if value.target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            symbols.add(value)
            return
        if value.target not in (
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.mod,
            operator.neg,
            operator.pos,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            min,
            max,
        ):
            reject()
        for source in value.all_input_nodes:
            bound(source)

    for values in node.args[1:3]:
        for value in cast("list[object]", values):
            bound(value)
    if node.target is _tracing_ops._for_loop_step:
        for value in cast("list[object]", node.args[4]):
            if value is not None:
                bound(value)
    output = info.graph.find_nodes(op="output")[0]
    captures = cast("list[Node]", node.args[3])
    outputs = cast("list[Node]", output.args[0])
    initialized: set[int] = set()
    for item in node.users:
        if item.target is not operator.getitem:
            reject()
        for user in item.users:
            if user.target is _tracing_ops._phi and user.args[1] is item:
                initial = cast("Node", user.args[0])
                fake = initial.meta.get("val")
                if (
                    initial not in captures
                    or not isinstance(fake, torch.Tensor)
                    or fake.ndim
                ):
                    reject()
                initialized.add(cast("int", item.args[1]))
    if initialized != set(range(len(outputs))):
        reject()
    for item in info.graph.nodes:
        if item.op in ("placeholder", "output") or item.target in (
            _tracing_ops._host_tensor,
            _tracing_ops._new_var,
        ):
            continue
        if isinstance(item.meta.get("val"), (int, torch.SymInt)):
            bound(item)
            continue
        fake = item.meta.get("val")
        if not isinstance(fake, torch.Tensor) or fake.ndim:
            reject()
        if item.target in (memory_ops.load, creation_ops.full):
            continue
        if not (
            isinstance(item.target, torch._ops.OpOverload)
            and (
                torch.Tag.pointwise in item.target.tags
                or item.target
                in (torch.ops.aten.scalar_tensor.default, torch.ops.aten.full.default)
            )
            and torch.Tag.nondeterministic_seeded not in item.target.tags
            and not item.target._schema.is_mutable
        ):
            reject()
    return frozenset(symbols)


def prove_local_atomics(graphs: list[GraphInfo]) -> frozenset[Node]:
    """Allow direct allocations, uniform captures, and final read-only consumers.

    View aliases, captured conditional mutation and reads before later mutation
    are excluded. Int32 atomic returns are independent materialized snapshots.
    Each allocation lives in its root or one immediate uniform branch arm.
    Root allocations are captured unchanged through scalar loops. A terminal
    uniform finalizer may consume fresh global data without local captures.
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
    local_branches: set[Node] = set()
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
        for node in info.graph.nodes:
            if node.target is _tracing_ops._if:
                if local_buffer_conditional_inputs(node, graphs) is not None:
                    local_branches.add(node)
                else:
                    terminal_finalizer_inputs(node, graphs)

    local_regions = {
        cast("int", graph_id)
        for branch in local_branches
        for graph_id in branch.args[1:3]
    }

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
            or (
                by_graph[allocation.graph].graph_id in parents
                and by_graph[allocation.graph].graph_id not in local_regions
            )
        ):
            reject("a constant one-dimensional root/arm allocation of int32/float32")
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
                        # Branch outputs can contain dead lexical locals. For
                        # local_regions the parent proof has no users or merged
                        # outputs, so none escapes. Loop identities are checked
                        # separately by the existing phi/capture rules.
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
                    if user in local_branches:
                        if alias not in (*user.args[3], *user.args[4]):
                            reject("an unchanged read-only conditional capture")
                        reads.append(positions[user])
                        continue
                    if user.target is atomic_ops.atomic_add and user.args[0] is alias:
                        if user.args[2] is alias or alias in user.args[1]:
                            reject("updates that do not read their mutable target")
                        if user.args[3] != "relaxed":
                            reject("relaxed local atomics")
                        if user.users and dtype != torch.int32:
                            reject("int32 returned local atomic values")
                        parent = user
                        while parent.graph is not allocation.graph:
                            entry = parents.get(by_graph[parent.graph].graph_id)
                            if entry is None:
                                reject("one root owner")
                            parent = entry[0]
                        updates.append(positions[parent])
                        continue
                    read_entry = parents.get(info.graph_id)
                    read_position = (
                        positions[user]
                        if user.graph is allocation.graph
                        else positions[read_entry[0]]
                        if read_entry is not None
                        and read_entry[0] in local_branches
                        and read_entry[0].graph is allocation.graph
                        else None
                    )
                    if (
                        read_position is not None
                        and info.graph_id in local_regions
                        and user.target is atomic_ops.atomic_add
                        and user.args[0] is not alias
                        and (user.args[2] is alias or alias in user.args[1])
                    ):
                        # The branch proof already restricts the destination to
                        # a fresh target in this arm. Captured/other completed
                        # buffers are read-only index/contribution sources.
                        reads.append(read_position)
                        continue
                    if (
                        read_position is not None
                        and user.target in (memory_ops.store, atomic_ops.atomic_add)
                        and user.args[2] is alias
                        and cast("Node", user.args[0]).target
                        is _tracing_ops._host_tensor
                    ):
                        reads.append(read_position)
                        continue
                    if read_position is not None and (
                        user.target
                        in (scan_ops._associative_scan, _tracing_ops._mask_to)
                        or (
                            isinstance(user.target, torch._ops.OpOverload)
                            and not user.target._schema.is_mutable
                            and isinstance(
                                user.meta.get("lowering"),
                                (PointwiseLowering, ReductionLowering),
                            )
                        )
                    ):
                        # Complete-fragment admission separately validates the
                        # supported scan/reduction and pure pointwise lowering.
                        # These tensor producers only read the resident target.
                        # The fragment emitter completes pending updates before
                        # reads. Dependencies keep the allocation live through
                        # every derived read, including lazy tensor recipes.
                        reads.append(read_position)
                        continue
                    reject("only read-only root consumers after local updates")
        if not updates or any(position <= max(updates) for position in reads):
            reject("initialization, then updates, then final reads")
    return allocations
