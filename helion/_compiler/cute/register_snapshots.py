"""Bounded immutable host snapshots with explicit per-thread coordinates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
from functools import cache
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node
from torch.fx.node import map_arg

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import memory_ops
from ..device_ir import RootGraphInfo
from ..indexing_strategy import SubscriptIndexing
from ..inductor_lowering import ReductionLowering
from .fragment_storage import configured_fragment_expr
from .local_atomic import atomic_target_origins
from .local_atomic import local_buffer_conditional_inputs
from .register_loads import host_load_is_readonly

if TYPE_CHECKING:
    from torch._inductor.ir import Reduction

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo

MAX_SLOTS = 32


@dataclass
class SnapshotOwner:
    """Only the active, in-bounds coordinate can read this thread's slot."""

    index: str
    slot: str
    size: int
    aliases: dict[str, str] = field(default_factory=dict)

    def prove(self, coordinate: str) -> None:
        symbol = sympy.Symbol(self.index, integer=True, nonnegative=True)

        def parse(node: ast.expr) -> sympy.Expr:
            if isinstance(node, ast.Name):
                if node.id == self.index:
                    return symbol
                if node.id in self.aliases:
                    return parse(ast.parse(self.aliases[node.id], mode="eval").body)
            elif isinstance(node, ast.Constant) and type(node.value) is int:
                return sympy.Integer(node.value)
            elif isinstance(node, ast.BinOp):
                left, right = parse(node.left), parse(node.right)
                if isinstance(node.op, ast.Add):
                    return sympy.expand(sympy.Add(left, right))
                if isinstance(node.op, ast.Sub):
                    return sympy.expand(sympy.Add(left, sympy.Mul(-1, right)))
                if isinstance(node.op, ast.Mult):
                    return sympy.expand(sympy.Mul(left, right))
                if isinstance(node.op, ast.FloorDiv) and right == 1:
                    return left
                if (
                    isinstance(node.op, ast.Mod)
                    and right.is_Integer
                    and int(cast("sympy.Integer", right)) >= self.size
                    and left == symbol
                ):
                    return symbol  # The active element loop proves 0 <= i < size.
            raise exc.InvalidConfig(
                "register snapshot coordinate is not its active owner"
            )

        result = parse(ast.parse(coordinate, mode="eval").body)
        if self.size == 1:
            result = result.subs(symbol, 0)
            symbol = sympy.Integer(0)
        if result != symbol:
            raise exc.InvalidConfig(
                "register snapshot coordinate is not its active owner"
            )


@dataclass(frozen=True)
class SnapshotBinding:
    """One initialized immutable rmem value, with the existing striped owner map."""

    source: Node
    name: str
    threads: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    domain_storage: frozenset[str]
    logical_shape: tuple[int, ...]


def frame_snapshot_captures(
    call: Node, graphs: list[GraphInfo], source: Node
) -> dict[int, tuple[int, ...]]:
    """Current-call invariant capture edges to one dominating initialized load.

    A register value is never a vector carry/branch result. Every edge follows
    the existing frame tree, with physical reads still checked by SnapshotOwner.
    """
    from .uniform_region_tree import uniform_local_regions

    try:
        tree = uniform_local_regions(graphs)
    except exc.InvalidConfig:
        return {}
    frames = {frame.graph.graph: frame for frame in tree.frames}
    if source.graph not in frames or call.graph not in frames:
        return {}
    if source.target is not memory_ops.load:
        return {}
    orders = {
        graph: {node: i for i, node in enumerate(graph.nodes)} for graph in frames
    }
    if source not in orders[source.graph] or call not in orders[call.graph]:
        return {}

    def ancestor(value: Node) -> Node | None:
        frame = frames.get(value.graph)
        if frame is None:
            return None
        if value in frame.placeholders:
            slot = frame.placeholders.index(value)
            if any(carried == slot for _, carried in frame.carry_map):
                return None
            return frame.captures[slot]
        if (
            value.target in (_tracing_ops._new_var, torch.ops.aten.alias.default)
            and len(value.args) == 1
            and not value.kwargs
            and isinstance(value.args[0], Node)
            and value.args[0].graph is value.graph
            and orders[value.graph][value.args[0]] < orders[value.graph][value]
        ):
            return value.args[0]
        return None

    def identity(value: Node) -> bool:
        seen: set[Node] = set()
        while value is not source:
            if value in seen:
                return False
            seen.add(value)
            parent = ancestor(value)
            if parent is None:
                return False
            left, right = value.meta.get("val"), parent.meta.get("val")
            if (
                not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or left.dtype != right.dtype
                or left.shape != right.shape
                or left.device != right.device
                or left.layout != right.layout
            ):
                return False
            value = parent
        return True

    # Walk outward to establish lexical dominance at the actual call site.
    position = call
    while position.graph is not source.graph:
        frame = frames.get(position.graph)
        if frame is None or frame.call is None:
            return {}
        position = frame.call
    if orders[source.graph][source] >= orders[source.graph][position]:
        return {}

    # Tensor coordinates must be direct iotas (possibly invariant captures).
    # Scalar coordinates use existing stable grid symbols, never loop carries.
    for index in cast("list[object]", source.args[1]):
        if not isinstance(index, Node):
            if type(index) is not int and index is not None and index != slice(None):
                return {}
            continue
        value = index
        seen: set[Node] = set()
        while (parent := ancestor(value)) is not None:
            if value in seen:
                return {}
            seen.add(value)
            value = parent
        if isinstance(index.meta.get("val"), torch.Tensor):
            if value.target is not torch.ops.prims.iota.default:
                return {}
        elif value.target is not _tracing_ops._get_symnode:
            return {}

    return {
        frame.graph.graph_id: tuple(
            slot
            for slot, entry in enumerate(frame.captures)
            if not any(carried == slot for _, carried in frame.carry_map)
            and identity(entry)
        )
        for frame in tree.frames
        if frame.call is call
    }


def frame_snapshot_recipes(
    call: Node,
    graphs: list[GraphInfo],
    env: CompileEnvironment,
    *,
    allow_unbound: bool = False,
) -> dict[int, dict[int, frozenset[Node]]]:
    """Typed same-owner recipes over initialized snapshots and invariant leaves.

    This proof does not turn a derived value into a load identity. The emitter
    separately initializes its own typed register slots at this actual call.
    Mutable carries, joins, remapped coordinates and arbitrary shared values
    are deliberately outside the recipe grammar.
    """
    from .uniform_region_tree import uniform_local_regions

    try:
        tree = uniform_local_regions(graphs)
    except exc.InvalidConfig:
        return {}
    frames = {frame.graph.graph: frame for frame in tree.frames}
    orders = {
        graph: {node: i for i, node in enumerate(graph.nodes)} for graph in frames
    }
    if call.graph not in orders or call not in orders[call.graph]:
        return {}

    carried_entries = {
        frame.captures[slot]
        for frame in tree.frames
        if frame.call is call
        for _, slot in frame.carry_map
    }

    def same_type(left: Node, right: Node) -> bool:
        a, b = left.meta.get("val"), right.meta.get("val")
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and a.device == b.device
            and a.layout == b.layout
        )

    def dominates(value: Node, user: Node) -> bool:
        while user.graph is not value.graph:
            frame = frames.get(user.graph)
            if frame is None or frame.call is None:
                return False
            user = frame.call
        return (
            value in orders[value.graph]
            and orders[value.graph][value] < orders[user.graph][user]
        )

    def recipe(entry: Node) -> frozenset[Node] | None:
        result = entry.meta.get("val")
        if not isinstance(result, torch.Tensor) or result.ndim != 1:
            return None
        active: set[Node] = set()

        @cache
        def visit(value: Node) -> frozenset[Node] | None:
            if value in active or value in carried_entries or value.graph not in frames:
                return None
            fake = value.meta.get("val")
            if not isinstance(fake, torch.Tensor) or (
                fake.device != result.device
                or fake.layout != result.layout
                or (fake.ndim != 0 and fake.shape != result.shape)
            ):
                return None
            active.add(value)
            try:
                frame = frames[value.graph]
                if value in frame.placeholders:
                    slot = frame.placeholders.index(value)
                    if any(carried == slot for _, carried in frame.carry_map):
                        return None
                    parent = frame.captures[slot]
                    if not same_type(value, parent):
                        return None
                    return visit(parent)
                if not dominates(value, call):
                    return None
                if value.target is memory_ops.load:
                    if fake.ndim == 1:
                        if (
                            not isinstance(value.args[0], Node)
                            or value.args[0].target is not _tracing_ops._host_tensor
                            or not host_load_is_readonly(
                                value, env, graphs, allow_unbound=allow_unbound
                            )
                            or not frame_snapshot_captures(call, graphs, value)
                        ):
                            return None
                        return frozenset((value,))
                    # Scalar host values must be readonly and addressed by a
                    # stable row symbol/literal, never by a mutable carry.
                    if not host_load_is_readonly(
                        value, env, graphs, allow_unbound=allow_unbound
                    ) or any(
                        isinstance(index, Node)
                        and index.target is not _tracing_ops._get_symnode
                        for index in cast("list[object]", value.args[1])
                    ):
                        return None
                    return frozenset()
                if value.target is torch.ops.prims.iota.default:
                    return (
                        frozenset()
                        if all(not isinstance(arg, Node) for arg in value.args)
                        and all(
                            not isinstance(arg, Node) for arg in value.kwargs.values()
                        )
                        else None
                    )
                if value.target is torch.ops.aten.scalar_tensor.default:
                    return (
                        frozenset()
                        if fake.ndim == 0
                        and len(value.args) == 1
                        and type(value.args[0]) in (bool, int, float)
                        and all(
                            not isinstance(arg, Node) for arg in value.kwargs.values()
                        )
                        else None
                    )
                if value.target in (
                    _tracing_ops._new_var,
                    torch.ops.aten.alias.default,
                ):
                    if (
                        len(value.args) != 1
                        or value.kwargs
                        or not isinstance(value.args[0], Node)
                        or not same_type(value, value.args[0])
                    ):
                        return None
                elif value.target is torch.ops.aten.view.dtype:
                    source = value.args[0]
                    if (
                        not isinstance(source, Node)
                        or len(value.args) != 2
                        or value.kwargs
                        or source.meta["val"].shape != fake.shape
                        or source.meta["val"].element_size() != fake.element_size()
                        or value.args[1] != fake.dtype
                    ):
                        return None
                elif value.target is _tracing_ops._mask_to:
                    if len(value.args) != 2 or type(value.args[1]) not in (
                        int,
                        float,
                        bool,
                    ):
                        return None
                elif isinstance(value.meta.get("lowering"), ReductionLowering):
                    lowering = cast("ReductionLowering", value.meta["lowering"])
                    if (
                        fake.ndim != 0
                        or fake.dtype not in (torch.int32, torch.int64)
                        or lowering.reduction_type not in ("sum", "min", "max")
                    ):
                        return None
                elif not (
                    isinstance(value.target, torch._ops.OpOverload)
                    and torch.Tag.pointwise in value.target.tags
                    and torch.Tag.nondeterministic_seeded not in value.target.tags
                    and not value.target._schema.is_mutable
                ):
                    return None
                operands: list[Node] = []
                map_arg((value.args, value.kwargs), lambda n: operands.append(n))
                found: set[Node] = set()
                for operand in operands:
                    if operand.graph is not value.graph or not dominates(
                        operand, value
                    ):
                        return None
                    leaves = visit(operand)
                    if leaves is None:
                        return None
                    found.update(leaves)
                return frozenset(found)
            finally:
                active.remove(value)

        return visit(entry)

    return {
        frame.graph.graph_id: {
            slot: leaves
            for slot, entry in enumerate(frame.captures)
            if not any(carried == slot for _, carried in frame.carry_map)
            and (leaves := recipe(entry))
        }
        for frame in tree.frames
        if frame.call is call
    }


def snapshot_capture_slots(
    call: Node, graphs: list[GraphInfo], source: Node
) -> tuple[int, ...]:
    """Only a dominating direct snapshot (or typed identity) can cross a while.

    This is intentionally not a pure-recipe/LICM proof. The caller separately
    proves whole-region readonly storage and every physical coordinate read.
    """
    from .resident_while import resident_while_plan

    try:
        plan = resident_while_plan(call, graphs)
    except exc.InvalidConfig:
        return ()
    if plan.composed:
        return frame_snapshot_captures(call, graphs, source).get(
            plan.body.graph.graph_id, ()
        )
    if source.graph is not plan.root.graph:
        return ()
    # Limit persistent domains to the existing direct iota coordinates. A
    # loaded/remapped index can hide a shared domain dependency whose lifetime
    # is not represented by the copied snapshot value.
    if any(
        isinstance(index, Node)
        and isinstance(index.meta.get("val"), torch.Tensor)
        and index.target is not torch.ops.prims.iota.default
        for index in cast("list[object]", source.args[1])
    ):
        return ()

    order = {node: i for i, node in enumerate(plan.root.graph.nodes)}
    if source not in order or order[source] >= order[call]:
        return ()

    def identity(value: Node) -> bool:
        while value is not source:
            if (
                value.graph is not source.graph
                or value.target
                not in (_tracing_ops._new_var, torch.ops.aten.alias.default)
                or len(value.args) != 1
                or value.kwargs
                or not isinstance(value.args[0], Node)
            ):
                return False
            parent = value.args[0]
            if parent not in order or order[parent] >= order[value]:
                return False
            left, right = value.meta.get("val"), parent.meta.get("val")
            if (
                not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or left.dtype != right.dtype
                or left.shape != right.shape
            ):
                return False
            value = parent
        return True

    return tuple(
        slot for slot in plan.invariant_slots if identity(plan.body.captures[slot])
    )


def snapshot_logical_shape(node: Node) -> tuple[int | torch.SymInt, ...]:
    target = cast("Node", node.args[0]).meta["val"]
    indices = [
        index.meta["val"] if isinstance(index, Node) else index
        for index in cast("list[Node | int | slice | None]", node.args[1])
    ]
    return tuple(SubscriptIndexing.compute_shape(target, indices))


def snapshot_chains(
    graphs: list[GraphInfo], env: CompileEnvironment, *, allow_unbound: bool = False
) -> dict[Node, frozenset[Node]]:
    """A direct rank-one snapshot may only escape through approved uniform captures.

    Reduction outputs stop the chain: their small result stays CTA-shared.
    Runtime coordinate checks additionally validate every lowered register read.
    """
    by_id = {graph.graph_id: graph for graph in graphs}
    targets = atomic_target_origins(graphs)

    def dimension(size: int | torch.SymInt) -> sympy.Basic:
        expr = size._sympy_() if isinstance(size, torch.SymInt) else sympy.Integer(size)
        return env.specialize_expr(
            cast("sympy.Expr", configured_fragment_expr(env, expr, lambda _bid: None))
        )

    def chain(load: Node) -> frozenset[Node]:
        fake = load.meta.get("val")
        if (
            load.target is not memory_ops.load
            or not load.users
            or not isinstance(fake, torch.Tensor)
            or fake.ndim != 1
            or not isinstance(load.args[0], Node)
            or load.args[0].target is not _tracing_ops._host_tensor
            or not host_load_is_readonly(load, env, graphs, allow_unbound=allow_unbound)
        ):
            return frozenset()

        logical = snapshot_logical_shape(load)

        def same(shape: object) -> bool:
            sizes = cast("tuple[int | torch.SymInt, ...]", shape)
            return len(sizes) == 1 and dimension(sizes[0]) == dimension(fake.shape[0])

        seen = {load}
        pending = list(load.users)
        dead_graphs = set(branch_graphs)
        while pending:
            node = pending.pop()
            if node in seen and node.target not in (
                _tracing_ops._if,
                _tracing_ops._while_loop,
            ):
                continue
            seen.add(node)
            value = node.meta.get("val")
            if node.target is _tracing_ops._while_loop:
                slots = snapshot_capture_slots(node, graphs, load)
                recipes = frame_snapshot_recipes(
                    node, graphs, env, allow_unbound=allow_unbound
                )
                slots = tuple(
                    sorted(
                        set(slots)
                        | {
                            slot
                            for slots_by_graph in recipes.values()
                            for slot, leaves in slots_by_graph.items()
                            if load in leaves
                        }
                    )
                )
                if not slots:
                    return frozenset()
                captures = cast("list[Node]", node.args[2])
                # Every reached capture needs its own identity or typed recipe
                # proof; initial bits never establish mutable invariance.
                if any(
                    entry in seen and slot not in slots
                    for slot, entry in enumerate(captures)
                ):
                    return frozenset()
                used = False
                for graph_id in node.args[:2]:
                    child = by_id[cast("int", graph_id)]
                    placeholders = list(child.graph.find_nodes(op="placeholder"))
                    for slot in slots:
                        placeholder = placeholders[slot]
                        used |= bool(placeholder.users)
                        if placeholder not in seen:
                            seen.add(placeholder)
                            pending.extend(placeholder.users)
                if not used:
                    return frozenset()
                continue
            if node.target is _tracing_ops._if:
                try:
                    legacy = (
                        not node.users
                        and local_buffer_conditional_inputs(node, graphs) is not None
                    )
                except exc.InvalidConfig:
                    legacy = False
                frame_slots = (
                    {} if legacy else frame_snapshot_captures(node, graphs, load)
                )
                if not legacy:
                    recipes = frame_snapshot_recipes(
                        node, graphs, env, allow_unbound=allow_unbound
                    )
                    frame_slots = {
                        gid: tuple(
                            sorted(
                                set(frame_slots.get(gid, ()))
                                | {
                                    slot
                                    for slot, leaves in entries.items()
                                    if load in leaves
                                }
                            )
                        )
                        for gid, entries in recipes.items()
                    }
                if not legacy and not frame_slots:
                    return frozenset()
                for side in range(2):
                    branch = by_id[cast("int", node.args[1 + side])]
                    dead_graphs.add(branch.graph)
                    for slot, (origin, placeholder) in enumerate(
                        zip(
                            cast("list[Node]", node.args[3 + side]),
                            branch.graph.find_nodes(op="placeholder"),
                            strict=True,
                        )
                    ):
                        if origin in seen:
                            if not legacy and slot not in frame_slots.get(
                                branch.graph_id, ()
                            ):
                                return frozenset()
                            if placeholder not in seen:
                                seen.add(placeholder)
                                pending.extend(placeholder.users)
                continue
            if node.op == "output" and node.graph in dead_graphs:
                continue
            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                    or value.dtype not in (torch.int32, torch.int64)
                    or lowering.reduction_type not in ("sum", "min", "max")
                    or not same(
                        cast("Reduction", lowering.buffer.data).reduction_ranges
                    )
                ):
                    return frozenset()
                continue
            if node.target is memory_ops.store:
                target = cast("Node", node.args[0])
                indices = [
                    i.meta["val"] if isinstance(i, Node) else i
                    for i in cast("list[Node | int | slice | None]", node.args[1])
                ]
                shape = SubscriptIndexing.compute_shape(target.meta["val"], indices)
                if target.target is not _tracing_ops._host_tensor or not (
                    len(shape) == len(logical) == 1
                    and dimension(shape[0]) == dimension(logical[0])
                ):
                    return frozenset()
                continue
            if node.target is atomic_ops.atomic_add:
                target = targets[node]
                if (
                    target.target is not creation_ops.full
                    or target.meta["val"].dtype != torch.int32
                    or node.args[3] != "relaxed"
                    or not isinstance(value, torch.Tensor)
                    or not (
                        same(value.shape)
                        or (
                            len(value.shape) == len(logical) == 1
                            and dimension(value.shape[0]) == dimension(logical[0])
                        )
                    )
                ):
                    return frozenset()
                continue
            if not isinstance(value, torch.Tensor) or not same(value.shape):
                return frozenset()
            if node.target not in (
                _tracing_ops._new_var,
                _tracing_ops._mask_to,
                torch.ops.aten.alias.default,
                torch.ops.aten.view.dtype,
            ) and not (
                isinstance(node.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in node.target.tags
                and torch.Tag.nondeterministic_seeded not in node.target.tags
                and not node.target._schema.is_mutable
            ):
                return frozenset()
            if not node.users:
                return frozenset()
            pending.extend(node.users)
        return frozenset(seen)

    from .uniform_region_tree import uniform_local_regions

    try:
        tree = uniform_local_regions(graphs)
        frame_graphs = {frame.graph.graph for frame in tree.frames}
        # The actual branch join proof selects scalar outputs only. Other FX
        # output entries are dead branch temporaries, not escaping values.
        branch_graphs = {
            frame.graph.graph
            for frame in tree.frames
            if frame.role in ("if_true", "if_false")
        }
    except exc.InvalidConfig:
        frame_graphs = set()
        branch_graphs = set()
    return {
        node: found
        for graph in graphs
        if isinstance(graph, RootGraphInfo) or graph.graph in frame_graphs
        for node in graph.graph.nodes
        if (found := chain(node))
    }
