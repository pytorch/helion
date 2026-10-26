"""Same-owner opaque producer DAGs ending at one collective publication."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import memory_ops
from .producer_cache import pure_producer
from .register_loads import host_load_is_readonly

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx import Graph
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


@dataclass(frozen=True)
class PureProducerPlan:
    lazy: frozenset[Node] = frozenset()
    publications: frozenset[Node] = frozenset()
    replicated: frozenset[Node] = frozenset()


def opaque_producer(node: Node) -> bool:
    return (
        node.target is inline_asm_ops.inline_asm_elementwise
        and len(node.args) == 6
        and not node.kwargs
        and node.args[4] is True
        and type(node.args[5]) is int
        and node.args[5] == 1
        and isinstance(node.args[3], torch.dtype)
        and isinstance(node.meta.get("val"), torch.Tensor)
    )


def _pointwise(node: Node) -> bool:
    # A view can preserve shape while changing the owner (e.g. square transpose).
    # The initial scope admits only pointwise identity coordinate maps, not views.
    return node.target is _tracing_ops._new_var or (
        pure_producer(node)
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    )


def _same_owner_plan(
    graph: Graph,
    env: CompileEnvironment,
    *,
    cache_nodes: frozenset[Node] = frozenset(),
    shape: Callable[[object], tuple[int, ...]] | None = None,
) -> PureProducerPlan:
    """Do not move an opaque operation to a different physical thread.

    Each selected producer has exactly one terminal, with identity pointwise
    coordinates and equal logical/physical shape on every path. That terminal
    is materialized with the same ordinary elements() owner as the original
    producer. Broadcasts, remaps, collectives and escaping multiple consumers
    remain publications. Every operation between definition and publication is
    pure; loads, mutations and control edges remain epoch boundaries.

    Publications are a fixed point: adding an earlier boundary must not make
    another producer execute once there and again in a later consumer.
    """
    ordered = tuple(graph.nodes)
    positions = {node: index for index, node in enumerate(ordered)}
    if any(
        _tracing_ops.is_for_loop_target(node.target)
        or node.target in (_tracing_ops._if, _tracing_ops._while_loop)
        for node in ordered
    ):
        return PureProducerPlan()
    opaque = frozenset(node for node in ordered if opaque_producer(node))
    if not opaque:
        return PureProducerPlan()

    def same_shape(left: Node, right: Node) -> bool:
        a, b = left.meta.get("val"), right.meta.get("val")
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        if a.ndim != b.ndim or not all(
            starmap(env.known_equal, zip(a.shape, b.shape, strict=True))
        ):
            return False
        if shape is not None:
            x, y = shape(a.shape), shape(b.shape)
            return x == y and all(size > 0 for size in x)
        return True

    def movable(node: Node) -> bool:
        return opaque_producer(node) or _pointwise(node)

    boundaries = set(cache_nodes)
    while True:
        additions: set[Node] = set()
        selected: set[Node] = set()
        for producer in opaque:
            if producer in boundaries:
                continue
            seen: dict[Node, frozenset[Node]] = {}

            def terminals(
                node: Node,
                producer: Node = producer,
                seen: dict[Node, frozenset[Node]] = seen,
            ) -> frozenset[Node]:
                if node in seen:
                    return seen[node]
                if (
                    node in boundaries
                    or not node.users
                    or any(
                        user.graph is not graph
                        or not movable(user)
                        or not same_shape(producer, user)
                        for user in node.users
                    )
                ):
                    result = frozenset((node,))
                else:
                    result = frozenset(
                        terminal for user in node.users for terminal in terminals(user)
                    )
                seen[node] = result
                return result

            sinks = terminals(producer)
            if len(sinks) != 1:
                continue
            terminal = next(iter(sinks))
            if terminal is producer or not same_shape(producer, terminal):
                continue
            interval = ordered[positions[producer] : positions[terminal] + 1]
            if any(not movable(node) for node in interval):
                continue
            selected.add(producer)
            additions.add(terminal)
        new = additions - boundaries
        if not new:
            # Only needed publications are forced; unrelated old caches remain
            # governed by the existing configuration and allocator.
            return PureProducerPlan(frozenset(selected), frozenset(additions))
        boundaries.update(new)


def pure_producer_plan(
    graph: Graph,
    env: CompileEnvironment,
    *,
    cache_nodes: frozenset[Node] = frozenset(),
    shape: Callable[[object], tuple[int, ...]] | None = None,
    graphs: Sequence[GraphInfo] = (),
) -> PureProducerPlan:
    """Replicate scalar arithmetic only within one proved owner publication.

    Scalar shape alone is not uniformity. Roots are literal constants or an
    already materialized, readonly host scalar load. Opaque programs, views,
    reductions, mutable values and captures are not uniform recipes. In
    particular, a pure scalar ASM may observe its executing lane and cannot
    become a recipe evaluated by another lane.

    Removing a scalar cache can join adjacent same-owner regions. Recompute
    their publication fixed point and retain a removal only when its entire
    consumer closure reaches one such publication, with no intervening effect
    or remap. All other producer caches remain unchanged.
    """
    if not cache_nodes:
        return _same_owner_plan(graph, env, cache_nodes=cache_nodes, shape=shape)
    ordered = tuple(graph.nodes)
    positions = {node: index for index, node in enumerate(ordered)}

    def scalar(node: Node) -> bool:
        value = node.meta.get("val")
        return (
            isinstance(value, torch.Tensor)
            and value.ndim == 0
            and (shape is None or shape(value.shape) == ())
        )

    uniform: set[Node] = set()
    for node in ordered:
        if not scalar(node):
            continue
        if node.target in (
            creation_ops.full,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
        ):
            # Do not infer immutability from metadata or an arbitrary host
            # expression. These creation nodes contain the actual literal.
            index = 0 if node.target is torch.ops.aten.scalar_tensor.default else 1
            if (
                len(node.args) > index
                and type(node.args[index]) in (bool, int, float)
                and not node.all_input_nodes
            ):
                uniform.add(node)
        elif node.target is memory_ops.load:
            source = node.args[0]
            if (
                isinstance(source, torch.fx.Node)
                and source in positions
                and source.target is _tracing_ops._host_tensor
                and graphs
                and host_load_is_readonly(node, env, graphs)
            ):
                # Rank-zero loads cannot enter lane_private_load or snapshot
                # schedules. Keep the existing load and its shared publication;
                # only dependent arithmetic is replicated, never this load.
                uniform.add(node)
        elif (
            _pointwise(node)
            and node.all_input_nodes
            and all(value in uniform for value in node.all_input_nodes)
        ):
            uniform.add(node)

    removed = {node for node in cache_nodes if node in uniform and _pointwise(node)}
    while True:
        remaining = cache_nodes - removed
        plan = _same_owner_plan(graph, env, cache_nodes=remaining, shape=shape)
        boundaries = remaining | plan.publications
        rejected: set[Node] = set()
        for candidate in removed:
            visited: set[Node] = set()
            terminals: set[Node] = set()
            pending = [candidate]
            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                if (
                    node in boundaries
                    or not node.users
                    or any(
                        user.graph is not graph
                        or not (opaque_producer(user) or _pointwise(user))
                        for user in node.users
                    )
                ):
                    terminals.add(node)
                else:
                    pending.extend(node.users)
            if len(terminals) != 1 or not visited.intersection(plan.lazy):
                rejected.add(candidate)
                continue
            terminal = next(iter(terminals))
            target = terminal.meta.get("val")
            if (
                terminal not in plan.publications
                or not isinstance(target, torch.Tensor)
                or target.ndim == 0
                or positions[terminal] <= positions[candidate]
            ):
                rejected.add(candidate)
                continue
            for node in visited:
                if node in uniform:
                    continue
                value = node.meta.get("val")
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != target.ndim
                    or not all(
                        starmap(
                            env.known_equal,
                            zip(value.shape, target.shape, strict=True),
                        )
                    )
                    or (shape is not None and shape(value.shape) != shape(target.shape))
                ):
                    rejected.add(candidate)
                    break
            interval = ordered[positions[candidate] : positions[terminal] + 1]
            if any(
                not (opaque_producer(node) or _pointwise(node)) for node in interval
            ):
                rejected.add(candidate)
        if not rejected:
            return PureProducerPlan(plan.lazy, plan.publications, frozenset(removed))
        removed.difference_update(rejected)
