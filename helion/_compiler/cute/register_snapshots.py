"""Bounded immutable host snapshots with explicit per-thread coordinates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

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
            or cast("Node", load.args[0]).target is not _tracing_ops._host_tensor
            or not host_load_is_readonly(load, env, graphs, allow_unbound=allow_unbound)
        ):
            return frozenset()

        logical = snapshot_logical_shape(load)

        def same(shape: object) -> bool:
            sizes = cast("tuple[int | torch.SymInt, ...]", shape)
            return len(sizes) == 1 and dimension(sizes[0]) == dimension(fake.shape[0])

        seen = {load}
        pending = list(load.users)
        dead_graphs = set()
        while pending:
            node = pending.pop()
            if node in seen and node.target is not _tracing_ops._if:
                continue
            seen.add(node)
            value = node.meta.get("val")
            if node.target is _tracing_ops._if:
                if node.users or local_buffer_conditional_inputs(node, graphs) is None:
                    return frozenset()
                for side in range(2):
                    branch = by_id[cast("int", node.args[1 + side])]
                    dead_graphs.add(branch.graph)
                    for origin, placeholder in zip(
                        cast("list[Node]", node.args[3 + side]),
                        branch.graph.find_nodes(op="placeholder"),
                        strict=True,
                    ):
                        if origin in seen and placeholder not in seen:
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

    return {
        node: found
        for graph in graphs
        if isinstance(graph, RootGraphInfo)
        for node in graph.graph.nodes
        if (found := chain(node))
    }
