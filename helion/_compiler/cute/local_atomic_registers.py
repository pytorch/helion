"""Same-coordinate lifetimes for private Int32 atomic return registers."""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

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

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo

# A register-pressure/code-size budget, independent of kernel or input shape.
MAX_SLOTS = 32


def local_atomic_register_chain(
    node: Node,
    env: CompileEnvironment,
    targets: dict[Node, Node],
    *,
    dead_outputs: bool,
) -> frozenset[Node]:
    """Keep all returned elements in their original physical owner.

    Callers separately prove the root's local allocation/epoch lifetime and
    configured slot budget. Only same-shape pointwise consumers, terminal
    host stores and unused private Int32 atomic updates are admitted. A store
    may use the result as its address or mask, but each expression is evaluated
    at the same coordinate.
    No loads, returned/global/ordered/Float32 atomic consumers, reductions,
    scans, remapping views or graph outputs may consume a register result.
    """
    fake = node.meta.get("val")
    if (
        node.target is not atomic_ops.atomic_add
        or not node.users
        or node.args[3] != "relaxed"
        or not isinstance(fake, torch.Tensor)
        or fake.dtype != torch.int32
        or not fake.ndim
        or targets[node].target is not creation_ops.full
    ):
        return frozenset()

    def same_size(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
        if env.known_equal(left, right):
            return True

        def logical(size: int | torch.SymInt) -> sympy.Expr:
            expr = (
                size._sympy_()
                if isinstance(size, torch.SymInt)
                else sympy.Integer(size)
            )
            # Indexed stores can normalize a complete static axis to a reduction
            # symbol. Match the emitter's full logical numel, never a tile hint
            # or a configuration-dependent non-reduction block size.
            return cast(
                "sympy.Expr", configured_fragment_expr(env, expr, lambda _block: None)
            )

        return env.specialize_expr(logical(left)) == env.specialize_expr(logical(right))

    def same_shape(shape: object) -> bool:
        sizes = cast("tuple[int | torch.SymInt, ...]", shape)
        return len(sizes) == fake.ndim and all(
            starmap(same_size, zip(fake.shape, sizes, strict=True))
        )

    seen = {node}
    pending = list(node.users)
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if user.graph is not node.graph:
            return frozenset()
        if user.op == "output" and dead_outputs:
            # The terminal local-buffer parent has no users or merged outputs.
            # Its lexical output list may still name dead arm-local temporaries.
            continue
        if user.target is memory_ops.store:
            host = cast("Node", user.args[0])
            if host.target is not _tracing_ops._host_tensor or user.users:
                return frozenset()
            indices = [
                index.meta["val"] if isinstance(index, Node) else index
                for index in cast("list[object]", user.args[1])
            ]
            if not same_shape(
                SubscriptIndexing.compute_shape(host.meta["val"], indices)
            ):
                return frozenset()
            continue
        if user.target is atomic_ops.atomic_add:
            target = targets[user]
            value = user.meta["val"]
            if (
                user.users
                or user.args[3] != "relaxed"
                or target.target is not creation_ops.full
                or target.meta["val"].dtype != torch.int32
                or not isinstance(value, torch.Tensor)
                or not same_shape(value.shape)
            ):
                return frozenset()
            continue
        value = user.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or not same_shape(value.shape)
            or not user.users
            or isinstance(user.meta.get("lowering"), ReductionLowering)
        ):
            return frozenset()
        if user.target not in (_tracing_ops._new_var, torch.ops.aten.alias.default):
            if not (
                isinstance(user.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in user.target.tags
                and torch.Tag.nondeterministic_seeded not in user.target.tags
                and not user.target._schema.is_mutable
            ):
                return frozenset()
        pending.extend(user.users)
    return frozenset(seen)


def local_atomic_register_chains(
    graphs: list[GraphInfo], env: CompileEnvironment
) -> dict[Node, frozenset[Node]]:
    """Share exact lexical admission between coverage and emission.

    The caller still proves local allocation lifetimes. Branch register values
    stay inside an immediate, already proved uniform arm: they never become a
    capture, loop carry or merged result. Global-ticket finalizers are unchanged.
    """
    by_id = {graph.graph_id: graph for graph in graphs}
    admitted: list[GraphInfo] = [
        graph for graph in graphs if isinstance(graph, RootGraphInfo)
    ]
    for root in tuple(admitted):
        for node in root.graph.nodes:
            if (
                node.target is _tracing_ops._if
                and local_buffer_conditional_inputs(node, graphs) is not None
            ):
                admitted.extend(by_id[cast("int", index)] for index in node.args[1:3])
    targets = atomic_target_origins(graphs)
    return {
        node: chain
        for graph in admitted
        for node in graph.graph.nodes
        if (
            chain := local_atomic_register_chain(
                node, env, targets, dead_outputs=not isinstance(graph, RootGraphInfo)
            )
        )
    }
