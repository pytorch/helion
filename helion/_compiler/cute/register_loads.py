"""Bounded lane-private load lifetimes for complete fragment roots."""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import memory_ops
from ..indexing_strategy import SubscriptIndexing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


def lane_private_load(node: Node, env: CompileEnvironment) -> bool:
    """Every consumer uses the same logical coordinate in this lexical graph.

    No value escapes through a carry, branch, opaque program, scan, reduction,
    shape-changing view, or addressable alias. The emitter separately proves
    at most one element per physical thread for the actual configuration.
    """
    if node.target is not memory_ops.load or not node.users:
        return False
    source = cast("Node", node.args[0])
    if source.target is not _tracing_ops._host_tensor:
        return False
    fake = node.meta.get("val")
    if not isinstance(fake, torch.Tensor) or not fake.ndim:
        return False

    def same_shape(shape: Sequence[int | torch.SymInt]) -> bool:
        return len(shape) == fake.ndim and all(
            starmap(env.known_equal, zip(fake.shape, shape, strict=True))
        )

    visited: set[Node] = set()
    pending = list(node.users)
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        if user.graph is not node.graph:
            return False
        value = user.meta.get("val")
        if user.target is atomic_ops.atomic_add:
            if (
                user.users
                or user.args[3] != "relaxed"
                or user.args[0] is node
                or user.args[0] in visited
                or not isinstance(value, torch.Tensor)
                or not same_shape(value.shape)
            ):
                return False
            continue
        if user.target is memory_ops.store:
            if user.args[2] not in visited and user.args[2] is not node:
                return False
            target = cast("Node", user.args[0])
            if target.target is not _tracing_ops._host_tensor:
                return False
            indices = [
                x.meta["val"] if isinstance(x, Node) else x
                for x in cast("list[object]", user.args[1])
            ]
            shape = SubscriptIndexing.compute_shape(target.meta["val"], indices)
            if not same_shape(shape):
                return False
            continue
        if not isinstance(value, torch.Tensor) or not same_shape(value.shape):
            return False
        if user.target not in (
            _tracing_ops._new_var,
            torch.ops.aten.alias.default,
            torch.ops.aten.view.dtype,
        ):
            if not (
                isinstance(user.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in user.target.tags
                and not user.target._schema.is_mutable
            ):
                return False
        if not user.users:
            return False
        pending.extend(user.users)
    return True


def host_load_is_readonly(
    node: Node,
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    *,
    allow_unbound: bool = False,
) -> bool:
    """Removing the load barrier must not reorder a possibly aliased write.

    Complete fragment roots have explicit store/atomic effects. Inspect the
    entire region, including nested loops and effects outside the value's user
    DAG. This deliberately declines even same-index writes: shape equality is
    not an address or cross-lane ordering proof. Local shared atomics cannot
    alias a host storage; their normal initialization/update barriers remain.
    """
    from ...language import creation_ops
    from .local_atomic import _reachable_graphs
    from .local_atomic import atomic_target_origins
    from .memory_ops import runtime_tensors_are_proven_disjoint

    source = cast("Node", node.args[0]).meta.get("val")
    if not isinstance(source, torch.Tensor):
        return False
    # Analysis IR can retain inactive rolled alternatives. Use the same
    # root/control-flow projection as the complete-fragment ownership proof.
    graphs = _reachable_graphs(list(graphs))
    if not any(node.graph is graph.graph for graph in graphs):
        return False
    nodes = [item for graph in graphs for item in graph.graph.nodes]
    atomic_targets = atomic_target_origins(list(graphs))
    source_storage = source.untyped_storage()
    for effect in nodes:
        if (
            effect.target is not memory_ops.store
            and effect.target not in atomic_ops.ATOMIC_OPS
        ):
            continue
        target_node = atomic_targets.get(effect, cast("Node", effect.args[0]))
        if effect in atomic_targets and target_node.target is creation_ops.full:
            # The complete-fragment ownership proof certifies these fresh
            # CTA-local shared allocations and their identity-preserving carries.
            continue
        if target_node.target is not _tracing_ops._host_tensor:
            return False
        target = target_node.meta.get("val")
        if not isinstance(target, torch.Tensor):
            return False
        target_storage = target.untyped_storage()
        if source_storage == target_storage:
            return False
        # Fake storage inequality alone does not prove that two runtime
        # arguments are disjoint (including distinct DLPack wrappers). Fresh
        # compiler allocations or the existing dispatch-keyed alias matrix do.
        if (
            source_storage not in env._symbolically_exact_layout_storages
            and target_storage not in env._symbolically_exact_layout_storages
            and not runtime_tensors_are_proven_disjoint(
                env, source, target, allow_unbound=allow_unbound
            )
        ):
            return False
    return True
