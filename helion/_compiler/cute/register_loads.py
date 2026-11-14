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
    from collections.abc import Collection
    from collections.abc import Iterator
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
    source = node.args[0]
    if not isinstance(source, Node) or source.target is not _tracing_ops._host_tensor:
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
            target = user.args[0]
            if (
                not isinstance(target, Node)
                or target.target is not _tracing_ops._host_tensor
            ):
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
    from .local_atomic import _reachable_graphs

    source_node = node.args[0]
    if not isinstance(source_node, Node):
        return False
    source = source_node.meta.get("val")
    if not isinstance(source, torch.Tensor):
        return False
    # Analysis IR can retain inactive rolled alternatives. Use the same
    # root/control-flow projection as the complete-fragment ownership proof.
    graphs = _reachable_graphs(list(graphs))
    if not any(node.graph is graph.graph for graph in graphs):
        return False
    return all(
        target is not None
        and not host_storages_may_alias(
            env, source, target, allow_unbound=allow_unbound
        )
        for _effect, target in _host_memory_effects(graphs, loads=False)
    )


def _host_memory_effects(
    graphs: Sequence[GraphInfo], *, loads: bool
) -> Iterator[tuple[Node, torch.Tensor | None]]:
    """Each store and atomic of ``graphs`` (and each load, with ``loads``)
    with the host tensor it touches, or ``None`` when none can be named."""
    from ...language import creation_ops
    from .local_atomic import atomic_target_origins

    atomic_targets = atomic_target_origins(list(graphs))
    for graph in graphs:
        for effect in graph.graph.nodes:
            if not (
                effect.target is memory_ops.store
                or effect.target in atomic_ops.ATOMIC_OPS
                or (loads and effect.target is memory_ops.load)
            ):
                continue
            target_node = atomic_targets.get(effect, effect.args[0])
            if not isinstance(target_node, Node):
                # Stack tensors carry indirect pointer tuples. Their pointees
                # may alias any storage even when the pointer tensor itself is
                # disjoint.
                yield effect, None
                continue
            if effect in atomic_targets and target_node.target is creation_ops.full:
                # The complete-fragment ownership proof certifies these fresh
                # CTA-local shared allocations and their identity-preserving
                # carries.
                continue
            target = target_node.meta.get("val")
            yield (
                effect,
                target
                if target_node.target is _tracing_ops._host_tensor
                and isinstance(target, torch.Tensor)
                else None,
            )


def host_storages_may_alias(
    env: CompileEnvironment,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Whether the runtime storages of two host tensors may overlap.

    Fake storage inequality alone does not prove that two runtime arguments
    are disjoint (including distinct DLPack wrappers). Every possible origin
    is checked: synthetic join storage can alias a fresh allocation, and
    invalidated input facts cannot be reused
    (``tensors_have_proven_disjoint_storage``).
    """
    from ..runtime_tensor_guards import tensors_have_proven_disjoint_storage

    if left.untyped_storage() == right.untyped_storage():
        return True
    return not tensors_have_proven_disjoint_storage(
        env, left, right, allow_unbound=allow_unbound
    )


def host_storage_is_exclusive(
    tensor: torch.Tensor,
    exempt: Collection[Node],
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    *,
    allow_unbound: bool = False,
) -> bool:
    """No load, atomic or store of the region outside ``exempt`` may touch
    ``tensor``'s storage: the destination-side companion of
    :func:`host_load_is_readonly`, over the same reachable graphs and atomic
    targets."""
    from .local_atomic import _reachable_graphs

    return all(
        effect in exempt
        or (
            target is not None
            and not host_storages_may_alias(
                env, target, tensor, allow_unbound=allow_unbound
            )
        )
        for effect, target in _host_memory_effects(
            _reachable_graphs(list(graphs)), loads=True
        )
    )


def has_opaque_inline_asm(graphs: Sequence[GraphInfo]) -> bool:
    """Inline assembly outside the scalar arithmetic envelope could hide a
    store no read-only or exclusivity proof sees."""
    from ...language.inline_asm_ops import inline_asm_elementwise
    from .pure_lane_packets import is_pure_scalar_assembly

    return any(
        node.target is inline_asm_elementwise
        and not is_pure_scalar_assembly(node.args[0], node.args[1], node.args[4])
        for graph in graphs
        for node in graph.graph.nodes
    )
