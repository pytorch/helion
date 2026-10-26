"""Straight-line, same-coordinate private integer atomic consumer regions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import memory_ops
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..inductor_lowering import PointwiseLowering
from ..variable_origin import GridOrigin
from .dead_zero_atomics import _boolean_update
from .dead_zero_atomics import _masks_imply_update
from .fragment_storage import configured_fragment_expr
from .local_atomic import atomic_target_origins
from .local_atomic_registers import local_atomic_register_chain
from .promote_output_axis import _fresh_tensors

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo

# Bound the region's live scalar state independently of workload geometry.
MAX_CONSUMERS = 4


@dataclass(frozen=True)
class AtomicConsumerRegion:
    nodes: tuple[Node, ...]
    atomics: tuple[Node, ...]
    targets: tuple[Node, ...]
    stores: tuple[Node, ...] = ()
    # Complete owner chains proved below to end inside this exact interval.
    local_returns: tuple[Node, ...] = ()


def _local_regions(
    graphs: list[GraphInfo], env: CompileEnvironment
) -> dict[Node, AtomicConsumerRegion]:
    """Plan only root-local integer updates with no intermediate observer.

    The ordinary local-atomic lifetime proof remains a prerequisite of fragment
    admission. The emitter additionally checks distinct physical storage and
    the configured owner map. This analysis does not move initialization or
    permit a read of a counter/sink through a lazy alias.
    """
    targets = atomic_target_origins(graphs)
    regions: dict[Node, AtomicConsumerRegion] = {}
    for info in graphs:
        if not isinstance(info, RootGraphInfo):
            continue
        ordered = tuple(info.graph.nodes)
        position = {node: index for index, node in enumerate(ordered)}
        consumed: set[Node] = set()
        for producer in ordered:
            fake = producer.meta.get("val")
            if (
                producer in consumed
                or producer.target is not atomic_ops.atomic_add
                or not isinstance(fake, torch.Tensor)
                or fake.ndim != 1
            ):
                continue
            chain = local_atomic_register_chain(
                producer, env, targets, dead_outputs=False
            )
            if not chain:
                continue
            atomics = tuple(
                node
                for node in ordered
                if node in chain and node.target is atomic_ops.atomic_add
            )
            if not 2 <= len(atomics) <= MAX_CONSUMERS + 1:
                continue
            if atomics[0] is not producer or any(node.users for node in atomics[1:]):
                continue
            first, last = position[producer], position[atomics[-1]]
            interval = ordered[first : last + 1]
            # A terminal host store or later/earlier user cannot be hidden by
            # selecting only the atomic subset of the existing owner chain.
            if not chain.issubset(interval):
                continue
            if any(
                node not in atomics
                and not (
                    node.target
                    in (
                        _tracing_ops._new_var,
                        torch.ops.aten.where.self,
                        torch.ops.aten.scalar_tensor.default,
                        torch.ops.aten.full.default,
                        torch.ops.prims.iota.default,
                    )
                    or (
                        isinstance(node.meta.get("lowering"), PointwiseLowering)
                        and isinstance(node.target, torch._ops.OpOverload)
                        and torch.Tag.pointwise in node.target.tags
                        and torch.Tag.nondeterministic_seeded not in node.target.tags
                        and not node.target._schema.is_mutable
                    )
                )
                for node in interval
            ):
                continue
            origins = tuple(targets[node] for node in atomics)
            if len(set(origins)) != len(origins) or any(
                origin.target is not creation_ops.full
                or origin.graph is not info.graph
                or position[origin] >= first
                or cast("torch.Tensor", origin.meta["val"]).dtype != torch.int32
                for origin in origins
            ):
                continue
            if any(
                node.target is atomic_ops.atomic_add and targets[node] in origins
                for node in ordered[:first]
            ):
                continue

            def reads_mutable(
                node: Node,
                seen: set[Node],
                producer: Node = producer,
                origins: tuple[Node, ...] = origins,
            ) -> bool:
                if node is producer or node in seen:
                    return False
                seen.add(node)
                if node in origins or node.target is atomic_ops.atomic_add:
                    return True
                return any(reads_mutable(arg, seen) for arg in node.all_input_nodes)

            # Exclude target argument itself; index/value must not observe any
            # state whose updates will now be interleaved. The producer result
            # is the one allowed immutable per-coordinate snapshot.
            inputs: list[Node] = []
            for node in atomics:
                map_arg((node.args[1:], node.kwargs), inputs.append)
            if any(reads_mutable(arg, set()) for arg in inputs):
                continue
            region = AtomicConsumerRegion(
                interval, atomics, origins, local_returns=(producer,)
            )
            regions[producer] = region
            consumed.update(interval)
    return regions


def _pure_element(node: Node) -> bool:
    if node.target in (
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        torch.ops.aten.alias.default,
        torch.ops.aten.where.self,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.full.default,
        torch.ops.prims.iota.default,
    ):
        return True
    if node.target is torch.ops.aten.view.dtype:
        source = node.args[0]
        before = source.meta.get("val") if isinstance(source, Node) else None
        after = node.meta.get("val")
        return (
            isinstance(before, torch.Tensor)
            and isinstance(after, torch.Tensor)
            and before.dtype.itemsize == after.dtype.itemsize
        )
    return (
        isinstance(node.meta.get("lowering"), PointwiseLowering)
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
        and torch.Tag.nondeterministic_seeded not in node.target.tags
        and not node.target._schema.is_mutable
    )


def _integer_constant(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, Node):
        if value.target in (_tracing_ops._new_var, torch.ops.aten.alias.default):
            return _integer_constant(value.args[0])
        if value.target is torch.ops.aten.scalar_tensor.default:
            fake = value.meta.get("val")
            if isinstance(fake, torch.Tensor) and fake.dtype in (
                torch.int32,
                torch.int64,
            ):
                return _integer_constant(value.args[0])
    return None


def _direct_ticket(value: object, producer: Node) -> bool:
    while isinstance(value, Node) and value is not producer:
        if value.target in (_tracing_ops._new_var, torch.ops.aten.alias.default) or (
            value.target is torch.ops.prims.convert_element_type.default
            and value.args[1] is torch.int64
            and isinstance(value.args[0], Node)
            and cast("torch.Tensor", value.args[0].meta["val"]).dtype is torch.int32
        ):
            value = value.args[0]
        else:
            return False
    return value is producer


def _unique_ticket_store(
    store: Node,
    producer: Node,
    origin: Node,
    info: RootGraphInfo,
    env: CompileEnvironment,
) -> bool:
    """A Boolean counter's returned positive-update tickets are unique.

    Require fresh host storage and all root grid coordinates verbatim, plus
    the direct ticket on the remaining contiguous axis. Bounds predicates may
    discard tickets but cannot duplicate them. No wrapping counter, address
    cast, remapping, input alias or independent grid dimension is admitted.
    """
    if store.kwargs or len(store.args) != 4 or store.users:
        return False
    target, indices, _value, mask = store.args
    if not (
        isinstance(target, Node)
        and target.target is _tracing_ops._host_tensor
        and isinstance(indices, (list, tuple))
        and isinstance(mask, Node)
        and _integer_constant(origin.args[1]) == 0
    ):
        return False
    tensor = target.meta.get("val")
    host = HostFunction.current()
    ir = host.device_ir
    if not isinstance(tensor, torch.Tensor) or tensor not in _fresh_tensors(
        host, integer_metadata_calls=True
    ):
        return False
    if not tensor.is_contiguous() or len(indices) != tensor.ndim:
        return False
    root_ids = ir.grid_block_ids[ir.root_ids.index(info.graph_id)]
    grid_indices = []
    ticket_axes = 0
    for index in indices:
        if _direct_ticket(index, producer):
            ticket_axes += 1
            continue
        if not isinstance(index, Node) or index.target is not _tracing_ops._get_symnode:
            return False
        scalar = index.meta.get("val")
        if not isinstance(scalar, torch.SymInt):
            return False
        symbol = host.expr_to_origin.get(scalar._sympy_())
        if symbol is None or type(symbol.origin) is not GridOrigin:
            return False
        grid_indices.append(symbol.origin.block_id)
    if ticket_axes != 1 or sorted(grid_indices) != sorted(root_ids):
        return False
    fake = cast("torch.Tensor", producer.meta["val"])
    extent = env.specialize_expr(
        cast(
            "sympy.Expr",
            configured_fragment_expr(
                env, sympy.sympify(fake.shape[0]), lambda _block: None
            ),
        )
    )
    if not isinstance(extent, sympy.Integer) or not 0 < int(extent) < 2**31:
        return False
    origin_fake = cast("torch.Tensor", origin.meta["val"])
    if tuple(origin_fake.shape) != (1,):
        return False
    update = _boolean_update(producer.args[2])
    if update is None or not _masks_imply_update(update, [mask]):
        return False
    index = producer.args[1]
    if not isinstance(index, (list, tuple)) or len(index) != 1:
        return False
    if _integer_constant(index[0]) == 0:
        return True
    selected = index[0]
    return (
        isinstance(selected, Node)
        and selected.target is torch.ops.aten.where.self
        and _integer_constant(selected.args[1]) == 0
        and _integer_constant(selected.args[2]) == 1
        and isinstance(selected.args[0], Node)
        and _masks_imply_update(selected.args[0], [mask])
    )


def atomic_consumer_regions(
    graphs: list[GraphInfo], env: CompileEnvironment
) -> dict[Node, AtomicConsumerRegion]:
    """Extend legacy private regions with adjacent independent owner chains.

    No effect, reduction, initialization or control boundary is crossed. The
    existing owner-chain proof authenticates every use of each returned ticket;
    only a unique-ticket fresh global store is added as a terminal effect.
    """
    legacy = _local_regions(graphs, env)
    targets = atomic_target_origins(graphs)
    result = dict(legacy)
    for info in graphs:
        if not isinstance(info, RootGraphInfo):
            continue
        ordered = tuple(info.graph.nodes)
        position = {node: i for i, node in enumerate(ordered)}
        candidates: list[AtomicConsumerRegion] = []
        for producer in ordered:
            fake = producer.meta.get("val")
            if (
                producer.target is not atomic_ops.atomic_add
                or not isinstance(fake, torch.Tensor)
                or fake.ndim != 1
            ):
                continue
            chain = local_atomic_register_chain(
                producer, env, targets, dead_outputs=False
            )
            if not chain:
                continue
            atomics = tuple(
                n for n in ordered if n in chain and n.target is atomic_ops.atomic_add
            )
            stores = tuple(
                n for n in ordered if n in chain and n.target is memory_ops.store
            )
            if (
                not 2 <= len(atomics) + len(stores) <= MAX_CONSUMERS + 1
                or len(stores) > 1
                or any(n.users for n in atomics[1:])
            ):
                continue
            effects = (*atomics, *stores)
            last = max(position[n] for n in effects)
            interval = ordered[position[producer] : last + 1]
            if not chain.issubset(interval) or any(
                n not in effects and not _pure_element(n) for n in interval
            ):
                continue
            origins = tuple(targets[n] for n in atomics)
            if len(set(origins)) != len(origins) or any(
                n.target is not creation_ops.full
                or n.graph is not info.graph
                or position[n] >= position[producer]
                or cast("torch.Tensor", n.meta["val"]).dtype is not torch.int32
                for n in origins
            ):
                continue
            if any(
                n.target is atomic_ops.atomic_add and targets[n] in origins
                for n in ordered[: position[producer]]
            ):
                continue
            if stores and not _unique_ticket_store(
                stores[0], producer, origins[0], info, env
            ):
                continue
            candidates.append(
                AtomicConsumerRegion(interval, atomics, origins, stores, (producer,))
            )

        def safe(region: AtomicConsumerRegion) -> bool:
            returned = {n for n in region.atomics if n.users}
            seen: set[Node] = set()

            def immutable(n: Node) -> bool:
                if n in returned or n in seen:
                    return True
                seen.add(n)
                if n in region.targets or n.target is atomic_ops.atomic_add:
                    return False
                if n.target is memory_ops.load:
                    source = n.args[0]
                    if (
                        not isinstance(source, Node)
                        or source.target is not _tracing_ops._host_tensor
                    ):
                        return False
                    storage = cast("torch.Tensor", source.meta["val"]).untyped_storage()
                    # Only the stores inside this interval can move relative
                    # to this load. Their fresh host-allocation proof survives
                    # runtime input aliasing; reject every view of that arena.
                    if any(
                        storage
                        == cast(
                            "torch.Tensor", cast("Node", store.args[0]).meta["val"]
                        ).untyped_storage()
                        for store in region.stores
                    ):
                        return False
                return all(immutable(arg) for arg in n.all_input_nodes)

            inputs: list[Node] = []
            for n in (*region.atomics, *region.stores):
                map_arg((n.args[1:], n.kwargs), inputs.append)
            return all(immutable(n) for n in inputs)

        joined: list[AtomicConsumerRegion] = []
        for candidate in candidates:
            if not safe(candidate):
                continue
            if joined:
                previous = joined[-1]
                between = ordered[
                    position[previous.nodes[-1]] + 1 : position[candidate.nodes[0]]
                ]
                first = position[previous.nodes[0]]
                combined = AtomicConsumerRegion(
                    ordered[first : position[candidate.nodes[-1]] + 1],
                    (*previous.atomics, *candidate.atomics),
                    (*previous.targets, *candidate.targets),
                    (*previous.stores, *candidate.stores),
                    (*previous.local_returns, *candidate.local_returns),
                )
                if (
                    position[previous.nodes[-1]] < position[candidate.nodes[0]]
                    and len(combined.atomics) + len(combined.stores)
                    <= MAX_CONSUMERS + 1
                    and len(combined.stores) <= 1
                    and not previous.stores
                    and all(_pure_element(n) for n in between)
                    and len(set(combined.targets)) == len(combined.targets)
                    and all(position[n] < first for n in candidate.targets)
                    and all(
                        all(
                            starmap(
                                env.known_equal,
                                zip(
                                    n.meta["val"].shape,
                                    previous.atomics[0].meta["val"].shape,
                                    strict=True,
                                ),
                            )
                        )
                        for n in candidate.atomics
                    )
                    and safe(combined)
                ):
                    joined[-1] = combined
                    continue
            joined.append(candidate)
        for region in joined:
            for node in region.nodes:
                result.pop(node, None)
            result[region.nodes[0]] = region
    return result
