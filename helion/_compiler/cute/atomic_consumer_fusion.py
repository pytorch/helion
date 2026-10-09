"""Straight-line, same-coordinate private integer atomic consumer regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ..device_ir import RootGraphInfo
from ..inductor_lowering import PointwiseLowering
from .local_atomic import atomic_target_origins
from .local_atomic_registers import local_atomic_register_chain

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo

# Bound the region's live scalar state independently of workload geometry.
MAX_CONSUMERS = 4


@dataclass(frozen=True)
class AtomicConsumerRegion:
    nodes: tuple[Node, ...]
    atomics: tuple[Node, ...]
    targets: tuple[Node, ...]


def atomic_consumer_regions(
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
            region = AtomicConsumerRegion(interval, atomics, origins)
            regions[producer] = region
            consumed.update(interval)
    return regions
