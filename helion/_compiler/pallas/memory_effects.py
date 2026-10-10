"""Memory effects of one scheduled device loop, including its child graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language import distributed_ops
from ...language import memory_ops
from ...language.atomic_ops import ATOMIC_OPS
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


MEMORY_REGION_META = "pallas_memory_region"


def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    compiler = HostFunction.current().compiler_state
    return bool(compiler.ssa_storages(first) & compiler.ssa_storages(second))


def _may_alias(first: torch.Tensor, second: torch.Tensor) -> bool:
    if _shares_storage(first, second):
        return True
    # Input views may start disjoint and alias on a later cached call. Only a
    # fresh compiler allocation proves disjointness without a runtime guard.
    env = CompileEnvironment.current()
    compiler = HostFunction.current().compiler_state
    first_source, second_source = (
        compiler.ssa_source(first),
        compiler.ssa_source(second),
    )
    return not (
        first_source is not None
        and env.tensor_storage_is_compiler_allocated(first_source)
        or second_source is not None
        and env.tensor_storage_is_compiler_allocated(second_source)
    )


def control_flow_child_graph_ids(node: torch.fx.Node) -> tuple[object, ...]:
    """Return the bodies reached by a loop, branch or while-loop call."""
    if _tracing_ops.is_for_loop_target(node.target):
        return node.args[:1]
    if node.target is _tracing_ops._if:
        return node.args[1:3]
    if node.target is _tracing_ops._while_loop:
        return node.args[:2]
    return ()


def _operand(node: torch.fx.Node, position: int) -> torch.Tensor | None:
    argument = node.args[position]
    value = argument.meta.get("val") if isinstance(argument, torch.fx.Node) else None
    return value if isinstance(value, torch.Tensor) else None


@dataclass(frozen=True)
class MemoryRegion:
    reads: tuple[tuple[torch.fx.Node, torch.Tensor], ...]
    writes: tuple[tuple[torch.fx.Node, torch.Tensor], ...]
    outer_accesses: tuple[torch.Tensor, ...]

    def is_written(self, tensor: torch.Tensor) -> bool:
        return any(_may_alias(tensor, other) for _, other in self.writes)

    def is_read(self, tensor: torch.Tensor) -> bool:
        return any(_may_alias(tensor, other) for _, other in self.reads)

    def write_aliases(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(other for _, other in self.writes if _may_alias(tensor, other))

    def load_sites(self, tensor: torch.Tensor) -> tuple[torch.fx.Node, ...]:
        compiler = HostFunction.current().compiler_state
        source = compiler.ssa_source(tensor)
        return tuple(
            node
            for node, candidate in self.reads
            if node.target is memory_ops.load
            and (
                candidate is tensor
                or source is not None
                and compiler.ssa_source(candidate) is source
            )
        )

    def has_outer_access(self, tensor: torch.Tensor) -> bool:
        # This is a placement check, not an input-disjointness proof: a distinct
        # read-only input outside the loop can retain its own BlockSpec.
        return any(_shares_storage(tensor, other) for other in self.outer_accesses)

    def has_other_write(self, node: torch.fx.Node, tensor: torch.Tensor) -> bool:
        return self.has_outer_access(tensor) or any(
            other is not node and _may_alias(tensor, destination)
            for other, destination in self.writes
        )


def plan_memory_regions(
    graphs: Sequence[GraphInfo], parents: Mapping[int, int]
) -> None:
    """Attach the nearest loop's effects to each node, without sibling effects."""
    from ..device_ir import ForLoopGraphInfo

    graph_by_id = {info.graph_id: info for info in graphs}
    descriptors = {
        node.meta[distributed_ops._REMOTE_COPY_DESCRIPTOR_ID_META]: node
        for info in graphs
        for node in info.graph.nodes
        if node.target is distributed_ops.make_async_remote_copy
    }

    def effects(
        nodes: Sequence[torch.fx.Node],
    ) -> tuple[
        list[tuple[torch.fx.Node, torch.Tensor]],
        list[tuple[torch.fx.Node, torch.Tensor]],
    ]:
        reads, writes = [], []
        for node in nodes:
            if (
                node.target in (memory_ops.load, memory_ops.store)
                or node.target in ATOMIC_OPS
            ):
                tensor = _operand(node, 0)
                if tensor is not None:
                    if node.target is not memory_ops.store:
                        reads.append((node, tensor))
                    if node.target is not memory_ops.load:
                        writes.append((node, tensor))
            elif node.target in (
                distributed_ops.make_async_remote_copy,
                distributed_ops.start_async_remote_copy_descriptor,
                distributed_ops.wait_async_remote_copy,
                distributed_ops.wait_recv_async_remote_copy,
            ):
                descriptor = (
                    node
                    if node.target is distributed_ops.make_async_remote_copy
                    else descriptors[node.args[0]]
                )
                source, destination = _operand(descriptor, 0), _operand(descriptor, 3)
                if source is not None:
                    reads.append((node, source))
                if destination is not None:
                    writes.append((node, destination))
        return reads, writes

    regions: dict[int, MemoryRegion] = {}
    for info in graphs:
        scope = info.graph_id
        while scope in parents and not isinstance(graph_by_id[scope], ForLoopGraphInfo):
            scope = parents[scope]
        if scope not in regions:
            reachable, pending = set(), [scope]
            nodes = []
            while pending:
                graph_id = pending.pop()
                if graph_id in reachable:
                    continue
                reachable.add(graph_id)
                graph = graph_by_id[graph_id].graph
                nodes.extend(graph.nodes)
                pending.extend(
                    child
                    for node in graph.nodes
                    for child in control_flow_child_graph_ids(node)
                    if isinstance(child, int)
                )
            reads, writes = effects(nodes)
            outer = []
            parent = parents.get(scope)
            while parent is not None:
                parent_reads, parent_writes = effects(
                    tuple(graph_by_id[parent].graph.nodes)
                )
                outer.extend(tensor for _, tensor in parent_reads)
                outer.extend(tensor for _, tensor in parent_writes)
                parent = parents.get(parent)
            regions[scope] = MemoryRegion(tuple(reads), tuple(writes), tuple(outer))
        for node in info.graph.nodes:
            node.meta[MEMORY_REGION_META] = regions[scope]


def memory_region(node: torch.fx.Node) -> MemoryRegion:
    region = node.meta[MEMORY_REGION_META]
    assert isinstance(region, MemoryRegion)
    return region
