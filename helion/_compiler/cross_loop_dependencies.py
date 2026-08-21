from __future__ import annotations

import ast
import dataclasses
import itertools
import math
from typing import TYPE_CHECKING
from typing import Literal

from .loop_dependency_checker import TileDependencyKind

if TYPE_CHECKING:
    from .device_ir import DeviceIR


CROSS_LOOP_ACCESS_ID_META = "_cross_loop_access_id"
CROSS_LOOP_ACCESS_MARKER_PREFIX = "__helion_cross_loop_access_wait__:"


def cross_loop_access_marker(access_id: int) -> str:
    """Return the inert AST marker placed immediately before one consumer load."""
    return f"{CROSS_LOOP_ACCESS_MARKER_PREFIX}{access_id}"


def cross_loop_access_marker_id(statement: ast.AST) -> int | None:
    """Decode an access marker without matching ordinary string expressions."""
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
        and statement.value.value.startswith(CROSS_LOOP_ACCESS_MARKER_PREFIX)
    ):
        return None
    suffix = statement.value.value.removeprefix(CROSS_LOOP_ACCESS_MARKER_PREFIX)
    return int(suffix) if suffix.isdigit() else None


def owner_root_by_graph_id(device_ir: DeviceIR) -> tuple[int, ...]:
    """Resolve every nested DeviceIR graph to its top-level root."""
    from .device_ir import NodeArgsGraphInfo
    from .device_ir import WhileLoopGraphInfo

    graph_id_by_graph = {
        id(graph_info.graph): graph_info.graph_id for graph_info in device_ir.graphs
    }
    owners = [-1] * len(device_ir.graphs)
    for root, graph_id in enumerate(device_ir.root_ids):
        owners[graph_id] = root

    changed = True
    while changed:
        changed = False
        for graph_info in device_ir.graphs:
            owner = owners[graph_info.graph_id]
            if owner >= 0:
                if isinstance(graph_info, WhileLoopGraphInfo):
                    cond_graph_id = graph_info.cond_graph_id
                    if owners[cond_graph_id] < 0:
                        owners[cond_graph_id] = owner
                        changed = True
                continue
            if not isinstance(graph_info, NodeArgsGraphInfo):
                continue
            parent_owners = {
                owners[parent_id]
                for node in graph_info.node_args
                if (parent_id := graph_id_by_graph.get(id(node.graph))) is not None
                and owners[parent_id] >= 0
            }
            if len(parent_owners) == 1:
                owners[graph_info.graph_id] = parent_owners.pop()
                changed = True
    return tuple(owners)


@dataclasses.dataclass(frozen=True)
class CrossLoopAccess:
    """The memory facts needed to prove a cross-root readiness relation."""

    access_id: int
    memory_op_index: int
    graph_id: int
    root: int
    allocation_id: int
    kind: Literal["load", "store"]
    tensor_name: str | None
    tensor_shape: tuple[int, ...]
    tensor_strides: tuple[int, ...]
    storage_offset: int
    subscript_dims: tuple[int, ...]
    subscript_affine_block_ids: tuple[int | None, ...]
    subscript_index_scales: tuple[int, ...]
    subscript_offsets: tuple[int | None, ...]
    subscript_is_scalar: tuple[bool, ...]
    has_explicit_mask: bool


@dataclasses.dataclass(frozen=True)
class AffinePredecessorAxis:
    """One producer task axis mapped through a tensor dimension."""

    producer_block_id: int
    tensor_dim: int
    producer_offset: int
    producer_is_scalar: bool
    consumer_block_id: int
    consumer_offset: int
    consumer_is_scalar: bool


@dataclasses.dataclass(frozen=True)
class AffinePredecessorMap:
    """A proven mapping from one consumer access to producer task coordinates."""

    producer_root: int
    producer_access_id: int
    consumer_access_id: int
    axes: tuple[AffinePredecessorAxis, ...]


@dataclasses.dataclass(frozen=True)
class UniformTaskPartitionAxis:
    """One producer coordinate derived from one consumer coordinate."""

    producer_block_id: int
    consumer_block_id: int
    scale: int
    offset: int


@dataclasses.dataclass(frozen=True)
class UniformTaskPartitionSegment:
    """One contiguous producer-coordinate interval for consumer coordinate 0."""

    begin: int
    length: int


@dataclasses.dataclass(frozen=True)
class UniformTaskPartition:
    """An exact producer partition indexed by consumer task coordinates.

    All producer axes except ``partition_producer_block_id`` are affine
    functions of one consumer axis. The partition axis is the union of the
    translated intervals in ``segments``. This is intentionally narrower than
    the general predecessor relation: relations that fan out, leave producer
    tasks unowned, or need more than one varying producer axis remain exact
    task events.
    """

    producer_axis_order: tuple[int, ...]
    consumer_axis_order: tuple[int, ...]
    producer_tasks: int
    consumer_tasks: int
    fanin: int
    outer_axes: tuple[UniformTaskPartitionAxis, ...]
    partition_producer_block_id: int
    partition_consumer_block_id: int
    partition_consumer_stride: int
    segments: tuple[UniformTaskPartitionSegment, ...]


@dataclasses.dataclass(frozen=True)
class ReadinessRequirement:
    """Readiness required by one consumer load on a RAW edge."""

    consumer_access_id: int
    producer_access_ids: tuple[int, ...]
    granularity: Literal["task", "root"]
    predecessor_map: AffinePredecessorMap | None


@dataclasses.dataclass(frozen=True)
class EventSpec:
    """Completion state published by one producer task family."""

    event_id: int
    producer_root: int
    granularity: Literal["task", "root"]


@dataclasses.dataclass(frozen=True)
class WaitSpec:
    """One consumer's wait on a producer completion event."""

    consumer_root: int
    consumer_access_id: int | None
    event_id: int
    placement: Literal["root_entry", "access"]
    predecessor_map: AffinePredecessorMap | None


@dataclasses.dataclass(frozen=True)
class CrossLoopDependencyEdge:
    """One allocation hazard between two source-ordered root families."""

    producer_root: int
    consumer_root: int
    allocation_id: int
    tensor_names: frozenset[str]
    kinds: frozenset[TileDependencyKind]
    producer_accesses: tuple[CrossLoopAccess, ...]
    consumer_accesses: tuple[CrossLoopAccess, ...]
    readiness: tuple[ReadinessRequirement, ...]

    @property
    def is_raw_only(self) -> bool:
        return self.kinds == frozenset((TileDependencyKind.READ_AFTER_WRITE,))

    @property
    def is_task_ready(self) -> bool:
        return bool(self.readiness) and all(
            requirement.granularity == "task" for requirement in self.readiness
        )


@dataclasses.dataclass(frozen=True)
class CrossLoopDependencyPlan:
    """Allocation-derived dependencies and their strongest proven readiness."""

    accesses: tuple[CrossLoopAccess, ...]
    edges: tuple[CrossLoopDependencyEdge, ...]
    events: tuple[EventSpec, ...]
    waits: tuple[WaitSpec, ...]

    def edges_between(
        self,
        producer_root: int,
        consumer_root: int,
    ) -> tuple[CrossLoopDependencyEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.producer_root == producer_root
            and edge.consumer_root == consumer_root
        )

    def event(self, event_id: int) -> EventSpec:
        return self.events[event_id]

    def waits_for_root(self, root: int) -> tuple[WaitSpec, ...]:
        return tuple(wait for wait in self.waits if wait.consumer_root == root)


def predecessor_task_ids(
    predecessor_map: AffinePredecessorMap,
    *,
    consumer_coordinates: dict[int, int],
    block_sizes: dict[int, int],
    producer_axis_order: tuple[int, ...],
    producer_axis_counts: dict[int, int],
) -> frozenset[int] | None:
    """Evaluate a proven map for one consumer task.

    ``producer_axis_order`` is the configured PID decomposition order, with the
    first axis varying fastest. ``None`` means the supplied launch geometry does
    not contain all coordinates needed by the map.
    """
    predecessor_ranges: dict[int, range] = {}
    for axis in predecessor_map.axes:
        consumer_coordinate = consumer_coordinates.get(axis.consumer_block_id)
        consumer_block = (
            1 if axis.consumer_is_scalar else block_sizes.get(axis.consumer_block_id)
        )
        producer_block = (
            1 if axis.producer_is_scalar else block_sizes.get(axis.producer_block_id)
        )
        producer_count = producer_axis_counts.get(axis.producer_block_id)
        if (
            consumer_coordinate is None
            or consumer_block is None
            or producer_block is None
            or producer_count is None
            or consumer_block <= 0
            or producer_block <= 0
            or producer_count <= 0
        ):
            return None
        begin = consumer_coordinate * consumer_block + axis.consumer_offset
        end = begin + consumer_block - 1
        first = max(0, (begin - axis.producer_offset) // producer_block)
        last = min(
            producer_count - 1,
            (end - axis.producer_offset) // producer_block,
        )
        if first > last:
            return frozenset()
        predecessor_ranges[axis.producer_block_id] = range(first, last + 1)

    if set(predecessor_ranges) != set(producer_axis_order):
        return None

    result: set[int] = set()
    for coordinates in itertools.product(
        *(predecessor_ranges[block_id] for block_id in producer_axis_order)
    ):
        task_id = 0
        multiplier = 1
        for block_id, coordinate in zip(producer_axis_order, coordinates, strict=True):
            task_id += coordinate * multiplier
            multiplier *= producer_axis_counts[block_id]
        result.add(task_id)
    return frozenset(result)


def prove_uniform_task_partition(
    predecessor_maps: tuple[AffinePredecessorMap, ...],
    *,
    consumer_axis_order: tuple[int, ...],
    consumer_axis_counts: dict[int, int],
    producer_axis_order: tuple[int, ...],
    producer_axis_counts: dict[int, int],
    block_sizes: dict[int, int],
) -> UniformTaskPartition | None:
    """Prove that exact predecessor sets form one uniform task partition.

    ``predecessor_task_ids`` remains the source of truth for dependency
    membership. This function only determines whether those exact sets admit a
    compact consumer-major traversal for a last-arrival continuation.
    """
    if not predecessor_maps:
        return None
    if (
        set(consumer_axis_order) != set(consumer_axis_counts)
        or set(producer_axis_order) != set(producer_axis_counts)
        or any(count <= 0 for count in consumer_axis_counts.values())
        or any(count <= 0 for count in producer_axis_counts.values())
    ):
        return None

    consumer_axis_by_producer: dict[int, int] = {}
    for predecessor_map in predecessor_maps:
        axes_by_producer = {
            axis.producer_block_id: axis for axis in predecessor_map.axes
        }
        if len(axes_by_producer) != len(predecessor_map.axes) or set(
            axes_by_producer
        ) != set(producer_axis_order):
            return None
        for producer_block_id, axis in axes_by_producer.items():
            if axis.consumer_block_id not in consumer_axis_counts:
                return None
            previous = consumer_axis_by_producer.setdefault(
                producer_block_id, axis.consumer_block_id
            )
            if previous != axis.consumer_block_id:
                return None

    # The compact lowering maps each producer axis from a distinct consumer
    # axis. Extra singleton consumer axes are harmless, but an unrepresented
    # non-singleton axis would duplicate predecessor ownership.
    mapped_consumer_axes = tuple(consumer_axis_by_producer.values())
    if len(set(mapped_consumer_axes)) != len(mapped_consumer_axes) or any(
        count > 1 and block_id not in mapped_consumer_axes
        for block_id, count in consumer_axis_counts.items()
    ):
        return None

    def task_coordinates(
        task_id: int,
        axis_order: tuple[int, ...],
        axis_counts: dict[int, int],
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        remainder = task_id
        for block_id in axis_order:
            count = axis_counts[block_id]
            result[block_id] = remainder % count
            remainder //= count
        if remainder:
            raise AssertionError("task ID exceeds its static coordinate domain")
        return result

    producer_tasks = math.prod(producer_axis_counts.values())
    consumer_tasks = math.prod(consumer_axis_counts.values())
    predecessor_sets: list[frozenset[int]] = []
    consumer_coordinates_by_task: list[dict[int, int]] = []
    owner_by_producer: list[int | None] = [None] * producer_tasks
    fanin: int | None = None
    for consumer_task in range(consumer_tasks):
        consumer_coordinates = task_coordinates(
            consumer_task, consumer_axis_order, consumer_axis_counts
        )
        predecessor_ids: set[int] = set()
        for predecessor_map in predecessor_maps:
            mapped = predecessor_task_ids(
                predecessor_map,
                consumer_coordinates=consumer_coordinates,
                block_sizes=block_sizes,
                producer_axis_order=producer_axis_order,
                producer_axis_counts=producer_axis_counts,
            )
            if mapped is None:
                return None
            predecessor_ids.update(mapped)
        if not predecessor_ids:
            return None
        if fanin is None:
            fanin = len(predecessor_ids)
        elif len(predecessor_ids) != fanin:
            return None
        for producer_task in predecessor_ids:
            if owner_by_producer[producer_task] is not None:
                return None
            owner_by_producer[producer_task] = consumer_task
        predecessor_sets.append(frozenset(predecessor_ids))
        consumer_coordinates_by_task.append(consumer_coordinates)

    if fanin is None or fanin <= 1 or any(owner is None for owner in owner_by_producer):
        return None

    producer_coordinates_by_task = [
        task_coordinates(task, producer_axis_order, producer_axis_counts)
        for task in range(producer_tasks)
    ]
    varying_producer_axes = [
        block_id
        for block_id in producer_axis_order
        if any(
            len(
                {
                    producer_coordinates_by_task[producer_task][block_id]
                    for producer_task in predecessors
                }
            )
            > 1
            for predecessors in predecessor_sets
        )
    ]
    if len(varying_producer_axes) != 1:
        return None
    partition_producer_block_id = varying_producer_axes[0]
    partition_consumer_block_id = consumer_axis_by_producer[partition_producer_block_id]

    outer_axes: list[UniformTaskPartitionAxis] = []
    for producer_block_id in producer_axis_order:
        if producer_block_id == partition_producer_block_id:
            continue
        consumer_block_id = consumer_axis_by_producer[producer_block_id]
        producer_by_consumer: dict[int, int] = {}
        for consumer_coordinates, predecessor_set in zip(
            consumer_coordinates_by_task, predecessor_sets, strict=True
        ):
            producer_coordinates = {
                producer_coordinates_by_task[producer_task][producer_block_id]
                for producer_task in predecessor_set
            }
            if len(producer_coordinates) != 1:
                return None
            producer_coordinate = producer_coordinates.pop()
            consumer_coordinate = consumer_coordinates[consumer_block_id]
            previous = producer_by_consumer.setdefault(
                consumer_coordinate, producer_coordinate
            )
            if previous != producer_coordinate:
                return None
        if set(producer_by_consumer) != set(
            range(consumer_axis_counts[consumer_block_id])
        ):
            return None
        offset = producer_by_consumer[0]
        scale = (
            producer_by_consumer[1] - offset
            if consumer_axis_counts[consumer_block_id] > 1
            else 0
        )
        if any(
            producer_by_consumer[coordinate] != offset + coordinate * scale
            for coordinate in range(consumer_axis_counts[consumer_block_id])
        ):
            return None
        outer_axes.append(
            UniformTaskPartitionAxis(
                producer_block_id=producer_block_id,
                consumer_block_id=consumer_block_id,
                scale=scale,
                offset=offset,
            )
        )

    partition_values_by_consumer: dict[int, frozenset[int]] = {}
    for consumer_coordinates, predecessor_set in zip(
        consumer_coordinates_by_task, predecessor_sets, strict=True
    ):
        values = frozenset(
            producer_coordinates_by_task[producer_task][partition_producer_block_id]
            for producer_task in predecessor_set
        )
        if len(values) != fanin:
            return None
        consumer_coordinate = consumer_coordinates[partition_consumer_block_id]
        previous = partition_values_by_consumer.setdefault(consumer_coordinate, values)
        if previous != values:
            return None
    partition_consumer_count = consumer_axis_counts[partition_consumer_block_id]
    if set(partition_values_by_consumer) != set(range(partition_consumer_count)):
        return None
    base_values = partition_values_by_consumer[0]
    partition_consumer_stride = (
        min(partition_values_by_consumer[1]) - min(base_values)
        if partition_consumer_count > 1
        else 0
    )
    if any(
        partition_values_by_consumer[coordinate]
        != frozenset(
            value + coordinate * partition_consumer_stride for value in base_values
        )
        for coordinate in range(partition_consumer_count)
    ):
        return None

    segments: list[UniformTaskPartitionSegment] = []
    sorted_values = sorted(base_values)
    segment_begin = sorted_values[0]
    previous_value = segment_begin
    for value in sorted_values[1:]:
        if value == previous_value + 1:
            previous_value = value
            continue
        segments.append(
            UniformTaskPartitionSegment(
                begin=segment_begin,
                length=previous_value - segment_begin + 1,
            )
        )
        segment_begin = value
        previous_value = value
    segments.append(
        UniformTaskPartitionSegment(
            begin=segment_begin,
            length=previous_value - segment_begin + 1,
        )
    )
    return UniformTaskPartition(
        producer_axis_order=producer_axis_order,
        consumer_axis_order=consumer_axis_order,
        producer_tasks=producer_tasks,
        consumer_tasks=consumer_tasks,
        fanin=fanin,
        outer_axes=tuple(outer_axes),
        partition_producer_block_id=partition_producer_block_id,
        partition_consumer_block_id=partition_consumer_block_id,
        partition_consumer_stride=partition_consumer_stride,
        segments=tuple(segments),
    )


def build_cross_loop_dependency_plan(
    accesses: tuple[CrossLoopAccess, ...],
    grid_block_ids: list[list[int]],
    *,
    noncanonical_task_origin_block_ids: frozenset[int] = frozenset(),
) -> CrossLoopDependencyPlan:
    """Build the minimal source-ordered allocation hazard graph.

    This pass is deliberately independent of code generation. It identifies the
    most recent writer and intervening readers of every allocation, then proves
    task readiness for the strict affine subset. Anything else remains a
    root-completion dependency.
    """
    root_count = len(grid_block_ids)
    accesses_by_root: list[list[CrossLoopAccess]] = [[] for _ in range(root_count)]
    for access in accesses:
        if 0 <= access.root < root_count and access.allocation_id >= 0:
            accesses_by_root[access.root].append(access)

    reads_by_root = [
        _accesses_by_allocation(root_accesses, "load")
        for root_accesses in accesses_by_root
    ]
    writes_by_root = [
        _accesses_by_allocation(root_accesses, "store")
        for root_accesses in accesses_by_root
    ]

    kinds_by_edge: dict[tuple[int, int, int], set[TileDependencyKind]] = {}
    latest_writer: dict[int, int] = {}
    readers_since_write: dict[int, set[int]] = {}

    def record(
        producer_root: int,
        consumer_root: int,
        allocation_id: int,
        kind: TileDependencyKind,
    ) -> None:
        kinds_by_edge.setdefault(
            (producer_root, consumer_root, allocation_id), set()
        ).add(kind)

    for consumer_root in range(root_count):
        reads = reads_by_root[consumer_root]
        writes = writes_by_root[consumer_root]
        for allocation_id in reads:
            if (producer_root := latest_writer.get(allocation_id)) is not None:
                record(
                    producer_root,
                    consumer_root,
                    allocation_id,
                    TileDependencyKind.READ_AFTER_WRITE,
                )
        for allocation_id in writes:
            if (producer_root := latest_writer.get(allocation_id)) is not None:
                record(
                    producer_root,
                    consumer_root,
                    allocation_id,
                    TileDependencyKind.WRITE_AFTER_WRITE,
                )
            for producer_root in readers_since_write.get(allocation_id, ()):
                record(
                    producer_root,
                    consumer_root,
                    allocation_id,
                    TileDependencyKind.WRITE_AFTER_READ,
                )

        for allocation_id in writes:
            latest_writer[allocation_id] = consumer_root
            readers_since_write.pop(allocation_id, None)
        for allocation_id in reads.keys() - writes.keys():
            readers_since_write.setdefault(allocation_id, set()).add(consumer_root)

    edges: list[CrossLoopDependencyEdge] = []
    for (producer_root, consumer_root, allocation_id), mutable_kinds in sorted(
        kinds_by_edge.items()
    ):
        kinds = frozenset(mutable_kinds)
        producer_reads = reads_by_root[producer_root].get(allocation_id, ())
        producer_writes = writes_by_root[producer_root].get(allocation_id, ())
        consumer_reads = reads_by_root[consumer_root].get(allocation_id, ())
        consumer_writes = writes_by_root[consumer_root].get(allocation_id, ())
        producer_accesses = tuple((*producer_reads, *producer_writes))
        consumer_accesses = tuple((*consumer_reads, *consumer_writes))
        readiness = (
            _build_raw_readiness(
                producer_root=producer_root,
                producer_stores=producer_writes,
                consumer_loads=consumer_reads,
                producer_grid_block_ids=grid_block_ids[producer_root],
                noncanonical_task_origin_block_ids=noncanonical_task_origin_block_ids,
            )
            if TileDependencyKind.READ_AFTER_WRITE in kinds
            else ()
        )
        edges.append(
            CrossLoopDependencyEdge(
                producer_root=producer_root,
                consumer_root=consumer_root,
                allocation_id=allocation_id,
                tensor_names=frozenset(
                    access.tensor_name
                    for access in (*producer_accesses, *consumer_accesses)
                    if access.tensor_name is not None
                ),
                kinds=kinds,
                producer_accesses=producer_accesses,
                consumer_accesses=consumer_accesses,
                readiness=readiness,
            )
        )

    events, waits = _build_event_plan(tuple(edges), grid_block_ids)
    return CrossLoopDependencyPlan(
        accesses=accesses,
        edges=tuple(edges),
        events=events,
        waits=waits,
    )


def _build_event_plan(
    edges: tuple[CrossLoopDependencyEdge, ...],
    grid_block_ids: list[list[int]],
) -> tuple[tuple[EventSpec, ...], tuple[WaitSpec, ...]]:
    """Lower allocation hazards to shared producer events and consumer waits.

    A root-completion requirement subsumes task waits between the same pair of
    roots. Task events are shared by every consumer of that producer root;
    publication occurs after the producer's unchanged task body.
    """
    events: list[EventSpec] = []
    event_by_key: dict[tuple[int, Literal["task", "root"]], int] = {}
    waits: list[WaitSpec] = []

    def event_id(producer_root: int, granularity: Literal["task", "root"]) -> int:
        key = (producer_root, granularity)
        if (existing := event_by_key.get(key)) is not None:
            return existing
        result = len(events)
        event_by_key[key] = result
        events.append(
            EventSpec(
                event_id=result,
                producer_root=producer_root,
                granularity=granularity,
            )
        )
        return result

    edges_by_pair: dict[tuple[int, int], list[CrossLoopDependencyEdge]] = {}
    for edge in edges:
        edges_by_pair.setdefault((edge.producer_root, edge.consumer_root), []).append(
            edge
        )

    for (producer_root, consumer_root), pair_edges in edges_by_pair.items():
        requires_root = any(
            not edge.is_raw_only
            or not edge.readiness
            or any(requirement.granularity == "root" for requirement in edge.readiness)
            for edge in pair_edges
        )
        if requires_root:
            waits.append(
                WaitSpec(
                    consumer_root=consumer_root,
                    consumer_access_id=None,
                    event_id=event_id(producer_root, "root"),
                    placement="root_entry",
                    predecessor_map=None,
                )
            )
            continue

        task_event_id = event_id(producer_root, "task")
        consumer_grid = set(grid_block_ids[consumer_root])
        for edge in pair_edges:
            for requirement in edge.readiness:
                assert requirement.predecessor_map is not None
                wait = WaitSpec(
                    consumer_root=consumer_root,
                    consumer_access_id=requirement.consumer_access_id,
                    event_id=task_event_id,
                    placement=(
                        "root_entry"
                        if all(
                            axis.consumer_block_id in consumer_grid
                            for axis in requirement.predecessor_map.axes
                        )
                        else "access"
                    ),
                    predecessor_map=requirement.predecessor_map,
                )
                if wait not in waits:
                    waits.append(wait)

    return tuple(events), tuple(waits)


def _accesses_by_allocation(
    accesses: list[CrossLoopAccess],
    kind: Literal["load", "store"],
) -> dict[int, tuple[CrossLoopAccess, ...]]:
    result: dict[int, list[CrossLoopAccess]] = {}
    for access in accesses:
        if access.kind == kind:
            result.setdefault(access.allocation_id, []).append(access)
    return {
        allocation_id: tuple(allocation_accesses)
        for allocation_id, allocation_accesses in result.items()
    }


def _build_raw_readiness(
    *,
    producer_root: int,
    producer_stores: tuple[CrossLoopAccess, ...],
    consumer_loads: tuple[CrossLoopAccess, ...],
    producer_grid_block_ids: list[int],
    noncanonical_task_origin_block_ids: frozenset[int],
) -> tuple[ReadinessRequirement, ...]:
    requirements: list[ReadinessRequirement] = []
    producer_access_ids = tuple(access.access_id for access in producer_stores)
    for consumer_load in consumer_loads:
        predecessor_map = _build_affine_predecessor_map(
            producer_root=producer_root,
            producer_stores=producer_stores,
            consumer_load=consumer_load,
            producer_grid_block_ids=producer_grid_block_ids,
            noncanonical_task_origin_block_ids=noncanonical_task_origin_block_ids,
        )
        requirements.append(
            ReadinessRequirement(
                consumer_access_id=consumer_load.access_id,
                producer_access_ids=producer_access_ids,
                granularity="task" if predecessor_map is not None else "root",
                predecessor_map=predecessor_map,
            )
        )
    return tuple(requirements)


def _build_affine_predecessor_map(
    *,
    producer_root: int,
    producer_stores: tuple[CrossLoopAccess, ...],
    consumer_load: CrossLoopAccess,
    producer_grid_block_ids: list[int],
    noncanonical_task_origin_block_ids: frozenset[int],
) -> AffinePredecessorMap | None:
    """Prove the strict affine subset used for task-level readiness."""
    if len(producer_stores) != 1 or not producer_grid_block_ids:
        return None
    store = producer_stores[0]
    if store.has_explicit_mask:
        return None
    if (
        store.tensor_shape != consumer_load.tensor_shape
        or store.tensor_strides != consumer_load.tensor_strides
        or store.storage_offset != consumer_load.storage_offset
    ):
        return None

    # A task proof must account for every tensor dimension. Bare integers,
    # slices, gathers, and other unresolved forms use root completion.
    expected_dims = tuple(range(len(store.tensor_shape)))
    if (
        store.subscript_dims != expected_dims
        or consumer_load.subscript_dims != expected_dims
        or len(store.subscript_affine_block_ids) != len(expected_dims)
        or len(consumer_load.subscript_affine_block_ids) != len(expected_dims)
        or len(store.subscript_index_scales) != len(expected_dims)
        or len(consumer_load.subscript_index_scales) != len(expected_dims)
        or len(store.subscript_offsets) != len(expected_dims)
        or len(consumer_load.subscript_offsets) != len(expected_dims)
        or len(store.subscript_is_scalar) != len(expected_dims)
        or len(consumer_load.subscript_is_scalar) != len(expected_dims)
        or any(scale != 1 for scale in store.subscript_index_scales)
        or any(scale != 1 for scale in consumer_load.subscript_index_scales)
        or any(offset is None for offset in store.subscript_offsets)
        or any(offset is None for offset in consumer_load.subscript_offsets)
    ):
        return None

    store_block_ids = store.subscript_affine_block_ids
    if set(store_block_ids) != set(producer_grid_block_ids):
        return None
    if any(
        block_id in noncanonical_task_origin_block_ids
        for block_id in (
            *producer_grid_block_ids,
            *(
                block_id
                for block_id in consumer_load.subscript_affine_block_ids
                if block_id is not None
            ),
        )
    ):
        return None

    axes: list[AffinePredecessorAxis] = []
    used_dims: set[int] = set()
    for producer_block_id in producer_grid_block_ids:
        producer_positions = [
            position
            for position, block_id in enumerate(store_block_ids)
            if block_id == producer_block_id
        ]
        if len(producer_positions) != 1:
            return None
        position = producer_positions[0]
        tensor_dim = store.subscript_dims[position]
        if tensor_dim in used_dims:
            return None
        used_dims.add(tensor_dim)

        consumer_positions = [
            consumer_position
            for consumer_position, dim in enumerate(consumer_load.subscript_dims)
            if dim == tensor_dim
            and consumer_load.subscript_affine_block_ids[consumer_position] is not None
        ]
        if len(consumer_positions) != 1:
            return None
        consumer_position = consumer_positions[0]
        consumer_block_id = consumer_load.subscript_affine_block_ids[consumer_position]
        producer_offset = store.subscript_offsets[position]
        consumer_offset = consumer_load.subscript_offsets[consumer_position]
        assert consumer_block_id is not None
        assert producer_offset is not None
        assert consumer_offset is not None
        axes.append(
            AffinePredecessorAxis(
                producer_block_id=producer_block_id,
                tensor_dim=tensor_dim,
                producer_offset=producer_offset,
                producer_is_scalar=store.subscript_is_scalar[position],
                consumer_block_id=consumer_block_id,
                consumer_offset=consumer_offset,
                consumer_is_scalar=consumer_load.subscript_is_scalar[consumer_position],
            )
        )

    return AffinePredecessorMap(
        producer_root=producer_root,
        producer_access_id=store.access_id,
        consumer_access_id=consumer_load.access_id,
        axes=tuple(axes),
    )
