from __future__ import annotations

import ast
import dataclasses
import enum
import itertools
import math
from typing import TYPE_CHECKING
from typing import Literal

import sympy

if TYPE_CHECKING:
    from .device_ir import DeviceIR


CROSS_LOOP_ACCESS_ID_META = "_cross_loop_access_id"
CROSS_LOOP_ACCESS_MARKER_PREFIX = "__helion_cross_loop_access_wait__:"


class TileDependencyKind(enum.Enum):
    """The memory hazard represented by a cross-loop dependency edge."""

    READ_AFTER_WRITE = "read_after_write"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


def cross_loop_access_marker(access_id: int) -> ast.stmt:
    """Return a tagged inert program point immediately before a consumer load."""
    from .ast_extension import create

    marker = create(ast.Expr, value=create(ast.Constant, value=None))
    setattr(marker, CROSS_LOOP_ACCESS_ID_META, access_id)
    return marker


def cross_loop_access_marker_id(statement: ast.AST) -> int | None:
    """Return the access attached to an explicit compiler program point."""
    access_id = getattr(statement, CROSS_LOOP_ACCESS_ID_META, None)
    if isinstance(access_id, int):
        return access_id

    # Accept the old inert string representation while cached/generated bodies
    # from the migration branch are still useful for comparison.
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
class LogicalTaskAxis:
    """One source-level axis in a root's logical task space.

    ``extent`` comes directly from the block-size registration performed while
    tracing ``hl.tile``.  It is independent of the later physical PID order or
    an L2 traversal chosen by a concrete configuration.
    """

    block_id: int
    extent: sympy.Expr | str | None
    canonical_origin: bool = True


@dataclasses.dataclass(frozen=True)
class TaskFamily:
    """One opaque top-level loop and its authoritative logical task domain."""

    root: int
    graph_id: int | None
    axes: tuple[LogicalTaskAxis, ...]
    access_ids: tuple[int, ...] = ()

    @property
    def logical_axis_order(self) -> tuple[int, ...]:
        return tuple(axis.block_id for axis in self.axes)

    @property
    def has_canonical_origin(self) -> bool:
        return all(axis.canonical_origin for axis in self.axes)

    def axis(self, block_id: int) -> LogicalTaskAxis | None:
        return next((axis for axis in self.axes if axis.block_id == block_id), None)


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
    subscript_is_full_slice: tuple[bool, ...] = ()
    is_atomic: bool = False


@dataclasses.dataclass(frozen=True)
class AllocationRegion:
    """A conservative region in allocation-address coordinates.

    ``address_interval`` is always a may-access hull.  When
    ``is_exact_contiguous`` is true, it is also the exact set of addresses.
    ``coordinate_bounds`` retain an exact rectangular view when one is known;
    they let equal-layout views prove disjointness or coverage without turning
    the dependency pass into a general symbolic set solver.
    """

    address_interval: tuple[int, int] | None
    is_exact_contiguous: bool
    layout: tuple[tuple[int, ...], tuple[int, ...], int] | None = None
    coordinate_bounds: tuple[tuple[int, int], ...] = ()
    coordinates_are_exact: bool = False


@dataclasses.dataclass(frozen=True)
class AccessDependency:
    """One source-ordered memory hazard over an allocation region."""

    kind: TileDependencyKind
    producer_access_id: int
    consumer_access_id: int
    region: AllocationRegion


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
    general predecessor relation: relations that fan out or need more than one
    varying producer axis remain exact task events. A disjoint physical prefix
    may participate while an unrelated producer suffix executes normally.
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
    producer_task_segments: tuple[UniformTaskPartitionSegment, ...]
    producer_key_by_task: tuple[int | None, ...]

    @property
    def participating_producer_tasks(self) -> int:
        return self.consumer_tasks * self.fanin

    @property
    def covers_producer_domain(self) -> bool:
        return self.participating_producer_tasks == self.producer_tasks


@dataclasses.dataclass(frozen=True)
class ReadinessRequirement:
    """Readiness required by one consumer access on an allocation edge."""

    kind: TileDependencyKind
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
    access_dependencies: tuple[AccessDependency, ...]
    readiness: tuple[ReadinessRequirement, ...]

    @property
    def is_raw_only(self) -> bool:
        return self.kinds == frozenset((TileDependencyKind.READ_AFTER_WRITE,))

    @property
    def has_complete_readiness(self) -> bool:
        return {requirement.kind for requirement in self.readiness} == set(self.kinds)

    @property
    def is_task_ready(self) -> bool:
        return self.has_complete_readiness and all(
            requirement.granularity == "task" for requirement in self.readiness
        )


@dataclasses.dataclass(frozen=True)
class CrossLoopDependencyPlan:
    """Allocation-derived dependencies and their strongest proven readiness."""

    task_families: tuple[TaskFamily, ...]
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

    if fanin is None or fanin <= 0:
        return None

    participating_producer_tasks = consumer_tasks * fanin
    participating_ids = [
        producer_task
        for producer_task, owner in enumerate(owner_by_producer)
        if owner is not None
    ]
    if len(participating_ids) != participating_producer_tasks:
        return None
    producer_task_segments: list[UniformTaskPartitionSegment] = []
    segment_begin = participating_ids[0]
    previous_task = segment_begin
    for producer_task in participating_ids[1:]:
        if producer_task == previous_task + 1:
            previous_task = producer_task
            continue
        producer_task_segments.append(
            UniformTaskPartitionSegment(
                begin=segment_begin,
                length=previous_task - segment_begin + 1,
            )
        )
        segment_begin = producer_task
        previous_task = producer_task
    producer_task_segments.append(
        UniformTaskPartitionSegment(
            begin=segment_begin,
            length=previous_task - segment_begin + 1,
        )
    )

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
    if not varying_producer_axes and fanin == 1:
        varying_producer_axes = [
            block_id
            for block_id in producer_axis_order
            if producer_axis_counts[block_id] > 1
        ][:1]
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
        producer_task_segments=tuple(producer_task_segments),
        producer_key_by_task=tuple(owner_by_producer),
    )


@dataclasses.dataclass(frozen=True)
class _ReachingAccess:
    root: int
    access: CrossLoopAccess
    region: AllocationRegion


def _access_region(
    access: CrossLoopAccess,
    task_family: TaskFamily,
) -> AllocationRegion:
    """Conservatively summarize one root's union of an access.

    Canonical non-scalar tile axes cover their source-level iteration extent
    independently of the configured block size. Unknown, scalar, masked, or
    indirect dimensions retain a may-access bound but are not allowed to kill
    an earlier reaching definition.
    """
    shape = access.tensor_shape
    strides = access.tensor_strides
    if len(shape) != len(strides) or any(size < 0 for size in shape):
        return AllocationRegion(None, False)

    position_by_dim: dict[int, int] = {}
    for position, tensor_dim in enumerate(access.subscript_dims):
        if tensor_dim in position_by_dim or not 0 <= tensor_dim < len(shape):
            return AllocationRegion(None, False)
        position_by_dim[tensor_dim] = position

    bounds: list[tuple[int, int]] = []
    exact_dimensions: list[bool] = []
    for tensor_dim, size in enumerate(shape):
        position = position_by_dim.get(tensor_dim)
        if position is None:
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        if position >= len(access.subscript_is_full_slice):
            return AllocationRegion(None, False)
        if access.subscript_is_full_slice[position]:
            bounds.append((0, size))
            exact_dimensions.append(not access.has_explicit_mask)
            continue
        if (
            position >= len(access.subscript_affine_block_ids)
            or position >= len(access.subscript_index_scales)
            or position >= len(access.subscript_offsets)
            or position >= len(access.subscript_is_scalar)
        ):
            return AllocationRegion(None, False)
        block_id = access.subscript_affine_block_ids[position]
        offset = access.subscript_offsets[position]
        axis = task_family.axis(block_id) if block_id is not None else None
        symbolic_extent = axis.extent if axis is not None else None
        if (
            axis is None
            or not axis.canonical_origin
            or not isinstance(symbolic_extent, int | sympy.Integer)
            or symbolic_extent < 0
            or access.subscript_index_scales[position] != 1
            or offset is None
            or access.subscript_is_scalar[position]
        ):
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        extent = int(symbolic_extent)
        begin = offset
        end = offset + extent
        if begin < 0 or end > size:
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        bounds.append((begin, end))
        exact_dimensions.append(not access.has_explicit_mask)

    return _allocation_region_from_bounds(
        access,
        tuple(bounds),
        tuple(exact_dimensions),
    )


def access_task_region(
    access: CrossLoopAccess,
    *,
    task_coordinates: dict[int, int],
    block_sizes: dict[int, int],
) -> AllocationRegion:
    """Return one configured logical task's conservative access footprint."""
    shape = access.tensor_shape
    if len(shape) != len(access.tensor_strides) or any(size < 0 for size in shape):
        return AllocationRegion(None, False)
    position_by_dim = {
        tensor_dim: position
        for position, tensor_dim in enumerate(access.subscript_dims)
        if 0 <= tensor_dim < len(shape)
    }
    if len(position_by_dim) != len(access.subscript_dims):
        return AllocationRegion(None, False)

    bounds: list[tuple[int, int]] = []
    exact_dimensions: list[bool] = []
    for tensor_dim, size in enumerate(shape):
        position = position_by_dim.get(tensor_dim)
        if position is None or position >= len(access.subscript_is_full_slice):
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        if access.subscript_is_full_slice[position]:
            bounds.append((0, size))
            exact_dimensions.append(not access.has_explicit_mask)
            continue
        if (
            position >= len(access.subscript_affine_block_ids)
            or position >= len(access.subscript_index_scales)
            or position >= len(access.subscript_offsets)
            or position >= len(access.subscript_is_scalar)
        ):
            return AllocationRegion(None, False)
        block_id = access.subscript_affine_block_ids[position]
        coordinate = task_coordinates.get(block_id) if block_id is not None else None
        offset = access.subscript_offsets[position]
        block_size = block_sizes.get(block_id) if block_id is not None else None
        if (
            coordinate is None
            or offset is None
            or block_size is None
            or block_size <= 0
            or access.subscript_index_scales[position] != 1
        ):
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        if access.subscript_is_scalar[position]:
            begin = coordinate + offset
            end = begin + 1
        else:
            begin = coordinate * block_size + offset
            end = begin + block_size
        if begin < 0 or begin >= size:
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        bounds.append((begin, min(end, size)))
        exact_dimensions.append(not access.has_explicit_mask)

    return _allocation_region_from_bounds(
        access,
        tuple(bounds),
        tuple(exact_dimensions),
    )


def _allocation_region_from_bounds(
    access: CrossLoopAccess,
    bounds: tuple[tuple[int, int], ...],
    exact_dimensions: tuple[bool, ...],
) -> AllocationRegion:
    shape = access.tensor_shape
    strides = access.tensor_strides
    if any(begin >= end for begin, end in bounds):
        return AllocationRegion(
            (access.storage_offset, access.storage_offset),
            True,
            (shape, strides, access.storage_offset),
            bounds,
            all(exact_dimensions),
        )

    address_begin = access.storage_offset
    address_end = access.storage_offset
    for (begin, end), stride in zip(bounds, strides, strict=True):
        first = begin * stride
        last = (end - 1) * stride
        address_begin += min(first, last)
        address_end += max(first, last)
    address_end += 1

    coordinates_are_exact = all(exact_dimensions)
    active_strides = sorted(
        (abs(stride), end - begin)
        for (begin, end), stride in zip(bounds, strides, strict=True)
        if end - begin > 1
    )
    expected_stride = 1
    is_contiguous = coordinates_are_exact
    for stride, length in active_strides:
        if stride != expected_stride:
            is_contiguous = False
            break
        expected_stride *= length

    return AllocationRegion(
        (address_begin, address_end),
        is_contiguous,
        (shape, strides, access.storage_offset),
        bounds,
        coordinates_are_exact,
    )


def allocation_regions_may_overlap(
    left: AllocationRegion,
    right: AllocationRegion,
) -> bool:
    left_interval = left.address_interval
    right_interval = right.address_interval
    if left_interval is not None and right_interval is not None:
        if (
            left_interval[1] <= right_interval[0]
            or right_interval[1] <= left_interval[0]
        ):
            return False
    return not (
        left.layout is not None
        and left.layout == right.layout
        and left.coordinate_bounds
        and len(left.coordinate_bounds) == len(right.coordinate_bounds)
        and any(
            left_end <= right_begin or right_end <= left_begin
            for (left_begin, left_end), (right_begin, right_end) in zip(
                left.coordinate_bounds,
                right.coordinate_bounds,
                strict=True,
            )
        )
    )


def _region_must_cover(cover: AllocationRegion, target: AllocationRegion) -> bool:
    cover_interval = cover.address_interval
    target_interval = target.address_interval
    if (
        cover.is_exact_contiguous
        and cover_interval is not None
        and target_interval is not None
        and cover_interval[0] <= target_interval[0]
        and target_interval[1] <= cover_interval[1]
    ):
        return True
    return (
        cover.coordinates_are_exact
        and cover.layout is not None
        and cover.layout == target.layout
        and len(cover.coordinate_bounds) == len(target.coordinate_bounds)
        and all(
            cover_begin <= target_begin and target_end <= cover_end
            for (cover_begin, cover_end), (target_begin, target_end) in zip(
                cover.coordinate_bounds,
                target.coordinate_bounds,
                strict=True,
            )
        )
    )


def _linear_region(begin: int, end: int) -> AllocationRegion:
    return AllocationRegion((begin, end), True)


def _intersect_regions(
    left: AllocationRegion,
    right: AllocationRegion,
) -> AllocationRegion:
    left_interval = left.address_interval
    right_interval = right.address_interval
    if left_interval is None or right_interval is None:
        return AllocationRegion(None, False)
    begin = max(left_interval[0], right_interval[0])
    end = min(left_interval[1], right_interval[1])
    if left.is_exact_contiguous and right.is_exact_contiguous:
        return _linear_region(begin, end)
    return AllocationRegion((begin, end), False)


def _subtract_regions(
    target: AllocationRegion,
    covers: tuple[AllocationRegion, ...],
) -> tuple[AllocationRegion, ...]:
    """Return the definitely-uncovered portion of ``target``.

    Exact contiguous regions can be split. Other layouts are retained unless
    one new write is proven to cover them completely. Retaining an imprecise
    region may add dependencies but can never lose a reaching definition.
    """
    pieces = (target,)
    for cover in covers:
        next_pieces: list[AllocationRegion] = []
        for piece in pieces:
            if _region_must_cover(cover, piece):
                continue
            piece_interval = piece.address_interval
            cover_interval = cover.address_interval
            if (
                piece.is_exact_contiguous
                and cover.is_exact_contiguous
                and piece_interval is not None
                and cover_interval is not None
            ):
                overlap_begin = max(piece_interval[0], cover_interval[0])
                overlap_end = min(piece_interval[1], cover_interval[1])
                if overlap_begin < overlap_end:
                    if piece_interval[0] < overlap_begin:
                        next_pieces.append(
                            _linear_region(piece_interval[0], overlap_begin)
                        )
                    if overlap_end < piece_interval[1]:
                        next_pieces.append(
                            _linear_region(overlap_end, piece_interval[1])
                        )
                    continue
            next_pieces.append(piece)
        pieces = tuple(next_pieces)
    return pieces


def _subtract_reaching_accesses(
    reaching: list[_ReachingAccess],
    writes: tuple[_ReachingAccess, ...],
) -> list[_ReachingAccess]:
    cover_regions = tuple(write.region for write in writes)
    return [
        _ReachingAccess(entry.root, entry.access, residual)
        for entry in reaching
        for residual in _subtract_regions(entry.region, cover_regions)
    ]


def build_cross_loop_dependency_plan(
    accesses: tuple[CrossLoopAccess, ...],
    grid_block_ids: list[list[int]] | None = None,
    *,
    task_families: tuple[TaskFamily, ...] | None = None,
    root_phases: tuple[int, ...] | None = None,
    noncanonical_task_origin_block_ids: frozenset[int] = frozenset(),
) -> CrossLoopDependencyPlan:
    """Build the minimal source-ordered allocation hazard graph.

    This pass is deliberately independent of code generation. It identifies the
    most recent writer and intervening readers of every allocation, then proves
    task readiness for the strict affine subset. Anything else remains a
    root-completion dependency.
    """
    if task_families is None:
        if grid_block_ids is None:
            raise TypeError("grid_block_ids or task_families must be provided")
        task_families = tuple(
            TaskFamily(
                root=root,
                graph_id=None,
                axes=tuple(
                    LogicalTaskAxis(
                        block_id=block_id,
                        extent=None,
                        canonical_origin=(
                            block_id not in noncanonical_task_origin_block_ids
                        ),
                    )
                    for block_id in block_ids
                ),
            )
            for root, block_ids in enumerate(grid_block_ids)
        )
    elif grid_block_ids is not None and tuple(
        tuple(block_ids) for block_ids in grid_block_ids
    ) != tuple(family.logical_axis_order for family in task_families):
        raise ValueError("grid_block_ids disagree with task_families")

    root_count = len(task_families)
    if root_phases is None:
        root_phases = (0,) * root_count
    elif len(root_phases) != root_count:
        raise ValueError("root_phases must have one entry per task family")
    grid_block_ids = [list(family.logical_axis_order) for family in task_families]
    accesses_by_root: list[list[CrossLoopAccess]] = [[] for _ in range(root_count)]
    for access in accesses:
        if 0 <= access.root < root_count and access.allocation_id >= 0:
            accesses_by_root[access.root].append(access)

    # Views can carry different source names at different roots while still
    # naming the same storage.  Keep one diagnostic alias set per allocation so
    # diagnostics can describe the DeviceIR edge without manufacturing one
    # duplicate edge per source spelling.
    tensor_names_by_allocation: dict[int, set[str]] = {}
    for access in accesses:
        if access.allocation_id >= 0 and access.tensor_name is not None:
            tensor_names_by_allocation.setdefault(access.allocation_id, set()).add(
                access.tensor_name
            )

    reads_by_root = [
        _accesses_by_allocation(root_accesses, "load")
        for root_accesses in accesses_by_root
    ]
    writes_by_root = [
        _accesses_by_allocation(root_accesses, "store")
        for root_accesses in accesses_by_root
    ]

    access_by_id = {access.access_id: access for access in accesses}
    region_by_access_id = {
        access.access_id: _access_region(access, task_families[access.root])
        for access in accesses
        if 0 <= access.root < root_count and access.allocation_id >= 0
    }
    dependencies_by_edge: dict[tuple[int, int, int], set[AccessDependency]] = {}
    reaching_writes: dict[int, list[_ReachingAccess]] = {}
    reaching_reads: dict[int, list[_ReachingAccess]] = {}

    def record(
        producer: _ReachingAccess,
        consumer: _ReachingAccess,
        kind: TileDependencyKind,
    ) -> None:
        dependencies_by_edge.setdefault(
            (producer.root, consumer.root, consumer.access.allocation_id), set()
        ).add(
            AccessDependency(
                kind=kind,
                producer_access_id=producer.access.access_id,
                consumer_access_id=consumer.access.access_id,
                region=_intersect_regions(producer.region, consumer.region),
            )
        )

    current_phase: int | None = None
    for consumer_root in range(root_count):
        phase = root_phases[consumer_root]
        if phase != current_phase:
            reaching_writes.clear()
            reaching_reads.clear()
            current_phase = phase
        reads = {
            allocation_id: tuple(
                _ReachingAccess(
                    consumer_root,
                    access,
                    region_by_access_id[access.access_id],
                )
                for access in allocation_accesses
            )
            for allocation_id, allocation_accesses in reads_by_root[
                consumer_root
            ].items()
        }
        writes = {
            allocation_id: tuple(
                _ReachingAccess(
                    consumer_root,
                    access,
                    region_by_access_id[access.access_id],
                )
                for access in allocation_accesses
            )
            for allocation_id, allocation_accesses in writes_by_root[
                consumer_root
            ].items()
        }

        for allocation_id, consumer_reads in reads.items():
            for consumer in consumer_reads:
                for producer in reaching_writes.get(allocation_id, ()):
                    if allocation_regions_may_overlap(producer.region, consumer.region):
                        record(
                            producer,
                            consumer,
                            TileDependencyKind.READ_AFTER_WRITE,
                        )
        for allocation_id, consumer_writes in writes.items():
            for consumer in consumer_writes:
                for producer in reaching_writes.get(allocation_id, ()):
                    if allocation_regions_may_overlap(producer.region, consumer.region):
                        record(
                            producer,
                            consumer,
                            TileDependencyKind.WRITE_AFTER_WRITE,
                        )
                for producer in reaching_reads.get(allocation_id, ()):
                    if allocation_regions_may_overlap(producer.region, consumer.region):
                        record(
                            producer,
                            consumer,
                            TileDependencyKind.WRITE_AFTER_READ,
                        )

        for allocation_id in reads.keys() | writes.keys():
            consumer_writes = writes.get(allocation_id, ())
            if consumer_writes:
                reaching_writes[allocation_id] = [
                    *_subtract_reaching_accesses(
                        reaching_writes.get(allocation_id, []), consumer_writes
                    ),
                    *consumer_writes,
                ]
                reaching_reads[allocation_id] = _subtract_reaching_accesses(
                    reaching_reads.get(allocation_id, []), consumer_writes
                )
            consumer_reads = reads.get(allocation_id, ())
            if consumer_reads:
                reaching_reads.setdefault(allocation_id, []).extend(
                    _ReachingAccess(consumer.root, consumer.access, residual)
                    for consumer in consumer_reads
                    for residual in _subtract_regions(
                        consumer.region,
                        tuple(write.region for write in consumer_writes),
                    )
                )

    edges: list[CrossLoopDependencyEdge] = []
    for (producer_root, consumer_root, allocation_id), dependency_set in sorted(
        dependencies_by_edge.items()
    ):
        access_dependencies = tuple(
            sorted(
                dependency_set,
                key=lambda dependency: (
                    dependency.kind.value,
                    dependency.producer_access_id,
                    dependency.consumer_access_id,
                    dependency.region.address_interval or (-1, -1),
                ),
            )
        )
        kinds = frozenset(dependency.kind for dependency in access_dependencies)
        producer_accesses = tuple(
            access_by_id[access_id]
            for access_id in sorted(
                {dependency.producer_access_id for dependency in access_dependencies}
            )
        )
        consumer_accesses = tuple(
            access_by_id[access_id]
            for access_id in sorted(
                {dependency.consumer_access_id for dependency in access_dependencies}
            )
        )
        readiness: tuple[ReadinessRequirement, ...] = ()
        for kind in TileDependencyKind:
            kind_dependencies = tuple(
                dependency
                for dependency in access_dependencies
                if dependency.kind == kind
            )
            for consumer_access_id in sorted(
                {dependency.consumer_access_id for dependency in kind_dependencies}
            ):
                producer_access_ids = sorted(
                    {
                        dependency.producer_access_id
                        for dependency in kind_dependencies
                        if dependency.consumer_access_id == consumer_access_id
                    }
                )
                readiness += _build_access_readiness(
                    kind=kind,
                    producer_root=producer_root,
                    producer_accesses=tuple(
                        access_by_id[access_id] for access_id in producer_access_ids
                    ),
                    consumer_accesses=(access_by_id[consumer_access_id],),
                    producer_grid_block_ids=grid_block_ids[producer_root],
                    noncanonical_task_origin_block_ids=(
                        noncanonical_task_origin_block_ids
                    ),
                )
        edges.append(
            CrossLoopDependencyEdge(
                producer_root=producer_root,
                consumer_root=consumer_root,
                allocation_id=allocation_id,
                tensor_names=frozenset(
                    tensor_names_by_allocation.get(allocation_id, ())
                ),
                kinds=kinds,
                producer_accesses=producer_accesses,
                consumer_accesses=consumer_accesses,
                access_dependencies=access_dependencies,
                readiness=readiness,
            )
        )

    events, waits = _build_event_plan(tuple(edges), grid_block_ids)
    return CrossLoopDependencyPlan(
        task_families=task_families,
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
        if any(not edge.has_complete_readiness for edge in pair_edges):
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

        consumer_grid = set(grid_block_ids[consumer_root])
        for edge in pair_edges:
            for requirement in edge.readiness:
                granularity = requirement.granularity
                predecessor_map = requirement.predecessor_map
                wait = WaitSpec(
                    consumer_root=consumer_root,
                    consumer_access_id=requirement.consumer_access_id,
                    event_id=event_id(producer_root, granularity),
                    placement=(
                        "root_entry"
                        if granularity == "task"
                        and predecessor_map is not None
                        and all(
                            axis.consumer_block_id in consumer_grid
                            for axis in predecessor_map.axes
                        )
                        else "access"
                    ),
                    predecessor_map=predecessor_map,
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


def _build_access_readiness(
    *,
    kind: TileDependencyKind,
    producer_root: int,
    producer_accesses: tuple[CrossLoopAccess, ...],
    consumer_accesses: tuple[CrossLoopAccess, ...],
    producer_grid_block_ids: list[int],
    noncanonical_task_origin_block_ids: frozenset[int],
) -> tuple[ReadinessRequirement, ...]:
    requirements: list[ReadinessRequirement] = []
    producer_access_ids = tuple(access.access_id for access in producer_accesses)
    for consumer_access in consumer_accesses:
        predecessor_map = _build_affine_predecessor_map(
            producer_root=producer_root,
            producer_accesses=producer_accesses,
            consumer_access=consumer_access,
            producer_grid_block_ids=producer_grid_block_ids,
            noncanonical_task_origin_block_ids=noncanonical_task_origin_block_ids,
        )
        requirements.append(
            ReadinessRequirement(
                kind=kind,
                consumer_access_id=consumer_access.access_id,
                producer_access_ids=producer_access_ids,
                granularity="task" if predecessor_map is not None else "root",
                predecessor_map=predecessor_map,
            )
        )
    return tuple(requirements)


def _build_affine_predecessor_map(
    *,
    producer_root: int,
    producer_accesses: tuple[CrossLoopAccess, ...],
    consumer_access: CrossLoopAccess,
    producer_grid_block_ids: list[int],
    noncanonical_task_origin_block_ids: frozenset[int],
) -> AffinePredecessorMap | None:
    """Prove the strict affine subset used for task-level readiness."""
    if len(producer_accesses) != 1 or not producer_grid_block_ids:
        return None
    producer_access = producer_accesses[0]
    if producer_access.has_explicit_mask:
        return None

    # A task proof must account for every tensor dimension. Bare integers,
    # slices, gathers, and other unresolved forms use root completion.
    producer_positions = _normalized_access_positions(producer_access)
    consumer_positions = _normalized_access_positions(consumer_access)
    if producer_positions is None or consumer_positions is None:
        return None
    if any(
        access.tensor_shape[access.subscript_dims[position]] == 1
        and not access.subscript_is_full_slice[position]
        and access.subscript_offsets[position] != 0
        for access, positions in (
            (producer_access, producer_positions),
            (consumer_access, consumer_positions),
        )
        for position in positions
    ):
        return None
    preserve_paired_size_one_axes = len(producer_positions) == len(
        consumer_positions
    ) and all(
        (
            producer_access.tensor_strides[
                producer_access.subscript_dims[producer_position]
            ]
            == consumer_access.tensor_strides[
                consumer_access.subscript_dims[consumer_position]
            ]
        )
        or (
            producer_access.tensor_shape[
                producer_access.subscript_dims[producer_position]
            ]
            == 1
            and consumer_access.tensor_shape[
                consumer_access.subscript_dims[consumer_position]
            ]
            == 1
        )
        for producer_position, consumer_position in zip(
            producer_positions, consumer_positions, strict=True
        )
    )
    if not preserve_paired_size_one_axes:
        producer_positions = tuple(
            position
            for position in producer_positions
            if producer_access.tensor_shape[producer_access.subscript_dims[position]]
            != 1
        )
        consumer_positions = tuple(
            position
            for position in consumer_positions
            if consumer_access.tensor_shape[consumer_access.subscript_dims[position]]
            != 1
        )
    producer_strides = tuple(
        producer_access.tensor_strides[producer_access.subscript_dims[position]]
        for position in producer_positions
    )
    consumer_strides = tuple(
        consumer_access.tensor_strides[consumer_access.subscript_dims[position]]
        for position in consumer_positions
    )
    if len(producer_strides) != len(consumer_strides) or any(
        producer_stride != consumer_stride
        and not (
            producer_access.tensor_shape[
                producer_access.subscript_dims[producer_position]
            ]
            == 1
            and consumer_access.tensor_shape[
                consumer_access.subscript_dims[consumer_position]
            ]
            == 1
        )
        for producer_position, consumer_position, producer_stride, consumer_stride in zip(
            producer_positions,
            consumer_positions,
            producer_strides,
            consumer_strides,
            strict=True,
        )
    ):
        return None
    dimension_pairs = tuple(zip(producer_positions, consumer_positions, strict=True))
    storage_delta = consumer_access.storage_offset - producer_access.storage_offset
    offset_adjustments = [0] * len(dimension_pairs)
    if storage_delta:
        adjustment_position = next(
            (
                index
                for index in range(len(dimension_pairs) - 1, -1, -1)
                if producer_strides[index] != 0
                and storage_delta % producer_strides[index] == 0
            ),
            None,
        )
        if adjustment_position is None:
            return None
        offset_adjustments[adjustment_position] = (
            storage_delta // producer_strides[adjustment_position]
        )

    producer_block_ids = tuple(
        producer_access.subscript_affine_block_ids[position]
        for position, _ in dimension_pairs
        if producer_access.subscript_affine_block_ids[position] is not None
    )
    if set(producer_block_ids) != set(producer_grid_block_ids):
        return None
    if any(
        block_id in noncanonical_task_origin_block_ids
        for block_id in (
            *producer_grid_block_ids,
            *(
                block_id
                for block_id in consumer_access.subscript_affine_block_ids
                if block_id is not None
            ),
        )
    ):
        return None

    axes: list[AffinePredecessorAxis] = []
    used_dims: set[int] = set()
    for producer_block_id in producer_grid_block_ids:
        matching_pairs = [
            (producer_position, consumer_position, offset_adjustment)
            for (producer_position, consumer_position), offset_adjustment in zip(
                dimension_pairs, offset_adjustments, strict=True
            )
            if producer_access.subscript_affine_block_ids[producer_position]
            == producer_block_id
        ]
        if len(matching_pairs) != 1:
            return None
        position, consumer_position, offset_adjustment = matching_pairs[0]
        tensor_dim = producer_access.subscript_dims[position]
        if tensor_dim in used_dims:
            return None
        used_dims.add(tensor_dim)

        consumer_block_id = consumer_access.subscript_affine_block_ids[
            consumer_position
        ]
        if consumer_block_id is None:
            return None
        producer_offset = producer_access.subscript_offsets[position]
        consumer_offset = consumer_access.subscript_offsets[consumer_position]
        assert producer_offset is not None
        assert consumer_offset is not None
        axes.append(
            AffinePredecessorAxis(
                producer_block_id=producer_block_id,
                tensor_dim=tensor_dim,
                producer_offset=producer_offset,
                producer_is_scalar=producer_access.subscript_is_scalar[position],
                consumer_block_id=consumer_block_id,
                consumer_offset=consumer_offset + offset_adjustment,
                consumer_is_scalar=consumer_access.subscript_is_scalar[
                    consumer_position
                ],
            )
        )

    return AffinePredecessorMap(
        producer_root=producer_root,
        producer_access_id=producer_access.access_id,
        consumer_access_id=consumer_access.access_id,
        axes=tuple(axes),
    )


def _normalized_access_positions(
    access: CrossLoopAccess,
) -> tuple[int, ...] | None:
    """Return fully affine view dimensions in logical order.

    The result still names positions in the access metadata.  Callers may drop
    size-one dimensions when comparing two views: those dimensions contribute
    neither address range nor task multiplicity, so inserting or removing them
    is an exact allocation-coordinate normalization.
    """
    expected_dims = tuple(range(len(access.tensor_shape)))
    if (
        access.subscript_dims != expected_dims
        or len(access.tensor_strides) != len(expected_dims)
        or len(access.subscript_affine_block_ids) != len(expected_dims)
        or len(access.subscript_index_scales) != len(expected_dims)
        or len(access.subscript_offsets) != len(expected_dims)
        or len(access.subscript_is_scalar) != len(expected_dims)
        or len(access.subscript_is_full_slice) != len(expected_dims)
    ):
        return None
    for position in range(len(expected_dims)):
        if access.subscript_is_full_slice[position]:
            continue
        if (
            access.subscript_affine_block_ids[position] is None
            or access.subscript_index_scales[position] != 1
            or access.subscript_offsets[position] is None
        ):
            return None
    return tuple(range(len(expected_dims)))
