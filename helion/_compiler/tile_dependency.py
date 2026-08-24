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


TILE_ACCESS_ID_META = "_cross_loop_access_id"
TILE_ACTION_SCOPE_IDS_META = "_tile_dependency_action_scope_ids"


class TileDependencyKind(enum.Enum):
    """The memory hazard represented by a cross-loop dependency edge."""

    READ_AFTER_WRITE = "read_after_write"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


def tile_access_marker(access_id: int) -> ast.stmt:
    """Return a tagged inert program point immediately before a consumer load."""
    from .ast_extension import create

    marker = create(ast.Expr, value=create(ast.Constant, value=None))
    setattr(marker, TILE_ACCESS_ID_META, access_id)
    return marker


def tile_access_marker_id(statement: ast.AST) -> int | None:
    """Return the access attached to an explicit compiler program point."""
    access_id = getattr(statement, TILE_ACCESS_ID_META, None)
    return access_id if isinstance(access_id, int) else None


def owner_root_by_graph_id(device_ir: DeviceIR) -> tuple[int, ...]:
    """Resolve every nested DeviceIR graph to its top-level root."""
    roots_by_graph: list[set[int]] = [set() for _ in device_ir.graphs]
    for scope in build_execution_scopes(device_ir):
        roots_by_graph[scope.graph_id].add(scope.root)
    return tuple(
        next(iter(roots)) if len(roots) == 1 else -1 for roots in roots_by_graph
    )


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

    axes: tuple[LogicalTaskAxis, ...]

    @property
    def logical_axis_order(self) -> tuple[int, ...]:
        return tuple(axis.block_id for axis in self.axes)

    def axis(self, block_id: int) -> LogicalTaskAxis | None:
        return next((axis for axis in self.axes if axis.block_id == block_id), None)


@dataclasses.dataclass(frozen=True)
class ExecutionScope:
    """One reachable DeviceIR callsite in an outer task's execution strand.

    ``graph_id`` identifies the called body, while ``callsite_path`` identifies
    this particular invocation of that body.  Nested loop actions inherit the
    worker assigned to their owning root task; this record describes their
    logical coordinate domain and program-order identity, not an independently
    movable scheduling unit.
    """

    scope_id: int
    root: int
    graph_id: int
    callsite_path: tuple[tuple[int, int], ...]
    parent_scope_id: int | None
    kind: Literal["root", "loop", "branch", "while_condition", "while_body"]
    local_axis_order: tuple[int, ...]
    logical_axis_order: tuple[int, ...]
    guaranteed: bool
    segmentable: bool

    @property
    def is_root(self) -> bool:
        return self.kind == "root"


def build_execution_scopes(device_ir: DeviceIR) -> tuple[ExecutionScope, ...]:
    """Build the reachable DeviceIR callsite tree used by dependency actions.

    A DeviceIR graph body is not itself a unique execution point: one body may
    be referenced by several callsites, and control-flow graphs have different
    execution guarantees from ordinary device loops.  Paths therefore use the
    lexical call node and child argument slot within each owning root.
    """
    from ..language import _tracing_ops
    from .device_ir import ForLoopGraphInfo

    scopes: list[ExecutionScope] = []

    def add_scope(
        *,
        root: int,
        graph_id: int,
        callsite_path: tuple[tuple[int, int], ...],
        parent_scope_id: int | None,
        kind: Literal["root", "loop", "branch", "while_condition", "while_body"],
        local_axis_order: tuple[int, ...],
        logical_axis_order: tuple[int, ...],
        guaranteed: bool,
        segmentable: bool,
    ) -> int:
        scope_id = len(scopes)
        scopes.append(
            ExecutionScope(
                scope_id=scope_id,
                root=root,
                graph_id=graph_id,
                callsite_path=callsite_path,
                parent_scope_id=parent_scope_id,
                kind=kind,
                local_axis_order=local_axis_order,
                logical_axis_order=logical_axis_order,
                guaranteed=guaranteed,
                segmentable=segmentable,
            )
        )
        return scope_id

    def walk(
        *,
        root: int,
        scope_id: int,
        ancestor_graph_ids: frozenset[int],
    ) -> None:
        scope = scopes[scope_id]
        graph = device_ir.graphs[scope.graph_id].graph
        for node_index, node in enumerate(graph.nodes):
            if node.op != "call_function":
                continue

            child_specs: list[
                tuple[
                    int,
                    int,
                    Literal["loop", "branch", "while_condition", "while_body"],
                    bool,
                ]
            ] = []
            if (
                _tracing_ops.is_for_loop_target(node.target)
                and node.args
                and isinstance(node.args[0], int)
            ):
                child_specs.append((0, node.args[0], "loop", scope.guaranteed))
            elif node.target is _tracing_ops._if and len(node.args) >= 3:
                if isinstance(node.args[1], int):
                    child_specs.append((1, node.args[1], "branch", False))
                if isinstance(node.args[2], int):
                    child_specs.append((2, node.args[2], "branch", False))
            elif node.target is _tracing_ops._while_loop and len(node.args) >= 2:
                if isinstance(node.args[0], int):
                    child_specs.append((0, node.args[0], "while_condition", False))
                if isinstance(node.args[1], int):
                    child_specs.append((1, node.args[1], "while_body", False))

            callsite_scope_ids: list[tuple[int, int]] = []
            for child_slot, child_graph_id, kind, guaranteed in child_specs:
                if not 0 <= child_graph_id < len(device_ir.graphs):
                    continue
                child_info = device_ir.graphs[child_graph_id]
                local_axes = (
                    tuple(child_info.block_ids)
                    if kind == "loop" and isinstance(child_info, ForLoopGraphInfo)
                    else ()
                )
                axes_are_unique = not set(local_axes).intersection(
                    scope.logical_axis_order
                )
                child_scope_id = add_scope(
                    root=root,
                    graph_id=child_graph_id,
                    callsite_path=(*scope.callsite_path, (node_index, child_slot)),
                    parent_scope_id=scope_id,
                    kind=kind,
                    local_axis_order=local_axes,
                    logical_axis_order=(*scope.logical_axis_order, *local_axes),
                    guaranteed=guaranteed,
                    segmentable=(
                        kind == "loop"
                        and guaranteed
                        and axes_are_unique
                        and not any(
                            axis in device_ir.noncanonical_task_origin_block_ids
                            for axis in local_axes
                        )
                    ),
                )
                callsite_scope_ids.append((child_slot, child_scope_id))
                if child_graph_id not in ancestor_graph_ids:
                    walk(
                        root=root,
                        scope_id=child_scope_id,
                        ancestor_graph_ids=ancestor_graph_ids
                        | frozenset((child_graph_id,)),
                    )
            if callsite_scope_ids:
                node.meta[TILE_ACTION_SCOPE_IDS_META] = tuple(callsite_scope_ids)

    for root, graph_id in enumerate(device_ir.root_ids):
        family = device_ir.task_families[root]
        root_scope_id = add_scope(
            root=root,
            graph_id=graph_id,
            callsite_path=(),
            parent_scope_id=None,
            kind="root",
            local_axis_order=family.logical_axis_order,
            logical_axis_order=family.logical_axis_order,
            guaranteed=True,
            segmentable=False,
        )
        walk(
            root=root,
            scope_id=root_scope_id,
            ancestor_graph_ids=frozenset((graph_id,)),
        )
    return tuple(scopes)


@dataclasses.dataclass(frozen=True)
class TileAccess:
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
    layout_is_static: bool
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
    producer_offset: int
    producer_is_scalar: bool
    consumer_block_id: int
    consumer_offset: int
    consumer_is_scalar: bool


@dataclasses.dataclass(frozen=True)
class AffinePredecessorMap:
    """A proven mapping from one consumer access to producer task coordinates."""

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

    outer_axes: tuple[UniformTaskPartitionAxis, ...]
    partition_producer_block_id: int
    partition_consumer_block_id: int
    partition_consumer_stride: int
    segments: tuple[UniformTaskPartitionSegment, ...]
    producer_key_by_task: tuple[int | None, ...]

    @property
    def producer_tasks(self) -> int:
        return len(self.producer_key_by_task)

    @property
    def consumer_tasks(self) -> int:
        return (
            max(
                (key for key in self.producer_key_by_task if key is not None),
                default=-1,
            )
            + 1
        )

    @property
    def fanin(self) -> int:
        return self.participating_producer_tasks // self.consumer_tasks

    @property
    def participating_producer_tasks(self) -> int:
        return sum(key is not None for key in self.producer_key_by_task)

    @property
    def covers_producer_domain(self) -> bool:
        return self.participating_producer_tasks == self.producer_tasks


@dataclasses.dataclass(frozen=True)
class ReadinessRequirement:
    """Readiness required by one consumer access on an allocation edge."""

    kind: TileDependencyKind
    consumer_access_id: int
    granularity: Literal["task", "root"]
    predecessor_map: AffinePredecessorMap | None


@dataclasses.dataclass(frozen=True)
class EventContribution:
    """One task family's contribution to a semantic readiness event."""

    producer_root: int


@dataclasses.dataclass(frozen=True)
class KeyedEvent:
    """A logical readiness-key domain with independent contributors.

    The initial dependency builder emits singleton contributor tuples. Keeping
    the contributor relation independent of event identity lets the graph pass
    later canonicalize joins without changing the consumer-use representation.
    A root-granularity singleton is the canonical ``FamilyDone`` event.
    """

    event_id: int
    contributors: tuple[EventContribution, ...]
    granularity: Literal["task", "root"]

    @property
    def producer_root(self) -> int:
        """Return the producer of a single-contributor event."""
        if len(self.contributors) != 1:
            raise ValueError("a multi-contributor event has no unique producer root")
        return self.contributors[0].producer_root

    @property
    def is_family_done(self) -> bool:
        """Whether this is the canonical whole-family completion event."""
        return self.granularity == "root" and len(self.contributors) == 1


@dataclasses.dataclass(frozen=True)
class EventUse:
    """One consumer's wait on a producer completion event."""

    consumer_root: int
    consumer_access_id: int | None
    event_id: int
    placement: Literal["root_entry", "access"]
    predecessor_map: AffinePredecessorMap | None


@dataclasses.dataclass(frozen=True)
class TileDependency:
    """One allocation hazard between two source-ordered root families."""

    producer_root: int
    consumer_root: int
    allocation_id: int
    tensor_names: frozenset[str]
    kinds: frozenset[TileDependencyKind]
    producer_accesses: tuple[TileAccess, ...]
    consumer_accesses: tuple[TileAccess, ...]
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
class TileDependencyGraph:
    """Allocation-derived dependencies and their strongest proven readiness."""

    task_families: tuple[TaskFamily, ...]
    accesses: tuple[TileAccess, ...]
    edges: tuple[TileDependency, ...]
    events: tuple[KeyedEvent, ...]
    waits: tuple[EventUse, ...]
    execution_scopes: tuple[ExecutionScope, ...] = ()
    scope_ids_by_access: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("keyed event IDs must be contiguous and source ordered")
        if any(not 0 <= use.event_id < len(self.events) for use in self.waits):
            raise ValueError("event use references an unknown keyed event")
        if tuple(scope.scope_id for scope in self.execution_scopes) != tuple(
            range(len(self.execution_scopes))
        ):
            raise ValueError("execution scope IDs must be contiguous")
        if any(
            not 0 <= scope_id < len(self.execution_scopes)
            for scope_ids in self.scope_ids_by_access
            for scope_id in scope_ids
        ):
            raise ValueError("access references an unknown execution scope")

    def edges_between(
        self,
        producer_root: int,
        consumer_root: int,
    ) -> tuple[TileDependency, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.producer_root == producer_root
            and edge.consumer_root == consumer_root
        )

    def event(self, event_id: int) -> KeyedEvent:
        return self.events[event_id]

    def waits_for_root(self, root: int) -> tuple[EventUse, ...]:
        return tuple(wait for wait in self.waits if wait.consumer_root == root)

    def uses_for_event(self, event_id: int) -> tuple[EventUse, ...]:
        """Return every use of one event, preserving source order."""
        return tuple(wait for wait in self.waits if wait.event_id == event_id)

    def events_contributed_by(self, root: int) -> tuple[KeyedEvent, ...]:
        """Return events receiving contributions from one task family."""
        return tuple(
            event
            for event in self.events
            if any(
                contributor.producer_root == root for contributor in event.contributors
            )
        )

    def family_done(self, root: int) -> KeyedEvent | None:
        """Return the canonical whole-family event for ``root``, if required."""
        events = tuple(
            event for event in self.events_contributed_by(root) if event.is_family_done
        )
        if len(events) > 1:
            raise AssertionError(f"task family {root} has multiple FamilyDone events")
        return events[0] if events else None

    def scopes_for_access(self, access_id: int) -> tuple[ExecutionScope, ...]:
        if not 0 <= access_id < len(self.scope_ids_by_access):
            return ()
        return tuple(
            self.execution_scopes[scope_id]
            for scope_id in self.scope_ids_by_access[access_id]
        )


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
        outer_axes=tuple(outer_axes),
        partition_producer_block_id=partition_producer_block_id,
        partition_consumer_block_id=partition_consumer_block_id,
        partition_consumer_stride=partition_consumer_stride,
        segments=tuple(segments),
        producer_key_by_task=tuple(owner_by_producer),
    )


@dataclasses.dataclass(frozen=True)
class _ReachingAccess:
    root: int
    access: TileAccess
    region: AllocationRegion


def _access_region(
    access: TileAccess,
    task_family: TaskFamily,
) -> AllocationRegion:
    """Conservatively summarize one root's union of an access.

    Canonical non-scalar tile axes cover their source-level iteration extent
    independently of the configured block size. Unknown, scalar, masked, or
    indirect dimensions retain a may-access bound but are not allowed to kill
    an earlier reaching definition.
    """
    if not access.layout_is_static:
        return AllocationRegion(None, False)
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
    access: TileAccess,
    *,
    task_coordinates: dict[int, int],
    block_sizes: dict[int, int],
) -> AllocationRegion:
    """Return one configured logical task's conservative access footprint."""
    if not access.layout_is_static:
        return AllocationRegion(None, False)
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
    access: TileAccess,
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
        and _layout_is_injective(left.layout)
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


def _layout_is_injective(
    layout: tuple[tuple[int, ...], tuple[int, ...], int],
) -> bool:
    """Conservatively prove that distinct coordinates have distinct addresses."""
    shape, strides, _storage_offset = layout
    span = 1
    for stride, size in sorted(
        (abs(stride), size)
        for size, stride in zip(shape, strides, strict=True)
        if size > 1
    ):
        if stride < span:
            return False
        span += stride * (size - 1)
    return True


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


def build_tile_dependency_graph(
    accesses: tuple[TileAccess, ...],
    grid_block_ids: list[list[int]] | None = None,
    *,
    device_ir: DeviceIR | None = None,
    task_families: tuple[TaskFamily, ...] | None = None,
    root_phases: tuple[int, ...] | None = None,
    noncanonical_task_origin_block_ids: frozenset[int] = frozenset(),
) -> TileDependencyGraph:
    """Build the minimal source-ordered allocation hazard graph.

    This pass is deliberately independent of code generation. It identifies the
    most recent writer and intervening readers of every allocation, then proves
    task readiness for the strict affine subset. Anything else remains a
    root-completion dependency.
    """
    if task_families is None and device_ir is not None:
        task_families = tuple(device_ir.task_families)
    if task_families is None:
        if grid_block_ids is None:
            raise TypeError(
                "device_ir, grid_block_ids, or task_families must be provided"
            )
        task_families = tuple(
            TaskFamily(
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
            for block_ids in grid_block_ids
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
    accesses_by_root: list[list[TileAccess]] = [[] for _ in range(root_count)]
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

    edges: list[TileDependency] = []
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
            TileDependency(
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
    execution_scopes = (
        build_execution_scopes(device_ir) if device_ir is not None else ()
    )
    scope_ids_by_graph: dict[int, list[int]] = {}
    for scope in execution_scopes:
        scope_ids_by_graph.setdefault(scope.graph_id, []).append(scope.scope_id)
    scope_ids_by_access: list[tuple[int, ...]] = [
        ()
        for _ in range(max((access.access_id for access in accesses), default=-1) + 1)
    ]
    for access in accesses:
        scope_ids_by_access[access.access_id] = tuple(
            scope_id
            for scope_id in scope_ids_by_graph.get(access.graph_id, ())
            if execution_scopes[scope_id].root == access.root
        )
    return TileDependencyGraph(
        task_families=task_families,
        accesses=accesses,
        edges=tuple(edges),
        events=events,
        waits=waits,
        execution_scopes=execution_scopes,
        scope_ids_by_access=tuple(scope_ids_by_access),
    )


def _build_event_plan(
    edges: tuple[TileDependency, ...],
    grid_block_ids: list[list[int]],
) -> tuple[tuple[KeyedEvent, ...], tuple[EventUse, ...]]:
    """Lower allocation hazards to shared producer events and consumer waits.

    A root-completion requirement subsumes task waits between the same pair of
    roots. Task events are shared by every consumer of that producer root;
    publication occurs after the producer's unchanged task body.
    """
    events: list[KeyedEvent] = []
    event_by_key: dict[tuple[int, Literal["task", "root"]], int] = {}
    waits: list[EventUse] = []

    def event_id(producer_root: int, granularity: Literal["task", "root"]) -> int:
        key = (producer_root, granularity)
        if (existing := event_by_key.get(key)) is not None:
            return existing
        result = len(events)
        event_by_key[key] = result
        events.append(
            KeyedEvent(
                event_id=result,
                contributors=(EventContribution(producer_root),),
                granularity=granularity,
            )
        )
        return result

    edges_by_pair: dict[tuple[int, int], list[TileDependency]] = {}
    for edge in edges:
        edges_by_pair.setdefault((edge.producer_root, edge.consumer_root), []).append(
            edge
        )

    for (producer_root, consumer_root), pair_edges in edges_by_pair.items():
        if any(not edge.has_complete_readiness for edge in pair_edges):
            waits.append(
                EventUse(
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
                wait = EventUse(
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
    accesses: list[TileAccess],
    kind: Literal["load", "store"],
) -> dict[int, tuple[TileAccess, ...]]:
    result: dict[int, list[TileAccess]] = {}
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
    producer_accesses: tuple[TileAccess, ...],
    consumer_accesses: tuple[TileAccess, ...],
    producer_grid_block_ids: list[int],
    noncanonical_task_origin_block_ids: frozenset[int],
) -> tuple[ReadinessRequirement, ...]:
    requirements: list[ReadinessRequirement] = []
    for consumer_access in consumer_accesses:
        predecessor_map = _build_affine_predecessor_map(
            producer_accesses=producer_accesses,
            consumer_access=consumer_access,
            producer_grid_block_ids=producer_grid_block_ids,
            noncanonical_task_origin_block_ids=noncanonical_task_origin_block_ids,
        )
        requirements.append(
            ReadinessRequirement(
                kind=kind,
                consumer_access_id=consumer_access.access_id,
                granularity="task" if predecessor_map is not None else "root",
                predecessor_map=predecessor_map,
            )
        )
    return tuple(requirements)


def _build_affine_predecessor_map(
    *,
    producer_accesses: tuple[TileAccess, ...],
    consumer_access: TileAccess,
    producer_grid_block_ids: list[int],
    noncanonical_task_origin_block_ids: frozenset[int],
) -> AffinePredecessorMap | None:
    """Prove the strict affine subset used for task-level readiness."""
    if len(producer_accesses) != 1 or not producer_grid_block_ids:
        return None
    producer_access = producer_accesses[0]
    if (
        producer_access.has_explicit_mask
        or not producer_access.layout_is_static
        or not consumer_access.layout_is_static
    ):
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
        # A one-dimensional shift has a unique coordinate interpretation.
        # In multiple dimensions, divisibility by one stride is insufficient:
        # e.g. offset 5 in a contiguous (4, 4) view is (1, 1), not (0, 5).
        # Keep those cases on root completion until allocation-coordinate
        # decomposition is represented explicitly.
        if (
            len(dimension_pairs) != 1
            or producer_strides[0] == 0
            or storage_delta % producer_strides[0]
        ):
            return None
        offset_adjustments[0] = storage_delta // producer_strides[0]

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
                producer_offset=producer_offset,
                producer_is_scalar=producer_access.subscript_is_scalar[position],
                consumer_block_id=consumer_block_id,
                consumer_offset=consumer_offset + offset_adjustment,
                consumer_is_scalar=consumer_access.subscript_is_scalar[
                    consumer_position
                ],
            )
        )

    return AffinePredecessorMap(axes=tuple(axes))


def _normalized_access_positions(
    access: TileAccess,
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
