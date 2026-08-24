from __future__ import annotations

import dataclasses
import enum
import math
from operator import itemgetter
from typing import TYPE_CHECKING
from typing import Literal

import sympy

if TYPE_CHECKING:
    import ast

    from .device_ir import DeviceIR


TILE_ACTION_SCOPE_IDS_META = "_tile_dependency_action_scope_ids"
TILE_ACTION_SCOPE_ID_ATTR = "_tile_dependency_action_scope_id"
DependencyPoint = tuple[int, int | None]


class TileDependencyKind(enum.Enum):
    """The memory hazard represented by a cross-loop dependency edge."""

    READ_AFTER_WRITE = "read_after_write"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


def tile_action_scope_id(node: ast.AST) -> int | None:
    """Return the stable DeviceIR execution scope attached to a lowered loop."""
    scope_id = getattr(node, TILE_ACTION_SCOPE_ID_ATTR, None)
    return scope_id if isinstance(scope_id, int) else None


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
class InstantiatedTaskFamily:
    """A logical task family instantiated for one kernel configuration."""

    logical_axis_order: tuple[int, ...]
    physical_axis_order: tuple[int, ...]
    axis_counts_items: tuple[tuple[int, int], ...]
    block_sizes_items: tuple[tuple[int, int], ...]
    logical_task_by_physical_task: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        task_order = self.logical_task_by_physical_task
        if task_order is not None and (
            len(task_order) != self.task_count
            or set(task_order) != set(range(self.task_count))
        ):
            raise ValueError("physical traversal must permute the logical task domain")

    @property
    def axis_counts(self) -> dict[int, int]:
        return dict(self.axis_counts_items)

    @property
    def block_sizes(self) -> dict[int, int]:
        return dict(self.block_sizes_items)

    @property
    def task_count(self) -> int:
        return math.prod(count for _, count in self.axis_counts_items)

    @property
    def physical_traversal(self) -> tuple[int, ...]:
        task_order = self.logical_task_by_physical_task
        return task_order if task_order is not None else tuple(range(self.task_count))

    def task_coordinates(self, task: int) -> dict[int, int]:
        """Decode one logical task ID using the authoritative axis order."""
        coordinates: dict[int, int] = {}
        remainder = task
        for block_id in self.logical_axis_order:
            count = self.axis_counts[block_id]
            coordinates[block_id] = remainder % count
            remainder //= count
        if remainder:
            raise AssertionError("task exceeds its logical coordinate domain")
        return coordinates


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


@dataclasses.dataclass(frozen=True)
class InstantiatedActionDomain:
    """One configured ordered-action domain inside an outer task strand."""

    scope_id: int
    root: int
    strand_axis_order: tuple[int, ...]
    logical_axis_order: tuple[int, ...]
    axis_counts_items: tuple[tuple[int, int], ...]
    block_sizes_items: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.logical_axis_order[: len(self.strand_axis_order)] != (
            self.strand_axis_order
        ):
            raise ValueError("action axes must begin with their task-strand axes")
        if len(set(self.logical_axis_order)) != len(self.logical_axis_order):
            raise ValueError("action axes must be unique")
        if tuple(axis for axis, _ in self.axis_counts_items) != (
            self.logical_axis_order
        ) or tuple(axis for axis, _ in self.block_sizes_items) != (
            self.logical_axis_order
        ):
            raise ValueError("action geometry must follow logical axis order")
        if any(count <= 0 for _, count in self.axis_counts_items):
            raise ValueError("action axis counts must be positive")
        if any(size <= 0 for _, size in self.block_sizes_items):
            raise ValueError("action block sizes must be positive")

    @property
    def axis_counts(self) -> dict[int, int]:
        return dict(self.axis_counts_items)

    @property
    def block_sizes(self) -> dict[int, int]:
        return dict(self.block_sizes_items)

    @property
    def nested_axis_order(self) -> tuple[int, ...]:
        return self.logical_axis_order[len(self.strand_axis_order) :]

    @property
    def strand_count(self) -> int:
        counts = self.axis_counts
        return math.prod(counts[axis] for axis in self.strand_axis_order)

    @property
    def actions_per_strand(self) -> int:
        counts = self.axis_counts
        return math.prod(counts[axis] for axis in self.nested_axis_order)

    @property
    def action_count(self) -> int:
        return self.strand_count * self.actions_per_strand

    def action_coordinates(self, action: int) -> dict[int, int]:
        """Decode a strand-major action ID into complete logical coordinates."""
        if not 0 <= action < self.action_count:
            raise IndexError(action)
        coordinates: dict[int, int] = {}
        counts = self.axis_counts
        strand_task, local_action = divmod(action, self.actions_per_strand)
        for axis in self.strand_axis_order:
            count = counts[axis]
            coordinates[axis] = strand_task % count
            strand_task //= count
        for axis in self.nested_axis_order:
            count = counts[axis]
            coordinates[axis] = local_action % count
            local_action //= count
        if strand_task or local_action:
            raise AssertionError("action exceeds its logical coordinate domain")
        return coordinates

    def strand_task(self, action: int) -> int:
        if not 0 <= action < self.action_count:
            raise IndexError(action)
        return action // self.actions_per_strand

    def action_from_coordinates(self, coordinates: dict[int, int]) -> int:
        """Encode complete logical coordinates as a strand-major action ID."""
        counts = self.axis_counts
        strand_task = 0
        multiplier = 1
        for axis in self.strand_axis_order:
            coordinate = coordinates[axis]
            count = counts[axis]
            if not 0 <= coordinate < count:
                raise IndexError(coordinate)
            strand_task += coordinate * multiplier
            multiplier *= count

        local_action = 0
        multiplier = 1
        for axis in self.nested_axis_order:
            coordinate = coordinates[axis]
            count = counts[axis]
            if not 0 <= coordinate < count:
                raise IndexError(coordinate)
            local_action += coordinate * multiplier
            multiplier *= count
        return strand_task * self.actions_per_strand + local_action


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
    graph_node_index: int = -1


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
    dependency_id: int = -1


@dataclasses.dataclass(frozen=True)
class ActionDependencyRelation:
    """Exact overlap from producer actions to one consumer action domain."""

    kind: TileDependencyKind
    dependency_id: int
    producer_access_id: int
    consumer_access_id: int
    consumer_scope_id: int
    predecessors_by_consumer_action: tuple[frozenset[tuple[int, int | None, int]], ...]


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

    @property
    def is_raw_only(self) -> bool:
        return self.kinds == frozenset((TileDependencyKind.READ_AFTER_WRITE,))


@dataclasses.dataclass(frozen=True)
class TileDependencyGraph:
    """Allocation-derived dependencies and DeviceIR execution scopes."""

    task_families: tuple[TaskFamily, ...]
    accesses: tuple[TileAccess, ...]
    edges: tuple[TileDependency, ...]
    execution_scopes: tuple[ExecutionScope, ...] = ()
    scope_ids_by_access: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
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

    def scopes_for_access(self, access_id: int) -> tuple[ExecutionScope, ...]:
        if not 0 <= access_id < len(self.scope_ids_by_access):
            return ()
        return tuple(
            self.execution_scopes[scope_id]
            for scope_id in self.scope_ids_by_access[access_id]
        )

    def dependency_points(
        self,
        dependency: AccessDependency,
    ) -> frozenset[DependencyPoint]:
        """Return every reachable consumer callsite for one memory hazard."""
        if not 0 <= dependency.consumer_access_id < len(self.scope_ids_by_access):
            return frozenset(((dependency.dependency_id, None),))
        scope_ids = self.scope_ids_by_access[dependency.consumer_access_id]
        return (
            frozenset((dependency.dependency_id, scope_id) for scope_id in scope_ids)
            if scope_ids
            else frozenset(((dependency.dependency_id, None),))
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


_ConfiguredDomain = InstantiatedTaskFamily | InstantiatedActionDomain


def _domain_task_count(domain: _ConfiguredDomain) -> int:
    return (
        domain.task_count
        if isinstance(domain, InstantiatedTaskFamily)
        else domain.action_count
    )


def _domain_coordinates(
    domain: _ConfiguredDomain,
    task: int,
) -> dict[int, int]:
    return (
        domain.task_coordinates(task)
        if isinstance(domain, InstantiatedTaskFamily)
        else domain.action_coordinates(task)
    )


def _access_predecessor_sets(
    *,
    producer_access: TileAccess,
    producer_domain: _ConfiguredDomain,
    consumer_access: TileAccess,
    consumer_domain: _ConfiguredDomain,
    dependency_region: AllocationRegion,
) -> tuple[frozenset[int], ...] | None:
    """Evaluate exact overlap for one configured access pair."""

    def indexed_regions(
        access: TileAccess,
        domain: _ConfiguredDomain,
    ) -> list[tuple[int, int, int, AllocationRegion]] | None:
        result: list[tuple[int, int, int, AllocationRegion]] = []
        for task in range(_domain_task_count(domain)):
            region = access_task_region(
                access,
                task_coordinates=_domain_coordinates(domain, task),
                block_sizes=domain.block_sizes,
            )
            interval = region.address_interval
            if interval is None:
                return None
            if interval[0] < interval[1] and allocation_regions_may_overlap(
                region, dependency_region
            ):
                result.append((interval[0], interval[1], task, region))
        result.sort(key=itemgetter(0, 1, 2))
        return result

    producer_regions = indexed_regions(producer_access, producer_domain)
    consumer_regions = indexed_regions(consumer_access, consumer_domain)
    if producer_regions is None or consumer_regions is None:
        return None

    predecessors = [set() for _ in range(_domain_task_count(consumer_domain))]
    # Regions are half-open, so ends precede starts at the same address.
    # Producer starts precede consumer starts so equal-start pairs are emitted
    # exactly once.
    producer_end = 0
    consumer_end = 1
    producer_start = 2
    consumer_start = 3
    sweep_events: list[tuple[int, int, int, AllocationRegion]] = []
    for begin, end, task, region in producer_regions:
        sweep_events.extend(
            (
                (begin, producer_start, task, region),
                (end, producer_end, task, region),
            )
        )
    for begin, end, task, region in consumer_regions:
        sweep_events.extend(
            (
                (begin, consumer_start, task, region),
                (end, consumer_end, task, region),
            )
        )
    sweep_events.sort(key=itemgetter(0, 1, 2))

    active_producers: dict[int, AllocationRegion] = {}
    active_consumers: dict[int, AllocationRegion] = {}
    for _, event_kind, task, region in sweep_events:
        if event_kind == producer_end:
            active_producers.pop(task)
        elif event_kind == consumer_end:
            active_consumers.pop(task)
        elif event_kind == producer_start:
            for consumer_task, consumer_region in active_consumers.items():
                if allocation_regions_may_overlap(region, consumer_region):
                    predecessors[consumer_task].add(task)
            active_producers[task] = region
        else:
            for producer_task, producer_region in active_producers.items():
                if allocation_regions_may_overlap(producer_region, region):
                    predecessors[task].add(producer_task)
            active_consumers[task] = region
    return tuple(frozenset(tasks) for tasks in predecessors)


def dependency_predecessor_sets(
    dependency: TileDependency,
    *,
    task_families: tuple[InstantiatedTaskFamily, ...],
    access_by_id: dict[int, TileAccess],
) -> tuple[frozenset[int], ...] | None:
    """Evaluate one configured root-task relation by allocation overlap."""
    producer = task_families[dependency.producer_root]
    consumer = task_families[dependency.consumer_root]
    predecessors = [set() for _ in range(consumer.task_count)]
    for access_dependency in dependency.access_dependencies:
        relation = _access_predecessor_sets(
            producer_access=access_by_id[access_dependency.producer_access_id],
            producer_domain=producer,
            consumer_access=access_by_id[access_dependency.consumer_access_id],
            consumer_domain=consumer,
            dependency_region=access_dependency.region,
        )
        if relation is None:
            return None
        for consumer_task, producer_tasks in enumerate(relation):
            predecessors[consumer_task].update(producer_tasks)
    return tuple(frozenset(tasks) for tasks in predecessors)


def instantiate_root_predecessor_sets(
    dependency_graph: TileDependencyGraph,
    *,
    task_families: tuple[InstantiatedTaskFamily, ...],
) -> dict[tuple[int, int], tuple[frozenset[int], ...] | None]:
    """Instantiate every root-pair relation with the canonical overlap proof.

    A pair is exact only when every allocation hazard between the two roots can
    be represented.  The returned predecessor sets already union all of those
    hazards, so downstream event construction never needs to inspect accesses
    or repeat the memory proof.
    """
    if len(task_families) != len(dependency_graph.task_families):
        raise ValueError("task family count disagrees with the dependency graph")

    access_by_id = {access.access_id: access for access in dependency_graph.accesses}
    edges_by_pair: dict[tuple[int, int], list[TileDependency]] = {}
    for dependency in dependency_graph.edges:
        edges_by_pair.setdefault(
            (dependency.producer_root, dependency.consumer_root), []
        ).append(dependency)

    result: dict[tuple[int, int], tuple[frozenset[int], ...] | None] = {}
    for pair, dependencies in edges_by_pair.items():
        if any(
            not axis.canonical_origin
            for root in pair
            for axis in dependency_graph.task_families[root].axes
        ):
            # Configured task coordinates are zero based. Until a nonzero or
            # strided source origin is first-class in the logical domain, using
            # them as allocation coordinates would prove the wrong relation.
            result[pair] = None
            continue
        consumer = task_families[pair[1]]
        predecessors = [set() for _ in range(consumer.task_count)]
        complete = True
        for dependency in dependencies:
            edge_predecessors = dependency_predecessor_sets(
                dependency,
                task_families=task_families,
                access_by_id=access_by_id,
            )
            if edge_predecessors is None:
                complete = False
                break
            for consumer_task, producer_tasks in enumerate(edge_predecessors):
                predecessors[consumer_task].update(producer_tasks)
        result[pair] = (
            tuple(frozenset(tasks) for tasks in predecessors) if complete else None
        )
    return result


def instantiate_action_domains(
    dependency_graph: TileDependencyGraph,
    *,
    task_families: tuple[InstantiatedTaskFamily, ...],
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[InstantiatedActionDomain, ...]:
    """Bind reachable DeviceIR scopes to one static logical configuration."""
    if len(task_families) != len(dependency_graph.task_families):
        raise ValueError("task family count disagrees with the dependency graph")

    result: list[InstantiatedActionDomain] = []
    for scope in dependency_graph.execution_scopes:
        family = task_families[scope.root]
        strand_axes = family.logical_axis_order
        if scope.logical_axis_order[: len(strand_axes)] != strand_axes:
            continue
        family_counts = family.axis_counts
        family_blocks = family.block_sizes
        counts: list[tuple[int, int]] = []
        blocks: list[tuple[int, int]] = []
        valid = True
        for axis in scope.logical_axis_order:
            if axis in family_counts:
                count = family_counts[axis]
                block = family_blocks[axis]
            elif (geometry := axis_geometry.get(axis)) is not None:
                count, block = geometry
            else:
                valid = False
                break
            if count <= 0 or block <= 0:
                valid = False
                break
            counts.append((axis, count))
            blocks.append((axis, block))
        if valid:
            result.append(
                InstantiatedActionDomain(
                    scope_id=scope.scope_id,
                    root=scope.root,
                    strand_axis_order=strand_axes,
                    logical_axis_order=scope.logical_axis_order,
                    axis_counts_items=tuple(counts),
                    block_sizes_items=tuple(blocks),
                )
            )
    return tuple(result)


def preceding_actions_for_access(
    dependency_graph: TileDependencyGraph,
    *,
    action_domains: tuple[InstantiatedActionDomain, ...],
    source_scope_id: int,
    consumer_scope_id: int,
    consumer_access_id: int,
) -> tuple[frozenset[int], ...] | None:
    """Map each consumer action to earlier source-scope actions in its strand.

    The source action's entry wait dominates an access in a descendant scope.
    For a lexically earlier sibling or descendant callsite, every source action
    under the shared enclosing action has completed before the access.  Other
    scope pairs are unordered and return ``None``.
    """
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    domain_by_scope = {domain.scope_id: domain for domain in action_domains}
    source_scope = scope_by_id[source_scope_id]
    consumer_scope = scope_by_id[consumer_scope_id]
    source_domain = domain_by_scope.get(source_scope_id)
    consumer_domain = domain_by_scope.get(consumer_scope_id)
    if (
        source_scope.root != consumer_scope.root
        or source_domain is None
        or consumer_domain is None
    ):
        return None
    try:
        consumer_access = next(
            access
            for access in dependency_graph.accesses
            if access.access_id == consumer_access_id
        )
    except StopIteration:
        return None
    if consumer_access.graph_node_index < 0:
        return None

    def lineage(scope_id: int) -> tuple[int, ...]:
        result: list[int] = []
        current: int | None = scope_id
        while current is not None:
            result.append(current)
            current = scope_by_id[current].parent_scope_id
        result.reverse()
        return tuple(result)

    source_lineage = lineage(source_scope_id)
    consumer_lineage = lineage(consumer_scope_id)
    common_length = 0
    for source_ancestor, consumer_ancestor in zip(
        source_lineage, consumer_lineage, strict=False
    ):
        if source_ancestor != consumer_ancestor:
            break
        common_length += 1
    if not common_length:
        return None

    # An action-entry wait dominates every access in that action and all of its
    # descendant callsites. Project each consumer action to that ancestor.
    if common_length == len(source_lineage):
        return tuple(
            frozenset(
                (
                    source_domain.action_from_coordinates(
                        consumer_domain.action_coordinates(consumer_action)
                    ),
                )
            )
            for consumer_action in range(consumer_domain.action_count)
        )

    common_scope_id = source_lineage[common_length - 1]
    source_child = scope_by_id[source_lineage[common_length]]
    source_node_index = source_child.callsite_path[-1][0]
    if common_length == len(consumer_lineage):
        consumer_node_index = consumer_access.graph_node_index
    else:
        consumer_child = scope_by_id[consumer_lineage[common_length]]
        consumer_node_index = consumer_child.callsite_path[-1][0]
    if source_node_index >= consumer_node_index:
        return None

    # The complete earlier subtree has run for the current common-ancestor
    # action. Group source actions by that shared coordinate prefix once, then
    # look up the group for every consumer action.
    common_domain = domain_by_scope[common_scope_id]
    source_actions_by_common_action: dict[int, set[int]] = {}
    for source_action in range(source_domain.action_count):
        common_action = common_domain.action_from_coordinates(
            source_domain.action_coordinates(source_action)
        )
        source_actions_by_common_action.setdefault(common_action, set()).add(
            source_action
        )
    return tuple(
        frozenset(
            source_actions_by_common_action.get(
                common_domain.action_from_coordinates(
                    consumer_domain.action_coordinates(consumer_action)
                ),
                (),
            )
        )
        for consumer_action in range(consumer_domain.action_count)
    )


def instantiate_action_relations(
    dependency_graph: TileDependencyGraph,
    *,
    task_families: tuple[InstantiatedTaskFamily, ...],
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[ActionDependencyRelation, ...]:
    """Prove canonical producer-action sets for every consumer access scope.

    One returned relation unions every reachable producer callsite for a
    source-level memory hazard. Missing or non-segmentable callsites make the
    relation unavailable rather than leaving the scheduler to reconstruct
    partial access semantics.
    """
    domains = instantiate_action_domains(
        dependency_graph,
        task_families=task_families,
        axis_geometry=axis_geometry,
    )
    domain_by_scope = {domain.scope_id: domain for domain in domains}
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    access_by_id = {access.access_id: access for access in dependency_graph.accesses}
    result: list[ActionDependencyRelation] = []
    for dependency in dependency_graph.edges:
        for access_dependency in dependency.access_dependencies:
            producer_access = access_by_id[access_dependency.producer_access_id]
            consumer_access = access_by_id[access_dependency.consumer_access_id]
            producer_scope_ids = dependency_graph.scope_ids_by_access[
                producer_access.access_id
            ]
            if not producer_scope_ids:
                continue
            for consumer_scope_id in dependency_graph.scope_ids_by_access[
                consumer_access.access_id
            ]:
                consumer_scope = scope_by_id[consumer_scope_id]
                consumer_domain = domain_by_scope.get(consumer_scope_id)
                if (
                    not consumer_scope.guaranteed
                    or consumer_domain is None
                    or (
                        not consumer_scope.is_root
                        and (
                            not consumer_scope.segmentable
                            or len(consumer_domain.nested_axis_order) != 1
                        )
                    )
                ):
                    continue
                result_predecessors: list[set[tuple[int, int | None, int]]] = [
                    set() for _ in range(consumer_domain.action_count)
                ]
                complete = True
                for producer_scope_id in producer_scope_ids:
                    producer_scope = scope_by_id[producer_scope_id]
                    producer_domain = domain_by_scope.get(producer_scope_id)
                    if (
                        not producer_scope.guaranteed
                        or producer_domain is None
                        or (
                            not producer_scope.is_root
                            and (
                                not producer_scope.segmentable
                                or len(producer_domain.nested_axis_order) != 1
                            )
                        )
                    ):
                        complete = False
                        break
                    access_predecessors = _access_predecessor_sets(
                        producer_access=producer_access,
                        producer_domain=producer_domain,
                        consumer_access=consumer_access,
                        consumer_domain=consumer_domain,
                        dependency_region=access_dependency.region,
                    )
                    if access_predecessors is None:
                        complete = False
                        break
                    normalized_scope_id = (
                        None if producer_scope.is_root else producer_scope_id
                    )
                    for consumer_action, producer_actions in enumerate(
                        access_predecessors
                    ):
                        result_predecessors[consumer_action].update(
                            (
                                producer_domain.root,
                                normalized_scope_id,
                                producer_action,
                            )
                            for producer_action in producer_actions
                        )
                if complete:
                    result.append(
                        ActionDependencyRelation(
                            kind=access_dependency.kind,
                            dependency_id=access_dependency.dependency_id,
                            producer_access_id=producer_access.access_id,
                            consumer_access_id=consumer_access.access_id,
                            consumer_scope_id=consumer_scope_id,
                            predecessors_by_consumer_action=tuple(
                                frozenset(actions) for actions in result_predecessors
                            ),
                        )
                    )
    return tuple(result)


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
    next_dependency_id = 0
    for (producer_root, consumer_root, allocation_id), dependency_set in sorted(
        dependencies_by_edge.items()
    ):
        ordered_dependencies = sorted(
            dependency_set,
            key=lambda dependency: (
                dependency.kind.value,
                dependency.producer_access_id,
                dependency.consumer_access_id,
                dependency.region.address_interval or (-1, -1),
            ),
        )
        access_dependencies = tuple(
            dataclasses.replace(
                dependency,
                dependency_id=next_dependency_id + index,
            )
            for index, dependency in enumerate(ordered_dependencies)
        )
        next_dependency_id += len(access_dependencies)
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
            )
        )

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
        execution_scopes=execution_scopes,
        scope_ids_by_access=tuple(scope_ids_by_access),
    )


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
