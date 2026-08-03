"""SparseCore program and dependency-derived pipeline schedule."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import math
from typing import TYPE_CHECKING

from .access import AccessKind
from .sc_base import SC_LANES
from .sc_base import SC_SHARED_BYTES
from .sc_base import SC_VMEM_BYTES
from .sc_base import SC_VMEM_MARGIN
from .sc_base import _reject
from .sc_base import shared_acc_bytes
from .sparsecore_access import AtomicAddAccess
from .sparsecore_access import CachedLoadAccess
from .sparsecore_access import DirectLoadAccess
from .sparsecore_access import IndirectLoadAccess
from .sparsecore_access import IndirectStoreAccess
from .sparsecore_access import SparseCoreAccess

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch


@dataclass(frozen=True)
class SparseCoreGeometry:
    item_block_id: int
    item_count: int
    tile_size: int
    num_cores: int
    num_subcores: int

    @property
    def subcores(self) -> int:
        return self.num_cores * self.num_subcores

    @property
    def items_per_subcore(self) -> int:
        return self.tile_size // self.subcores

    @property
    def blocks(self) -> int:
        return math.ceil(self.item_count / self.tile_size)

    @property
    def padded_items(self) -> int:
        return self.blocks * self.tile_size


class TaskKind(Enum):
    INITIALIZE = "initialize"
    ONCE_TRANSFER = "once_transfer"
    SYNC_TRANSFER = "sync_transfer"
    ASYNC_START = "async_start"
    ASYNC_WAIT = "async_wait"
    COMPUTE = "compute"
    STORE = "store"
    FINALIZE = "finalize"


class SchedulePhase(Enum):
    INITIALIZE = "initialize"
    STEADY = "steady"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class Task:
    id: int
    kind: TaskKind
    node: torch.fx.Node
    dependencies: frozenset[int]


@dataclass(frozen=True)
class ScheduleStage:
    lag: int
    tasks: tuple[Task, ...]
    phase: SchedulePhase = SchedulePhase.STEADY


@dataclass(frozen=True)
class SparseCoreSchedule:
    stages: tuple[ScheduleStage, ...]
    depth: int

    @property
    def steady_stages(self) -> tuple[ScheduleStage, ...]:
        return tuple(
            stage for stage in self.stages if stage.phase is SchedulePhase.STEADY
        )


@dataclass
class SparseCoreProgram:
    graph: torch.fx.Graph
    geometry: SparseCoreGeometry
    accesses: tuple[SparseCoreAccess, ...]
    access_by_node: dict[torch.fx.Node, SparseCoreAccess] = field(init=False)
    index_nodes: frozenset[torch.fx.Node] = field(init=False)
    schedule: SparseCoreSchedule | None = None

    def __post_init__(self) -> None:
        self.access_by_node = {access.site.node: access for access in self.accesses}
        if len(self.access_by_node) != len(self.accesses):
            raise AssertionError("duplicate SparseCore access node")
        self.index_nodes = frozenset(
            access.index_node
            for access in self.accesses
            if isinstance(
                access, (IndirectLoadAccess, IndirectStoreAccess, AtomicAddAccess)
            )
        )
        offset_loads = [
            access
            for access in self.accesses
            if isinstance(access, IndirectLoadAccess) and access.index_offset
        ]
        for access in offset_loads:
            index_access = self.access_by_node.get(access.index_node)
            if isinstance(index_access, IndirectLoadAccess):
                _reject(
                    "schedule",
                    "static offsets on dependent indices are not implemented",
                    node=access.site.node,
                )
            users = [
                candidate
                for candidate in self.accesses
                if isinstance(candidate, IndirectLoadAccess)
                and candidate.index_node is access.index_node
            ]
            if len(users) > 1:
                _reject(
                    "schedule",
                    "an index adjusted by a static offset cannot be shared",
                    node=access.site.node,
                )

    @property
    def loads(self) -> tuple[SparseCoreAccess, ...]:
        return tuple(
            access for access in self.accesses if access.site.kind is AccessKind.LOAD
        )

    @property
    def stores(self) -> tuple[SparseCoreAccess, ...]:
        return tuple(
            access
            for access in self.accesses
            if access.site.kind is not AccessKind.LOAD
        )

    @property
    def rebuilt_outputs(self) -> tuple[torch.Tensor, ...]:
        """Logical outputs whose SC implementation does not preserve input data."""
        return tuple(
            access.site.tensor
            for access in self.stores
            if isinstance(access, AtomicAddAccess)
        )


def load_buffer_shape(
    program: SparseCoreProgram, access: SparseCoreAccess
) -> tuple[int, ...]:
    """Scratch shape for one load and one pipeline slot."""
    if isinstance(access, CachedLoadAccess):
        return access.layout.storage_shape
    if isinstance(access, IndirectLoadAccess):
        assert access.stream is not None
        entries = access.stream.elements_per_item
        entry_size = access.layout.value_size // entries
        return (program.geometry.items_per_subcore * entries, entry_size)
    if isinstance(access, DirectLoadAccess) and access.site.node in program.index_nodes:
        assert access.stream is not None
        return (program.geometry.items_per_subcore * access.stream.elements_per_item,)
    return access.layout.storage_shape


class _ScheduleBuilder:
    def __init__(self, program: SparseCoreProgram) -> None:
        self.program = program
        self.tasks: list[Task] = []
        self.value_task: dict[torch.fx.Node, int | None] = {}

    def add(
        self,
        kind: TaskKind,
        node: torch.fx.Node,
        dependencies: Iterable[int | None] = (),
    ) -> int:
        task_id = len(self.tasks)
        task = Task(
            task_id,
            kind,
            node,
            frozenset(dep for dep in dependencies if dep is not None),
        )
        self.tasks.append(task)
        return task_id

    def value(self, node: torch.fx.Node) -> int | None:
        if node in self.value_task:
            return self.value_task[node]
        access = self.program.access_by_node.get(node)
        if access is not None and access.site.kind is AccessKind.LOAD:
            return self.load(access)
        if node.op in ("placeholder", "output"):
            self.value_task[node] = None
            return None

        from ...language._tracing_ops import _get_symnode
        from ...language._tracing_ops import _host_tensor

        if node.target in (_host_tensor, _get_symnode):
            self.value_task[node] = None
            return None

        dependencies = [self.value(parent) for parent in node.all_input_nodes]
        task_id = self.add(TaskKind.COMPUTE, node, dependencies)
        self.value_task[node] = task_id
        return task_id

    def load(self, access: SparseCoreAccess) -> int:
        node = access.site.node
        prior = self.value_task.get(node)
        if prior is not None:
            return prior
        dependencies = [self.value(dep) for dep in access.dependencies]
        if isinstance(access, CachedLoadAccess):
            available = self.add(TaskKind.ONCE_TRANSFER, node, dependencies)
        elif isinstance(access, IndirectLoadAccess) or (
            isinstance(access, DirectLoadAccess)
            and node not in self.program.index_nodes
        ):
            start = self.add(TaskKind.ASYNC_START, node, dependencies)
            available = self.add(TaskKind.ASYNC_WAIT, node, (start,))
        else:
            available = self.add(TaskKind.SYNC_TRANSFER, node, dependencies)
        self.value_task[node] = available
        return available

    def store(self, access: SparseCoreAccess) -> None:
        dependencies = [self.value(dep) for dep in access.dependencies]
        if isinstance(access, AtomicAddAccess):
            dependencies.append(self.add(TaskKind.INITIALIZE, access.site.node))
        store = self.add(TaskKind.STORE, access.site.node, dependencies)
        if isinstance(access, AtomicAddAccess):
            self.add(TaskKind.FINALIZE, access.site.node, (store,))

    def build(self) -> SparseCoreSchedule:
        # Stable task order keeps generated code and errors deterministic.
        for access in self.program.loads:
            self.load(access)
        for access in self.program.stores:
            self.store(access)

        lags: dict[int, int] = {}
        for task in self.tasks:
            dependency_lag = max((lags[dep] for dep in task.dependencies), default=0)
            lags[task.id] = dependency_lag + (
                1 if task.kind is TaskKind.ASYNC_WAIT else 0
            )

        initialization = tuple(
            task
            for task in self.tasks
            if task.kind in (TaskKind.INITIALIZE, TaskKind.ONCE_TRANSFER)
        )
        finalization = tuple(
            task for task in self.tasks if task.kind is TaskKind.FINALIZE
        )
        by_lag: dict[int, list[Task]] = {}
        for task in self.tasks:
            if task.kind in (
                TaskKind.INITIALIZE,
                TaskKind.ONCE_TRANSFER,
                TaskKind.FINALIZE,
            ):
                continue
            lag = lags[task.id]
            by_lag.setdefault(lag, []).append(task)
        stages: list[ScheduleStage] = []
        if initialization:
            stages.append(ScheduleStage(0, initialization, SchedulePhase.INITIALIZE))
        stages.extend(
            ScheduleStage(lag, tuple(tasks)) for lag, tasks in sorted(by_lag.items())
        )
        if finalization:
            stages.append(
                ScheduleStage(
                    max(lags[task.id] for task in finalization),
                    finalization,
                    SchedulePhase.FINALIZE,
                )
            )
        depth = max(by_lag, default=0) + 1
        return SparseCoreSchedule(tuple(stages), depth)


def _verify_resources(program: SparseCoreProgram, schedule: SparseCoreSchedule) -> None:
    ring_bytes = sum(
        math.prod(load_buffer_shape(program, access))
        * access.layout.storage_dtype.itemsize
        * schedule.depth
        for access in program.loads
        if not isinstance(access, CachedLoadAccess)
    )
    cached_bytes = sum(
        math.prod(load_buffer_shape(program, access))
        * access.layout.storage_dtype.itemsize
        for access in program.loads
        if isinstance(access, CachedLoadAccess)
    )
    output_bytes = sum(
        math.prod(access.layout.storage_shape) * access.layout.storage_dtype.itemsize
        for access in program.stores
    )
    output_bytes += sum(
        SC_LANES * access.layout.value_size * access.layout.storage_dtype.itemsize
        for access in program.stores
        if isinstance(access, AtomicAddAccess)
    )
    used = ring_bytes + cached_bytes + output_bytes
    limit = SC_VMEM_BYTES - SC_VMEM_MARGIN
    if used > limit:
        _reject(
            "resource",
            f"VMEM program uses {used} bytes at pipeline depth {schedule.depth}; "
            f"limit is {limit}",
        )
    shared_bytes = sum(
        shared_acc_bytes(int(access.site.tensor.shape[0]), access.layout.value_size)
        for access in program.stores
        if isinstance(access, AtomicAddAccess)
    )
    if shared_bytes > SC_SHARED_BYTES:
        _reject(
            "resource",
            f"shared-memory accumulators use {shared_bytes} bytes; "
            f"limit is {SC_SHARED_BYTES}",
        )


def schedule_sparsecore_program(program: SparseCoreProgram) -> SparseCoreSchedule:
    """Schedule transfers and compute from their dependencies."""
    schedule = _ScheduleBuilder(program).build()
    _verify_resources(program, schedule)
    program.schedule = schedule
    return schedule
