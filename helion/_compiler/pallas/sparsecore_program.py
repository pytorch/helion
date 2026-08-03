"""SparseCore program and dependency-derived pipeline schedule."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import math
from typing import TYPE_CHECKING

from .memory_access import MemoryAccessKind
from .sc_base import SC_VMEM_BYTES
from .sc_base import SC_VMEM_MARGIN
from .sc_base import _reject
from .sparsecore_plan import DirectLoadPlan
from .sparsecore_plan import IndirectLoadPlan
from .sparsecore_plan import IndirectStorePlan
from .sparsecore_plan import SparseCoreMemoryPlan

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch


class TaskKind(Enum):
    SYNC_TRANSFER = "sync_transfer"
    ASYNC_START = "async_start"
    ASYNC_WAIT = "async_wait"
    COMPUTE = "compute"
    STORE = "store"


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


@dataclass(frozen=True)
class SparseCoreSchedule:
    stages: tuple[ScheduleStage, ...]
    depth: int


@dataclass
class SparseCoreProgram:
    graph: torch.fx.Graph
    memory_plans: tuple[SparseCoreMemoryPlan, ...]
    item_count: int
    tile_size: int
    num_cores: int
    num_subcores: int
    plan_by_node: dict[torch.fx.Node, SparseCoreMemoryPlan] = field(init=False)
    index_nodes: frozenset[torch.fx.Node] = field(init=False)
    schedule: SparseCoreSchedule | None = None

    @property
    def total_subcores(self) -> int:
        return self.num_cores * self.num_subcores

    @property
    def items_per_subcore(self) -> int:
        return self.tile_size // self.total_subcores

    @property
    def blocks(self) -> int:
        return math.ceil(self.item_count / self.tile_size)

    @property
    def padded_items(self) -> int:
        return self.blocks * self.tile_size

    def __post_init__(self) -> None:
        self.plan_by_node = {plan.access.node: plan for plan in self.memory_plans}
        if len(self.plan_by_node) != len(self.memory_plans):
            raise AssertionError("duplicate SparseCore memory plan")
        self.index_nodes = frozenset(
            plan.index_node
            for plan in self.memory_plans
            if isinstance(plan, (IndirectLoadPlan, IndirectStorePlan))
        )
        offset_loads = [
            plan
            for plan in self.memory_plans
            if isinstance(plan, IndirectLoadPlan) and plan.index_offset
        ]
        for plan in offset_loads:
            index_plan = self.plan_by_node.get(plan.index_node)
            if isinstance(index_plan, IndirectLoadPlan):
                _reject(
                    "schedule",
                    "static offsets on dependent indices are not implemented",
                    node=plan.access.node,
                )
            users = [
                candidate
                for candidate in self.memory_plans
                if isinstance(candidate, IndirectLoadPlan)
                and candidate.index_node is plan.index_node
            ]
            if len(users) > 1:
                _reject(
                    "schedule",
                    "an index adjusted by a static offset cannot be shared",
                    node=plan.access.node,
                )

    @property
    def loads(self) -> tuple[SparseCoreMemoryPlan, ...]:
        return tuple(
            plan
            for plan in self.memory_plans
            if plan.access.kind is MemoryAccessKind.LOAD
        )

    @property
    def stores(self) -> tuple[SparseCoreMemoryPlan, ...]:
        return tuple(
            plan
            for plan in self.memory_plans
            if plan.access.kind is not MemoryAccessKind.LOAD
        )


def load_buffer_shape(
    program: SparseCoreProgram, plan: SparseCoreMemoryPlan
) -> tuple[int, ...]:
    """Scratch shape for one load and one pipeline slot."""
    if isinstance(plan, IndirectLoadPlan):
        assert plan.stream is not None
        entries = plan.stream.elements_per_item
        entry_size = plan.layout.value_size // entries
        return (program.items_per_subcore * entries, entry_size)
    if isinstance(plan, DirectLoadPlan) and plan.access.node in program.index_nodes:
        assert plan.stream is not None
        return (program.items_per_subcore * plan.stream.elements_per_item,)
    return plan.layout.storage_shape


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
        plan = self.program.plan_by_node.get(node)
        if plan is not None and plan.access.kind is MemoryAccessKind.LOAD:
            return self.load(plan)
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

    def load(self, plan: SparseCoreMemoryPlan) -> int:
        node = plan.access.node
        prior = self.value_task.get(node)
        if prior is not None:
            return prior
        dependencies = [self.value(dep) for dep in plan.dependencies]
        if isinstance(plan, IndirectLoadPlan) or (
            isinstance(plan, DirectLoadPlan) and node not in self.program.index_nodes
        ):
            start = self.add(TaskKind.ASYNC_START, node, dependencies)
            available = self.add(TaskKind.ASYNC_WAIT, node, (start,))
        else:
            available = self.add(TaskKind.SYNC_TRANSFER, node, dependencies)
        self.value_task[node] = available
        return available

    def store(self, plan: SparseCoreMemoryPlan) -> None:
        dependencies = [self.value(dep) for dep in plan.dependencies]
        self.add(TaskKind.STORE, plan.access.node, dependencies)

    def build(self) -> SparseCoreSchedule:
        # Stable task order keeps generated code and errors deterministic.
        for plan in self.program.loads:
            self.load(plan)
        for plan in self.program.stores:
            self.store(plan)

        lags: dict[int, int] = {}
        for task in self.tasks:
            dependency_lag = max((lags[dep] for dep in task.dependencies), default=0)
            lags[task.id] = dependency_lag + (
                1 if task.kind is TaskKind.ASYNC_WAIT else 0
            )

        by_lag: dict[int, list[Task]] = {}
        for task in self.tasks:
            lag = lags[task.id]
            by_lag.setdefault(lag, []).append(task)
        stages = tuple(
            ScheduleStage(lag, tuple(tasks)) for lag, tasks in sorted(by_lag.items())
        )
        depth = max(by_lag, default=0) + 1
        return SparseCoreSchedule(stages, depth)


def _verify_resources(program: SparseCoreProgram, schedule: SparseCoreSchedule) -> None:
    ring_bytes = sum(
        math.prod(load_buffer_shape(program, plan))
        * plan.layout.storage_dtype.itemsize
        * schedule.depth
        for plan in program.loads
    )
    output_bytes = sum(
        math.prod(plan.layout.storage_shape) * plan.layout.storage_dtype.itemsize
        for plan in program.stores
    )
    used = ring_bytes + output_bytes
    limit = SC_VMEM_BYTES - SC_VMEM_MARGIN
    if used > limit:
        _reject(
            "resource",
            f"VMEM program uses {used} bytes at pipeline depth {schedule.depth}; "
            f"limit is {limit}",
        )


def schedule_sparsecore_program(program: SparseCoreProgram) -> SparseCoreSchedule:
    """Schedule transfers and compute from their dependencies."""
    schedule = _ScheduleBuilder(program).build()
    _verify_resources(program, schedule)
    program.schedule = schedule
    return schedule
