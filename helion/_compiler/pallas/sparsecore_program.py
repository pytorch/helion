"""SparseCore program and dependency-derived pipeline schedule."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import math
from typing import TYPE_CHECKING

from .sparsecore_base import SC_VMEM_BYTES
from .sparsecore_base import SC_VMEM_MARGIN
from .sparsecore_base import _reject
from .sparsecore_plan import DirectLoadPlan
from .sparsecore_plan import DirectStorePlan
from .sparsecore_plan import IndirectLoadPlan
from .sparsecore_plan import IndirectStorePlan
from .sparsecore_plan import SparseCoreMemoryPlan

if TYPE_CHECKING:
    import torch


class TaskKind(Enum):
    SYNC_TRANSFER = "sync_transfer"
    ASYNC_START = "async_start"
    ASYNC_WAIT = "async_wait"
    STORE = "store"


@dataclass(frozen=True)
class Task:
    kind: TaskKind
    node: torch.fx.Node


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
    def item_blocks(self) -> int:
        return math.ceil(self.item_count / self.tile_size)

    @property
    def padded_items(self) -> int:
        return self.item_blocks * self.tile_size

    def __post_init__(self) -> None:
        self.plan_by_node = {plan.access.node: plan for plan in self.memory_plans}
        if len(self.plan_by_node) != len(self.memory_plans):
            raise AssertionError("duplicate SparseCore memory plan")
        self.index_nodes = frozenset(
            plan.index_node
            for plan in self.memory_plans
            if isinstance(plan, (IndirectLoadPlan, IndirectStorePlan))
        )
        for plan in self.memory_plans:
            if not isinstance(plan, (IndirectLoadPlan, IndirectStorePlan)):
                continue
            index_plan = self.plan_by_node.get(plan.index_node)
            if not isinstance(index_plan, (DirectLoadPlan, IndirectLoadPlan)):
                _reject(
                    "access_pattern",
                    "indirect index must be produced by a memory load",
                    node=plan.access.node,
                )
            if isinstance(plan, IndirectStorePlan) and isinstance(
                index_plan, IndirectLoadPlan
            ):
                _reject(
                    "access_pattern",
                    "dependent indices for indirect stores are not implemented",
                    node=plan.access.node,
                )
        for index_node in self.index_nodes:
            for user in index_node.users:
                plan = self.plan_by_node.get(user)
                if (
                    isinstance(plan, (IndirectLoadPlan, IndirectStorePlan))
                    and plan.index_node is index_node
                    and plan.access.value_node is not index_node
                ):
                    continue
                _reject(
                    "access_pattern",
                    "a load used as an indirect index cannot also be used as a value",
                    node=index_node,
                )

    @property
    def loads(self) -> tuple[DirectLoadPlan | IndirectLoadPlan, ...]:
        return tuple(
            plan
            for plan in self.memory_plans
            if isinstance(plan, (DirectLoadPlan, IndirectLoadPlan))
        )

    @property
    def stores(self) -> tuple[DirectStorePlan | IndirectStorePlan, ...]:
        return tuple(
            plan
            for plan in self.memory_plans
            if isinstance(plan, (DirectStorePlan, IndirectStorePlan))
        )


def load_buffer_shape(
    program: SparseCoreProgram, plan: DirectLoadPlan | IndirectLoadPlan
) -> tuple[int, ...]:
    """Scratch shape for one load and one pipeline slot."""
    if isinstance(plan, IndirectLoadPlan):
        entries = plan.transfer.elements_per_item
        entry_size = plan.layout.elements_per_item // entries
        return (program.items_per_subcore * entries, entry_size)
    if isinstance(plan, DirectLoadPlan) and plan.access.node in program.index_nodes:
        return (program.items_per_subcore * plan.transfer.elements_per_item,)
    return plan.layout.storage_shape


class _ScheduleBuilder:
    def __init__(self, program: SparseCoreProgram) -> None:
        self.program = program
        self.tasks: list[tuple[int, Task]] = []
        self.value_lags: dict[torch.fx.Node, int] = {}

    def add(self, lag: int, kind: TaskKind, node: torch.fx.Node) -> None:
        self.tasks.append((lag, Task(kind, node)))

    def value_lag(self, node: torch.fx.Node) -> int:
        if node in self.value_lags:
            return self.value_lags[node]
        plan = self.program.plan_by_node.get(node)
        if isinstance(plan, (DirectLoadPlan, IndirectLoadPlan)):
            return self.load(plan)
        if node.op in ("placeholder", "output"):
            self.value_lags[node] = 0
            return 0

        from ...language._tracing_ops import _get_symnode
        from ...language._tracing_ops import _host_tensor

        if node.target in (_host_tensor, _get_symnode):
            self.value_lags[node] = 0
            return 0

        lag = max(
            (self.value_lag(parent) for parent in node.all_input_nodes), default=0
        )
        self.value_lags[node] = lag
        return lag

    def load(self, plan: DirectLoadPlan | IndirectLoadPlan) -> int:
        node = plan.access.node
        prior = self.value_lags.get(node)
        if prior is not None:
            return prior
        lag = (
            self.value_lag(plan.index_node) if isinstance(plan, IndirectLoadPlan) else 0
        )
        if isinstance(plan, IndirectLoadPlan) or (
            isinstance(plan, DirectLoadPlan) and node not in self.program.index_nodes
        ):
            self.add(lag, TaskKind.ASYNC_START, node)
            lag += 1
            self.add(lag, TaskKind.ASYNC_WAIT, node)
        else:
            self.add(lag, TaskKind.SYNC_TRANSFER, node)
        self.value_lags[node] = lag
        return lag

    def store(self, plan: DirectStorePlan | IndirectStorePlan) -> None:
        value_node = plan.access.value_node
        assert value_node is not None
        lag = self.value_lag(value_node)
        if isinstance(plan, IndirectStorePlan):
            lag = max(lag, self.value_lag(plan.index_node))
        self.add(lag, TaskKind.STORE, plan.access.node)

    def build(self) -> SparseCoreSchedule:
        # Stable task order keeps generated code and errors deterministic.
        for plan in self.program.loads:
            self.load(plan)
        for plan in self.program.stores:
            self.store(plan)

        by_lag: dict[int, list[Task]] = {}
        for lag, task in self.tasks:
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


def schedule_sparsecore_program(program: SparseCoreProgram) -> None:
    """Build the transfer schedule from FX value dependencies."""
    schedule = _ScheduleBuilder(program).build()
    _verify_resources(program, schedule)
    program.schedule = schedule
