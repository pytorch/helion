from __future__ import annotations

import dataclasses
import itertools
from typing import Literal

from .. import exc
from .tile_dependency import EventUse
from .tile_dependency import InstantiatedActionDomain
from .tile_dependency import InstantiatedTaskFamily
from .tile_dependency import TileDependency
from .tile_dependency import TileDependencyGraph
from .tile_dependency import TileDependencyKind
from .tile_dependency import UniformTaskPartition
from .tile_dependency import dependency_predecessor_sets
from .tile_dependency import instantiate_action_domains
from .tile_dependency import instantiate_action_relations
from .tile_dependency import predecessor_task_ids
from .tile_dependency import prove_uniform_task_partition

CROSS_LOOP_NUM_WORKERS_CONFIG = "cross_loop_num_workers"
CROSS_LOOP_NUM_WORKERS_DEFAULT = 0


@dataclasses.dataclass(frozen=True)
class WorkerScheduleSegment:
    """One compact task-family run in a static persistent-worker schedule.

    ``schedule_begin`` is a linearized position over ``worker_count`` workers.
    Task offset ``i`` is assigned through linear schedule offset::

        offset = schedule_begin + i * schedule_step
        worker = worker_begin + offset % worker_count
        position = offset // worker_count

    Several segments can therefore describe arbitrary numbers of waves without
    materializing one schedule entry per runtime task.
    """

    root: int
    task_begin: int
    task_count: int
    worker_begin: int
    worker_count: int
    schedule_begin: int
    task_step: int = 1
    schedule_step: int = 1
    task_period: int | None = None
    task_period_step: int | None = None
    schedule_period: int | None = None
    schedule_period_step: int | None = None

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError(f"root must be nonnegative, got {self.root}")
        if self.task_begin < 0:
            raise ValueError(f"task_begin must be nonnegative, got {self.task_begin}")
        if self.task_count <= 0:
            raise ValueError(f"task_count must be positive, got {self.task_count}")
        if self.worker_begin < 0:
            raise ValueError(
                f"worker_begin must be nonnegative, got {self.worker_begin}"
            )
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        if self.schedule_begin < 0:
            raise ValueError(
                f"schedule_begin must be nonnegative, got {self.schedule_begin}"
            )
        if self.task_step == 0:
            raise ValueError("task_step must be nonzero")
        if self.schedule_step <= 0:
            raise ValueError(
                f"schedule_step must be positive, got {self.schedule_step}"
            )
        if (self.task_period is None) != (self.task_period_step is None):
            raise ValueError("periodic task order requires both task period fields")
        if self.task_period is not None and self.task_period <= 0:
            raise ValueError(f"task_period must be positive, got {self.task_period}")
        if (self.schedule_period is None) != (self.schedule_period_step is None):
            raise ValueError(
                "periodic schedule order requires both schedule period fields"
            )
        if self.schedule_period is not None and self.schedule_period <= 0:
            raise ValueError(
                f"schedule_period must be positive, got {self.schedule_period}"
            )
        tasks = tuple(self.task_for_offset(offset) for offset in range(self.task_count))
        if min(tasks) < 0:
            raise ValueError("worker schedule segment contains a negative task")
        if len(set(tasks)) != len(tasks):
            raise ValueError("worker schedule segment repeats a task")
        schedule_offsets = tuple(
            self.schedule_for_offset(offset) for offset in range(self.task_count)
        )
        if len(set(schedule_offsets)) != len(schedule_offsets):
            raise ValueError("worker schedule segment repeats a schedule position")

    def task_for_offset(self, task_offset: int) -> int:
        """Return the logical task at one offset within this segment."""
        if not 0 <= task_offset < self.task_count:
            raise IndexError(task_offset)
        if self.task_period is None:
            return self.task_begin + task_offset * self.task_step
        assert self.task_period_step is not None
        return (
            self.task_begin
            + task_offset % self.task_period * self.task_step
            + task_offset // self.task_period * self.task_period_step
        )

    def schedule_for_offset(self, task_offset: int) -> int:
        """Return the linearized worker-stream position for one task offset."""
        if not 0 <= task_offset < self.task_count:
            raise IndexError(task_offset)
        if self.schedule_period is None:
            return self.schedule_begin + task_offset * self.schedule_step
        assert self.schedule_period_step is not None
        return (
            self.schedule_begin
            + task_offset % self.schedule_period * self.schedule_step
            + task_offset // self.schedule_period * self.schedule_period_step
        )

    def placement(self, task: int) -> tuple[int, int] | None:
        """Return ``(worker, position)`` when this segment owns ``task``."""
        if self.task_period is None:
            task_delta = task - self.task_begin
            if task_delta % self.task_step:
                return None
            task_offset = task_delta // self.task_step
            if not 0 <= task_offset < self.task_count:
                return None
        else:
            task_offset = next(
                (
                    offset
                    for offset in range(self.task_count)
                    if self.task_for_offset(offset) == task
                ),
                None,
            )
            if task_offset is None:
                return None
        schedule_offset = self.schedule_for_offset(task_offset)
        return (
            self.worker_begin + schedule_offset % self.worker_count,
            schedule_offset // self.worker_count,
        )

    def task_at(self, worker: int, position: int) -> int | None:
        """Return the task at one worker position, if this segment owns it."""
        worker_offset = worker - self.worker_begin
        if not 0 <= worker_offset < self.worker_count or position < 0:
            return None
        schedule_offset = position * self.worker_count + worker_offset
        schedule_delta = schedule_offset - self.schedule_begin
        if schedule_delta < 0:
            return None
        if self.schedule_period is None:
            if schedule_delta % self.schedule_step:
                return None
            task_offset = schedule_delta // self.schedule_step
        else:
            assert self.schedule_period_step is not None
            outer = schedule_delta // self.schedule_period_step
            inner_delta = schedule_delta % self.schedule_period_step
            if inner_delta % self.schedule_step:
                return None
            inner = inner_delta // self.schedule_step
            if inner >= self.schedule_period:
                return None
            task_offset = outer * self.schedule_period + inner
        if task_offset >= self.task_count:
            return None
        return self.task_for_offset(task_offset)


def _fit_task_sequence(
    tasks: tuple[int, ...],
) -> tuple[int, int | None, int | None] | None:
    """Fit a one- or two-dimensional affine enumeration of logical tasks."""
    if not tasks:
        return None
    if len(tasks) == 1:
        return (1, None, None)
    task_step = tasks[1] - tasks[0]
    if task_step and all(
        task == tasks[0] + offset * task_step for offset, task in enumerate(tasks)
    ):
        return (task_step, None, None)

    period = next(
        (
            offset
            for offset in range(1, len(tasks))
            if tasks[offset] - tasks[offset - 1] != task_step
        ),
        None,
    )
    if period is None or task_step == 0:
        return None
    period_step = tasks[period] - tasks[0]
    if all(
        task == tasks[0] + offset % period * task_step + offset // period * period_step
        for offset, task in enumerate(tasks)
    ):
        return (task_step, period, period_step)
    return None


@dataclasses.dataclass(frozen=True)
class WorkerSchedule:
    """Compressed static execution order for all persistent workers."""

    worker_count: int
    segments: tuple[WorkerScheduleSegment, ...]

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        for segment in self.segments:
            if segment.worker_begin + segment.worker_count > self.worker_count:
                raise ValueError(
                    "worker schedule segment exceeds the resident worker domain"
                )

    def placement(self, root: int, task: int) -> tuple[int, int] | None:
        """Return the unique static placement of one logical task."""
        placements = tuple(
            placement
            for segment in self.segments
            if segment.root == root
            if (placement := segment.placement(task)) is not None
        )
        if len(placements) > 1:
            raise AssertionError(f"task ({root}, {task}) has multiple placements")
        return placements[0] if placements else None

    def task_at(self, worker: int, position: int) -> tuple[int, int] | None:
        """Return the unique ``(root, task)`` at one static schedule position."""
        tasks = tuple(
            (segment.root, task)
            for segment in self.segments
            if (task := segment.task_at(worker, position)) is not None
        )
        if len(tasks) > 1:
            raise AssertionError(
                f"worker {worker} position {position} has multiple tasks"
            )
        return tasks[0] if tasks else None

    def segments_for_root(self, root: int) -> tuple[WorkerScheduleSegment, ...]:
        """Return the compressed static relation for one task family."""
        return tuple(segment for segment in self.segments if segment.root == root)

    def task_order(self, root: int) -> tuple[int, ...]:
        """Return one root's tasks in linearized static execution order."""
        placed_tasks: list[tuple[int, int]] = []
        for segment in self.segments_for_root(root):
            for task_offset in range(segment.task_count):
                schedule_offset = segment.schedule_for_offset(task_offset)
                task = segment.task_for_offset(task_offset)
                placed_tasks.append((schedule_offset, task))
        placed_tasks.sort()
        if any(
            left_offset == right_offset
            for (left_offset, _), (right_offset, _) in itertools.pairwise(placed_tasks)
        ):
            raise AssertionError(f"root {root} has overlapping schedule segments")
        return tuple(task for _offset, task in placed_tasks)

    def workers_for_tasks(
        self,
        root: int,
        tasks: tuple[int, ...],
    ) -> frozenset[int]:
        """Return workers that may execute the selected static tasks."""
        workers: set[int] = set()
        for task in tasks:
            placement = self.placement(root, task)
            if placement is not None:
                workers.add(placement[0])
        return frozenset(workers)

    def without_tasks(
        self,
        excluded: frozenset[tuple[int, int]],
    ) -> WorkerSchedule:
        """Remove dynamically executed tasks while retaining static positions.

        Positions deliberately keep their original values. A removed local task
        executes synchronously at its final contributor, so closing the hole
        would silently change the order of later work on that worker.
        """
        if not excluded:
            return self
        excluded_by_root: dict[int, set[int]] = {}
        for root, task in excluded:
            excluded_by_root.setdefault(root, set()).add(task)
        segments: list[WorkerScheduleSegment] = []
        for segment in self.segments:
            root_excluded = excluded_by_root.get(segment.root)
            if not root_excluded:
                segments.append(segment)
                continue
            for task_offset in range(segment.task_count):
                task = segment.task_for_offset(task_offset)
                if task in root_excluded:
                    continue
                segments.append(
                    WorkerScheduleSegment(
                        root=segment.root,
                        task_begin=task,
                        task_count=1,
                        worker_begin=segment.worker_begin,
                        worker_count=segment.worker_count,
                        schedule_begin=segment.schedule_for_offset(task_offset),
                    )
                )
        return WorkerSchedule(worker_count=self.worker_count, segments=tuple(segments))

    def replacing_root(
        self,
        root: int,
        segments: tuple[WorkerScheduleSegment, ...],
    ) -> WorkerSchedule:
        """Return a schedule with one task family's placement replaced."""
        result: list[WorkerScheduleSegment] = []
        inserted = False
        for segment in self.segments:
            if segment.root != root:
                result.append(segment)
            elif not inserted:
                result.extend(segments)
                inserted = True
        if not inserted:
            result.extend(segments)
        return WorkerSchedule(worker_count=self.worker_count, segments=tuple(result))


def build_baseline_worker_schedule(
    task_families: tuple[InstantiatedTaskFamily, ...],
    worker_count: int,
) -> WorkerSchedule:
    """Represent the existing source-ordered persistent traversal exactly."""
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    segments: list[WorkerScheduleSegment] = []
    position_begin = 0
    for root, family in enumerate(task_families):
        task_count = family.task_count
        if task_count <= 0:
            continue
        task_order = family.physical_traversal
        run_begin = 0
        while run_begin < task_count:
            task_step = (
                task_order[run_begin + 1] - task_order[run_begin]
                if run_begin + 1 < task_count
                else 1
            )
            if task_step == 0:
                task_step = 1
            run_end = run_begin + 1
            while run_end < task_count and task_order[run_end] == (
                task_order[run_begin] + (run_end - run_begin) * task_step
            ):
                run_end += 1
            segments.append(
                WorkerScheduleSegment(
                    root=root,
                    task_begin=task_order[run_begin],
                    task_count=run_end - run_begin,
                    task_step=task_step,
                    worker_begin=0,
                    worker_count=worker_count,
                    schedule_begin=position_begin * worker_count + run_begin,
                )
            )
            run_begin = run_end
        position_begin += (task_count + worker_count - 1) // worker_count
    return WorkerSchedule(worker_count=worker_count, segments=tuple(segments))


def _family_placements_at_position(
    worker_schedule: WorkerSchedule,
    *,
    root: int,
    family: InstantiatedTaskFamily,
    position: int,
    unavailable_workers: frozenset[int] = frozenset(),
) -> tuple[WorkerSchedule, ...]:
    """Return dense placements for one complete family in free worker runs."""
    if family.task_count > worker_schedule.worker_count:
        return ()
    task_order = family.physical_traversal
    task_shape = _fit_task_sequence(task_order)
    if task_shape is None:
        return ()
    free_workers = [
        worker
        for worker in range(worker_schedule.worker_count)
        if worker not in unavailable_workers
        and (
            (occupant := worker_schedule.task_at(worker, position)) is None
            or occupant[0] == root
        )
    ]
    task_step, task_period, task_period_step = task_shape
    result: list[WorkerSchedule] = []
    run_end = len(free_workers)
    while run_end:
        run_begin = run_end - 1
        while run_begin and free_workers[run_begin - 1] == free_workers[run_begin] - 1:
            run_begin -= 1
        if run_end - run_begin >= family.task_count:
            worker_begin = free_workers[run_end - family.task_count]
            result.append(
                worker_schedule.replacing_root(
                    root,
                    (
                        WorkerScheduleSegment(
                            root=root,
                            task_begin=task_order[0],
                            task_count=family.task_count,
                            worker_begin=worker_begin,
                            worker_count=family.task_count,
                            schedule_begin=position * family.task_count,
                            task_step=task_step,
                            task_period=task_period,
                            task_period_step=task_period_step,
                        ),
                    ),
                )
            )
        run_end = run_begin
    return tuple(result)


def _is_valid_worker_schedule(
    event_graph: InstantiatedEventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> bool:
    try:
        validate_worker_schedule(event_graph, worker_schedule, local_triggers)
    except ValueError:
        return False
    return True


def _static_ancestors(
    task: tuple[int, int],
    *,
    worker_schedule: WorkerSchedule,
    local_predecessors: dict[tuple[int, int], frozenset[tuple[int, int]]],
    cache: dict[tuple[int, int], frozenset[tuple[int, int]]],
    visiting: frozenset[tuple[int, int]] = frozenset(),
) -> frozenset[tuple[int, int]]:
    """Contract local execution to the statically scheduled tasks enabling it."""
    if task in cache:
        return cache[task]
    if worker_schedule.placement(*task) is not None:
        result = frozenset((task,))
    elif task in visiting:
        raise ValueError("local trigger graph contains a cycle")
    elif (predecessors := local_predecessors.get(task)) is None:
        result = frozenset()
    else:
        result = frozenset(
            ancestor
            for predecessor in predecessors
            for ancestor in _static_ancestors(
                predecessor,
                worker_schedule=worker_schedule,
                local_predecessors=local_predecessors,
                cache=cache,
                visiting=visiting | frozenset((task,)),
            )
        )
    cache[task] = result
    return result


def place_ready_families(
    event_graph: InstantiatedEventGraph,
    original_schedule: WorkerSchedule,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[WorkerSchedule, tuple[LocalTrigger, ...]]:
    """Move complete ready families into idle capacity during a producer tail.

    Final-arrival execution is useful when no separate workers are available.
    When a complete consumer family fits on workers that are free while some
    of its static ancestors still have queued work, a direct event wait avoids
    extending those producer streams.  This is derived from schedule liveness,
    independent of the roots' operations or graph topology.
    """
    result = worker_schedule
    remaining_triggers = local_triggers
    local_predecessors = _local_trigger_predecessors(event_graph, remaining_triggers)
    static_ancestors_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    candidate_roots = sorted(
        {
            event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
            for trigger in remaining_triggers
        }
    )
    for root in candidate_roots:
        family = event_graph.task_families[root]
        if family.task_count > result.worker_count:
            continue
        root_triggers = tuple(
            trigger
            for trigger in remaining_triggers
            if event_graph.event(trigger.event_index)
            .uses[trigger.use_index]
            .consumer_root
            == root
        )
        readiness_positions: list[int] = []
        ancestor_placements: set[tuple[int, int]] = set()
        valid = True
        for task in range(family.task_count):
            ancestors = _static_ancestors(
                (root, task),
                worker_schedule=result,
                local_predecessors=local_predecessors,
                cache=static_ancestors_cache,
            )
            placements = tuple(
                placement
                for ancestor in ancestors
                if (placement := result.placement(*ancestor)) is not None
            )
            if not placements:
                valid = False
                break
            ancestor_placements.update(placements)
            readiness_positions.append(
                max(position for _worker, position in placements)
            )
        if not valid:
            continue

        original_positions = tuple(
            placement[1]
            for task in range(family.task_count)
            if (placement := original_schedule.placement(root, task)) is not None
        )
        if len(original_positions) != family.task_count:
            continue
        original_position = min(original_positions)
        remaining_without_root = tuple(
            trigger for trigger in remaining_triggers if trigger not in root_triggers
        )
        for position in range(min(readiness_positions) + 1, original_position):
            unfinished_workers = frozenset(
                worker
                for worker, ancestor_position in ancestor_placements
                if ancestor_position >= position
            )
            if not unfinished_workers:
                break
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_position(
                        result,
                        root=root,
                        family=family,
                        position=position,
                        unavailable_workers=unfinished_workers,
                    )
                    if _is_valid_worker_schedule(
                        event_graph, candidate, remaining_without_root
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            result = candidate
            remaining_triggers = remaining_without_root
            local_predecessors = _local_trigger_predecessors(
                event_graph, remaining_triggers
            )
            static_ancestors_cache.clear()
            break
    return result, remaining_triggers


def worker_count_breakpoints(
    event_graph: InstantiatedEventGraph,
) -> tuple[int, ...]:
    """Return worker counts aligned to complete event-key prefixes.

    The tuning input is a kernel-wide worker-count target.  Snapping it to
    these graph-derived boundaries avoids tuning arbitrary partial key groups
    while remaining independent of any particular dependency-chain shape.
    """
    result = {family.task_count for family in event_graph.task_families}
    for event in event_graph.events:
        required_tasks_by_root: dict[int, set[int]] = {}
        contributors_by_key = event.contributor_tasks_by_key
        for key in range(event.key_count):
            for root, task in contributors_by_key[key]:
                required_tasks_by_root.setdefault(root, set()).add(task)
            if required_tasks_by_root:
                result.add(max(map(len, required_tasks_by_root.values())))
    return tuple(sorted(worker_count for worker_count in result if worker_count > 0))


def resolve_worker_count(
    event_graph: InstantiatedEventGraph,
    *,
    default_worker_count: int,
    requested_worker_count: int,
) -> int:
    """Resolve a tuning target to the nearest complete event-key boundary."""
    if requested_worker_count < 0:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG} must be nonnegative, got "
            f"{requested_worker_count}"
        )
    if requested_worker_count == CROSS_LOOP_NUM_WORKERS_DEFAULT:
        return default_worker_count
    return min(
        worker_count_breakpoints(event_graph),
        key=lambda worker_count: (
            abs(worker_count - requested_worker_count),
            worker_count,
        ),
    )


def build_worker_schedule(
    event_graph: InstantiatedEventGraph,
    task_families: tuple[InstantiatedTaskFamily, ...],
    *,
    worker_count: int,
) -> tuple[
    WorkerSchedule,
    tuple[LocalTrigger, ...],
    tuple[CountedEventPlan, ...],
]:
    """Derive local and static task placement for one worker count."""
    baseline = build_baseline_worker_schedule(task_families, worker_count)
    local_triggers = choose_local_triggers(
        event_graph,
        baseline,
        worker_limit=worker_count,
    )
    ordered = order_local_contributors_by_key(
        event_graph,
        baseline,
        local_triggers,
    )
    local_triggers = choose_local_triggers(
        event_graph,
        ordered,
        worker_limit=worker_count,
    )
    local_tasks = frozenset(
        (use.consumer_root, task)
        for trigger in local_triggers
        for use in (event_graph.event(trigger.event_index).uses[trigger.use_index],)
        for task, required_keys in enumerate(use.required_keys_by_task)
        if required_keys
    )
    schedule = ordered.without_tasks(local_tasks)
    schedule, ordered_action_events = place_ordered_action_consumers(
        event_graph,
        schedule,
        local_triggers,
    )
    schedule, local_triggers = place_ready_families(
        event_graph,
        ordered,
        schedule,
        local_triggers,
    )
    validate_worker_schedule(event_graph, schedule, local_triggers)
    return schedule, local_triggers, ordered_action_events


@dataclasses.dataclass(frozen=True)
class LocalTrigger:
    """A task use executed by whichever contributor makes the final arrival."""

    event_index: int
    use_index: int
    possible_workers_by_key: tuple[frozenset[int], ...]

    @property
    def possible_workers(self) -> frozenset[int]:
        return frozenset(
            worker for workers in self.possible_workers_by_key for worker in workers
        )


@dataclasses.dataclass(frozen=True)
class InstantiatedEventContribution:
    """Concrete producer-task to event-key relation for one family."""

    producer_root: int
    keys_by_task: tuple[frozenset[int], ...]
    producer_scope_id: int | None = None


@dataclasses.dataclass(frozen=True)
class InstantiatedEventUse:
    """Concrete event-key requirements at one consumer program point."""

    consumer_root: int
    consumer_access_id: int | None
    placement: Literal["root_entry", "access"]
    required_keys_by_task: tuple[frozenset[int], ...]
    consumer_scope_id: int | None = None


@dataclasses.dataclass(frozen=True)
class InstantiatedKeyedEvent:
    """One configuration-specific event with independent contributions/uses."""

    event_id: int
    source_event_ids: tuple[int, ...]
    key_count: int
    contributions: tuple[InstantiatedEventContribution, ...]
    uses: tuple[InstantiatedEventUse, ...]
    family_done_root: int | None = None

    @property
    def contributor_tasks_by_key(
        self,
    ) -> tuple[frozenset[tuple[int, int]], ...]:
        """Return the producer task instances contributing to every key."""
        contributors: list[set[tuple[int, int]]] = [
            set() for _ in range(self.key_count)
        ]
        for contribution in self.contributions:
            for producer_task, keys in enumerate(contribution.keys_by_task):
                for key in keys:
                    if not 0 <= key < self.key_count:
                        raise ValueError(
                            f"event key {key} is outside [0, {self.key_count})"
                        )
                    contributors[key].add((contribution.producer_root, producer_task))
        return tuple(frozenset(tasks) for tasks in contributors)

    @property
    def expected_arrivals(self) -> tuple[int, ...]:
        """Derive arrival counts from producer relations."""
        return tuple(len(tasks) for tasks in self.contributor_tasks_by_key)

    @property
    def is_family_done(self) -> bool:
        return self.family_done_root is not None


@dataclasses.dataclass(frozen=True)
class InstantiatedEventGraph:
    """Configuration-specific full event DAG used by scheduling proofs."""

    task_families: tuple[InstantiatedTaskFamily, ...]
    events: tuple[InstantiatedKeyedEvent, ...]
    action_domains: tuple[InstantiatedActionDomain, ...] = ()

    def event(self, event_id: int) -> InstantiatedKeyedEvent:
        return self.events[event_id]

    def events_contributed_by(self, root: int) -> tuple[InstantiatedKeyedEvent, ...]:
        return tuple(
            event
            for event in self.events
            if any(
                contribution.producer_root == root
                for contribution in event.contributions
            )
        )

    def uses_for_root(self, root: int) -> tuple[InstantiatedEventUse, ...]:
        return tuple(
            use
            for event in self.events
            for use in event.uses
            if use.consumer_root == root
        )

    def action_domain(self, scope_id: int) -> InstantiatedActionDomain:
        return next(
            domain for domain in self.action_domains if domain.scope_id == scope_id
        )


@dataclasses.dataclass(frozen=True)
class TaskToKeySegment:
    """One compact run of a flattened task-to-event-key mapping."""

    task_begin: int
    task_count: int
    tasks_per_key: int
    first_key: int
    key_stride: int
    key_period: int | None = None


@dataclasses.dataclass(frozen=True)
class CountedEventContribution:
    """One root's contributions to a keyed readiness event."""

    source_event_ids: tuple[int, ...]
    producer_root: int
    task_to_key: tuple[int | None, ...]
    partition: UniformTaskPartition | None = None

    @property
    def task_to_key_segments(self) -> tuple[TaskToKeySegment, ...]:
        return _compress_task_to_key(self.task_to_key)

    @property
    def expected_arrivals(self) -> int:
        arrivals: dict[int, int] = {}
        for key in self.task_to_key:
            if key is not None:
                arrivals[key] = arrivals.get(key, 0) + 1
        counts = set(arrivals.values())
        if len(counts) != 1:
            raise ValueError("keyed event contributor has nonuniform fan-in")
        return counts.pop()


@dataclasses.dataclass(frozen=True)
class CountedEventUse:
    """One configured consumer relation for a counted event."""

    consumer_root: int
    key_by_task: tuple[int, ...]
    consumer_access_id: int | None = None
    consumer_scope_id: int | None = None

    @property
    def task_to_key_segments(self) -> tuple[TaskToKeySegment, ...]:
        return _compress_task_to_key(self.key_by_task)


@dataclasses.dataclass(frozen=True)
class CountedEventPlan:
    """A logical key space receiving contributions from one or more roots.

    Each contributor has an independently proved task-to-key relation. The
    expected count is derived by summing those relations; the event therefore
    represents both ordinary continuations and generic multi-predecessor joins.
    Consumer uses are independent of event identity. ``local_trigger_use``
    identifies the optional use executed by the final arriving contributor.
    """

    contributors: tuple[CountedEventContribution, ...]
    uses: tuple[CountedEventUse, ...]
    local_trigger_use: int | None = None
    graph_event_index: int | None = None

    @property
    def local_use(self) -> CountedEventUse | None:
        if self.local_trigger_use is None:
            return None
        return self.uses[self.local_trigger_use]

    @property
    def source_event_ids(self) -> tuple[int, ...]:
        return tuple(
            event_id
            for contributor in self.contributors
            for event_id in contributor.source_event_ids
        )

    @property
    def key_count(self) -> int:
        """Return the complete event-key domain used by producers or consumers."""
        return (
            max(
                (
                    *(
                        key
                        for contributor in self.contributors
                        for key in contributor.task_to_key
                        if key is not None
                    ),
                    *(key for use in self.uses for key in use.key_by_task),
                ),
                default=-1,
            )
            + 1
        )

    @property
    def expected_arrivals(self) -> int:
        counts = set(self.expected_arrivals_by_key)
        if len(counts) != 1:
            raise ValueError("keyed event has nonuniform total fan-in")
        return counts.pop()

    @property
    def expected_arrivals_by_key(self) -> tuple[int, ...]:
        arrivals = [0] * self.key_count
        for contributor in self.contributors:
            for key in contributor.task_to_key:
                if key is not None:
                    arrivals[key] += 1
        return tuple(arrivals)

    @property
    def is_single_contributor(self) -> bool:
        return len(self.contributors) == 1

    @property
    def single_contributor(self) -> CountedEventContribution:
        if not self.is_single_contributor:
            raise ValueError("keyed event has multiple contributors")
        return self.contributors[0]

    @property
    def event_id(self) -> int:
        event_ids = self.single_contributor.source_event_ids
        if len(event_ids) != 1:
            raise ValueError("keyed event has no source event")
        return event_ids[0]

    @property
    def producer_root(self) -> int:
        return self.single_contributor.producer_root

    @property
    def partition(self) -> UniformTaskPartition:
        partition = self.single_contributor.partition
        if partition is None:
            raise ValueError("keyed event has no uniform partition")
        return partition


def place_ordered_action_consumers(
    event_graph: InstantiatedEventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[WorkerSchedule, tuple[CountedEventPlan, ...]]:
    """Place strands with nested waits and derive their milestone events.

    Exact action dependencies remain the semantic source of truth. This pass
    uses only worker positions and same-strand action order to quotient adjacent
    actions into maximal segments with the same effective readiness frontier.
    It does not inspect operation kinds or recognize a graph topology.
    """
    uses_by_consumer: dict[
        int,
        list[tuple[InstantiatedKeyedEvent, InstantiatedEventUse]],
    ] = {}
    for event in event_graph.events:
        for use in event.uses:
            if use.consumer_scope_id is not None:
                uses_by_consumer.setdefault(use.consumer_root, []).append((event, use))

    local_predecessors = _local_trigger_predecessors(event_graph, local_triggers)
    static_ancestors_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    result = worker_schedule
    plans: list[CountedEventPlan] = []
    for consumer_root, event_uses in sorted(uses_by_consumer.items()):
        # Several nested scopes in one strand require a joint program-order
        # analysis. Keep their exact events, but do not move the strand until
        # that general liveness composition is available.
        if len(event_uses) != 1:
            continue
        event, use = event_uses[0]
        assert use.consumer_scope_id is not None
        domain = event_graph.action_domain(use.consumer_scope_id)
        consumer = event_graph.task_families[consumer_root]
        if (
            domain.root != consumer_root
            or len(domain.nested_axis_order) != 1
            or domain.strand_count != consumer.task_count
            or len(use.required_keys_by_task) != domain.action_count
            or consumer.task_count > result.worker_count
        ):
            continue

        contributors_by_key = event.contributor_tasks_by_key
        predecessors_by_action: list[frozenset[tuple[int, int]]] = []
        readiness_by_strand: list[list[int]] = [[] for _ in range(domain.strand_count)]
        ancestor_placements: set[tuple[int, int]] = set()
        valid = True
        for action, required_keys in enumerate(use.required_keys_by_task):
            predecessors = frozenset(
                predecessor
                for key in required_keys
                for predecessor in contributors_by_key[key]
            )
            predecessors_by_action.append(predecessors)
            if not predecessors:
                valid = False
                break
            ancestors = frozenset(
                ancestor
                for predecessor in predecessors
                for ancestor in _static_ancestors(
                    predecessor,
                    worker_schedule=result,
                    local_predecessors=local_predecessors,
                    cache=static_ancestors_cache,
                )
            )
            placements = tuple(
                placement
                for ancestor in ancestors
                if (placement := result.placement(*ancestor)) is not None
            )
            if not placements:
                valid = False
                break
            ancestor_placements.update(placements)
            readiness_by_strand[domain.strand_task(action)].append(
                max(position for _worker, position in placements)
            )
        if not valid or any(
            len(readiness) != domain.actions_per_strand
            for readiness in readiness_by_strand
        ):
            continue

        current_consumer_positions = tuple(
            placement[1]
            for task in range(consumer.task_count)
            if (placement := result.placement(consumer_root, task)) is not None
        )
        if len(current_consumer_positions) != consumer.task_count:
            continue
        original_position = min(current_consumer_positions)
        chosen: tuple[int, WorkerSchedule] | None = None
        earliest_readiness = min(
            position for pattern in readiness_by_strand for position in pattern
        )
        for position in range(earliest_readiness + 1, original_position):
            busy_workers = frozenset(
                worker
                for worker, ancestor_position in ancestor_placements
                if ancestor_position >= position
            )
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_position(
                        result,
                        root=consumer_root,
                        family=consumer,
                        position=position,
                        unavailable_workers=busy_workers,
                    )
                    if _is_valid_worker_schedule(
                        event_graph,
                        candidate,
                        local_triggers,
                    )
                ),
                None,
            )
            if candidate is not None:
                chosen = (position, candidate)
                break
        if chosen is None:
            continue

        consumer_position, candidate = chosen
        effective_patterns = tuple(
            tuple(max(position, consumer_position - 1) for position in pattern)
            for pattern in readiness_by_strand
        )
        boundaries = [0]
        for action_offset in range(1, domain.actions_per_strand):
            if any(
                pattern[action_offset] != pattern[action_offset - 1]
                for pattern in effective_patterns
            ):
                boundaries.append(action_offset)
        boundaries.append(domain.actions_per_strand)
        if len(boundaries) <= 2:
            continue

        key_by_signature: dict[frozenset[tuple[int, int]], int] = {}
        ordered_signatures: list[frozenset[tuple[int, int]]] = []
        key_by_action = [-1] * domain.action_count
        for strand_task in range(domain.strand_count):
            strand_begin = strand_task * domain.actions_per_strand
            for segment_begin, segment_end in itertools.pairwise(boundaries):
                signature = frozenset(
                    predecessor
                    for action in range(
                        strand_begin + segment_begin,
                        strand_begin + segment_end,
                    )
                    for predecessor in predecessors_by_action[action]
                )
                if not signature:
                    valid = False
                    break
                key = key_by_signature.setdefault(signature, len(key_by_signature))
                if key == len(ordered_signatures):
                    ordered_signatures.append(signature)
                for action in range(
                    strand_begin + segment_begin,
                    strand_begin + segment_end,
                ):
                    key_by_action[action] = key
            if not valid:
                break
        if not valid or any(key < 0 for key in key_by_action):
            continue

        contribution_keys: dict[int, list[int | None]] = {}
        for key, signature in enumerate(ordered_signatures):
            for producer_root, producer_task in signature:
                task_to_key = contribution_keys.get(producer_root)
                if task_to_key is None:
                    new_task_to_key: list[int | None] = [
                        None
                        for _ in range(
                            event_graph.task_families[producer_root].task_count
                        )
                    ]
                    contribution_keys[producer_root] = new_task_to_key
                    task_to_key = new_task_to_key
                previous = task_to_key[producer_task]
                if previous is not None and previous != key:
                    valid = False
                    break
                task_to_key[producer_task] = key
            if not valid:
                break
        if not valid:
            continue

        result = candidate
        plans.append(
            CountedEventPlan(
                contributors=tuple(
                    CountedEventContribution(
                        source_event_ids=event.source_event_ids,
                        producer_root=producer_root,
                        task_to_key=tuple(contribution_keys[producer_root]),
                    )
                    for producer_root in sorted(contribution_keys)
                ),
                uses=(
                    CountedEventUse(
                        consumer_root=consumer_root,
                        key_by_task=tuple(key_by_action),
                        consumer_access_id=use.consumer_access_id,
                        consumer_scope_id=use.consumer_scope_id,
                    ),
                ),
                graph_event_index=event.event_id,
            )
        )
    return result, tuple(plans)


@dataclasses.dataclass(frozen=True)
class CrossLoopSchedule:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    event_graph: InstantiatedEventGraph
    worker_schedule: WorkerSchedule
    local_triggers: tuple[LocalTrigger, ...]
    task_ready_edges: frozenset[tuple[int, int]]
    task_waits_by_root: dict[int, tuple[EventUse, ...]]
    counted_events: tuple[CountedEventPlan, ...]
    worker_limit: int

    @property
    def root_completion_edges(self) -> frozenset[tuple[int, int]]:
        """Return family-completion relations represented by one-key events."""
        return frozenset(
            (event.family_done_root, use.consumer_root)
            for plan in self.counted_events
            if plan.graph_event_index is not None
            for event in (self.event_graph.event(plan.graph_event_index),)
            if event.family_done_root is not None
            for use in plan.uses
        )


def lower_family_done_events(
    event_graph: InstantiatedEventGraph,
    edges: frozenset[tuple[int, int]],
) -> tuple[InstantiatedEventGraph, tuple[CountedEventPlan, ...]]:
    """Represent selected whole-family waits as canonical one-key events.

    The semantic event receives one contribution per logical task. Codegen is
    free to aggregate those arrivals by worker after proving the worker's
    complete task stream has finished.
    """
    consumers_by_producer: dict[int, set[int]] = {}
    for producer_root, consumer_root in edges:
        consumers_by_producer.setdefault(producer_root, set()).add(consumer_root)

    events = [
        dataclasses.replace(event, uses=()) if event.is_family_done else event
        for event in event_graph.events
    ]
    family_event_by_root = {
        event.family_done_root: event
        for event in events
        if event.family_done_root is not None
    }
    plans: list[CountedEventPlan] = []
    for producer_root, consumer_roots in sorted(consumers_by_producer.items()):
        selected_uses = tuple(
            InstantiatedEventUse(
                consumer_root=consumer_root,
                consumer_access_id=None,
                placement="root_entry",
                required_keys_by_task=tuple(
                    frozenset((0,))
                    for _ in range(event_graph.task_families[consumer_root].task_count)
                ),
            )
            for consumer_root in sorted(consumer_roots)
        )
        family_event = family_event_by_root.get(producer_root)
        if family_event is None:
            producer = event_graph.task_families[producer_root]
            family_event = InstantiatedKeyedEvent(
                event_id=len(events),
                source_event_ids=(),
                key_count=1,
                contributions=(
                    InstantiatedEventContribution(
                        producer_root=producer_root,
                        keys_by_task=tuple(
                            frozenset((0,)) for _ in range(producer.task_count)
                        ),
                    ),
                ),
                uses=selected_uses,
                family_done_root=producer_root,
            )
            events.append(family_event)
            family_event_by_root[producer_root] = family_event
        else:
            family_event = dataclasses.replace(family_event, uses=selected_uses)
            events[family_event.event_id] = family_event
        plans.append(
            CountedEventPlan(
                contributors=(
                    CountedEventContribution(
                        source_event_ids=family_event.source_event_ids,
                        producer_root=producer_root,
                        task_to_key=tuple(
                            0
                            for _ in range(
                                event_graph.task_families[producer_root].task_count
                            )
                        ),
                    ),
                ),
                uses=tuple(
                    CountedEventUse(
                        consumer_root=use.consumer_root,
                        key_by_task=tuple(0 for _ in use.required_keys_by_task),
                    )
                    for use in selected_uses
                ),
                graph_event_index=family_event.event_id,
            )
        )
    return dataclasses.replace(event_graph, events=tuple(events)), tuple(plans)


def lower_counted_events(
    event_graph: InstantiatedEventGraph,
    local_triggers: tuple[LocalTrigger, ...],
    dependency_graph: TileDependencyGraph | None = None,
) -> tuple[CountedEventPlan, ...]:
    """Adapt semantic events to the scalar-key counter representation.

    Event identity, contributors, uses, and local execution all come from the
    full DAG. Relations this representation cannot encode monotonically
    coarsen to ``FamilyDone`` rather than selecting another topology-specific
    schedule.
    """
    local_trigger_by_use = {
        (trigger.event_index, trigger.use_index) for trigger in local_triggers
    }
    result: list[CountedEventPlan] = []
    for event in event_graph.events:
        if event.is_family_done or not event.key_count:
            continue
        arrival_counts = set(event.expected_arrivals)
        if len(arrival_counts) != 1 or not next(iter(arrival_counts)):
            continue

        contributions: list[CountedEventContribution] = []
        lowerable = True
        for contribution in event.contributions:
            if any(len(keys) > 1 for keys in contribution.keys_by_task):
                lowerable = False
                break
            task_to_key = tuple(
                next(iter(keys)) if keys else None for keys in contribution.keys_by_task
            )
            partition = None
            if dependency_graph is not None and len(event.uses) == 1:
                use = event.uses[0]
                semantic_waits = tuple(
                    wait
                    for wait in dependency_graph.waits_for_root(use.consumer_root)
                    if wait.event_id in event.source_event_ids
                    and dependency_graph.event(wait.event_id).producer_root
                    == contribution.producer_root
                    and wait.placement == "root_entry"
                    and wait.predecessor_map is not None
                )
                if semantic_waits:
                    producer = event_graph.task_families[contribution.producer_root]
                    consumer = event_graph.task_families[use.consumer_root]
                    candidate = prove_uniform_task_partition(
                        tuple(
                            wait.predecessor_map
                            for wait in semantic_waits
                            if wait.predecessor_map is not None
                        ),
                        consumer_axis_order=consumer.logical_axis_order,
                        consumer_axis_counts=consumer.axis_counts,
                        producer_axis_order=producer.logical_axis_order,
                        producer_axis_counts=producer.axis_counts,
                        block_sizes={**producer.block_sizes, **consumer.block_sizes},
                    )
                    if (
                        candidate is not None
                        and candidate.producer_key_by_task == task_to_key
                    ):
                        partition = candidate
            contributions.append(
                CountedEventContribution(
                    source_event_ids=event.source_event_ids,
                    producer_root=contribution.producer_root,
                    task_to_key=task_to_key,
                    partition=partition,
                )
            )
        if not lowerable:
            continue

        uses: list[CountedEventUse] = []
        use_indices: list[int] = []
        for use_index, use in enumerate(event.uses):
            if (
                use.consumer_scope_id is not None
                or use.placement != "root_entry"
                or any(len(keys) != 1 for keys in use.required_keys_by_task)
            ):
                continue
            uses.append(
                CountedEventUse(
                    consumer_root=use.consumer_root,
                    key_by_task=tuple(
                        next(iter(keys)) for keys in use.required_keys_by_task
                    ),
                    consumer_access_id=use.consumer_access_id,
                )
            )
            use_indices.append(use_index)
        if not uses:
            continue
        selected_local_uses = [
            lowered_index
            for lowered_index, use_index in enumerate(use_indices)
            if (event.event_id, use_index) in local_trigger_by_use
        ]
        if len(selected_local_uses) > 1:
            raise ValueError("one counted event cannot have multiple local executors")
        result.append(
            CountedEventPlan(
                contributors=tuple(contributions),
                uses=tuple(uses),
                local_trigger_use=(
                    selected_local_uses[0] if selected_local_uses else None
                ),
                graph_event_index=event.event_id,
            )
        )
    return tuple(result)


def choose_local_triggers(
    event_graph: InstantiatedEventGraph,
    worker_schedule: WorkerSchedule,
    *,
    worker_limit: int,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[LocalTrigger, ...]:
    """Choose final-arrival execution from complete exact task readiness.

    A one-task family has no task-level parallelism to expose, so it remains in
    the static schedule. A family larger than the resident grid also remains
    static when its completion is consumed coarsely.
    """
    family_done_roots = {
        event.family_done_root
        for event in event_graph.events
        if event.family_done_root is not None and event.uses
    }
    return tuple(
        trigger
        for trigger in derive_local_triggers(event_graph, worker_schedule)
        if (
            use := event_graph.event(trigger.event_index).uses[trigger.use_index]
        ).consumer_root
        not in excluded_roots
        and len(set(event_graph.event(trigger.event_index).expected_arrivals)) == 1
        and event_graph.task_families[use.consumer_root].task_count > 1
        and not (
            use.consumer_root in family_done_roots
            and event_graph.task_families[use.consumer_root].task_count > worker_limit
        )
    )


def choose_counted_events(
    event_graph: InstantiatedEventGraph,
    local_triggers: tuple[LocalTrigger, ...],
    dependency_graph: TileDependencyGraph,
    *,
    excluded_direct_roots: frozenset[int] = frozenset(),
) -> tuple[CountedEventPlan, ...]:
    """Select every uniformly keyed root-entry event for one lowering path.

    Access-local consumers keep their program-point lowering, but excluding one
    use does not discard independent uses of the same semantic event.  A
    one-key event is whole-family completion and remains on the aggregated
    root-completion path unless it owns a local trigger.
    """
    local_uses = {
        (trigger.event_index, trigger.use_index) for trigger in local_triggers
    }
    selected: list[CountedEventPlan] = []
    for event in lower_counted_events(
        event_graph,
        local_triggers,
        dependency_graph,
    ):
        retained_use_indices = tuple(
            use_index
            for use_index, use in enumerate(event.uses)
            if use_index == event.local_trigger_use
            or use.consumer_root not in excluded_direct_roots
        )
        if not retained_use_indices or (
            event.local_trigger_use is None and event.key_count <= 1
        ):
            continue
        selected.append(
            dataclasses.replace(
                event,
                uses=tuple(event.uses[index] for index in retained_use_indices),
                local_trigger_use=(
                    retained_use_indices.index(event.local_trigger_use)
                    if event.local_trigger_use is not None
                    else None
                ),
            )
        )
    selected_local_count = sum(
        event.local_trigger_use is not None for event in selected
    )
    if selected_local_count != len(local_uses):
        raise AssertionError("not every selected local trigger has a lowering event")
    return tuple(selected)


def _compress_task_to_key(
    task_to_key: tuple[int | None, ...],
) -> tuple[TaskToKeySegment, ...]:
    runs: list[tuple[int, int, int]] = []
    task = 0
    while task < len(task_to_key):
        key = task_to_key[task]
        if key is None:
            task += 1
            continue
        begin = task
        task += 1
        while task < len(task_to_key) and task_to_key[task] == key:
            task += 1
        runs.append((begin, task - begin, key))

    segments: list[TaskToKeySegment] = []
    run = 0
    while run < len(runs):
        begin, tasks_per_key, first_key = runs[run]
        end_run = run + 1
        key_stride = 0
        if end_run < len(runs):
            next_begin, next_length, next_key = runs[end_run]
            if next_begin == begin + tasks_per_key and next_length == tasks_per_key:
                key_stride = next_key - first_key
                end_run += 1
                while end_run < len(runs):
                    candidate_begin, candidate_length, candidate_key = runs[end_run]
                    if (
                        candidate_begin != begin + (end_run - run) * tasks_per_key
                        or candidate_length != tasks_per_key
                        or candidate_key != first_key + (end_run - run) * key_stride
                    ):
                        break
                    end_run += 1
        segments.append(
            TaskToKeySegment(
                task_begin=begin,
                task_count=(end_run - run) * tasks_per_key,
                tasks_per_key=tasks_per_key,
                first_key=first_key,
                key_stride=key_stride,
            )
        )
        run = end_run
    repeated: list[TaskToKeySegment] = []
    for segment in segments:
        if repeated:
            previous = repeated[-1]
            period = previous.key_period or previous.task_count
            if (
                segment.task_begin == previous.task_begin + previous.task_count
                and segment.task_count == period
                and segment.tasks_per_key == previous.tasks_per_key
                and segment.first_key == previous.first_key
                and segment.key_stride == previous.key_stride
                and segment.key_period is None
            ):
                repeated[-1] = dataclasses.replace(
                    previous,
                    task_count=previous.task_count + segment.task_count,
                    key_period=period,
                )
                continue
        repeated.append(segment)
    return tuple(repeated)


def instantiate_event_graph(
    dependency_graph: TileDependencyGraph,
    task_families: tuple[InstantiatedTaskFamily, ...],
) -> InstantiatedEventGraph:
    """Instantiate the semantic task/event DAG for one kernel configuration.

    This pass is intentionally independent of execution policy. Exact task
    events retain their complete predecessor sets. A relation that cannot be
    evaluated for this configuration is redirected to the producer's canonical
    one-key family-completion event rather than being partially guessed.
    """
    if len(task_families) != len(dependency_graph.task_families):
        raise ValueError("task family count disagrees with the dependency graph")

    exact_events: list[InstantiatedKeyedEvent] = []
    family_done_uses: dict[
        int,
        dict[tuple[int, int | None, Literal["root_entry", "access"]], list[set[int]]],
    ] = {}

    def add_family_done_use(
        producer_root: int,
        use: EventUse,
    ) -> None:
        consumer_tasks = task_families[use.consumer_root].task_count
        use_key = (
            use.consumer_root,
            use.consumer_access_id,
            use.placement,
        )
        required = family_done_uses.setdefault(producer_root, {}).setdefault(
            use_key,
            [set() for _ in range(consumer_tasks)],
        )
        for keys in required:
            keys.add(0)

    for semantic_event in dependency_graph.events:
        if len(semantic_event.contributors) != 1:
            raise ValueError(
                "semantic multi-contributor events require an explicit task-to-key "
                "relation"
            )
        producer_root = semantic_event.producer_root
        producer = task_families[producer_root]
        semantic_uses = dependency_graph.uses_for_event(semantic_event.event_id)

        if semantic_event.is_family_done:
            for use in semantic_uses:
                add_family_done_use(producer_root, use)
            continue

        uses_by_program_point: dict[
            tuple[int, int | None, Literal["root_entry", "access"]],
            list[set[int]],
        ] = {}
        fallback_uses: list[EventUse] = []
        for use in semantic_uses:
            predecessor_map = use.predecessor_map
            consumer = task_families[use.consumer_root]
            if predecessor_map is None:
                fallback_uses.append(use)
                continue
            required_keys: list[set[int]] = []
            for consumer_task in range(consumer.task_count):
                predecessors = predecessor_task_ids(
                    predecessor_map,
                    consumer_coordinates=consumer.task_coordinates(consumer_task),
                    block_sizes={**producer.block_sizes, **consumer.block_sizes},
                    producer_axis_order=producer.logical_axis_order,
                    producer_axis_counts=producer.axis_counts,
                )
                if predecessors is None:
                    break
                required_keys.append(set(predecessors))
            if len(required_keys) != consumer.task_count:
                fallback_uses.append(use)
                continue
            use_key = (
                use.consumer_root,
                use.consumer_access_id,
                use.placement,
            )
            aggregate = uses_by_program_point.setdefault(
                use_key,
                [set() for _ in range(consumer.task_count)],
            )
            for consumer_task, keys in enumerate(required_keys):
                aggregate[consumer_task].update(keys)

        for use in fallback_uses:
            add_family_done_use(producer_root, use)

        exact_uses = tuple(
            InstantiatedEventUse(
                consumer_root=consumer_root,
                consumer_access_id=consumer_access_id,
                placement=placement,
                required_keys_by_task=tuple(frozenset(keys) for keys in required),
            )
            for (
                consumer_root,
                consumer_access_id,
                placement,
            ), required in sorted(
                uses_by_program_point.items(),
                key=lambda item: (
                    item[0][0],
                    -1 if item[0][1] is None else item[0][1],
                    item[0][2],
                ),
            )
        )
        if exact_uses:
            exact_events.append(
                InstantiatedKeyedEvent(
                    event_id=len(exact_events),
                    source_event_ids=(semantic_event.event_id,),
                    key_count=producer.task_count,
                    contributions=(
                        InstantiatedEventContribution(
                            producer_root=producer_root,
                            keys_by_task=tuple(
                                frozenset((task,))
                                for task in range(producer.task_count)
                            ),
                        ),
                    ),
                    uses=exact_uses,
                )
            )

    result = list(exact_events)
    for producer_root, use_relations in sorted(family_done_uses.items()):
        producer = task_families[producer_root]
        result.append(
            InstantiatedKeyedEvent(
                event_id=len(result),
                source_event_ids=tuple(
                    event.event_id
                    for event in dependency_graph.events_contributed_by(producer_root)
                    if event.is_family_done
                ),
                key_count=1,
                contributions=(
                    InstantiatedEventContribution(
                        producer_root=producer_root,
                        keys_by_task=tuple(
                            frozenset((0,)) for _ in range(producer.task_count)
                        ),
                    ),
                ),
                uses=tuple(
                    InstantiatedEventUse(
                        consumer_root=consumer_root,
                        consumer_access_id=consumer_access_id,
                        placement=placement,
                        required_keys_by_task=tuple(
                            frozenset(keys) for keys in required
                        ),
                    )
                    for (
                        consumer_root,
                        consumer_access_id,
                        placement,
                    ), required in sorted(
                        use_relations.items(),
                        key=lambda item: (
                            item[0][0],
                            -1 if item[0][1] is None else item[0][1],
                            item[0][2],
                        ),
                    )
                ),
                family_done_root=producer_root,
            )
        )

    return InstantiatedEventGraph(task_families=task_families, events=tuple(result))


def add_ordered_action_events(
    event_graph: InstantiatedEventGraph,
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
) -> InstantiatedEventGraph:
    """Add exact nested consumer actions to the configured keyed-event DAG.

    This pass is a predecessor-signature quotient over allocation-derived
    action relations. It has no knowledge of FFN, attention, reductions, or
    operation kinds. Nested producer publication is enabled separately once
    its completion point can be emitted safely; until then, only root-scope
    producers contribute to these additional events.
    """
    action_domains = instantiate_action_domains(
        dependency_graph,
        task_families=event_graph.task_families,
        axis_geometry=axis_geometry,
    )
    domain_by_scope = {domain.scope_id: domain for domain in action_domains}
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    relations = instantiate_action_relations(
        dependency_graph,
        task_families=event_graph.task_families,
        axis_geometry=axis_geometry,
    )
    relations_by_key: dict[
        tuple[TileDependencyKind, int, int, int, int],
        list[tuple[frozenset[int], ...]],
    ] = {}
    for relation in relations:
        relations_by_key.setdefault(
            (
                relation.kind,
                relation.producer_access_id,
                relation.consumer_access_id,
                relation.producer_scope_id,
                relation.consumer_scope_id,
            ),
            [],
        ).append(relation.predecessors_by_consumer_action)

    events = list(event_graph.events)
    for consumer_scope in dependency_graph.execution_scopes:
        if consumer_scope.is_root or not consumer_scope.segmentable:
            continue
        consumer_domain = domain_by_scope.get(consumer_scope.scope_id)
        if consumer_domain is None:
            continue

        matching_dependencies: list[
            tuple[
                TileDependencyKind,
                int,
                int,
                int,
                tuple[frozenset[int], ...],
            ]
        ] = []
        complete = True
        for edge in dependency_graph.edges:
            for access_dependency in edge.access_dependencies:
                consumer_scopes = dependency_graph.scope_ids_by_access[
                    access_dependency.consumer_access_id
                ]
                if consumer_scope.scope_id not in consumer_scopes:
                    continue
                producer_scopes = dependency_graph.scope_ids_by_access[
                    access_dependency.producer_access_id
                ]
                if not producer_scopes:
                    complete = False
                    break
                for producer_scope_id in producer_scopes:
                    producer_scope = scope_by_id[producer_scope_id]
                    producer_domain = domain_by_scope.get(producer_scope_id)
                    relation_key = (
                        access_dependency.kind,
                        access_dependency.producer_access_id,
                        access_dependency.consumer_access_id,
                        producer_scope_id,
                        consumer_scope.scope_id,
                    )
                    proved_relations = relations_by_key.get(relation_key)
                    if (
                        not producer_scope.is_root
                        or producer_domain is None
                        or not proved_relations
                    ):
                        complete = False
                        break
                    matching_dependencies.extend(
                        (
                            access_dependency.kind,
                            access_dependency.producer_access_id,
                            access_dependency.consumer_access_id,
                            producer_scope_id,
                            predecessor_sets,
                        )
                        for predecessor_sets in proved_relations
                    )
                if not complete:
                    break
            if not complete:
                break
        if not complete or not matching_dependencies:
            continue

        signatures: list[frozenset[tuple[int, int]]] = []
        for consumer_action in range(consumer_domain.action_count):
            predecessors: set[tuple[int, int]] = set()
            for (
                _kind,
                _producer_access_id,
                _consumer_access_id,
                producer_scope_id,
                predecessor_sets,
            ) in matching_dependencies:
                producer_domain = domain_by_scope[producer_scope_id]
                predecessors.update(
                    (producer_domain.root, producer_domain.strand_task(action))
                    for action in predecessor_sets[consumer_action]
                )
            signatures.append(frozenset(predecessors))

        key_by_signature: dict[frozenset[tuple[int, int]], int] = {}
        ordered_signatures: list[frozenset[tuple[int, int]]] = []
        key_by_action: list[int | None] = []
        for signature in signatures:
            if not signature:
                key_by_action.append(None)
                continue
            key = key_by_signature.setdefault(signature, len(key_by_signature))
            key_by_action.append(key)
            if key == len(ordered_signatures):
                ordered_signatures.append(signature)
        if not ordered_signatures:
            continue

        contribution_keys: dict[int, list[set[int]]] = {}
        for key, signature in enumerate(ordered_signatures):
            for producer_root, producer_task in signature:
                keys_by_task = contribution_keys.setdefault(
                    producer_root,
                    [
                        set()
                        for _ in range(
                            event_graph.task_families[producer_root].task_count
                        )
                    ],
                )
                keys_by_task[producer_task].add(key)
        source_access_ids = {
            access_id
            for (
                _kind,
                _producer_access_id,
                access_id,
                _producer_scope_id,
                _predecessor_sets,
            ) in matching_dependencies
        }
        source_event_ids = tuple(
            sorted(
                {
                    wait.event_id
                    for wait in dependency_graph.waits
                    if wait.consumer_root == consumer_scope.root
                    and wait.consumer_access_id in source_access_ids
                }
            )
        )
        events.append(
            InstantiatedKeyedEvent(
                event_id=len(events),
                source_event_ids=source_event_ids,
                key_count=len(ordered_signatures),
                contributions=tuple(
                    InstantiatedEventContribution(
                        producer_root=producer_root,
                        keys_by_task=tuple(
                            frozenset(keys) for keys in contribution_keys[producer_root]
                        ),
                    )
                    for producer_root in sorted(contribution_keys)
                ),
                uses=(
                    InstantiatedEventUse(
                        consumer_root=consumer_scope.root,
                        consumer_access_id=(
                            next(iter(source_access_ids))
                            if len(source_access_ids) == 1
                            else None
                        ),
                        placement="access",
                        required_keys_by_task=tuple(
                            frozenset(()) if key is None else frozenset((key,))
                            for key in key_by_action
                        ),
                        consumer_scope_id=consumer_scope.scope_id,
                    ),
                ),
            )
        )
    return dataclasses.replace(
        event_graph,
        events=tuple(events),
        action_domains=action_domains,
    )


def canonicalize_ready_events(
    event_graph: InstantiatedEventGraph,
) -> InstantiatedEventGraph:
    """Group complete root-entry predecessor signatures into ready keys.

    This is a graph rewrite over direct event relations. It does not recognize
    model topology or choose an executor. A producer task may contribute to
    several keys, several roots may contribute to one key, and several consumer
    tasks may require the same key.
    """
    original_events = event_graph.events
    events = list(original_events)

    contributors_by_event_key: dict[
        int,
        tuple[frozenset[tuple[int, int]], ...],
    ] = {event.event_id: event.contributor_tasks_by_key for event in original_events}
    uses_to_remove: set[tuple[int, int]] = set()

    for consumer_root, consumer in enumerate(event_graph.task_families):
        source_uses = tuple(
            (event.event_id, use_index, use)
            for event in original_events
            if not event.is_family_done
            for use_index, use in enumerate(event.uses)
            if use.consumer_root == consumer_root and use.placement == "root_entry"
        )
        if not source_uses:
            continue

        signatures: list[frozenset[tuple[int, int]]] = []
        for consumer_task in range(consumer.task_count):
            predecessors: set[tuple[int, int]] = set()
            for event_id, _use_index, use in source_uses:
                for key in use.required_keys_by_task[consumer_task]:
                    predecessors.update(contributors_by_event_key[event_id][key])
            signatures.append(frozenset(predecessors))

        already_canonical = len(source_uses) == 1 and all(
            len(keys) <= 1 for keys in source_uses[0][2].required_keys_by_task
        )
        if already_canonical:
            continue

        key_by_signature: dict[frozenset[tuple[int, int]], int] = {}
        key_by_consumer_task: list[int | None] = []
        ordered_signatures: list[frozenset[tuple[int, int]]] = []
        for signature in signatures:
            if not signature:
                key_by_consumer_task.append(None)
                continue
            key = key_by_signature.setdefault(signature, len(key_by_signature))
            key_by_consumer_task.append(key)
            if key == len(ordered_signatures):
                ordered_signatures.append(signature)
        if not ordered_signatures:
            continue

        contribution_keys: dict[int, list[set[int]]] = {}
        for key, signature in enumerate(ordered_signatures):
            for producer_root, producer_task in signature:
                keys_by_task = contribution_keys.setdefault(
                    producer_root,
                    [
                        set()
                        for _ in range(
                            event_graph.task_families[producer_root].task_count
                        )
                    ],
                )
                keys_by_task[producer_task].add(key)

        canonical_event = InstantiatedKeyedEvent(
            event_id=len(events),
            source_event_ids=tuple(
                sorted(
                    {
                        source_event_id
                        for event_id, _use_index, _use in source_uses
                        for source_event_id in original_events[
                            event_id
                        ].source_event_ids
                    }
                )
            ),
            key_count=len(ordered_signatures),
            contributions=tuple(
                InstantiatedEventContribution(
                    producer_root=producer_root,
                    keys_by_task=tuple(
                        frozenset(keys) for keys in contribution_keys[producer_root]
                    ),
                )
                for producer_root in sorted(contribution_keys)
            ),
            uses=(
                InstantiatedEventUse(
                    consumer_root=consumer_root,
                    consumer_access_id=None,
                    placement="root_entry",
                    required_keys_by_task=tuple(
                        frozenset(()) if key is None else frozenset((key,))
                        for key in key_by_consumer_task
                    ),
                ),
            ),
        )

        uses_to_remove.update(
            (event_id, use_index) for event_id, use_index, _use in source_uses
        )
        events.append(canonical_event)

    for event_id, event in enumerate(original_events):
        retained_uses = tuple(
            use
            for use_index, use in enumerate(event.uses)
            if (event_id, use_index) not in uses_to_remove
        )
        if retained_uses != event.uses:
            events[event_id] = dataclasses.replace(event, uses=retained_uses)

    merged_events: dict[
        tuple[
            int,
            tuple[InstantiatedEventContribution, ...],
            int | None,
        ],
        InstantiatedKeyedEvent,
    ] = {}
    for event in events:
        if not event.uses:
            continue
        signature = (event.key_count, event.contributions, event.family_done_root)
        previous = merged_events.get(signature)
        if previous is None:
            merged_events[signature] = event
            continue
        merged_events[signature] = dataclasses.replace(
            previous,
            source_event_ids=tuple(
                sorted({*previous.source_event_ids, *event.source_event_ids})
            ),
            uses=tuple(dict.fromkeys((*previous.uses, *event.uses))),
        )
    retained_events = tuple(merged_events.values())
    return dataclasses.replace(
        event_graph,
        events=tuple(
            dataclasses.replace(event, event_id=event_id)
            for event_id, event in enumerate(retained_events)
        ),
    )


def canonicalize_task_readiness(
    event_graph: InstantiatedEventGraph,
    dependency_graph: TileDependencyGraph,
) -> InstantiatedEventGraph:
    """Promote complete exact predecessor sets to root-entry readiness.

    Access-local placement describes where a wait is minimally required. When
    every incoming memory relation for a task family has an exact task overlap,
    the full predecessor signature also proves a conservative root-entry wait
    and makes the complete opaque task eligible for local or direct execution.
    This is a graph normalization over all incoming edges, not a topology
    matcher for any particular chain.
    """
    access_by_id = {access.access_id: access for access in dependency_graph.accesses}
    edges_by_consumer: dict[int, list[TileDependency]] = {}
    for edge in dependency_graph.edges:
        edges_by_consumer.setdefault(edge.consumer_root, []).append(edge)

    replacement_events: list[InstantiatedKeyedEvent] = []
    replaced_consumers: set[int] = set()
    for consumer_root, incoming_edges in sorted(edges_by_consumer.items()):
        consumer = event_graph.task_families[consumer_root]
        predecessors_by_task: list[set[tuple[int, int]]] = [
            set() for _ in range(consumer.task_count)
        ]
        valid = True
        for edge in incoming_edges:
            predecessor_sets = dependency_predecessor_sets(
                edge,
                task_families=event_graph.task_families,
                access_by_id=access_by_id,
            )
            if predecessor_sets is None:
                valid = False
                break
            for consumer_task, producer_tasks in enumerate(predecessor_sets):
                predecessors_by_task[consumer_task].update(
                    (edge.producer_root, producer_task)
                    for producer_task in producer_tasks
                )
        signatures = tuple(frozenset(tasks) for tasks in predecessors_by_task)
        if not valid or any(not signature for signature in signatures):
            continue

        key_by_signature: dict[frozenset[tuple[int, int]], int] = {}
        ordered_signatures: list[frozenset[tuple[int, int]]] = []
        key_by_task: list[int] = []
        for signature in signatures:
            key = key_by_signature.setdefault(signature, len(key_by_signature))
            key_by_task.append(key)
            if key == len(ordered_signatures):
                ordered_signatures.append(signature)
        contribution_keys: dict[int, list[set[int]]] = {}
        for key, signature in enumerate(ordered_signatures):
            for producer_root, producer_task in signature:
                keys_by_task = contribution_keys.setdefault(
                    producer_root,
                    [
                        set()
                        for _ in range(
                            event_graph.task_families[producer_root].task_count
                        )
                    ],
                )
                keys_by_task[producer_task].add(key)

        semantic_event_ids = tuple(
            sorted(
                {use.event_id for use in dependency_graph.waits_for_root(consumer_root)}
            )
        )
        replacement_events.append(
            InstantiatedKeyedEvent(
                event_id=-1,
                source_event_ids=semantic_event_ids,
                key_count=len(ordered_signatures),
                contributions=tuple(
                    InstantiatedEventContribution(
                        producer_root=producer_root,
                        keys_by_task=tuple(
                            frozenset(keys) for keys in contribution_keys[producer_root]
                        ),
                    )
                    for producer_root in sorted(contribution_keys)
                ),
                uses=(
                    InstantiatedEventUse(
                        consumer_root=consumer_root,
                        consumer_access_id=None,
                        placement="root_entry",
                        required_keys_by_task=tuple(
                            frozenset((key,)) for key in key_by_task
                        ),
                    ),
                ),
            )
        )
        replaced_consumers.add(consumer_root)

    retained_events = [
        dataclasses.replace(
            event,
            uses=tuple(
                use for use in event.uses if use.consumer_root not in replaced_consumers
            ),
        )
        for event in event_graph.events
    ]
    events = [event for event in retained_events if event.uses]
    events.extend(replacement_events)
    return dataclasses.replace(
        event_graph,
        events=tuple(
            dataclasses.replace(event, event_id=event_id)
            for event_id, event in enumerate(events)
        ),
    )


def derive_local_triggers(
    event_graph: InstantiatedEventGraph,
    worker_schedule: WorkerSchedule,
) -> tuple[LocalTrigger, ...]:
    """Select complete one-task-per-key uses for final-arrival execution."""
    prerequisite_uses_by_task: dict[
        tuple[int, int],
        list[tuple[int, int]],
    ] = {}
    for event in event_graph.events:
        for use_index, use in enumerate(event.uses):
            if use.consumer_scope_id is not None:
                continue
            for consumer_task, keys in enumerate(use.required_keys_by_task):
                if keys:
                    prerequisite_uses_by_task.setdefault(
                        (use.consumer_root, consumer_task), []
                    ).append((event.event_id, use_index))

    possible_workers_by_task: dict[tuple[int, int], frozenset[int]] = {}
    for root, family in enumerate(event_graph.task_families):
        for task in range(family.task_count):
            placement = worker_schedule.placement(root, task)
            if placement is not None:
                possible_workers_by_task[(root, task)] = frozenset((placement[0],))

    candidates: list[
        tuple[
            int,
            int,
            int,
            InstantiatedKeyedEvent,
            InstantiatedEventUse,
            tuple[int, ...],
        ]
    ] = []
    for event in event_graph.events:
        if event.is_family_done or len(event.uses) != 1:
            continue
        if any(
            len(keys) > 1
            for contribution in event.contributions
            for keys in contribution.keys_by_task
        ):
            continue
        use_index = 0
        use = event.uses[use_index]
        if use.consumer_scope_id is not None or use.placement != "root_entry":
            continue
        if not event.key_count or any(not count for count in event.expected_arrivals):
            continue
        consumer_task_by_key: list[int | None] = [None] * event.key_count
        valid = True
        for consumer_task, required_keys in enumerate(use.required_keys_by_task):
            if len(required_keys) != 1 or prerequisite_uses_by_task.get(
                (use.consumer_root, consumer_task)
            ) != [(event.event_id, use_index)]:
                valid = False
                break
            key = next(iter(required_keys))
            if consumer_task_by_key[key] is not None:
                valid = False
                break
            consumer_task_by_key[key] = consumer_task
        if not valid or any(task is None for task in consumer_task_by_key):
            continue

        candidates.append(
            (
                use.consumer_root,
                event.event_id,
                use_index,
                event,
                use,
                tuple(task for task in consumer_task_by_key if task is not None),
            )
        )

    candidate_count_by_contributor: dict[tuple[int, int], int] = {}
    for _root, _event_id, _use_index, event, _use, _tasks in candidates:
        for contributor in {
            contributor
            for contributors in event.contributor_tasks_by_key
            for contributor in contributors
        }:
            candidate_count_by_contributor[contributor] = (
                candidate_count_by_contributor.get(contributor, 0) + 1
            )

    result: list[LocalTrigger] = []
    for _consumer_root, event_id, use_index, event, use, tasks_by_key in sorted(
        candidates
    ):
        if any(
            candidate_count_by_contributor[contributor] != 1
            for contributors in event.contributor_tasks_by_key
            for contributor in contributors
        ):
            continue
        valid = True
        possible_workers_by_key: list[frozenset[int]] = []
        for key, contributors in enumerate(event.contributor_tasks_by_key):
            possible_workers = frozenset(
                worker
                for contributor in contributors
                for worker in possible_workers_by_task.get(contributor, ())
            )
            if not possible_workers:
                valid = False
                break
            possible_workers_by_key.append(possible_workers)
            consumer_task = tasks_by_key[key]
            possible_workers_by_task[(use.consumer_root, consumer_task)] = (
                possible_workers
            )
        if valid:
            result.append(
                LocalTrigger(
                    event_index=event_id,
                    use_index=use_index,
                    possible_workers_by_key=tuple(possible_workers_by_key),
                )
            )
    return tuple(result)


def order_local_contributors_by_key(
    event_graph: InstantiatedEventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> WorkerSchedule:
    """Order eligible static contributors by ready key.

    Key-major ordering is a topology-independent consequence of choosing a
    final-arrival executor: completing one key at a time exposes local work as
    early as possible. The transformation is applied only when one event
    contribution bijectively covers a complete static task family and its
    inverse is representable by affine schedule segments. Other families keep
    their existing traversal.
    """
    local_tasks = _local_trigger_predecessors(event_graph, local_triggers).keys()
    local_roots = {root for root, _task in local_tasks}
    replacement_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}

    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        if len(event.contributions) != 1:
            continue
        contribution = event.contributions[0]
        root = contribution.producer_root
        if root in local_roots or root in replacement_by_root:
            continue
        family = event_graph.task_families[root]
        if len(contribution.keys_by_task) != family.task_count or any(
            len(keys) != 1 for keys in contribution.keys_by_task
        ):
            continue

        tasks_by_key: list[list[int]] = [[] for _ in range(event.key_count)]
        for task, keys in enumerate(contribution.keys_by_task):
            key = next(iter(keys))
            if not 0 <= key < event.key_count:
                raise ValueError(f"event key {key} is outside its declared domain")
            tasks_by_key[key].append(task)
        fan_ins = {len(tasks) for tasks in tasks_by_key}
        if len(fan_ins) != 1 or not fan_ins or not (fan_in := fan_ins.pop()):
            continue

        placements = tuple(
            worker_schedule.placement(root, task) for task in range(family.task_count)
        )
        if any(placement is None for placement in placements):
            continue
        schedule_offsets = sorted(
            position * worker_schedule.worker_count + worker
            for placement in placements
            if placement is not None
            for worker, position in (placement,)
        )
        schedule_begin = schedule_offsets[0]
        if schedule_offsets != list(
            range(schedule_begin, schedule_begin + family.task_count)
        ):
            continue

        replacement: list[WorkerScheduleSegment] = []
        within_key = 0
        while within_key < fan_in:
            selected: (
                tuple[
                    int,
                    tuple[int, ...],
                    tuple[int, int | None, int | None],
                ]
                | None
            ) = None
            for end in range(fan_in, within_key, -1):
                task_sequence = tuple(
                    task for tasks in tasks_by_key for task in tasks[within_key:end]
                )
                if (task_shape := _fit_task_sequence(task_sequence)) is not None:
                    selected = (end, task_sequence, task_shape)
                    break
            if selected is None:
                replacement = []
                break
            end, task_sequence, task_shape = selected
            task_step, task_period, task_period_step = task_shape
            within_count = end - within_key
            replacement.append(
                WorkerScheduleSegment(
                    root=root,
                    task_begin=task_sequence[0],
                    task_count=len(task_sequence),
                    task_step=task_step,
                    worker_begin=0,
                    worker_count=worker_schedule.worker_count,
                    schedule_begin=schedule_begin + within_key,
                    task_period=task_period,
                    task_period_step=task_period_step,
                    schedule_period=within_count,
                    schedule_period_step=fan_in,
                )
            )
            within_key = end
        if replacement:
            replacement_by_root[root] = tuple(replacement)

    if not replacement_by_root:
        return worker_schedule
    segments: list[WorkerScheduleSegment] = []
    inserted_roots: set[int] = set()
    for segment in worker_schedule.segments:
        root_replacement = replacement_by_root.get(segment.root)
        if root_replacement is None:
            segments.append(segment)
        elif segment.root not in inserted_roots:
            segments.extend(root_replacement)
            inserted_roots.add(segment.root)
    return WorkerSchedule(
        worker_count=worker_schedule.worker_count, segments=tuple(segments)
    )


def _local_trigger_predecessors(
    event_graph: InstantiatedEventGraph,
    local_triggers: tuple[LocalTrigger, ...],
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    """Return the complete predecessor set for every locally executed task."""
    result: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        use = event.uses[trigger.use_index]
        contributors_by_key = event.contributor_tasks_by_key
        for consumer_task, required_keys in enumerate(use.required_keys_by_task):
            if len(required_keys) != 1:
                raise ValueError("a local trigger requires exactly one key per task")
            key = next(iter(required_keys))
            task = (use.consumer_root, consumer_task)
            if task in result:
                raise ValueError(f"task {task} has multiple local triggers")
            result[task] = contributors_by_key[key]
    return result


def validate_worker_schedule(
    event_graph: InstantiatedEventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...] = (),
) -> None:
    """Prove that static worker order and event dependencies are acyclic.

    A blocking wait adds an edge from every contributing producer task to the
    consumer task. Consecutive tasks on one persistent worker add execution-
    order edges. A cycle in their union is a concrete static deadlock: every
    task in the cycle waits for work sequenced after it on some worker.

    Access-local waits are conservatively treated as task-entry prerequisites.
    This may reject a legal schedule but cannot admit a cyclic one. Local
    triggers are added to this proof when they replace their baseline static
    placements.
    """
    task_nodes = {
        (root, task)
        for root, family in enumerate(event_graph.task_families)
        for task in range(family.task_count)
    }
    local_predecessors = _local_trigger_predecessors(event_graph, local_triggers)
    static_tasks = task_nodes - local_predecessors.keys()
    tasks_by_worker: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(worker_schedule.worker_count)
    ]
    for root, task in sorted(task_nodes):
        placement = worker_schedule.placement(root, task)
        if (root, task) in local_predecessors:
            if placement is not None:
                raise ValueError(
                    f"locally executed task ({root}, {task}) also has a static placement"
                )
            continue
        if placement is None:
            raise ValueError(f"task ({root}, {task}) has no static placement")
        worker, position = placement
        tasks_by_worker[worker].append((position, (root, task)))

    successors: dict[tuple[int, int], set[tuple[int, int]]] = {
        task: set() for task in static_tasks
    }
    indegree = dict.fromkeys(static_tasks, 0)

    static_ancestors_cache: dict[
        tuple[int, int],
        frozenset[tuple[int, int]],
    ] = {}

    def add_edge(
        producer: tuple[int, int],
        consumer: tuple[int, int],
    ) -> None:
        if producer == consumer or consumer in successors[producer]:
            return
        successors[producer].add(consumer)
        indegree[consumer] += 1

    for worker_tasks in tasks_by_worker:
        worker_tasks.sort()
        if any(
            left_position == right_position
            for (left_position, _), (right_position, _) in itertools.pairwise(
                worker_tasks
            )
        ):
            raise ValueError("multiple tasks occupy one worker schedule position")
        for (_, producer), (_, consumer) in itertools.pairwise(worker_tasks):
            add_edge(producer, consumer)

    for event in event_graph.events:
        contributors_by_key = event.contributor_tasks_by_key
        for use in event.uses:
            consumer_tasks = (
                range(len(use.required_keys_by_task))
                if use.consumer_scope_id is None
                else (
                    event_graph.action_domain(use.consumer_scope_id).strand_task(action)
                    for action in range(len(use.required_keys_by_task))
                )
            )
            for consumer_task, required_keys in zip(
                consumer_tasks,
                use.required_keys_by_task,
                strict=True,
            ):
                consumer = (use.consumer_root, consumer_task)
                if consumer in local_predecessors:
                    continue
                for key in required_keys:
                    for producer in contributors_by_key[key]:
                        ancestors = _static_ancestors(
                            producer,
                            worker_schedule=worker_schedule,
                            local_predecessors=local_predecessors,
                            cache=static_ancestors_cache,
                        )
                        if not ancestors:
                            raise ValueError(f"task {producer} has no executor")
                        for ancestor in ancestors:
                            add_edge(ancestor, consumer)

    ready = [task for task, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        task = ready.pop()
        visited += 1
        for successor in successors[task]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(static_tasks):
        blocked = sorted(task for task, degree in indegree.items() if degree)
        raise ValueError(
            f"worker schedule contains a dependency/order cycle involving {blocked[:8]}"
        )


def build_cross_loop_schedule(
    *,
    dependency_plan: TileDependencyGraph,
    task_families: tuple[InstantiatedTaskFamily, ...],
    axis_geometry: dict[int, tuple[int, int]],
    excluded_roots: frozenset[int],
    preordered_edges: frozenset[tuple[int, int]],
    physical_worker_limit: int,
    requested_worker_count: int = CROSS_LOOP_NUM_WORKERS_DEFAULT,
) -> CrossLoopSchedule:
    """Derive all generic readiness strategies without inspecting root bodies."""
    event_graph = canonicalize_ready_events(
        canonicalize_task_readiness(
            instantiate_event_graph(dependency_plan, task_families),
            dependency_plan,
        )
    )
    event_graph = add_ordered_action_events(
        event_graph,
        dependency_plan,
        axis_geometry=axis_geometry,
    )
    worker_limit = resolve_worker_count(
        event_graph,
        default_worker_count=physical_worker_limit,
        requested_worker_count=requested_worker_count,
    )
    candidate_task_pairs, waits_by_root = _select_available_waits(
        dependency_plan=dependency_plan,
        task_families=task_families,
        excluded_roots=excluded_roots,
    )
    action_capable_pairs = {
        (contribution.producer_root, use.consumer_root)
        for event in event_graph.events
        for contribution in event.contributions
        for use in event.uses
        if use.consumer_scope_id is not None
    }
    candidate_task_pairs = frozenset(candidate_task_pairs | action_capable_pairs)
    waits_by_root = {
        root: tuple(
            wait
            for wait in waits
            if (
                dependency_plan.event(wait.event_id).producer_root,
                wait.consumer_root,
            )
            in candidate_task_pairs
        )
        for root, waits in waits_by_root.items()
    }
    waits_by_root = {root: waits for root, waits in waits_by_root.items() if waits}
    retained_waits = waits_by_root
    try:
        worker_schedule, local_triggers, ordered_action_events = build_worker_schedule(
            event_graph,
            task_families,
            worker_count=worker_limit,
        )
    except ValueError as error:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG}={requested_worker_count} does not "
            "admit a progress-safe worker schedule"
        ) from error

    ordered_action_roots = frozenset(
        use.consumer_root for plan in ordered_action_events for use in plan.uses
    )
    counted_events = (
        *choose_counted_events(
            event_graph,
            local_triggers,
            dependency_plan,
            excluded_direct_roots=ordered_action_roots,
        ),
        *ordered_action_events,
    )
    selected_task_pairs = {
        (contributor.producer_root, use.consumer_root)
        for event in counted_events
        for contributor in event.contributors
        for use in event.uses
    }
    retained_waits = {
        root: tuple(
            wait
            for wait in waits
            if (
                dependency_plan.event(wait.event_id).producer_root,
                root,
            )
            not in selected_task_pairs
        )
        for root, waits in retained_waits.items()
    }
    retained_waits = {root: waits for root, waits in retained_waits.items() if waits}
    retained_wait_pairs = {
        (dependency_plan.event(wait.event_id).producer_root, consumer_root)
        for consumer_root, waits in retained_waits.items()
        for wait in waits
    }
    lowered_task_pairs = frozenset(selected_task_pairs | retained_wait_pairs)
    # Recompute coverage from the mechanisms that will actually be emitted.
    # Dependency analysis may prove a finer relation than the selected emitter
    # can materialize. Such a relation must monotonically coarsen to root
    # completion; retaining a task-ready classification without an emitter
    # would remove the dependency entirely.
    root_completion_edges = _select_root_completion_edges(
        dependencies=dependency_plan.edges,
        fully_task_ready_edges=lowered_task_pairs,
        preordered_edges=preordered_edges,
    )
    task_ready_edges = lowered_task_pairs
    root_order_edges = set(root_completion_edges) | set(preordered_edges)
    redundant_wait_pairs = {
        (dependency_plan.event(wait.event_id).producer_root, consumer_root)
        for consumer_root, waits in retained_waits.items()
        for wait in waits
        if _is_ordered_by_root_completion(
            dependency_plan.event(wait.event_id).producer_root,
            consumer_root,
            root_order_edges,
        )
    }
    if redundant_wait_pairs:
        retained_waits = {
            consumer_root: tuple(
                wait
                for wait in waits
                if (
                    dependency_plan.event(wait.event_id).producer_root,
                    consumer_root,
                )
                not in redundant_wait_pairs
            )
            for consumer_root, waits in retained_waits.items()
        }
        retained_waits = {
            root: waits for root, waits in retained_waits.items() if waits
        }
        task_ready_edges = frozenset(task_ready_edges - redundant_wait_pairs)
    _validate_schedule_coverage(
        dependencies=dependency_plan.edges,
        task_ready_edges=task_ready_edges,
        root_completion_edges=root_completion_edges,
    )
    event_graph, family_done_events = lower_family_done_events(
        event_graph,
        root_completion_edges,
    )
    counted_events = (*counted_events, *family_done_events)
    validate_worker_schedule(event_graph, worker_schedule, local_triggers)
    return CrossLoopSchedule(
        event_graph=event_graph,
        worker_schedule=worker_schedule,
        local_triggers=local_triggers,
        task_ready_edges=task_ready_edges,
        task_waits_by_root=retained_waits,
        counted_events=counted_events,
        worker_limit=worker_limit,
    )


def _validate_schedule_coverage(
    *,
    dependencies: tuple[TileDependency, ...],
    task_ready_edges: frozenset[tuple[int, int]],
    root_completion_edges: frozenset[tuple[int, int]],
) -> None:
    """Verify that every dependence has an emitted synchronization path."""
    covered = set(task_ready_edges) | set(root_completion_edges)
    root_order_edges = set(root_completion_edges)
    for dependency in dependencies:
        pair = (dependency.producer_root, dependency.consumer_root)
        if pair in covered or _is_ordered_by_root_completion(*pair, root_order_edges):
            continue
        raise exc.CrossLoopSchedulingError(
            f"{dependency.producer_root}->{dependency.consumer_root} through "
            f"allocations {sorted(dependency.tensor_names)!r} has no cross-loop "
            "synchronization path"
        )


def _select_root_completion_edges(
    *,
    dependencies: tuple[TileDependency, ...],
    fully_task_ready_edges: frozenset[tuple[int, int]],
    preordered_edges: frozenset[tuple[int, int]],
) -> frozenset[tuple[int, int]]:
    """Choose the minimal source-ordered root-completion fallback edges."""
    ordered_root_edges: set[tuple[int, int]] = set()
    for dependency in sorted(
        dependencies,
        key=lambda edge: (
            edge.consumer_root - edge.producer_root,
            edge.producer_root,
            edge.consumer_root,
        ),
    ):
        pair = (dependency.producer_root, dependency.consumer_root)
        if pair in fully_task_ready_edges or pair in preordered_edges:
            continue
        if _is_ordered_by_root_completion(*pair, ordered_root_edges):
            continue
        ordered_root_edges.add(pair)
    return frozenset(ordered_root_edges)


def _is_ordered_by_root_completion(
    producer: int,
    consumer: int,
    edges: set[tuple[int, int]],
) -> bool:
    """Return whether whole-root ordering transitively covers one pair."""
    pending = [producer]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current == consumer:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(target for source, target in edges if source == current)
    return False


def _select_available_waits(
    *,
    dependency_plan: TileDependencyGraph,
    task_families: tuple[InstantiatedTaskFamily, ...],
    excluded_roots: frozenset[int],
) -> tuple[
    frozenset[tuple[int, int]],
    dict[int, tuple[EventUse, ...]],
]:
    kinds_by_pair: dict[tuple[int, int], set[TileDependencyKind]] = {}
    for dependency in dependency_plan.edges:
        pair = (dependency.producer_root, dependency.consumer_root)
        kinds_by_pair.setdefault(pair, set()).update(dependency.kinds)

    waits_by_pair: dict[tuple[int, int], list[EventUse]] = {}
    for wait in dependency_plan.waits:
        event = dependency_plan.event(wait.event_id)
        waits_by_pair.setdefault((event.producer_root, wait.consumer_root), []).append(
            wait
        )

    fully_task_ready_pairs: set[tuple[int, int]] = set()
    waits_by_root: dict[int, list[EventUse]] = {}
    for pair in kinds_by_pair:
        producer_root, consumer_root = pair
        producer = task_families[producer_root]
        consumer = task_families[consumer_root]
        pair_waits = waits_by_pair.get(pair, ())
        all_task_waits = tuple(
            wait
            for wait in pair_waits
            if dependency_plan.event(wait.event_id).granularity == "task"
        )
        task_waits = tuple(
            wait for wait in all_task_waits if wait.placement == "root_entry"
        )
        declared_root_waits = tuple(
            wait
            for wait in pair_waits
            if dependency_plan.event(wait.event_id).granularity == "root"
        )
        can_use_tasks = (
            producer_root not in excluded_roots
            and consumer_root not in excluded_roots
            # A one-task event is exactly root completion. Keep one canonical
            # representation so singleton producers do not allocate and poll
            # a second, equivalent task-event array.
            and producer.task_count > 1
            and consumer.task_count > 0
            and bool(task_waits)
            and len(task_waits) == len(all_task_waits)
        )
        producer_axes = set(producer.logical_axis_order)
        consumer_axes = set(consumer.logical_axis_order)
        for wait in task_waits:
            predecessor_map = wait.predecessor_map
            if (
                predecessor_map is None
                or {axis.producer_block_id for axis in predecessor_map.axes}
                != producer_axes
            ):
                can_use_tasks = False
                break
            required_consumer_axes = {
                axis.consumer_block_id for axis in predecessor_map.axes
            }
            if not required_consumer_axes <= consumer_axes:
                can_use_tasks = False
                break

        if not can_use_tasks:
            continue

        selected_waits = waits_by_root.setdefault(consumer_root, [])
        existing = {
            (
                wait.event_id,
                wait.predecessor_map.axes,
                None,
            )
            for wait in selected_waits
            if wait.predecessor_map is not None
        }
        for wait in task_waits:
            assert wait.predecessor_map is not None
            key = (
                wait.event_id,
                wait.predecessor_map.axes,
                None,
            )
            if key not in existing:
                selected_waits.append(wait)
                existing.add(key)

        if not declared_root_waits:
            fully_task_ready_pairs.add(pair)

    return (
        frozenset(fully_task_ready_pairs),
        {root: tuple(waits) for root, waits in waits_by_root.items()},
    )
