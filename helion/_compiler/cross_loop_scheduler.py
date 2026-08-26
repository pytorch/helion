from __future__ import annotations

import dataclasses
from functools import cache
from functools import cached_property
import heapq
import itertools
import operator

import sympy

from .. import exc
from .tile_dependency import DependencyPoint
from .tile_dependency import LogicalDomain
from .tile_dependency import LogicalRelation
from .tile_dependency import TileDependencyGraph
from .tile_dependency import instantiate_root_domains
from .tile_dependency import instantiate_scope_domains
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import logical_axis_symbol
from .tile_dependency import preceding_scope_relation

CROSS_LOOP_NUM_WORKERS_CONFIG = "cross_loop_num_workers"
CROSS_LOOP_NUM_WORKERS_DEFAULT = 0

# Some validation and compaction paths deliberately expand the configured task
# DAG.  Keep that work bounded; larger affine schedules retain their symbolic
# representation and the baseline compressed schedule.
_EXPLICIT_SCHEDULE_TASK_LIMIT = 4096
_CLC_COMMAND_TASK_LIMIT = 8192
_EXPLICIT_SCHEDULE_EVENT_KEY_LIMIT = 65_536
_EXPLICIT_SCHEDULE_EDGE_LIMIT = 1_048_576


class ClcCommandPlanUnavailable(exc.CrossLoopSchedulingError):
    """The graph is valid, but exceeds the bounded CLC command planner."""


@dataclasses.dataclass
class _ClcMaterializationBudget:
    """Bound all explicitly materialized CLC graph incidences."""

    limit: int | None
    used: int = 0

    def consume(self, count: int, description: str) -> None:
        self.used += count
        if self.limit is not None and self.used > self.limit:
            raise ClcCommandPlanUnavailable(
                f"CLC command planning exceeded its {self.limit}-item "
                f"materialization budget while {description}"
            )


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
    task_relation: LogicalRelation | None = None

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
        if self.task_relation is not None:
            if (
                self.task_relation.source_domain.size != self.task_count
                or self.task_relation.source_domain.kind != "worker"
                or self.task_relation.target_domain.kind != "scope"
                or not self.task_relation.pieces
            ):
                raise ValueError(
                    "symbolic worker schedule relation has incompatible domains"
                )
        else:
            tasks = tuple(
                self.task_for_offset(offset) for offset in range(self.task_count)
            )
            if min(tasks) < 0:
                raise ValueError("worker schedule segment contains a negative task")
            if len(set(tasks)) != len(tasks):
                raise ValueError("worker schedule segment repeats a task")
        if (
            self.schedule_period is not None
            and self.schedule_period_step is not None
            and self.schedule_period_step
            <= (self.schedule_period - 1) * self.schedule_step
        ):
            raise ValueError("periodic worker schedule must be strictly ordered")

    def task_for_offset(self, task_offset: int) -> int:
        """Return the logical task at one offset within this segment."""
        if not 0 <= task_offset < self.task_count:
            raise IndexError(task_offset)
        if self.task_relation is not None:
            source_coordinates = self.task_relation.source_domain.coordinates(
                task_offset
            )
            targets = self.task_relation.target_coordinates(source_coordinates)
            if len(targets) != 1:
                raise AssertionError(
                    "symbolic schedule ordinal does not map to one logical task"
                )
            return self.task_relation.target_domain.index(
                dict(
                    zip(
                        self.task_relation.target_domain.axis_order,
                        next(iter(targets)),
                        strict=True,
                    )
                )
            )
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

    @cached_property
    def _task_offset_by_task(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for offset in range(self.task_count):
            task = self.task_for_offset(offset)
            if task in result:
                raise AssertionError("worker schedule segment repeats a task")
            result[task] = offset
        return result

    def placement(self, task: int) -> tuple[int, int] | None:
        """Return ``(worker, position)`` when this segment owns ``task``."""
        if self.task_relation is not None:
            inverse = self.task_relation.inverse()
            offsets = (
                inverse.targets(task)
                if inverse is not None
                else frozenset(
                    (self._task_offset_by_task[task],)
                    if task in self._task_offset_by_task
                    else ()
                )
            )
            if len(offsets) > 1:
                raise AssertionError("symbolic schedule maps one task more than once")
            if not offsets:
                return None
            task_offset = next(iter(offsets))
        elif self.task_period is None:
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


def _flat_domain_index_expression(domain: LogicalDomain) -> sympy.Expr:
    """Return the canonical flattened index of a logical coordinate."""
    result: sympy.Expr = sympy.Integer(0)
    multiplier = 1
    for axis in domain.axis_order:
        result += logical_axis_symbol(axis) * multiplier  # pyrefly: ignore[unsupported-operation]
        multiplier *= domain.axis_counts[axis]
    return sympy.simplify(result)


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
        """Return one task's placement without expanding the full schedule."""
        placements = tuple(
            placement
            for segment in self.segments_for_root(root)
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

    def workers_for_root(self, root: int) -> frozenset[int]:
        """Return the compact worker support of one statically placed family."""
        return frozenset(
            worker
            for segment in self.segments_for_root(root)
            for worker in range(
                segment.worker_begin,
                segment.worker_begin + segment.worker_count,
            )
        )

    def _placement_axes(self) -> tuple[int, int]:
        maximum = max(
            (
                axis
                for segment in self.segments
                if segment.task_relation is not None
                for domain in (
                    segment.task_relation.source_domain,
                    segment.task_relation.target_domain,
                )
                for axis in domain.axis_order
            ),
            default=0,
        )
        worker_axis = maximum + 1
        return worker_axis, worker_axis + 1

    @cached_property
    def placement_domain(self) -> LogicalDomain:
        """The worker/stream-position coordinates of static execution."""
        worker_axis, position_axis = self._placement_axes()
        maximum_position = max(
            (
                segment.schedule_for_offset(segment.task_count - 1)
                // segment.worker_count
                for segment in self.segments
            ),
            default=0,
        )
        return LogicalDomain(
            axis_order=(worker_axis, position_axis),
            axis_counts_items=(
                (worker_axis, self.worker_count),
                (position_axis, maximum_position + 1),
            ),
            kind="worker",
        )

    @cached_property
    def position_domain(self) -> LogicalDomain:
        """The projected stream-position coordinate used for frontier math."""
        position_axis = self.placement_domain.axis_order[1]
        return LogicalDomain(
            axis_order=(position_axis,),
            axis_counts_items=(
                (position_axis, self.placement_domain.axis_counts[position_axis]),
            ),
            kind="value",
        )

    def placement_relation(
        self,
        segment: WorkerScheduleSegment,
    ) -> LogicalRelation | None:
        """Map one segment's ordinal coordinates to worker and stream position."""
        relation = segment.task_relation
        if relation is None:
            return None
        ordinal = _flat_domain_index_expression(relation.source_domain)
        if segment.schedule_period is None:
            schedule_offset = segment.schedule_begin + ordinal * segment.schedule_step  # pyrefly: ignore[unsupported-operation]
        else:
            assert segment.schedule_period_step is not None
            schedule_offset = (
                segment.schedule_begin
                + sympy.Mod(ordinal, segment.schedule_period) * segment.schedule_step  # pyrefly: ignore[unsupported-operation]
                + sympy.floor(ordinal / segment.schedule_period)  # pyrefly: ignore[unsupported-operation]
                * segment.schedule_period_step  # pyrefly: ignore[unsupported-operation]
            )
        worker = segment.worker_begin + sympy.Mod(  # pyrefly: ignore[unsupported-operation]
            schedule_offset,
            segment.worker_count,
        )
        position = sympy.floor(schedule_offset / segment.worker_count)
        return LogicalRelation.point_map(
            relation.source_domain,
            self.placement_domain,
            (  # pyrefly: ignore[bad-argument-type]
                (
                    tuple(
                        (axis, 0, relation.source_domain.axis_counts[axis], 1)
                        for axis in relation.source_domain.axis_order
                    ),
                    (worker, position),
                ),
            ),
        )

    def position_relation(
        self,
        segment: WorkerScheduleSegment,
    ) -> LogicalRelation | None:
        """Project one symbolic placement relation to stream position."""
        placement = self.placement_relation(segment)
        return (
            None
            if placement is None
            else placement.project_target(self.position_domain)
        )

    def last_positions_for_root(self, root: int) -> dict[int, int] | None:
        """Return each participating worker's final stream position."""
        result: dict[int, int] = {}
        for segment in self.segments_for_root(root):
            if segment.schedule_period is not None or segment.schedule_step != 1:
                return None
            begin = segment.schedule_begin
            end = begin + segment.task_count
            for local_worker in range(segment.worker_count):
                first = begin + (local_worker - begin) % segment.worker_count
                if first >= end:
                    continue
                last = first + (end - 1 - first) // segment.worker_count * (
                    segment.worker_count
                )
                worker = segment.worker_begin + local_worker
                result[worker] = max(
                    result.get(worker, -1),
                    last // segment.worker_count,
                )
        return result

    def position_bounds_for_root(self, root: int) -> tuple[int, int] | None:
        """Return the first and last occupied stream positions for one root."""
        segments = self.segments_for_root(root)
        if not segments:
            return None
        positions = tuple(
            (
                segment.schedule_for_offset(0) // segment.worker_count,
                segment.schedule_for_offset(segment.task_count - 1)
                // segment.worker_count,
            )
            for segment in segments
        )
        return (
            min(begin for begin, _end in positions),
            max(end for _begin, end in positions),
        )

    def contiguous_global_interval(self, root: int) -> tuple[int, int] | None:
        """Return one dense global schedule interval without task expansion."""
        segments = sorted(
            self.segments_for_root(root),
            key=lambda segment: segment.schedule_begin,
        )
        if not segments:
            return None
        begin = segments[0].schedule_begin
        end = begin
        for segment in segments:
            if (
                segment.worker_begin != 0
                or segment.worker_count != self.worker_count
                or segment.schedule_period is not None
                or segment.schedule_step != 1
                or segment.schedule_begin != end
            ):
                return None
            end += segment.task_count
        return begin, end

    def without_roots(self, roots: frozenset[int]) -> WorkerSchedule:
        """Remove complete locally executed families without task expansion."""
        if not roots:
            return self
        segments = tuple(
            segment for segment in self.segments if segment.root not in roots
        )
        if not segments:
            raise ValueError("cannot construct an empty worker schedule")
        return WorkerSchedule(
            worker_count=self.worker_count,
            segments=segments,
        )

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
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...],
    worker_count: int,
) -> WorkerSchedule:
    """Represent the existing source-ordered persistent traversal exactly."""
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    segments: list[WorkerScheduleSegment] = []
    position_begin = 0
    if len(root_domains) != len(root_traversals):
        raise ValueError("root domains and traversals must have equal length")
    for root, (domain, traversal) in enumerate(
        zip(root_domains, root_traversals, strict=True)
    ):
        task_count = domain.size
        if task_count <= 0:
            continue
        active_workers = min(worker_count, task_count)
        segments.append(
            WorkerScheduleSegment(
                root=root,
                task_begin=0,
                task_count=task_count,
                worker_begin=0,
                worker_count=active_workers,
                schedule_begin=position_begin * active_workers,
                task_relation=traversal,
            )
        )
        position_begin += (task_count + worker_count - 1) // worker_count
    return WorkerSchedule(worker_count=worker_count, segments=tuple(segments))


def _family_placements_at_position(
    worker_schedule: WorkerSchedule,
    *,
    root: int,
    task_domain: LogicalDomain,
    task_traversal: LogicalRelation,
    position: int,
    unavailable_workers: frozenset[int] = frozenset(),
    pack_high: bool = True,
) -> tuple[WorkerSchedule, ...]:
    """Return dense placements for one complete family in free worker runs."""
    if task_domain.size > worker_schedule.worker_count:
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
    result: list[WorkerSchedule] = []
    run_end = len(free_workers)
    while run_end:
        run_begin = run_end - 1
        while run_begin and free_workers[run_begin - 1] == free_workers[run_begin] - 1:
            run_begin -= 1
        if run_end - run_begin >= task_domain.size:
            worker_begin = free_workers[
                run_end - task_domain.size if pack_high else run_begin
            ]
            result.append(
                worker_schedule.replacing_root(
                    root,
                    (
                        WorkerScheduleSegment(
                            root=root,
                            task_begin=0,
                            task_count=task_domain.size,
                            worker_begin=worker_begin,
                            worker_count=task_domain.size,
                            schedule_begin=position * task_domain.size,
                            task_relation=task_traversal,
                        ),
                    ),
                )
            )
        run_end = run_begin
    return tuple(result if pack_high else reversed(result))


def place_initial_ready_families(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
) -> WorkerSchedule:
    """Pack independent source families into idle slots of earlier waves."""
    incoming_roots = {
        use.consumer_root
        for event in event_graph.events
        for use in event.uses
        if any(
            contribution.producer_root != use.consumer_root
            for contribution in event.contributions
        )
    }
    result = worker_schedule
    for root, task_domain in enumerate(event_graph.root_domains):
        if root in incoming_roots or task_domain.size > result.worker_count:
            continue
        current_bounds = result.position_bounds_for_root(root)
        if current_bounds is None:
            continue
        original_position = current_bounds[0]
        for position in range(original_position):
            candidate = next(
                iter(
                    _family_placements_at_position(
                        result,
                        root=root,
                        task_domain=task_domain,
                        task_traversal=event_graph.root_traversals[root],
                        position=position,
                    )
                ),
                None,
            )
            if candidate is not None:
                result = candidate
                break
    return result


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
    event_graph: EventGraph,
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
    candidate_roots = sorted(
        {
            event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
            for trigger in remaining_triggers
        }
    )
    for root in candidate_roots:
        task_domain = event_graph.root_domains[root]
        if task_domain.size > result.worker_count:
            continue
        root_triggers = tuple(
            trigger
            for trigger in remaining_triggers
            if event_graph.event(trigger.event_index)
            .uses[trigger.use_index]
            .consumer_root
            == root
        )
        if len(root_triggers) != 1:
            continue
        trigger = root_triggers[0]
        trigger_event = event_graph.event(trigger.event_index)
        trigger_use = trigger_event.uses[trigger.use_index]
        completion = _event_completion_positions(
            event_graph,
            trigger_event,
            worker_schedule=result,
            local_triggers=remaining_triggers,
        )
        if completion is None:
            continue
        completion_positions, static_relations = completion
        readiness = trigger_use.keys.then(completion_positions)
        readiness_bounds = None if readiness is None else readiness.value_bounds()
        if readiness_bounds is None:
            continue
        if any(not relation.has_total_source() for _root, relation in static_relations):
            continue
        prerequisite_roots = _transitive_static_prerequisite_roots(
            event_graph,
            static_relations,
            remaining_triggers,
        )
        if prerequisite_roots is None:
            continue
        ancestor_placements: set[tuple[int, int]] = set()
        for prerequisite_root in prerequisite_roots:
            last_positions = result.last_positions_for_root(prerequisite_root)
            if last_positions is None:
                ancestor_placements.clear()
                break
            ancestor_placements.update(last_positions.items())
        if not ancestor_placements:
            continue

        original_bounds = original_schedule.position_bounds_for_root(root)
        if original_bounds is None:
            continue
        original_position = original_bounds[0]
        remaining_without_root = tuple(
            trigger for trigger in remaining_triggers if trigger not in root_triggers
        )
        for position in range(readiness_bounds[0] + 1, original_position):
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
                        task_domain=task_domain,
                        task_traversal=event_graph.root_traversals[root],
                        position=position,
                        unavailable_workers=unfinished_workers,
                        pack_high=False,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            result = candidate
            remaining_triggers = remaining_without_root
            break
    return result, remaining_triggers


def resolve_worker_count(
    event_graph: EventGraph,
    *,
    default_worker_count: int,
    requested_worker_count: int,
) -> int:
    """Honor an explicit resident-worker target.

    WorkerSchedule supports families spanning multiple waves, and its graph
    validator is the authority on progress safety.  Rounding a requested
    cohort to an event-key boundary can erase exactly the partial wave that a
    later ready family should occupy.
    """
    if requested_worker_count < 0:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG} must be nonnegative, got "
            f"{requested_worker_count}"
        )
    if requested_worker_count == CROSS_LOOP_NUM_WORKERS_DEFAULT:
        return default_worker_count
    if requested_worker_count > default_worker_count:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG}={requested_worker_count} exceeds "
            f"the configured launch capacity {default_worker_count}"
        )
    return requested_worker_count


def build_worker_schedule(
    event_graph: EventGraph,
    *,
    worker_count: int,
) -> tuple[
    WorkerSchedule,
    tuple[LocalTrigger, ...],
    tuple[CountedEventPlan, ...],
    EventGraph,
]:
    """Derive local and static task placement for one worker count."""
    baseline = build_baseline_worker_schedule(
        event_graph.root_domains,
        event_graph.root_traversals,
        worker_count,
    )
    baseline = place_initial_ready_families(event_graph, baseline)
    nested_wait_roots = frozenset(
        use.consumer_root
        for event in event_graph.events
        for use in event.uses
        if use.consumer_scope_id is not None
    )
    local_triggers = choose_local_triggers(
        event_graph,
        baseline,
        worker_limit=worker_count,
        excluded_roots=nested_wait_roots,
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
        excluded_roots=nested_wait_roots,
    )
    local_roots = frozenset(
        use.consumer_root
        for trigger in local_triggers
        for use in (event_graph.event(trigger.event_index).uses[trigger.use_index],)
    )
    schedule = ordered.without_roots(local_roots)
    schedule, nested_scope_events = place_nested_scope_consumers(
        event_graph,
        schedule,
        local_triggers,
    )
    nested_dependency_points = frozenset(
        dependency_point
        for event in nested_scope_events
        for use in event.uses
        for dependency_point in use.dependency_points
    )
    scheduled_event_graph = _without_root_uses_for_dependencies(
        event_graph, nested_dependency_points
    )
    schedule, local_triggers = place_ready_families(
        scheduled_event_graph,
        ordered,
        schedule,
        local_triggers,
    )
    return schedule, local_triggers, nested_scope_events, scheduled_event_graph


@dataclasses.dataclass(frozen=True)
class LocalTrigger:
    """A task use executed by whichever contributor makes the final arrival."""

    event_index: int
    use_index: int


@dataclasses.dataclass(frozen=True)
class EventContribution:
    """A producer execution scope's symbolic contribution to one event."""

    producer_root: int
    keys: LogicalRelation
    producer_scope_id: int | None = None

    @property
    def expected_arrivals(self) -> int:
        inverse = self.keys.inverse()
        cardinality = None if inverse is None else inverse.fiber_cardinality()
        count = None if cardinality is None else cardinality.constant_value()
        if count is None:
            raise ValueError("symbolic contributor has nonuniform fan-in")
        return count


@dataclasses.dataclass(frozen=True)
class EventUse:
    """A consumer execution scope's symbolic requirements from one event."""

    consumer_root: int
    keys: LogicalRelation
    dependency_points: frozenset[DependencyPoint] = frozenset()
    consumer_scope_id: int | None = None


@dataclasses.dataclass(frozen=True)
class KeyedEvent:
    """One symbolic readiness event shared by scheduling and lowering."""

    event_id: int
    key_domain: LogicalDomain
    contributions: tuple[EventContribution, ...]
    uses: tuple[EventUse, ...]

    def __post_init__(self) -> None:
        if self.key_domain.kind != "event":
            raise ValueError("event key domain must have event kind")
        if self.key_domain.identity != self.event_id:
            raise ValueError("event key domain identity must match its event ID")
        if self.key_domain.axis_order != tuple(range(len(self.key_domain.axis_order))):
            raise ValueError("event key axes must use canonical local ordinals")
        if self.key_domain.block_sizes_items:
            raise ValueError("event key domains must not inherit scope block sizes")
        if any(
            contribution.keys.target_domain != self.key_domain
            for contribution in self.contributions
        ) or any(use.keys.target_domain != self.key_domain for use in self.uses):
            raise ValueError("event relations must target the event key domain")

    @property
    def key_count(self) -> int:
        return self.key_domain.size

    @property
    def family_done_root(self) -> int | None:
        if (
            self.key_count == 1
            and len(self.contributions) == 1
            and self.contributions[0].producer_scope_id is None
            and self.contributions[0].keys.is_total()
        ):
            return self.contributions[0].producer_root
        return None

    @property
    def is_family_done(self) -> bool:
        return self.family_done_root is not None


@dataclasses.dataclass(frozen=True)
class EventGraph:
    """Configured symbolic readiness DAG and its execution-scope domains."""

    root_domains: tuple[LogicalDomain, ...]
    root_traversals: tuple[LogicalRelation, ...]
    scope_domains: tuple[LogicalDomain | None, ...]
    events: tuple[KeyedEvent, ...]

    def __post_init__(self) -> None:
        if len(self.root_domains) != len(self.root_traversals):
            raise ValueError("event graph root domains must match traversals")
        for domain, traversal in zip(
            self.root_domains,
            self.root_traversals,
            strict=True,
        ):
            if (
                traversal.target_domain != domain
                or traversal.source_domain.size != domain.size
                or traversal.source_domain.kind != "worker"
                or not traversal.pieces
            ):
                raise ValueError(
                    "each root traversal must have compatible typed domains"
                )
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("event IDs must be dense and source ordered")

    def event(self, event_id: int) -> KeyedEvent:
        return self.events[event_id]

    def events_contributed_by(self, root: int) -> tuple[KeyedEvent, ...]:
        return tuple(
            event
            for event in self.events
            if any(
                contribution.producer_root == root
                for contribution in event.contributions
            )
        )

    def uses_for_root(self, root: int) -> tuple[EventUse, ...]:
        return tuple(
            use
            for event in self.events
            for use in event.uses
            if use.consumer_root == root
        )

    def scope_domain(self, scope_id: int) -> LogicalDomain:
        domain = self.scope_domains[scope_id]
        if domain is None:
            raise ValueError(f"execution scope {scope_id} has no configured domain")
        return domain

    def nested_axes(self, root: int, scope_id: int) -> tuple[int, ...]:
        root_axes = frozenset(self.root_domains[root].axis_order)
        return tuple(
            axis
            for axis in self.scope_domain(scope_id).axis_order
            if axis not in root_axes
        )

    def source_traversal(self, root: int, scope_id: int | None) -> tuple[int, ...]:
        root_axes = self.root_domains[root].axis_order
        if scope_id is None:
            return root_axes
        return (*self.nested_axes(root, scope_id), *root_axes)

    def required_keys_by_strand(self, use: EventUse) -> LogicalRelation | None:
        """Project a checkpoint's requirements onto its owning root strands.

        Projection failure is a legality failure for strand-level scheduling;
        it must not trigger enumeration of the nested action domain.
        """
        root_domain = self.root_domains[use.consumer_root]
        if use.consumer_scope_id is None:
            if use.keys.source_domain != root_domain:
                raise ValueError("root event use has the wrong source domain")
            return use.keys
        return use.keys.project_source(root_domain)

    def materialized_required_keys_by_strand(
        self,
        use: EventUse,
        *,
        budget: _ClcMaterializationBudget | None = None,
    ) -> tuple[frozenset[int], ...]:
        """Return every event key required anywhere in each root strand.

        A nested checkpoint can depend on its loop coordinate, so existentially
        projecting the symbolic relation onto the root domain is not always
        expressible by ``LogicalRelation``.  The schedule validator only needs
        a conservative root-entry prerequisite: union all nested requirements
        belonging to the same root task. This exact materialized fallback is
        used only by bounded schedule validation and command planning.
        """
        strand_keys = self.required_keys_by_strand(use)
        if strand_keys is not None:
            result: list[frozenset[int]] = []
            for task in range(strand_keys.source_domain.size):
                keys = strand_keys.targets(task)
                if budget is not None:
                    budget.consume(len(keys), "materializing consumer incidences")
                result.append(keys)
            return tuple(result)

        if use.consumer_scope_id is None:
            raise ValueError("root event use cannot be projected onto its strands")
        root_domain = self.root_domains[use.consumer_root]
        scope_domain = self.scope_domain(use.consumer_scope_id)
        source_traversal = self.source_traversal(
            use.consumer_root,
            use.consumer_scope_id,
        )
        result: list[set[int]] = [set() for _ in range(root_domain.size)]
        for source_task in range(use.keys.source_domain.size):
            required_keys = use.keys.targets(
                source_task,
                source_traversal=source_traversal,
            )
            if budget is not None:
                budget.consume(
                    len(required_keys),
                    "materializing nested consumer incidences",
                )
            coordinates = scope_domain.coordinates(
                source_task,
                traversal=source_traversal,
            )
            root_task = root_domain.index(
                {axis: coordinates[axis] for axis in root_domain.axis_order}
            )
            result[root_task].update(required_keys)
        return tuple(frozenset(keys) for keys in result)

    def uniform_expected_arrivals(self, event: KeyedEvent) -> int | None:
        """Return constant fan-in without enumerating event keys."""
        total = 0
        for contribution in event.contributions:
            cardinality = _contribution_fiber_cardinality(contribution.keys)
            count = None if cardinality is None else cardinality.constant_value()
            if count is None:
                return None
            total += count
        return total

    def materialized_contributor_tasks_by_key(
        self,
        event: KeyedEvent,
    ) -> tuple[frozenset[tuple[int, int]], ...]:
        """Enumerate producer strands only for the small-domain validator."""
        result: list[set[tuple[int, int]]] = [set() for _ in range(event.key_count)]
        for contribution in event.contributions:
            strand_keys = contribution.keys.project_source(
                self.root_domains[contribution.producer_root]
            )
            if strand_keys is None:
                raise ValueError(
                    "event contribution cannot be projected onto producer strands"
                )
            inverse = strand_keys.inverse()
            if inverse is not None:
                for key, tasks in enumerate(inverse.materialize()):
                    result[key].update(
                        (contribution.producer_root, task) for task in tasks
                    )
            else:
                for task, keys in enumerate(strand_keys.materialize()):
                    for key in keys:
                        result[key].add((contribution.producer_root, task))
        return tuple(frozenset(tasks) for tasks in result)


@cache
def _contribution_fiber_cardinality(
    keys: LogicalRelation,
) -> LogicalRelation | None:
    inverse = keys.inverse()
    return None if inverse is None else inverse.fiber_cardinality()


@cache
def uniform_preimage_cardinality(keys: LogicalRelation) -> int | None:
    """Return a relation's constant preimage size, with an exact fallback.

    The symbolic inverse is preferable, but some valid mixed-radix maps have a
    compact forward form whose inverse is outside the current relation grammar.
    Enumerating configured source strands is exact and lets those relations use
    the same counted-event lowering instead of falling back to a grid barrier.
    """
    cardinality = _contribution_fiber_cardinality(keys)
    count = None if cardinality is None else cardinality.constant_value()
    if count is not None:
        return count
    if len(keys.source_domain.axis_order) != 1 or len(keys.target_domain.axis_order) <= 1:
        return None
    counts = [0] * keys.target_domain.size
    for targets in keys.materialize():
        for target in targets:
            counts[target] += 1
    if not counts or not counts[0] or any(value != counts[0] for value in counts[1:]):
        return None
    return counts[0]


def _counted_contribution_is_lowerable(contribution: EventContribution) -> bool:
    """Keep scheduler eligibility identical to counted-event code generation."""
    return (
        _contribution_fiber_cardinality(contribution.keys) is not None
        and contribution.keys.canonical_single_valued() is not None
    )


def _local_contribution_is_lowerable(contribution: EventContribution) -> bool:
    """Accept any contribution whose per-key fan-in can be emitted exactly."""
    return (
        (
            uniform_preimage_cardinality(contribution.keys) is not None
            or _contribution_fiber_cardinality(contribution.keys) is not None
        )
        and contribution.keys.canonical_single_valued() is not None
    )


def _canonical_event_domain(domain: LogicalDomain) -> LogicalDomain:
    """Name quotient coordinates locally rather than borrowing scope axes."""
    return LogicalDomain(
        axis_order=tuple(range(len(domain.axis_order))),
        axis_counts_items=tuple(
            (event_axis, count)
            for event_axis, (_scope_axis, count) in enumerate(domain.axis_counts_items)
        ),
        kind="event",
    )


def _canonical_event_relation(
    relation: LogicalRelation,
    key_domain: LogicalDomain,
) -> LogicalRelation:
    """Express one candidate relation in its event-local coordinate chart."""
    old_domain = relation.target_domain
    if (
        old_domain.kind != "event"
        or old_domain.identity is not None
        or tuple(old_domain.axis_counts.values())
        != tuple(key_domain.axis_counts.values())
    ):
        raise AssertionError("event relation does not match its quotient geometry")
    renamed_axes = dict(zip(old_domain.axis_order, key_domain.axis_order, strict=True))
    return LogicalRelation(
        source_domain=relation.source_domain,
        target_domain=key_domain,
        pieces=tuple(
            dataclasses.replace(
                piece,
                target_ranges=tuple(
                    (renamed_axes[axis], begin, end, step)
                    for axis, begin, end, step in piece.target_ranges
                ),
            )
            for piece in relation.pieces
        ),
    )


def _add_event_candidate(
    pending: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ],
    *,
    key_domain: LogicalDomain,
    contributions: tuple[EventContribution, ...],
    uses: tuple[EventUse, ...],
) -> None:
    """Group fanout by the complete producer partition before assigning IDs."""
    canonical_domain = _canonical_event_domain(key_domain)
    canonical_contributions = tuple(
        dataclasses.replace(
            contribution,
            keys=_canonical_event_relation(contribution.keys, canonical_domain),
        )
        for contribution in contributions
    )
    canonical_uses = tuple(
        dataclasses.replace(
            use,
            keys=_canonical_event_relation(use.keys, canonical_domain),
        )
        for use in uses
    )
    grouped_uses = pending.setdefault(
        (canonical_domain, canonical_contributions),
        [],
    )
    for use in canonical_uses:
        matching_index = next(
            (
                index
                for index, previous in enumerate(grouped_uses)
                if previous.consumer_root == use.consumer_root
                and previous.consumer_scope_id == use.consumer_scope_id
                and previous.keys == use.keys
            ),
            None,
        )
        if matching_index is None:
            grouped_uses.append(use)
            continue
        previous = grouped_uses[matching_index]
        grouped_uses[matching_index] = dataclasses.replace(
            previous,
            dependency_points=previous.dependency_points | use.dependency_points,
        )


def _finalize_keyed_events(
    pending: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ],
) -> tuple[KeyedEvent, ...]:
    """Assign deterministic IDs after the readiness quotient is complete."""

    def retarget(relation: LogicalRelation, domain: LogicalDomain) -> LogicalRelation:
        result = relation.retarget(domain)
        if result is None:
            raise AssertionError("event identity assignment changed key geometry")
        return result

    events: list[KeyedEvent] = []
    for event_id, ((key_domain, contributions), uses) in enumerate(pending.items()):
        identified_domain = dataclasses.replace(key_domain, identity=event_id)
        events.append(
            KeyedEvent(
                event_id=event_id,
                key_domain=identified_domain,
                contributions=tuple(
                    dataclasses.replace(
                        contribution,
                        keys=retarget(contribution.keys, identified_domain),
                    )
                    for contribution in contributions
                ),
                uses=tuple(
                    dataclasses.replace(
                        use,
                        keys=retarget(use.keys, identified_domain),
                    )
                    for use in uses
                ),
            )
        )
    return tuple(events)


def _without_root_uses_for_dependencies(
    event_graph: EventGraph,
    dependency_points: frozenset[DependencyPoint],
) -> EventGraph:
    """Remove root-entry alternatives covered by selected action checkpoints."""
    if not dependency_points:
        return event_graph
    events: list[KeyedEvent] = []
    for event in event_graph.events:
        uses: list[EventUse] = []
        for use in event.uses:
            if use.consumer_scope_id is not None:
                uses.append(use)
                continue
            remaining = use.dependency_points - dependency_points
            if remaining:
                uses.append(dataclasses.replace(use, dependency_points=remaining))
        events.append(dataclasses.replace(event, uses=tuple(uses)))
    return dataclasses.replace(event_graph, events=tuple(events))


@dataclasses.dataclass(frozen=True)
class CountedEventPlan:
    """A logical key space receiving contributions from one or more roots.

    Each contributor has an independently proved task-to-key relation. The
    expected count is derived by summing those relations; the event therefore
    represents both ordinary continuations and generic multi-predecessor joins.
    Consumer uses are independent of event identity. ``local_trigger_use``
    identifies the optional use executed by the final arriving contributor.
    """

    contributors: tuple[EventContribution, ...]
    uses: tuple[EventUse, ...]
    key_domain: LogicalDomain
    local_trigger_use: int | None = None
    graph_event_index: int | None = None

    @property
    def local_use(self) -> EventUse | None:
        if self.local_trigger_use is None:
            return None
        return self.uses[self.local_trigger_use]

    @property
    def key_count(self) -> int:
        """Return the complete event-key domain used by producers or consumers."""
        return self.key_domain.size

    @property
    def expected_arrivals(self) -> int:
        return sum(contributor.expected_arrivals for contributor in self.contributors)

    @property
    def is_single_contributor(self) -> bool:
        return len(self.contributors) == 1

    @property
    def single_contributor(self) -> EventContribution:
        if not self.is_single_contributor:
            raise ValueError("keyed event has multiple contributors")
        return self.contributors[0]

    @property
    def producer_root(self) -> int:
        return self.single_contributor.producer_root


@dataclasses.dataclass(frozen=True)
class ReadinessPlan:
    """Dispatch-independent synchronization selected from one dependency graph."""

    event_graph: EventGraph
    counted_events: tuple[CountedEventPlan, ...]

    @property
    def root_completion_edges(self) -> frozenset[tuple[int, int]]:
        """Return family-completion relations represented by one-key events."""
        return frozenset(
            (family_done_root, use.consumer_root)
            for plan in self.counted_events
            if plan.graph_event_index is not None
            for event in (self.event_graph.event(plan.graph_event_index),)
            if (family_done_root := event.family_done_root) is not None
            for use in plan.uses
        )


@dataclasses.dataclass(frozen=True)
class _ScopeReadiness:
    """Configured readiness of one nested scope on its owning task strands."""

    event: KeyedEvent
    use: EventUse
    domain: LogicalDomain
    readiness: LogicalRelation
    ancestor_placements: frozenset[tuple[int, int]]


def _merge_relations_by_root(
    relations: tuple[tuple[int, LogicalRelation], ...],
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    merged: dict[int, LogicalRelation] = {}
    for root, relation in relations:
        previous = merged.get(root)
        if previous is None:
            merged[root] = relation
            continue
        union = previous.union(relation)
        if union is None:
            return None
        merged[root] = union
    return tuple(sorted(merged.items()))


def _static_contribution_relations(
    event_graph: EventGraph,
    *,
    root: int,
    scope_id: int | None,
    keys: LogicalRelation,
    local_trigger_by_root: dict[int, LocalTrigger],
    visiting: frozenset[int] = frozenset(),
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    """Contract local execution to relations from statically scheduled roots."""
    root_domain = event_graph.root_domains[root]
    root_keys = keys if scope_id is None else keys.project_source(root_domain)
    if root_keys is None:
        return None
    trigger = local_trigger_by_root.get(root)
    if trigger is None:
        return ((root, root_keys),)
    if root in visiting:
        return None
    trigger_event = event_graph.event(trigger.event_index)
    trigger_use = trigger_event.uses[trigger.use_index]
    inverse_use = trigger_use.keys.inverse()
    key_to_target = None if inverse_use is None else inverse_use.then(root_keys)
    if key_to_target is None:
        return None
    expanded: list[tuple[int, LogicalRelation]] = []
    for contribution in trigger_event.contributions:
        upstream_keys = contribution.keys.then(key_to_target)
        if upstream_keys is None:
            return None
        upstream = _static_contribution_relations(
            event_graph,
            root=contribution.producer_root,
            scope_id=contribution.producer_scope_id,
            keys=upstream_keys,
            local_trigger_by_root=local_trigger_by_root,
            visiting=visiting | frozenset((root,)),
        )
        if upstream is None:
            return None
        expanded.extend(upstream)
    return _merge_relations_by_root(tuple(expanded))


def _event_static_contributions(
    event_graph: EventGraph,
    event: KeyedEvent,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    local_trigger_by_root = {
        event_graph.event(trigger.event_index)
        .uses[trigger.use_index]
        .consumer_root: trigger
        for trigger in local_triggers
    }
    expanded: list[tuple[int, LogicalRelation]] = []
    for contribution in event.contributions:
        static_relations = _static_contribution_relations(
            event_graph,
            root=contribution.producer_root,
            scope_id=contribution.producer_scope_id,
            keys=contribution.keys,
            local_trigger_by_root=local_trigger_by_root,
        )
        if static_relations is None:
            return None
        expanded.extend(static_relations)
    return _merge_relations_by_root(tuple(expanded))


def _transitive_static_prerequisite_roots(
    event_graph: EventGraph,
    static_relations: tuple[tuple[int, LogicalRelation], ...],
    local_triggers: tuple[LocalTrigger, ...],
) -> frozenset[int] | None:
    """Close static contributors through waits earlier on their task strands."""
    roots = {root for root, _relation in static_relations}
    pending = list(roots)
    while pending:
        consumer_root = pending.pop()
        for event in event_graph.events:
            if not any(use.consumer_root == consumer_root for use in event.uses):
                continue
            upstream = _event_static_contributions(
                event_graph,
                event,
                local_triggers,
            )
            if upstream is None:
                return None
            for producer_root, _relation in upstream:
                if producer_root == consumer_root:
                    return None
                if producer_root not in roots:
                    roots.add(producer_root)
                    pending.append(producer_root)
    return frozenset(roots)


def _event_completion_positions(
    event_graph: EventGraph,
    event: KeyedEvent,
    *,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[LogicalRelation, tuple[tuple[int, LogicalRelation], ...]] | None:
    """Return event-key frontiers and their ultimate static contributors."""
    static_relations = _event_static_contributions(
        event_graph,
        event,
        local_triggers,
    )
    if static_relations is None:
        return None
    maxima: list[LogicalRelation] = []
    for root, keys in static_relations:
        root_domain = event_graph.root_domains[root]
        if keys.source_domain != root_domain:
            return None
        for segment in worker_schedule.segments_for_root(root):
            task_relation = segment.task_relation
            if task_relation is None or task_relation.target_domain != root_domain:
                return None
            ordinal_keys = task_relation.then(keys)
            inverse = None if ordinal_keys is None else ordinal_keys.inverse()
            positions = worker_schedule.position_relation(segment)
            maximum = (
                None
                if inverse is None or positions is None
                else inverse.fiber_maximum(positions)
            )
            if maximum is None:
                return None
            maxima.append(maximum)
    if not maxima:
        return None
    combined = maxima[0]
    for relation in maxima[1:]:
        union = combined.union(relation)
        if union is None:
            return None
        combined = union
    identity = LogicalRelation.identity(
        worker_schedule.position_domain,
        worker_schedule.position_domain,
    )
    maximum = combined.fiber_maximum(identity)
    return None if maximum is None else (maximum, static_relations)


def _scope_readiness(
    event_graph: EventGraph,
    event: KeyedEvent,
    use: EventUse,
    *,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> _ScopeReadiness | None:
    """Project one exact action relation onto static worker completion positions."""
    assert use.consumer_scope_id is not None
    domain = event_graph.scope_domain(use.consumer_scope_id)
    nested_axes = event_graph.nested_axes(
        use.consumer_root,
        use.consumer_scope_id,
    )
    if len(nested_axes) != 1 or use.keys.source_domain != domain:
        return None

    completion = _event_completion_positions(
        event_graph,
        event,
        worker_schedule=worker_schedule,
        local_triggers=local_triggers,
    )
    if completion is None:
        return None
    completion_positions, static_relations = completion
    action_readiness = use.keys.then(completion_positions)
    if action_readiness is None or not action_readiness.is_total_function():
        return None
    if any(not relation.has_total_source() for _root, relation in static_relations):
        return None
    prerequisite_roots = _transitive_static_prerequisite_roots(
        event_graph,
        static_relations,
        local_triggers,
    )
    if prerequisite_roots is None:
        return None
    ancestor_placements: set[tuple[int, int]] = set()
    for root in prerequisite_roots:
        last_positions = worker_schedule.last_positions_for_root(root)
        if last_positions is None:
            return None
        ancestor_placements.update(last_positions.items())
    return _ScopeReadiness(
        event=event,
        use=use,
        domain=domain,
        readiness=action_readiness,
        ancestor_placements=frozenset(ancestor_placements),
    )


def _segmented_scope_event(
    event_graph: EventGraph,
    event: KeyedEvent,
    use: EventUse,
    boundaries: tuple[int, ...],
) -> CountedEventPlan | None:
    """Coarsen one exact nested dependency into contiguous action segments."""
    consumer_scope_id = use.consumer_scope_id
    assert consumer_scope_id is not None
    domain = event_graph.scope_domain(consumer_scope_id)
    nested_axes = event_graph.nested_axes(
        use.consumer_root,
        consumer_scope_id,
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    segments = tuple(itertools.pairwise(boundaries))
    if not segments or any(begin >= end for begin, end in segments):
        return None
    used_axes = use.keys.source_axes_used()
    if used_axes is None or nested_axis not in used_axes:
        return None
    reduced_domain = LogicalDomain(
        axis_order=used_axes,
        axis_counts_items=tuple((axis, domain.axis_counts[axis]) for axis in used_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in used_axes
            if axis in domain.block_sizes
        ),
        kind="scope",
        identity=domain.identity,
    )
    outer_axes = tuple(axis for axis in used_axes if axis != nested_axis)
    key_domain = LogicalDomain(
        axis_order=tuple(range(len(outer_axes) + 1)),
        axis_counts_items=(
            (0, len(segments)),
            *(
                (event_axis, reduced_domain.axis_counts[source_axis])
                for event_axis, source_axis in enumerate(outer_axes, start=1)
            ),
        ),
        kind="event",
    )
    stage_keys = LogicalRelation.point_map(
        reduced_domain,
        key_domain,
        tuple(
            (
                tuple(
                    (
                        (axis, segment_begin, segment_end, 1)
                        if axis == nested_axis
                        else (axis, 0, reduced_domain.axis_counts[axis], 1)
                    )
                    for axis in reduced_domain.axis_order
                ),
                (
                    sympy.Integer(stage),
                    *(logical_axis_symbol(axis) for axis in outer_axes),
                ),
            )
            for stage, (segment_begin, segment_end) in enumerate(segments)
        ),
    )
    inverse_use = use.keys.inverse()
    reduced_inverse = (
        None if inverse_use is None else inverse_use.project_target(reduced_domain)
    )
    coarsening = None if reduced_inverse is None else reduced_inverse.then(stage_keys)
    if coarsening is None:
        return None
    contribution_relations = tuple(
        contribution.keys.then(coarsening) for contribution in event.contributions
    )
    if any(relation is None for relation in contribution_relations):
        return None
    action_keys = stage_keys.lift_source(domain)
    if action_keys is None or action_keys.canonical_single_valued() is None:
        return None

    contributors = tuple(
        EventContribution(
            producer_root=contribution.producer_root,
            producer_scope_id=contribution.producer_scope_id,
            keys=relation,
        )
        for contribution, relation in zip(
            event.contributions,
            contribution_relations,
            strict=True,
        )
        if relation is not None
    )
    if any(
        not _counted_contribution_is_lowerable(contributor)
        for contributor in contributors
    ):
        # Coarsening a broadcast use can make one producer task feed segments
        # on both sides of the frontier. Counted-event codegen emits one key
        # per task, so retain the exact dependency and let coverage select the
        # conservative root-completion fallback instead.
        return None

    return CountedEventPlan(
        contributors=contributors,
        uses=(
            EventUse(
                consumer_root=use.consumer_root,
                dependency_points=use.dependency_points,
                consumer_scope_id=use.consumer_scope_id,
                keys=action_keys,
            ),
        ),
        graph_event_index=event.event_id,
        key_domain=key_domain,
    )


def _scope_milestones(
    event_graph: EventGraph,
    readiness: _ScopeReadiness,
    *,
    consumer_position: int,
) -> CountedEventPlan | None:
    """Split a nested scope loop at the selected schedule frontier."""
    domain = readiness.domain
    root_domain = event_graph.root_domains[readiness.use.consumer_root]
    actions_per_strand = domain.size // root_domain.size
    consumer_scope_id = readiness.use.consumer_scope_id
    assert consumer_scope_id is not None
    nested_axes = event_graph.nested_axes(
        readiness.use.consumer_root,
        consumer_scope_id,
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes

    def ready(action_offset: int) -> bool | None:
        value_bounds = readiness.readiness.value_bounds({nested_axis: action_offset})
        if value_bounds is None:
            return None
        # Producers at the same stream position execute concurrently on other
        # workers. They permit placement at this position, but are not ready at
        # admission: their consumer actions belong after the single frontier.
        # ``ancestor_placements`` separately prevents self-deadlock on a
        # producer's own worker.
        return value_bounds[1] < consumer_position

    first_ready = ready(0)
    last_ready = ready(actions_per_strand - 1)
    if first_ready is None or last_ready is None:
        return None
    if not first_ready:
        frontier = 0
    elif last_ready:
        frontier = actions_per_strand
    else:
        lower = 0
        upper = actions_per_strand - 1
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            midpoint_ready = ready(midpoint)
            if midpoint_ready is None:
                return None
            if midpoint_ready:
                lower = midpoint
            else:
                upper = midpoint
        frontier = upper
    boundaries = tuple(sorted({0, frontier, actions_per_strand}))
    return _segmented_scope_event(
        event_graph,
        readiness.event,
        readiness.use,
        boundaries,
    )


def place_nested_scope_consumers(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[WorkerSchedule, tuple[CountedEventPlan, ...]]:
    """Place strands with nested waits and derive their milestone events.

    Exact action dependencies remain the semantic source of truth. This pass
    uses only worker positions and same-strand action order to select one
    admission frontier for the original nested loop.
    It does not inspect operation kinds or recognize a graph topology.
    """
    uses_by_consumer: dict[
        int,
        list[tuple[KeyedEvent, EventUse]],
    ] = {}
    for event in event_graph.events:
        for use in event.uses:
            if use.consumer_scope_id is not None:
                uses_by_consumer.setdefault(use.consumer_root, []).append((event, use))

    result = worker_schedule
    plans: list[CountedEventPlan] = []
    for consumer_root, event_uses in sorted(uses_by_consumer.items()):
        task_domain = event_graph.root_domains[consumer_root]
        if task_domain.size > result.worker_count:
            continue

        # A preceding scope may already carry every dependency point needed by
        # a later scope.  The implication was proved from DeviceIR program
        # order when the event graph was built, so the later wait is redundant.
        uncovered_event_uses: list[tuple[KeyedEvent, EventUse]] = []
        preceding_dependency_points: set[DependencyPoint] = set()
        for event, use in sorted(
            event_uses,
            key=lambda item: (
                item[1].consumer_scope_id
                if item[1].consumer_scope_id is not None
                else -1,
                item[0].event_id,
            ),
        ):
            if use.dependency_points and use.dependency_points <= (
                preceding_dependency_points
            ):
                continue
            uncovered_event_uses.append((event, use))
            preceding_dependency_points.update(use.dependency_points)

        readiness = tuple(
            _scope_readiness(
                event_graph,
                event,
                use,
                worker_schedule=result,
                local_triggers=local_triggers,
            )
            for event, use in uncovered_event_uses
        )
        if not readiness or any(item is None for item in readiness):
            continue
        ordered_readiness = tuple(item for item in readiness if item is not None)

        current_consumer_bounds = result.position_bounds_for_root(consumer_root)
        if current_consumer_bounds is None:
            continue
        original_position = current_consumer_bounds[0]
        readiness_bounds = tuple(
            item.readiness.value_bounds() for item in ordered_readiness
        )
        if any(bounds is None for bounds in readiness_bounds):
            continue
        earliest_readiness = min(
            bounds[0] for bounds in readiness_bounds if bounds is not None
        )
        ancestor_placements = frozenset(
            placement
            for item in ordered_readiness
            for placement in item.ancestor_placements
        )
        chosen: tuple[WorkerSchedule, tuple[CountedEventPlan, ...]] | None = None
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
                        task_domain=task_domain,
                        task_traversal=event_graph.root_traversals[consumer_root],
                        position=position,
                        unavailable_workers=busy_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            milestone_results = tuple(
                _scope_milestones(
                    event_graph,
                    item,
                    consumer_position=position,
                )
                for item in ordered_readiness
            )
            if any(item is None for item in milestone_results):
                continue
            chosen = (
                candidate,
                tuple(plan for plan in milestone_results if plan is not None),
            )
            break
        if chosen is None:
            continue
        result, scope_plans = chosen
        plans.extend(scope_plans)
    return result, tuple(plans)


@dataclasses.dataclass(frozen=True)
class CrossLoopSchedule:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    readiness: ReadinessPlan
    worker_schedule: WorkerSchedule
    local_triggers: tuple[LocalTrigger, ...]
    worker_limit: int

    @property
    def event_graph(self) -> EventGraph:
        return self.readiness.event_graph

    @property
    def counted_events(self) -> tuple[CountedEventPlan, ...]:
        return self.readiness.counted_events

    @property
    def root_completion_edges(self) -> frozenset[tuple[int, int]]:
        return self.readiness.root_completion_edges


@dataclasses.dataclass(frozen=True)
class ClcCommandRange:
    """One root's physical task ordinals in the flattened command table."""

    root: int
    begin: int
    end: int

    @property
    def task_count(self) -> int:
        return self.end - self.begin


@dataclasses.dataclass(frozen=True)
class ClcCommand:
    """One logical task in the dependency-topological CLC stream."""

    worker: int
    position_begin: int
    position_end: int
    task: tuple[int, int]

    def __post_init__(self) -> None:
        if self.worker < 0:
            raise ValueError(f"worker must be nonnegative, got {self.worker}")
        if self.position_begin < 0 or self.position_end <= self.position_begin:
            raise ValueError("invalid CLC command position interval")


@dataclasses.dataclass(frozen=True)
class ClcCommandPlan:
    """A dependency-topological stream of logical worker commands."""

    base_schedule: CrossLoopSchedule
    command_ranges: tuple[ClcCommandRange, ...]
    commands: tuple[ClcCommand, ...]
    task_order: tuple[int, ...]

    @property
    def event_graph(self) -> EventGraph:
        return self.base_schedule.event_graph

    @property
    def readiness(self) -> ReadinessPlan:
        return self.base_schedule.readiness

    @property
    def counted_events(self) -> tuple[CountedEventPlan, ...]:
        return self.base_schedule.counted_events

    @property
    def root_completion_edges(self) -> frozenset[tuple[int, int]]:
        return self.base_schedule.root_completion_edges

    @property
    def launch_token_count(self) -> int:
        return max(1, self.command_count)

    @property
    def command_count(self) -> int:
        return len(self.commands)

    @property
    def uses_cancellation(self) -> bool:
        return self.command_count > 1

    def traversal_for_root(self, root: int) -> LogicalRelation:
        return self.event_graph.root_traversals[root]


def lower_family_done_events(
    event_graph: EventGraph,
    dependency_graph: TileDependencyGraph,
    edges: frozenset[tuple[int, int]],
) -> tuple[EventGraph, tuple[CountedEventPlan, ...]]:
    """Represent selected whole-family waits as canonical one-key events.

    The semantic event receives one contribution per logical task. Codegen is
    free to aggregate those arrivals by worker after proving the worker's
    complete task stream has finished.
    """
    consumers_by_producer: dict[int, set[int]] = {}
    for producer_root, consumer_root in edges:
        consumers_by_producer.setdefault(producer_root, set()).add(consumer_root)

    events = [
        dataclasses.replace(event, uses=())
        if event.family_done_root is not None
        else event
        for event in event_graph.events
    ]
    family_event_by_root = {
        family_done_root: event
        for event in events
        if (family_done_root := event.family_done_root) is not None
    }
    plans: list[CountedEventPlan] = []
    for producer_root, consumer_roots in sorted(consumers_by_producer.items()):
        selected_uses = tuple(
            EventUse(
                consumer_root=consumer_root,
                consumer_scope_id=None,
                keys=LogicalRelation.total(
                    event_graph.root_domains[consumer_root],
                    LogicalDomain(
                        axis_order=(),
                        axis_counts_items=(),
                        kind="event",
                        identity=(
                            family_event_by_root[producer_root].event_id
                            if producer_root in family_event_by_root
                            else len(events)
                        ),
                    ),
                ),
                dependency_points=frozenset(
                    dependency_point
                    for dependency in dependency_graph.edges_between(
                        producer_root, consumer_root
                    )
                    for access_dependency in dependency.access_dependencies
                    for dependency_point in dependency_graph.dependency_points(
                        access_dependency
                    )
                ),
            )
            for consumer_root in sorted(consumer_roots)
        )
        family_event = family_event_by_root.get(producer_root)
        if family_event is None:
            key_domain = selected_uses[0].keys.target_domain
            family_event = KeyedEvent(
                event_id=len(events),
                key_domain=key_domain,
                contributions=(
                    EventContribution(
                        producer_root=producer_root,
                        producer_scope_id=None,
                        keys=LogicalRelation.total(
                            event_graph.root_domains[producer_root],
                            key_domain,
                        ),
                    ),
                ),
                uses=selected_uses,
            )
            events.append(family_event)
            family_event_by_root[producer_root] = family_event
        else:
            family_event = dataclasses.replace(family_event, uses=selected_uses)
            events[family_event.event_id] = family_event
        plans.append(
            CountedEventPlan(
                contributors=(
                    EventContribution(
                        producer_root=producer_root,
                        keys=family_event.contributions[0].keys,
                    ),
                ),
                uses=tuple(
                    EventUse(
                        consumer_root=use.consumer_root,
                        dependency_points=use.dependency_points,
                        keys=use.keys,
                    )
                    for use in selected_uses
                ),
                graph_event_index=family_event.event_id,
                key_domain=family_event.key_domain,
            )
        )
    return dataclasses.replace(event_graph, events=tuple(events)), tuple(plans)


def choose_local_triggers(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule | None,
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
        family_done_root
        for event in event_graph.events
        if (family_done_root := event.family_done_root) is not None and event.uses
    }
    return tuple(
        trigger
        for trigger in derive_local_triggers(event_graph, worker_schedule)
        if (
            use := event_graph.event(trigger.event_index).uses[trigger.use_index]
        ).consumer_root
        not in excluded_roots
        and event_graph.root_domains[use.consumer_root].size > 1
        and not (
            use.consumer_root in family_done_roots
            and event_graph.root_domains[use.consumer_root].size > worker_limit
        )
    )


def choose_counted_events(
    event_graph: EventGraph,
    local_triggers: tuple[LocalTrigger, ...],
    *,
    excluded_dependency_points: frozenset[DependencyPoint] = frozenset(),
) -> tuple[CountedEventPlan, ...]:
    """Select root-entry events representable by the counted-event emitter.

    Nested consumers keep their program-point lowering. Excluding one use does
    not discard independent uses of the same semantic event. A one-key event is
    whole-family completion and remains on the aggregated root-completion path
    unless it owns a local trigger. Unsupported relations monotonically fall
    back to root completion during coverage selection.
    """
    local_uses = {
        (trigger.event_index, trigger.use_index) for trigger in local_triggers
    }
    selected: list[CountedEventPlan] = []
    for event in event_graph.events:
        has_local_use = any(
            (event.event_id, use_index) in local_uses
            for use_index in range(len(event.uses))
        )
        contributions_are_lowerable = all(
            _counted_contribution_is_lowerable(contribution)
            for contribution in event.contributions
        ) or (
            has_local_use
            and all(
                _local_contribution_is_lowerable(contribution)
                for contribution in event.contributions
            )
        )
        if (
            event.family_done_root is not None
            or not event.key_count
            or not contributions_are_lowerable
        ):
            continue
        retained_uses: list[EventUse] = []
        selected_local_uses: list[int] = []
        for use_index, use in enumerate(event.uses):
            if (
                use.consumer_scope_id is not None
                or use.keys.canonical_single_valued() is None
            ):
                continue
            remaining = use.dependency_points - excluded_dependency_points
            is_local = (event.event_id, use_index) in local_uses
            if not is_local and not remaining:
                continue
            if is_local:
                selected_local_uses.append(len(retained_uses))
            retained_uses.append(
                EventUse(
                    consumer_root=use.consumer_root,
                    keys=use.keys,
                    dependency_points=(
                        use.dependency_points if is_local else frozenset(remaining)
                    ),
                )
            )
        if len(selected_local_uses) > 1:
            raise ValueError("one counted event cannot have multiple local executors")
        local_trigger_use = selected_local_uses[0] if selected_local_uses else None
        if not retained_uses or (local_trigger_use is None and event.key_count <= 1):
            continue
        selected.append(
            CountedEventPlan(
                contributors=event.contributions,
                uses=tuple(retained_uses),
                local_trigger_use=local_trigger_use,
                graph_event_index=event.event_id,
                key_domain=event.key_domain,
            )
        )
    selected_local_count = sum(
        event.local_trigger_use is not None for event in selected
    )
    if selected_local_count != len(local_uses):
        raise AssertionError("not every selected local trigger has a lowering event")
    return tuple(selected)


def build_keyed_events(
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
    publishable_scope_ids: frozenset[int] | None = None,
) -> tuple[KeyedEvent, ...] | None:
    """Build the canonical symbolic event graph from memory dependencies.

    This is the sole event-construction path. It never constructs a per-task
    predecessor set. Unsupported relations coarsen to one family-completion
    event for the affected root pair.
    """
    root_domains = instantiate_root_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    if any(domain is None for domain in root_domains):
        return None
    concrete_root_domains = tuple(
        domain for domain in root_domains if domain is not None
    )
    symbolic_dependencies = instantiate_symbolic_dependencies(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    exact_dependencies = tuple(
        dependency
        for dependency in symbolic_dependencies
        if dependency.relation is not None
        and dependency.relation.source_axes_used() is not None
    )
    all_dependency_points_by_pair: dict[tuple[int, int], set[DependencyPoint]] = {}
    for edge in dependency_graph.edges:
        pair = (edge.producer_root, edge.consumer_root)
        for access_dependency in edge.access_dependencies:
            all_dependency_points_by_pair.setdefault(pair, set()).update(
                dependency_graph.dependency_points(access_dependency)
            )

    scope_domains = instantiate_scope_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    implied_points: dict[DependencyPoint, set[DependencyPoint]] = {}
    for source in exact_dependencies:
        source_scope_id = source.consumer_scope_id
        if source_scope_id is None or scope_by_id[source_scope_id].is_root:
            continue
        source_relation = source.relation
        assert source_relation is not None
        source_point = (
            source.dependency_id,
            source.producer_scope_id,
            source_scope_id,
        )
        for later in exact_dependencies:
            later_scope_id = later.consumer_scope_id
            later_relation = later.relation
            if (
                later is source
                or later_scope_id is None
                or later_relation is None
                or source.consumer_root != later.consumer_root
                or source.producer_root != later.producer_root
                or source.producer_scope_id != later.producer_scope_id
                or source_relation.target_domain != later_relation.target_domain
            ):
                continue
            preceding = preceding_scope_relation(
                dependency_graph,
                scope_domains=scope_domains,
                source_scope_id=source_scope_id,
                consumer_scope_id=later_scope_id,
                consumer_access_id=later.consumer_access_id,
            )
            acquired = None if preceding is None else preceding.then(source_relation)
            if acquired is not None and acquired.covers(later_relation):
                implied_points.setdefault(source_point, set()).add(
                    (
                        later.dependency_id,
                        later.producer_scope_id,
                        later_scope_id,
                    )
                )

    exact_relations: dict[
        tuple[int, int | None, LogicalDomain],
        dict[
            tuple[int, int | None, LogicalDomain],
            list[tuple[LogicalRelation, DependencyPoint]],
        ],
    ] = {}
    represented_dependency_points_by_pair: dict[
        tuple[int, int], set[DependencyPoint]
    ] = {}

    def add_exact_relation(
        *,
        producer_root: int,
        producer_scope_id: int | None,
        consumer_root: int,
        consumer_scope_id: int | None,
        relation: LogicalRelation,
        dependency_points: frozenset[DependencyPoint],
    ) -> None:
        consumer = (consumer_root, consumer_scope_id, relation.source_domain)
        producer = (producer_root, producer_scope_id, relation.target_domain)
        exact_relations.setdefault(consumer, {}).setdefault(producer, []).extend(
            (relation, dependency_point) for dependency_point in dependency_points
        )
        represented_dependency_points_by_pair.setdefault(
            (producer_root, consumer_root), set()
        ).update(dependency_points)

    for dependency in exact_dependencies:
        relation = dependency.relation
        assert relation is not None
        dependency_point = (
            dependency.dependency_id,
            dependency.producer_scope_id,
            dependency.consumer_scope_id,
        )
        exact_points = frozenset(
            (dependency_point, *implied_points.get(dependency_point, ()))
        )
        producer_scope = (
            None
            if dependency.producer_scope_id is None
            else scope_by_id[dependency.producer_scope_id]
        )
        consumer_scope = (
            None
            if dependency.consumer_scope_id is None
            else scope_by_id[dependency.consumer_scope_id]
        )
        producer_is_root = producer_scope is None or producer_scope.is_root
        consumer_is_root = consumer_scope is None or consumer_scope.is_root
        producer_scope_is_usable = producer_is_root or (
            producer_scope is not None
            and producer_scope.segmentable
            and (
                publishable_scope_ids is None
                or dependency.producer_scope_id in publishable_scope_ids
            )
        )
        consumer_scope_is_usable = consumer_is_root or (
            consumer_scope is not None and consumer_scope.segmentable
        )
        if producer_scope_is_usable and consumer_scope_is_usable:
            add_exact_relation(
                producer_root=dependency.producer_root,
                producer_scope_id=(
                    None if producer_is_root else dependency.producer_scope_id
                ),
                consumer_root=dependency.consumer_root,
                consumer_scope_id=(
                    None if consumer_is_root else dependency.consumer_scope_id
                ),
                relation=relation,
                dependency_points=exact_points,
            )

        if producer_is_root and consumer_is_root:
            continue
        root_relation = relation
        if not consumer_is_root:
            projected = root_relation.project_source(
                concrete_root_domains[dependency.consumer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        if not producer_is_root:
            projected = root_relation.project_target(
                concrete_root_domains[dependency.producer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        add_exact_relation(
            producer_root=dependency.producer_root,
            producer_scope_id=None,
            consumer_root=dependency.consumer_root,
            consumer_scope_id=None,
            relation=root_relation,
            dependency_points=exact_points,
        )

    pending_events: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ] = {}

    def add_producer_keyed_events(
        *,
        consumer_root: int,
        consumer_scope_id: int | None,
        relations: list[
            tuple[
                tuple[int, int | None, LogicalDomain],
                LogicalRelation,
                frozenset[DependencyPoint],
            ]
        ],
    ) -> None:
        """Keep a finer exact key when a consumer quotient needs fanout."""
        for producer, relation, dependency_points in relations:
            producer_root, producer_scope_id, producer_domain = producer
            key_domain = dataclasses.replace(
                producer_domain,
                kind="event",
                identity=None,
            )
            use_relation = relation.retarget(key_domain)
            if use_relation is None:
                raise AssertionError("producer-key event geometry must match")
            _add_event_candidate(
                pending_events,
                key_domain=key_domain,
                contributions=(
                    EventContribution(
                        producer_root=producer_root,
                        producer_scope_id=producer_scope_id,
                        keys=LogicalRelation.identity(
                            producer_domain,
                            key_domain,
                        ),
                    ),
                ),
                uses=(
                    EventUse(
                        consumer_root=consumer_root,
                        consumer_scope_id=consumer_scope_id,
                        keys=use_relation,
                        dependency_points=dependency_points,
                    ),
                ),
            )

    for consumer, producers in sorted(
        exact_relations.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
        ),
    ):
        consumer_root, consumer_scope_id, consumer_domain = consumer
        merged_relations: list[
            tuple[
                tuple[int, int | None, LogicalDomain],
                LogicalRelation,
                frozenset[DependencyPoint],
            ]
        ] = []
        key_axes: set[int] = set()
        for producer, relation_points in sorted(
            producers.items(),
            key=lambda item: (
                item[0][0],
                -1 if item[0][1] is None else item[0][1],
            ),
        ):
            relation, first_point = relation_points[0]
            dependency_points = {first_point}
            for next_relation, dependency_point in relation_points[1:]:
                union = relation.union(next_relation)
                if union is None:
                    return None
                relation = union
                dependency_points.add(dependency_point)
            used_axes = relation.source_axes_used()
            if used_axes is None:
                return None
            key_axes.update(used_axes)
            merged_relations.append((producer, relation, frozenset(dependency_points)))

        inverses = tuple(
            relation.inverse() for _producer, relation, _ in merged_relations
        )
        if any(inverse is None for inverse in inverses):
            add_producer_keyed_events(
                consumer_root=consumer_root,
                consumer_scope_id=consumer_scope_id,
                relations=merged_relations,
            )
            continue

        ordered_key_axes = tuple(
            axis for axis in consumer_domain.axis_order if axis in key_axes
        )
        consumer_counts = consumer_domain.axis_counts
        consumer_blocks = consumer_domain.block_sizes
        key_domain = LogicalDomain(
            axis_order=ordered_key_axes,
            axis_counts_items=tuple(
                (axis, consumer_counts[axis]) for axis in ordered_key_axes
            ),
            block_sizes_items=tuple(
                (axis, consumer_blocks[axis])
                for axis in ordered_key_axes
                if axis in consumer_blocks
            ),
            kind="event",
        )
        use_relation = LogicalRelation.projection(consumer_domain, key_domain)
        if use_relation is None:
            return None
        contributions: list[EventContribution] = []
        dependency_points: set[DependencyPoint] = set()
        for (producer, _relation, relation_points), inverse in zip(
            merged_relations,
            inverses,
            strict=True,
        ):
            producer_root, producer_scope_id, producer_domain = producer
            contribution_relation = (
                None if inverse is None else inverse.project_target(key_domain)
            )
            if contribution_relation is None:
                return None
            contributions.append(
                EventContribution(
                    producer_root=producer_root,
                    producer_scope_id=producer_scope_id,
                    keys=contribution_relation,
                )
            )
            dependency_points.update(relation_points)
        if any(
            not contribution.keys.is_single_valued() for contribution in contributions
        ):
            add_producer_keyed_events(
                consumer_root=consumer_root,
                consumer_scope_id=consumer_scope_id,
                relations=merged_relations,
            )
            continue
        _add_event_candidate(
            pending_events,
            key_domain=key_domain,
            contributions=tuple(contributions),
            uses=(
                EventUse(
                    consumer_root=consumer_root,
                    consumer_scope_id=consumer_scope_id,
                    keys=use_relation,
                    dependency_points=frozenset(dependency_points),
                ),
            ),
        )

    failed_consumers_by_producer: dict[int, dict[int, set[DependencyPoint]]] = {}
    for (
        producer_root,
        consumer_root,
    ), dependency_points in all_dependency_points_by_pair.items():
        remaining_points = (
            dependency_points
            - represented_dependency_points_by_pair.get(
                (producer_root, consumer_root), set()
            )
        )
        if not remaining_points:
            continue
        failed_consumers_by_producer.setdefault(producer_root, {})[consumer_root] = (
            remaining_points
        )
    for producer_root, consumer_points in sorted(failed_consumers_by_producer.items()):
        key_domain = LogicalDomain(
            axis_order=(),
            axis_counts_items=(),
            kind="event",
        )
        producer_domain = concrete_root_domains[producer_root]
        uses: list[EventUse] = []
        for consumer_root, dependency_points in sorted(consumer_points.items()):
            uses.append(
                EventUse(
                    consumer_root=consumer_root,
                    consumer_scope_id=None,
                    keys=LogicalRelation.total(
                        concrete_root_domains[consumer_root],
                        key_domain,
                    ),
                    dependency_points=frozenset(dependency_points),
                )
            )
        _add_event_candidate(
            pending_events,
            key_domain=key_domain,
            contributions=(
                EventContribution(
                    producer_root=producer_root,
                    producer_scope_id=None,
                    keys=LogicalRelation.total(
                        producer_domain,
                        key_domain,
                    ),
                ),
            ),
            uses=tuple(uses),
        )
    return _finalize_keyed_events(pending_events)


def _root_completion_events(
    dependency_graph: TileDependencyGraph,
    root_domains: tuple[LogicalDomain, ...],
) -> tuple[KeyedEvent, ...]:
    """Conservatively represent every cross-root hazard by family completion."""
    consumers_by_producer: dict[int, dict[int, set[DependencyPoint]]] = {}
    for dependency in dependency_graph.edges:
        points = {
            dependency_point
            for access_dependency in dependency.access_dependencies
            for dependency_point in dependency_graph.dependency_points(
                access_dependency
            )
        }
        consumers_by_producer.setdefault(dependency.producer_root, {}).setdefault(
            dependency.consumer_root,
            set(),
        ).update(points)

    pending: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ] = {}
    for producer_root, consumers in sorted(consumers_by_producer.items()):
        key_domain = LogicalDomain(
            axis_order=(),
            axis_counts_items=(),
            kind="event",
        )
        _add_event_candidate(
            pending,
            key_domain=key_domain,
            contributions=(
                EventContribution(
                    producer_root=producer_root,
                    producer_scope_id=None,
                    keys=LogicalRelation.total(
                        root_domains[producer_root],
                        key_domain,
                    ),
                ),
            ),
            uses=tuple(
                EventUse(
                    consumer_root=consumer_root,
                    consumer_scope_id=None,
                    keys=LogicalRelation.total(
                        root_domains[consumer_root],
                        key_domain,
                    ),
                    dependency_points=frozenset(points),
                )
                for consumer_root, points in sorted(consumers.items())
            ),
        )
    return _finalize_keyed_events(pending)


def build_event_graph(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...],
    axis_geometry: dict[int, tuple[int, int]],
    publishable_scope_ids: frozenset[int] | None = None,
) -> EventGraph:
    """Bind the symbolic readiness DAG for one selected configuration."""
    configured_root_domains = instantiate_root_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    if (
        any(domain is None for domain in configured_root_domains)
        or tuple(domain for domain in configured_root_domains if domain is not None)
        != root_domains
    ):
        raise ValueError("configured root domains disagree with the dependency graph")
    events = build_keyed_events(
        dependency_graph,
        axis_geometry=axis_geometry,
        publishable_scope_ids=publishable_scope_ids,
    )
    if events is None:
        events = _root_completion_events(dependency_graph, root_domains)
    return EventGraph(
        root_domains=root_domains,
        root_traversals=root_traversals,
        scope_domains=instantiate_scope_domains(
            dependency_graph,
            axis_geometry=axis_geometry,
        ),
        events=events,
    )


def derive_local_triggers(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule | None = None,
) -> tuple[LocalTrigger, ...]:
    """Select complete one-task-per-key uses for final-arrival execution.

    Static dispatch passes a worker schedule so continuations are selected only
    when a producer can execute them. Dynamic dispatch has no fixed placement;
    in that case the dependency relation alone determines eligibility.
    """
    required_points_by_root: dict[int, set[DependencyPoint]] = {}
    root_uses_by_dependency_point: dict[
        tuple[int, DependencyPoint],
        list[KeyedEvent],
    ] = {}
    for event in event_graph.events:
        for use in event.uses:
            if use.consumer_scope_id is None:
                required_points_by_root.setdefault(use.consumer_root, set()).update(
                    use.dependency_points
                )
                for dependency_point in use.dependency_points:
                    root_uses_by_dependency_point.setdefault(
                        (use.consumer_root, dependency_point), []
                    ).append(event)
    def missing_requirements_are_dominated(
        *,
        consumer_root: int,
        missing: frozenset[DependencyPoint],
        trigger_producer_roots: frozenset[int],
    ) -> bool:
        """Prove omitted singleton prerequisites precede the trigger producers.

        A continuation may read values beyond the allocation whose final arrival
        elects it. A missing family-completion dependency is nevertheless ready
        when that same family event gates a trigger contributor at root entry.
        This event-level proof distinguishes different producer callsites within
        one root; root ancestry alone is not sufficient.
        """
        for dependency_point in missing:
            events = root_uses_by_dependency_point.get(
                (consumer_root, dependency_point), ()
            )
            family_events = tuple(
                event for event in events if event.family_done_root is not None
            )
            if not family_events or any(
                not any(
                    event_use.consumer_scope_id is None
                    and event_use.consumer_root in trigger_producer_roots
                    for event_use in family_event.uses
                )
                for family_event in family_events
            ):
                return False
        return True

    candidates: list[
        tuple[
            int,
            int,
            int,
            KeyedEvent,
            EventUse,
            tuple[tuple[int, LogicalRelation], ...],
        ]
    ] = []
    for event in event_graph.events:
        if (
            event.family_done_root is not None
            or len(event.uses) != 1
            or any(
                contribution.producer_scope_id is not None
                for contribution in event.contributions
            )
        ):
            continue
        if any(
            not _local_contribution_is_lowerable(contribution)
            for contribution in event.contributions
        ):
            continue
        use_index = 0
        use = event.uses[use_index]
        if use.consumer_scope_id is not None:
            continue
        if not event.key_count:
            continue
        inverse_use = use.keys.inverse()
        missing_requirements = frozenset(
            required_points_by_root.get(use.consumer_root, ())
        ) - use.dependency_points
        trigger_producer_roots = frozenset(
            contribution.producer_root for contribution in event.contributions
        )
        if (
            (
                missing_requirements
                and not missing_requirements_are_dominated(
                    consumer_root=use.consumer_root,
                    missing=missing_requirements,
                    trigger_producer_roots=trigger_producer_roots,
                )
            )
            or not use.keys.is_total_function()
            or inverse_use is None
            or not inverse_use.is_total_function()
        ):
            continue
        producer_relations = _merge_relations_by_root(
            tuple(
                (contribution.producer_root, contribution.keys)
                for contribution in event.contributions
            )
        )
        if producer_relations is None:
            continue

        candidates.append(
            (
                use.consumer_root,
                event.event_id,
                use_index,
                event,
                use,
                producer_relations,
            )
        )

    conflicting_candidates: set[tuple[int, int]] = set()
    candidates_by_consumer_root: dict[int, list[tuple[int, int]]] = {}
    for consumer_root, event_id, use_index, *_rest in candidates:
        candidates_by_consumer_root.setdefault(consumer_root, []).append(
            (event_id, use_index)
        )
    for root_candidates in candidates_by_consumer_root.values():
        if len(root_candidates) > 1:
            conflicting_candidates.update(root_candidates)
    candidates_by_producer_root: dict[
        int,
        list[tuple[int, LogicalRelation]],
    ] = {}
    for candidate_index, candidate in enumerate(candidates):
        for producer_root, relation in candidate[-1]:
            candidates_by_producer_root.setdefault(producer_root, []).append(
                (candidate_index, relation)
            )
    for root_candidates in candidates_by_producer_root.values():
        for (left_index, left), (right_index, right) in itertools.combinations(
            root_candidates, 2
        ):
            if not left.has_disjoint_source_support(right):
                conflicting_candidates.update(
                    (
                        (candidates[left_index][1], candidates[left_index][2]),
                        (candidates[right_index][1], candidates[right_index][2]),
                    )
                )

    possible_workers_by_root = (
        {
            root: worker_schedule.workers_for_root(root)
            for root in range(len(event_graph.root_domains))
        }
        if worker_schedule is not None
        else None
    )

    result: list[LocalTrigger] = []
    for (
        _consumer_root,
        event_id,
        use_index,
        event,
        use,
        _producer_relations,
    ) in sorted(candidates, key=operator.itemgetter(slice(3))):
        if (event_id, use_index) in conflicting_candidates:
            continue
        if possible_workers_by_root is not None:
            possible_workers = frozenset(
                worker
                for contribution in event.contributions
                for worker in possible_workers_by_root[contribution.producer_root]
            )
            if not possible_workers:
                continue
            possible_workers_by_root[use.consumer_root] = possible_workers
        result.append(
            LocalTrigger(
                event_index=event_id,
                use_index=use_index,
            )
        )
    return tuple(result)


def order_local_contributors_by_key(
    event_graph: EventGraph,
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
    local_roots = {
        event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
        for trigger in local_triggers
    }
    replacement_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}

    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        if len(event.contributions) != 1:
            continue
        contribution = event.contributions[0]
        if contribution.producer_scope_id is not None:
            continue
        root = contribution.producer_root
        if root in local_roots or root in replacement_by_root:
            continue
        task_domain = event_graph.root_domains[root]
        inverse = contribution.keys.inverse()
        task_relation = None if inverse is None else inverse.fiber_enumeration()
        if (
            task_relation is None
            or task_relation.target_domain != event_graph.root_domains[root]
            or task_relation.source_domain.size != task_domain.size
        ):
            continue

        schedule_interval = worker_schedule.contiguous_global_interval(root)
        if (
            schedule_interval is None
            or schedule_interval[1] - schedule_interval[0] != task_domain.size
        ):
            continue
        schedule_begin = schedule_interval[0]
        replacement_by_root[root] = (
            WorkerScheduleSegment(
                root=root,
                task_begin=0,
                task_count=task_domain.size,
                worker_begin=0,
                worker_count=worker_schedule.worker_count,
                schedule_begin=schedule_begin,
                task_relation=task_relation,
            ),
        )

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
    event_graph: EventGraph,
    local_triggers: tuple[LocalTrigger, ...],
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    """Return the complete predecessor set for every locally executed task."""
    result: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        use = event.uses[trigger.use_index]
        contributors_by_key = event_graph.materialized_contributor_tasks_by_key(event)
        for consumer_task, required_keys in enumerate(
            use.keys.materialize(
                source_traversal=event_graph.source_traversal(
                    use.consumer_root,
                    use.consumer_scope_id,
                )
            )
        ):
            if len(required_keys) != 1:
                raise ValueError("a local trigger requires exactly one key per task")
            key = next(iter(required_keys))
            task = (use.consumer_root, consumer_task)
            if task in result:
                raise ValueError(f"task {task} has multiple local triggers")
            result[task] = contributors_by_key[key]
    return result


def validate_worker_schedule(
    event_graph: EventGraph,
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
        for root, domain in enumerate(event_graph.root_domains)
        for task in range(domain.size)
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

    graph_nodes = {("task", root, task) for root, task in static_tasks}
    successors: dict[tuple[str, int, int], set[tuple[str, int, int]]] = {
        node: set() for node in graph_nodes
    }
    indegree = dict.fromkeys(graph_nodes, 0)

    static_ancestors_cache: dict[
        tuple[int, int],
        frozenset[tuple[int, int]],
    ] = {}

    def add_edge(
        producer: tuple[str, int, int],
        consumer: tuple[str, int, int],
    ) -> None:
        if producer not in successors:
            successors[producer] = set()
            indegree[producer] = 0
        if consumer not in successors:
            successors[consumer] = set()
            indegree[consumer] = 0
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
            add_edge(
                ("task", *producer),
                ("task", *consumer),
            )

    for event in event_graph.events:
        contributors_by_key = event_graph.materialized_contributor_tasks_by_key(event)
        consumers_by_key: list[set[tuple[int, int]]] = [
            set() for _ in range(event.key_count)
        ]
        for use in event.uses:
            for consumer_task, required_keys in enumerate(
                event_graph.materialized_required_keys_by_strand(use)
            ):
                consumer = (use.consumer_root, consumer_task)
                if consumer in local_predecessors:
                    continue
                for key in required_keys:
                    consumers_by_key[key].add(consumer)
        for key, consumers in enumerate(consumers_by_key):
            if not consumers:
                continue
            event_node = ("event", event.event_id, key)
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
                    add_edge(("task", *ancestor), event_node)
            for consumer in consumers:
                add_edge(event_node, ("task", *consumer))

    ready = [task for task, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        task = ready.pop()
        visited += 1
        for successor in successors[task]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(indegree):
        blocked = sorted(node for node, degree in indegree.items() if degree)
        raise ValueError(
            f"worker schedule contains a dependency/order cycle involving {blocked[:8]}"
        )


def finalize_readiness_plan(
    *,
    event_graph: EventGraph,
    counted_events: tuple[CountedEventPlan, ...],
    dependency_plan: TileDependencyGraph,
    preordered_edges: frozenset[tuple[int, int]],
) -> ReadinessPlan:
    """Finalize dependency coverage once, independently of task dispatch."""
    covered_dependency_points = frozenset(
        dependency_point
        for event in counted_events
        for use in event.uses
        for dependency_point in use.dependency_points
    )
    root_completion_edges = _select_root_completion_edges(
        dependency_graph=dependency_plan,
        covered_dependency_points=covered_dependency_points,
        preordered_edges=preordered_edges,
    )
    root_order_edges = set(root_completion_edges) | set(preordered_edges)
    retained_counted_events: list[CountedEventPlan] = []
    for event in counted_events:
        retained_use_indices = tuple(
            use_index
            for use_index, use in enumerate(event.uses)
            if use_index == event.local_trigger_use
            or not all(
                _is_ordered_by_root_completion(
                    contributor.producer_root,
                    use.consumer_root,
                    root_order_edges,
                )
                for contributor in event.contributors
            )
        )
        if not retained_use_indices:
            continue
        retained_counted_events.append(
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
    counted_events = tuple(retained_counted_events)
    covered_dependency_points = frozenset(
        dependency_point
        for event in counted_events
        for use in event.uses
        for dependency_point in use.dependency_points
    )
    _validate_schedule_coverage(
        dependency_graph=dependency_plan,
        covered_dependency_points=covered_dependency_points,
        root_completion_edges=root_completion_edges,
        preordered_edges=preordered_edges,
    )
    event_graph, family_done_events = lower_family_done_events(
        event_graph,
        dependency_plan,
        root_completion_edges,
    )
    return ReadinessPlan(
        event_graph=event_graph,
        counted_events=(*counted_events, *family_done_events),
    )


def build_cross_loop_schedule(
    *,
    dependency_plan: TileDependencyGraph,
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...],
    axis_geometry: dict[int, tuple[int, int]],
    preordered_edges: frozenset[tuple[int, int]],
    physical_worker_limit: int,
    requested_worker_count: int = CROSS_LOOP_NUM_WORKERS_DEFAULT,
    publishable_scope_ids: frozenset[int] | None = None,
) -> CrossLoopSchedule:
    """Derive all generic readiness strategies without inspecting root bodies."""
    event_graph = build_event_graph(
        dependency_plan,
        root_domains=root_domains,
        root_traversals=root_traversals,
        axis_geometry=axis_geometry,
        publishable_scope_ids=publishable_scope_ids,
    )
    worker_limit = resolve_worker_count(
        event_graph,
        default_worker_count=physical_worker_limit,
        requested_worker_count=requested_worker_count,
    )
    try:
        (
            worker_schedule,
            local_triggers,
            nested_scope_events,
            event_graph,
        ) = build_worker_schedule(
            event_graph,
            worker_count=worker_limit,
        )
    except ValueError as error:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG}={requested_worker_count} does not "
            "admit a progress-safe worker schedule"
        ) from error

    nested_scope_dependency_points = frozenset(
        dependency_point
        for plan in nested_scope_events
        for use in plan.uses
        for dependency_point in use.dependency_points
    )
    counted_events = (
        *choose_counted_events(
            event_graph,
            local_triggers,
            excluded_dependency_points=nested_scope_dependency_points,
        ),
        *nested_scope_events,
    )
    readiness = finalize_readiness_plan(
        event_graph=event_graph,
        counted_events=counted_events,
        dependency_plan=dependency_plan,
        preordered_edges=preordered_edges,
    )
    worker_schedule = compact_ready_explicit_families(
        readiness,
        worker_schedule,
        local_triggers,
    )
    return CrossLoopSchedule(
        readiness=readiness,
        worker_schedule=worker_schedule,
        local_triggers=local_triggers,
        worker_limit=worker_limit,
    )


def build_clc_command_plan(base_schedule: CrossLoopSchedule) -> ClcCommandPlan:
    """Quotient the canonical worker schedule into topological CLC commands."""
    event_graph = base_schedule.event_graph
    task_count = sum(domain.size for domain in event_graph.root_domains)
    if task_count > _CLC_COMMAND_TASK_LIMIT:
        raise ClcCommandPlanUnavailable(
            "CLC command planning would materialize "
            f"{task_count} tasks; the current limit is "
            f"{_CLC_COMMAND_TASK_LIMIT}"
        )
    event_key_count = sum(plan.key_count for plan in base_schedule.counted_events)
    if event_key_count > _EXPLICIT_SCHEDULE_EVENT_KEY_LIMIT:
        raise ClcCommandPlanUnavailable(
            "CLC command planning would materialize "
            f"{event_key_count} event keys; the current limit is "
            f"{_EXPLICIT_SCHEDULE_EVENT_KEY_LIMIT}"
        )

    local_roots = frozenset(
        event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
        for trigger in base_schedule.local_triggers
    )
    command_ranges: list[ClcCommandRange] = []
    task_begin = 0
    for root, domain in enumerate(event_graph.root_domains):
        if root not in local_roots:
            command_ranges.append(
                ClcCommandRange(
                    root=root,
                    begin=task_begin,
                    end=task_begin + domain.size,
                )
            )
        task_begin += domain.size
    command_ranges_tuple = tuple(command_ranges)
    explicit_tasks = tuple(
        (command_range.root, task)
        for command_range in command_ranges_tuple
        for task in range(command_range.task_count)
    )
    commands = _clc_task_commands(
        base_schedule.worker_schedule,
        explicit_tasks,
    )
    commands = _topologically_order_clc_commands(
        commands,
        base_schedule.readiness,
        max_edges=_EXPLICIT_SCHEDULE_EDGE_LIMIT,
    )
    flat_task_by_logical = _clc_flat_task_by_logical(
        event_graph,
        command_ranges_tuple,
    )
    task_order = tuple(flat_task_by_logical[command.task] for command in commands)
    if len(task_order) != len(set(task_order)) or frozenset(task_order) != frozenset(
        flat_task_by_logical.values()
    ):
        raise AssertionError(
            "CLC commands do not cover every explicit task exactly once"
        )
    return ClcCommandPlan(
        base_schedule=base_schedule,
        command_ranges=command_ranges_tuple,
        commands=commands,
        task_order=task_order,
    )


def _clc_contributor_tasks_by_key(
    event_graph: EventGraph,
    contributors: tuple[EventContribution, ...],
    key_count: int,
    *,
    budget: _ClcMaterializationBudget | None = None,
) -> tuple[frozenset[tuple[int, int]], ...]:
    result: list[set[tuple[int, int]]] = [set() for _ in range(key_count)]
    for contributor in contributors:
        producer_root = contributor.producer_root
        strand_keys = contributor.keys.project_source(
            event_graph.root_domains[producer_root]
        )
        if strand_keys is None:
            raise ClcCommandPlanUnavailable(
                "CLC event contribution cannot be projected onto producer tasks"
        )
        for task in range(strand_keys.source_domain.size):
            keys = strand_keys.targets(task)
            if budget is not None:
                budget.consume(len(keys), "materializing producer incidences")
            for key in keys:
                result[key].add((producer_root, task))
    return tuple(frozenset(tasks) for tasks in result)


def _clc_local_predecessors(
    readiness: ReadinessPlan,
    *,
    budget: _ClcMaterializationBudget | None = None,
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    event_graph = readiness.event_graph
    result: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for plan in readiness.counted_events:
        local_use = plan.local_use
        if local_use is None:
            continue
        required_keys = event_graph.required_keys_by_strand(local_use)
        if required_keys is None:
            raise ClcCommandPlanUnavailable(
                "CLC continuation readiness cannot be projected onto its tasks"
            )
        producers_by_key = _clc_contributor_tasks_by_key(
            event_graph,
            plan.contributors,
            plan.key_count,
            budget=budget,
        )
        for consumer_task in range(required_keys.source_domain.size):
            keys = required_keys.targets(consumer_task)
            if budget is not None:
                budget.consume(
                    len(keys),
                    "materializing continuation consumer incidences",
                )
            if len(keys) != 1:
                raise ClcCommandPlanUnavailable(
                    "CLC continuation requires exactly one event key per task"
                )
            (key,) = tuple(keys)
            task = (local_use.consumer_root, consumer_task)
            if task in result:
                raise ClcCommandPlanUnavailable(
                    f"CLC task {task} has multiple continuation triggers"
                )
            result[task] = producers_by_key[key]
    return result


def _clc_explicit_ancestors(
    task: tuple[int, int],
    *,
    explicit_tasks: frozenset[tuple[int, int]],
    local_predecessors: dict[tuple[int, int], frozenset[tuple[int, int]]],
    cache: dict[tuple[int, int], frozenset[tuple[int, int]]],
    budget: _ClcMaterializationBudget | None = None,
    visiting: frozenset[tuple[int, int]] = frozenset(),
) -> frozenset[tuple[int, int]]:
    cached = cache.get(task)
    if cached is not None:
        return cached
    if task in explicit_tasks:
        result = frozenset((task,))
    elif task in visiting:
        raise exc.CrossLoopSchedulingError("CLC continuation graph contains a cycle")
    else:
        predecessors = local_predecessors.get(task)
        if predecessors is None:
            raise exc.CrossLoopSchedulingError(
                f"CLC continuation task {task} has no trigger"
            )
        result = frozenset(
            ancestor
            for predecessor in predecessors
            for ancestor in _clc_explicit_ancestors(
                predecessor,
                explicit_tasks=explicit_tasks,
                local_predecessors=local_predecessors,
                cache=cache,
                budget=budget,
                visiting=visiting | frozenset((task,)),
            )
        )
        if not result:
            raise exc.CrossLoopSchedulingError(
                f"CLC continuation task {task} has no explicit ancestor"
            )
    if budget is not None:
        budget.consume(len(result), "contracting continuation ancestors")
    cache[task] = result
    return result


def _clc_explicit_task_predecessors(
    readiness: ReadinessPlan,
    command_ranges: tuple[ClcCommandRange, ...],
    *,
    max_edges: int | None = None,
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    """Contract local continuations into one explicit task dependency DAG."""
    budget = _ClcMaterializationBudget(max_edges)
    event_graph = readiness.event_graph
    explicit_tasks = frozenset(
        (command_range.root, task)
        for command_range in command_ranges
        for task in range(command_range.task_count)
    )
    predecessors: dict[tuple[int, int], set[tuple[int, int]]] = {
        task: set() for task in explicit_tasks
    }
    local_predecessors = _clc_local_predecessors(
        readiness,
        budget=budget,
    )
    ancestor_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    explicit_roots = {command_range.root for command_range in command_ranges}

    for plan in readiness.counted_events:
        producers_by_key = _clc_contributor_tasks_by_key(
            event_graph,
            plan.contributors,
            plan.key_count,
            budget=budget,
        )
        explicit_producers_by_key = tuple(
            frozenset(
                ancestor
                for producer in producers
                for ancestor in _clc_explicit_ancestors(
                    producer,
                    explicit_tasks=explicit_tasks,
                    local_predecessors=local_predecessors,
                    cache=ancestor_cache,
                    budget=budget,
                )
            )
            for producers in producers_by_key
        )
        for use in plan.uses:
            if use.consumer_root not in explicit_roots:
                continue
            try:
                required_keys = event_graph.materialized_required_keys_by_strand(
                    use,
                    budget=budget,
                )
            except ValueError as error:
                raise ClcCommandPlanUnavailable(str(error)) from error
            for consumer_task, keys in enumerate(required_keys):
                consumer = (use.consumer_root, consumer_task)
                task_predecessors = predecessors[consumer]
                previous_count = len(task_predecessors)
                task_predecessors.update(
                    producer
                    for key in keys
                    for producer in explicit_producers_by_key[key]
                    if producer != consumer
                )
                budget.consume(
                    len(task_predecessors) - previous_count,
                    "expanding dense dependency edges",
                )
    return {
        task: frozenset(task_predecessors)
        for task, task_predecessors in predecessors.items()
    }


def _clc_flat_task_by_logical(
    event_graph: EventGraph,
    command_ranges: tuple[ClcCommandRange, ...],
) -> dict[tuple[int, int], int]:
    """Map logical tasks to their stable flattened physical task IDs."""
    result: dict[tuple[int, int], int] = {}
    for command_range in command_ranges:
        traversal = event_graph.root_traversals[command_range.root]
        for physical_task in range(command_range.task_count):
            logical_tasks = traversal.targets(physical_task)
            if len(logical_tasks) != 1:
                raise ClcCommandPlanUnavailable(
                    f"CLC root {command_range.root} traversal is not single-valued"
                )
            (logical_task,) = logical_tasks
            key = (command_range.root, logical_task)
            if key in result:
                raise ClcCommandPlanUnavailable(
                    f"CLC root {command_range.root} traversal repeats task "
                    f"{logical_task}"
                )
            result[key] = command_range.begin + physical_task
    return result


def _clc_task_commands(
    worker_schedule: WorkerSchedule,
    tasks: tuple[tuple[int, int], ...],
) -> tuple[ClcCommand, ...]:
    """Make each explicit logical task independently schedulable."""
    commands: list[ClcCommand] = []
    for task in tasks:
        placement = worker_schedule.placement(*task)
        if placement is None:
            raise ClcCommandPlanUnavailable(f"CLC task {task} has no worker placement")
        worker, position = placement
        commands.append(
            ClcCommand(
                worker=worker,
                position_begin=position,
                position_end=position + 1,
                task=task,
            )
        )
    return tuple(commands)


def _topologically_order_clc_commands(
    commands: tuple[ClcCommand, ...],
    readiness: ReadinessPlan,
    *,
    max_edges: int | None = None,
) -> tuple[ClcCommand, ...]:
    """Rank commands through compact event-key nodes.

    A counted event is a hyperedge from all contributors of one key to every
    consumer of that key.  Keeping one zero-cost node per key avoids expanding
    dense all-to-all dependencies while preserving exactly the same partial-
    residency argument: a command is issued only after every predecessor
    command has appeared earlier in the ticket stream.
    """
    budget = _ClcMaterializationBudget(max_edges)
    task_location: dict[tuple[int, int], int] = {}
    commands_by_worker: dict[int, list[int]] = {}
    for command_index, command in enumerate(commands):
        commands_by_worker.setdefault(command.worker, []).append(command_index)
        if command.task in task_location:
            raise AssertionError(
                f"CLC task {command.task} appears in multiple commands"
            )
        task_location[command.task] = command_index

    successors: list[set[int]] = [set() for _ in commands]
    indegree = [0] * len(commands)
    priorities: list[tuple[int, int, int, int]] = [
        (1, command.position_begin, command.worker, command_index)
        for command_index, command in enumerate(commands)
    ]
    def add_event_node(plan_index: int, key: int) -> int:
        node = len(successors)
        successors.append(set())
        indegree.append(0)
        # Event nodes have no work.  Process them before ready task nodes so
        # their consumers enter the same ready frontier as with direct edges.
        priorities.append((0, plan_index, key, node))
        return node

    def add_edge(producer: int, consumer: int) -> None:
        if producer == consumer or consumer in successors[producer]:
            return
        successors[producer].add(consumer)
        indegree[consumer] += 1
        budget.consume(1, "building compact dependency edges")

    for worker_commands in commands_by_worker.values():
        worker_commands.sort(key=lambda index: commands[index].position_begin)
        for producer, consumer in itertools.pairwise(worker_commands):
            add_edge(producer, consumer)

    event_graph = readiness.event_graph
    local_predecessors = _clc_local_predecessors(
        readiness,
        budget=budget,
    )
    ancestor_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    explicit_tasks = frozenset(task_location)
    for plan_index, plan in enumerate(readiness.counted_events):
        producers_by_key = _clc_contributor_tasks_by_key(
            event_graph,
            plan.contributors,
            plan.key_count,
            budget=budget,
        )
        consumers_by_key: list[set[tuple[int, int]]] = [
            set() for _ in range(plan.key_count)
        ]
        for use in plan.uses:
            try:
                required_keys_by_task = (
                    event_graph.materialized_required_keys_by_strand(
                        use,
                        budget=budget,
                    )
                )
            except ValueError as error:
                raise ClcCommandPlanUnavailable(str(error)) from error
            for consumer_task, required_keys in enumerate(required_keys_by_task):
                consumer = (use.consumer_root, consumer_task)
                if consumer not in explicit_tasks:
                    continue
                for key in required_keys:
                    consumers_by_key[key].add(consumer)

        for key, consumers in enumerate(consumers_by_key):
            if not consumers:
                continue
            event_node = add_event_node(plan_index, key)
            for producer in producers_by_key[key]:
                ancestors = _clc_explicit_ancestors(
                    producer,
                    explicit_tasks=explicit_tasks,
                    local_predecessors=local_predecessors,
                    cache=ancestor_cache,
                    budget=budget,
                )
                if not ancestors:
                    raise ClcCommandPlanUnavailable(
                        f"CLC task {producer} has no explicit executor"
                    )
                for ancestor in ancestors:
                    add_edge(task_location[ancestor], event_node)
            for consumer in consumers:
                add_edge(event_node, task_location[consumer])

    ready = [
        priorities[index] for index, degree in enumerate(indegree) if degree == 0
    ]
    heapq.heapify(ready)
    ordered: list[ClcCommand] = []
    while ready:
        *_priority, node = heapq.heappop(ready)
        if node < len(commands):
            ordered.append(commands[node])
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, priorities[successor])
    if any(indegree):
        raise ClcCommandPlanUnavailable("CLC command quotient contains a cycle")
    if len(ordered) != len(commands):
        raise AssertionError("CLC command order omitted an explicit task")
    return tuple(ordered)


def compact_ready_explicit_families(
    readiness: ReadinessPlan,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> WorkerSchedule:
    """Pack complete ready families into otherwise idle resident-worker slots.

    This is a structural compaction of the validated task graph.  It does not
    inspect operators or estimate their cost: a family is moved only when all
    of its explicit predecessors are placed in earlier worker positions, the
    complete family fits in one resident wave, and the resulting schedule is
    still acyclic.  Occupied partial waves are preferred so the pass closes
    holes instead of merely renumbering empty positions.
    """
    event_graph = readiness.event_graph
    if sum(domain.size for domain in event_graph.root_domains) > (
        _EXPLICIT_SCHEDULE_TASK_LIMIT
    ):
        return worker_schedule
    local_roots = frozenset(
        event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
        for trigger in local_triggers
    )
    nested_roots = frozenset(
        root
        for plan in readiness.counted_events
        for root, scope_id in (
            *((contribution.producer_root, contribution.producer_scope_id)
              for contribution in plan.contributors),
            *((use.consumer_root, use.consumer_scope_id) for use in plan.uses),
        )
        if scope_id is not None
    )
    command_ranges: list[ClcCommandRange] = []
    task_begin = 0
    for root, domain in enumerate(event_graph.root_domains):
        if root not in local_roots:
            command_ranges.append(
                ClcCommandRange(
                    root=root,
                    begin=task_begin,
                    end=task_begin + domain.size,
                )
            )
        task_begin += domain.size
    try:
        predecessors = _clc_explicit_task_predecessors(
            readiness,
            tuple(command_ranges),
            max_edges=_EXPLICIT_SCHEDULE_EDGE_LIMIT,
        )
    except ClcCommandPlanUnavailable:
        return worker_schedule

    result = worker_schedule
    for root, domain in enumerate(event_graph.root_domains):
        if (
            root in local_roots
            or root in nested_roots
            or domain.size > result.worker_count
        ):
            continue
        current_bounds = result.position_bounds_for_root(root)
        if current_bounds is None:
            continue
        root_predecessors = frozenset(
            predecessor
            for task in range(domain.size)
            for predecessor in predecessors[(root, task)]
        )
        if not root_predecessors or any(
            producer_root == root for producer_root, _task in root_predecessors
        ):
            continue
        predecessor_placements = tuple(
            result.placement(*predecessor) for predecessor in root_predecessors
        )
        if any(placement is None for placement in predecessor_placements):
            continue
        ready_after = max(
            placement[1]
            for placement in predecessor_placements
            if placement is not None
        )
        candidates: list[tuple[bool, int, WorkerSchedule]] = []
        for position in range(ready_after + 1, current_bounds[0]):
            occupied_count = sum(
                result.task_at(worker, position) is not None
                for worker in range(result.worker_count)
            )
            placements = _family_placements_at_position(
                result,
                root=root,
                task_domain=domain,
                task_traversal=event_graph.root_traversals[root],
                position=position,
                pack_high=0 < occupied_count < domain.size,
            )
            if not placements:
                continue
            candidates.append((occupied_count > 0, position, placements[0]))
        if not candidates:
            continue
        occupied_candidates = tuple(
            candidate for candidate in candidates if candidate[0]
        )
        ordered_candidates = sorted(
            occupied_candidates or tuple(candidates),
            key=lambda candidate: candidate[1],
            reverse=bool(occupied_candidates),
        )
        for _occupied, _position, candidate in ordered_candidates:
            try:
                validate_worker_schedule(
                    event_graph,
                    candidate,
                    local_triggers,
                )
            except ValueError:
                continue
            result = candidate
            break
    return result


def _validate_schedule_coverage(
    *,
    dependency_graph: TileDependencyGraph,
    covered_dependency_points: frozenset[DependencyPoint],
    root_completion_edges: frozenset[tuple[int, int]],
    preordered_edges: frozenset[tuple[int, int]],
) -> None:
    """Verify that every dependence has an emitted synchronization path."""
    root_order_edges = set(root_completion_edges) | set(preordered_edges)
    for dependency in dependency_graph.edges:
        pair = (dependency.producer_root, dependency.consumer_root)
        if _is_ordered_by_root_completion(*pair, root_order_edges):
            continue
        uncovered = tuple(
            dependency_point
            for access_dependency in dependency.access_dependencies
            for dependency_point in dependency_graph.dependency_points(
                access_dependency
            )
            if dependency_point not in covered_dependency_points
        )
        if not uncovered:
            continue
        raise exc.CrossLoopSchedulingError(
            f"{dependency.producer_root}->{dependency.consumer_root} through "
            f"allocations {sorted(dependency.tensor_names)!r} has no cross-loop "
            f"synchronization path for dependencies {uncovered!r}"
        )


def _select_root_completion_edges(
    *,
    dependency_graph: TileDependencyGraph,
    covered_dependency_points: frozenset[DependencyPoint],
    preordered_edges: frozenset[tuple[int, int]],
) -> frozenset[tuple[int, int]]:
    """Choose the minimal source-ordered root-completion fallback edges."""
    selected_edges: set[tuple[int, int]] = set()
    ordered_root_edges = set(preordered_edges)
    for dependency in sorted(
        dependency_graph.edges,
        key=lambda edge: (
            edge.consumer_root - edge.producer_root,
            edge.producer_root,
            edge.consumer_root,
        ),
    ):
        pair = (dependency.producer_root, dependency.consumer_root)
        if all(
            dependency_graph.dependency_points(access_dependency)
            <= covered_dependency_points
            for access_dependency in dependency.access_dependencies
        ):
            continue
        if _is_ordered_by_root_completion(*pair, ordered_root_edges):
            continue
        selected_edges.add(pair)
        ordered_root_edges.add(pair)
    return frozenset(selected_edges)


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
