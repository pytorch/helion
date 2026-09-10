from __future__ import annotations

import dataclasses
from functools import cache
from functools import cached_property
import heapq
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
from torch.utils._sympy.functions import CeilDiv
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

from .. import exc
from . import tile_dependency
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import DependencyObligation
from .tile_dependency import TileDependencyGraph
from .tile_dependency import _CoordinateRelationPiece
from .tile_dependency import _logical_expression_bounds
from .tile_dependency import _simplify_logical_expression
from .tile_dependency import consumer_to_preceding_site_relation
from .tile_dependency import coordinate_axis_symbol
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import nested_logical_axes
from .tile_dependency import pid_task_order

if TYPE_CHECKING:
    from collections.abc import Callable

WorkerInterval = tuple[int, int]

_SOURCE_LAUNCH_STAGE = 0
_RESIDENT_LAUNCH_STAGE = 1


def _normalize_intervals(
    intervals: tuple[WorkerInterval, ...] | list[WorkerInterval],
) -> tuple[WorkerInterval, ...]:
    """Return a sorted, disjoint union of half-open integer intervals."""
    result: list[WorkerInterval] = []
    for begin, end in sorted(intervals):
        if begin >= end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return tuple(result)


def _subtract_intervals(
    minuend: tuple[WorkerInterval, ...],
    subtrahend: tuple[WorkerInterval, ...],
) -> tuple[WorkerInterval, ...]:
    """Return ``minuend - subtrahend`` using only interval endpoints."""
    result: list[WorkerInterval] = []
    subtrahend = _normalize_intervals(subtrahend)
    for begin, end in _normalize_intervals(minuend):
        cursor = begin
        for remove_begin, remove_end in subtrahend:
            if remove_end <= cursor:
                continue
            if remove_begin >= end:
                break
            if cursor < remove_begin:
                result.append((cursor, min(remove_begin, end)))
            cursor = max(cursor, remove_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


def _interval_cardinality(intervals: tuple[WorkerInterval, ...]) -> int:
    """Return the exact number of integers in disjoint half-open intervals."""
    return sum(end - begin for begin, end in _normalize_intervals(intervals))


@dataclasses.dataclass(frozen=True)
class WorkerScheduleSegment:
    """One root's ownership relation in a static persistent-worker schedule.

    In a normalized :class:`WorkerSchedule`, ``task_order`` maps global
    ``(launch stage, worker, wave)`` coordinates directly to logical tasks.
    Its source support is allowed to be partial.  The three integer placement
    fields remain only as the migration-compatible spelling of a dense run;
    :class:`WorkerSchedule` immediately normalizes that spelling into the
    relation and all semantic queries consume the relation.

    Before normalization, constructors may still pass a dense logical task
    order plus ``dispatch_offset``::

        dispatch_index = dispatch_offset + task_order_index
        worker = worker_begin + dispatch_index % worker_count
        worker_step = dispatch_index // worker_count

    This compatibility form is never a second accepted schedule: construction
    succeeds only when it can be converted exactly to the authoritative
    relation.
    """

    root: int
    task_order: CoordinateRelation
    worker_begin: int
    worker_count: int
    dispatch_offset: int

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError(f"root must be nonnegative, got {self.root}")
        if self.worker_begin < 0:
            raise ValueError(
                f"worker_begin must be nonnegative, got {self.worker_begin}"
            )
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        if self.dispatch_offset < 0:
            raise ValueError(
                f"dispatch_offset must be nonnegative, got {self.dispatch_offset}"
            )
        if self.task_order.target_domain.kind != "site" or (
            not self.task_order.pieces
            and self.task_order.target_domain.size_expr.is_zero is not True
        ):
            raise ValueError(
                "symbolic worker schedule relation has incompatible domains"
            )
        if self.task_order.source_domain.kind not in ("task_order", "worker"):
            raise ValueError(
                "worker schedule source must be a task order or schedule domain"
            )
        if self.task_order.source_domain.kind == "worker" and (
            len(self.task_order.source_domain.axis_order) != 3
            or self.task_order.source_domain.axis_count_expressions[
                self.task_order.source_domain.axis_order[0]
            ]
            != 2
        ):
            raise ValueError("normalized worker schedule has incompatible axes")

    @cached_property
    def task_count_expr(self) -> sympy.Expr:
        """Exact number of schedule slots owned by this segment relation."""
        if self.task_order.source_domain.kind == "task_order":
            return self.task_order.source_domain.size_expr
        cardinality = self.task_order.source_support_cardinality()
        if cardinality is None:
            raise ValueError("worker schedule support cardinality is not exact")
        return sympy.sympify(cardinality)

    @cached_property
    def task_count(self) -> int:
        """Concrete compatibility view of :attr:`task_count_expr`."""
        task_count = sympy.simplify(self.task_count_expr)
        if task_count.free_symbols or not isinstance(task_count, sympy.Integer):
            raise ValueError("worker schedule task count is symbolic")
        return int(task_count)

    @property
    def is_normalized(self) -> bool:
        """Return whether ``task_order`` is the authoritative schedule map."""
        return self.task_order.source_domain.kind == "worker"

    @cached_property
    def launch_stage(self) -> int | None:
        """Return the sole launch stage occupied by this relation, if proved."""
        if not self.is_normalized:
            return None
        canonical = self.task_order.canonical_single_valued()
        if canonical is None:
            canonical = self.task_order
        launch_stage_axis = canonical.source_domain.axis_order[0]
        bounds = {
            (begin, end, step)
            for piece in canonical.pieces
            for axis, begin, end, step in piece.source_bounds_items
            if axis == launch_stage_axis
        }
        if len(bounds) != 1:
            return None
        (launch_stage_bound,) = bounds
        begin, end, step = launch_stage_bound
        if (
            step != 1
            or end != begin + 1  # pyrefly: ignore[unsupported-operation]
            or begin not in (_SOURCE_LAUNCH_STAGE, _RESIDENT_LAUNCH_STAGE)
        ):
            return None
        return int(begin)

    @cached_property
    def logical_task_order(self) -> CoordinateRelation | None:
        """Derive the dense logical traversal represented by this run.

        This is compatibility and code-rendering machinery.  Placement and
        correctness use ``task_order`` itself after normalization.
        """
        if not self.is_normalized:
            return self.task_order
        try:
            task_count = self.task_count_expr
        except ValueError:
            return None
        ordinal_axis = (
            max(
                (
                    *self.task_order.source_domain.axis_order,
                    *self.task_order.target_domain.axis_order,
                ),
                default=0,
            )
            + 1
        )
        ordinal_domain = CoordinateDomain(
            axis_order=(ordinal_axis,),
            axis_counts_items=((ordinal_axis, task_count),),
            kind="task_order",
            _allow_empty=task_count.is_zero is True,
        )
        if task_count.is_zero is True:
            result = CoordinateRelation(
                source_domain=ordinal_domain,
                target_domain=self.task_order.target_domain,
                pieces=(),
            )
            converse = CoordinateRelation(
                source_domain=self.task_order.target_domain,
                target_domain=ordinal_domain,
                pieces=(),
            )
            tile_dependency._remember_exact_converse(result, converse)
            return result
        ordinal = coordinate_axis_symbol(ordinal_axis)
        dispatch_index = self.dispatch_offset + ordinal  # pyrefly: ignore[unsupported-operation]
        launch_stage_axis, worker_axis, wave_axis = (
            self.task_order.source_domain.axis_order
        )
        launch_stage = self.launch_stage
        if launch_stage is None:
            return None
        launch_stage_bound = (launch_stage, launch_stage + 1, 1)

        if task_count.free_symbols:
            # Prove the compatibility fields directly against an exact dense
            # ordinalization of the authoritative placement support.  This is
            # bounded by relation pieces and never samples runtime task counts.
            support_relation = CoordinateRelation.point_map(
                self.task_order.source_domain,
                ordinal_domain,
                tuple(
                    (piece.source_bounds_items, (sympy.Integer(0),))
                    for piece in self.task_order.pieces
                ),
            ).coalesce_adjacent_source_boxes()
            support_ordinal = support_relation._ordinalized_source_support
            authoritative_dispatch = (
                None if support_ordinal is None else support_ordinal.converse()
            )
            compatibility_dispatch = CoordinateRelation.point_map(
                ordinal_domain,
                self.task_order.source_domain,
                (
                    (
                        ((ordinal_axis, 0, task_count, 1),),
                        (
                            sympy.Integer(launch_stage),
                            self.worker_begin
                            + sympy.Mod(dispatch_index, self.worker_count),
                            sympy.floor(dispatch_index / self.worker_count),
                        ),
                    ),
                ),
            )
            if (
                support_ordinal is not None
                and authoritative_dispatch is not None
                and authoritative_dispatch.is_total_function()
                and authoritative_dispatch.is_pointwise_equal_on_same_support(
                    compatibility_dispatch
                )
            ):
                logical_order = tile_dependency._factor_through_source_ordinalization(
                    self.task_order,
                    support_ordinal,
                    authoritative_dispatch,
                )
                logical_to_ordinal = (
                    None if logical_order is None else logical_order.converse()
                )
                if (
                    logical_order is not None
                    and logical_order.is_total_function()
                    and logical_to_ordinal is not None
                    and logical_to_ordinal.is_single_valued()
                ):
                    return logical_order
            return None

        # Retain the bounded concrete compatibility proof for older segmented
        # schedules whose source support has no compact ordinalization.
        task_count_int = int(task_count)
        canonical = self.task_order.canonical_single_valued()
        if canonical is None:
            return None
        schedule_ordinal = (
            coordinate_axis_symbol(wave_axis) * self.worker_count  # pyrefly: ignore[unsupported-operation]
            + coordinate_axis_symbol(worker_axis)
            - self.worker_begin
            - self.dispatch_offset
        )
        for piece in canonical.pieces:
            bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            worker_begin, worker_end, _worker_step = bounds[worker_axis]
            ordinal_bounds = _logical_expression_bounds(
                schedule_ordinal,
                domain=canonical.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                bounds[launch_stage_axis] != launch_stage_bound
                or worker_begin < self.worker_begin
                or worker_end > self.worker_begin + self.worker_count
                or ordinal_bounds is None
                or ordinal_bounds[0] < 0  # pyrefly: ignore[unsupported-operation]
                or ordinal_bounds[1] >= task_count_int  # pyrefly: ignore[unsupported-operation]
            ):
                return None
        substitutions = {
            coordinate_axis_symbol(launch_stage_axis): sympy.Integer(launch_stage),
            coordinate_axis_symbol(worker_axis): self.worker_begin  # pyrefly: ignore[unsupported-operation]
            + sympy.Mod(dispatch_index, self.worker_count),  # pyrefly: ignore[unsupported-operation]
            coordinate_axis_symbol(wave_axis): sympy.floor(
                dispatch_index / self.worker_count
            ),
        }
        full_ordinal_bounds = ((ordinal_axis, 0, task_count_int, 1),)
        pieces: list[
            tuple[tuple[tuple[int, int, int, int], ...], tuple[sympy.Expr, ...]]
        ] = []
        for piece in self.task_order.pieces:
            source_bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            if source_bounds[launch_stage_axis] != launch_stage_bound:
                continue
            worker_begin, worker_end, worker_step = source_bounds[worker_axis]
            wave_begin, wave_end, wave_step = source_bounds[wave_axis]
            if wave_end == wave_begin + 1:
                ordinal_begin = (
                    wave_begin * self.worker_count
                    + worker_begin
                    - self.worker_begin
                    - self.dispatch_offset
                )
                ordinal_end = ordinal_begin + worker_end - worker_begin
                ordinal_step = worker_step
            elif (
                worker_begin == self.worker_begin
                and worker_end == self.worker_begin + self.worker_count
                and worker_step == 1
            ):
                ordinal_begin = wave_begin * self.worker_count - self.dispatch_offset
                ordinal_end = (
                    (wave_end - 1) * self.worker_count
                    - self.dispatch_offset
                    + self.worker_count
                )
                ordinal_step = wave_step
                if wave_step != 1:
                    return None
            elif worker_end == worker_begin + 1 and worker_step == 1:
                ordinal_begin = (
                    wave_begin * self.worker_count
                    + worker_begin
                    - self.worker_begin
                    - self.dispatch_offset
                )
                ordinal_end = (
                    (wave_end - 1) * self.worker_count
                    + worker_begin
                    - self.worker_begin
                    - self.dispatch_offset
                    + 1
                )
                ordinal_step = wave_step * self.worker_count
            else:
                return None
            ordinal_begin = max(0, ordinal_begin)
            ordinal_end = min(task_count_int, ordinal_end)
            if ordinal_begin >= ordinal_end:
                continue
            ordinal_bounds = ((ordinal_axis, ordinal_begin, ordinal_end, ordinal_step),)
            target_expressions: list[sympy.Expr] = []
            for _axis, begin, end, step in piece.target_ranges:
                if step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
                    return None
                expression = (
                    ordinal
                    if sympy.simplify(
                        begin
                        - (  # pyrefly: ignore[unsupported-operation]
                            coordinate_axis_symbol(worker_axis)  # pyrefly: ignore[unsupported-operation]
                            - self.worker_begin  # pyrefly: ignore[unsupported-operation]
                            + coordinate_axis_symbol(wave_axis) * self.worker_count  # pyrefly: ignore[unsupported-operation]
                            - self.dispatch_offset
                        )
                    )
                    == 0
                    else cast("sympy.Expr", begin.xreplace(substitutions))
                )
                target_expressions.append(
                    _simplify_logical_expression(
                        expression,
                        domain=ordinal_domain,
                        source_bounds=full_ordinal_bounds,
                    )
                )
            pieces.append(
                (
                    ordinal_bounds,
                    tuple(target_expressions),
                )
            )
        result = CoordinateRelation.point_map(
            ordinal_domain,
            self.task_order.target_domain,
            tuple(pieces),
        )
        compact = result.coalesce_adjacent_source_boxes()
        if compact.is_total_function():
            result = compact
        elif launch_stage != _SOURCE_LAUNCH_STAGE:
            return None
        if not result.is_total_function():
            return None
        schedule_to_ordinal = CoordinateRelation.point_map(
            self.task_order.source_domain,
            ordinal_domain,
            (
                (
                    tuple(
                        (
                            axis,
                            0,
                            self.task_order.source_domain.axis_count_expressions[axis],
                            1,
                        )
                        for axis in self.task_order.source_domain.axis_order
                    ),
                    (schedule_ordinal,),
                ),
            ),
        )
        logical_to_schedule = self.task_order.converse()
        logical_to_ordinal = (
            None
            if logical_to_schedule is None
            else logical_to_schedule.then(schedule_to_ordinal)
        )
        if logical_to_ordinal is None:
            return None
        tile_dependency._remember_exact_converse(result, logical_to_ordinal)
        return result

    def logical_task_wave_relation(
        self,
        wave_domain: CoordinateDomain,
    ) -> CoordinateRelation | None:
        """Derive this dense rendering run's task-order-to-wave map."""
        logical_order = self.logical_task_order
        if logical_order is None:
            return None
        if logical_order.source_domain.size_expr.is_zero is True:
            return CoordinateRelation(
                source_domain=logical_order.source_domain,
                target_domain=wave_domain,
                pieces=(),
            )
        local_ordinal = _flat_domain_index_expression(logical_order.source_domain)
        wave = sympy.floor(  # pyrefly: ignore[bad-argument-type]
            (self.dispatch_offset + local_ordinal) / self.worker_count  # pyrefly: ignore[unsupported-operation]
        )
        return CoordinateRelation.point_map(
            logical_order.source_domain,
            wave_domain,
            (
                (
                    tuple(
                        (
                            axis,
                            0,
                            logical_order.source_domain.axis_count_expressions[axis],
                            1,
                        )
                        for axis in logical_order.source_domain.axis_order
                    ),
                    (wave,),
                ),
            ),
        )

    def dispatch_index(self, task_order_index: int) -> int:
        """Return the linearized dispatch index for one ordered task."""
        if not 0 <= task_order_index < self.task_count:
            raise IndexError(task_order_index)
        return self.dispatch_offset + task_order_index

    def occupies(self, worker: int, worker_step: int) -> bool:
        """Return whether this segment occupies one resident worker step."""
        if self.is_normalized:
            launch_stage_axis, worker_axis, wave_axis = (
                self.task_order.source_domain.axis_order
            )
            targets = self.task_order.target_coordinates(
                {
                    launch_stage_axis: _RESIDENT_LAUNCH_STAGE,
                    worker_axis: worker,
                    wave_axis: worker_step,
                }
            )
            if len(targets) > 1:
                raise AssertionError("one schedule slot maps to multiple tasks")
            return bool(targets)
        worker_offset = worker - self.worker_begin
        if not 0 <= worker_offset < self.worker_count or worker_step < 0:
            return False
        dispatch_index = worker_step * self.worker_count + worker_offset
        task_order_index = dispatch_index - self.dispatch_offset
        return 0 <= task_order_index < self.task_count

    def worker_step_bounds(self, worker: int) -> tuple[int, int] | None:
        """Return this segment's first and last resident step on one worker."""
        if self.is_normalized and self.launch_stage != _RESIDENT_LAUNCH_STAGE:
            return None
        worker_offset = worker - self.worker_begin
        if not 0 <= worker_offset < self.worker_count:
            return None
        begin = self.dispatch_offset
        end = begin + self.task_count
        first = begin + (worker_offset - begin) % self.worker_count
        if first >= end:
            return None
        last = first + (end - 1 - first) // self.worker_count * self.worker_count
        return first // self.worker_count, last // self.worker_count

    def workers(self) -> frozenset[int]:
        """Materialize active resident workers for diagnostics and tests."""
        return frozenset(
            worker
            for begin, end in self.worker_intervals()
            for worker in range(begin, end)
        )

    def worker_step_runs(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return resident ``(worker begin, end, first step, last step)`` runs.

        A dense dispatch has only constant-many changes: at the first-dispatch
        wrap and the final-dispatch wrap.  Partitioning at those endpoints
        proves the bounds for arbitrary task and worker counts without
        iterating either domain.  Production optimized schedules impose the
        narrower wave-aligned form when constructing their rank certificate.
        """
        if self.is_normalized:
            canonical = self.task_order.canonical_single_valued()
            if canonical is None and self.task_order.is_single_valued():
                canonical = self.task_order
            if canonical is None:
                raise ValueError("worker schedule relation is not single-valued")
            launch_stage_axis, worker_axis, wave_axis = (
                canonical.source_domain.axis_order
            )
            support_by_waves: dict[
                tuple[int, int, int], list[tuple[int, int, int]]
            ] = {}
            for piece in canonical.pieces:
                bounds = {
                    axis: (begin, end, step)
                    for axis, begin, end, step in piece.source_bounds_items
                }
                if bounds[launch_stage_axis] != (
                    _RESIDENT_LAUNCH_STAGE,
                    _RESIDENT_LAUNCH_STAGE + 1,
                    1,
                ):
                    continue
                worker_begin, worker_end, worker_step = bounds[worker_axis]
                wave_begin, wave_end, wave_step = bounds[wave_axis]
                support_by_waves.setdefault(
                    (wave_begin, wave_end, wave_step), []
                ).append((worker_begin, worker_end, worker_step))
            runs: list[tuple[int, int, int, int]] = []
            for (
                wave_begin,
                wave_end,
                wave_step,
            ), worker_support in support_by_waves.items():
                if wave_step != 1:
                    raise ValueError("worker schedule wave support must be contiguous")
                support_begin = min(begin for begin, _end, _step in worker_support)
                support_end = max(end for _begin, end, _step in worker_support)
                support_count = sum(
                    len(range(begin, end, step)) for begin, end, step in worker_support
                )
                if support_count != support_end - support_begin:
                    raise ValueError("worker schedule has noncontiguous worker support")
                for index, (begin, end, step) in enumerate(worker_support):
                    for other_begin, other_end, other_step in worker_support[
                        index + 1 :
                    ]:
                        overlap_begin = max(begin, other_begin)
                        overlap_end = min(end, other_end)
                        if overlap_begin >= overlap_end:
                            continue
                        if (other_begin - begin) % math.gcd(step, other_step) == 0:
                            raise ValueError("worker schedule support overlaps")
                runs.append((support_begin, support_end, wave_begin, wave_end - 1))
            return tuple(sorted(dict.fromkeys(runs), key=operator.itemgetter(2, 0)))

        worker_count = self.worker_count
        first_residue = self.dispatch_offset % worker_count
        active_count = min(self.task_count, worker_count)
        first_end = first_residue + active_count
        support = (
            ((first_residue, first_end),)
            if first_end <= worker_count
            else ((0, first_end - worker_count), (first_residue, worker_count))
        )
        final_residue = (self.dispatch_offset + self.task_count - 1) % worker_count
        structural_cuts = {0, worker_count, first_residue, final_residue + 1}
        structural_cuts.update(
            endpoint for interval in support for endpoint in interval
        )
        cuts = sorted(structural_cuts)
        runs: list[tuple[int, int, int, int]] = []
        for begin, end in itertools.pairwise(cuts):
            if begin >= end or not any(
                support_begin <= begin and end <= support_end
                for support_begin, support_end in support
            ):
                continue
            begin_bounds = self.worker_step_bounds(self.worker_begin + begin)
            end_bounds = self.worker_step_bounds(self.worker_begin + end - 1)
            if begin_bounds is None or begin_bounds != end_bounds:
                raise AssertionError("worker-step structural partition is incomplete")
            runs.append(
                (
                    self.worker_begin + begin,
                    self.worker_begin + end,
                    *begin_bounds,
                )
            )
        return tuple(runs)

    def worker_intervals(self) -> tuple[WorkerInterval, ...]:
        """Return the active-worker support without worker enumeration."""
        return _normalize_intervals(
            [(begin, end) for begin, end, _first, _last in self.worker_step_runs()]
        )


def _flat_domain_index_expression(domain: CoordinateDomain) -> sympy.Expr:
    """Return the canonical flattened index of a logical coordinate."""
    result: sympy.Expr = sympy.Integer(0)
    multiplier: sympy.Expr = sympy.Integer(1)
    counts = domain.axis_count_expressions
    for axis in domain.axis_order:
        result += coordinate_axis_symbol(axis) * multiplier  # pyrefly: ignore[unsupported-operation]
        multiplier = sympy.simplify(multiplier * counts[axis])
    return sympy.simplify(result)


def _worker_schedule_domain(
    worker_count: int,
    wave_count: int | sympy.Expr,
    axes: tuple[int, int, int],
) -> CoordinateDomain:
    """Return the shared launch-stage, worker, and wave schedule domain."""
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    wave_count_expr = sympy.sympify(wave_count)
    if (
        wave_count_expr.is_integer is not True
        or wave_count_expr.is_nonnegative is not True
    ):
        raise ValueError("wave count must be a nonnegative integer expression")
    launch_stage_axis, worker_axis, wave_axis = axes
    return CoordinateDomain(
        axis_order=axes,
        axis_counts_items=(
            (launch_stage_axis, 2),
            (worker_axis, worker_count),
            (wave_axis, wave_count_expr),
        ),
        kind="worker",
        _allow_empty=wave_count_expr.is_zero is True,
    )


def _normalize_dense_schedule_segment(
    segment: WorkerScheduleSegment,
    schedule_domain: CoordinateDomain,
    *,
    launch_stage: int = _RESIDENT_LAUNCH_STAGE,
) -> WorkerScheduleSegment:
    """Convert one legacy dense run to its exact schedule ownership relation."""
    if launch_stage not in (_SOURCE_LAUNCH_STAGE, _RESIDENT_LAUNCH_STAGE):
        raise ValueError(f"invalid launch stage {launch_stage}")
    launch_stage_axis, worker_axis, wave_axis = schedule_domain.axis_order
    if segment.is_normalized:
        source = segment.task_order.source_domain
        if source == schedule_domain:
            return segment
        source_counts = source.axis_count_expressions
        schedule_counts = schedule_domain.axis_count_expressions
        if (
            source.axis_order != schedule_domain.axis_order
            or not _equal_integer_expressions(
                source_counts[launch_stage_axis],
                schedule_counts[launch_stage_axis],
            )
            or not _equal_integer_expressions(
                source_counts[worker_axis],
                schedule_counts[worker_axis],
            )
            or sympy.simplify(
                schedule_counts[wave_axis] - source_counts[wave_axis]
            ).is_nonnegative
            is not True
        ):
            raise ValueError("worker schedule segments do not share one domain")
        rebased_task_order = segment.task_order.rebase_source_domain(schedule_domain)
        if rebased_task_order is None:
            raise ValueError("worker schedule segment cannot widen its source domain")
        return dataclasses.replace(segment, task_order=rebased_task_order)

    logical_order = segment.task_order
    worker = coordinate_axis_symbol(worker_axis)
    wave = coordinate_axis_symbol(wave_axis)
    local_ordinal = (
        wave * segment.worker_count  # pyrefly: ignore[unsupported-operation]
        + worker
        - segment.worker_begin
        - segment.dispatch_offset
    )
    ordinal_axis = (
        max(
            (
                *schedule_domain.axis_order,
                *logical_order.source_domain.axis_order,
                *logical_order.target_domain.axis_order,
            ),
            default=0,
        )
        + 1
    )
    ordinal_domain = CoordinateDomain(
        axis_order=(ordinal_axis,),
        axis_counts_items=((ordinal_axis, logical_order.source_domain.size),),
        kind="task_order",
    )
    flat_logical_order = _flat_task_order_relation(logical_order, ordinal_domain)
    if (
        flat_logical_order is None
        or not flat_logical_order.is_total_function()
        or flat_logical_order.converse() is None
    ):
        raise ValueError("logical task order cannot be flattened exactly")

    def dispatch_boxes(
        ordinal_begin: int,
        ordinal_end: int,
        ordinal_step: int,
    ) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        """Map one dense ordinal interval to constant-many schedule boxes."""
        if ordinal_step != 1:
            ordinal_values = range(ordinal_begin, ordinal_end, ordinal_step)
            if not ordinal_values:
                return ()
            dispatch_begin = segment.dispatch_offset + ordinal_values[0]
            dispatch_last = segment.dispatch_offset + ordinal_values[-1]
            first_wave, first_worker = divmod(dispatch_begin, segment.worker_count)
            last_wave, last_worker = divmod(dispatch_last, segment.worker_count)
            if first_wave == last_wave:
                return (
                    (
                        (
                            launch_stage_axis,
                            launch_stage,
                            launch_stage + 1,
                            1,
                        ),
                        (
                            worker_axis,
                            segment.worker_begin + first_worker,
                            segment.worker_begin + last_worker + 1,
                            ordinal_step,
                        ),
                        (wave_axis, first_wave, first_wave + 1, 1),
                    ),
                )
            if ordinal_step % segment.worker_count == 0 and first_worker == last_worker:
                return (
                    (
                        (
                            launch_stage_axis,
                            launch_stage,
                            launch_stage + 1,
                            1,
                        ),
                        (
                            worker_axis,
                            segment.worker_begin + first_worker,
                            segment.worker_begin + first_worker + 1,
                            1,
                        ),
                        (
                            wave_axis,
                            first_wave,
                            last_wave + 1,
                            ordinal_step // segment.worker_count,
                        ),
                    ),
                )
            raise ValueError("strided task order crosses worker-wave boundaries")
        dispatch_begin = segment.dispatch_offset + ordinal_begin
        dispatch_end = segment.dispatch_offset + ordinal_end
        first_wave, first_worker = divmod(dispatch_begin, segment.worker_count)
        final_wave, final_worker = divmod(dispatch_end, segment.worker_count)
        boxes: list[tuple[tuple[int, int, int, int], ...]] = []

        def add(
            worker_offset_begin: int, worker_offset_end: int, begin: int, end: int
        ) -> None:
            if worker_offset_begin >= worker_offset_end or begin >= end:
                return
            boxes.append(
                (
                    (
                        launch_stage_axis,
                        launch_stage,
                        launch_stage + 1,
                        1,
                    ),
                    (
                        worker_axis,
                        segment.worker_begin + worker_offset_begin,
                        segment.worker_begin + worker_offset_end,
                        1,
                    ),
                    (wave_axis, begin, end, 1),
                )
            )

        if first_wave == final_wave:
            add(first_worker, final_worker, first_wave, first_wave + 1)
            return tuple(boxes)
        add(first_worker, segment.worker_count, first_wave, first_wave + 1)
        middle_begin = first_wave + 1
        middle_end = final_wave
        add(0, segment.worker_count, middle_begin, middle_end)
        add(0, final_worker, final_wave, final_wave + 1)
        return tuple(boxes)

    (ordinal_axis,) = ordinal_domain.axis_order
    ordinal_symbol = coordinate_axis_symbol(ordinal_axis)
    schedule_to_ordinal = CoordinateRelation.point_map(
        schedule_domain,
        ordinal_domain,
        tuple(
            (source_bounds, (local_ordinal,))
            for source_bounds in dispatch_boxes(0, segment.task_count, 1)
        ),
    )
    dispatch_index = segment.dispatch_offset + ordinal_symbol
    ordinal_to_schedule = CoordinateRelation.point_map(
        ordinal_domain,
        schedule_domain,
        (
            (
                ((ordinal_axis, 0, segment.task_count, 1),),
                (
                    sympy.Integer(launch_stage),
                    segment.worker_begin
                    + sympy.Mod(dispatch_index, segment.worker_count),
                    cast(
                        "sympy.Expr",
                        FloorDiv(dispatch_index, segment.worker_count),
                    ),
                ),
            ),
        ),
    )
    tile_dependency._remember_exact_converse(
        schedule_to_ordinal,
        ordinal_to_schedule,
    )
    schedule_relation = schedule_to_ordinal.then(flat_logical_order)
    if schedule_relation is None:
        schedule_pieces: list[
            tuple[tuple[tuple[int, int, int, int], ...], tuple[sympy.Expr, ...]]
        ] = []
        for piece in flat_logical_order.pieces:
            ((piece_axis, piece_begin, piece_end, piece_step),) = (
                piece.source_bounds_items
            )
            if piece_axis != ordinal_axis:
                raise ValueError("dense worker schedule has the wrong ordinal axis")
            if any(
                step != 1 or sympy.simplify(end - begin) != 1  # pyrefly: ignore[unsupported-operation]
                for _axis, begin, end, step in piece.target_ranges
            ):
                raise ValueError("dense worker schedule requires a logical point map")
            for source_bounds in dispatch_boxes(
                piece_begin,
                piece_end,
                piece_step,
            ):
                target_expressions = tuple(
                    cast(
                        "sympy.Expr",
                        begin.xreplace({ordinal_symbol: local_ordinal}),
                    )
                    for _axis, begin, _end, _step in piece.target_ranges
                )
                schedule_pieces.append((source_bounds, target_expressions))
        schedule_relation = CoordinateRelation.point_map(
            schedule_domain,
            logical_order.target_domain,
            tuple(schedule_pieces),
        )
        logical_to_ordinal = tile_dependency._memoized_exact_converse(
            flat_logical_order
        ) or flat_logical_order.converse()
        schedule_inverse = (
            None
            if logical_to_ordinal is None
            else logical_to_ordinal.then(ordinal_to_schedule)
        )
        if schedule_inverse is not None:
            tile_dependency._remember_exact_converse(
                schedule_relation,
                schedule_inverse,
            )
    if (
        schedule_relation is None
        or tile_dependency._memoized_exact_converse(schedule_relation) is None
        or not schedule_relation.is_single_valued()
        or schedule_relation.source_support_cardinality()
        != logical_order.source_domain.size
    ):
        raise ValueError("dense worker run does not have exact schedule support")
    return dataclasses.replace(segment, task_order=schedule_relation)


def _task_order_ordinal_domain(
    task_order: CoordinateRelation,
) -> CoordinateDomain:
    """Create a collision-free scalar domain for one task-order traversal."""
    ordinal_axis = (
        max(
            (
                *task_order.source_domain.axis_order,
                *task_order.target_domain.axis_order,
            ),
            default=0,
        )
        + 1
    )
    return CoordinateDomain(
        axis_order=(ordinal_axis,),
        axis_counts_items=((ordinal_axis, task_order.source_domain.size_expr),),
        kind="task_order",
        _allow_empty=task_order.source_domain.size_expr.is_zero is True,
    )


def _flat_task_order_relation(
    task_order: CoordinateRelation,
    ordinal_domain: CoordinateDomain,
    *,
    ordinal_begin: int | sympy.Expr = 0,
) -> CoordinateRelation | None:
    """Map a flat emitted ordinal interval directly to logical tasks."""
    task_count = task_order.source_domain.size_expr
    ordinal_count = ordinal_domain.size_expr
    ordinal_begin = sympy.sympify(ordinal_begin)
    if (
        len(ordinal_domain.axis_order) != 1
        or not tile_dependency._is_provably_nonnegative(ordinal_begin, None)
        or not tile_dependency._is_provably_nonnegative(
            sympy.simplify(ordinal_count - ordinal_begin - task_count),
            None,
        )
    ):
        return None
    if task_count.is_zero is True:
        if task_order.target_domain.size_expr.is_zero is not True:
            return None
        result = CoordinateRelation(
            source_domain=ordinal_domain,
            target_domain=task_order.target_domain,
            pieces=(),
        )
        converse = CoordinateRelation(
            source_domain=task_order.target_domain,
            target_domain=ordinal_domain,
            pieces=(),
        )
        tile_dependency._remember_exact_converse(result, converse)
        return result
    task_order_converse = tile_dependency._memoized_exact_converse(task_order)
    if task_order_converse is None:
        task_order_converse = task_order.derive_converse_and_target_counts()[0]
        if task_order_converse is not None:
            tile_dependency._remember_exact_converse(
                task_order,
                task_order_converse,
            )
    if task_order_converse is None:
        return None
    if (
        _equal_integer_expressions(ordinal_begin, 0)
        and len(task_order.source_domain.axis_order) == 1
        and _equal_integer_expressions(task_count, ordinal_count)
    ):
        renamed = task_order.rename_source_axes(ordinal_domain)
        if renamed is not None:
            if renamed.converse() is None:
                return None
            return renamed
    (ordinal_axis,) = ordinal_domain.axis_order
    local_flat_index = coordinate_axis_symbol(ordinal_axis) - ordinal_begin  # pyrefly: ignore[unsupported-operation]
    local_coordinates: list[sympy.Expr] = []
    local_stride: sympy.Expr = sympy.Integer(1)
    source_axis_order = task_order.source_domain.axis_order
    for index, axis in enumerate(source_axis_order):
        axis_count = task_order.source_domain.axis_count_expressions[axis]
        quotient = (
            local_flat_index
            if local_stride == 1
            else cast("sympy.Expr", FloorDiv(local_flat_index, local_stride))
        )
        local_coordinates.append(
            cast(
                "sympy.Expr",
                quotient
                if index == len(source_axis_order) - 1
                else sympy.Mod(quotient, axis_count)
                if not axis_count.free_symbols
                else sympy.simplify(
                    quotient
                    - cast("sympy.Expr", FloorDiv(quotient, axis_count)) * axis_count
                ),
            )
        )
        local_stride = sympy.simplify(local_stride * axis_count)
    ordinal_to_local = CoordinateRelation.point_map(
        ordinal_domain,
        task_order.source_domain,
        (
            (
                (
                    (
                        ordinal_axis,
                        ordinal_begin,
                        ordinal_begin + task_count,
                        1,
                    ),
                ),
                tuple(local_coordinates),
            ),
        ),
    )
    local_ordinal: sympy.Expr = sympy.sympify(ordinal_begin)
    local_stride: sympy.Expr = sympy.Integer(1)
    for axis in source_axis_order:
        local_ordinal = sympy.simplify(
            local_ordinal + coordinate_axis_symbol(axis) * local_stride
        )
        local_stride = sympy.simplify(
            local_stride * task_order.source_domain.axis_count_expressions[axis]
        )
    local_to_ordinal = CoordinateRelation.point_map(
        task_order.source_domain,
        ordinal_domain,
        (
            (
                tuple(
                    (
                        axis,
                        0,
                        task_order.source_domain.axis_count_expressions[axis],
                        1,
                    )
                    for axis in task_order.source_domain.axis_order
                ),
                (local_ordinal,),
            ),
        ),
    )
    tile_dependency._remember_exact_converse(ordinal_to_local, local_to_ordinal)
    composed = ordinal_to_local.then(task_order)
    if (
        composed is not None
        and tile_dependency._memoized_exact_converse(composed) is not None
    ):
        return composed

    # A compact multidimensional PID order can partition its fastest digit
    # into equal adjacent ranges with a static target offset.  Scalar
    # flattening turns that guard into a periodic set that the ordinary point
    # composition cannot express.  Fold it only as a rescue after the native
    # forms fail, and accept the rewrite only when its total point map remains
    # symbolically proved.
    compact_task_order = task_order.coalesce_adjacent_source_boxes(
        fold_static_offsets=True
    )
    if (
        compact_task_order != task_order
        and compact_task_order.is_total_function()
        and compact_task_order.converse() is not None
        and (composed := ordinal_to_local.then(compact_task_order)) is not None
        and tile_dependency._memoized_exact_converse(composed) is not None
    ):
        return composed

    if task_order.parameter_symbols or ordinal_domain.parameter_symbols:
        # The fallback below expands only static relation boxes.  Dynamic
        # traversals must stay in the exact point-composition paths above.
        return None

    # A piecewise PID order can constrain a fast source digit while leaving
    # slower digits free.  Its flattened support is then a bounded union of
    # arithmetic progressions.  Derive those progressions from relation boxes;
    # never enumerate logical tasks.
    source_strides: dict[int, int] = {}
    stride = 1
    for axis in task_order.source_domain.axis_order:
        source_strides[axis] = stride
        stride *= task_order.source_domain.axis_counts[axis]
    flattened_pieces: list[
        tuple[tuple[tuple[int, int, int, int], ...], tuple[sympy.Expr, ...]]
    ] = []
    ordinal_symbol = coordinate_axis_symbol(ordinal_axis)
    for piece in task_order.pieces:
        bounds = {
            axis: (begin, end, step)
            for axis, begin, end, step in piece.source_bounds_items
        }
        varying_axis = max(
            task_order.source_domain.axis_order,
            key=lambda axis: len(range(*bounds[axis])),
        )
        fixed_axes = tuple(
            axis for axis in task_order.source_domain.axis_order if axis != varying_axis
        )
        fixed_count = math.prod(len(range(*bounds[axis])) for axis in fixed_axes)
        if fixed_count > tile_dependency._MAX_RELATION_PIECES - len(flattened_pieces):
            return None
        varying_begin, varying_end, varying_step = bounds[varying_axis]
        varying_values = range(varying_begin, varying_end, varying_step)
        if not varying_values:
            continue
        for fixed_values in itertools.product(
            *(range(*bounds[axis]) for axis in fixed_axes)
        ):
            fixed_coordinates = dict(zip(fixed_axes, fixed_values, strict=True))
            flat_begin = (
                sum(
                    fixed_coordinates[axis] * source_strides[axis]
                    for axis in fixed_axes
                )
                + varying_begin * source_strides[varying_axis]
            )
            flat_step = varying_step * source_strides[varying_axis]
            flat_end = flat_begin + (len(varying_values) - 1) * flat_step + 1
            substitutions = {
                coordinate_axis_symbol(axis): sympy.Integer(value)
                for axis, value in fixed_coordinates.items()
            }
            substitutions[coordinate_axis_symbol(varying_axis)] = (
                varying_begin  # pyrefly: ignore[unsupported-operation]
                + sympy.floor(  # pyrefly: ignore[bad-argument-type]
                    (ordinal_symbol - ordinal_begin - flat_begin)  # pyrefly: ignore[unsupported-operation]
                    / source_strides[varying_axis]
                )
            )
            target_expressions: list[sympy.Expr] = []
            for _axis, begin, end, target_step in piece.target_ranges:
                if (
                    target_step != 1 or sympy.simplify(end - begin) != 1  # pyrefly: ignore[unsupported-operation]
                ):
                    return None
                target_expressions.append(
                    cast("sympy.Expr", begin.xreplace(substitutions))
                )
            flattened_pieces.append(
                (
                    (
                        (
                            ordinal_axis,
                            ordinal_begin + flat_begin,
                            ordinal_begin + flat_end,
                            flat_step,
                        ),
                    ),
                    tuple(target_expressions),
                )
            )
    result = CoordinateRelation.point_map(
        ordinal_domain,
        task_order.target_domain,
        tuple(flattened_pieces),
    )
    result_converse = task_order_converse.then(local_to_ordinal)
    if result_converse is None:
        return None
    tile_dependency._remember_exact_converse(result, result_converse)
    if not result.is_total_function():
        return None
    return result


def _concrete_task_order_slice(
    task_order: CoordinateRelation,
    begin: int,
    count: int,
) -> CoordinateRelation | None:
    """Retain the bounded concrete slice fallback for manual relations."""

    def retain_exact_converse(
        relation: CoordinateRelation,
    ) -> CoordinateRelation | None:
        converse = tile_dependency._memoized_exact_converse(relation)
        if converse is None:
            converse = relation.derive_converse_and_target_counts()[0]
            if converse is not None:
                tile_dependency._remember_exact_converse(relation, converse)
        return converse

    if begin < 0 or count <= 0 or begin + count > task_order.source_domain.size:
        return None
    if begin == 0 and count == task_order.source_domain.size:
        return task_order if retain_exact_converse(task_order) is not None else None
    task_order_converse = retain_exact_converse(task_order)
    if task_order_converse is None:
        return None

    # Preserve the configured Cartesian task-order domain when the flat slice
    # is one rectangle.  Besides producing much smaller formulas, this keeps
    # piecewise PID permutations invertible for exact-once validation.
    source_axes = task_order.source_domain.axis_order
    source_counts = task_order.source_domain.axis_counts
    begin_coordinates = task_order.source_domain.coordinates(begin)
    source_stride = 1
    for split_index, split_axis in enumerate(source_axes):
        split_count = source_counts[split_axis]
        if (
            begin % source_stride
            or count % source_stride
            or (span := count // source_stride) > split_count
            or begin_coordinates[split_axis] + span > split_count
        ):
            source_stride *= split_count
            continue
        rectangular_domain = CoordinateDomain(
            axis_order=source_axes,
            axis_counts_items=tuple(
                (
                    axis,
                    source_counts[axis]
                    if axis in source_axes[:split_index]
                    else span
                    if axis == split_axis
                    else 1,
                )
                for axis in source_axes
            ),
            kind="task_order",
        )
        rectangular_to_source = CoordinateRelation.point_map(
            rectangular_domain,
            task_order.source_domain,
            (
                (
                    tuple(
                        (axis, 0, rectangular_domain.axis_counts[axis], 1)
                        for axis in source_axes
                    ),
                    tuple(
                        coordinate_axis_symbol(axis) + begin_coordinates[axis]  # pyrefly: ignore[unsupported-operation]
                        for axis in source_axes
                    ),
                ),
            ),
        )
        rectangular_converse = CoordinateRelation.point_map(
            task_order.source_domain,
            rectangular_domain,
            (
                (
                    tuple(
                        (
                            axis,
                            begin_coordinates[axis],
                            begin_coordinates[axis]
                            + rectangular_domain.axis_counts[axis],
                            1,
                        )
                        for axis in source_axes
                    ),
                    tuple(
                        coordinate_axis_symbol(axis) - begin_coordinates[axis]  # pyrefly: ignore[unsupported-operation]
                        for axis in source_axes
                    ),
                ),
            ),
        )
        tile_dependency._remember_exact_converse(
            rectangular_to_source,
            rectangular_converse,
        )
        rectangular = rectangular_to_source.then(task_order)
        if rectangular is not None and retain_exact_converse(rectangular) is not None:
            return rectangular
        source_stride *= split_count

    axes = (
        *task_order.source_domain.axis_order,
        *task_order.target_domain.axis_order,
    )
    slice_axis = max(axes, default=0) + 1
    slice_domain = CoordinateDomain(
        axis_order=(slice_axis,),
        axis_counts_items=((slice_axis, count),),
        kind="task_order",
    )
    flat_index = coordinate_axis_symbol(slice_axis) + begin  # pyrefly: ignore[unsupported-operation]
    coordinates: list[sympy.Expr] = []
    stride = 1
    source_cuts = {0, count}
    for axis in task_order.source_domain.axis_order:
        axis_count = task_order.source_domain.axis_counts[axis]
        coordinates.append(
            cast(
                "sympy.Expr",
                sympy.Mod(
                    sympy.floor(flat_index / stride),  # pyrefly: ignore[unsupported-operation]
                    axis_count,
                ),
            )
        )
        if axis_count > 1:
            period = stride * axis_count
            boundary = (begin // period + 1) * period
            boundary_count = max(0, (begin + count - 1 - boundary) // period + 1)
            if boundary_count > tile_dependency._MAX_RELATION_PIECES - len(
                source_cuts
            ):
                return None
            while boundary < begin + count:
                source_cuts.add(boundary - begin)
                boundary += period
        stride *= axis_count
    slice_to_source = CoordinateRelation.point_map(
        slice_domain,
        task_order.source_domain,
        tuple(
            (
                ((slice_axis, piece_begin, piece_end, 1),),
                tuple(coordinates),
            )
            for piece_begin, piece_end in itertools.pairwise(sorted(source_cuts))
        ),
    )
    source_to_slice = retain_exact_converse(slice_to_source)
    if source_to_slice is None:
        return None
    result = slice_to_source.then(task_order)
    if result is not None and retain_exact_converse(result) is not None:
        return result

    # A configured task order can be piecewise over a non-leading source axis.
    # A flat slice then crosses those pieces periodically rather than at one
    # contiguous cut (for example, eight tasks from each half of a 16-wide
    # axis).  Partition the bounded slice into affine progressions aligned to
    # the existing pieces.  This remains symbolic: it enumerates relation
    # boxes, not logical tasks or target coordinates.
    source_strides: dict[int, int] = {}
    stride = 1
    for axis in task_order.source_domain.axis_order:
        source_strides[axis] = stride
        stride *= task_order.source_domain.axis_counts[axis]
    aligned_pieces: list[
        tuple[tuple[tuple[int, int, int, int], ...], tuple[sympy.Expr, ...]]
    ] = []
    slice_end = begin + count
    for task_order_piece in task_order.pieces:
        if any(
            step != 1 or sympy.simplify(end - piece_begin) != 1
            for _axis, piece_begin, end, step in task_order_piece.target_ranges
        ):
            return None
        substitutions = {
            coordinate_axis_symbol(axis): expression
            for axis, expression in zip(
                task_order.source_domain.axis_order,
                coordinates,
                strict=True,
            )
        }
        target_expressions = tuple(
            cast("sympy.Expr", piece_begin.xreplace(substitutions))
            for _axis, piece_begin, _end, _step in task_order_piece.target_ranges
        )
        bounds = {
            axis: (piece_begin, piece_end, piece_step)
            for axis, piece_begin, piece_end, piece_step in (
                task_order_piece.source_bounds_items
            )
        }
        varying_axis = max(
            task_order.source_domain.axis_order,
            key=lambda axis: len(range(*bounds[axis])),
        )
        fixed_axes = tuple(
            axis for axis in task_order.source_domain.axis_order if axis != varying_axis
        )
        varying_begin, varying_end, varying_step = bounds[varying_axis]
        varying_count = len(range(varying_begin, varying_end, varying_step))
        if not varying_count:
            continue
        fixed_value_count = 1
        for axis in fixed_axes:
            fixed_value_count *= len(range(*bounds[axis]))
        if fixed_value_count > tile_dependency._MAX_RELATION_PIECES - len(
            aligned_pieces
        ):
            return None
        for fixed_values in itertools.product(
            *(range(*bounds[axis]) for axis in fixed_axes)
        ):
            fixed_coordinates = dict(zip(fixed_axes, fixed_values, strict=True))
            flat_begin = (
                sum(
                    fixed_coordinates[axis] * source_strides[axis]
                    for axis in fixed_axes
                )
                + varying_begin * source_strides[varying_axis]
            )
            flat_step = varying_step * source_strides[varying_axis]
            first_position = max(0, (begin - flat_begin + flat_step - 1) // flat_step)
            final_position = min(
                varying_count,
                (slice_end - flat_begin + flat_step - 1) // flat_step,
            )
            if first_position >= final_position:
                continue
            first_flat = flat_begin + first_position * flat_step
            final_flat = flat_begin + (final_position - 1) * flat_step
            aligned_pieces.append(
                (
                    (
                        (
                            slice_axis,
                            first_flat - begin,
                            final_flat - begin + 1,
                            flat_step,
                        ),
                    ),
                    target_expressions,
                )
            )
    result = CoordinateRelation.point_map(
        slice_domain,
        task_order.target_domain,
        tuple(aligned_pieces),
    )
    result_converse = task_order_converse.then(source_to_slice)
    if result_converse is None:
        result_converse = retain_exact_converse(result)
        if result_converse is None:
            return None
    tile_dependency._remember_exact_converse(result, result_converse)
    return result


def _task_order_slice(
    task_order: CoordinateRelation,
    ordinal_begin: int | sympy.Expr,
    task_count: int | sympy.Expr,
) -> CoordinateRelation | None:
    """Return one exact dense ordinal slice of a task traversal."""
    ordinal_begin = sympy.simplify(sympy.sympify(ordinal_begin))
    task_count = sympy.simplify(sympy.sympify(task_count))
    if ordinal_begin.is_integer is not True or task_count.is_integer is not True:
        return None

    # Preserve the established bounded construction for wholly concrete
    # relations.  In particular, manually assembled piecewise orders can use
    # its relation-box fallback without changing their representation.
    if (
        not task_order.parameter_symbols
        and not ordinal_begin.free_symbols
        and not task_count.free_symbols
    ):
        if not isinstance(ordinal_begin, sympy.Integer) or not isinstance(
            task_count, sympy.Integer
        ):
            return None
        return _concrete_task_order_slice(
            task_order,
            int(ordinal_begin),
            int(task_count),
        )

    ordinal_count = task_order.source_domain.size_expr
    if (
        not tile_dependency._is_provably_nonnegative(ordinal_begin, None)
        or not tile_dependency._is_provably_nonnegative(task_count, None)
        or not tile_dependency._is_provably_nonnegative(
            sympy.simplify(ordinal_count - ordinal_begin - task_count),
            None,
        )
    ):
        return None

    if _equal_integer_expressions(ordinal_begin, 0) and _equal_integer_expressions(
        task_count,
        ordinal_count,
    ):
        return task_order if task_order.converse() is not None else None

    slice_axis = (
        max(
            (
                *task_order.source_domain.axis_order,
                *task_order.target_domain.axis_order,
            ),
            default=0,
        )
        + 2
    )
    slice_domain = CoordinateDomain(
        axis_order=(slice_axis,),
        axis_counts_items=((slice_axis, task_count),),
        kind="task_order",
        _allow_empty=task_count.is_zero is True,
    )
    if task_count.is_zero is True:
        result = CoordinateRelation(
            source_domain=slice_domain,
            target_domain=task_order.target_domain,
            pieces=(),
        )
        converse = CoordinateRelation(
            source_domain=task_order.target_domain,
            target_domain=slice_domain,
            pieces=(),
        )
        tile_dependency._remember_exact_converse(result, converse)
        return result

    ordinal_domain = _task_order_ordinal_domain(task_order)
    flat_task_order = _flat_task_order_relation(task_order, ordinal_domain)
    if flat_task_order is None:
        return None
    (ordinal_axis,) = ordinal_domain.axis_order
    slice_ordinal = coordinate_axis_symbol(slice_axis)
    ordinal = coordinate_axis_symbol(ordinal_axis)
    slice_to_ordinal = CoordinateRelation.point_map(
        slice_domain,
        ordinal_domain,
        (
            (
                ((slice_axis, 0, task_count, 1),),
                (slice_ordinal + ordinal_begin,),  # pyrefly: ignore[unsupported-operation]
            ),
        ),
    )
    # Keep the inverse's source box rectangular and let the slice target
    # domain clip ordinals outside the selected interval.  This is the exact
    # converse of the translation above, and lets ordinary point composition
    # restrict mixed-radix logical coordinates without expanding a symbolic
    # prefix into runtime-many Cartesian boxes.
    ordinal_to_slice = CoordinateRelation.point_map(
        ordinal_domain,
        slice_domain,
        (
            (
                ((ordinal_axis, 0, ordinal_count, 1),),
                (ordinal - ordinal_begin,),  # pyrefly: ignore[unsupported-operation]
            ),
        ),
    )
    tile_dependency._remember_exact_converse(slice_to_ordinal, ordinal_to_slice)
    result = slice_to_ordinal.then(flat_task_order)
    if result is None or tile_dependency._memoized_exact_converse(result) is None:
        return None
    return result


def _equal_integer_expressions(
    left: int | sympy.Expr,
    right: int | sympy.Expr,
) -> bool:
    """Compare two integer expressions without choosing parameter values."""
    difference = sympy.Add(
        sympy.sympify(left),
        sympy.Mul(-1, sympy.sympify(right)),
    )
    return sympy.simplify(difference) == 0


def _ceildiv_nonnegative_expression(
    numerator: int | sympy.Expr,
    denominator: int,
) -> sympy.Expr:
    """Return exact integer ceildiv for a nonnegative parameter expression."""
    return cast("sympy.Expr", CeilDiv(sympy.sympify(numerator), denominator))


def _parametric_task_axis_order(
    task_order: CoordinateRelation,
    domain: CoordinateDomain,
) -> tuple[int, ...] | None:
    """Return the exact ungrouped PID-axis order of one symbolic root."""
    axis_order = task_order.source_domain.axis_order
    if set(axis_order) != set(domain.axis_order) or task_order != pid_task_order(
        domain,
        axis_order,
    ):
        return None
    return axis_order


def _parametric_root_major_relation(
    schedule_domain: CoordinateDomain,
    target_domain: CoordinateDomain,
    first_slot: sympy.Expr,
    worker_count: int,
    task_axis_order: tuple[int, ...] | None = None,
) -> CoordinateRelation:
    """Return one root's exact placement in a packed root-major stream."""
    launch_stage_axis, worker_axis, wave_axis = schedule_domain.axis_order
    worker = coordinate_axis_symbol(worker_axis)
    wave = coordinate_axis_symbol(wave_axis)
    task_count = target_domain.size_expr
    if task_count.is_zero is True:
        relation = CoordinateRelation(
            source_domain=schedule_domain,
            target_domain=target_domain,
            pieces=(),
        )
        converse = CoordinateRelation(
            source_domain=target_domain,
            target_domain=schedule_domain,
            pieces=(),
        )
        tile_dependency._remember_exact_converse(relation, converse)
        return relation
    first_wave = cast("sympy.Expr", FloorDiv(first_slot, worker_count))
    first_worker = sympy.Mod(first_slot, worker_count)
    first_count = SymbolicMin(task_count, worker_count - first_worker)
    remaining = SymbolicMax(sympy.simplify(task_count - first_count), 0)
    full_waves = cast("sympy.Expr", FloorDiv(remaining, worker_count))
    tail_count = sympy.Mod(remaining, worker_count)
    middle_wave_begin = sympy.simplify(first_wave + 1)
    middle_wave_end = sympy.simplify(middle_wave_begin + full_waves)
    final_wave_end = sympy.simplify(
        middle_wave_end + _ceildiv_nonnegative_expression(tail_count, worker_count)
    )
    logical_task = sympy.simplify(wave * worker_count + worker - first_slot)
    task_axis_order = (
        target_domain.axis_order if task_axis_order is None else task_axis_order
    )
    if set(task_axis_order) != set(target_domain.axis_order):
        raise ValueError("task axis order must permute the target axes")
    logical_coordinates: dict[int, sympy.Expr] = {}
    stride: sympy.Expr = sympy.Integer(1)
    for index, axis in enumerate(task_axis_order):
        count = sympy.sympify(target_domain.axis_count_expressions[axis])
        quotient = (
            logical_task
            if stride == 1
            else cast("sympy.Expr", FloorDiv(logical_task, stride))
        )
        logical_coordinates[axis] = (
            quotient
            if index == len(task_axis_order) - 1
            else sympy.Mod(quotient, count)
            if not count.free_symbols
            else sympy.simplify(
                quotient - cast("sympy.Expr", FloorDiv(quotient, count)) * count
            )
        )
        stride = sympy.simplify(stride * count)
    target_coordinates = tuple(
        logical_coordinates[axis] for axis in target_domain.axis_order
    )
    aligned_full_waves = cast("sympy.Expr", FloorDiv(task_count, worker_count))
    is_wave_aligned = _equal_integer_expressions(first_worker, 0) and (
        _equal_integer_expressions(
            aligned_full_waves * worker_count,
            task_count,
        )
    )
    source_pieces = (
        (
            (
                (
                    launch_stage_axis,
                    _RESIDENT_LAUNCH_STAGE,
                    _RESIDENT_LAUNCH_STAGE + 1,
                    1,
                ),
                (worker_axis, 0, worker_count, 1),
                (
                    wave_axis,
                    first_wave,
                    sympy.simplify(first_wave + aligned_full_waves),
                    1,
                ),
            ),
            target_coordinates,
        ),
    ) if is_wave_aligned else (
        (
            (
                (
                    launch_stage_axis,
                    _RESIDENT_LAUNCH_STAGE,
                    _RESIDENT_LAUNCH_STAGE + 1,
                    1,
                ),
                (
                    worker_axis,
                    first_worker,
                    first_worker + first_count,
                    1,
                ),
                (
                    wave_axis,
                    first_wave,
                    sympy.simplify(first_wave + 1),
                    1,
                ),
            ),
            target_coordinates,
        ),
        (
            (
                (
                    launch_stage_axis,
                    _RESIDENT_LAUNCH_STAGE,
                    _RESIDENT_LAUNCH_STAGE + 1,
                    1,
                ),
                (worker_axis, 0, worker_count, 1),
                (wave_axis, middle_wave_begin, middle_wave_end, 1),
            ),
            target_coordinates,
        ),
        (
            (
                (
                    launch_stage_axis,
                    _RESIDENT_LAUNCH_STAGE,
                    _RESIDENT_LAUNCH_STAGE + 1,
                    1,
                ),
                (worker_axis, 0, tail_count, 1),
                (wave_axis, middle_wave_end, final_wave_end, 1),
            ),
            target_coordinates,
        ),
    )
    relation = CoordinateRelation.point_map(
        schedule_domain,
        target_domain,
        source_pieces,
    )
    relation = dataclasses.replace(
        relation,
        pieces=tuple(
            piece
            for piece in relation.pieces
            if not any(
                sympy.simplify(end - begin).is_nonpositive is True  # pyrefly: ignore[unsupported-operation]
                for _axis, begin, end, _step in piece.source_bounds_items
            )
        ),
    )
    task_ordinal: sympy.Expr = sympy.Integer(0)
    stride = sympy.Integer(1)
    for axis in task_axis_order:
        task_ordinal = sympy.simplify(
            task_ordinal + coordinate_axis_symbol(axis) * stride
        )
        stride = sympy.simplify(stride * target_domain.axis_count_expressions[axis])
    global_slot = sympy.simplify(first_slot + task_ordinal)
    converse = CoordinateRelation.point_map(
        target_domain,
        schedule_domain,
        (
            (
                tuple(
                    (
                        axis,
                        0,
                        target_domain.axis_count_expressions[axis],
                        1,
                    )
                    for axis in target_domain.axis_order
                ),
                (
                    sympy.Integer(_RESIDENT_LAUNCH_STAGE),
                    sympy.Mod(global_slot, worker_count),
                    cast("sympy.Expr", FloorDiv(global_slot, worker_count)),
                ),
            ),
        ),
    )
    tile_dependency._remember_exact_converse(relation, converse)
    return relation


def _packed_ordinal_slice_relation(
    schedule_domain: CoordinateDomain,
    ordinal_domain: CoordinateDomain,
    first_slot: sympy.Expr,
    worker_count: int,
    ordinal_begin: int | sympy.Expr,
    ordinal_end: int | sympy.Expr,
) -> CoordinateRelation | None:
    """Place one contiguous task-order interval in the packed slot stream."""
    interval_count = sympy.simplify(ordinal_end - ordinal_begin)  # pyrefly: ignore[unsupported-operation]
    if len(ordinal_domain.axis_order) != 1 or interval_count.is_nonnegative is not True:
        return None
    (ordinal_axis,) = ordinal_domain.axis_order
    interval_domain = CoordinateDomain(
        axis_order=(ordinal_axis,),
        axis_counts_items=((ordinal_axis, interval_count),),
        kind="task_order",
        identity=ordinal_domain.identity,
    )
    local = _parametric_root_major_relation(
        schedule_domain,
        interval_domain,
        sympy.simplify(first_slot + ordinal_begin),
        worker_count,
        interval_domain.axis_order,
    )
    interval_coordinate = coordinate_axis_symbol(ordinal_axis)
    translation = CoordinateRelation.point_map(
        interval_domain,
        ordinal_domain,
        (
            (
                ((ordinal_axis, 0, interval_count, 1),),
                (interval_coordinate + ordinal_begin,),  # pyrefly: ignore[unsupported-operation]
            ),
        ),
    )
    translation_converse = CoordinateRelation.point_map(
        ordinal_domain,
        interval_domain,
        (
            (
                ((ordinal_axis, ordinal_begin, ordinal_end, 1),),
                (interval_coordinate - ordinal_begin,),  # pyrefly: ignore[unsupported-operation]
            ),
        ),
    )
    tile_dependency._remember_exact_converse(translation, translation_converse)
    return local.then(translation)


def _packed_root_major_task_order_relation(
    schedule_domain: CoordinateDomain,
    task_order: CoordinateRelation,
    first_slot: sympy.Expr,
    worker_count: int,
) -> CoordinateRelation | None:
    """Compose packed schedule slots with one configured logical traversal."""
    if (
        not task_order.is_bijection_from_source_support()
        or task_order.converse() is None
    ):
        return None
    packed_source = _parametric_root_major_relation(
        schedule_domain,
        task_order.source_domain,
        first_slot,
        worker_count,
        task_order.source_domain.axis_order,
    )
    if (
        (composed := packed_source.then(task_order)) is not None
        and tile_dependency._memoized_exact_converse(composed) is not None
    ):
        return composed

    # Piecewise orders such as L2 grouping may guard only an interval of their
    # scalar traversal. Align the packed relation to those existing pieces so
    # point composition can prove each preimage directly. This loop scales
    # with relation complexity, never with tasks, workers, or waves.
    ordinal_domain = _task_order_ordinal_domain(task_order)
    flat_task_order = _flat_task_order_relation(task_order, ordinal_domain)
    if flat_task_order is None:
        return None
    if not tile_dependency._relation_product_is_within_budget(
        len(flat_task_order.pieces),
        3,
    ):
        return None
    (ordinal_axis,) = ordinal_domain.axis_order
    packed_order: CoordinateRelation | None = None
    for piece in flat_task_order.pieces:
        if len(piece.source_bounds_items) != 1:
            return None
        axis, begin, end, step = piece.source_bounds_items[0]
        if axis != ordinal_axis or step != 1:
            return None
        packed_slice = _packed_ordinal_slice_relation(
            schedule_domain,
            ordinal_domain,
            first_slot,
            worker_count,
            begin,
            end,
        )
        if packed_slice is None:
            return None
        if (
            packed_order is not None
            and not tile_dependency._relation_product_is_within_budget(
                len(packed_order.pieces),
                len(packed_slice.pieces),
            )
        ):
            return None
        packed_order = (
            packed_slice if packed_order is None else packed_order.union(packed_slice)
        )
        if packed_order is None:
            return None
    if packed_order is None:
        return None
    composed = packed_order.then(flat_task_order)
    return (
        composed
        if composed is not None
        and tile_dependency._memoized_exact_converse(composed) is not None
        else None
    )


def _parametric_root_major_schedule_geometry_from_parts(
    worker_count: int,
    segments: tuple[WorkerScheduleSegment, ...],
) -> tuple[tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr], ...] | None:
    """Recognize and prove the compiler's exact parametric root-major relation.

    Each result item is ``(segment, first_slot, task_count)``. This structural
    proof is shared by schedule validation, barrier ownership, and codegen, so
    none of those consumers reconstructs a schedule from runtime shape hints.
    """
    if not segments:
        return None
    if not all(segment.is_normalized for segment in segments):
        return None
    schedule_domain = segments[0].task_order.source_domain
    if any(segment.task_order.source_domain != schedule_domain for segment in segments):
        return None
    if schedule_domain.kind != "worker" or len(schedule_domain.axis_order) != 3:
        return None
    launch_stage_axis, worker_axis, wave_axis = schedule_domain.axis_order
    schedule_counts = schedule_domain.axis_count_expressions
    if not _equal_integer_expressions(
        schedule_counts[launch_stage_axis], 2
    ) or not _equal_integer_expressions(schedule_counts[worker_axis], worker_count):
        return None

    first_slot: sympy.Expr = sympy.Integer(0)
    result: list[tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr]] = []
    previous_root = -1
    for segment in segments:
        relation = segment.task_order
        target_domain = relation.target_domain
        if (
            segment.root <= previous_root
            or segment.worker_begin != 0
            or segment.worker_count != worker_count
        ):
            return None
        previous_root = segment.root
        task_count = target_domain.size_expr
        statically_empty = task_count.is_zero is True and not relation.pieces
        if statically_empty:
            expected = _parametric_root_major_relation(
                schedule_domain,
                target_domain,
                first_slot,
                worker_count,
                target_domain.axis_order,
            )
            if relation != expected:
                return None
        else:
            if not relation.has_same_source_support(
                _parametric_root_major_relation(
                    schedule_domain,
                    target_domain,
                    first_slot,
                    worker_count,
                    target_domain.axis_order,
                ),
            ):
                return None
        result.append((segment, first_slot, task_count))
        first_slot = sympy.simplify(sympy.Add(first_slot, task_count))

    if not _equal_integer_expressions(
        schedule_counts[wave_axis],
        _ceildiv_nonnegative_expression(first_slot, worker_count),
    ):
        return None
    return tuple(result)


class _WorkerScheduleChronologyError(ValueError):
    """Raised when segment tuple order disagrees with a resident strand."""


def _validate_normalized_worker_schedule(
    worker_count: int,
    segments: tuple[WorkerScheduleSegment, ...],
) -> None:
    """Prove exact ownership from the authoritative schedule relations."""
    if not segments:
        return None
    schedule_domain = segments[0].task_order.source_domain
    if schedule_domain.kind != "worker" or len(schedule_domain.axis_order) != 3:
        raise ValueError("worker schedule has an incompatible placement domain")
    launch_stage_axis, worker_axis, _wave_axis = schedule_domain.axis_order
    counts = schedule_domain.axis_count_expressions
    if not _equal_integer_expressions(counts[launch_stage_axis], 2) or not (
        _equal_integer_expressions(counts[worker_axis], worker_count)
    ):
        raise ValueError("worker schedule has incompatible launch or worker bounds")
    if any(segment.task_order.source_domain != schedule_domain for segment in segments):
        raise ValueError("worker schedule segments do not share one domain")
    total_piece_count = sum(len(segment.task_order.pieces) for segment in segments)
    if total_piece_count > tile_dependency._MAX_RELATION_PIECES or not (
        tile_dependency._relation_product_is_within_budget(
            total_piece_count,
            total_piece_count,
        )
    ):
        raise ValueError("worker schedule exceeds the symbolic relation budget")

    for index, segment in enumerate(segments):
        relation = segment.task_order
        statically_empty = (
            relation.target_domain.size_expr.is_zero is True and not relation.pieces
        )
        if not statically_empty and segment.launch_stage is None:
            raise ValueError("worker schedule segment is not an exact point map")
        if not statically_empty and relation.converse() is None:
            raise ValueError("worker schedule segment has no exact converse")
        if any(
            not relation.has_disjoint_source_support(other.task_order)
            for other in segments[index + 1 :]
        ):
            raise ValueError("worker schedule segment support overlaps")

    for root in sorted({segment.root for segment in segments}):
        root_segments = tuple(segment for segment in segments if segment.root == root)
        target_domain = root_segments[0].task_order.target_domain
        if any(
            segment.task_order.target_domain != target_domain
            for segment in root_segments
        ):
            raise ValueError("one root uses multiple logical task domains")
        combined = root_segments[0].task_order
        for segment in root_segments[1:]:
            combined = combined.union(segment.task_order)
            if combined is None:
                raise ValueError("worker schedule root relation exceeds proof budget")
        if not combined.is_bijection_from_source_support():
            raise ValueError("worker schedule does not own each logical task once")


@dataclasses.dataclass(frozen=True)
class WorkerSchedule:
    """The authoritative ownership and chronology of persistent work.

    Every normalized segment maps the same global schedule coordinate space
    directly to logical tasks.  Tuple order is retained only as a stable
    rendering order for the concrete compatibility lowering; the ``wave``
    coordinate is semantic chronology.
    """

    worker_count: int
    segments: tuple[WorkerScheduleSegment, ...]

    def __getstate__(self) -> dict[str, object]:
        """Serialize semantic fields only, never derived geometry caches."""
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore immutable semantic fields with all derived caches empty."""
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name, state[field.name])

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        input_segments = self.segments
        has_legacy_segments = any(not segment.is_normalized for segment in input_segments)
        normalized_axes = {
            segment.task_order.source_domain.axis_order
            for segment in input_segments
            if segment.is_normalized
        }
        if len(normalized_axes) > 1:
            raise ValueError("worker schedule segments use different schedule axes")
        if input_segments and not has_legacy_segments:
            schedule_domains = {
                segment.task_order.source_domain for segment in input_segments
            }
            if len(schedule_domains) != 1:
                raise ValueError("worker schedule segments do not share one domain")
            (schedule_domain,) = schedule_domains
        else:
            if normalized_axes:
                (schedule_axes,) = normalized_axes
            else:
                minimum_axis = min(
                    (
                        axis
                        for segment in input_segments
                        for domain in (
                            segment.task_order.source_domain,
                            segment.task_order.target_domain,
                        )
                        for axis in domain.axis_order
                    ),
                    default=0,
                )
                schedule_axes = (
                    minimum_axis - 3,
                    minimum_axis - 2,
                    minimum_axis - 1,
                )
            _launch_stage_axis, _worker_axis, wave_axis = schedule_axes
            nonempty_segments = tuple(
                segment
                for segment in input_segments
                if segment.task_count_expr.is_zero is not True
            )
            maximum_wave = max(
                (
                    (
                        segment.dispatch_index(segment.task_count - 1)
                        // segment.worker_count
                    )
                    if not segment.is_normalized
                    else segment.task_order.source_domain.axis_counts[wave_axis] - 1
                    for segment in nonempty_segments
                ),
                default=-1,
            )
            schedule_domain = _worker_schedule_domain(
                self.worker_count,
                maximum_wave + 1,
                schedule_axes,
            )
        normalized_segments = tuple(
            _normalize_dense_schedule_segment(segment, schedule_domain)
            for segment in input_segments
        )
        object.__setattr__(self, "segments", normalized_segments)
        _validate_normalized_worker_schedule(
            self.worker_count,
            normalized_segments,
        )

        # The dense compatibility fields are still used by the legacy concrete
        # renderer.  Check their tuple chronology only when this instance was
        # constructed from that spelling; relation validation above is the
        # semantic proof for every schedule form.
        if not has_legacy_segments:
            return

        prior_runs: list[tuple[int, int, int]] = []
        for segment in self.segments:
            if segment.worker_begin + segment.worker_count > self.worker_count:
                raise ValueError(
                    "worker schedule segment exceeds the resident worker domain"
                )
            for begin, end, first_step, last_step in segment.worker_step_runs():
                for prior_begin, prior_end, prior_last_step in prior_runs:
                    if (
                        max(begin, prior_begin) < min(end, prior_end)
                        and first_step <= prior_last_step
                    ):
                        overlap_begin = max(begin, prior_begin)
                        raise _WorkerScheduleChronologyError(
                            "worker schedule tuple order disagrees with worker-step "
                            f"order for worker {overlap_begin}: {first_step} follows "
                            f"{prior_last_step}"
                        )
                prior_runs.append((begin, end, last_step))

    def root_at(self, worker: int, worker_step: int) -> int | None:
        """Return the task family occupying one resident worker step."""
        roots = tuple(
            segment.root
            for segment in self.segments
            if segment.occupies(worker, worker_step)
        )
        if len(roots) > 1:
            raise AssertionError(
                f"worker {worker} step {worker_step} has multiple tasks"
            )
        return roots[0] if roots else None

    def segments_for_root(self, root: int) -> tuple[WorkerScheduleSegment, ...]:
        """Return the compressed static relation for one task family."""
        return tuple(segment for segment in self.segments if segment.root == root)

    def workers_for_root(self, root: int) -> frozenset[int]:
        """Materialize one root's resident worker support for diagnostics."""
        return frozenset(
            worker
            for begin, end in self.worker_intervals_for_root(root)
            for worker in range(begin, end)
        )

    def worker_intervals_for_root(self, root: int) -> tuple[WorkerInterval, ...]:
        """Return one root's concrete compact resident-worker support.

        Parameterized schedules expose their exact support through
        :class:`RootBarrierPublicationPlan.participant_order` instead of
        pretending that a runtime-rotated cohort is one static interval.
        """
        return root_barrier_publication_plan(self, root).participant_intervals

    def active_worker_count_for_root(self, root: int) -> int | sympy.Expr:
        """Return one root's exact real resident-owner count."""
        return root_barrier_publication_plan(self, root).resident_arrival_count

    @cached_property
    def placement_domain(self) -> CoordinateDomain:
        """The common launch-stage, worker, and wave schedule domain."""
        if self.segments:
            return self.segments[0].task_order.source_domain
        return _worker_schedule_domain(self.worker_count, 1, (-3, -2, -1))

    @cached_property
    def worker_step_domain(self) -> CoordinateDomain:
        """The projected worker-step coordinate used for readiness math."""
        wave_axis = self.placement_domain.axis_order[2]
        wave_count = self.placement_domain.axis_count_expressions[wave_axis]
        return CoordinateDomain(
            axis_order=(wave_axis,),
            axis_counts_items=(
                (
                    wave_axis,
                    wave_count,
                ),
            ),
            kind="value",
            _allow_empty=wave_count.is_zero is True,
        )

    def last_worker_steps_for_root(self, root: int) -> dict[int, int]:
        """Return each participating resident worker's final occupied step."""
        result: dict[int, int] = {}
        for segment in self.segments_for_root(root):
            for (
                worker_begin,
                worker_end,
                _first_wave,
                last_wave,
            ) in segment.worker_step_runs():
                for worker in range(worker_begin, worker_end):
                    result[worker] = max(result.get(worker, -1), last_wave)
        return result

    def worker_step_bounds_for_root(self, root: int) -> tuple[int, int] | None:
        """Return the first and last occupied resident steps for one root."""
        segments = self.segments_for_root(root)
        if not segments:
            return None
        worker_steps = tuple(
            (first_wave, last_wave)
            for segment in segments
            for _worker_begin, _worker_end, first_wave, last_wave in (
                segment.worker_step_runs()
            )
        )
        if not worker_steps:
            return None
        return (
            min(begin for begin, _end in worker_steps),
            max(end for _begin, end in worker_steps),
        )

    def dense_assignment(self, root: int) -> tuple[int, int, int, int] | None:
        """Return one root's dense worker and schedule interval.

        The tuple contains ``worker_begin``, ``worker_count``,
        ``dispatch_offset``, and ``task_count``.  A root split across different
        worker ranges or separated schedule intervals is not dense.
        """
        segments = sorted(
            self.segments_for_root(root),
            key=lambda segment: segment.dispatch_offset,
        )
        if not segments:
            return None
        worker_begin = segments[0].worker_begin
        worker_count = segments[0].worker_count
        dispatch_offset = segments[0].dispatch_offset
        schedule_end = dispatch_offset
        for segment in segments:
            if (
                segment.logical_task_order is None
                or segment.worker_begin != worker_begin
                or segment.worker_count != worker_count
                or segment.dispatch_offset != schedule_end
            ):
                return None
            schedule_end += segment.task_count
        return (
            worker_begin,
            worker_count,
            dispatch_offset,
            schedule_end - dispatch_offset,
        )

    def contiguous_global_interval(self, root: int) -> tuple[int, int] | None:
        """Return one dense global schedule interval without task expansion."""
        assignment = self.dense_assignment(root)
        if assignment is None:
            return None
        worker_begin, worker_count, dispatch_offset, task_count = assignment
        if worker_begin or worker_count != self.worker_count:
            return None
        return dispatch_offset, dispatch_offset + task_count

    def without_roots(self, roots: frozenset[int]) -> WorkerSchedule:
        """Remove complete locally executed families without task expansion."""
        if not roots:
            return self
        return WorkerSchedule(
            worker_count=self.worker_count,
            segments=tuple(
                segment for segment in self.segments if segment.root not in roots
            ),
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


_DERIVED_ROOT_MAJOR_GEOMETRY_ATTRIBUTE = "_derived_root_major_geometry"


def _remember_root_major_schedule_geometry(
    worker_schedule: WorkerSchedule,
    geometry: tuple[tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr], ...],
) -> None:
    """Retain builder-derived root-major geometry outside schedule semantics."""
    if tuple(item[0] for item in geometry) != worker_schedule.segments:
        raise AssertionError("root-major geometry does not cover schedule segments")
    first_slot: sympy.Expr = sympy.Integer(0)
    for segment, actual_first_slot, task_count in geometry:
        if not _equal_integer_expressions(actual_first_slot, first_slot) or not (
            _equal_integer_expressions(
                task_count,
                segment.task_order.target_domain.size_expr,
            )
        ):
            raise AssertionError("root-major geometry disagrees with schedule domains")
        first_slot = sympy.simplify(first_slot + task_count)
    worker_schedule.__dict__[_DERIVED_ROOT_MAJOR_GEOMETRY_ATTRIBUTE] = geometry


def _parametric_root_major_schedule_geometry(
    worker_schedule: WorkerSchedule,
) -> tuple[tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr], ...] | None:
    """Return the proved parameterized schedule geometry, when supported."""
    memoized = worker_schedule.__dict__.get(_DERIVED_ROOT_MAJOR_GEOMETRY_ATTRIBUTE)
    if memoized is not None:
        return cast(
            "tuple[tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr], ...]",
            memoized,
        )
    geometry = _parametric_root_major_schedule_geometry_from_parts(
        worker_schedule.worker_count,
        worker_schedule.segments,
    )
    if geometry is not None:
        _remember_root_major_schedule_geometry(worker_schedule, geometry)
    return geometry


@cache
def _root_task_placement_relation(
    worker_schedule: WorkerSchedule,
    root: int,
) -> CoordinateRelation | None:
    """Map every logical task of one root to its unique schedule slot.

    Segment relations are authoritative.  Their exact converses are combined
    directly rather than rebuilding task order from dispatch arithmetic.
    """
    segments = worker_schedule.segments_for_root(root)
    if not segments:
        return None
    pieces = []
    root_domain = segments[0].task_order.target_domain
    for segment in segments:
        if segment.task_order.target_domain != root_domain:
            return None
        converse = segment.task_order.converse()
        if (
            converse is None
            or converse.source_domain != root_domain
            or converse.target_domain != worker_schedule.placement_domain
            or not converse.is_single_valued()
        ):
            return None
        pieces.extend(converse.pieces)
    result = CoordinateRelation(
        source_domain=root_domain,
        target_domain=worker_schedule.placement_domain,
        pieces=tuple(pieces),
    )
    return result if result.is_total_function() else None


@cache
def _root_task_slot_relation(
    worker_schedule: WorkerSchedule,
    root: int,
) -> CoordinateRelation | None:
    """Map every resident logical task to ``wave * workers + worker``.

    The root placement relation is the sole ownership source.  This derived
    scalar view discards launch stage only after proving that every represented
    task is resident, and it remains bounded by the ordinary relation budget.
    """
    placement = _root_task_placement_relation(worker_schedule, root)
    if (
        placement is None
        or len(placement.pieces) > tile_dependency._MAX_RELATION_PIECES
    ):
        return None
    launch_stage_axis, worker_axis, wave_axis = (
        worker_schedule.placement_domain.axis_order
    )
    wave_count = worker_schedule.placement_domain.axis_count_expressions[wave_axis]
    slot_count = sympy.simplify(worker_schedule.worker_count * wave_count)
    slot_domain = CoordinateDomain(
        axis_order=(wave_axis,),
        axis_counts_items=((wave_axis, slot_count),),
        kind="value",
        _allow_empty=slot_count.is_zero is True,
    )
    pieces: list[
        tuple[
            tuple[tuple[int, int | sympy.Expr, int | sympy.Expr, int], ...],
            tuple[sympy.Expr, ...],
        ]
    ] = []
    for piece in placement.pieces:
        targets = {
            axis: (begin, end, step) for axis, begin, end, step in piece.target_ranges
        }
        if set(targets) != {launch_stage_axis, worker_axis, wave_axis}:
            return None
        if any(
            step != 1 or not _equal_integer_expressions(end - begin, 1)
            for begin, end, step in targets.values()
        ):
            return None
        launch_stage = targets[launch_stage_axis][0]
        if not _equal_integer_expressions(
            launch_stage,
            _RESIDENT_LAUNCH_STAGE,
        ):
            return None
        worker = targets[worker_axis][0]
        wave = targets[wave_axis][0]
        pieces.append(
            (
                piece.source_bounds_items,
                (
                    _simplify_logical_expression(
                        sympy.simplify(wave * worker_schedule.worker_count + worker),
                        domain=placement.source_domain,
                        source_bounds=piece.source_bounds_items,
                    ),
                ),
            )
        )
    result = CoordinateRelation.point_map(
        placement.source_domain,
        slot_domain,
        tuple(pieces),
    )
    return result if result.is_total_function() else None


@cache
def _root_task_wave_relation(
    worker_schedule: WorkerSchedule,
    root: int,
) -> CoordinateRelation | None:
    """Map every logical task of one root to its authoritative wave."""
    # Prefer the compact dense rendering certificate when every segment has
    # one.  This preserves formulas such as ``floor(task / worker_count)``;
    # taking the converse of worker/wave support first would split the same
    # function into one constant piece per wave and can obscure later event
    # frontier calculations.  ``logical_task_order`` has already proved that
    # each certificate exactly matches its authoritative segment support.
    dense_waves: CoordinateRelation | None = None
    dense_supported = True
    for segment in worker_schedule.segments_for_root(root):
        logical_order = segment.logical_task_order
        waves_by_ordinal = segment.logical_task_wave_relation(
            worker_schedule.worker_step_domain
        )
        logical_to_ordinal = None if logical_order is None else logical_order.converse()
        segment_waves = (
            None
            if logical_to_ordinal is None or waves_by_ordinal is None
            else logical_to_ordinal.then(waves_by_ordinal)
        )
        if segment_waves is None:
            dense_supported = False
            break
        dense_waves = (
            segment_waves if dense_waves is None else dense_waves.union(segment_waves)
        )
        if dense_waves is None:
            dense_supported = False
            break
    if dense_supported and dense_waves is not None and dense_waves.is_total_function():
        return dense_waves

    placement = _root_task_placement_relation(worker_schedule, root)
    if placement is None:
        return None
    waves = placement.project_target(worker_schedule.worker_step_domain)
    if waves is None or not waves.is_total_function():
        return None
    full_source_bounds = tuple(
        (axis, 0, waves.source_domain.axis_count_expressions[axis], 1)
        for axis in waves.source_domain.axis_order
    )
    for piece in waves.pieces:
        (_target_axis, begin, end, step) = piece.target_ranges[0]
        if step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
            continue
        candidate = CoordinateRelation.point_map(
            waves.source_domain,
            waves.target_domain,
            ((full_source_bounds, (begin,)),),
        )
        if candidate.is_total_function() and candidate.is_pointwise_equal_to(waves):
            return candidate
    return waves


@cache
def _maximum_value_by_key(
    keys_by_item: CoordinateRelation,
    values_by_item: CoordinateRelation,
) -> CoordinateRelation | None:
    """Map each key to the maximum scalar value of its contributing items."""
    if (
        keys_by_item.source_domain != values_by_item.source_domain
        or len(values_by_item.target_domain.axis_order) != 1
    ):
        return None
    items_by_key = keys_by_item.converse()
    maximum = (
        None
        if items_by_key is None
        else items_by_key.max_target_value_by_source(values_by_item)
    )
    return (
        maximum
        if maximum is not None and maximum.canonical_single_valued() is not None
        else None
    )


@cache
def _maximum_required_value_by_consumer(
    keys_by_consumer: CoordinateRelation,
    value_by_key: CoordinateRelation,
) -> CoordinateRelation | None:
    """Map each consumer item to its maximum required scalar key value."""
    values_by_consumer = keys_by_consumer.then(value_by_key)
    if values_by_consumer is None:
        # Set-valued key requirements need not compose into another relation,
        # but their extrema can still be derived directly from the same two
        # relations.
        maximum = keys_by_consumer.max_target_value_by_source(value_by_key)
    else:
        identity = CoordinateRelation.identity(
            value_by_key.target_domain,
            value_by_key.target_domain,
        )
        maximum = values_by_consumer.max_target_value_by_source(identity)
    return (
        maximum
        if maximum is not None and maximum.canonical_single_valued() is not None
        else None
    )


def _strict_root_slot_progress(
    worker_schedule: WorkerSchedule,
    dependencies: tuple[tuple[int, int, CoordinateRelation], ...],
) -> bool:
    """Prove root-level dependency edges strictly increase global slot rank.

    Each relation maps a consumer root task to all producer root tasks that it
    waits for.  Empty support is allowed; otherwise the exact maximum producer
    slot must precede the owning consumer slot.  Worker-strand edges also
    increase this rank by construction, so the proof is sufficient for
    same-wave lower-worker admission without enumerating workers or tasks.
    """
    slots_by_root: dict[int, CoordinateRelation] = {}
    for producer_root, consumer_root, producers_by_consumer in dependencies:
        producer_slots = slots_by_root.get(producer_root)
        if producer_slots is None:
            producer_slots = _root_task_slot_relation(
                worker_schedule,
                producer_root,
            )
            if producer_slots is None:
                return False
            slots_by_root[producer_root] = producer_slots
        consumer_slots = slots_by_root.get(consumer_root)
        if consumer_slots is None:
            consumer_slots = _root_task_slot_relation(
                worker_schedule,
                consumer_root,
            )
            if consumer_slots is None:
                return False
            slots_by_root[consumer_root] = consumer_slots
        if (
            producers_by_consumer.source_domain != consumer_slots.source_domain
            or producers_by_consumer.target_domain != producer_slots.source_domain
        ):
            return False
        latest_producer_slot = producers_by_consumer.max_target_value_by_source(
            producer_slots
        )
        if latest_producer_slot is None:
            return False
        if not latest_producer_slot.pieces:
            support_cardinality = producers_by_consumer.source_support_cardinality()
            if support_cardinality is None or not _equal_integer_expressions(
                support_cardinality,
                0,
            ):
                return False
            continue
        if not latest_producer_slot.is_pointwise_strictly_less_than_where_defined(
            consumer_slots
        ):
            return False
    return True


@cache
def _maximum_root_wave_by_key(
    worker_schedule: WorkerSchedule,
    root: int,
    keys_by_task: CoordinateRelation,
) -> CoordinateRelation | None:
    """Map each emitted key to its latest producer wave from schedule support."""
    segments = worker_schedule.segments_for_root(root)
    if not segments or any(
        segment.task_order.target_domain != keys_by_task.source_domain
        for segment in segments
    ):
        return None
    task_waves = _root_task_wave_relation(worker_schedule, root)
    direct = (
        None if task_waves is None else _maximum_value_by_key(keys_by_task, task_waves)
    )
    if direct is not None:
        return direct
    identity = CoordinateRelation.identity(
        worker_schedule.worker_step_domain,
        worker_schedule.worker_step_domain,
    )
    maxima: list[CoordinateRelation] = []
    for segment in segments:
        keys_by_slot = segment.task_order.then(keys_by_task)
        slots_by_key = None if keys_by_slot is None else keys_by_slot.converse()
        waves_by_key = (
            None
            if slots_by_key is None
            else slots_by_key.project_target(worker_schedule.worker_step_domain)
        )
        maximum = (
            None
            if waves_by_key is None
            else waves_by_key.max_target_value_by_source(identity)
        )
        if maximum is None:
            return None
        maxima.append(maximum)
    combined = maxima[0]
    for maximum in maxima[1:]:
        combined = combined.union(maximum)
        if combined is None:
            return None
    return combined.max_target_value_by_source(identity)


@dataclasses.dataclass(frozen=True)
class RootBarrierPublication:
    """Workers that publish one root after a particular segment occurrence."""

    segment_index: int
    worker_intervals: tuple[WorkerInterval, ...]


@dataclasses.dataclass(frozen=True)
class RootBarrierPublicationPlan:
    """The sole derivation of root-barrier ownership and arrival mass.

    ``participant_order`` is an exact worker-to-dense-owner relation when the
    participant cohort depends on runtime parameters.  Concrete schedules use
    ``participant_intervals`` as a proved strength reduction.  Every emission
    site contributes ``unit_contribution``; code generation must consume the
    counts and support here rather than reconstructing schedule geometry.
    """

    root: int
    participant_intervals: tuple[WorkerInterval, ...]
    participant_order: CoordinateRelation | None
    publications: tuple[RootBarrierPublication, ...]
    resident_arrival_count: int | sympy.Expr
    continuation_arrival_count: int | sympy.Expr
    source_stage_arrival_count: int
    real_arrival_count: int | sympy.Expr
    effective_arrival_count: int | sympy.Expr
    maximum_arrival_count: int
    unit_contribution: int = 1

    def __post_init__(self) -> None:
        if self.unit_contribution != 1:
            raise ValueError("root barriers require unit publication contributions")
        expected_real = sympy.simplify(
            sympy.Add(
                sympy.sympify(self.resident_arrival_count),
                sympy.sympify(self.continuation_arrival_count),
                self.source_stage_arrival_count,
            )
        )
        if not _equal_integer_expressions(self.real_arrival_count, expected_real):
            raise ValueError("root-barrier real arrival count is inconsistent")
        if any(
            sympy.sympify(count).is_nonnegative is not True
            for count in (
                self.resident_arrival_count,
                self.continuation_arrival_count,
                self.source_stage_arrival_count,
                self.real_arrival_count,
                self.effective_arrival_count,
            )
        ):
            raise ValueError("root-barrier arrival counts must be nonnegative")
        expected_effective = SymbolicMax(
            sympy.Integer(1), sympy.sympify(self.real_arrival_count)
        )
        if not _equal_integer_expressions(
            self.effective_arrival_count,
            expected_effective,
        ):
            raise ValueError(
                "root-barrier effective arrival count must be max(real, 1)"
            )
        if self.maximum_arrival_count <= 0:
            raise ValueError("root-barrier maximum arrival count must be positive")
        if not _is_provably_at_most(
            self.effective_arrival_count,
            self.maximum_arrival_count,
        ):
            raise ValueError("root-barrier arrival count exceeds its epoch bound")
        if self.participant_order is None:
            return
        if self.continuation_arrival_count != 0 or self.source_stage_arrival_count:
            raise ValueError(
                "symbolic resident ownership cannot overlap external publication"
            )
        if not _equal_integer_expressions(
            self.participant_order.target_domain.size_expr,
            self.effective_arrival_count,
        ):
            raise ValueError("participant order does not cover effective arrivals")
        if not self.participant_order.is_bijection_from_source_support():
            raise ValueError("participant order is not an exact support bijection")

    @property
    def parameter_symbols(self) -> frozenset[sympy.Symbol]:
        """Return every runtime parameter used by publication semantics."""
        symbols: set[sympy.Symbol] = set()
        if self.participant_order is not None:
            symbols.update(self.participant_order.parameter_symbols)
        for count in (
            self.resident_arrival_count,
            self.continuation_arrival_count,
            self.real_arrival_count,
            self.effective_arrival_count,
        ):
            symbols.update(cast("set[sympy.Symbol]", sympy.sympify(count).free_symbols))
        return frozenset(symbols)


def _concrete_or_symbolic_integer(expression: int | sympy.Expr) -> int | sympy.Expr:
    """Keep symbolic integers symbolic while normalizing exact constants."""
    simplified = sympy.simplify(sympy.sympify(expression))
    return int(simplified) if isinstance(simplified, sympy.Integer) else simplified


def _is_provably_at_most(expression: int | sympy.Expr, upper_bound: int) -> bool:
    """Prove a small integer-expression upper bound without sampling it."""
    expression = sympy.simplify(sympy.sympify(expression))
    difference = sympy.simplify(sympy.Integer(upper_bound) - expression)
    if difference.is_nonnegative is True:
        return True
    if expression.func in (sympy.Min, SymbolicMin):
        return any(
            _is_provably_at_most(argument, upper_bound) for argument in expression.args
        )
    if expression.func in (sympy.Max, SymbolicMax):
        return all(
            _is_provably_at_most(argument, upper_bound) for argument in expression.args
        )
    return False


def _root_major_participant_order_from_geometry(
    segment: WorkerScheduleSegment,
    first_slot: sympy.Expr,
    task_count: sympy.Expr,
    worker_count: int,
) -> tuple[CoordinateRelation, int | sympy.Expr, int | sympy.Expr] | None:
    """Project a previously proved root-major placement to dense arrivals."""
    if not _equal_integer_expressions(
        segment.task_order.target_domain.size_expr,
        task_count,
    ):
        return None

    _launch_stage_axis, worker_axis, _wave_axis = (
        segment.task_order.source_domain.axis_order
    )
    worker_domain = CoordinateDomain(
        axis_order=(worker_axis,),
        axis_counts_items=((worker_axis, worker_count),),
        kind="worker",
    )
    real_arrival_count = _concrete_or_symbolic_integer(
        SymbolicMin(sympy.Integer(worker_count), task_count)
    )
    effective_arrival_count = _concrete_or_symbolic_integer(
        SymbolicMax(sympy.Integer(1), sympy.sympify(real_arrival_count))
    )
    participant_axis = (
        min(
            (
                *segment.task_order.source_domain.axis_order,
                *segment.task_order.target_domain.axis_order,
            )
        )
        - 1
    )
    participant_domain = CoordinateDomain(
        axis_order=(participant_axis,),
        axis_counts_items=((participant_axis, effective_arrival_count),),
        kind="value",
    )
    worker = coordinate_axis_symbol(worker_axis)
    first_worker = sympy.Mod(first_slot, worker_count)
    local_ordinal = sympy.Mod(
        worker + worker_count - first_worker,  # pyrefly: ignore[unsupported-operation]
        worker_count,
    )
    participant_order = CoordinateRelation.point_map(
        worker_domain,
        participant_domain,
        (
            (
                ((worker_axis, 0, worker_count, 1),),
                (local_ordinal,),
            ),
        ),
    )
    if not participant_order.is_single_valued():
        raise AssertionError("root-major participant order is not single-valued")
    participant = coordinate_axis_symbol(participant_axis)
    workers_by_participant = CoordinateRelation.point_map(
        participant_domain,
        worker_domain,
        (
            (
                (
                    (
                        participant_axis,
                        0,
                        effective_arrival_count,
                        1,
                    ),
                ),
                (sympy.Mod(participant + first_worker, worker_count),),
            ),
        ),
    )
    if not workers_by_participant.is_total_function():
        raise AssertionError("root-major participant converse is not total")
    tile_dependency._remember_exact_converse(
        participant_order,
        workers_by_participant,
    )
    return participant_order, real_arrival_count, effective_arrival_count


@cache
def root_barrier_publication_plan(
    worker_schedule: WorkerSchedule,
    root: int,
    readiness_counters: tuple[ReadinessCounterPlan, ...] = (),
) -> RootBarrierPublicationPlan:
    """Assign every participating worker to its final root occurrence once.

    The reverse scan and interval subtraction are the authoritative derivation
    used for both the barrier arrival count and the codegen emission sites.
    No worker or task is enumerated.
    """
    continuation_domains = tuple(
        consumer.keys_by_consumer.source_domain
        for counter in readiness_counters
        if (consumer := counter.continuation_consumer) is not None
        and consumer.consumer_root == root
    )
    if len(continuation_domains) > 1:
        raise ValueError("one root cannot have multiple continuation owners")
    if continuation_domains and not all(
        counter.continuation_consumer is None
        or counter.continuation_consumer.consumer_root != root
        or counter.continuation_consumer.keys_by_consumer.is_total_function()
        for counter in readiness_counters
    ):
        raise ValueError("a continuation must cover its complete root")
    continuation_arrival_count: int | sympy.Expr = (
        _concrete_or_symbolic_integer(continuation_domains[0].size_expr)
        if continuation_domains
        else 0
    )
    source_segments = tuple(
        segment
        for segment in worker_schedule.segments_for_root(root)
        if segment.launch_stage == _SOURCE_LAUNCH_STAGE
    )
    if len(source_segments) > 1:
        raise ValueError("one root cannot have multiple source-stage owners")
    source_stage_arrival_count = (
        source_segments[0].task_order.target_domain.size if source_segments else 0
    )

    if (
        parameterized_geometry := _parametric_root_major_schedule_geometry(
            worker_schedule
        )
    ) is not None:
        matching = tuple(
            segment_index
            for segment_index, (segment, _first_wave, _task_count) in enumerate(
                parameterized_geometry
            )
            if segment.root == root
        )
        if len(matching) > 1:
            raise ValueError(
                f"parameterized root-major schedule has no unique root {root}"
            )
        if matching:
            segment_index = matching[0]
            segment, first_slot, task_count = parameterized_geometry[segment_index]
            participant = _root_major_participant_order_from_geometry(
                segment,
                first_slot,
                task_count,
                worker_schedule.worker_count,
            )
            if participant is None:
                raise ValueError("root-major participant support is not proved")
            participant_order, real_arrival_count, effective_arrival_count = participant
            if continuation_arrival_count != 0 or source_stage_arrival_count:
                raise ValueError(
                    "parameterized resident ownership overlaps another execution role"
                )
            return RootBarrierPublicationPlan(
                root=root,
                participant_intervals=(),
                participant_order=participant_order,
                publications=(RootBarrierPublication(segment_index, ()),),
                resident_arrival_count=real_arrival_count,
                continuation_arrival_count=0,
                source_stage_arrival_count=0,
                real_arrival_count=real_arrival_count,
                effective_arrival_count=effective_arrival_count,
                maximum_arrival_count=worker_schedule.worker_count,
            )
        if continuation_arrival_count == 0 and source_stage_arrival_count == 0:
            raise ValueError(f"parameterized root-major schedule has no root {root}")
    later_workers: tuple[WorkerInterval, ...] = ()
    reverse_publications: list[RootBarrierPublication] = []
    for segment_index in reversed(range(len(worker_schedule.segments))):
        segment = worker_schedule.segments[segment_index]
        if segment.root != root or segment.launch_stage == _SOURCE_LAUNCH_STAGE:
            continue
        segment_workers = segment.worker_intervals()
        publication_workers = _subtract_intervals(segment_workers, later_workers)
        if publication_workers:
            reverse_publications.append(
                RootBarrierPublication(segment_index, publication_workers)
            )
        later_workers = _normalize_intervals([*later_workers, *segment_workers])

    publications = tuple(reversed(reverse_publications))
    published_intervals = _normalize_intervals(
        [
            interval
            for publication in publications
            for interval in publication.worker_intervals
        ]
    )
    published_count = sum(
        _interval_cardinality(publication.worker_intervals)
        for publication in publications
    )
    if published_intervals != later_workers or published_count != _interval_cardinality(
        later_workers
    ):
        raise AssertionError("root publication ownership is not an exact partition")
    resident_arrival_count = published_count
    real_arrival_count = _concrete_or_symbolic_integer(
        sympy.Add(
            resident_arrival_count,
            sympy.sympify(continuation_arrival_count),
            source_stage_arrival_count,
        )
    )
    if not isinstance(real_arrival_count, int) or real_arrival_count <= 0:
        raise ValueError(
            "root-barrier publication requires a proved positive bounded owner set"
        )
    maximum_arrival_count = int(real_arrival_count)
    return RootBarrierPublicationPlan(
        root=root,
        participant_intervals=later_workers,
        participant_order=None,
        publications=publications,
        resident_arrival_count=resident_arrival_count,
        continuation_arrival_count=continuation_arrival_count,
        source_stage_arrival_count=source_stage_arrival_count,
        real_arrival_count=real_arrival_count,
        effective_arrival_count=real_arrival_count,
        maximum_arrival_count=maximum_arrival_count,
    )


def _build_root_major_worker_schedule(
    root_domains: tuple[CoordinateDomain, ...],
    root_task_orders: tuple[CoordinateRelation, ...],
    worker_count: int,
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> WorkerSchedule:
    """Build an exact packed schedule over constant or runtime task counts.

    Roots retain source order, but each starts at the immediately following
    global slot rather than at a rounded wave boundary.  Consequently every
    prerequisite edge still points from a smaller slot to a larger slot, while
    otherwise idle workers may begin the next root during the preceding root's
    final partial wave.  Full resident-worker admission makes that slot rank a
    progress proof; runtime counters remain the readiness authority.
    """
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    if len(root_domains) != len(root_task_orders):
        raise ValueError("root domains and task orders must have equal length")
    if any(root < 0 or root >= len(root_domains) for root in excluded_roots):
        raise ValueError("excluded root is out of range")
    if not root_domains:
        raise ValueError("root-major schedule requires at least one root")
    if any(
        task_order.target_domain != domain
        for domain, task_order in zip(root_domains, root_task_orders, strict=True)
    ):
        raise ValueError("root task orders must target their root domains")
    scheduled_roots = tuple(
        (root, domain)
        for root, domain in enumerate(root_domains)
        if root not in excluded_roots
    )
    if not scheduled_roots:
        raise ValueError("root-major schedule cannot exclude every root")

    minimum_axis = min(
        (axis for domain in root_domains for axis in domain.axis_order),
        default=0,
    )
    schedule_axes = (minimum_axis - 3, minimum_axis - 2, minimum_axis - 1)
    first_slot: sympy.Expr = sympy.Integer(0)
    root_first_slots: list[sympy.Expr] = []
    for _root, domain in scheduled_roots:
        task_count = domain.size_expr
        root_first_slots.append(first_slot)
        first_slot = sympy.simplify(sympy.Add(first_slot, task_count))
    schedule_domain = _worker_schedule_domain(
        worker_count,
        _ceildiv_nonnegative_expression(first_slot, worker_count),
        schedule_axes,
    )
    segments: list[WorkerScheduleSegment] = []
    for (root, _domain), root_first_slot in zip(
        scheduled_roots,
        root_first_slots,
        strict=True,
    ):
        relation = _packed_root_major_task_order_relation(
            schedule_domain,
            root_task_orders[root],
            root_first_slot,
            worker_count,
        )
        if relation is None:
            raise ValueError("root task order cannot be composed with packed slots")
        segments.append(
            WorkerScheduleSegment(
                root=root,
                task_order=relation,
                worker_begin=0,
                worker_count=worker_count,
                dispatch_offset=0,
            )
        )
    worker_schedule = WorkerSchedule(
        worker_count=worker_count,
        segments=tuple(segments),
    )
    _remember_root_major_schedule_geometry(
        worker_schedule,
        tuple(
            (segment, root_first_slot, segment.task_order.target_domain.size_expr)
            for segment, root_first_slot in zip(
                worker_schedule.segments,
                root_first_slots,
                strict=True,
            )
        ),
    )
    return worker_schedule


def build_baseline_worker_schedule(
    root_domains: tuple[CoordinateDomain, ...],
    root_task_orders: tuple[CoordinateRelation, ...],
    worker_count: int,
) -> WorkerSchedule:
    """Represent the existing source-ordered persistent task order exactly."""
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    segments: list[WorkerScheduleSegment] = []
    worker_step_begin = 0
    if len(root_domains) != len(root_task_orders):
        raise ValueError("root domains and task orders must have equal length")
    for root, (domain, task_order) in enumerate(
        zip(root_domains, root_task_orders, strict=True)
    ):
        task_count = domain.size
        active_workers = min(worker_count, task_count)
        segments.append(
            WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=0,
                worker_count=active_workers,
                dispatch_offset=worker_step_begin * active_workers,
            )
        )
        worker_step_begin += (task_count + worker_count - 1) // worker_count
    return WorkerSchedule(worker_count=worker_count, segments=tuple(segments))


def _family_placements_at_worker_step(
    worker_schedule: WorkerSchedule,
    *,
    root: int,
    task_domain: CoordinateDomain,
    task_order: CoordinateRelation,
    worker_step: int,
    unavailable_workers: frozenset[int] = frozenset(),
) -> tuple[WorkerSchedule, ...]:
    """Return dense placements for one complete family in free worker runs."""
    if task_domain.size > worker_schedule.worker_count:
        return ()
    # Preserve every prerequisite implied by earlier roots while looking for
    # idle capacity. ``replacing_root`` separately verifies that the proposed
    # placement agrees with the executable segment-tuple chronology.
    source_order_unavailable_workers = frozenset(
        worker
        for preceding_root in range(root)
        for worker, last_worker_step in worker_schedule.last_worker_steps_for_root(
            preceding_root
        ).items()
        if last_worker_step >= worker_step
    )
    free_workers = [
        worker
        for worker in range(worker_schedule.worker_count)
        if worker not in unavailable_workers
        and worker not in source_order_unavailable_workers
        and (
            (occupant_root := worker_schedule.root_at(worker, worker_step)) is None
            or occupant_root == root
        )
    ]
    result: list[WorkerSchedule] = []
    run_end = len(free_workers)
    while run_end:
        run_begin = run_end - 1
        while run_begin and free_workers[run_begin - 1] == free_workers[run_begin] - 1:
            run_begin -= 1
        if run_end - run_begin >= task_domain.size:
            worker_begin = free_workers[run_end - task_domain.size]
            candidate_segment = WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=worker_begin,
                worker_count=task_domain.size,
                dispatch_offset=worker_step * task_domain.size,
            )
            try:
                candidate = worker_schedule.replacing_root(
                    root,
                    (candidate_segment,),
                )
            except _WorkerScheduleChronologyError:
                # This is a speculative placement.  Earlier scheduling passes
                # can already have moved an independent later root ahead in
                # worker time, while ``replacing_root`` preserves the replaced
                # root's tuple position.  In that case the candidate has no
                # executable linearization in the current tuple order.  Keep
                # searching rather than invalidating the conservative plan.
                pass
            else:
                result.append(candidate)
        run_end = run_begin
    return tuple(result)


def place_ready_families(
    readiness_graph: ReadinessGraph,
    original_schedule: WorkerSchedule,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> tuple[WorkerSchedule, tuple[FinalArrivalContinuation, ...]]:
    """Move complete ready families into idle capacity during a producer tail.

    Final-arrival execution is useful when no separate workers are available.
    When a complete consumer family fits on workers that are free while some
    of its static ancestors still have queued work, a direct event wait avoids
    extending those producer streams.  This is derived from schedule liveness,
    independent of the roots' operations or graph topology.
    """
    result = worker_schedule
    remaining_continuations = continuations
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph, continuations
    )
    candidate_roots = sorted(
        {
            readiness_graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
            for continuation in remaining_continuations
        }
    )
    for root in candidate_roots:
        task_domain = readiness_graph.root_domains[root]
        if task_domain.size > result.worker_count:
            continue
        root_continuations = tuple(
            continuation
            for continuation in remaining_continuations
            if readiness_graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
            == root
        )
        if len(root_continuations) != 1:
            continue
        continuation = root_continuations[0]
        continuation_event = readiness_graph.event(continuation.event_id)
        continuation_consumer = continuation_event.consumers[
            continuation.consumer_index
        ]
        ready_after = _event_ready_after_worker_steps(
            readiness_graph,
            continuation_event,
            worker_schedule=result,
            continuation_by_root=continuation_by_root,
        )
        if ready_after is None:
            continue
        ready_after_worker_steps, prerequisite_roots = ready_after
        consumer_ready_after = continuation_consumer.keys_by_consumer.then(
            ready_after_worker_steps
        )
        readiness_bounds = (
            None
            if consumer_ready_after is None
            else consumer_ready_after.value_bounds()
        )
        if readiness_bounds is None:
            continue
        prerequisite_worker_steps: set[tuple[int, int]] = set()
        for prerequisite_root in prerequisite_roots:
            last_worker_steps = result.last_worker_steps_for_root(prerequisite_root)
            prerequisite_worker_steps.update(last_worker_steps.items())
        if not prerequisite_worker_steps:
            continue

        original_bounds = original_schedule.worker_step_bounds_for_root(root)
        if original_bounds is None:
            continue
        original_worker_step = original_bounds[0]
        remaining_after_placement = tuple(
            continuation
            for continuation in remaining_continuations
            if continuation not in root_continuations
        )
        for worker_step in range(readiness_bounds[0] + 1, original_worker_step):
            unfinished_workers = frozenset(
                worker
                for worker, prerequisite_worker_step in prerequisite_worker_steps
                if prerequisite_worker_step >= worker_step
            )
            if not unfinished_workers:
                break
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_worker_step(
                        result,
                        root=root,
                        task_domain=task_domain,
                        task_order=readiness_graph.root_task_orders[root],
                        worker_step=worker_step,
                        unavailable_workers=unfinished_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            result = candidate
            remaining_continuations = remaining_after_placement
            continuation_by_root.pop(root)
            break
    return result, remaining_continuations


def build_worker_schedule(
    readiness_graph: ReadinessGraph,
    continuation_candidates: tuple[FinalArrivalContinuation, ...],
    *,
    worker_count: int,
) -> tuple[
    WorkerSchedule,
    tuple[FinalArrivalContinuation, ...],
    tuple[ReadinessCounterPlan, ...],
]:
    """Derive local and static task placement for one worker count."""
    baseline = build_baseline_worker_schedule(
        readiness_graph.root_domains,
        readiness_graph.root_task_orders,
        worker_count,
    )
    nested_wait_roots = frozenset(
        readiness_consumer.consumer_root
        for event in readiness_graph.events
        for readiness_consumer in event.consumers
        if readiness_consumer.consumer_site_id is not None
    )
    continuations = choose_final_arrival_continuations(
        readiness_graph,
        continuation_candidates,
        baseline,
        excluded_roots=nested_wait_roots,
    )
    ordered = order_continuation_producers_by_readiness_key(
        readiness_graph,
        baseline,
        continuations,
    )
    continuation_roots = frozenset(
        readiness_consumer.consumer_root
        for continuation in continuations
        for readiness_consumer in (
            readiness_graph.event(continuation.event_id).consumers[
                continuation.consumer_index
            ],
        )
    )
    schedule = ordered.without_roots(continuation_roots)
    schedule, nested_loop_counters = place_nested_loop_consumers(
        readiness_graph,
        schedule,
        continuations,
    )
    schedule, continuations = place_ready_families(
        readiness_graph,
        ordered,
        schedule,
        continuations,
    )
    return schedule, continuations, nested_loop_counters


@dataclasses.dataclass(frozen=True)
class FinalArrivalContinuation:
    """A consumer task executed by whichever producer makes the final arrival."""

    event_id: int
    consumer_index: int


def _continuations_by_consumer_root(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> dict[int, FinalArrivalContinuation]:
    """Index final-arrival continuations by their consumer root."""
    return {
        readiness_graph.event(continuation.event_id)
        .consumers[continuation.consumer_index]
        .consumer_root: continuation
        for continuation in continuations
    }


@dataclasses.dataclass(frozen=True)
class ReadinessProducer:
    """One producer execution site's requirements for a readiness event."""

    producer_root: int
    producers_by_key: CoordinateRelation
    producer_site_id: int | None = None

    @cached_property
    def _publication_and_arrival_count_relations(
        self,
    ) -> tuple[CoordinateRelation | None, CoordinateRelation | None]:
        """Derive publication and arrival counts from one target-set proof."""
        return self.producers_by_key.derive_converse_and_target_counts()

    @property
    def keys_by_producer(self) -> CoordinateRelation | None:
        """Return the derived publication relation, when representable."""
        return self._publication_and_arrival_count_relations[0]

    @property
    def arrival_count_by_key(self) -> CoordinateRelation | None:
        """Return the exact symbolic number of arrivals per readiness key."""
        return self._publication_and_arrival_count_relations[1]


def _uniform_arrival_count(
    producers: tuple[ReadinessProducer, ...],
) -> int | None:
    """Return one constant arrival count for an event, when it has one."""
    bounds = _arrival_count_bounds(producers)
    if bounds is None or bounds[0] != bounds[1]:
        return None
    return bounds[0]


def _arrival_count_bounds(
    producers: tuple[ReadinessProducer, ...],
) -> tuple[int, int] | None:
    """Return proved minimum and maximum arrivals over all readiness keys."""
    minimum = 0
    maximum = 0
    cardinalities: list[CoordinateRelation] = []
    for readiness_producer in producers:
        cardinality = readiness_producer.arrival_count_by_key
        if cardinality is None or not cardinality.is_total_function():
            return None
        bounds = cardinality.value_bounds()
        if bounds is None:
            return None
        cardinalities.append(cardinality)
        minimum += bounds[0]
        maximum += bounds[1]
    if minimum != maximum and any(
        cardinality.canonical_single_valued() is None for cardinality in cardinalities
    ):
        # Nonuniform codegen evaluates the exact count at each key.  A bounds
        # proof alone cannot supply that value.
        return None
    return minimum, maximum


@dataclasses.dataclass(frozen=True)
class ReadinessConsumer:
    """A consumer execution site's symbolic requirements from one event."""

    consumer_root: int
    keys_by_consumer: CoordinateRelation
    covered_obligations: frozenset[DependencyObligation] = frozenset()
    consumer_site_id: int | None = None


def _readiness_key_domain(
    producers: tuple[ReadinessProducer, ...],
    consumers: tuple[ReadinessConsumer, ...],
) -> CoordinateDomain:
    """Validate and return the shared readiness-key domain of one event."""
    if not producers:
        raise ValueError("an event requires at least one producer")
    readiness_key_domain = producers[0].producers_by_key.source_domain
    if any(
        readiness_producer.producers_by_key.source_domain != readiness_key_domain
        for readiness_producer in producers[1:]
    ) or any(
        readiness_consumer.keys_by_consumer.target_domain != readiness_key_domain
        for readiness_consumer in consumers
    ):
        raise ValueError("event relations must share one readiness-key domain")
    return readiness_key_domain


@dataclasses.dataclass(frozen=True)
class ReadinessEvent:
    """One symbolic readiness event shared by scheduling and lowering."""

    producers: tuple[ReadinessProducer, ...]
    consumers: tuple[ReadinessConsumer, ...]

    def __post_init__(self) -> None:
        readiness_key_domain = _readiness_key_domain(self.producers, self.consumers)
        if readiness_key_domain.kind != "event":
            raise ValueError("readiness-key domain must have event kind")
        if readiness_key_domain.identity is None or readiness_key_domain.identity < 0:
            raise ValueError("readiness-key domain must have a nonnegative identity")
        if readiness_key_domain.axis_order != tuple(
            range(len(readiness_key_domain.axis_order))
        ):
            raise ValueError("readiness-key axes must use canonical local indices")
        if readiness_key_domain.block_sizes_items:
            raise ValueError("readiness-key domains must not inherit site block sizes")

    @property
    def readiness_key_domain(self) -> CoordinateDomain:
        """Return the readiness-key domain owned by every event relation."""
        return self.producers[0].producers_by_key.source_domain

    @property
    def event_id(self) -> int:
        """Return the event identity owned by its readiness-key domain."""
        identity = self.readiness_key_domain.identity
        assert identity is not None
        return identity

    @property
    def readiness_key_count_expr(self) -> sympy.Expr:
        """Return the possibly parameterized readiness-key count."""
        return self.readiness_key_domain.size_expr

    @property
    def readiness_key_count(self) -> int:
        """Return the concrete readiness-key count."""
        return self.readiness_key_domain.size

    @property
    def root_barrier_producer_root(self) -> int | None:
        if (
            _equal_integer_expressions(self.readiness_key_count_expr, 1)
            and len(self.producers) == 1
            and self.producers[0].producer_site_id is None
            and self.producers[0].producers_by_key.is_total()
        ):
            return self.producers[0].producer_root
        return None


def _validate_root_task_orders(
    root_task_orders: tuple[CoordinateRelation, ...],
) -> None:
    """Validate the one configured PID-to-logical relation per root."""
    root_identities: set[int] = set()
    for root, task_order in enumerate(root_task_orders):
        identity = task_order.target_domain.identity
        if (
            task_order.source_domain.kind != "task_order"
            or task_order.target_domain.kind != "site"
            or not _equal_integer_expressions(
                task_order.source_domain.size_expr,
                task_order.target_domain.size_expr,
            )
            or (
                not task_order.pieces
                and task_order.target_domain.size_expr.is_zero is not True
            )
        ):
            raise ValueError(
                "each root task order must have compatible typed domains"
            )
        if (
            task_order.source_domain.identity != identity
            or (identity is not None and identity in root_identities)
            or not task_order.is_bijection_from_source_support()
        ):
            raise ValueError(
                f"configured task order for root {root} is not an exact bijection"
            )
        if identity is not None:
            root_identities.add(identity)


@dataclasses.dataclass(frozen=True)
class ReadinessGraph:
    """Configured symbolic readiness DAG and root task orders."""

    root_task_orders: tuple[CoordinateRelation, ...]
    events: tuple[ReadinessEvent, ...]

    def __post_init__(self) -> None:
        _validate_root_task_orders(self.root_task_orders)
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("event IDs must be dense and source ordered")

    @property
    def root_domains(self) -> tuple[CoordinateDomain, ...]:
        """Return the task domains owned by the configured root task orders."""
        return tuple(task_order.target_domain for task_order in self.root_task_orders)

    def event(self, event_id: int) -> ReadinessEvent:
        return self.events[event_id]


def _supports_readiness_counter_lowering(
    readiness_producer: ReadinessProducer,
) -> bool:
    """Keep scheduler eligibility identical to counted-event code generation."""
    publication = readiness_producer.keys_by_producer
    return (
        readiness_producer.arrival_count_by_key is not None
        and publication is not None
        and publication.canonical_single_valued() is not None
    )


def _canonical_readiness_key_domain(domain: CoordinateDomain) -> CoordinateDomain:
    """Name quotient coordinates locally rather than borrowing site axes."""
    if domain.kind != "event" or domain.identity is not None:
        raise AssertionError("event quotient domain must be unidentified")
    return CoordinateDomain(
        axis_order=tuple(range(len(domain.axis_order))),
        axis_counts_items=tuple(
            (event_axis, count)
            for event_axis, (_site_axis, count) in enumerate(domain.axis_counts_items)
        ),
        kind="event",
    )


def _record_readiness_event(
    pending: dict[
        tuple[CoordinateDomain, tuple[ReadinessProducer, ...]],
        ReadinessEvent,
    ],
    *,
    readiness_key_domain: CoordinateDomain,
    producers: tuple[ReadinessProducer, ...],
    consumers: tuple[ReadinessConsumer, ...],
) -> None:
    """Canonicalize and group one semantic event by producer partition.

    Lowering eligibility is intentionally not part of event identity. Later
    scheduling phases either lower this exact semantic event or conservatively
    cover its obligations with root barriers.
    """
    if _readiness_key_domain(producers, consumers) != readiness_key_domain:
        raise AssertionError("event relations do not share their quotient domain")
    canonical_domain = _canonical_readiness_key_domain(readiness_key_domain)
    canonical_producers: list[ReadinessProducer] = []
    for readiness_producer in producers:
        producers_by_key = readiness_producer.producers_by_key.rename_source_axes(
            canonical_domain
        )
        if producers_by_key is None:
            raise AssertionError("event relation does not match its quotient geometry")
        canonical_producers.append(
            dataclasses.replace(
                readiness_producer,
                producers_by_key=producers_by_key,
            )
        )
    canonical_consumers: list[ReadinessConsumer] = []
    for readiness_consumer in consumers:
        keys_by_consumer = readiness_consumer.keys_by_consumer.rename_target_axes(
            canonical_domain
        )
        if keys_by_consumer is None:
            raise AssertionError("event relation does not match its quotient geometry")
        canonical_consumers.append(
            dataclasses.replace(
                readiness_consumer,
                keys_by_consumer=keys_by_consumer,
            )
        )
    canonical_producers_tuple = tuple(canonical_producers)
    signature = canonical_domain, canonical_producers_tuple
    previous_event = pending.get(signature)
    if previous_event is None:
        event_id = len(pending)
        identified_domain = dataclasses.replace(canonical_domain, identity=event_id)
        identified_producers: list[ReadinessProducer] = []
        for readiness_producer in canonical_producers_tuple:
            producers_by_key = readiness_producer.producers_by_key.rename_source_axes(
                identified_domain
            )
            if producers_by_key is None:
                raise AssertionError(
                    "event identity assignment changed readiness-key geometry"
                )
            identified_producers.append(
                dataclasses.replace(
                    readiness_producer,
                    producers_by_key=producers_by_key,
                )
            )
        event_producers = tuple(identified_producers)
        previous_consumers: tuple[ReadinessConsumer, ...] = ()
    else:
        event_id = previous_event.event_id
        identified_domain = previous_event.readiness_key_domain
        event_producers = previous_event.producers
        previous_consumers = previous_event.consumers

    grouped_consumers = list(previous_consumers)
    for canonical_consumer in canonical_consumers:
        keys_by_consumer = canonical_consumer.keys_by_consumer.rename_target_axes(
            identified_domain
        )
        if keys_by_consumer is None:
            raise AssertionError(
                "event identity assignment changed readiness-key geometry"
            )
        readiness_consumer = dataclasses.replace(
            canonical_consumer,
            keys_by_consumer=keys_by_consumer,
        )
        matching_index = next(
            (
                index
                for index, previous in enumerate(grouped_consumers)
                if previous.consumer_root == readiness_consumer.consumer_root
                and previous.consumer_site_id == readiness_consumer.consumer_site_id
                and previous.keys_by_consumer == readiness_consumer.keys_by_consumer
            ),
            None,
        )
        if matching_index is None:
            grouped_consumers.append(readiness_consumer)
            continue
        previous = grouped_consumers[matching_index]
        grouped_consumers[matching_index] = dataclasses.replace(
            previous,
            covered_obligations=(
                previous.covered_obligations | readiness_consumer.covered_obligations
            ),
        )
    pending[signature] = ReadinessEvent(
        producers=event_producers,
        consumers=tuple(grouped_consumers),
    )


@dataclasses.dataclass(frozen=True)
class ReadinessCounterPlan:
    """A readiness-key space receiving arrivals from one or more roots.

    Each producer has an independently proved readiness-key-to-producer
    relation. The
    expected count is derived from its producer sets, so the event represents
    both ordinary continuations and generic multi-predecessor joins.
    Consumers are independent of event identity. ``continuation_consumer_index``
    identifies the optional consumer executed by the final arriving producer.
    """

    producers: tuple[ReadinessProducer, ...]
    consumers: tuple[ReadinessConsumer, ...]
    continuation_consumer_index: int | None = None

    def __post_init__(self) -> None:
        _readiness_key_domain(self.producers, self.consumers)

    @property
    def readiness_key_domain(self) -> CoordinateDomain:
        """Return the readiness-key domain owned by every event relation."""
        return self.producers[0].producers_by_key.source_domain

    @property
    def continuation_consumer(self) -> ReadinessConsumer | None:
        if self.continuation_consumer_index is None:
            return None
        return self.consumers[self.continuation_consumer_index]

    @property
    def readiness_key_count_expr(self) -> sympy.Expr:
        """Return the possibly parameterized readiness-key count."""
        return self.readiness_key_domain.size_expr

    @property
    def readiness_key_count(self) -> int:
        """Return the complete readiness-key count."""
        return self.readiness_key_domain.size

    @property
    def parameter_symbols(self) -> frozenset[sympy.Symbol]:
        """Return every host-backed parameter used by this counter plan."""
        symbols: set[sympy.Symbol] = set()
        for producer in self.producers:
            symbols.update(producer.producers_by_key.parameter_symbols)
        for consumer in self.consumers:
            symbols.update(consumer.keys_by_consumer.parameter_symbols)
        return frozenset(symbols)

    def uniform_arrival_count(self) -> int | None:
        """Return constant fan-in without enumerating readiness keys."""
        return _uniform_arrival_count(self.producers)

    def arrival_count_bounds(self) -> tuple[int, int] | None:
        """Return proved fan-in bounds without enumerating readiness keys."""
        return _arrival_count_bounds(self.producers)


@dataclasses.dataclass(frozen=True)
class _EmittedPrerequisite:
    """One consumer prerequisite exactly as synchronization emits it.

    A prerequisite is either one whole-root barrier edge or one consumer arm
    of an exact counter plan.  Candidate generation and symbolic acceptance
    consume this same descriptor sequence, so they cannot silently disagree
    about which emitted dependencies exist.
    """

    barrier_producer_root: int | None = None
    barrier_consumer_root: int | None = None
    counter_plan: ReadinessCounterPlan | None = None
    counter_consumer_index: int | None = None

    def __post_init__(self) -> None:
        is_barrier = (
            self.barrier_producer_root is not None
            and self.barrier_consumer_root is not None
            and self.counter_plan is None
            and self.counter_consumer_index is None
        )
        is_counter = (
            self.barrier_producer_root is None
            and self.barrier_consumer_root is None
            and self.counter_plan is not None
            and self.counter_consumer_index is not None
            and 0 <= self.counter_consumer_index < len(self.counter_plan.consumers)
        )
        if not (is_barrier ^ is_counter):
            raise ValueError("prerequisite must describe exactly one emitted mechanism")

    @property
    def consumer_root(self) -> int:
        """Return the logical root whose execution is guarded."""
        if self.barrier_consumer_root is not None:
            return self.barrier_consumer_root
        assert self.counter_plan is not None
        assert self.counter_consumer_index is not None
        return self.counter_plan.consumers[self.counter_consumer_index].consumer_root

    @property
    def counter_consumer(self) -> ReadinessConsumer | None:
        """Return the exact-counter consumer, or ``None`` for a barrier."""
        if self.counter_plan is None:
            return None
        assert self.counter_consumer_index is not None
        return self.counter_plan.consumers[self.counter_consumer_index]


@cache
def _emitted_prerequisites(
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> tuple[_EmittedPrerequisite, ...]:
    """Return the canonical dependency view shared by proposal and proof."""
    result = [
        _EmittedPrerequisite(
            barrier_producer_root=producer_root,
            barrier_consumer_root=consumer_root,
        )
        for producer_root, consumer_root in sorted(root_barrier_edges)
    ]
    for plan in readiness_counters:
        for consumer_index in range(len(plan.consumers)):
            if consumer_index == plan.continuation_consumer_index:
                continue
            result.append(
                _EmittedPrerequisite(
                    counter_plan=plan,
                    counter_consumer_index=consumer_index,
                )
            )
    return tuple(result)


def _parameterized_prerequisites_follow_root_order(
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> bool:
    """Prove every emitted wait points forward in packed root order.

    Together with full resident-worker admission, this is the progress
    certificate for sharing a physical wave at root boundaries: a waiting
    task cannot precede any task that may release it on the same strand.
    """
    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        if prerequisite.barrier_producer_root is not None:
            producer_roots = (prerequisite.barrier_producer_root,)
        else:
            assert prerequisite.counter_plan is not None
            producer_roots = tuple(
                producer.producer_root
                for producer in prerequisite.counter_plan.producers
            )
        if any(
            producer_root >= prerequisite.consumer_root
            for producer_root in producer_roots
        ):
            return False
    return True


@cache
def _emitted_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
) -> tuple[FinalArrivalContinuation, ...] | None:
    """Recover continuation identities from the plans consumed by codegen.

    ``ReadinessCounterPlan.continuation_consumer_index`` is the emitted source
    of truth.  The readiness-key domain retains the semantic event identity;
    matching the plan against that event's canonical lowering gives the
    consumer index needed for recursive producer contraction without carrying
    a second independently-derived continuation list into proposal or proof.
    """
    result: list[FinalArrivalContinuation] = []
    consumer_roots: set[int] = set()
    for plan in readiness_counters:
        continuation_consumer = plan.continuation_consumer
        if continuation_consumer is None:
            continue
        event_id = plan.readiness_key_domain.identity
        if event_id is None or not 0 <= event_id < len(readiness_graph.events):
            return None
        event = readiness_graph.event(event_id)
        lowering_relations = _counter_lowering_relations(event)
        if lowering_relations is None:
            return None
        lowered_producers, lowered_consumers = lowering_relations
        if lowered_producers != plan.producers:
            return None
        matches = tuple(
            FinalArrivalContinuation(event_id, consumer_index)
            for consumer_index, consumer in enumerate(lowered_consumers)
            if consumer == continuation_consumer
        )
        if len(matches) != 1:
            return None
        consumer_root = continuation_consumer.consumer_root
        if consumer_root in consumer_roots:
            return None
        consumer_roots.add(consumer_root)
        result.append(matches[0])
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class _NestedLoopReadiness:
    """Configured readiness of one nested loop in task-local program order."""

    event: ReadinessEvent
    readiness_consumer: ReadinessConsumer
    ready_after_worker_step: CoordinateRelation
    prerequisite_worker_steps: frozenset[tuple[int, int]]


def _merge_relations_by_root(
    relations: tuple[tuple[int, CoordinateRelation], ...],
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    merged: dict[int, CoordinateRelation] = {}
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


def _static_producer_relations(
    readiness_graph: ReadinessGraph,
    *,
    root: int,
    site_id: int | None,
    readiness_keys: CoordinateRelation,
    continuation_by_root: dict[int, FinalArrivalContinuation],
    visiting: frozenset[int] = frozenset(),
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    """Contract continuations to relations from statically scheduled roots."""
    root_domain = readiness_graph.root_domains[root]
    root_keys = (
        readiness_keys
        if site_id is None
        else readiness_keys.project_source(root_domain)
    )
    if root_keys is None:
        return None
    continuation = continuation_by_root.get(root)
    if continuation is None:
        return ((root, root_keys),)
    if root in visiting:
        return None
    continuation_event = readiness_graph.event(continuation.event_id)
    continuation_consumer = continuation_event.consumers[continuation.consumer_index]
    converse_consumer = continuation_consumer.keys_by_consumer.converse()
    key_to_target = (
        None if converse_consumer is None else converse_consumer.then(root_keys)
    )
    if key_to_target is None:
        return None
    expanded: list[tuple[int, CoordinateRelation]] = []
    for readiness_producer in continuation_event.producers:
        publication = readiness_producer.keys_by_producer
        upstream_keys = None if publication is None else publication.then(key_to_target)
        if upstream_keys is None:
            return None
        upstream = _static_producer_relations(
            readiness_graph,
            root=readiness_producer.producer_root,
            site_id=readiness_producer.producer_site_id,
            readiness_keys=upstream_keys,
            continuation_by_root=continuation_by_root,
            visiting=visiting | frozenset((root,)),
        )
        if upstream is None:
            return None
        expanded.extend(upstream)
    return _merge_relations_by_root(tuple(expanded))


def _event_static_producers(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    return _readiness_static_producers(
        readiness_graph,
        event.producers,
        continuation_by_root,
    )


def _readiness_static_producers(
    readiness_graph: ReadinessGraph,
    producers: tuple[ReadinessProducer, ...],
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    """Contract a producer set through final-arrival continuation roots."""
    expanded: list[tuple[int, CoordinateRelation]] = []
    for readiness_producer in producers:
        publication = readiness_producer.keys_by_producer
        if publication is None:
            return None
        static_relations = _static_producer_relations(
            readiness_graph,
            root=readiness_producer.producer_root,
            site_id=readiness_producer.producer_site_id,
            readiness_keys=publication,
            continuation_by_root=continuation_by_root,
        )
        if static_relations is None:
            return None
        expanded.extend(static_relations)
    return _merge_relations_by_root(tuple(expanded))


def _transitive_static_prerequisite_roots(
    readiness_graph: ReadinessGraph,
    static_relations: tuple[tuple[int, CoordinateRelation], ...],
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> frozenset[int] | None:
    """Close static producers through waits earlier in task-local program order."""
    roots = {root for root, _relation in static_relations}
    pending = list(roots)
    while pending:
        consumer_root = pending.pop()
        for event in readiness_graph.events:
            if not any(
                readiness_consumer.consumer_root == consumer_root
                for readiness_consumer in event.consumers
            ):
                continue
            upstream = _event_static_producers(
                readiness_graph,
                event,
                continuation_by_root,
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


def _event_ready_after_worker_steps(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    *,
    worker_schedule: WorkerSchedule,
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> tuple[CoordinateRelation, frozenset[int]] | None:
    """Return when each readiness key becomes ready and its static producers."""
    static_relations = _event_static_producers(
        readiness_graph,
        event,
        continuation_by_root,
    )
    if static_relations is None:
        return None
    prerequisite_roots = _transitive_static_prerequisite_roots(
        readiness_graph,
        static_relations,
        continuation_by_root,
    )
    if prerequisite_roots is None:
        return None
    maxima: list[CoordinateRelation] = []
    for root, keys_by_task in static_relations:
        root_domain = readiness_graph.root_domains[root]
        if keys_by_task.source_domain != root_domain:
            return None
        maximum = _maximum_root_wave_by_key(
            worker_schedule,
            root,
            keys_by_task,
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
    identity = CoordinateRelation.identity(
        worker_schedule.worker_step_domain,
        worker_schedule.worker_step_domain,
    )
    maximum = combined.max_target_value_by_source(identity)
    return None if maximum is None else (maximum, prerequisite_roots)


def _nested_loop_readiness(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
    *,
    worker_schedule: WorkerSchedule,
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> _NestedLoopReadiness | None:
    """Map each nested-loop iteration to its prerequisite worker step."""
    assert readiness_consumer.consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None

    event_ready_after = _event_ready_after_worker_steps(
        readiness_graph,
        event,
        worker_schedule=worker_schedule,
        continuation_by_root=continuation_by_root,
    )
    if event_ready_after is None:
        return None
    ready_after_worker_steps, prerequisite_roots = event_ready_after
    iteration_ready_after_worker_step = readiness_consumer.keys_by_consumer.then(
        ready_after_worker_steps
    )
    if (
        iteration_ready_after_worker_step is None
        or not iteration_ready_after_worker_step.is_total_function()
    ):
        return None
    prerequisite_worker_steps: set[tuple[int, int]] = set()
    for root in prerequisite_roots:
        last_worker_steps = worker_schedule.last_worker_steps_for_root(root)
        prerequisite_worker_steps.update(last_worker_steps.items())
    return _NestedLoopReadiness(
        event=event,
        readiness_consumer=readiness_consumer,
        ready_after_worker_step=iteration_ready_after_worker_step,
        prerequisite_worker_steps=frozenset(prerequisite_worker_steps),
    )


def _segmented_nested_loop_counter(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
    boundaries: tuple[int | sympy.Expr, ...],
) -> ReadinessCounterPlan | None:
    """Coarsen one exact nested dependency into contiguous loop segments."""
    consumer_site_id = readiness_consumer.consumer_site_id
    assert consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    nested_extent = sympy.sympify(domain.axis_count_expressions[nested_axis])
    normalized_boundaries = tuple(
        sympy.simplify(sympy.sympify(boundary)) for boundary in boundaries
    )
    segments = tuple(itertools.pairwise(normalized_boundaries))
    if (
        not segments
        or any(boundary.is_integer is not True for boundary in normalized_boundaries)
        or any(
            not boundary.free_symbols <= domain.parameter_symbols
            for boundary in normalized_boundaries
        )
        or sympy.simplify(normalized_boundaries[0]) != 0
        or sympy.simplify(normalized_boundaries[-1] - nested_extent) != 0
        or any(
            not tile_dependency._is_provably_nonnegative(begin, None)
            or not tile_dependency._is_provably_nonnegative(
                sympy.simplify(end - begin - 1),
                None,
            )
            or not tile_dependency._is_provably_nonnegative(
                sympy.simplify(nested_extent - end),
                None,
            )
            for begin, end in segments
        )
    ):
        return None
    used_axes = readiness_consumer.keys_by_consumer.source_axes_affecting_targets()
    if used_axes is None or nested_axis not in used_axes:
        return None
    domain_counts = domain.axis_count_expressions
    reduced_domain = CoordinateDomain(
        axis_order=used_axes,
        axis_counts_items=tuple((axis, domain_counts[axis]) for axis in used_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in used_axes
            if axis in domain.block_sizes
        ),
        kind="site",
        identity=domain.identity,
        _allow_empty=domain._allow_empty,
    )
    outer_axes = tuple(axis for axis in used_axes if axis != nested_axis)
    reduced_counts = reduced_domain.axis_count_expressions
    readiness_key_domain = CoordinateDomain(
        axis_order=tuple(range(len(outer_axes) + 1)),
        axis_counts_items=(
            (0, len(segments)),
            *(
                (event_axis, reduced_counts[source_axis])
                for event_axis, source_axis in enumerate(outer_axes, start=1)
            ),
        ),
        kind="event",
        _allow_empty=reduced_domain._allow_empty,
    )
    keys_by_reduced_iteration = CoordinateRelation.point_map(
        reduced_domain,
        readiness_key_domain,
        tuple(
            (
                tuple(
                    (
                        (axis, segment_begin, segment_end, 1)
                        if axis == nested_axis
                        else (axis, 0, reduced_counts[axis], 1)
                    )
                    for axis in reduced_domain.axis_order
                ),
                (
                    sympy.Integer(stage),
                    *(coordinate_axis_symbol(axis) for axis in outer_axes),
                ),
            )
            for stage, (segment_begin, segment_end) in enumerate(segments)
        ),
    )
    outer_event_axis_by_source = dict(
        zip(outer_axes, range(1, len(outer_axes) + 1), strict=True)
    )
    reduced_iterations_by_key = CoordinateRelation(
        source_domain=readiness_key_domain,
        target_domain=reduced_domain,
        pieces=tuple(
            _CoordinateRelationPiece(
                source_bounds_items=(
                    (0, stage, stage + 1, 1),
                    *(
                        (
                            event_axis,
                            0,
                            reduced_counts[source_axis],
                            1,
                        )
                        for source_axis, event_axis in (
                            outer_event_axis_by_source.items()
                        )
                    ),
                ),
                target_ranges=tuple(
                    (
                        (axis, segment_begin, segment_end, 1)
                        if axis == nested_axis
                        else (
                            axis,
                            coordinate_axis_symbol(
                                outer_event_axis_by_source[axis]
                            ),
                            coordinate_axis_symbol(
                                outer_event_axis_by_source[axis]
                            )
                            + 1,
                            1,
                        )
                    )
                    for axis in reduced_domain.axis_order
                ),
            )
            for stage, (segment_begin, segment_end) in enumerate(segments)
        ),
    )
    # Boundary validation above proves that these pieces are an exact
    # partition, including their symbolic endpoints.  Retain that structural
    # proof instead of asking the generic converse search to rediscover the
    # same segmentation by specializing or enumerating the nested extent.
    tile_dependency._remember_exact_converse(
        keys_by_reduced_iteration,
        reduced_iterations_by_key,
    )
    # Remove task-local coordinates that provably do not affect the producer
    # set before taking the converse. The unreduced relation may be many-to-one
    # even though the reduced relation is exactly invertible.
    reduced_consumer = readiness_consumer.keys_by_consumer.project_source(
        reduced_domain
    )
    reduced_converse = None if reduced_consumer is None else reduced_consumer.converse()
    key_coarsening = (
        None
        if reduced_converse is None
        else reduced_converse.then(keys_by_reduced_iteration)
    )
    if key_coarsening is None:
        return None
    # This is a scheduling-derived coarsening of an already lowerable event,
    # not a second dependency fact. Derive both directions from the
    # authoritative producer sets and the one proved segment quotient.  The
    # generic converse search cannot rediscover runtime-valued segment cuts;
    # retaining the structural inverse here avoids specialization or sampling.
    publication_relations = tuple(
        (
            None
            if readiness_producer.keys_by_producer is None
            else readiness_producer.keys_by_producer.then(key_coarsening)
        )
        for readiness_producer in event.producers
    )
    if any(relation is None for relation in publication_relations):
        return None
    producers_by_key_relations: list[CoordinateRelation] = []
    semantic_keys_by_segment: CoordinateRelation | None = None
    for readiness_producer, publication in zip(
        event.producers,
        publication_relations,
        strict=True,
    ):
        assert publication is not None
        producers_by_key = publication.converse()
        if producers_by_key is None:
            if semantic_keys_by_segment is None:
                semantic_keys_by_segment = key_coarsening.converse()
            if semantic_keys_by_segment is None:
                return None
            producers_by_key = semantic_keys_by_segment.then(
                readiness_producer.producers_by_key
            )
            if producers_by_key is None:
                return None
            tile_dependency._remember_exact_converse(producers_by_key, publication)
        producers_by_key_relations.append(producers_by_key)
    keys_by_consumer = keys_by_reduced_iteration.lift_source(domain)
    if keys_by_consumer is None:
        return None

    return ReadinessCounterPlan(
        producers=tuple(
            ReadinessProducer(
                producer_root=readiness_producer.producer_root,
                producer_site_id=readiness_producer.producer_site_id,
                producers_by_key=relation,
            )
            for readiness_producer, relation in zip(
                event.producers,
                tuple(producers_by_key_relations),
                strict=True,
            )
            if relation is not None
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=readiness_consumer.consumer_root,
                covered_obligations=readiness_consumer.covered_obligations,
                consumer_site_id=readiness_consumer.consumer_site_id,
                keys_by_consumer=keys_by_consumer,
            ),
        ),
    )


def _uniform_nested_readiness_frontier(
    ready_after_worker_step: CoordinateRelation,
    nested_axis: int,
) -> CoordinateRelation | None:
    """Maximize a nested iteration's producer wave over every owning CTA.

    Outer axes which provably do not affect readiness are factored before the
    maximum.  Recomposition proves that this is an exact product, rather than
    inferring independence merely because an axis is absent from the target
    expression.  This distinction also makes a runtime-empty outer domain
    safe: the reduced proof is only used after it has been lifted back to the
    original, conditionally empty source domain.

    The result has one source coordinate, ``nested_axis``, and maps it to the
    latest producer wave required by any owning CTA.  A retained outer axis
    is maximized as a complete relation fiber; a fiber which may be empty or
    whose maximum is not exact declines conservatively.
    """
    domain = ready_after_worker_step.source_domain
    if (
        nested_axis not in domain.axis_order
        or len(ready_after_worker_step.target_domain.axis_order) != 1
        or not ready_after_worker_step.is_total_function()
    ):
        return None
    affecting_axes = ready_after_worker_step.source_axes_affecting_targets()
    if affecting_axes is None:
        return None
    retained_axes = tuple(
        axis
        for axis in domain.axis_order
        if axis == nested_axis or axis in affecting_axes
    )
    counts = domain.axis_count_expressions
    reduced_domain = CoordinateDomain(
        axis_order=retained_axes,
        axis_counts_items=tuple((axis, counts[axis]) for axis in retained_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in retained_axes
            if axis in domain.block_sizes
        ),
        kind=domain.kind,
        identity=domain.identity,
        _allow_empty=domain._allow_empty,
    )
    reduced = ready_after_worker_step.project_source(reduced_domain)
    recomposed = None if reduced is None else reduced.lift_source(domain)
    if (
        reduced is None
        or recomposed is None
        or not recomposed.is_pointwise_equal_on_same_support(
            ready_after_worker_step
        )
        or not reduced.is_total_function()
    ):
        return None

    nested_domain = CoordinateDomain(
        axis_order=(nested_axis,),
        axis_counts_items=((nested_axis, counts[nested_axis]),),
        block_sizes_items=(
            ((nested_axis, domain.block_sizes[nested_axis]),)
            if nested_axis in domain.block_sizes
            else ()
        ),
        kind=domain.kind,
        identity=domain.identity,
        _allow_empty=domain._allow_empty,
    )
    if reduced_domain == nested_domain:
        frontier = reduced
    else:
        nested_iteration = coordinate_axis_symbol(nested_axis)
        retained_iterations_by_nested = CoordinateRelation(
            source_domain=nested_domain,
            target_domain=reduced_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=(
                        (nested_axis, 0, counts[nested_axis], 1),
                    ),
                    target_ranges=tuple(
                        (
                            (axis, nested_iteration, nested_iteration + 1, 1)
                            if axis == nested_axis
                            else (axis, 0, counts[axis], 1)
                        )
                        for axis in retained_axes
                    ),
                ),
            ),
        )
        frontier = retained_iterations_by_nested.max_target_value_by_source(reduced)
    canonical = None if frontier is None else frontier.canonical_single_valued()
    return (
        canonical
        if canonical is not None and canonical.is_total_function()
        else None
    )


def _nested_ready_prefix_boundaries(
    frontier: CoordinateRelation,
    consumer_worker_step: int,
) -> tuple[int | sympy.Expr, ...] | None:
    """Derive the exact prefix ready strictly before one admission wave.

    Composition with a wave-prefix relation computes the exact preimage of
    ``producer_wave < consumer_worker_step``.  The result is accepted only
    when its complete source support is one prefix.  Consequently wrapping or
    otherwise nonmonotone readiness declines instead of relying on sampled
    endpoints.
    """
    if (
        consumer_worker_step < 0
        or len(frontier.source_domain.axis_order) != 1
        or len(frontier.target_domain.axis_order) != 1
        or not frontier.is_total_function()
    ):
        return None
    (nested_axis,) = frontier.source_domain.axis_order
    nested_extent = sympy.sympify(
        frontier.source_domain.axis_count_expressions[nested_axis]
    )
    if not tile_dependency._is_provably_nonnegative(nested_extent - 1, None):
        return None
    if consumer_worker_step == 0:
        return (sympy.Integer(0), nested_extent)

    (wave_axis,) = frontier.target_domain.axis_order
    wave_count = sympy.sympify(
        frontier.target_domain.axis_count_expressions[wave_axis]
    )
    if tile_dependency._is_provably_nonnegative(
        wave_count - consumer_worker_step,
        None,
    ):
        ready_wave_end: int | sympy.Expr = sympy.Integer(consumer_worker_step)
    elif tile_dependency._is_provably_nonnegative(
        consumer_worker_step - wave_count,
        None,
    ):
        ready_wave_end = wave_count
    else:
        return None

    ready_value_domain = CoordinateDomain((), (), kind="value")
    ready_values = CoordinateRelation(
        source_domain=frontier.target_domain,
        target_domain=ready_value_domain,
        pieces=(
            _CoordinateRelationPiece(
                source_bounds_items=((wave_axis, 0, ready_wave_end, 1),),
                target_ranges=(),
            ),
        ),
    )
    ready_iterations = frontier.then(ready_values)
    ready_iterations = (
        None
        if ready_iterations is None
        else ready_iterations.canonical_single_valued()
    )
    if ready_iterations is None:
        return None
    ready_iterations = ready_iterations.coalesce_adjacent_source_boxes()
    if not ready_iterations.pieces:
        split_iteration = sympy.Integer(0)
    elif len(ready_iterations.pieces) == 1:
        (piece,) = ready_iterations.pieces
        if len(piece.source_bounds_items) != 1:
            return None
        axis, begin, end, step = piece.source_bounds_items[0]
        if axis != nested_axis or step != 1 or sympy.simplify(begin) != 0:
            return None
        split_iteration = sympy.simplify(end)
    else:
        return None

    if sympy.simplify(split_iteration) == 0 or sympy.simplify(
        split_iteration - nested_extent
    ) == 0:
        return (sympy.Integer(0), nested_extent)
    if not tile_dependency._is_provably_nonnegative(
        split_iteration - 1,
        None,
    ) or not tile_dependency._is_provably_nonnegative(
        nested_extent - split_iteration - 1,
        None,
    ):
        return None
    return (sympy.Integer(0), split_iteration, nested_extent)


def _concrete_nested_ready_prefix_boundaries(
    frontier: CoordinateRelation,
    consumer_worker_step: int,
) -> tuple[int, ...] | None:
    """Retain the old binary search as a proved-monotone concrete oracle."""
    if not _scalar_relation_is_nondecreasing(frontier):
        return None
    (nested_axis,) = frontier.source_domain.axis_order
    nested_extent = frontier.source_domain.axis_counts[nested_axis]

    def ready(nested_iteration: int) -> bool | None:
        value_bounds = frontier.value_bounds({nested_axis: nested_iteration})
        return (
            None
            if value_bounds is None
            else value_bounds[1] < consumer_worker_step
        )

    first_ready = ready(0)
    last_ready = ready(nested_extent - 1)
    if first_ready is None or last_ready is None:
        return None
    if not first_ready:
        split_iteration = 0
    elif last_ready:
        split_iteration = nested_extent
    else:
        lower = 0
        upper = nested_extent - 1
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            midpoint_ready = ready(midpoint)
            if midpoint_ready is None:
                return None
            if midpoint_ready:
                lower = midpoint
            else:
                upper = midpoint
        split_iteration = upper
    return tuple(sorted({0, split_iteration, nested_extent}))


def _split_nested_loop_at_readiness(
    readiness_graph: ReadinessGraph,
    nested_readiness: _NestedLoopReadiness,
    *,
    consumer_worker_step: int,
) -> ReadinessCounterPlan | None:
    """Split a nested loop at its exact uniform readiness frontier."""
    domain = nested_readiness.readiness_consumer.keys_by_consumer.source_domain
    consumer_site_id = nested_readiness.readiness_consumer.consumer_site_id
    assert consumer_site_id is not None
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[nested_readiness.readiness_consumer.consumer_root],
        domain,
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    frontier = _uniform_nested_readiness_frontier(
        nested_readiness.ready_after_worker_step,
        nested_axis,
    )
    if frontier is None:
        return None
    boundaries = _nested_ready_prefix_boundaries(frontier, consumer_worker_step)
    if boundaries is None and not frontier.parameter_symbols:
        boundaries = _concrete_nested_ready_prefix_boundaries(
            frontier,
            consumer_worker_step,
        )
    if boundaries is None:
        return None
    return _segmented_nested_loop_counter(
        readiness_graph,
        nested_readiness.event,
        nested_readiness.readiness_consumer,
        boundaries,
    )


def _nested_loop_entry_counter(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
) -> ReadinessCounterPlan | None:
    """Coarsen exact iteration readiness to one wait per owning root task."""
    consumer_site_id = readiness_consumer.consumer_site_id
    assert consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None
    nested_iterations_per_task = domain.axis_count_expressions[nested_axes[0]]
    return _segmented_nested_loop_counter(
        readiness_graph,
        event,
        readiness_consumer,
        (0, nested_iterations_per_task),
    )


def place_nested_loop_consumers(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> tuple[WorkerSchedule, tuple[ReadinessCounterPlan, ...]]:
    """Place root tasks with nested waits and derive their readiness counters.

    Exact nested-iteration dependencies remain the semantic source of truth.
    This pass uses only worker steps and task-local program order to select one
    split point for the original nested loop.
    It does not inspect operation kinds or recognize a graph topology.
    """
    consumers_by_root: dict[
        int,
        list[tuple[ReadinessEvent, ReadinessConsumer]],
    ] = {}
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph, continuations
    )
    for event in readiness_graph.events:
        for readiness_consumer in event.consumers:
            if readiness_consumer.consumer_site_id is not None:
                consumers_by_root.setdefault(
                    readiness_consumer.consumer_root, []
                ).append((event, readiness_consumer))

    result = worker_schedule
    plans: list[ReadinessCounterPlan] = []
    for consumer_root, event_consumers in sorted(consumers_by_root.items()):
        task_domain = readiness_graph.root_domains[consumer_root]

        # A preceding site may already carry every dependency obligation needed by
        # a later site.  The implication was proved from DeviceIR program
        # order when the readiness graph was built, so the later wait is redundant.
        uncovered_consumers: list[tuple[ReadinessEvent, ReadinessConsumer]] = []
        preceding_obligations: set[DependencyObligation] = set()
        for event, readiness_consumer in sorted(
            event_consumers,
            key=lambda item: (
                item[1].consumer_site_id
                if item[1].consumer_site_id is not None
                else -1,
                item[0].event_id,
            ),
        ):
            if (
                readiness_consumer.covered_obligations
                and readiness_consumer.covered_obligations <= (preceding_obligations)
            ):
                continue
            uncovered_consumers.append((event, readiness_consumer))
            preceding_obligations.update(readiness_consumer.covered_obligations)

        nested_loop_entry_plans = tuple(
            plan
            for event, readiness_consumer in uncovered_consumers
            if (
                plan := _nested_loop_entry_counter(
                    readiness_graph, event, readiness_consumer
                )
            )
            is not None
        )
        if task_domain.size > result.worker_count:
            plans.extend(nested_loop_entry_plans)
            continue

        nested_readiness = tuple(
            _nested_loop_readiness(
                readiness_graph,
                event,
                readiness_consumer,
                worker_schedule=result,
                continuation_by_root=continuation_by_root,
            )
            for event, readiness_consumer in uncovered_consumers
        )
        if not nested_readiness or any(item is None for item in nested_readiness):
            plans.extend(nested_loop_entry_plans)
            continue
        ordered_readiness = tuple(item for item in nested_readiness if item is not None)

        current_consumer_bounds = result.worker_step_bounds_for_root(consumer_root)
        if current_consumer_bounds is None:
            plans.extend(nested_loop_entry_plans)
            continue
        original_worker_step = current_consumer_bounds[0]
        readiness_bounds = tuple(
            item.ready_after_worker_step.value_bounds() for item in ordered_readiness
        )
        if any(bounds is None for bounds in readiness_bounds):
            plans.extend(nested_loop_entry_plans)
            continue
        earliest_ready_after_step = min(
            bounds[0] for bounds in readiness_bounds if bounds is not None
        )
        prerequisite_worker_steps = frozenset(
            placement
            for item in ordered_readiness
            for placement in item.prerequisite_worker_steps
        )
        chosen: tuple[WorkerSchedule, tuple[ReadinessCounterPlan, ...]] | None = None
        for worker_step in range(earliest_ready_after_step + 1, original_worker_step):
            busy_workers = frozenset(
                worker
                for worker, prerequisite_worker_step in prerequisite_worker_steps
                if prerequisite_worker_step >= worker_step
            )
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_worker_step(
                        result,
                        root=consumer_root,
                        task_domain=task_domain,
                        task_order=readiness_graph.root_task_orders[consumer_root],
                        worker_step=worker_step,
                        unavailable_workers=busy_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            milestone_results = tuple(
                _split_nested_loop_at_readiness(
                    readiness_graph,
                    item,
                    consumer_worker_step=worker_step,
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
            plans.extend(nested_loop_entry_plans)
            continue
        result, nested_loop_plans = chosen
        plans.extend(nested_loop_plans)
    return result, tuple(plans)


@dataclasses.dataclass(frozen=True)
class StaticPipelinePlan:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    worker_schedule: WorkerSchedule
    root_task_orders: tuple[CoordinateRelation, ...]
    readiness_counters: tuple[ReadinessCounterPlan, ...]
    root_barrier_edges: frozenset[tuple[int, int]]
    transient_source_root: int | None = None

    def __post_init__(self) -> None:
        _validate_root_task_orders(self.root_task_orders)
        root_count = len(self.root_task_orders)
        root_domains = tuple(
            task_order.target_domain for task_order in self.root_task_orders
        )
        for segment in self.worker_schedule.segments:
            if (
                segment.root >= root_count
                or segment.task_order.target_domain
                != self.root_task_orders[segment.root].target_domain
            ):
                raise ValueError(
                    "worker schedule segment disagrees with its configured root domain"
                )
        for producer_root, consumer_root in self.root_barrier_edges:
            if not (0 <= producer_root < root_count and 0 <= consumer_root < root_count):
                raise ValueError("root-barrier edge references an unknown root")

        continuation_roots: set[int] = set()
        for counter in self.readiness_counters:
            if not _supports_exact_counter_plan_lowering(
                counter,
                root_domains,
            ):
                raise ValueError("readiness counter has no exact lowering")
            for consumer_index, consumer in enumerate(counter.consumers):
                if consumer_index == counter.continuation_consumer_index:
                    continuation_roots.add(consumer.consumer_root)

        if _current_renderer_lowerable_counters(
            self.readiness_counters,
            root_domains,
        ) != self.readiness_counters:
            raise ValueError("readiness counter is unsupported by the current renderer")

        scheduled_roots = {segment.root for segment in self.worker_schedule.segments}
        if scheduled_roots & continuation_roots:
            raise ValueError("continuation roots must not retain resident ownership")
        if scheduled_roots | continuation_roots != set(range(root_count)):
            raise ValueError("pipeline plan does not own every configured root")

        source_segments = tuple(
            segment
            for segment in self.worker_schedule.segments
            if segment.launch_stage == _SOURCE_LAUNCH_STAGE
        )
        if self.transient_source_root is None:
            if source_segments:
                raise ValueError("a source-stage segment requires source ownership")
        elif (
            not 0 <= self.transient_source_root < root_count
            or len(source_segments) != 1
            or source_segments[0].root != self.transient_source_root
        ):
            raise ValueError("source ownership requires one matching stage-zero segment")

        # Freeze the sole root-publication derivation with the selected plan.
        # Code generation consumes this cache and must not reconstruct owner
        # support or arrival counts from schedule geometry.
        _ = self.root_barrier_publication_plans

    @cached_property
    def root_barrier_publication_plans(
        self,
    ) -> tuple[RootBarrierPublicationPlan | None, ...]:
        """Return the finalized publication plan for every applicable root."""
        root_count = len(self.root_task_orders)
        producer_roots = {
            producer for producer, _consumer in self.root_barrier_edges
        }
        publication_roots = producer_roots | {
            segment.root for segment in self.worker_schedule.segments
        }
        root_level_counters = tuple(
            counter
            for counter in self.readiness_counters
            if all(consumer.consumer_site_id is None for consumer in counter.consumers)
        )
        return tuple(
            root_barrier_publication_plan(
                self.worker_schedule,
                root,
                root_level_counters,
            )
            if root in publication_roots
            else None
            for root in range(root_count)
        )


def choose_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    candidates: tuple[FinalArrivalContinuation, ...],
    worker_schedule: WorkerSchedule,
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[FinalArrivalContinuation, ...]:
    """Choose final-arrival execution from complete exact task readiness.

    A statically one-task family has no task-level parallelism to expose, so it
    remains in the static schedule.  A parameterized family may be empty, one
    task, or many tasks at runtime; final-arrival ownership is exact in every
    case and must not be selected from one sampled extent.
    """
    has_possible_worker_by_root = {
        root: bool(worker_schedule.segments_for_root(root))
        for root in range(len(readiness_graph.root_domains))
    }
    reachable_candidates: list[FinalArrivalContinuation] = []
    for continuation in candidates:
        event = readiness_graph.event(continuation.event_id)
        readiness_consumer = event.consumers[continuation.consumer_index]
        if not any(
            has_possible_worker_by_root[readiness_producer.producer_root]
            for readiness_producer in event.producers
        ):
            continue
        has_possible_worker_by_root[readiness_consumer.consumer_root] = True
        reachable_candidates.append(continuation)

    result: list[FinalArrivalContinuation] = []
    for continuation in reachable_candidates:
        readiness_consumer = readiness_graph.event(continuation.event_id).consumers[
            continuation.consumer_index
        ]
        consumer_root = readiness_consumer.consumer_root
        if consumer_root in excluded_roots:
            continue
        domain = readiness_graph.root_domains[consumer_root]
        if not _equal_integer_expressions(domain.size_expr, 1):
            result.append(continuation)
    return tuple(result)


def _counter_lowering_relations(
    event: ReadinessEvent,
) -> tuple[tuple[ReadinessProducer, ...], tuple[ReadinessConsumer, ...]] | None:
    """Return an exact lowerable representation of one semantic event.

    The readiness graph retains its finest exact key space.  A single producer
    whose publication is not a function may still admit the established
    producer-set quotient as a counter-rendering strength reduction.  Derive
    that quotient from the event's authoritative relation rather than replacing
    the event in the graph.
    """
    if all(
        _supports_readiness_counter_lowering(readiness_producer)
        for readiness_producer in event.producers
    ):
        return event.producers, event.consumers
    if len(event.producers) != 1:
        return None
    (semantic_producer,) = event.producers
    normalized_producers_by_key = (
        semantic_producer.producers_by_key.coalesce_adjacent_source_boxes()
    )
    quotient = normalized_producers_by_key.producer_set_quotient()
    if quotient is None:
        return None
    quotient_keys_by_semantic_key, quotient_producers_by_key = quotient
    quotient_key_domain = dataclasses.replace(
        _canonical_readiness_key_domain(quotient_keys_by_semantic_key.target_domain),
        identity=event.event_id,
    )
    quotient_keys_by_semantic_key = quotient_keys_by_semantic_key.rename_target_axes(
        quotient_key_domain
    )
    quotient_producers_by_key = quotient_producers_by_key.rename_source_axes(
        quotient_key_domain
    )
    if quotient_keys_by_semantic_key is None or quotient_producers_by_key is None:
        return None
    lowered_producer = dataclasses.replace(
        semantic_producer,
        producers_by_key=quotient_producers_by_key,
    )
    if not _supports_readiness_counter_lowering(lowered_producer):
        return None
    lowered_consumers: list[ReadinessConsumer] = []
    for consumer in event.consumers:
        keys_by_consumer = consumer.keys_by_consumer.then(quotient_keys_by_semantic_key)
        if keys_by_consumer is None:
            return None
        lowered_consumers.append(
            dataclasses.replace(consumer, keys_by_consumer=keys_by_consumer)
        )
    return (lowered_producer,), tuple(lowered_consumers)


def choose_readiness_counters(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
    *,
    excluded_obligations: frozenset[DependencyObligation] = frozenset(),
) -> tuple[ReadinessCounterPlan, ...]:
    """Select root-entry events representable by readiness counters.

    Nested consumers keep their execution-site lowering. Excluding one
    consumer does not discard independent consumers of the same semantic
    event. A whole-root one-key event remains on the aggregated root-barrier
    path, while a one-key event covering only a strict producer subset retains
    its exact counter. Unsupported relations monotonically fall back to a root
    barrier during coverage selection.
    """
    continuation_consumers = {
        (continuation.event_id, continuation.consumer_index)
        for continuation in continuations
    }
    selected: list[ReadinessCounterPlan] = []
    for event in readiness_graph.events:
        if event.root_barrier_producer_root is not None:
            continue
        lowering_relations = _counter_lowering_relations(event)
        if lowering_relations is None:
            continue
        lowered_producers, lowered_consumers = lowering_relations
        retained_consumers: list[ReadinessConsumer] = []
        continuation_consumer_indices: list[int] = []
        for consumer_index, readiness_consumer in enumerate(lowered_consumers):
            if (
                readiness_consumer.consumer_site_id is not None
                or readiness_consumer.keys_by_consumer.canonical_single_valued() is None
            ):
                continue
            remaining = readiness_consumer.covered_obligations - excluded_obligations
            is_continuation = (
                event.event_id,
                consumer_index,
            ) in continuation_consumers
            if not is_continuation and not remaining:
                continue
            if is_continuation:
                continuation_consumer_indices.append(len(retained_consumers))
            retained_consumers.append(
                ReadinessConsumer(
                    consumer_root=readiness_consumer.consumer_root,
                    keys_by_consumer=readiness_consumer.keys_by_consumer,
                    covered_obligations=(
                        readiness_consumer.covered_obligations
                        if is_continuation
                        else frozenset(remaining)
                    ),
                )
            )
        if len(continuation_consumer_indices) > 1:
            raise ValueError("one readiness counter cannot have multiple continuations")
        continuation_consumer_index = (
            continuation_consumer_indices[0] if continuation_consumer_indices else None
        )
        # A one-key event may cover a strict producer subset. Whole-root
        # one-key events were rejected through ``root_barrier_producer_root``.
        if not retained_consumers:
            continue
        candidate = ReadinessCounterPlan(
            producers=lowered_producers,
            consumers=tuple(retained_consumers),
            continuation_consumer_index=continuation_consumer_index,
        )
        if _supports_exact_counter_plan_lowering(
            candidate,
            readiness_graph.root_domains,
        ):
            selected.append(candidate)
    selected_continuation_count = sum(
        counter_plan.continuation_consumer_index is not None
        for counter_plan in selected
    )
    if selected_continuation_count != len(continuation_consumers):
        raise AssertionError(
            "not every final-arrival continuation has a readiness counter"
        )
    return tuple(selected)


def build_readiness_events(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[CoordinateDomain, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    publishable_site_ids: frozenset[int] | None = None,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> tuple[ReadinessEvent, ...]:
    """Build canonical symbolic readiness events from memory dependencies.

    This is the sole event-construction path. It never constructs a per-task
    producer set. Unsupported relations coarsen to one root-barrier
    event for the affected root pair.
    """
    symbolic_dependencies = instantiate_symbolic_dependencies(
        dependency_graph,
        root_domains=root_domains,
        site_domains=site_domains,
        prove_nonnegative=prove_nonnegative,
    )
    site_by_id = {site.site_id: site for site in dependency_graph.execution_sites}
    exact_dependencies = tuple(
        dependency
        for dependency in symbolic_dependencies
        if dependency.producers_by_consumer is not None
        and dependency.producers_by_consumer.source_axes_affecting_targets() is not None
    )
    all_obligations_by_pair: dict[tuple[int, int], set[DependencyObligation]] = {}
    for edge in dependency_graph.edges:
        pair = (edge.producer_root, edge.consumer_root)
        for access_dependency in edge.access_dependencies:
            all_obligations_by_pair.setdefault(pair, set()).update(
                dependency_graph.dependency_obligations(access_dependency)
            )

    implied_obligations: dict[DependencyObligation, set[DependencyObligation]] = {}
    for preceding_dependency in exact_dependencies:
        preceding_site_id = preceding_dependency.consumer_site_id
        if preceding_site_id is None or site_by_id[preceding_site_id].is_root:
            continue
        preceding_producers = preceding_dependency.producers_by_consumer
        assert preceding_producers is not None
        preceding_obligation = (
            preceding_dependency.dependency_id,
            preceding_dependency.producer_site_id,
            preceding_site_id,
        )
        for later_dependency in exact_dependencies:
            later_site_id = later_dependency.consumer_site_id
            later_producers = later_dependency.producers_by_consumer
            if (
                later_dependency is preceding_dependency
                or later_site_id is None
                or later_producers is None
                or preceding_dependency.consumer_root != later_dependency.consumer_root
                or preceding_dependency.producer_root != later_dependency.producer_root
                or preceding_dependency.producer_site_id
                != later_dependency.producer_site_id
                or preceding_producers.target_domain != later_producers.target_domain
            ):
                continue
            preceding = consumer_to_preceding_site_relation(
                dependency_graph,
                site_domains=site_domains,
                preceding_site_id=preceding_site_id,
                consumer_site_id=later_site_id,
                consumer_access_id=later_dependency.consumer_access_id,
            )
            acquired = (
                None if preceding is None else preceding.then(preceding_producers)
            )
            if acquired is not None and acquired.covers(later_producers):
                implied_obligations.setdefault(preceding_obligation, set()).add(
                    (
                        later_dependency.dependency_id,
                        later_dependency.producer_site_id,
                        later_site_id,
                    )
                )

    exact_relations: dict[
        tuple[int, int | None, CoordinateDomain],
        dict[
            tuple[int, int | None, CoordinateDomain],
            list[tuple[CoordinateRelation, DependencyObligation]],
        ],
    ] = {}

    def add_exact_relation(
        *,
        producer_root: int,
        producer_site_id: int | None,
        consumer_root: int,
        consumer_site_id: int | None,
        relation: CoordinateRelation,
        covered_obligations: frozenset[DependencyObligation],
    ) -> None:
        consumer = (consumer_root, consumer_site_id, relation.source_domain)
        producer = (producer_root, producer_site_id, relation.target_domain)
        exact_relations.setdefault(consumer, {}).setdefault(producer, []).extend(
            (relation, obligation) for obligation in covered_obligations
        )

    for dependency in exact_dependencies:
        relation = dependency.producers_by_consumer
        assert relation is not None
        obligation = (
            dependency.dependency_id,
            dependency.producer_site_id,
            dependency.consumer_site_id,
        )
        exact_obligations = frozenset(
            (obligation, *implied_obligations.get(obligation, ()))
        )
        producer_site = (
            None
            if dependency.producer_site_id is None
            else site_by_id[dependency.producer_site_id]
        )
        consumer_site = (
            None
            if dependency.consumer_site_id is None
            else site_by_id[dependency.consumer_site_id]
        )
        producer_is_root = producer_site is None or producer_site.is_root
        consumer_is_root = consumer_site is None or consumer_site.is_root
        producer_site_is_usable = producer_is_root or (
            producer_site is not None
            and producer_site.can_split_loop
            and (
                publishable_site_ids is None
                or dependency.producer_site_id in publishable_site_ids
            )
        )
        consumer_site_is_usable = consumer_is_root or (
            consumer_site is not None and consumer_site.can_split_loop
        )
        if producer_site_is_usable and consumer_site_is_usable:
            add_exact_relation(
                producer_root=dependency.producer_root,
                producer_site_id=(
                    None if producer_is_root else dependency.producer_site_id
                ),
                consumer_root=dependency.consumer_root,
                consumer_site_id=(
                    None if consumer_is_root else dependency.consumer_site_id
                ),
                relation=relation,
                covered_obligations=exact_obligations,
            )

        if producer_is_root and consumer_is_root:
            continue
        root_relation = relation
        if not producer_is_root:
            projected = root_relation.project_target(
                root_domains[dependency.producer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        if not consumer_is_root:
            projected = root_relation.project_source(
                root_domains[dependency.consumer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        add_exact_relation(
            producer_root=dependency.producer_root,
            producer_site_id=None,
            consumer_root=dependency.consumer_root,
            consumer_site_id=None,
            relation=root_relation,
            covered_obligations=exact_obligations,
        )

    pending_events: dict[
        tuple[CoordinateDomain, tuple[ReadinessProducer, ...]],
        ReadinessEvent,
    ] = {}
    represented_obligations: set[DependencyObligation] = set()

    def record_readiness_event(
        *,
        readiness_key_domain: CoordinateDomain,
        producers: tuple[ReadinessProducer, ...],
        consumers: tuple[ReadinessConsumer, ...],
    ) -> None:
        _record_readiness_event(
            pending_events,
            readiness_key_domain=readiness_key_domain,
            producers=producers,
            consumers=consumers,
        )
        for readiness_consumer in consumers:
            represented_obligations.update(readiness_consumer.covered_obligations)

    def add_producer_key_events(
        *,
        consumer_root: int,
        consumer_site_id: int | None,
        relations: list[
            tuple[
                tuple[int, int | None, CoordinateDomain],
                CoordinateRelation,
                frozenset[DependencyObligation],
            ]
        ],
    ) -> None:
        """Keep finer readiness keys when a consumer quotient needs fanout."""
        for producer, relation, obligations in relations:
            producer_root, producer_site_id, producer_domain = producer
            readiness_key_domain = dataclasses.replace(
                producer_domain,
                kind="event",
                identity=None,
            )
            keys_by_consumer = relation.rename_target_axes(readiness_key_domain)
            if keys_by_consumer is None:
                raise AssertionError("producer-keyed readiness geometry must match")
            record_readiness_event(
                readiness_key_domain=readiness_key_domain,
                producers=(
                    ReadinessProducer(
                        producer_root=producer_root,
                        producer_site_id=producer_site_id,
                        producers_by_key=CoordinateRelation.identity(
                            readiness_key_domain,
                            producer_domain,
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=consumer_root,
                        consumer_site_id=consumer_site_id,
                        keys_by_consumer=keys_by_consumer,
                        covered_obligations=obligations,
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
        consumer_root, consumer_site_id, consumer_domain = consumer
        merged_relations: list[
            tuple[
                tuple[int, int | None, CoordinateDomain],
                CoordinateRelation,
                frozenset[DependencyObligation],
            ]
        ] = []
        readiness_key_axis_set: set[int] = set()
        quotient_is_supported = True
        for producer, relation_points in sorted(
            producers.items(),
            key=lambda item: (
                item[0][0],
                -1 if item[0][1] is None else item[0][1],
            ),
        ):
            relation, first_point = relation_points[0]
            obligations = {first_point}
            for next_relation, obligation in relation_points[1:]:
                union = relation.union(next_relation)
                if union is None:
                    quotient_is_supported = False
                    break
                relation = union
                obligations.add(obligation)
            if not quotient_is_supported:
                break
            relation = relation.coalesce_adjacent_target_boxes()
            used_axes = relation.source_axes_affecting_targets()
            if used_axes is None:
                quotient_is_supported = False
                break
            readiness_key_axis_set.update(used_axes)
            merged_relations.append((producer, relation, frozenset(obligations)))

        if not quotient_is_supported:
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=[
                    (producer, relation, frozenset((obligation,)))
                    for producer, relation_points in sorted(
                        producers.items(),
                        key=lambda item: (
                            item[0][0],
                            -1 if item[0][1] is None else item[0][1],
                        ),
                    )
                    for relation, obligation in relation_points
                ],
            )
            continue

        if any(
            left_points & right_points
            for left_index, (_left, _left_relation, left_points) in enumerate(
                merged_relations
            )
            for _right, _right_relation, right_points in merged_relations[
                left_index + 1 :
            ]
        ):
            # The same memory obligation was represented at more than one
            # producer site. These are alternative synchronization points,
            # not independent arrivals to one joined event.
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue

        readiness_key_axes = tuple(
            axis
            for axis in consumer_domain.axis_order
            if axis in readiness_key_axis_set
        )
        consumer_counts = consumer_domain.axis_count_expressions
        consumer_blocks = consumer_domain.block_sizes
        readiness_key_domain = CoordinateDomain(
            axis_order=readiness_key_axes,
            axis_counts_items=tuple(
                (axis, consumer_counts[axis]) for axis in readiness_key_axes
            ),
            block_sizes_items=tuple(
                (axis, consumer_blocks[axis])
                for axis in readiness_key_axes
                if axis in consumer_blocks
            ),
            kind="event",
        )
        keys_by_consumer = CoordinateRelation.projection(
            consumer_domain, readiness_key_domain
        )
        if keys_by_consumer is None:
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue
        event_producers: list[ReadinessProducer] = []
        covered_obligations: set[DependencyObligation] = set()
        for producer, relation, relation_points in merged_relations:
            producer_root, producer_site_id, _producer_domain = producer
            producers_by_key = relation.factor_through(keys_by_consumer)
            if producers_by_key is None:
                break
            event_producers.append(
                ReadinessProducer(
                    producer_root=producer_root,
                    producer_site_id=producer_site_id,
                    producers_by_key=producers_by_key,
                )
            )
            covered_obligations.update(relation_points)
        else:
            record_readiness_event(
                readiness_key_domain=readiness_key_domain,
                producers=tuple(event_producers),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=consumer_root,
                        consumer_site_id=consumer_site_id,
                        keys_by_consumer=keys_by_consumer,
                        covered_obligations=frozenset(covered_obligations),
                    ),
                ),
            )
            continue

        if len(event_producers) != len(merged_relations):
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue

    failed_consumers_by_producer: dict[int, dict[int, set[DependencyObligation]]] = {}
    for (
        producer_root,
        consumer_root,
    ), obligations in all_obligations_by_pair.items():
        remaining_obligations = obligations - represented_obligations
        if not remaining_obligations:
            continue
        failed_consumers_by_producer.setdefault(producer_root, {})[consumer_root] = (
            remaining_obligations
        )
    for producer_root, obligations_by_consumer in sorted(
        failed_consumers_by_producer.items()
    ):
        readiness_key_domain = CoordinateDomain(
            axis_order=(),
            axis_counts_items=(),
            kind="event",
        )
        producer_domain = root_domains[producer_root]
        consumers: list[ReadinessConsumer] = []
        for consumer_root, obligations in sorted(obligations_by_consumer.items()):
            consumers.append(
                ReadinessConsumer(
                    consumer_root=consumer_root,
                    consumer_site_id=None,
                    keys_by_consumer=CoordinateRelation.total(
                        root_domains[consumer_root],
                        readiness_key_domain,
                    ),
                    covered_obligations=frozenset(obligations),
                )
            )
        _record_readiness_event(
            pending_events,
            readiness_key_domain=readiness_key_domain,
            producers=(
                ReadinessProducer(
                    producer_root=producer_root,
                    producer_site_id=None,
                    producers_by_key=CoordinateRelation.total(
                        readiness_key_domain,
                        producer_domain,
                    ),
                ),
            ),
            consumers=tuple(consumers),
        )
    return tuple(pending_events.values())


def build_readiness_graph(
    dependency_graph: TileDependencyGraph,
    *,
    root_task_orders: tuple[CoordinateRelation, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    publishable_site_ids: frozenset[int] | None = None,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> ReadinessGraph:
    """Bind the symbolic readiness DAG for one selected configuration."""
    root_domains = tuple(task_order.target_domain for task_order in root_task_orders)
    events = build_readiness_events(
        dependency_graph,
        root_domains=root_domains,
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
        prove_nonnegative=prove_nonnegative,
    )
    return ReadinessGraph(
        root_task_orders=root_task_orders,
        events=events,
    )


def derive_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
) -> tuple[FinalArrivalContinuation, ...]:
    """Derive complete one-task-per-readiness-key continuation candidates."""
    required_obligations_by_root: dict[int, set[DependencyObligation]] = {}
    for event in readiness_graph.events:
        for readiness_consumer in event.consumers:
            required_obligations_by_root.setdefault(
                readiness_consumer.consumer_root, set()
            ).update(readiness_consumer.covered_obligations)

    candidates: list[
        tuple[
            int,
            int,
            int,
            tuple[tuple[int, CoordinateRelation], ...],
        ]
    ] = []
    for event in readiness_graph.events:
        if (
            event.root_barrier_producer_root is not None
            or len(event.consumers) != 1
            or any(
                readiness_producer.producer_site_id is not None
                for readiness_producer in event.producers
            )
        ):
            continue
        lowering_relations = _counter_lowering_relations(event)
        if lowering_relations is None:
            continue
        lowered_producers, lowered_consumers = lowering_relations
        consumer_index = 0
        (readiness_consumer,) = lowered_consumers
        if readiness_consumer.consumer_site_id is not None:
            continue
        candidate_plan = ReadinessCounterPlan(
            producers=lowered_producers,
            consumers=(readiness_consumer,),
            continuation_consumer_index=consumer_index,
        )
        if not _supports_exact_counter_plan_lowering(
            candidate_plan,
            readiness_graph.root_domains,
        ):
            continue
        if not readiness_consumer.covered_obligations.issuperset(
            required_obligations_by_root.get(readiness_consumer.consumer_root, ())
        ):
            continue
        producer_relations = _merge_relations_by_root(
            tuple(
                (
                    readiness_producer.producer_root,
                    cast("CoordinateRelation", readiness_producer.keys_by_producer),
                )
                for readiness_producer in lowered_producers
            )
        )
        if producer_relations is None or len(producer_relations) != len(
            {item.producer_root for item in lowered_producers}
        ):
            continue

        candidates.append(
            (
                readiness_consumer.consumer_root,
                event.event_id,
                consumer_index,
                producer_relations,
            )
        )

    conflicting_candidates: set[tuple[int, int]] = set()
    candidates_by_consumer_root: dict[int, list[tuple[int, int]]] = {}
    for consumer_root, event_id, consumer_index, *_rest in candidates:
        candidates_by_consumer_root.setdefault(consumer_root, []).append(
            (event_id, consumer_index)
        )
    for root_candidates in candidates_by_consumer_root.values():
        if len(root_candidates) > 1:
            conflicting_candidates.update(root_candidates)
    candidates_by_producer_root: dict[
        int,
        list[tuple[int, CoordinateRelation]],
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

    result: list[FinalArrivalContinuation] = []
    for (
        _consumer_root,
        event_id,
        consumer_index,
        _producer_relations,
    ) in sorted(candidates, key=operator.itemgetter(slice(3))):
        if (event_id, consumer_index) in conflicting_candidates:
            continue
        result.append(
            FinalArrivalContinuation(
                event_id=event_id,
                consumer_index=consumer_index,
            )
        )
    return tuple(result)


def order_continuation_producers_by_readiness_key(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> WorkerSchedule:
    """Order eligible static producers by readiness key.

    Readiness-key-major ordering completes one readiness key at a time so
    final-arrival work becomes ready as early as possible. It is legal only
    when one producer compactly enumerates a complete static task family; all
    other families keep their existing task order.
    """
    continuation_roots = {
        readiness_graph.event(continuation.event_id)
        .consumers[continuation.consumer_index]
        .consumer_root
        for continuation in continuations
    }
    replacement_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}

    for continuation in continuations:
        event = readiness_graph.event(continuation.event_id)
        if len(event.producers) != 1:
            continue
        readiness_producer = event.producers[0]
        if readiness_producer.producer_site_id is not None:
            continue
        root = readiness_producer.producer_root
        if root in continuation_roots or root in replacement_by_root:
            continue
        task_domain = readiness_graph.root_domains[root]
        task_order = readiness_producer.producers_by_key.enumerate_targets_by_source()
        if (
            task_order is None
            or task_order.target_domain != task_domain
            or task_order.source_domain.size != task_domain.size
        ):
            continue

        schedule_interval = worker_schedule.contiguous_global_interval(root)
        if (
            schedule_interval is None
            or schedule_interval[1] - schedule_interval[0] != task_domain.size
        ):
            continue
        replacement_by_root[root] = (
            WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=0,
                worker_count=worker_schedule.worker_count,
                dispatch_offset=schedule_interval[0],
            ),
        )

    if not replacement_by_root:
        return worker_schedule
    segments: list[WorkerScheduleSegment] = []
    inserted_roots: set[int] = set()
    for segment in worker_schedule.segments:
        replacement = replacement_by_root.get(segment.root)
        if replacement is None:
            segments.append(segment)
        elif segment.root not in inserted_roots:
            segments.extend(replacement)
            inserted_roots.add(segment.root)
    return WorkerSchedule(worker_schedule.worker_count, tuple(segments))


_MAX_GLOBAL_LIST_EDGES = 2_000_000
_MAX_GLOBAL_LIST_SEGMENTS = 4096


def _combined_producer_tasks_by_key(
    plan: ReadinessCounterPlan,
    producer_root: int,
    producer_domain: CoordinateDomain,
) -> CoordinateRelation | None:
    """Return the distinct root tasks behind each key for one producer root."""
    result: CoordinateRelation | None = None
    found = False
    for producer in plan.producers:
        if producer.producer_root != producer_root:
            continue
        found = True
        projected = producer.producers_by_key.project_target(producer_domain)
        if projected is None:
            return None
        result = projected if result is None else result.union(projected)
        if result is None:
            return None
    return result if found else None


def _validate_worker_schedule_tasks(
    worker_schedule: WorkerSchedule,
    root_task_orders: tuple[CoordinateRelation, ...],
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> bool:
    """Prove exact ownership, with explicitly excluded roots absent."""
    if any(root < 0 or root >= len(root_task_orders) for root in excluded_roots):
        return False
    for root, reference_task_order in enumerate(root_task_orders):
        root_domain = reference_task_order.target_domain
        segments = worker_schedule.segments_for_root(root)
        if root in excluded_roots:
            if segments:
                return False
            continue
        if not segments:
            return False
        if any(
            segment.task_order.source_domain != worker_schedule.placement_domain
            or segment.task_order.target_domain != root_domain
            for segment in segments
        ):
            return False
        placement = _root_task_placement_relation(worker_schedule, root)
        if placement is not None:
            continue
        # The traversal compatibility path uses concrete ordinal domains.
        # A parameterized schedule is valid only when its authoritative
        # placement relation itself proves exact ownership.
        if root_domain.parameter_symbols or any(
            segment.task_order.parameter_symbols for segment in segments
        ):
            return False
        if sum(segment.task_count for segment in segments) != root_domain.size:
            return False
        traversal = _root_schedule_traversal(segments, reference_task_order)
        if traversal is None or traversal.logical_task_to_scheduled_ordinal is None:
            return False
    return True


@dataclasses.dataclass(frozen=True)
class _ScheduledRootTraversal:
    """One symbolic view of the executable traversal for a scheduled root.

    The segment tuple is the source of truth.  This certificate is its cached
    symbolic derivation: codegen consumes the forward relation and ordinal
    ranges, while optimized-schedule legality checks require the exact inverse
    relation.  Some existing PID permutations cannot express either direction
    as one relation; codegen still consumes these certified ranges, but they
    are ineligible for optimized schedule acceptance.  Keeping every available
    view here prevents either side from rebuilding a subtly different
    traversal.
    """

    segment_ordinal_ranges: tuple[
        tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr], ...
    ]
    scheduled_ordinal_to_logical_task: CoordinateRelation | None
    logical_task_to_scheduled_ordinal: CoordinateRelation | None
    matches_reference: bool


@cache
def _logical_task_to_order_ordinal(
    task_order: CoordinateRelation,
    ordinal_domain: CoordinateDomain,
    *,
    ordinal_begin: int | sympy.Expr = 0,
) -> CoordinateRelation | None:
    """Invert one emitted traversal into its scalar ordinal coordinates.

    The native multidimensional relation and its flattened spelling describe
    the same ``task_order``. Different symbolic permutations can make only
    one of those forms invertible, so this is the single certificate used by
    both exact-once validation and progress-support checks.
    """
    task_count = task_order.source_domain.size_expr
    ordinal_count = ordinal_domain.size_expr
    ordinal_begin = sympy.sympify(ordinal_begin)
    if (
        len(ordinal_domain.axis_order) != 1
        or not tile_dependency._is_provably_nonnegative(ordinal_begin, None)
        or not tile_dependency._is_provably_nonnegative(
            sympy.simplify(ordinal_count - ordinal_begin - task_count),
            None,
        )
    ):
        return None
    if task_count.is_zero is True:
        if task_order.target_domain.size_expr.is_zero is not True:
            return None
        result = CoordinateRelation(
            source_domain=task_order.target_domain,
            target_domain=ordinal_domain,
            pieces=(),
        )
        converse = CoordinateRelation(
            source_domain=ordinal_domain,
            target_domain=task_order.target_domain,
            pieces=(),
        )
        tile_dependency._remember_exact_converse(result, converse)
        return result
    local_to_ordinal = CoordinateRelation.point_map(
        task_order.source_domain,
        ordinal_domain,
        (
            (
                tuple(
                    (
                        axis,
                        0,
                        task_order.source_domain.axis_count_expressions[axis],
                        1,
                    )
                    for axis in task_order.source_domain.axis_order
                ),
                (
                    ordinal_begin  # pyrefly: ignore[unsupported-operation]
                    + _flat_domain_index_expression(task_order.source_domain),
                ),
            ),
        ),
    )
    logical_to_local = task_order.converse()
    if logical_to_local is not None and not logical_to_local.is_single_valued():
        logical_to_local = None
    result: CoordinateRelation | None
    if logical_to_local is None:
        result = None
    elif len(task_order.source_domain.axis_order) == 1:
        (local_axis,) = task_order.source_domain.axis_order
        (ordinal_axis,) = ordinal_domain.axis_order
        if _equal_integer_expressions(
            ordinal_begin,
            0,
        ) and logical_to_local.target_domain == ordinal_domain:
            result = logical_to_local
        else:
            result = CoordinateRelation(
                source_domain=logical_to_local.source_domain,
                target_domain=ordinal_domain,
                pieces=tuple(
                    dataclasses.replace(
                        piece,
                        target_ranges=tuple(
                            (
                                ordinal_axis,
                                begin + ordinal_begin,  # pyrefly: ignore[unsupported-operation]
                                end + ordinal_begin,  # pyrefly: ignore[unsupported-operation]
                                step,
                            )
                            for axis, begin, end, step in piece.target_ranges
                            if axis == local_axis
                        ),
                    )
                    for piece in logical_to_local.pieces
                ),
            )
    else:
        result = logical_to_local.then(local_to_ordinal)
    if result is None:
        # A woven PID traversal may not have a representable multidimensional
        # inverse even though the inverse of its actual emitted order is exact.
        forward = _flat_task_order_relation(
            task_order,
            ordinal_domain,
            ordinal_begin=ordinal_begin,
        )
        result = None if forward is None else forward.converse()
    return result if result is not None and result.is_single_valued() else None


def _logical_task_to_reference_ordinal(
    reference_task_order: CoordinateRelation,
    ordinal_domain: CoordinateDomain,
) -> CoordinateRelation | None:
    """Map a logical CTA to its canonical PID ordinal symbolically."""
    if not _equal_integer_expressions(
        reference_task_order.source_domain.size_expr,
        ordinal_domain.size_expr,
    ):
        return None
    result = _logical_task_to_order_ordinal(
        reference_task_order,
        ordinal_domain,
    )
    return result if result is not None and result.is_total_function() else None


@cache
def _root_schedule_traversal(
    segments: tuple[WorkerScheduleSegment, ...],
    reference_task_order: CoordinateRelation,
) -> _ScheduledRootTraversal | None:
    """Derive the one symbolic traversal certificate consumed downstream.

    Codegen maps each concatenated segment ordinal through the corresponding
    ``segment.task_order`` before recovering the root's canonical PID.  The
    schedule therefore need not reproduce the reference traversal.  When the
    union of directly composed inverse relations is representable and total,
    equal cardinality proves exact-once ownership without materializing either
    domain.  Optimized candidates require that inverse.  Existing conservative
    PID permutations may retain only the segment ranges used by codegen.
    """
    root_domain = reference_task_order.target_domain
    if not segments:
        return None
    try:
        segment_task_counts = tuple(segment.task_count_expr for segment in segments)
    except ValueError:
        return None
    if not _equal_integer_expressions(
        sympy.Add(*segment_task_counts),
        root_domain.size_expr,
    ):
        return None
    ordinal_axis = (
        max(
            (
                *root_domain.axis_order,
                *(
                    axis
                    for segment in segments
                    for axis in segment.task_order.source_domain.axis_order
                ),
            ),
            default=0,
        )
        + 1
    )
    ordinal_domain = CoordinateDomain(
        axis_order=(ordinal_axis,),
        axis_counts_items=((ordinal_axis, root_domain.size_expr),),
        kind="task_order",
        _allow_empty=root_domain.size_expr.is_zero is True,
    )
    root_to_scheduled: CoordinateRelation | None = None
    scheduled_to_root: CoordinateRelation | None = None
    inverse_relation_supported = True
    forward_relation_supported = True
    segment_ordinal_ranges: list[
        tuple[WorkerScheduleSegment, sympy.Expr, sympy.Expr]
    ] = []
    ordinal_begin: sympy.Expr = sympy.Integer(0)
    for segment, segment_task_count in zip(
        segments,
        segment_task_counts,
        strict=True,
    ):
        logical_order = segment.logical_task_order
        if (
            logical_order is None
            or logical_order.target_domain != root_domain
            or not logical_order.is_total_function()
        ):
            return None
        forward_piece = _flat_task_order_relation(
            logical_order,
            ordinal_domain,
            ordinal_begin=ordinal_begin,
        )
        inverse_piece = _logical_task_to_order_ordinal(
            logical_order,
            ordinal_domain,
            ordinal_begin=ordinal_begin,
        )
        if inverse_piece is None or not inverse_relation_supported:
            inverse_relation_supported = False
            root_to_scheduled = None
        else:
            root_to_scheduled = (
                inverse_piece
                if root_to_scheduled is None
                else root_to_scheduled.union(inverse_piece)
            )
            if root_to_scheduled is None:
                inverse_relation_supported = False
        if forward_piece is None or not forward_relation_supported:
            forward_relation_supported = False
            scheduled_to_root = None
        else:
            scheduled_to_root = (
                forward_piece
                if scheduled_to_root is None
                else scheduled_to_root.union(forward_piece)
            )
            if scheduled_to_root is None:
                forward_relation_supported = False
        segment_ordinal_ranges.append(
            (
                segment,
                ordinal_begin,
                sympy.simplify(ordinal_begin + segment_task_count),
            )
        )
        ordinal_begin = sympy.simplify(ordinal_begin + segment_task_count)
    if (
        not _equal_integer_expressions(
            reference_task_order.source_domain.size_expr,
            root_domain.size_expr,
        )
        or not reference_task_order.is_total_function()
    ):
        return None
    if root_to_scheduled is not None and not root_to_scheduled.is_total_function():
        root_to_scheduled = None

    if scheduled_to_root is not None and not scheduled_to_root.is_single_valued():
        scheduled_to_root = None
    if scheduled_to_root is not None and not scheduled_to_root.is_total_function():
        scheduled_to_root = None
    if scheduled_to_root is not None:
        # Segment boundaries are placement details, not logical traversal
        # discontinuities.  Joining identical adjacent point maps can expose a
        # single mixed-radix bijection whose inverse is provable only over the
        # complete root domain (for example, complementary prefix/suffix
        # slices).  The joined forward relation remains derived directly from
        # the authoritative schedule relation.
        joined_forward = scheduled_to_root.coalesce_adjacent_source_boxes()
        if joined_forward.is_total_function():
            scheduled_to_root = joined_forward
        if root_to_scheduled is None:
            converse = scheduled_to_root.converse()
            if (
                converse is not None
                and converse.is_single_valued()
                and converse.is_total_function()
            ):
                root_to_scheduled = converse
    reference_ordinal = _logical_task_to_reference_ordinal(
        reference_task_order,
        ordinal_domain,
    )

    return _ScheduledRootTraversal(
        segment_ordinal_ranges=tuple(segment_ordinal_ranges),
        scheduled_ordinal_to_logical_task=scheduled_to_root,
        logical_task_to_scheduled_ordinal=root_to_scheduled,
        matches_reference=(
            root_to_scheduled is not None
            and reference_ordinal is not None
            and (
                root_to_scheduled.is_pointwise_equal_to(reference_ordinal)
                or root_to_scheduled.is_pointwise_equal_on_same_support(
                    reference_ordinal
                )
            )
        ),
    )


def root_schedule_matches_reference(
    segments: tuple[WorkerScheduleSegment, ...],
    reference_task_order: CoordinateRelation,
) -> bool:
    """Prove that a segmented traversal preserves canonical PID order."""
    traversal = _root_schedule_traversal(segments, reference_task_order)
    return traversal is not None and traversal.matches_reference


def _task_step_relations(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[CoordinateRelation | None, ...] | None:
    """Return logical-task-to-wave functions from schedule converses."""
    result: list[CoordinateRelation | None] = []
    for root in range(len(readiness_graph.root_task_orders)):
        if root in excluded_roots:
            result.append(None)
            continue
        task_steps = _root_task_wave_relation(worker_schedule, root)
        if task_steps is None:
            return None
        result.append(task_steps)
    return tuple(result)


def _keys_by_consumer_root_task(
    readiness_graph: ReadinessGraph,
    consumer: ReadinessConsumer,
) -> CoordinateRelation | None:
    """Project every nested checkpoint requirement onto its owning CTA."""
    root_domain = readiness_graph.root_domains[consumer.consumer_root]
    if consumer.keys_by_consumer.source_domain == root_domain:
        return consumer.keys_by_consumer
    return consumer.keys_by_consumer.project_source(root_domain)


@cache
def _external_source_frontiers(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    source_root: int,
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    """Map resident tasks to their latest required source ticket.

    This frontier is a scheduling priority only.  Source tickets are issued
    before resident tickets, which proves progress, but only runtime counters
    establish that the corresponding source work has completed.
    """
    source_segment = _transient_source_schedule_segment(
        worker_schedule,
        source_root,
    )
    ticket_to_logical = (
        None if source_segment is None else source_segment.logical_task_order
    )
    continuations = _emitted_final_arrival_continuations(
        readiness_graph,
        readiness_counters,
    )
    if ticket_to_logical is None or continuations is None:
        return None
    ticket_domain = ticket_to_logical.source_domain
    ticket_identity = CoordinateRelation.identity(ticket_domain, ticket_domain)
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph,
        continuations,
    )
    parts_by_consumer: dict[int, list[CoordinateRelation]] = {}
    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        consumer_root = prerequisite.consumer_root
        if prerequisite.barrier_producer_root is not None:
            producer_domain = readiness_graph.root_domains[
                prerequisite.barrier_producer_root
            ]
            static_relations = _static_producer_relations(
                readiness_graph,
                root=prerequisite.barrier_producer_root,
                site_id=None,
                readiness_keys=CoordinateRelation.identity(
                    producer_domain,
                    producer_domain,
                ),
                continuation_by_root=continuation_by_root,
            )
            if static_relations is None:
                return None
            if any(root == source_root for root, _relation in static_relations):
                consumer_domain = readiness_graph.root_domains[consumer_root]
                parts_by_consumer.setdefault(consumer_root, []).append(
                    CoordinateRelation.point_map(
                        consumer_domain,
                        ticket_domain,
                        (
                            (
                                tuple(
                                    (
                                        axis,
                                        0,
                                        consumer_domain.axis_counts[axis],
                                        1,
                                    )
                                    for axis in consumer_domain.axis_order
                                ),
                                (sympy.Integer(ticket_domain.size - 1),),
                            ),
                        ),
                    )
                )
            continue

        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        static_relations = _readiness_static_producers(
            readiness_graph,
            plan.producers,
            continuation_by_root,
        )
        if consumer_keys is None or static_relations is None:
            return None
        source_relations = tuple(
            relation for root, relation in static_relations if root == source_root
        )
        if not source_relations:
            continue
        source_keys = source_relations[0]
        for relation in source_relations[1:]:
            union = source_keys.union(relation)
            if union is None:
                return None
            source_keys = union
        keys_by_ticket = ticket_to_logical.then(source_keys)
        if keys_by_ticket is not None:
            compact_keys_by_ticket = keys_by_ticket.coalesce_adjacent_source_boxes(
                fold_static_offsets=True
            )
            if compact_keys_by_ticket.is_total_function():
                keys_by_ticket = compact_keys_by_ticket
        frontier_by_key = (
            None
            if keys_by_ticket is None
            else _maximum_value_by_key(
                keys_by_ticket,
                ticket_identity,
            )
        )
        frontier = (
            None
            if frontier_by_key is None
            else _maximum_required_value_by_consumer(
                consumer_keys,
                frontier_by_key,
            )
        )
        if frontier is None:
            return None
        parts_by_consumer.setdefault(consumer_root, []).append(frontier)

    result: list[tuple[int, CoordinateRelation]] = []
    for consumer_root, parts in sorted(parts_by_consumer.items()):
        combined = parts[0]
        for part in parts[1:]:
            union = combined.union(part)
            if union is None:
                return None
            combined = union
        frontier = combined.max_target_value_by_source(ticket_identity)
        if frontier is None or not frontier.is_total_function():
            return None
        canonical_frontier = frontier.canonical_single_valued()
        if canonical_frontier is None or any(
            step != 1 or sympy.simplify(end - begin) != 1  # pyrefly: ignore[unsupported-operation]
            for piece in canonical_frontier.pieces
            for _axis, begin, end, step in piece.target_ranges
        ):
            return None
        simplified_frontier = CoordinateRelation.point_map(
            canonical_frontier.source_domain,
            canonical_frontier.target_domain,
            tuple(
                (
                    piece.source_bounds_items,
                    tuple(
                        _simplify_logical_expression(
                            begin,
                            domain=canonical_frontier.source_domain,
                            source_bounds=piece.source_bounds_items,
                        )
                        for _axis, begin, _end, _step in piece.target_ranges
                    ),
                )
                for piece in canonical_frontier.pieces
            ),
        )
        if simplified_frontier.is_total_function():
            frontier = simplified_frontier
        result.append((consumer_root, frontier))
    return tuple(result)


def _expression_is_nondecreasing_in(
    expression: sympy.Expr,
    source_symbol: sympy.Symbol,
) -> bool:
    """Conservatively prove monotonicity in one integer coordinate.

    This recognizes only order-preserving operations in the logical relation
    IR.  In particular, ``floor`` and ``ceiling`` preserve the order of their
    argument, whereas ``Mod`` is deliberately unsupported because it wraps.
    The proof is structural and therefore does not inspect concrete source
    points.
    """
    expression = sympy.simplify(expression)
    if source_symbol not in expression.free_symbols:
        return not expression.free_symbols
    if expression.free_symbols != {source_symbol}:
        return False
    if expression == source_symbol:
        return True
    if isinstance(expression, sympy.Add):
        return all(
            _expression_is_nondecreasing_in(term, source_symbol)
            for term in expression.args
        )
    if isinstance(expression, sympy.Mul):
        coefficient = sympy.Integer(1)
        varying_terms: list[sympy.Expr] = []
        for factor in expression.args:
            if factor.free_symbols:
                varying_terms.append(cast("sympy.Expr", factor))
            else:
                coefficient *= factor  # pyrefly: ignore[unsupported-operation]
        return (
            len(varying_terms) == 1
            and coefficient.is_real is True
            and coefficient >= 0
            and _expression_is_nondecreasing_in(
                varying_terms[0],
                source_symbol,
            )
        )
    if expression.func in (sympy.floor, sympy.ceiling):
        return _expression_is_nondecreasing_in(
            cast("sympy.Expr", expression.args[0]),
            source_symbol,
        )
    if expression.func in (sympy.Min, sympy.Max):
        return all(
            _expression_is_nondecreasing_in(cast("sympy.Expr", argument), source_symbol)
            for argument in expression.args
        )
    return False


def _scalar_relation_is_nondecreasing(relation: CoordinateRelation) -> bool:
    """Prove a total scalar relation is nondecreasing in source order."""
    canonical = relation.canonical_single_valued()
    if canonical is None:
        return False
    canonical = canonical.coalesce_adjacent_source_boxes()
    if (
        not canonical.is_total_function()
        or len(canonical.source_domain.axis_order) != 1
        or len(canonical.target_domain.axis_order) != 1
    ):
        return False
    (source_axis,) = canonical.source_domain.axis_order
    source_symbol = coordinate_axis_symbol(source_axis)
    cursor = 0
    previous_last: sympy.Expr | None = None
    for piece in sorted(
        canonical.pieces,
        key=lambda item: item.source_bounds_items[0][1],
    ):
        if len(piece.source_bounds_items) != 1 or len(piece.target_ranges) != 1:
            return False
        piece_axis, begin, end, source_step = piece.source_bounds_items[0]
        _target_axis, value, value_end, target_step = piece.target_ranges[0]
        if (
            piece_axis != source_axis
            or begin != cursor
            or source_step != 1
            or target_step != 1
            or sympy.simplify(value_end - value) != 1  # pyrefly: ignore[unsupported-operation]
        ):
            return False
        first = sympy.simplify(value.xreplace({source_symbol: sympy.Integer(begin)}))
        last = sympy.simplify(value.xreplace({source_symbol: sympy.Integer(end - 1)}))
        if (
            first.free_symbols
            or last.free_symbols
            or first.is_integer is not True
            or last.is_integer is not True
            or (previous_last is not None and first < previous_last)
        ):
            return False
        if end - begin > 1 and not _expression_is_nondecreasing_in(
            value,
            source_symbol,
        ):
            return False
        previous_last = last
        cursor = end
    return cursor == canonical.source_domain.size


def _consumer_major_producer_order(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    excluded_roots: frozenset[int],
    external_source_frontiers: tuple[tuple[int, CoordinateRelation], ...] = (),
) -> WorkerSchedule:
    """Normalize readiness cohorts for the global-list proposal.

    A partially ready root is ordered first by the event that admits each of
    its CTAs.  A root without such an incoming counter may instead be ordered
    by the downstream consumer whose fan-in it completes.  Both orders come
    exclusively from emitted readiness relations and are accepted only as
    symbolic exact-once permutations of a complete root.  This normalization
    is ephemeral: a rejected global proposal leaves the input schedule intact.
    """
    continuations = _emitted_final_arrival_continuations(
        readiness_graph,
        readiness_counters,
    )
    if continuations is None:
        return worker_schedule
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph,
        continuations,
    )
    ordinal_axis = (
        max(
            axis
            for task_order in readiness_graph.root_task_orders
            for domain in (task_order.source_domain, task_order.target_domain)
            for axis in domain.axis_order
        )
        + 1
    )
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )

    def exact_task_order(
        grouped_tasks: CoordinateRelation | None,
        task_domain: CoordinateDomain,
    ) -> tuple[CoordinateRelation, CoordinateRelation] | None:
        task_order = (
            None
            if grouped_tasks is None
            else grouped_tasks.enumerate_targets_by_source()
        )
        ordinal_domain = CoordinateDomain(
            axis_order=(ordinal_axis,),
            axis_counts_items=((ordinal_axis, task_domain.size),),
            kind="task_order",
        )
        if (
            task_order is None
            or task_order.target_domain != task_domain
            or task_order.source_domain.size != task_domain.size
            or not task_order.is_total_function()
            or len(task_order.pieces) > tile_dependency._MAX_RELATION_PIECES
        ):
            return None
        ordinal = _logical_task_to_order_ordinal(task_order, ordinal_domain)
        if ordinal is None or not ordinal.is_total_function():
            return None
        return task_order, ordinal

    def unique_orders(
        candidates: dict[
            int,
            list[tuple[CoordinateRelation, CoordinateRelation]],
        ],
        unsupported: set[int],
    ) -> dict[int, CoordinateRelation]:
        result: dict[int, CoordinateRelation] = {}
        for root, root_candidates in candidates.items():
            if root in unsupported:
                continue
            reference_ordinal = root_candidates[0][1]
            if any(
                not candidate_ordinal.is_pointwise_equal_to(reference_ordinal)
                for _task_order, candidate_ordinal in root_candidates[1:]
            ):
                continue
            result[root] = root_candidates[0][0]
        return result

    def replace_dense_orders(
        schedule: WorkerSchedule,
        task_orders: dict[int, CoordinateRelation],
    ) -> WorkerSchedule:
        replacements: dict[int, WorkerScheduleSegment] = {}
        for root, task_order in task_orders.items():
            schedule_interval = schedule.contiguous_global_interval(root)
            if (
                schedule_interval is None
                or schedule_interval[1] - schedule_interval[0]
                != readiness_graph.root_domains[root].size
            ):
                continue
            replacements[root] = WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=0,
                worker_count=schedule.worker_count,
                dispatch_offset=schedule_interval[0],
            )
        if not replacements:
            return schedule
        segments: list[WorkerScheduleSegment] = []
        inserted_roots: set[int] = set()
        for segment in schedule.segments:
            replacement = replacements.get(segment.root)
            if replacement is None:
                segments.append(segment)
            elif segment.root not in inserted_roots:
                segments.append(replacement)
                inserted_roots.add(segment.root)
        try:
            return WorkerSchedule(schedule.worker_count, tuple(segments))
        except ValueError:
            # Reordering is a speculative optimization.  If the proposed
            # dense orders cannot be normalized into one exact ownership
            # relation, retain the complete input schedule atomically.
            return schedule

    # First derive counter-admission cohorts.  An exact earlier-stage source
    # frontier may replace that order below when source release is the tighter
    # constraint; otherwise a newly released subset remains compact before
    # downstream completion is optimized.
    incoming_counter_roots: set[int] = set()
    admission_candidates: dict[
        int,
        list[tuple[CoordinateRelation, CoordinateRelation]],
    ] = {}
    unsupported_admission_roots: set[int] = set()
    for prerequisite in prerequisites:
        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        if plan is None or consumer is None:
            continue
        consumer_root = consumer.consumer_root
        if consumer_root in excluded_roots:
            continue
        incoming_counter_roots.add(consumer_root)
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        tasks_by_key = None if consumer_keys is None else consumer_keys.converse()
        candidate = exact_task_order(
            tasks_by_key,
            readiness_graph.root_domains[consumer_root],
        )
        if candidate is None:
            unsupported_admission_roots.add(consumer_root)
        else:
            admission_candidates.setdefault(consumer_root, []).append(candidate)

    admission_orders = unique_orders(
        admission_candidates,
        unsupported_admission_roots,
    )
    source_orders: dict[int, CoordinateRelation] = {}
    for root, source_frontier in external_source_frontiers:
        canonical_frontier = source_frontier.canonical_single_valued()
        canonical_frontier = (
            None
            if canonical_frontier is None
            else canonical_frontier.coalesce_adjacent_source_boxes()
        )
        affecting_axes = (
            None
            if canonical_frontier is None
            else canonical_frontier.source_axes_affecting_targets()
        )
        if affecting_axes is None or len(affecting_axes) != 1:
            continue
        (frontier_axis,) = affecting_axes
        root_domain = readiness_graph.root_domains[root]
        source_axis_order = (
            *(axis for axis in root_domain.axis_order if axis != frontier_axis),
            frontier_axis,
        )
        grouped_tasks = CoordinateRelation.identity(
            root_domain,
            root_domain,
        ).reorder_source_axes(source_axis_order)
        candidate = exact_task_order(grouped_tasks, root_domain)
        if candidate is None:
            continue
        task_order, logical_to_ordinal = candidate
        ordinal_to_logical = logical_to_ordinal.converse()
        ordered_frontier = (
            None
            if ordinal_to_logical is None
            else ordinal_to_logical.then(source_frontier)
        )
        if ordered_frontier is not None and _scalar_relation_is_nondecreasing(
            ordered_frontier
        ):
            source_orders[root] = task_order
    # A source-major order is accepted only with its own monotonicity and
    # exact-bijection certificates.  The final all-prerequisite rank proof
    # remains authoritative for resident dependencies.
    selected_admission_orders = {
        **admission_orders,
        **source_orders,
    }
    normalized_schedule = replace_dense_orders(
        worker_schedule,
        selected_admission_orders,
    )

    completion_candidates: dict[
        int,
        list[tuple[CoordinateRelation, CoordinateRelation]],
    ] = {}
    unsupported_completion_roots: set[int] = set()
    for prerequisite in prerequisites:
        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        if plan is None or consumer is None:
            # A whole-root barrier carries no useful cohort order.
            continue
        static_relations = _readiness_static_producers(
            readiness_graph,
            plan.producers,
            continuation_by_root,
        )
        if static_relations is None:
            continue
        consumer_segments = normalized_schedule.segments_for_root(
            consumer.consumer_root
        )
        consumer_traversal = _root_schedule_traversal(
            consumer_segments,
            readiness_graph.root_task_orders[consumer.consumer_root],
        )
        consumer_order = (
            None
            if consumer_traversal is None
            else consumer_traversal.scheduled_ordinal_to_logical_task
        )
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        ordered_consumer_keys = (
            None
            if consumer_order is None or consumer_keys is None
            else consumer_order.then(consumer_keys)
        )
        for producer_root, keys_by_producer in static_relations:
            if (
                producer_root in excluded_roots
                or producer_root in incoming_counter_roots
            ):
                continue
            producers_by_key = keys_by_producer.converse()
            producers_by_consumer = (
                None
                if ordered_consumer_keys is None or producers_by_key is None
                else ordered_consumer_keys.then(producers_by_key)
            )
            candidate = exact_task_order(
                producers_by_consumer,
                readiness_graph.root_domains[producer_root],
            )
            if candidate is None:
                unsupported_completion_roots.add(producer_root)
                continue
            completion_candidates.setdefault(producer_root, []).append(candidate)

    completion_orders = unique_orders(
        completion_candidates,
        unsupported_completion_roots,
    )
    return replace_dense_orders(normalized_schedule, completion_orders)


def _keys_at_first_consumer_checkpoint(
    readiness_graph: ReadinessGraph,
    consumer: ReadinessConsumer,
) -> CoordinateRelation | None:
    """Map each owning CTA to the keys waited on at nested iteration zero."""
    root_domain = readiness_graph.root_domains[consumer.consumer_root]
    site_domain = consumer.keys_by_consumer.source_domain
    if site_domain == root_domain:
        return consumer.keys_by_consumer
    nested_axes = nested_logical_axes(root_domain, site_domain)
    if len(nested_axes) != 1:
        return None
    root_axes = frozenset(root_domain.axis_order)
    root_to_checkpoint = CoordinateRelation.point_map(
        root_domain,
        site_domain,
        (
            (
                tuple(
                    (axis, 0, root_domain.axis_counts[axis], 1)
                    for axis in root_domain.axis_order
                ),
                tuple(
                    coordinate_axis_symbol(axis)
                    if axis in root_axes
                    else sympy.Integer(0)
                    for axis in site_domain.axis_order
                ),
            ),
        ),
    )
    return root_to_checkpoint.then(consumer.keys_by_consumer)


def _producer_frontier_by_consumer_task(
    readiness_graph: ReadinessGraph,
    *,
    worker_schedule: WorkerSchedule,
    consumer_keys: CoordinateRelation,
    producer: ReadinessProducer,
) -> CoordinateRelation | None:
    """Map each consumer CTA to the latest required producer wave."""
    producer_domain = readiness_graph.root_domains[producer.producer_root]
    keys_by_task = producer.keys_by_producer
    if keys_by_task is None:
        return None
    if keys_by_task.source_domain != producer_domain:
        keys_by_task = keys_by_task.project_source(producer_domain)
    if keys_by_task is None:
        return None
    key_frontier = _maximum_root_wave_by_key(
        worker_schedule,
        producer.producer_root,
        keys_by_task,
    )
    if key_frontier is None:
        return None
    # Each producer arm is proved separately.  A union of partial frontiers
    # must never masquerade as a total combined prerequisite.
    return _maximum_required_value_by_consumer(
        consumer_keys,
        key_frontier,
    )


def _all_tasks_frontier(
    consumer_domain: CoordinateDomain,
    producer_steps: CoordinateRelation,
) -> CoordinateRelation | None:
    """Return a conservative symbolic frontier for a whole-root barrier."""
    if not producer_steps.is_total_function():
        return None
    bounds = producer_steps.value_bounds()
    if bounds is None:
        return None
    rank_domain = producer_steps.target_domain
    return CoordinateRelation.point_map(
        consumer_domain,
        rank_domain,
        (
            (
                tuple(
                    (axis, 0, consumer_domain.axis_counts[axis], 1)
                    for axis in consumer_domain.axis_order
                ),
                (sympy.Integer(bounds[1]),),
            ),
        ),
    )


def _has_symbolic_worker_rank(
    worker_schedule: WorkerSchedule,
) -> bool:
    """Prove every resident strand edge strictly increases worker step."""
    prior_runs: list[tuple[int, int, int]] = []
    for segment in worker_schedule.segments:
        for begin, end, first_step, last_step in segment.worker_step_runs():
            if any(
                max(begin, prior_begin) < min(end, prior_end)
                and first_step <= prior_last_step
                for prior_begin, prior_end, prior_last_step in prior_runs
            ):
                return False
            prior_runs.append((begin, end, last_step))
    return True


def _segments_share_worker(
    left: WorkerScheduleSegment,
    right: WorkerScheduleSegment,
) -> bool:
    """Return whether two symbolic segment occurrences share a worker."""
    return any(
        max(left_begin, right_begin) < min(left_end, right_end)
        for left_begin, left_end in left.worker_intervals()
        for right_begin, right_end in right.worker_intervals()
    )


@cache
def _segment_dependency_support_overlaps(
    producer_segment: WorkerScheduleSegment,
    keys_by_producer: CoordinateRelation,
    consumer_segment: WorkerScheduleSegment,
    keys_by_consumer: CoordinateRelation,
) -> bool | None:
    """Prove whether two segment occurrences can share a readiness key.

    The finite graph below contains segment occurrences, never logical CTAs.
    These compositions restrict the canonical producer/consumer relations by
    each segment's symbolic task support.  A failure to derive an exact
    relation declines the candidate instead of guessing from materialized
    points.
    """
    consumer_order = consumer_segment.logical_task_order
    producer_order = producer_segment.logical_task_order
    if consumer_order is None or producer_order is None:
        return None
    consumer_segment_keys = consumer_order.then(keys_by_consumer)
    consumers_by_key = (
        None if consumer_segment_keys is None else consumer_segment_keys.converse()
    )
    producers_reaching_consumer = (
        None if consumers_by_key is None else keys_by_producer.then(consumers_by_key)
    )
    # Restrict by the same exact traversal certificate used for global
    # exact-once validation; only its logical source support matters here.
    producer_support = _logical_task_to_order_ordinal(
        producer_order,
        _task_order_ordinal_domain(producer_order),
    )
    if producers_reaching_consumer is None or producer_support is None:
        return None
    if producer_support.source_domain != producers_reaching_consumer.source_domain:
        return None
    return not producer_support.has_disjoint_source_support(producers_reaching_consumer)


def _has_acyclic_symbolic_segment_precedence(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    continuation_by_root: dict[int, FinalArrivalContinuation],
    excluded_roots: frozenset[int],
) -> bool:
    """Prove global progress on the finite symbolic segment graph.

    A node is one compressed schedule segment.  Edges encode both resident
    strand order and emitted waits.  Every concrete CTA execution edge maps
    to one of these symbolic edges (or stays within one segment), so an
    acyclic segment graph is a conservative certificate that independently
    rules out cycles assembled from several individually safe waits.
    """
    segments = worker_schedule.segments
    successors: list[set[int]] = [set() for _ in segments]
    edge_attempt_count = 0
    segments_by_root: dict[int, list[int]] = {}
    for index, segment in enumerate(segments):
        segments_by_root.setdefault(segment.root, []).append(index)

    def add_edge(producer_index: int, consumer_index: int) -> bool:
        nonlocal edge_attempt_count
        edge_attempt_count += 1
        if edge_attempt_count > _MAX_GLOBAL_LIST_EDGES:
            return False
        if producer_index == consumer_index:
            return False
        successors[producer_index].add(consumer_index)
        return True

    # Tuple order is executable order.  Any two occurrences sharing a worker
    # therefore induce this strand precedence; nonconsecutive edges are merely
    # transitive edges and cannot manufacture a cycle.
    for earlier_index, earlier in enumerate(segments):
        for later_index in range(earlier_index + 1, len(segments)):
            edge_attempt_count += 1
            if edge_attempt_count > _MAX_GLOBAL_LIST_EDGES:
                return False
            if _segments_share_worker(earlier, segments[later_index]):
                successors[earlier_index].add(later_index)

    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        consumer_indices = segments_by_root.get(prerequisite.consumer_root)
        if not consumer_indices:
            return False
        if prerequisite.barrier_producer_root is not None:
            producer_root = prerequisite.barrier_producer_root
            producer_domain = readiness_graph.root_domains[producer_root]
            static_relations = _static_producer_relations(
                readiness_graph,
                root=producer_root,
                site_id=None,
                readiness_keys=CoordinateRelation.identity(
                    producer_domain,
                    producer_domain,
                ),
                continuation_by_root=continuation_by_root,
            )
            if static_relations is None:
                return False
            for static_root, _relation in static_relations:
                if static_root in excluded_roots:
                    continue
                producer_indices = segments_by_root.get(static_root)
                if not producer_indices:
                    return False
                for producer_index in producer_indices:
                    for consumer_index in consumer_indices:
                        if not add_edge(producer_index, consumer_index):
                            return False
            continue

        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        static_relations = _readiness_static_producers(
            readiness_graph,
            plan.producers,
            continuation_by_root,
        )
        if consumer_keys is None or static_relations is None:
            return False
        for static_root, keys_by_producer in static_relations:
            if static_root in excluded_roots:
                continue
            producer_indices = segments_by_root.get(static_root)
            if not producer_indices:
                return False
            for producer_index in producer_indices:
                producer_segment = segments[producer_index]
                for consumer_index in consumer_indices:
                    edge_attempt_count += 1
                    if edge_attempt_count > _MAX_GLOBAL_LIST_EDGES:
                        return False
                    overlaps = _segment_dependency_support_overlaps(
                        producer_segment,
                        keys_by_producer,
                        segments[consumer_index],
                        consumer_keys,
                    )
                    if overlaps is None:
                        return False
                    if overlaps:
                        if producer_index == consumer_index:
                            return False
                        successors[producer_index].add(consumer_index)

    indegree = [0] * len(segments)
    for segment_successors in successors:
        for successor in segment_successors:
            indegree[successor] += 1
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        index = ready.pop()
        visited += 1
        for successor in successors[index]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return visited == len(segments)


def _schedule_is_progress_safe(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    transient_source_root: int | None = None,
    require_strict_rank: bool = False,
) -> bool:
    """Prove progress symbolically, without constructing a CTA DAG.

    Resident strand edges and ordinary semantic edges strictly increase the
    scalar worker-step rank. A source-first transient root occupies an earlier
    ticket role and has no waits. Its edges into resident work are therefore
    progress-safe even when the resident CTA reaches its emitted wait before
    the source finishes; all source tickets have already been issued.  When a
    disjoint-worker wait intentionally violates the scalar rank, one global
    segment-precedence DAG proves that several such waits cannot form a cycle.
    """
    continuations = _emitted_final_arrival_continuations(
        readiness_graph,
        readiness_counters,
    )
    if continuations is None:
        return False
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph, continuations
    )
    continuation_roots = frozenset(continuation_by_root)
    excluded_roots = continuation_roots | (
        frozenset()
        if transient_source_root is None
        else frozenset((transient_source_root,))
    )
    if (
        (
            transient_source_root is not None
            and _transient_source_schedule_segment(
                worker_schedule,
                transient_source_root,
            )
            is None
        )
        or any(worker_schedule.segments_for_root(root) for root in continuation_roots)
        or not _has_symbolic_worker_rank(worker_schedule)
    ):
        return False
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )
    if any(
        prerequisite.consumer_root in excluded_roots for prerequisite in prerequisites
    ):
        return False
    task_steps = _task_step_relations(
        worker_schedule,
        readiness_graph,
        excluded_roots=excluded_roots,
    )
    if task_steps is None:
        return False

    needs_segment_precedence_proof = False

    for prerequisite in prerequisites:
        if prerequisite.barrier_producer_root is not None:
            producer_root = prerequisite.barrier_producer_root
            consumer_root = prerequisite.consumer_root
            consumer_domain = readiness_graph.root_domains[consumer_root]
            consumer_steps = task_steps[consumer_root]
            producer_domain = readiness_graph.root_domains[producer_root]
            static_relations = _static_producer_relations(
                readiness_graph,
                root=producer_root,
                site_id=None,
                readiness_keys=CoordinateRelation.identity(
                    producer_domain,
                    producer_domain,
                ),
                continuation_by_root=continuation_by_root,
            )
            if static_relations is None or consumer_steps is None:
                return False
            strictly_ranked = True
            for static_root, _relation in static_relations:
                if static_root == transient_source_root:
                    # Every source ticket is issued before any resident ticket.
                    # The barrier still gates completion and visibility at
                    # runtime, but launch-stage order proves progress.
                    continue
                producer_steps = task_steps[static_root]
                if producer_steps is None:
                    return False
                frontier = _all_tasks_frontier(
                    consumer_domain,
                    producer_steps,
                )
                if (
                    frontier is None
                    or not frontier.is_total_function()
                    or (not frontier.is_pointwise_strictly_less_than(consumer_steps))
                ):
                    strictly_ranked = False
            if not strictly_ranked and require_strict_rank:
                return False
            if not strictly_ranked:
                needs_segment_precedence_proof = True
            continue

        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        consumer_keys = (
            _keys_by_consumer_root_task(readiness_graph, consumer)
            if require_strict_rank or consumer.consumer_site_id is None
            else _keys_at_first_consumer_checkpoint(readiness_graph, consumer)
        )
        if consumer_keys is None:
            return False
        static_relations = _readiness_static_producers(
            readiness_graph,
            plan.producers,
            continuation_by_root,
        )
        if static_relations is None:
            return False
        strictly_ranked = True
        for producer_root, keys_by_producer in static_relations:
            if producer_root == transient_source_root:
                # This arm occupies the earlier launch stage.  In a mixed
                # join, every resident arm is still proved below.
                continue
            producers_by_key = keys_by_producer.converse()
            if producers_by_key is None:
                return False
            producer_steps = task_steps[producer_root]
            consumer_steps = task_steps[consumer.consumer_root]
            if producer_steps is None or consumer_steps is None:
                return False
            frontier = _producer_frontier_by_consumer_task(
                readiness_graph,
                worker_schedule=worker_schedule,
                consumer_keys=consumer_keys,
                producer=ReadinessProducer(
                    producer_root=producer_root,
                    producers_by_key=producers_by_key,
                ),
            )
            if frontier is None:
                strictly_ranked = False
                continue
            if frontier.is_pointwise_strictly_less_than_where_defined(consumer_steps):
                continue
            strictly_ranked = False
        if not strictly_ranked and require_strict_rank:
            return False
        if not require_strict_rank and (
            consumer.consumer_site_id is not None or not strictly_ranked
        ):
            # A nested consumer can wait again after admission.  The global
            # segment graph includes all checkpoints, whereas the scalar fast
            # path above intentionally checks only entry readiness.
            needs_segment_precedence_proof = True
    return not needs_segment_precedence_proof or (
        _has_acyclic_symbolic_segment_precedence(
            worker_schedule,
            readiness_graph,
            readiness_counters,
            root_barrier_edges,
            continuation_by_root=continuation_by_root,
            excluded_roots=excluded_roots,
        )
    )


def _schedule_has_strict_root_slot_progress(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    transient_source_root: int | None = None,
) -> bool:
    """Prove exact root-entry counter waits increase resident global slots.

    This intentionally declines root barriers, nested execution sites,
    final-arrival continuations, and a separate source launch stage.  Those
    mechanisms need a rank spanning more than resident root-entry slots.
    """
    if root_barrier_edges or transient_source_root is not None:
        return False
    continuations = _emitted_final_arrival_continuations(
        readiness_graph,
        readiness_counters,
    )
    if continuations is None or continuations:
        return False
    if (
        any(
            not _supports_exact_counter_plan_lowering(
                plan,
                readiness_graph.root_domains,
            )
            for plan in readiness_counters
        )
        or any(
            producer.producer_site_id is not None
            for plan in readiness_counters
            for producer in plan.producers
        )
        or any(
            consumer.consumer_site_id is not None
            for plan in readiness_counters
            for consumer in plan.consumers
        )
    ):
        return False

    dependencies: list[tuple[int, int, CoordinateRelation]] = []
    for prerequisite in _emitted_prerequisites(readiness_counters, frozenset()):
        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        consumer_domain = readiness_graph.root_domains[consumer.consumer_root]
        if consumer.keys_by_consumer.source_domain != consumer_domain:
            return False
        static_relations = _readiness_static_producers(
            readiness_graph,
            plan.producers,
            {},
        )
        if static_relations is None:
            return False
        for producer_root, keys_by_producer in static_relations:
            producers_by_key = keys_by_producer.converse()
            producers_by_consumer = (
                None
                if producers_by_key is None
                else consumer.keys_by_consumer.then(producers_by_key)
            )
            if producers_by_consumer is None:
                return False
            dependencies.append(
                (
                    producer_root,
                    consumer.consumer_root,
                    producers_by_consumer,
                )
            )
    return _strict_root_slot_progress(worker_schedule, tuple(dependencies))


@dataclasses.dataclass(frozen=True)
class _PlacedRun:
    root: int
    source_begin: int
    task_count: int
    worker_begin: int
    worker_count: int
    worker_step: int


def _root_schema_criticality(
    root_count: int,
    edges: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    """Return shape-independent ``(slack, -top)`` classes for root DAGs."""
    successors = [set() for _ in range(root_count)]
    indegree = [0] * root_count
    for producer, consumer in edges:
        if producer == consumer:
            continue
        if consumer not in successors[producer]:
            successors[producer].add(consumer)
            indegree[consumer] += 1
    ready = [root for root, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        root = heapq.heappop(ready)
        order.append(root)
        for consumer in sorted(successors[root]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, consumer)
    if len(order) != root_count:
        # A cyclic quotient needs the affine SCC-rank proof from Phase 4.
        return None

    predecessors = [set() for _ in range(root_count)]
    for producer, root_successors in enumerate(successors):
        for consumer in root_successors:
            predecessors[consumer].add(producer)
    top = [1] * root_count
    for root in order:
        top[root] = 1 + max(
            (top[producer] for producer in predecessors[root]),
            default=0,
        )
    bottom = [1] * root_count
    for root in reversed(order):
        bottom[root] = 1 + max(
            (bottom[consumer] for consumer in successors[root]),
            default=0,
        )
    horizon = max(
        (top[root] + bottom[root] - 1 for root in range(root_count)),
        default=0,
    )
    return tuple(
        (horizon - top[root] - bottom[root] + 1, -top[root])
        for root in range(root_count)
    )


def _scalar_relation_maximum_on_interval(
    relation: CoordinateRelation,
    begin: int,
    end: int,
) -> tuple[bool, int] | None:
    """Bound a partial scalar function over one concrete ordinal interval.

    The boolean distinguishes an interval outside the relation's support from
    a represented value.  Every returned maximum is proved from relation
    bounds; no source point is sampled.
    """
    if begin < 0 or end < begin or len(relation.source_domain.axis_order) != 1:
        return None
    if len(relation.target_domain.axis_order) != 1:
        return None
    canonical = relation.canonical_single_valued()
    if canonical is None:
        return None
    (source_axis,) = canonical.source_domain.axis_order
    maxima: list[int] = []
    for piece in canonical.pieces:
        ((piece_axis, piece_begin, piece_end, piece_step),) = piece.source_bounds_items
        if piece_axis != source_axis or piece_step != 1:
            return None
        overlap_begin = max(begin, piece_begin)
        overlap_end = min(end, piece_end)
        if overlap_begin >= overlap_end:
            continue
        if len(piece.target_ranges) != 1:
            return None
        _target_axis, value, value_end, value_step = piece.target_ranges[0]
        if value_step != 1 or sympy.simplify(value_end - value) != 1:  # pyrefly: ignore[unsupported-operation]
            return None
        bounds = _logical_expression_bounds(
            value,
            domain=canonical.source_domain,
            source_bounds=((source_axis, overlap_begin, overlap_end, 1),),
        )
        if (
            bounds is None or bounds[1].free_symbols or bounds[1].is_integer is not True  # pyrefly: ignore[missing-attribute]
        ):
            return None
        maxima.append(int(bounds[1]))
    return (False, -1) if not maxima else (True, max(maxima))


def _scalar_relation_constant_prefix_end(
    relation: CoordinateRelation,
    begin: int,
    end: int,
) -> int | None:
    """Return the maximal constant prefix of a proved nondecreasing relation."""
    if begin >= end:
        return begin
    first = _scalar_relation_maximum_on_interval(relation, begin, begin + 1)
    whole = _scalar_relation_maximum_on_interval(relation, begin, end)
    if first is None or whole is None or not first[0] or not whole[0]:
        return None
    if whole[1] == first[1]:
        return end
    low = begin + 1
    high = end
    while low + 1 < high:
        middle = (low + high) // 2
        prefix = _scalar_relation_maximum_on_interval(
            relation,
            begin,
            middle,
        )
        if prefix is None or not prefix[0]:
            return None
        if prefix[1] == first[1]:
            low = middle
        else:
            high = middle
    return low


def _event_frontier_list_schedule(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    transient_source_root: int | None = None,
) -> WorkerSchedule | None:
    """List-schedule concrete root/event frontiers without a per-CTA DAG."""
    continuations = _emitted_final_arrival_continuations(
        readiness_graph,
        readiness_counters,
    )
    if continuations is None:
        return None
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph,
        continuations,
    )
    excluded_roots = frozenset(continuation_by_root) | (
        frozenset()
        if transient_source_root is None
        else frozenset((transient_source_root,))
    )
    external_source_frontiers = (
        ()
        if transient_source_root is None
        else _external_source_frontiers(
            readiness_graph,
            worker_schedule,
            readiness_counters,
            root_barrier_edges,
            transient_source_root,
        )
    )
    if external_source_frontiers is None:
        return None
    ordered_schedule = _consumer_major_producer_order(
        readiness_graph,
        worker_schedule,
        readiness_counters,
        root_barrier_edges,
        excluded_roots=excluded_roots,
        external_source_frontiers=external_source_frontiers,
    )
    scheduled_roots = tuple(
        root
        for root in range(len(readiness_graph.root_domains))
        if root not in excluded_roots
    )
    root_orders: dict[int, CoordinateRelation] = {}
    for root in scheduled_roots:
        traversal = _root_schedule_traversal(
            ordered_schedule.segments_for_root(root),
            readiness_graph.root_task_orders[root],
        )
        if traversal is None or traversal.scheduled_ordinal_to_logical_task is None:
            return None
        order = traversal.scheduled_ordinal_to_logical_task
        if len(order.source_domain.axis_order) != 1 or not order.is_total_function():
            return None
        root_orders[root] = order

    external_frontiers: dict[int, CoordinateRelation] = {}
    for root, logical_frontier in external_source_frontiers:
        order = root_orders.get(root)
        ordered_frontier = None if order is None else order.then(logical_frontier)
        if ordered_frontier is None or not _scalar_relation_is_nondecreasing(
            ordered_frontier
        ):
            return None
        external_frontiers[root] = ordered_frontier

    incoming_frontiers: dict[int, list[tuple[int, CoordinateRelation]]] = {
        root: [] for root in scheduled_roots
    }
    schema_edges: set[tuple[int, int]] = set()
    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        consumer_root = prerequisite.consumer_root
        if consumer_root in excluded_roots:
            continue
        consumer_order = root_orders.get(consumer_root)
        if consumer_order is None:
            return None
        if prerequisite.barrier_producer_root is not None:
            producer_root = prerequisite.barrier_producer_root
            producer_domain = readiness_graph.root_domains[producer_root]
            static_relations = _static_producer_relations(
                readiness_graph,
                root=producer_root,
                site_id=None,
                readiness_keys=CoordinateRelation.identity(
                    producer_domain,
                    producer_domain,
                ),
                continuation_by_root=continuation_by_root,
            )
            if static_relations is None:
                return None
            for static_root, _relation in static_relations:
                schema_edges.add((static_root, consumer_root))
                if static_root in excluded_roots:
                    continue
                producer_order = root_orders.get(static_root)
                if producer_order is None:
                    return None
                frontier = CoordinateRelation.point_map(
                    consumer_order.source_domain,
                    producer_order.source_domain,
                    (
                        (
                            tuple(
                                (
                                    axis,
                                    0,
                                    consumer_order.source_domain.axis_counts[axis],
                                    1,
                                )
                                for axis in consumer_order.source_domain.axis_order
                            ),
                            (sympy.Integer(producer_order.source_domain.size - 1),),
                        ),
                    ),
                )
                incoming_frontiers[consumer_root].append((static_root, frontier))
            continue

        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        ordered_consumer_keys = (
            None if consumer_keys is None else consumer_order.then(consumer_keys)
        )
        static_relations = _readiness_static_producers(
            readiness_graph,
            plan.producers,
            continuation_by_root,
        )
        if ordered_consumer_keys is None or static_relations is None:
            return None
        for producer_root, keys_by_producer in static_relations:
            schema_edges.add((producer_root, consumer_root))
            if producer_root in excluded_roots:
                continue
            producer_order = root_orders.get(producer_root)
            if producer_order is None:
                return None
            keys_by_producer_order = producer_order.then(keys_by_producer)
            ordinal_identity = CoordinateRelation.identity(
                producer_order.source_domain,
                producer_order.source_domain,
            )
            producer_frontier_by_key = (
                None
                if keys_by_producer_order is None
                else _maximum_value_by_key(keys_by_producer_order, ordinal_identity)
            )
            frontier = (
                None
                if producer_frontier_by_key is None
                else _maximum_required_value_by_consumer(
                    ordered_consumer_keys,
                    producer_frontier_by_key,
                )
            )
            if frontier is None or frontier.canonical_single_valued() is None:
                return None
            incoming_frontiers[consumer_root].append((producer_root, frontier))

    criticality = _root_schema_criticality(
        len(readiness_graph.root_domains),
        frozenset(schema_edges),
    )
    if criticality is None:
        return None

    order_piece_ends: dict[int, tuple[int, ...]] = {}
    for root, order in root_orders.items():
        canonical = order.canonical_single_valued()
        if canonical is None:
            return None
        (ordinal_axis,) = canonical.source_domain.axis_order
        if any(
            axis != ordinal_axis or step != 1
            for piece in canonical.pieces
            for axis, _begin, _end, step in piece.source_bounds_items
        ):
            return None
        order_piece_ends[root] = tuple(
            sorted(
                {
                    end
                    for piece in canonical.pieces
                    for _axis, _begin, end, _step in piece.source_bounds_items
                }
            )
        )

    cursors = dict.fromkeys(scheduled_roots, 0)
    root_ends = {root: root_orders[root].source_domain.size for root in scheduled_roots}

    def interval_is_admissible(
        root: int,
        begin: int,
        end: int,
        producer_cursors: dict[int, int],
    ) -> bool | None:
        for producer_root, frontier in incoming_frontiers[root]:
            maximum = _scalar_relation_maximum_on_interval(frontier, begin, end)
            if maximum is None:
                return None
            has_value, value = maximum
            if has_value and value >= producer_cursors[producer_root]:
                return False
        return True

    def admissible_prefix_end(
        root: int,
        begin: int,
        limit: int,
        producer_cursors: dict[int, int],
    ) -> int | None:
        if begin >= limit:
            return begin
        whole = interval_is_admissible(root, begin, limit, producer_cursors)
        if whole is None:
            return None
        if whole:
            return limit
        first = interval_is_admissible(root, begin, begin + 1, producer_cursors)
        if first is None or not first:
            return begin if first is False else None
        low = begin + 1
        high = limit
        while low + 1 < high:
            middle = (low + high) // 2
            prefix = interval_is_admissible(
                root,
                begin,
                middle,
                producer_cursors,
            )
            if prefix is None:
                return None
            if prefix:
                low = middle
            else:
                high = middle
        return low

    def next_piece_end(root: int, cursor: int) -> int | None:
        return next(
            (end for end in order_piece_ends[root] if cursor < end),
            None,
        )

    placed_runs: list[_PlacedRun] = []
    worker_step = 0
    while any(cursors[root] < root_ends[root] for root in scheduled_roots):
        wave_start_cursors = dict(cursors)
        next_worker = 0
        while next_worker < worker_schedule.worker_count:
            candidates: list[
                tuple[
                    tuple[int, int, int, int, int, int, int, int],
                    int,
                    int,
                ]
            ] = []
            for root in scheduled_roots:
                cursor = cursors[root]
                if cursor >= root_ends[root]:
                    continue
                piece_end = next_piece_end(root, cursor)
                if piece_end is None:
                    return None
                limit = min(
                    root_ends[root],
                    piece_end,
                    cursor + worker_schedule.worker_count - next_worker,
                )
                if external_frontier := external_frontiers.get(root):
                    external_end = _scalar_relation_constant_prefix_end(
                        external_frontier,
                        cursor,
                        limit,
                    )
                    if external_end is None:
                        return None
                    limit = min(limit, external_end)
                # Stop at the next producer frontier needed by any current
                # consumer cohort.  This is the exact event boundary that
                # lets another arm close a join in the remaining wave slots;
                # it is derived from the emitted prerequisite relation rather
                # than from a configurable chunk size.
                for consumer_root in scheduled_roots:
                    consumer_cursor = cursors[consumer_root]
                    if consumer_cursor >= root_ends[consumer_root]:
                        continue
                    for producer_root, frontier in incoming_frontiers[consumer_root]:
                        if producer_root != root:
                            continue
                        required = _scalar_relation_maximum_on_interval(
                            frontier,
                            consumer_cursor,
                            consumer_cursor + 1,
                        )
                        if required is None:
                            return None
                        has_value, value = required
                        if has_value and cursor <= value:
                            limit = min(limit, value + 1)
                candidate_end = admissible_prefix_end(
                    root,
                    cursor,
                    limit,
                    wave_start_cursors,
                )
                if candidate_end is None:
                    return None
                if candidate_end == cursor:
                    continue

                hypothetical_cursors = dict(cursors)
                hypothetical_cursors[root] = candidate_end
                released_roots: list[int] = []
                for consumer_root in scheduled_roots:
                    consumer_cursor = cursors[consumer_root]
                    if consumer_cursor >= root_ends[consumer_root]:
                        continue
                    before = interval_is_admissible(
                        consumer_root,
                        consumer_cursor,
                        consumer_cursor + 1,
                        cursors,
                    )
                    after = interval_is_admissible(
                        consumer_root,
                        consumer_cursor,
                        consumer_cursor + 1,
                        hypothetical_cursors,
                    )
                    if before is None or after is None:
                        return None
                    if not before and after:
                        released_roots.append(consumer_root)
                base = criticality[root]
                release_class = min(
                    (criticality[consumer_root] for consumer_root in released_roots),
                    default=base,
                )
                effective = min(base, release_class)
                closes_effective_event = any(
                    criticality[consumer_root] == effective
                    for consumer_root in released_roots
                )
                external_release = -1
                if external_frontier := external_frontiers.get(root):
                    external_maximum = _scalar_relation_maximum_on_interval(
                        external_frontier,
                        cursor,
                        candidate_end,
                    )
                    if external_maximum is None or not external_maximum[0]:
                        return None
                    external_release = external_maximum[1]
                priority = (
                    effective[0],
                    0 if root in external_frontiers else 1,
                    effective[1],
                    0 if effective == base else 1,
                    0 if closes_effective_event else 1,
                    external_release,
                    root,
                    cursor,
                )
                candidates.append((priority, root, candidate_end))

            if not candidates:
                break
            _priority, root, candidate_end = min(candidates)
            cursor = cursors[root]
            count = candidate_end - cursor
            placed_runs.append(
                _PlacedRun(
                    root=root,
                    source_begin=cursor,
                    task_count=count,
                    worker_begin=next_worker,
                    worker_count=count,
                    worker_step=worker_step,
                )
            )
            cursors[root] = candidate_end
            next_worker += count
        if next_worker == 0:
            return None
        worker_step += 1

    merged_runs: list[_PlacedRun] = []
    for run in placed_runs:
        if merged_runs:
            previous = merged_runs[-1]
            extends_full_width_run = (
                previous.worker_begin == 0
                and previous.worker_count == worker_schedule.worker_count
                and previous.task_count % previous.worker_count == 0
                and run.worker_begin == 0
                and run.worker_count <= worker_schedule.worker_count
                and previous.worker_step + previous.task_count // previous.worker_count
                == run.worker_step
            )
            if (
                previous.root == run.root
                and previous.source_begin + previous.task_count == run.source_begin
                and extends_full_width_run
            ):
                merged_runs[-1] = dataclasses.replace(
                    previous,
                    task_count=previous.task_count + run.task_count,
                )
                continue
            extends_same_wave = (
                previous.worker_step == run.worker_step
                and previous.worker_begin + previous.worker_count == run.worker_begin
            )
            if (
                previous.root == run.root
                and previous.source_begin + previous.task_count == run.source_begin
                and extends_same_wave
            ):
                merged_runs[-1] = dataclasses.replace(
                    previous,
                    task_count=previous.task_count + run.task_count,
                    worker_count=previous.worker_count + run.worker_count,
                )
                continue
        merged_runs.append(run)
    if len(merged_runs) > _MAX_GLOBAL_LIST_SEGMENTS:
        return None

    segments: list[WorkerScheduleSegment] = (
        []
        if transient_source_root is None
        else list(ordered_schedule.segments_for_root(transient_source_root))
    )
    for run in merged_runs:
        task_order = _task_order_slice(
            root_orders[run.root],
            run.source_begin,
            run.task_count,
        )
        if task_order is None:
            return None
        segments.append(
            WorkerScheduleSegment(
                root=run.root,
                task_order=task_order,
                worker_begin=run.worker_begin,
                worker_count=run.worker_count,
                dispatch_offset=run.worker_step * run.worker_count,
            )
        )
    try:
        result = WorkerSchedule(worker_schedule.worker_count, tuple(segments))
    except ValueError:
        return None
    if not _validate_worker_schedule_tasks(
        result,
        readiness_graph.root_task_orders,
        excluded_roots=frozenset(continuation_by_root),
    ):
        return None
    if not _schedule_is_progress_safe(
        result,
        readiness_graph,
        readiness_counters,
        root_barrier_edges,
        transient_source_root=transient_source_root,
        require_strict_rank=True,
    ):
        return None
    return result


def _resident_task_counts_by_root(
    worker_schedule: WorkerSchedule,
    root_count: int,
) -> tuple[sympy.Expr, ...] | None:
    """Return exact resident task mass from authoritative schedule support."""
    counts = [sympy.Integer(0) for _ in range(root_count)]
    for segment in worker_schedule.segments:
        if not 0 <= segment.root < root_count:
            return None
        count = segment.task_order.source_support_cardinality()
        if count is None or (
            not sympy.sympify(count).is_zero
            and segment.launch_stage != _RESIDENT_LAUNCH_STAGE
        ):
            return None
        counts[segment.root] = sympy.simplify(
            counts[segment.root] + sympy.sympify(count)
        )
    return tuple(counts)


def _singleton_scalar_relation_value(
    relation: CoordinateRelation,
) -> sympy.Expr | None:
    """Extract one proved scalar value from a total singleton relation."""
    canonical = relation.canonical_single_valued()
    if (
        canonical is None
        or canonical.source_domain.axis_order
        or not canonical.is_total_function()
        or len(canonical.pieces) != 1
    ):
        return None
    (piece,) = canonical.pieces
    if piece.source_bounds_items or len(piece.target_ranges) != 1:
        return None
    _axis, begin, end, step = piece.target_ranges[0]
    if step != 1 or not _equal_integer_expressions(end - begin, 1):
        return None
    return sympy.simplify(begin)


def _resident_schedule_occupied_wave_count(
    worker_schedule: WorkerSchedule,
) -> sympy.Expr | None:
    """Return the exact number of occupied resident waves, including zero.

    Schedule relations live in a bounded worker/wave domain, so that domain's
    wave extent is always an upper bound.  When the exact source-support mass
    fills the minimum possible number of waves, the bound is attained by the
    pigeonhole principle.  This also proves the runtime-empty case: zero tasks
    have a zero-wave minimum without asking an extremum operation to represent
    a conditionally empty fiber.

    Non-minimal schedules use the ordinary relation projection and extremum
    operations.  If a symbolic root may be empty and its maximum therefore has
    no total representation, decline rather than sampling a nonempty extent.
    """
    if not worker_schedule.segments:
        return sympy.Integer(0)

    task_counts = _resident_task_counts_by_root(
        worker_schedule,
        max(segment.root for segment in worker_schedule.segments) + 1,
    )
    if task_counts is None:
        return None
    task_count = sympy.simplify(sympy.Add(*task_counts))
    wave_count = sympy.sympify(worker_schedule.worker_step_domain.size_expr)
    minimum_wave_count = _ceildiv_nonnegative_expression(
        task_count,
        worker_schedule.worker_count,
    )
    if _equal_integer_expressions(wave_count, minimum_wave_count):
        return sympy.simplify(wave_count)

    singleton_domain = CoordinateDomain((), (), kind="event")
    maximum_waves: list[sympy.Expr] = []
    for root in sorted({segment.root for segment in worker_schedule.segments}):
        segments = worker_schedule.segments_for_root(root)
        root_domain = segments[0].task_order.target_domain
        if root_domain.size_expr.is_zero is True:
            continue
        maximum = _maximum_root_wave_by_key(
            worker_schedule,
            root,
            CoordinateRelation.total(root_domain, singleton_domain),
        )
        value = None if maximum is None else _singleton_scalar_relation_value(maximum)
        if value is None:
            return None
        maximum_waves.append(value)

    if not maximum_waves:
        return sympy.Integer(0) if task_count.is_zero is True else None
    result = sympy.simplify(sympy.Max(*maximum_waves) + 1)
    if not tile_dependency._is_provably_nonnegative(
        sympy.simplify(wave_count - result),
        None,
    ):
        return None
    return result


def _global_unit_list_schedule(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    transient_source_root: int | None = None,
) -> WorkerSchedule | None:
    """Apply event-frontier list scheduling without materializing CTA nodes."""
    if transient_source_root is None and not _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        return worker_schedule
    if transient_source_root is not None:
        source_schedule = _with_transient_source_schedule_segment(
            worker_schedule,
            readiness_graph.root_task_orders,
            transient_source_root,
        )
        if source_schedule is None:
            return None
        worker_schedule = source_schedule
    candidate = _event_frontier_list_schedule(
        readiness_graph,
        worker_schedule,
        readiness_counters,
        root_barrier_edges,
        transient_source_root=transient_source_root,
    )
    if candidate is None or transient_source_root is not None:
        return candidate

    # Unit-task list scheduling must not lengthen an otherwise equivalent
    # resident schedule.  This is a structural makespan certificate, not a
    # latency model: candidates with the same exact root coverage may replace
    # the input only when their final occupied wave is no later.
    root_count = len(readiness_graph.root_domains)
    input_task_counts = _resident_task_counts_by_root(worker_schedule, root_count)
    candidate_task_counts = _resident_task_counts_by_root(candidate, root_count)
    if (
        input_task_counts is None
        or candidate_task_counts is None
        or any(
            not _equal_integer_expressions(input_count, candidate_count)
            for input_count, candidate_count in zip(
                input_task_counts,
                candidate_task_counts,
                strict=True,
            )
        )
    ):
        return worker_schedule

    input_horizon = _resident_schedule_occupied_wave_count(worker_schedule)
    candidate_horizon = _resident_schedule_occupied_wave_count(candidate)
    if (
        input_horizon is None
        or candidate_horizon is None
        or not tile_dependency._is_provably_nonnegative(
            sympy.simplify(input_horizon - candidate_horizon),
            None,
        )
    ):
        return worker_schedule
    return candidate


def _transient_source_candidate(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    worker_count: int,
) -> int | None:
    """Find the sole oversubscribed source in the emitted prerequisite graph."""
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )
    consumer_roots = {prerequisite.consumer_root for prerequisite in prerequisites}
    producer_roots: set[int] = set()
    for prerequisite in prerequisites:
        if prerequisite.barrier_producer_root is not None:
            producer_roots.add(prerequisite.barrier_producer_root)
            continue
        assert prerequisite.counter_plan is not None
        producer_roots.update(
            producer.producer_root for producer in prerequisite.counter_plan.producers
        )
    source_roots = tuple(sorted(producer_roots - consumer_roots))
    if len(source_roots) != 1:
        return None
    (source_root,) = source_roots
    if readiness_graph.root_domains[
        source_root
    ].size <= worker_count or not _has_strict_partial_source_signal(
        readiness_graph,
        source_root,
        readiness_counters,
        root_barrier_edges,
    ):
        return None
    return source_root


@cache
def _source_segment_ticket_order(
    segment: WorkerScheduleSegment,
) -> CoordinateRelation | None:
    """Derive the logical-task-to-ticket bijection from one dense segment."""
    ticket_to_logical = segment.logical_task_order
    logical_to_ticket = (
        None
        if ticket_to_logical is None
        else _logical_task_to_order_ordinal(
            ticket_to_logical,
            ticket_to_logical.source_domain,
        )
    )
    if (
        ticket_to_logical is None
        or not ticket_to_logical.is_total_function()
        or logical_to_ticket is None
        or not logical_to_ticket.is_total_function()
    ):
        return None
    return logical_to_ticket


def _transient_source_schedule_segment(
    worker_schedule: WorkerSchedule,
    source_root: int,
) -> WorkerScheduleSegment | None:
    """Recognize one exact dense launch-stage-zero source-ticket relation."""
    if source_root < 0:
        return None
    segments = worker_schedule.segments_for_root(source_root)
    if len(segments) != 1:
        return None
    (segment,) = segments
    if (
        segment.launch_stage != _SOURCE_LAUNCH_STAGE
        or segment.worker_begin != 0
        or segment.worker_count != worker_schedule.worker_count
        or segment.dispatch_offset != 0
        or _source_segment_ticket_order(segment) is None
    ):
        return None
    return segment


def _with_transient_source_schedule_segment(
    worker_schedule: WorkerSchedule,
    root_task_orders: tuple[CoordinateRelation, ...],
    source_root: int,
) -> WorkerSchedule | None:
    """Move one concrete root into the schedule's source-ticket stage."""
    if not 0 <= source_root < len(root_task_orders):
        return None
    if (
        _transient_source_schedule_segment(
            worker_schedule,
            source_root,
        )
        is not None
        and worker_schedule.segments_for_root(source_root)[0].task_order.target_domain
        == root_task_orders[source_root].target_domain
    ):
        return worker_schedule

    resident_schedule = worker_schedule.without_roots(frozenset((source_root,)))
    placement_domain = resident_schedule.placement_domain
    launch_stage_axis, worker_axis, wave_axis = placement_domain.axis_order
    source_task_order = root_task_orders[source_root]
    source_task_count = source_task_order.source_domain.size
    source_wave_count = (
        source_task_count + worker_schedule.worker_count - 1
    ) // worker_schedule.worker_count
    wave_count = max(
        placement_domain.axis_counts[wave_axis],
        source_wave_count,
    )
    combined_domain = _worker_schedule_domain(
        worker_schedule.worker_count,
        wave_count,
        (launch_stage_axis, worker_axis, wave_axis),
    )
    resident_segments: list[WorkerScheduleSegment] = []
    for segment in resident_schedule.segments:
        rebased_task_order = segment.task_order.rebase_source_domain(combined_domain)
        if rebased_task_order is None:
            return None
        resident_segments.append(
            dataclasses.replace(segment, task_order=rebased_task_order)
        )
    try:
        source_segment = _normalize_dense_schedule_segment(
            WorkerScheduleSegment(
                root=source_root,
                task_order=source_task_order,
                worker_begin=0,
                worker_count=worker_schedule.worker_count,
                dispatch_offset=0,
            ),
            combined_domain,
            launch_stage=_SOURCE_LAUNCH_STAGE,
        )
        result = WorkerSchedule(
            worker_count=worker_schedule.worker_count,
            segments=(source_segment, *resident_segments),
        )
    except ValueError:
        return None
    return (
        result
        if _transient_source_schedule_segment(
            result,
            source_root,
        )
        is not None
        else None
    )


def _source_ticket_order(
    worker_schedule: WorkerSchedule,
    source_root: int,
) -> CoordinateRelation | None:
    """Derive the logical-task-to-ticket map from the source segment."""
    source_segment = _transient_source_schedule_segment(
        worker_schedule,
        source_root,
    )
    if source_segment is None:
        return None
    return _source_segment_ticket_order(source_segment)


def _has_strict_partial_source_signal(
    readiness_graph: ReadinessGraph,
    source_root: int,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> bool:
    """Find an emitted wait with a strict-subset source contribution."""
    source_domain = readiness_graph.root_domains[source_root]
    root_order_edges = set(root_barrier_edges)
    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        if plan is None or consumer is None:
            continue
        if _is_ordered_by_root_barrier(
            source_root,
            consumer.consumer_root,
            root_order_edges,
        ):
            continue
        source_tasks_by_key = _combined_producer_tasks_by_key(
            plan,
            source_root,
            source_domain,
        )
        if source_tasks_by_key is None:
            continue
        source_count_by_key = source_tasks_by_key.target_count_by_source()
        source_count_by_wait = (
            None
            if source_count_by_key is None
            else consumer.keys_by_consumer.then(source_count_by_key)
        )
        bounds = (
            None
            if source_count_by_wait is None
            else source_count_by_wait.value_bounds()
        )
        if (
            source_count_by_wait is not None
            and source_count_by_wait.pieces
            and bounds is not None
            and bounds[0] > 0
            and bounds[1] < source_domain.size
        ):
            return True
    return False


def _has_valid_transient_source_schedule(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    source_root: int,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> bool:
    """Check source-ticket invariants after exact schedule ownership is proved."""
    if not 0 <= source_root < len(readiness_graph.root_task_orders):
        return False
    if _source_ticket_order(worker_schedule, source_root) is None:
        return False
    if any(
        segment.launch_stage
        != (
            _SOURCE_LAUNCH_STAGE
            if segment.root == source_root
            else _RESIDENT_LAUNCH_STAGE
        )
        for segment in worker_schedule.segments
    ):
        return False
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )
    return not any(
        prerequisite.consumer_root == source_root for prerequisite in prerequisites
    )


def _parameterized_counter_fan_in_bounds(
    plan: ReadinessCounterPlan,
) -> tuple[int, int] | None:
    """Return the proved static fan-in bounds accepted by parametric schedules.

    Ordinary resident waits may fan out from one key to multiple consumer
    tasks and may have different exact arrival counts by key. Final-arrival
    continuations retain their stricter uniform one-task-per-key contract.
    """
    fan_in_bounds = plan.arrival_count_bounds()
    if (
        len(plan.producers) != 1
        or not plan.consumers
        or fan_in_bounds is None
        or not 0 < fan_in_bounds[0] <= fan_in_bounds[1] < 2**32
    ):
        return None
    (producer,) = plan.producers
    publication = producer.keys_by_producer
    if (
        producer.producer_site_id is not None
        or publication is None
        or publication.canonical_single_valued() is None
    ):
        return None
    if not all(
        consumer.consumer_site_id is None
        and producer.producer_root < consumer.consumer_root
        and consumer.keys_by_consumer.canonical_single_valued() is not None
        for consumer in plan.consumers
    ):
        return None
    if plan.continuation_consumer_index is not None:
        if (
            fan_in_bounds[0] != fan_in_bounds[1]
            or fan_in_bounds[0] == 1
            or len(plan.consumers) != 1
        ):
            return None
        (consumer,) = plan.consumers
        converse = consumer.keys_by_consumer.converse()
        if (
            not consumer.keys_by_consumer.is_positional_bijection()
            or converse is None
            or not converse.is_positional_bijection()
        ):
            return None
    return fan_in_bounds


def _parameterized_uniform_counter_fan_in(
    plan: ReadinessCounterPlan,
) -> int | None:
    """Return a parameterized counter's uniform fan-in, when it has one."""
    bounds = _parameterized_counter_fan_in_bounds(plan)
    if bounds is None or bounds[0] != bounds[1]:
        return None
    return bounds[0]


def _supports_exact_counter_plan_lowering(
    plan: ReadinessCounterPlan,
    root_domains: tuple[CoordinateDomain, ...],
) -> bool:
    """Return whether the selected counter has one exact supported lowering.

    These are immutable plan facts shared by static and parameterized
    schedules.  Renderer-specific restrictions do not belong here.
    """
    def endpoint_has_supported_domain(
        root: int,
        site_id: int | None,
        domain: CoordinateDomain,
    ) -> bool:
        if not 0 <= root < len(root_domains):
            return False
        root_domain = root_domains[root]
        if site_id is None:
            return domain == root_domain
        root_counts = root_domain.axis_count_expressions
        site_counts = domain.axis_count_expressions
        return (
            domain.kind == "site"
            and domain.identity == site_id
            and len(nested_logical_axes(root_domain, domain)) == 1
            and all(
                axis in site_counts
                and _equal_integer_expressions(site_counts[axis], count)
                for axis, count in root_counts.items()
            )
        )

    continuation_index = plan.continuation_consumer_index
    if not plan.consumers or (
        continuation_index is not None
        and not 0 <= continuation_index < len(plan.consumers)
    ):
        return False
    for producer in plan.producers:
        if not endpoint_has_supported_domain(
            producer.producer_root,
            producer.producer_site_id,
            producer.producers_by_key.target_domain,
        ) or not _supports_readiness_counter_lowering(producer):
            return False
    for consumer in plan.consumers:
        if not endpoint_has_supported_domain(
            consumer.consumer_root,
            consumer.consumer_site_id,
            consumer.keys_by_consumer.source_domain,
        ) or consumer.keys_by_consumer.canonical_single_valued() is None:
            return False

    nested_consumers = tuple(
        consumer
        for consumer in plan.consumers
        if consumer.consumer_site_id is not None
    )
    if nested_consumers and (
        len(plan.consumers) != 1 or continuation_index is not None
    ):
        return False

    if continuation_index is None:
        return True

    continuation_consumer = plan.consumers[continuation_index]
    converse = continuation_consumer.keys_by_consumer.converse()
    fan_in = plan.uniform_arrival_count()
    return (
        fan_in is not None
        and fan_in > 0
        and continuation_consumer.keys_by_consumer.is_total_function()
        and converse is not None
        and converse.is_total_function()
    )


def _supports_current_parameterized_renderer(
    plan: ReadinessCounterPlan,
    root_domains: tuple[CoordinateDomain, ...],
) -> bool:
    """Return whether today's parameterized renderer supports this exact plan."""
    return (
        _supports_exact_counter_plan_lowering(plan, root_domains)
        and _parameterized_counter_fan_in_bounds(plan) is not None
    )


def _current_renderer_lowerable_counters(
    plans: tuple[ReadinessCounterPlan, ...],
    root_domains: tuple[CoordinateDomain, ...],
) -> tuple[ReadinessCounterPlan, ...]:
    """Filter exact plans through today's parameterized/epoch renderer contract."""
    common = tuple(
        plan
        for plan in plans
        if _supports_exact_counter_plan_lowering(plan, root_domains)
    )
    parameterized_roots = any(domain.parameter_symbols for domain in root_domains)
    renderable = tuple(
        plan
        for plan in common
        if not (parameterized_roots or plan.parameter_symbols)
        or _supports_current_parameterized_renderer(plan, root_domains)
    )
    uses_epoch_framing = parameterized_roots or any(
        plan.parameter_symbols for plan in renderable
    )
    if not uses_epoch_framing:
        return renderable
    return tuple(
        plan
        for plan in renderable
        if (bounds := plan.arrival_count_bounds()) is not None
        and 0 < bounds[0] <= bounds[1] < 2**32
    )


def _supports_parameterized_fan_in_one_counter(
    plan: ReadinessCounterPlan,
) -> bool:
    """Return whether one counter belongs to the positional recurrence."""
    if _parameterized_uniform_counter_fan_in(plan) != 1:
        return False
    (producer,) = plan.producers
    publication = producer.keys_by_producer
    return (
        plan.continuation_consumer_index is None
        and publication is not None
        and producer.producers_by_key.is_positional_bijection()
        and publication.is_positional_bijection()
        and all(
            consumer.keys_by_consumer.is_positional_bijection()
            and (converse := consumer.keys_by_consumer.converse()) is not None
            and converse.is_positional_bijection()
            for consumer in plan.consumers
        )
    )


def _finalize_emitted_synchronization(
    *,
    dependency_graph: TileDependencyGraph,
    root_domains: tuple[CoordinateDomain, ...],
    readiness_counters: tuple[ReadinessCounterPlan, ...],
) -> tuple[tuple[ReadinessCounterPlan, ...], frozenset[tuple[int, int]]]:
    """Select fallback barriers and remove counter consumers they subsume."""
    # Dropping an unsupported plan lets coverage select the ordinary
    # root-barrier fallback from the same obligations. The two-pass renderer
    # filter first rejects unsupported parameterized plans, then enforces the
    # shared epoch bound across every retained sibling.
    readiness_counters = _current_renderer_lowerable_counters(
        readiness_counters,
        root_domains,
    )
    covered_obligations = frozenset(
        obligation
        for counter_plan in readiness_counters
        for readiness_consumer in counter_plan.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    root_barrier_edges = _select_root_barrier_edges(
        dependency_graph=dependency_graph,
        covered_obligations=covered_obligations,
    )
    root_order_edges = set(root_barrier_edges)
    retained_readiness_counters: list[ReadinessCounterPlan] = []
    for counter_plan in readiness_counters:
        retained_consumer_indices = tuple(
            consumer_index
            for consumer_index, readiness_consumer in enumerate(counter_plan.consumers)
            if consumer_index == counter_plan.continuation_consumer_index
            or not all(
                _is_ordered_by_root_barrier(
                    readiness_producer.producer_root,
                    readiness_consumer.consumer_root,
                    root_order_edges,
                )
                for readiness_producer in counter_plan.producers
            )
        )
        if not retained_consumer_indices:
            continue
        retained_readiness_counters.append(
            dataclasses.replace(
                counter_plan,
                consumers=tuple(
                    counter_plan.consumers[index] for index in retained_consumer_indices
                ),
                continuation_consumer_index=(
                    retained_consumer_indices.index(
                        counter_plan.continuation_consumer_index
                    )
                    if counter_plan.continuation_consumer_index is not None
                    else None
                ),
            )
        )
    retained = tuple(retained_readiness_counters)
    covered_obligations = frozenset(
        obligation
        for counter_plan in retained
        for readiness_consumer in counter_plan.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    _validate_schedule_coverage(
        dependency_graph=dependency_graph,
        covered_obligations=covered_obligations,
        root_barrier_edges=root_barrier_edges,
    )
    return retained, root_barrier_edges


def build_static_pipeline_plan(
    *,
    dependency_graph: TileDependencyGraph,
    root_task_orders: tuple[CoordinateRelation, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    worker_count: int,
    publishable_site_ids: frozenset[int] | None = None,
    allow_transient_source: bool = False,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> StaticPipelinePlan:
    """Derive all generic readiness strategies without inspecting root bodies."""
    root_domains = tuple(task_order.target_domain for task_order in root_task_orders)
    readiness_graph = build_readiness_graph(
        dependency_graph=dependency_graph,
        root_task_orders=root_task_orders,
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
        prove_nonnegative=prove_nonnegative,
    )
    continuation_candidates = derive_final_arrival_continuations(readiness_graph)
    if any(domain.parameter_symbols for domain in root_domains):
        try:
            worker_schedule = _build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                worker_count,
            )
        except ValueError as error:
            raise exc.InvalidConfig(
                "cross_loop_schedule='static_pipeline' cannot represent this "
                "parameterized root schedule"
            ) from error
        continuation_candidates = choose_final_arrival_continuations(
            readiness_graph,
            continuation_candidates,
            worker_schedule,
        )
        sink_roots = frozenset(range(len(root_domains))) - frozenset(
            edge.producer_root for edge in dependency_graph.edges
        )
        selected_continuations: list[FinalArrivalContinuation] = []
        for continuation in continuation_candidates:
            event = readiness_graph.event(continuation.event_id)
            readiness_consumer = event.consumers[continuation.consumer_index]
            provisional_plan = ReadinessCounterPlan(
                producers=event.producers,
                consumers=(readiness_consumer,),
                continuation_consumer_index=0,
            )
            fan_in = _parameterized_uniform_counter_fan_in(provisional_plan)
            if readiness_consumer.consumer_root in sink_roots and (
                fan_in is not None and fan_in > 1
            ):
                selected_continuations.append(continuation)
        continuations = tuple(selected_continuations)
        selected_counters = choose_readiness_counters(
            readiness_graph,
            continuations,
        )
        readiness_counters = tuple(
            plan
            for plan in selected_counters
            if _supports_current_parameterized_renderer(plan, root_domains)
        )
        if sum(
            plan.continuation_consumer_index is not None for plan in readiness_counters
        ) != len(continuations):
            # Continuation ownership is atomic: if any selected sink cannot be
            # represented by the emitted counter subset, retain every root in
            # the conservative schedule.
            continuations = ()
            readiness_counters = tuple(
                plan
                for plan in choose_readiness_counters(readiness_graph, ())
                if _supports_current_parameterized_renderer(plan, root_domains)
            )
        readiness_counters, root_barrier_edges = _finalize_emitted_synchronization(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            readiness_counters=readiness_counters,
        )
        if not _parameterized_prerequisites_follow_root_order(
            readiness_counters,
            root_barrier_edges,
        ):
            raise AssertionError(
                "parameterized schedule has a backward root prerequisite"
            )
        continuation_roots = frozenset(
            readiness_consumer.consumer_root
            for plan in readiness_counters
            if (readiness_consumer := plan.continuation_consumer) is not None
        )
        if continuation_roots:
            worker_schedule = _build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                worker_count,
                excluded_roots=continuation_roots,
            )
        return StaticPipelinePlan(
            worker_schedule=worker_schedule,
            root_task_orders=root_task_orders,
            readiness_counters=readiness_counters,
            root_barrier_edges=root_barrier_edges,
            transient_source_root=None,
        )

    try:
        (
            worker_schedule,
            continuations,
            nested_loop_counters,
        ) = build_worker_schedule(
            readiness_graph,
            continuation_candidates,
            worker_count=worker_count,
        )
    except ValueError as error:
        raise exc.InvalidConfig(
            f"the num_sm_multiplier grid of {worker_count} workers does not "
            "admit a progress-safe cross-loop schedule"
        ) from error

    nested_loop_obligations = frozenset(
        obligation
        for plan in nested_loop_counters
        for readiness_consumer in plan.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    candidate_readiness_counters = (
        *choose_readiness_counters(
            readiness_graph,
            continuations,
            excluded_obligations=nested_loop_obligations,
        ),
        *nested_loop_counters,
    )
    dropped_unlowerable_counter = (
        _current_renderer_lowerable_counters(
            candidate_readiness_counters,
            root_domains,
        )
        != candidate_readiness_counters
    )
    # Recompute coverage from the mechanisms that will actually be emitted.
    # Dependency analysis may prove a finer relation than the selected emitter
    # can materialize. Such a relation must monotonically coarsen to root
    # barrier; retaining a task-ready classification without an emitter
    # would remove the dependency entirely.
    readiness_counters, root_barrier_edges = _finalize_emitted_synchronization(
        dependency_graph=dependency_graph,
        root_domains=root_domains,
        readiness_counters=candidate_readiness_counters,
    )
    if dropped_unlowerable_counter:
        # Local placement may have used the finer event before finalization.
        # Once that event falls back to a stronger whole-root barrier, restart
        # from the conservative root-major schedule so the replacement cannot
        # introduce a same-worker cycle.
        worker_schedule = build_baseline_worker_schedule(
            root_domains,
            root_task_orders,
            worker_count,
        )
        continuations = ()
        readiness_counters, root_barrier_edges = _finalize_emitted_synchronization(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            readiness_counters=choose_readiness_counters(readiness_graph, ()),
        )
    transient_source_root: int | None = None
    globally_scheduled = None
    if not continuations:
        transient_candidate = (
            _transient_source_candidate(
                readiness_graph,
                readiness_counters,
                root_barrier_edges,
                worker_count=worker_count,
            )
            if allow_transient_source
            else None
        )
        if transient_candidate is not None:
            transient_schedule = _global_unit_list_schedule(
                readiness_graph,
                worker_schedule,
                readiness_counters,
                root_barrier_edges,
                transient_source_root=transient_candidate,
            )
            if transient_schedule is not None and _has_valid_transient_source_schedule(
                transient_schedule,
                readiness_graph,
                transient_candidate,
                readiness_counters,
                root_barrier_edges,
            ):
                globally_scheduled = transient_schedule
                transient_source_root = transient_candidate
    if globally_scheduled is None:
        globally_scheduled = _global_unit_list_schedule(
            readiness_graph,
            worker_schedule,
            readiness_counters,
            root_barrier_edges,
        )
    if globally_scheduled is not None:
        worker_schedule = globally_scheduled
    if dropped_unlowerable_counter and not _schedule_is_progress_safe(
        worker_schedule,
        readiness_graph,
        readiness_counters,
        root_barrier_edges,
        transient_source_root=transient_source_root,
    ):
        raise AssertionError(
            "replay-unsafe counter fallback did not preserve schedule progress"
        )
    return StaticPipelinePlan(
        worker_schedule=worker_schedule,
        root_task_orders=root_task_orders,
        readiness_counters=readiness_counters,
        root_barrier_edges=root_barrier_edges,
        transient_source_root=transient_source_root,
    )


def _validate_schedule_coverage(
    *,
    dependency_graph: TileDependencyGraph,
    covered_obligations: frozenset[DependencyObligation],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> None:
    """Verify that every dependence has an emitted synchronization path."""
    root_order_edges = set(root_barrier_edges)
    for dependency in dependency_graph.edges:
        pair = (dependency.producer_root, dependency.consumer_root)
        if _is_ordered_by_root_barrier(*pair, root_order_edges):
            continue
        uncovered = tuple(
            obligation
            for access_dependency in dependency.access_dependencies
            for obligation in dependency_graph.dependency_obligations(access_dependency)
            if obligation not in covered_obligations
        )
        if not uncovered:
            continue
        raise exc.CrossLoopSchedulingError(
            f"{dependency.producer_root}->{dependency.consumer_root} through "
            f"allocations {sorted(dependency.tensor_names)!r} has no cross-loop "
            f"synchronization path for dependencies {uncovered!r}"
        )


def _select_root_barrier_edges(
    *,
    dependency_graph: TileDependencyGraph,
    covered_obligations: frozenset[DependencyObligation],
) -> frozenset[tuple[int, int]]:
    """Choose the minimal source-ordered root-barrier fallback edges."""
    selected_edges: set[tuple[int, int]] = set()
    ordered_root_edges: set[tuple[int, int]] = set()
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
            dependency_graph.dependency_obligations(access_dependency)
            <= covered_obligations
            for access_dependency in dependency.access_dependencies
        ):
            continue
        if _is_ordered_by_root_barrier(*pair, ordered_root_edges):
            continue
        selected_edges.add(pair)
        ordered_root_edges.add(pair)
    return frozenset(selected_edges)


def _is_ordered_by_root_barrier(
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
