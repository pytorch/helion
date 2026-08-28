from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from .. import exc
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cross_loop_scheduler import CountedEventPlan
from .cross_loop_scheduler import EventContribution
from .cross_loop_scheduler import EventUse
from .cross_loop_scheduler import build_cross_loop_schedule
from .host_function import HostFunction
from .program_id import _ast_fingerprint
from .program_id import _clone_ast_value
from .program_id import _clone_opaque_statements_with_loop_rewrite
from .program_id import _clone_opaque_statements_with_scope_stages
from .program_id import typed_program_id
from .tile_dependency import LogicalRelation
from .tile_dependency import instantiate_root_domains
from .tile_dependency import logical_axis_symbol

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .program_id import ForEachProgramID
    from .program_id import PersistentProgramIDs


# Independent keyed-event counters occupy distinct cache lines so polling one
# dependency does not contend with publication to another. Cross-loop event
# state is currently CUDA-only, where 128 bytes is a conservative L2 line
# alignment. The layout is expressed in bytes rather than a model-shaped
# counter count.
_CROSS_LOOP_COUNTER_ALIGNMENT_BYTES = 128
_CROSS_LOOP_COUNTER_DTYPE = torch.uint32
_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS = (
    _CROSS_LOOP_COUNTER_ALIGNMENT_BYTES // _CROSS_LOOP_COUNTER_DTYPE.itemsize
)


def emit_cross_loop_schedule(
    owner: ForEachProgramID,
    strategy: PersistentProgramIDs,
    device_function: DeviceFunction,
    total_expr: str,
) -> list[ast.stmt]:
    """Emit monotonic arrival-counter phases for a persistent launch.

    Workers publish release arrivals and consumers acquire-poll the
    corresponding counters. The targets are epoch-scaled, so fixed CUDA
    Graph arguments need neither a reset kernel nor a host-side epoch update.
    """
    static_task_counts = owner._static_case_task_counts(device_function)
    if static_task_counts is None:
        CompileEnvironment.current().has_barrier = True
        device_function.triton_minimum_resident_programs = strategy.grid_size_expr
        return owner._emit_phase_loops(strategy, device_function, total_expr)

    worker = typed_program_id(0)
    epoch_var = device_function.new_var("tile_dependency_epoch", dce=False)
    base_body = owner._prepare_persistent_body(
        cast("list[ast.stmt]", device_function.body),
        device_function,
        strategy.virtual_pid_var,
    )
    case_bodies = owner._extract_case_bodies(base_body)
    opaque_case_fingerprints = tuple(_ast_fingerprint(body) for body in case_bodies)
    case_offsets: list[int] = []
    running_offset = 0
    for task_count in static_task_counts:
        case_offsets.append(running_offset)
        running_offset += task_count
    dependency_plan = HostFunction.current().device_ir.tile_dependency_graph
    assert dependency_plan is not None
    indexing = device_function.config.get("indexing", ())

    def uses_tensor_descriptor(memory_op_index: int) -> bool:
        if isinstance(indexing, str):
            return indexing == "tensor_descriptor"
        return (
            isinstance(indexing, (list, tuple))
            and 0 <= memory_op_index < len(indexing)
            and indexing[memory_op_index] == "tensor_descriptor"
        )

    unpublishable_scope_ids = frozenset(
        scope_id
        for access in dependency_plan.accesses
        if access.kind == "store" and uses_tensor_descriptor(access.memory_op_index)
        for scope_id in dependency_plan.scope_ids_by_access[access.access_id]
    )
    publishable_scope_ids = (
        frozenset(
            scope.scope_id
            for scope in dependency_plan.execution_scopes
            if not scope.is_root and scope.scope_id not in unpublishable_scope_ids
        )
        if unpublishable_scope_ids
        else None
    )
    configured_worker_count = CompileEnvironment.current().config_spec.num_sm * cast(
        "int", device_function.config.get("num_sm_multiplier", 1)
    )
    axis_geometry = {
        block_id: geometry
        for block_id in range(len(CompileEnvironment.current().block_sizes))
        if (geometry := owner._static_block_axis_geometry(block_id, device_function))
        is not None
    }
    configured_root_domains = instantiate_root_domains(
        dependency_plan,
        axis_geometry=axis_geometry,
    )
    if any(domain is None for domain in configured_root_domains):
        raise exc.CrossLoopSchedulingError(
            "cross-loop scheduling requires static root domains"
        )
    root_domains = tuple(
        domain for domain in configured_root_domains if domain is not None
    )
    root_traversals = owner._root_physical_traversals(
        device_function,
        root_domains,
    )
    assert root_traversals is not None
    assert [domain.size for domain in root_domains] == static_task_counts
    cross_loop_schedule = build_cross_loop_schedule(
        dependency_plan=dependency_plan,
        root_domains=root_domains,
        root_traversals=root_traversals,
        axis_geometry=axis_geometry,
        worker_count=configured_worker_count,
        publishable_scope_ids=publishable_scope_ids,
    )
    root_completion_edges = set(cross_loop_schedule.root_completion_edges)
    family_done_event_plans = tuple(
        plan
        for plan in cross_loop_schedule.counted_events
        if plan.graph_event_index is not None
        and cross_loop_schedule.event_graph.event(plan.graph_event_index).is_family_done
    )
    nested_scope_event_plans = tuple(
        plan
        for plan in cross_loop_schedule.counted_events
        if any(use.consumer_scope_id is not None for use in plan.uses)
    )
    counted_event_plans = tuple(
        plan
        for plan in cross_loop_schedule.counted_events
        if plan not in family_done_event_plans and plan not in nested_scope_event_plans
    )
    all_counted_event_plans = (
        *counted_event_plans,
        *nested_scope_event_plans,
    )
    launch_worker_count = cross_loop_schedule.worker_schedule.worker_count

    static_workers_by_root = {
        root: cross_loop_schedule.worker_schedule.workers_for_root(root)
        for root in range(len(root_domains))
    }
    local_task_count_by_root: dict[int, int] = {}
    for trigger in cross_loop_schedule.local_triggers:
        event = cross_loop_schedule.event_graph.event(trigger.event_index)
        use = event.uses[trigger.use_index]
        if not use.keys.is_total_function():
            raise AssertionError("a local event must cover its complete root")
        local_task_count_by_root[use.consumer_root] = use.keys.source_domain.size

    def active_worker_count(root: int) -> int:
        return len(static_workers_by_root[root])

    def root_completion_arrival_count(root: int) -> int:
        return active_worker_count(root) + local_task_count_by_root.get(root, 0)

    # Reject grids that cannot residently fit when the device is otherwise
    # idle. Concurrent-stream residency remains an explicit unresolved
    # contract for this non-cooperative static lowering.
    device_function.triton_minimum_resident_programs = strategy.grid_size_expr
    device_function.preamble.extend(strategy._persistent_setup_statements(total_expr))
    counted_event_offsets: dict[CountedEventPlan, int] = {}
    counted_event_counter_count = 0
    counted_event_key_stride = _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    for plan in all_counted_event_plans:
        counted_event_offsets[plan] = counted_event_counter_count
        counted_event_counter_count += plan.key_count * counted_event_key_stride
    root_completion_producer_roots = sorted(
        {producer for producer, _ in root_completion_edges}
    )
    root_completion_indices = {
        root: index for index, root in enumerate(root_completion_producer_roots)
    }
    state_count = 0

    def reserve_state(count: int) -> int | None:
        nonlocal state_count
        if not count:
            return None
        state_count = (
            (state_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
            // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
            * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        )
        offset = state_count
        state_count += count
        return offset

    counted_event_state_offset = reserve_state(counted_event_counter_count)
    root_completion_state_offset = reserve_state(
        len(root_completion_producer_roots) * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    static_state_base = str(
        (launch_worker_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
        // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    state_arg = owner._register_cross_loop_state(
        device_function,
        name_hint="tile_dependency_state",
        numel=f"{static_state_base} + {state_count}",
        dtype=_CROSS_LOOP_COUNTER_DTYPE,
    )

    def state_section(offset: int | None) -> str | None:
        if offset is None:
            return None
        return f"{state_arg} + ({static_state_base}) + {offset}"

    epoch_arg = state_arg
    counted_event_arg = state_section(counted_event_state_offset)
    root_completion_counter_arg = state_section(root_completion_state_offset)

    stage_root_ranges = owner._cross_loop_stage_root_ranges()
    result: list[ast.stmt] = [
        statement_from_string(f"{epoch_var} = tl.load({epoch_arg} + {worker}) + 1")
    ]
    consumed_on_ready_roots = {
        local_use.consumer_root
        for plan in counted_event_plans
        if (local_use := plan.local_use) is not None
    }

    root_completion_incoming: dict[int, tuple[int, ...]] = {
        consumer: tuple(
            sorted(
                producer
                for producer, target in root_completion_edges
                if target == consumer
            )
        )
        for consumer in {consumer for _, consumer in root_completion_edges}
    }

    def root_completion_counter(root: int) -> str:
        assert root_completion_counter_arg is not None
        return (
            f"{root_completion_counter_arg} + "
            f"{root_completion_indices[root] * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}"
        )

    def root_completion_dependency(root: int) -> tuple[str, str]:
        arrivals = root_completion_arrival_count(root)
        return (
            root_completion_counter(root),
            f"tl.cast({epoch_var}, tl.uint32) * tl.cast({arrivals}, tl.uint32)",
        )

    def root_completion_input_dependencies(
        root: int,
    ) -> tuple[tuple[str, str], ...]:
        producers = root_completion_incoming.get(root, ())
        return tuple(root_completion_dependency(producer) for producer in producers)

    def root_completion_publication(root: int) -> list[ast.stmt]:
        if root not in root_completion_indices:
            return []
        completion_counter = root_completion_counter(root)
        arrivals = root_completion_arrival_count(root)
        result = [owner._cross_loop_publication_barrier(device_function)]
        if arrivals == 1:
            result.append(
                statement_from_string(
                    f"tl.atomic_xchg({completion_counter}, {epoch_var}, "
                    "sem='release', scope='gpu')"
                )
            )
        else:
            result.append(
                statement_from_string(
                    f"tl.atomic_add({completion_counter}, 1, "
                    "sem='release', scope='gpu')"
                )
            )
        return result

    root_physical_axis_order: list[tuple[int, ...]] = []
    root_logical_axis_order: list[tuple[int, ...]] = []
    root_axis_counts: list[dict[int, int]] = []
    for root, domain in enumerate(root_domains):
        geometry = owner._static_case_geometry(root, device_function)
        if geometry is None:
            raise AssertionError("configured root geometry became dynamic")
        physical_axis_order, _axis_counts, _block_sizes = geometry
        root_physical_axis_order.append(physical_axis_order)
        root_logical_axis_order.append(domain.axis_order)
        root_axis_counts.append(domain.axis_counts)
    root_events_by_producer: dict[
        int,
        list[
            tuple[
                CountedEventPlan,
                EventContribution,
            ]
        ],
    ] = {}
    producer_events_by_scope: dict[
        int,
        list[
            tuple[
                CountedEventPlan,
                EventContribution,
            ]
        ],
    ] = {}
    for plan in all_counted_event_plans:
        for contributor in plan.contributors:
            if contributor.producer_scope_id is None:
                root_events_by_producer.setdefault(
                    contributor.producer_root, []
                ).append((plan, contributor))
            else:
                producer_events_by_scope.setdefault(
                    contributor.producer_scope_id, []
                ).append((plan, contributor))
    nested_producer_roots = {
        contributor.producer_root
        for contributions in producer_events_by_scope.values()
        for _plan, contributor in contributions
    }
    scheduled_task_roots = {
        root
        for root, traversal in enumerate(root_traversals)
        if any(
            segment.task_relation != traversal
            for segment in cross_loop_schedule.worker_schedule.segments_for_root(root)
        )
    }
    counted_event_uses_by_waiting_root: dict[
        int,
        list[tuple[CountedEventPlan, EventUse]],
    ] = {}
    for plan in counted_event_plans:
        for use_index, use in enumerate(plan.uses):
            if use_index == plan.local_trigger_use:
                continue
            if use.consumer_scope_id is not None:
                continue
            counted_event_uses_by_waiting_root.setdefault(use.consumer_root, []).append(
                (plan, use)
            )
    nested_scope_events_by_consumer: dict[
        int,
        list[tuple[CountedEventPlan, EventUse]],
    ] = {}
    for plan in nested_scope_event_plans:
        for use in plan.uses:
            if use.consumer_scope_id is None:
                continue
            nested_scope_events_by_consumer.setdefault(use.consumer_root, []).append(
                (plan, use)
            )

    def flat_task_coordinates(
        task: str,
        axis_order: tuple[int, ...],
        counts: dict[int, int],
    ) -> dict[int, str]:
        coordinates: dict[int, str] = {}
        multiplier = 1
        for block_id in axis_order:
            count = counts[block_id]
            if count == 1:
                coordinates[block_id] = "0"
            elif multiplier == 1:
                coordinates[block_id] = f"(({task}) % {count})"
            else:
                coordinates[block_id] = f"((({task}) // {multiplier}) % {count})"
            multiplier *= counts[block_id]
        return coordinates

    def relation_expression(
        expression: sympy.Expr,
        coordinates: dict[int, str],
    ) -> str:
        """Render the restricted logical-relation expression grammar."""
        if isinstance(expression, sympy.Integer):
            return str(int(expression))
        if isinstance(expression, sympy.Symbol):
            axis = next(
                (
                    axis
                    for axis in coordinates
                    if logical_axis_symbol(axis) == expression
                ),
                None,
            )
            if axis is None:
                raise AssertionError(f"unknown logical relation symbol {expression}")
            return f"({coordinates[axis]})"
        if isinstance(expression, sympy.Add):
            return " + ".join(
                f"({relation_expression(cast('sympy.Expr', term), coordinates)})"
                for term in expression.as_ordered_terms()
            )
        if isinstance(expression, sympy.Mul):
            return " * ".join(
                f"({relation_expression(factor, coordinates)})"
                for factor in expression.as_ordered_factors()
            )
        if expression.func in (sympy.floor, sympy.ceiling):
            numerator, denominator = sympy.fraction(sympy.together(expression.args[0]))
            if not isinstance(denominator, sympy.Integer) or denominator <= 0:
                raise AssertionError(
                    "logical floor/ceiling requires a positive static divisor"
                )
            numerator_expr = relation_expression(numerator, coordinates)
            if expression.func == sympy.floor:
                return f"(({numerator_expr}) // {int(denominator)})"
            return f"(-((-({numerator_expr})) // {int(denominator)}))"
        if expression.func in (sympy.Min, sympy.Max):
            function = "tl.minimum" if expression.func == sympy.Min else "tl.maximum"
            rendered = relation_expression(
                cast("sympy.Expr", expression.args[0]), coordinates
            )
            for argument in expression.args[1:]:
                rendered = (
                    f"{function}(({rendered}), "
                    f"({relation_expression(cast('sympy.Expr', argument), coordinates)}))"
                )
            return rendered
        if isinstance(expression, sympy.Mod):
            numerator, denominator = expression.args
            if not isinstance(denominator, sympy.Integer) or denominator <= 0:
                raise AssertionError(
                    "logical modulo requires a positive static divisor"
                )
            return (
                f"(({relation_expression(cast('sympy.Expr', numerator), coordinates)}) % "
                f"{int(denominator)})"
            )
        if isinstance(expression, sympy.Rational):
            if expression.q == 1:
                return str(expression.p)
            return f"({expression.p} / {expression.q})"
        raise AssertionError(f"unsupported logical relation expression {expression!r}")

    def relation_source_membership(
        bounds: tuple[tuple[int, int, int, int], ...],
        coordinates: dict[int, str],
    ) -> str:
        conditions: list[str] = []
        for axis, begin, end, step in bounds:
            coordinate = coordinates[axis]
            conditions.extend(
                (
                    f"({coordinate}) >= {begin}",
                    f"({coordinate}) < {end}",
                )
            )
            if step != 1:
                conditions.append(f"(({coordinate}) - {begin}) % {step} == 0")
        return " and ".join(conditions) or "True"

    def relation_point_coordinates(
        relation: LogicalRelation,
        source_coordinates: dict[int, str],
    ) -> tuple[dict[int, str], str]:
        """Render an at-most-one-valued relation without task tables."""
        canonical = relation.canonical_single_valued()
        if canonical is None or not canonical.pieces:
            raise AssertionError("event relation is not single-valued")
        memberships: list[str] = []
        values_by_axis: dict[int, list[tuple[str, str]]] = {
            axis: [] for axis in canonical.target_domain.axis_order
        }
        for piece in canonical.pieces:
            source_membership = relation_source_membership(
                piece.source_bounds_items,
                source_coordinates,
            )
            target_memberships: list[str] = []
            piece_values: dict[int, str] = {}
            for axis, begin, end, step in piece.target_ranges:
                if (
                    step != 1
                    or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                    != 1
                ):
                    raise AssertionError("event relation target is not one point")
                value = relation_expression(begin, source_coordinates)
                piece_values[axis] = value
                target_memberships.extend(
                    (
                        f"({value}) >= 0",
                        f"({value}) < {canonical.target_domain.axis_counts[axis]}",
                    )
                )
            membership = " and ".join((source_membership, *target_memberships))
            memberships.append(membership)
            for axis, value in piece_values.items():
                values_by_axis[axis].append((membership, value))

        def select(values: list[tuple[str, str]]) -> str:
            expressions = tuple(dict.fromkeys(value for _membership, value in values))
            if len(expressions) == 1:
                return expressions[0]
            result = values[-1][1]
            for membership, value in reversed(values[:-1]):
                result = f"tl.where({membership}, {value}, {result})"
            return result

        return (
            {axis: select(values) for axis, values in values_by_axis.items()},
            (
                "True"
                if canonical.is_total_function()
                else " or ".join(f"({membership})" for membership in memberships)
            ),
        )

    def relation_flat_target(
        relation: LogicalRelation,
        source_coordinates: dict[int, str],
    ) -> tuple[str, str]:
        target_coordinates, membership = relation_point_coordinates(
            relation,
            source_coordinates,
        )
        terms: list[str] = []
        multiplier = 1
        for axis in relation.target_domain.axis_order:
            count = relation.target_domain.axis_counts[axis]
            if count != 1:
                coordinate = target_coordinates[axis]
                terms.append(
                    f"({coordinate})"
                    if multiplier == 1
                    else f"({coordinate}) * {multiplier}"
                )
            multiplier *= count
        return " + ".join(terms) or "0", membership

    from .tile_strategy import L2GroupingProgramIDs

    def logical_coordinates_for_physical_task(
        root: int,
        physical_task: str,
    ) -> dict[int, str]:
        """Apply the root's existing PID traversal to one physical task."""
        axis_order = root_physical_axis_order[root]
        counts = root_axis_counts[root]
        case = owner.cases[root]
        if not isinstance(case, L2GroupingProgramIDs) or (
            len(axis_order) >= 2
            and (counts[axis_order[1]] == 1 or case.group_size >= counts[axis_order[0]])
        ):
            return flat_task_coordinates(physical_task, axis_order, counts)

        assert len(axis_order) >= 2
        first_axis, second_axis = axis_order[:2]
        first_count = counts[first_axis]
        second_count = counts[second_axis]
        group_size = case.group_size
        inner_size = first_count * second_count
        group_span = group_size * second_count
        inner_task = (
            f"(({physical_task}) % {inner_size})"
            if len(axis_order) > 2
            else f"({physical_task})"
        )
        group = f"(({inner_task}) // {group_span})"
        first_in_group = f"({group}) * {group_size}"
        actual_group_size = (
            f"tl.minimum({first_count} - ({first_in_group}), {group_size})"
        )
        within_group = f"(({inner_task}) % {group_span})"
        coordinates = {
            first_axis: (
                f"({first_in_group}) + (({within_group}) % ({actual_group_size}))"
            ),
            second_axis: f"({within_group}) // ({actual_group_size})",
        }
        multiplier = inner_size
        for block_id in axis_order[2:]:
            count = counts[block_id]
            coordinates[block_id] = (
                "0"
                if count == 1
                else f"((({physical_task}) // {multiplier}) % {count})"
            )
            multiplier *= counts[block_id]
        return coordinates

    def physical_task_for_logical_coordinates(
        root: int,
        coordinates: dict[int, str],
    ) -> str:
        """Invert the root's PID traversal without changing its body."""
        axis_order = root_physical_axis_order[root]
        counts = root_axis_counts[root]
        case = owner.cases[root]
        if not isinstance(case, L2GroupingProgramIDs) or (
            len(axis_order) >= 2
            and (counts[axis_order[1]] == 1 or case.group_size >= counts[axis_order[0]])
        ):
            terms: list[str] = []
            multiplier = 1
            for block_id in axis_order:
                count = counts[block_id]
                if count != 1:
                    coordinate = coordinates[block_id]
                    terms.append(
                        f"({coordinate})"
                        if multiplier == 1
                        else f"({coordinate}) * {multiplier}"
                    )
                multiplier *= count
            return " + ".join(terms) or "0"

        assert len(axis_order) >= 2
        first_axis, second_axis = axis_order[:2]
        first_count = counts[first_axis]
        second_count = counts[second_axis]
        group_size = case.group_size
        first_coordinate = coordinates[first_axis]
        second_coordinate = coordinates[second_axis]
        group = f"(({first_coordinate}) // {group_size})"
        first_in_group = f"({group}) * {group_size}"
        actual_group_size = (
            f"tl.minimum({first_count} - ({first_in_group}), {group_size})"
        )
        inner_task = (
            f"({group}) * {group_size * second_count} + "
            f"({second_coordinate}) * ({actual_group_size}) + "
            f"({first_coordinate}) - ({first_in_group})"
        )
        terms = [f"({inner_task})"]
        multiplier = first_count * second_count
        for block_id in axis_order[2:]:
            count = counts[block_id]
            if count != 1:
                terms.append(f"({coordinates[block_id]}) * {multiplier}")
            multiplier *= count
        return " + ".join(terms)

    def logical_task_from_coordinates(
        root: int,
        coordinates: dict[int, str],
    ) -> str:
        terms: list[str] = []
        multiplier = 1
        for block_id in root_logical_axis_order[root]:
            count = root_axis_counts[root][block_id]
            if count != 1:
                coordinate = coordinates[block_id]
                terms.append(
                    f"({coordinate})"
                    if multiplier == 1
                    else f"({coordinate}) * {multiplier}"
                )
            multiplier *= count
        return " + ".join(terms) or "0"

    def body_with_scope_waits(
        plan: CountedEventPlan,
        use: EventUse,
        body: list[ast.stmt],
        consumer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        assert use.consumer_scope_id is not None
        domain = cross_loop_schedule.event_graph.scope_domain(use.consumer_scope_id)
        nested_axes = cross_loop_schedule.event_graph.nested_axes(
            use.consumer_root,
            use.consumer_scope_id,
        )
        if len(nested_axes) != 1:
            raise AssertionError(
                "nested scope lowering currently requires one loop axis"
            )
        (nested_axis,) = nested_axes
        boundaries = tuple(
            sorted(
                {
                    boundary
                    for piece in use.keys.pieces
                    for axis, begin, end, _step in piece.source_bounds_items
                    if axis == nested_axis
                    for boundary in (begin, end)
                    if 0 < boundary < domain.axis_counts[nested_axis]
                }
            )
        )
        stage_offsets = (0, *boundaries)
        stage_waits: list[tuple[ast.stmt, ...]] = []
        for action_offset in stage_offsets:
            scope_coordinates = {
                **consumer_coordinates,
                nested_axis: str(action_offset),
            }
            event_key, membership = relation_flat_target(
                use.keys,
                scope_coordinates,
            )
            if membership == "False":
                raise AssertionError("nested scope stage has no event key")
            stage_waits.append(
                tuple(
                    owner._wait_for_counter(
                        device_function=device_function,
                        counter=counted_event_counter(plan, event_key),
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({counted_event_expected_arrivals(plan, event_key)}, tl.uint32)"
                        ),
                        prefix="tile_dependency_scope_wait",
                    )
                )
            )
        return _clone_opaque_statements_with_scope_stages(
            body,
            scope_id=use.consumer_scope_id,
            split_iteration_offsets=boundaries,
            stage_waits=tuple(stage_waits),
        )

    def counted_event_counter(plan: CountedEventPlan, key: str) -> str:
        assert counted_event_arg is not None
        return (
            f"{counted_event_arg} + {counted_event_offsets[plan]} + "
            f"({key}) * {counted_event_key_stride}"
        )

    def counted_event_uniform_arrivals(
        plan: CountedEventPlan,
    ) -> int | None:
        total = 0
        for contributor in plan.contributors:
            cardinality = contributor.arrivals_per_key
            count = None if cardinality is None else cardinality.constant_value()
            if count is None:
                return None
            total += count
        return total

    def counted_event_expected_arrivals(
        plan: CountedEventPlan,
        key: str,
    ) -> str:
        uniform = counted_event_uniform_arrivals(plan)
        if uniform is not None:
            return str(uniform)
        key_coordinates = flat_task_coordinates(
            key,
            plan.key_domain.axis_order,
            plan.key_domain.axis_counts,
        )
        expressions: list[str] = []
        for contributor in plan.contributors:
            cardinality = contributor.arrivals_per_key
            if cardinality is None:
                raise AssertionError("event fan-in is not symbolically known")
            values, _membership = relation_point_coordinates(
                cardinality,
                key_coordinates,
            )
            expressions.append(values[cardinality.target_domain.axis_order[0]])
        return " + ".join(f"({expression})" for expression in expressions)

    def emit_counted_event_for_key(
        plan: CountedEventPlan,
        key: str,
    ) -> list[ast.stmt]:
        local_use = plan.local_use
        if local_use is None:
            counter = counted_event_counter(plan, key)
            if counted_event_uniform_arrivals(plan) == 1:
                return [
                    statement_from_string(
                        f"tl.atomic_xchg({counter}, {epoch_var}, "
                        "sem='release', scope='gpu')"
                    )
                ]
            return [
                statement_from_string(
                    f"tl.atomic_add({counter}, 1, sem='release', scope='gpu')"
                )
            ]

        inverse_use = local_use.keys.inverse()
        if inverse_use is None or not inverse_use.is_total_function():
            raise AssertionError("local event use must bijectively cover its consumer")
        if inverse_use.is_positional_bijection():
            consumer_task_expression = key
        else:
            key_coordinates = flat_task_coordinates(
                key,
                plan.key_domain.axis_order,
                plan.key_domain.axis_counts,
            )
            consumer_coordinates, _membership = relation_point_coordinates(
                inverse_use,
                key_coordinates,
            )
            consumer_task_expression = logical_task_from_coordinates(
                local_use.consumer_root,
                consumer_coordinates,
            )
        consumer_task = device_function.new_var(
            "tile_dependency_continuation_task", dce=True
        )
        assignments = [
            statement_from_string(f"{consumer_task} = {consumer_task_expression}")
        ]
        on_ready_root = local_use.consumer_root
        consumer_coordinates = flat_task_coordinates(
            consumer_task,
            root_logical_axis_order[on_ready_root],
            root_axis_counts[on_ready_root],
        )
        consumer_logical_pid = (
            f"{case_offsets[on_ready_root]} + "
            f"{physical_task_for_logical_coordinates(on_ready_root, consumer_coordinates)}"
        )
        consumer_extra_arguments = (consumer_task,)
        previous = device_function.new_var(
            "tile_dependency_continuation_previous", dce=False
        )
        consumer_call = owner._outline_opaque_tile_body(
            device_function,
            root=on_ready_root,
            logical_pid=consumer_logical_pid,
            body=body_with_scope_publications(
                on_ready_root,
                case_bodies[on_ready_root],
                consumer_coordinates,
            ),
            extra_argument_names=consumer_extra_arguments,
        )
        consumer_publications: list[ast.stmt] = []
        for nested_event, nested_contributor in root_events_by_producer.get(
            on_ready_root, ()
        ):
            consumer_publications.extend(
                emit_counted_event_from_producer_coordinates(
                    nested_event,
                    nested_contributor,
                    consumer_coordinates,
                )
            )

        last_arrival_body = [consumer_call]
        if consumer_publications:
            last_arrival_body.append(
                owner._cross_loop_publication_barrier(device_function)
            )
            last_arrival_body.extend(consumer_publications)
        last_arrival_body.extend(root_completion_publication(on_ready_root))
        expected_arrivals = counted_event_uniform_arrivals(plan)
        if expected_arrivals is None:
            raise AssertionError("local execution requires uniform event fan-in")
        if expected_arrivals == 1:
            return [*assignments, *last_arrival_body]
        arrival_counter = counted_event_counter(plan, key)
        return [
            *assignments,
            *owner._emit_counted_event_on_ready(
                counter=arrival_counter,
                epoch=epoch_var,
                expected_arrivals=expected_arrivals,
                previous=previous,
                on_ready=last_arrival_body,
            ),
        ]

    def emit_counted_event_from_producer_coordinates(
        plan: CountedEventPlan,
        contributor: EventContribution,
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        publication = contributor.producer_to_keys
        if publication is None:
            raise AssertionError("event publication relation is unavailable")
        key, membership = relation_flat_target(
            publication,
            producer_coordinates,
        )
        publications = emit_counted_event_for_key(plan, key)
        if membership == "True":
            return publications
        return [
            create(
                ast.If,
                test=expr_from_string(membership),
                body=publications,
                orelse=[],
            )
        ]

    def body_with_scope_publications(
        root: int,
        body: list[ast.stmt],
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        """Publish nested scope events without moving the owning strand."""
        from .tile_dependency import tile_dependency_scope_id

        scope_ids = {
            scope_id
            for scope_id, contributions in producer_events_by_scope.items()
            if any(
                contributor.producer_root == root
                for _plan, contributor in contributions
            )
        }
        if not scope_ids:
            return body
        emitted_scope_ids: set[int] = set()

        def rewrite(loop: ast.For) -> list[ast.stmt] | None:
            scope_id = tile_dependency_scope_id(loop)
            if scope_id is None or scope_id not in scope_ids:
                return None
            if scope_id in emitted_scope_ids:
                raise AssertionError(
                    "one dependency scope must identify one lowered loop"
                )
            nested_axes = cross_loop_schedule.event_graph.nested_axes(root, scope_id)
            if (
                len(nested_axes) != 1
                or not isinstance(loop.target, ast.Name)
                or not isinstance(loop.iter, ast.Call)
                or len(loop.iter.args) < 2
            ):
                raise AssertionError(
                    "nested scope publication requires one range-like loop axis"
                )
            nested_axis = nested_axes[0]
            begin = ast.unparse(loop.iter.args[0])
            step = ast.unparse(loop.iter.args[2]) if len(loop.iter.args) >= 3 else "1"
            scope_coordinates = {
                **producer_coordinates,
                nested_axis: f"(({loop.target.id}) - ({begin})) // ({step})",
            }

            publications: list[ast.stmt] = []
            for plan, contributor in producer_events_by_scope[scope_id]:
                publication = contributor.producer_to_keys
                if publication is None:
                    raise AssertionError(
                        "nested event publication relation is unavailable"
                    )
                key, membership = relation_flat_target(
                    publication,
                    scope_coordinates,
                )
                event_publications = emit_counted_event_for_key(plan, key)
                if membership == "True":
                    publications.extend(event_publications)
                else:
                    publications.append(
                        create(
                            ast.If,
                            test=expr_from_string(membership),
                            body=event_publications,
                            orelse=[],
                        )
                    )

            cloned = cast("ast.For", _clone_ast_value(loop))
            computation = _ast_fingerprint(cloned.body)
            cloned.body.extend(
                [
                    owner._cross_loop_publication_barrier(device_function),
                    *publications,
                ]
            )
            if _ast_fingerprint(cloned.body[: -len(publications) - 1]) != computation:
                raise AssertionError(
                    "nested scope publication changed the loop computation"
                )
            emitted_scope_ids.add(scope_id)
            return [cloned]

        result = _clone_opaque_statements_with_loop_rewrite(body, rewrite)
        if emitted_scope_ids != scope_ids:
            missing = sorted(scope_ids - emitted_scope_ids)
            raise AssertionError(f"missing nested producer scopes {missing}")
        return result

    def scheduled_logical_task_expression(
        root: int,
        schedule_ordinal: str,
    ) -> str | None:
        """Map one root-local schedule position to its logical task ID."""
        if root not in scheduled_task_roots:
            return None
        worker_schedule = cross_loop_schedule.worker_schedule
        root_domain = root_domains[root]
        segments = worker_schedule.segments_for_root(root)
        schedule_interval = worker_schedule.contiguous_global_interval(root)
        if (
            schedule_interval is None
            or schedule_interval[1] - schedule_interval[0] != root_domain.size
        ):
            raise AssertionError(
                f"root {root} does not occupy one contiguous schedule interval"
            )
        schedule_begin = schedule_interval[0]

        expression = ""
        for segment in reversed(segments):
            ordinal_begin = segment.schedule_begin - schedule_begin
            ordinal_delta = f"(({schedule_ordinal}) - {ordinal_begin})"
            if segment.schedule_period is None:
                task_offset = f"({ordinal_delta} // {segment.schedule_step})"
                membership = (
                    f"({ordinal_delta}) >= 0 and "
                    f"({ordinal_delta} % {segment.schedule_step}) == 0 and "
                    f"({task_offset}) < {segment.task_count}"
                )
            else:
                assert segment.schedule_period_step is not None
                within_period = f"({ordinal_delta} % {segment.schedule_period_step})"
                inner = f"({within_period} // {segment.schedule_step})"
                task_offset = (
                    f"({ordinal_delta} // {segment.schedule_period_step}) * "
                    f"{segment.schedule_period} + {inner}"
                )
                membership = (
                    f"({ordinal_delta}) >= 0 and "
                    f"({within_period} % {segment.schedule_step}) == 0 and "
                    f"({inner}) < {segment.schedule_period} and "
                    f"({task_offset}) < {segment.task_count}"
                )
            if segment.task_relation is not None:
                ordinal_coordinates = flat_task_coordinates(
                    task_offset,
                    segment.task_relation.source_domain.axis_order,
                    segment.task_relation.source_domain.axis_counts,
                )
                task_coordinates, relation_membership = relation_point_coordinates(
                    segment.task_relation,
                    ordinal_coordinates,
                )
                if relation_membership != "True":
                    membership = f"({membership}) and ({relation_membership})"
                segment_task = logical_task_from_coordinates(
                    root,
                    task_coordinates,
                )
            elif segment.task_period is None:
                segment_task = (
                    f"{segment.task_begin} + ({task_offset}) * {segment.task_step}"
                )
            else:
                assert segment.task_period_step is not None
                segment_task = (
                    f"{segment.task_begin} + "
                    f"(({task_offset}) % {segment.task_period}) * "
                    f"{segment.task_step} + "
                    f"(({task_offset}) // {segment.task_period}) * "
                    f"{segment.task_period_step}"
                )
            if not expression:
                expression = segment_task
                continue
            expression = f"tl.where({membership}, {segment_task}, {expression})"
        if not expression:
            raise AssertionError(f"root {root} has no static schedule")
        return expression

    def task_scheduled_body(
        root: int,
        local_task: str,
        logical_pid: str,
        extra_argument_names: tuple[str, ...],
        *,
        force_noinline: bool = False,
    ) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        has_task_scheduling = root in nested_scope_events_by_consumer
        producer_events = tuple(root_events_by_producer.get(root, ()))
        scheduled_local_task = local_task
        scheduled_logical_pid = logical_pid
        scheduled_coordinates: dict[int, str] | None = None
        if producer_events or root in nested_producer_roots:
            has_task_scheduling = True
        if (
            logical_task_expr := scheduled_logical_task_expression(
                root,
                local_task,
            )
        ) is not None:
            scheduled_task = device_function.new_var(
                "tile_dependency_scheduled_logical_task", dce=True
            )
            physical_task = device_function.new_var(
                "tile_dependency_scheduled_physical_task", dce=True
            )
            scheduled_coordinates = flat_task_coordinates(
                scheduled_task,
                root_logical_axis_order[root],
                root_axis_counts[root],
            )
            physical_task_expr = physical_task_for_logical_coordinates(
                root, scheduled_coordinates
            )
            body.extend(
                [
                    statement_from_string(f"{scheduled_task} = {logical_task_expr}"),
                    statement_from_string(f"{physical_task} = {physical_task_expr}"),
                ]
            )
            scheduled_local_task = physical_task
            scheduled_logical_pid = f"{case_offsets[root]} + {physical_task}"
        if scheduled_coordinates is None:
            scheduled_coordinates = logical_coordinates_for_physical_task(
                root, scheduled_local_task
            )
        for (
            incoming_counted_event,
            incoming_use,
        ) in counted_event_uses_by_waiting_root.get(root, ()):
            has_task_scheduling = True
            assert counted_event_arg is not None
            event_key, membership = relation_flat_target(
                incoming_use.keys,
                scheduled_coordinates,
            )
            wait = owner._wait_for_counter(
                device_function=device_function,
                counter=counted_event_counter(
                    incoming_counted_event,
                    event_key,
                ),
                target=(
                    f"tl.cast({epoch_var}, tl.uint32) * "
                    f"tl.cast({counted_event_expected_arrivals(incoming_counted_event, event_key)}, tl.uint32)"
                ),
                prefix="tile_dependency_keyed_event_wait",
            )
            if not incoming_use.keys.is_total_function():
                body.append(
                    create(
                        ast.If,
                        test=expr_from_string(membership),
                        body=wait,
                        orelse=[],
                    )
                )
            else:
                body.extend(wait)
        nested_event_uses = nested_scope_events_by_consumer.get(root, ())
        if nested_event_uses:
            # Instrument the original logical loop before segmentation.
            # Split ranges then retain the original scope-coordinate
            # expression instead of rebasing publication IDs per segment.
            scheduled_root_body = body_with_scope_publications(
                root,
                case_bodies[root],
                scheduled_coordinates,
            )
            for scope_plan, scope_use in sorted(
                nested_event_uses,
                key=lambda item: (
                    item[1].consumer_scope_id
                    if item[1].consumer_scope_id is not None
                    else -1
                ),
            ):
                scheduled_root_body = body_with_scope_waits(
                    scope_plan,
                    scope_use,
                    scheduled_root_body,
                    scheduled_coordinates,
                )
            opaque_call = owner._outline_cross_loop_region(
                device_function,
                name_hint=f"tile_dependency_root_{root}",
                body=[
                    statement_from_string(
                        f"{owner.shared_pid_var} = {scheduled_logical_pid}"
                    ),
                    *scheduled_root_body,
                ],
                extra_argument_names=extra_argument_names,
            )
        else:
            opaque_call = owner._outline_opaque_tile_body(
                device_function,
                root=root,
                logical_pid=scheduled_logical_pid,
                body=body_with_scope_publications(
                    root,
                    case_bodies[root],
                    scheduled_coordinates,
                ),
                extra_argument_names=extra_argument_names,
                noinline=force_noinline,
            )
        body.append(opaque_call)
        publications: list[ast.stmt] = []
        if publications or producer_events:
            has_task_scheduling = True
            body.append(owner._cross_loop_publication_barrier(device_function))
            body.extend(publications)
        for counted_event, contributor in producer_events:
            body.extend(
                emit_counted_event_from_producer_coordinates(
                    counted_event,
                    contributor,
                    scheduled_coordinates,
                )
            )
        if not has_task_scheduling:
            return body
        return [
            owner._outline_cross_loop_region(
                device_function,
                name_hint=f"tile_dependency_root_{root}_scheduled_task",
                body=body,
                extra_argument_names=extra_argument_names,
                noinline=True,
            )
        ]

    dense_assignment_by_root: dict[int, tuple[int, int, int]] = {}
    for root, root_domain in enumerate(root_domains):
        segments = sorted(
            cross_loop_schedule.worker_schedule.segments_for_root(root),
            key=lambda segment: segment.schedule_begin,
        )
        if not segments:
            continue
        if sum(segment.task_count for segment in segments) != root_domain.size:
            raise exc.CrossLoopSchedulingError(
                "partially static task families are not supported yet"
            )
        worker_begin = segments[0].worker_begin
        worker_count = segments[0].worker_count
        if any(
            segment.worker_begin != worker_begin
            or segment.worker_count != worker_count
            or segment.schedule_step != 1
            or segment.schedule_period is not None
            for segment in segments
        ):
            raise exc.CrossLoopSchedulingError(
                f"root {root} has a noncontiguous static worker assignment"
            )
        schedule_begin = segments[0].schedule_begin
        if schedule_begin % worker_count or any(
            segment.schedule_begin
            != schedule_begin
            + sum(previous.task_count for previous in segments[:index])
            for index, segment in enumerate(segments)
        ):
            raise exc.CrossLoopSchedulingError(
                f"root {root} has a non-dense static worker schedule"
            )
        dense_assignment_by_root[root] = (
            worker_begin,
            worker_count,
            root_domain.size,
        )

    def static_root_body(root: int) -> list[ast.stmt]:
        if root in consumed_on_ready_roots:
            return []
        assignment = dense_assignment_by_root.get(root)
        if assignment is None:
            return []
        worker_begin, worker_count, task_count = assignment
        task_dispatch: list[ast.stmt]
        if task_count == 1:
            task_dispatch = task_scheduled_body(
                root,
                "0",
                str(case_offsets[root]),
                (epoch_var,),
                force_noinline=True,
            )
        else:
            local_task = f"({strategy.virtual_pid_var}) - {case_offsets[root]}"
            task_dispatch = [
                create(
                    ast.For,
                    target=create(
                        ast.Name,
                        id=strategy.virtual_pid_var,
                        ctx=ast.Store(),
                    ),
                    iter=expr_from_string(
                        f"tl.range((({worker}) - {worker_begin}) + "
                        f"({case_offsets[root]}), "
                        f"({case_offsets[root] + task_count}), {worker_count})"
                    ),
                    body=task_scheduled_body(
                        root,
                        local_task,
                        strategy.virtual_pid_var,
                        (strategy.virtual_pid_var,),
                    ),
                    orelse=[],
                    type_comment=None,
                )
            ]
        incoming_roots = root_completion_incoming.get(root, ())
        publishes_completion = root in root_completion_indices
        if (
            worker_begin == 0
            and worker_count == launch_worker_count
            and not incoming_roots
            and not publishes_completion
        ):
            return task_dispatch

        active_body = owner._wait_for_dependencies(
            device_function=device_function,
            dependencies=root_completion_input_dependencies(root),
            prefix="tile_dependency_root_completion_wait",
        )
        active_body.extend(task_dispatch)
        if publishes_completion:
            active_body.extend(root_completion_publication(root))
        condition = (
            f"({worker}) == {worker_begin}"
            if worker_count == 1
            else (
                f"({worker}) >= {worker_begin} and "
                f"({worker}) < {worker_begin + worker_count}"
            )
        )
        return [
            create(
                ast.If,
                test=expr_from_string(condition),
                body=active_body,
                orelse=[],
            )
        ]

    for root_begin, root_end in stage_root_ranges:
        for root in range(root_begin, root_end):
            result.extend(static_root_body(root))
    result.append(
        statement_from_string(f"tl.store({epoch_arg} + {worker}, {epoch_var})")
    )
    if (
        tuple(_ast_fingerprint(body) for body in case_bodies)
        != opaque_case_fingerprints
    ):
        raise AssertionError(
            "tile-dependency lowering mutated an opaque source tile body"
        )
    return result
