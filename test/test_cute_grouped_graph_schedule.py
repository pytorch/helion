from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from test.test_cute_chained_body_program import _source as root_source
from test.test_cute_chained_group_guards import _convert
from test.test_cute_chained_group_guards import _dot
from test.test_cute_chained_group_guards import _input
from test.test_cute_chained_loop_workspace import _loop_plan
from test.test_cute_chained_preparation_frame import _safe as frame_safe
from test.test_cute_chained_preparation_frame import _synthetic
from test.test_cute_chained_recurrence_workspace import _cut_for
from test.test_cute_chained_recurrence_workspace import _safe as recurrence_safe
from test.test_cute_chained_root_weighted_pair import _code as action_source
from test.test_cute_prepared_graph_schedule import _region

from helion._compiler.cute import chained_body_program as bodies
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import prepared_graph_schedule as schedules
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion.language import memory_ops


def _groups(region):
    return tuple(
        ContractionGroup((i,), (StageGeometry((32, 32, 32), False),))
        for i in range(len(region.contractions))
    )


def _schedule(region, groups, **kwargs):
    return schedules.schedule_contraction_groups(
        schedules.ContractionGraph(region),
        groups,
        ((32, 32, 32),) * len(region.contractions),
        partitions=(frozenset(spec.node for spec in region.contractions),),
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["accumulator", "value", "nested"])
def test_unordered_groups_use_actual_dependencies_and_original_ties(kind):
    region = _region(kind, independent_first=True)
    groups = _groups(region)
    result = _schedule(region, groups[::-1])
    assert tuple(item.group for item in result.groups) == groups
    assert all(item.group is groups[i] for i, item in enumerate(result.groups))
    assert [(item.read_event, item.publication_event) for item in result.groups] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert result.transition_event == 6
    for node in region.nodes:
        node.name = "unrelated_" + node.name
    assert _schedule(region, groups[::-1]) == result


def test_logical_identity_is_not_the_lifetime_event():
    region = _region(independent_first=True)
    # The logical identity tuple need not be the FX/topological order. The
    # physical root adapter separately enforces its consecutive completion.
    region = replace(region, contractions=region.contractions[::-1])
    result = _schedule(region, _groups(region))
    assert [item.group.stages for item in result.groups] == [(2,), (1,), (0,)]
    assert [item.read_event for item in result.groups] == [0, 2, 4]


def test_group_members_stay_atomic_and_events_preserve_original_width():
    region = _region(independent_first=True)
    groups = _groups(region)
    joined = ContractionGroup((0, 1), groups[0].geometries + groups[1].geometries)
    result = _schedule(region, (groups[2], joined))
    assert [item.group for item in result.groups] == [joined, groups[2]]
    assert [(item.read_event, item.publication_event) for item in result.groups] == [
        (0, 1),
        (4, 5),
    ]
    assert result.transition_event == 6


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "member_order", "gap", "geometry", "dependent"]
)
def test_malformed_physical_groups_reject_before_events(failure):
    region = _region(independent_first=True)
    groups = _groups(region)
    if failure == "missing":
        groups = groups[:-1]
    elif failure == "duplicate":
        groups = (*groups, groups[0])
    elif failure == "geometry":
        groups = (
            replace(groups[0], geometries=(StageGeometry((16, 32, 32), False),)),
            *groups[1:],
        )
    else:
        members = {"member_order": (1, 0), "gap": (0, 2), "dependent": (1, 2)}[failure]
        joined = ContractionGroup(members, (groups[0].geometries[0],) * 2)
        groups = (
            joined,
            *(group for group in groups if group.stages[0] not in members),
        )
    with pytest.raises(chain._UnsupportedChain):
        _schedule(region, groups)


@pytest.mark.parametrize("cut", ["partition", "materialization"])
def test_original_role_and_materialization_cuts_are_not_crossed(cut):
    region = _region(independent_first=True)
    singletons = _groups(region)
    joined = ContractionGroup(
        (0, 1), singletons[0].geometries + singletons[1].geometries
    )
    if cut == "partition":
        graph = schedules.ContractionGraph(region)
        with pytest.raises(chain._UnsupportedChain, match="original roles"):
            schedules.schedule_contraction_groups(
                graph,
                (joined, singletons[2]),
                ((32, 32, 32),) * 3,
                partitions=(
                    frozenset({region.contractions[0].node}),
                    frozenset(spec.node for spec in region.contractions[1:]),
                ),
            )
    else:
        # Insert a genuine graph node between the two otherwise independent
        # contractions, retaining their original operand definitions.
        first, second, _ = region.contractions
        with region.graph.inserting_before(second.node):
            marker = region.graph.call_function(torch.neg, (first.node,))
        region = replace(region, nodes=tuple(region.graph.nodes))
        with pytest.raises(chain._UnsupportedChain, match="materialization"):
            _schedule(
                region,
                (joined, singletons[2]),
                materializations=frozenset({marker}),
            )


def test_live_out_definition_is_not_its_original_publication_event():
    region = _region(independent_first=True)
    first, second, _ = region.contractions
    with region.graph.inserting_before(second.node):
        live_out = region.graph.call_function(torch.neg, (first.node,))
    region = replace(region, nodes=tuple(region.graph.nodes), live_outs=(live_out,))
    singletons = _groups(region)
    joined = ContractionGroup(
        (0, 1), singletons[0].geometries + singletons[1].geometries
    )
    result = _schedule(region, (singletons[2], joined))
    assert result.groups[0].group is joined
    # The actual terminal effect keeps its dependency. This stage-order pass
    # does not emit the effect at the pure value definition or move its store.
    graph = schedules.ContractionGraph(region)
    assert graph.effects == ((live_out, frozenset({first.node})),)


def _terminal_loop():
    graph = Graph()
    operand = _input(graph, "operand", (16, 16), torch.bfloat16)
    destination = _input(graph, "destination", (16, 16), torch.float32)
    states = tuple(
        _input(graph, f"state_{i}", (16, 16), torch.float32) for i in range(2)
    )
    prepared = _convert(graph, _dot(graph, operand, operand), torch.bfloat16)
    first = _dot(graph, prepared, operand, states[0])
    graph.call_function(memory_ops.store, (destination, [slice(None)], first, None))
    last = _dot(graph, prepared, operand, states[1])
    plan = _loop_plan(graph, states, (first, last))
    geometry = StageGeometry((16, 16, 16), False)
    plan = replace(
        plan,
        contraction_groups=(
            ContractionGroup((0,), (geometry,)),
            ContractionGroup((1, 2), (geometry, geometry)),
        ),
    )
    cut, shapes = _cut_for(plan, (plan.dots[0], prepared), (operand, destination))
    assert plan.region is not None and plan.loop is not None
    placement = schedules.LoopTerminalStorePlacement(
        plan.loop, plan.region.stores, cut.storage_proof_key, frozenset(cut.recurrence)
    )
    return plan, cut, shapes, placement


def _terminal_schedule(plan, cut, placement, graph=None):
    assert plan.region is not None and plan.contraction_groups is not None
    return schedules.schedule_contraction_groups(
        schedules.ContractionGraph(plan.region) if graph is None else graph,
        plan.contraction_groups,
        plan.shapes,
        partitions=(frozenset(cut.preparation), frozenset(cut.recurrence)),
        terminal_stores=placement,
    )


def test_actual_store_cut_requires_original_loop_terminal_placement():
    plan, cut, shapes, placement = _terminal_loop()
    with pytest.raises(chain._UnsupportedChain, match="materialization"):
        _terminal_schedule(plan, cut, None)
    schedule = _terminal_schedule(plan, cut, placement)
    assert schedule.store_events == ((placement.stores[0], 6),)
    with patch.object(
        schedules,
        "schedule_contraction_groups",
        wraps=schedules.schedule_contraction_groups,
    ) as order:
        workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None and order.call_count == 1
    assert order.call_args.kwargs["terminal_stores"].loop is plan.loop
    assert workspace.transition_event == 6
    assert workspace.layout.region("chain_1_c").live_until == 7
    recurrence_safe(workspace)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_store",
        "foreign_store",
        "storage_key",
        "partition",
        "loop_graph",
        "store_operand",
        "inventory",
    ],
)
def test_terminal_placement_rejects_changed_original_authority(mutation):
    plan, cut, _shapes, placement = _terminal_loop()
    assert plan.region is not None
    graph = schedules.ContractionGraph(plan.region)
    if mutation == "missing_store":
        placement = replace(placement, stores=())
    elif mutation == "foreign_store":
        foreign = _terminal_loop()[3]
        placement = replace(placement, stores=foreign.stores)
    elif mutation == "storage_key":
        placement = replace(placement, storage_key="foreign")
    elif mutation == "partition":
        placement = replace(placement, recurrence=frozenset(cut.preparation))
    elif mutation == "loop_graph":
        placement = replace(placement, loop=_terminal_loop()[3].loop)
    elif mutation == "inventory":
        graph = schedules.ContractionGraph(replace(plan.region, stores=()))
    else:
        store = placement.stores[0]
        store.args = (*store.args[:2], plan.dots[-1], store.args[3])
    with pytest.raises(chain._UnsupportedChain):
        _terminal_schedule(plan, cut, placement, graph)


def test_group_order_checks_deep_revision_once():
    region = _region()
    graph = schedules.ContractionGraph(region)
    original = schedules.ContractionGraph.check
    calls = []

    def check(current):
        calls.append(current)
        return original(current)

    with patch.object(schedules.ContractionGraph, "check", check):
        schedules.schedule_contraction_groups(
            graph,
            _groups(region),
            ((32, 32, 32),) * 2,
            partitions=(frozenset(spec.node for spec in region.contractions),),
        )
    assert calls == [graph]
    region.contractions[0].node.args = region.contractions[0].node.args[::-1]
    with pytest.raises(chain._UnsupportedChain, match="dependency graph changed"):
        schedules.schedule_contraction_groups(
            graph,
            _groups(region),
            ((32, 32, 32),) * 2,
            partitions=(frozenset(spec.node for spec in region.contractions),),
        )


@pytest.mark.parametrize("kind", ["dot", "collective", "cache"])
def test_real_loop_builders_consume_unordered_bindings_before_allocation(kind):
    plan, cut, shapes = _synthetic(kind)
    groups = tuple(
        ContractionGroup((i,), (StageGeometry(shape, False),))
        for i, shape in enumerate(plan.shapes)
    )
    ordered = replace(plan, contraction_groups=groups)
    unordered = replace(plan, contraction_groups=groups[::-1])
    expected_frame = plan_preparation_frame(ordered, cut, shapes)
    expected_recurrence = plan_recurrence_workspace(ordered, cut, shapes)
    original = schedules.schedule_contraction_groups
    seen = []

    def observe(graph, bindings, *args, **kwargs):
        assert bindings == groups[::-1]
        result = original(graph, bindings, *args, **kwargs)
        seen.append(result)
        return result

    with patch.object(schedules, "schedule_contraction_groups", observe):
        frame = plan_preparation_frame(unordered, cut, shapes)
        recurrence = plan_recurrence_workspace(unordered, cut, shapes)
    assert frame == expected_frame and frame is not None
    assert recurrence == expected_recurrence and recurrence is not None
    assert len(seen) == 2
    assert [stage.read_event for stage in recurrence.stages] == [
        seen[1].groups[2].read_event
    ]
    frame_safe(frame)
    recurrence_safe(recurrence)


@pytest.mark.parametrize("client", ["materialized", "actions"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_actual_ordinary_root_uses_shared_schedule_before_actions(client, dtype):
    original = schedules.schedule_contraction_groups
    seen = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    with (
        patch.object(schedules, "schedule_contraction_groups", observe),
        patch.object(
            bodies, "emit_body_program", wraps=bodies.emit_body_program
        ) as emit,
    ):
        source = (
            root_source(dtype)
            if client == "materialized"
            else action_source(dtype=dtype)
        )
    assert source and len(seen) == 1 and emit.call_count == 1
    body = emit.call_args.kwargs["root" if client == "materialized" else "root_actions"]
    assert body.consumed
    if client == "materialized":
        assert [action.event for action in body.program.actions] == [
            event
            for item in seen[0].groups
            for event in (item.read_event, item.publication_event)
        ]
    else:
        assert [action.stage for action in body.actions] == [
            item.group.stages[0] for item in seen[0].groups
        ]
