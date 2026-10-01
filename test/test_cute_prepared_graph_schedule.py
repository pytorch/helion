from __future__ import annotations

from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from test.test_cute_contraction_region import _call
from test.test_cute_contraction_region import _input

from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.cute.prepared_continuation import ContractionIssue
from helion._compiler.cute.prepared_graph_schedule import ContractionGraph
from helion._compiler.cute.prepared_graph_schedule import DependencyKind
from helion._compiler.cute.prepared_graph_schedule import IssuePlacement
from helion._compiler.cute.prepared_graph_schedule import order_issue_placements
from helion._compiler.device_ir import RootGraphInfo
from helion.language import memory_ops
from helion.language.matmul_ops import dot


def _region(kind="accumulator", independent_first=False):
    graph = Graph()
    a = _input(graph, "matrix", torch.empty((32, 32), dtype=torch.bfloat16))
    b = _input(graph, "other", torch.empty((32, 32), dtype=torch.bfloat16))
    fp32 = torch.empty((32, 32), dtype=torch.float32)
    independent = None
    if independent_first:
        independent = _call(graph, dot, (a, b, None, torch.float32), fp32)
    first = _call(graph, dot, (a, b, None, torch.float32), fp32)
    if kind == "accumulator":
        left, acc = a, first
    else:
        value = _call(
            graph,
            torch.ops.prims.convert_element_type.default,
            (first, torch.bfloat16),
            a.meta["val"],
        )
        if kind == "nested":
            # Dependency in a nested index/mask-like argument, not the payload.
            value = _call(
                graph, torch.ops.aten.where.self, (value, a, b), a.meta["val"]
            )
        left, acc = value, None
    second = _call(graph, dot, (left, b, acc, torch.float32), fp32)
    graph.output((independent, second) if independent is not None else (second,))
    region = collect_contraction_region(RootGraphInfo(0, graph))
    assert region is not None
    return region


@pytest.mark.parametrize("kind", ("accumulator", "value", "nested"))
def test_exact_use_def_and_precision_edges(kind):
    region = _region(kind)
    selected = ContractionGraph(region)
    (edge,) = selected.dependencies
    first, second = region.contractions
    assert edge.producer is first and edge.consumer is second
    assert edge.kind is (
        DependencyKind.ACCUMULATOR if kind == "accumulator" else DependencyKind.VALUE
    )
    assert bool(edge.path) is (kind != "accumulator")
    assert selected.order() == (first, second)
    assert selected.effects == ((region.live_outs[0], frozenset({second.node})),)


def _placements(region):
    owner, role = object(), object()
    result = []
    for spec in region.contractions:
        for begin, end in ((0, 1), (1, 2)):
            result.append(
                IssuePlacement(
                    ContractionIssue(
                        region,
                        spec,
                        begin,
                        end,
                        16,
                        begin != 0 or spec.accumulator is not None,
                        end == 2,
                    ),
                    owner,
                    role,
                    (0, 32),
                )
            )
    return tuple(result)


def test_unordered_bindings_follow_data_and_k_edges():
    region = _region(independent_first=True)
    graph = ContractionGraph(region)
    placements = _placements(region)
    (expected,) = order_issue_placements(graph, placements)
    (reversed_order,) = order_issue_placements(graph, tuple(reversed(placements)))
    assert expected == reversed_order
    assert tuple(item.issue.spec for item in expected[::2]) == graph.order()
    # Renaming nodes cannot become a semantic role assignment.
    for node in region.graph.nodes:
        node.name = "renamed_" + node.name
    assert order_issue_placements(graph, placements) == (expected,)


@pytest.mark.parametrize(
    "failure", ("missing", "duplicate", "gap", "overlap", "role", "owner")
)
def test_issue_interval_and_owner_rejections(failure):
    region = _region()
    placements = _placements(region)
    graph = ContractionGraph(region)
    first = placements[0]
    if failure == "missing":
        broken = placements[:-1]
    elif failure == "duplicate":
        broken = (*placements, first)
    elif failure == "gap":
        broken = (replace(first, issue=replace(first.issue, begin=1)), *placements[1:])
    elif failure == "overlap":
        broken = (*placements, replace(first, member=(16, 32)))
    elif failure == "role":
        broken = (replace(first, role=object()), *placements[1:])
    else:
        broken = (replace(first, owner=object()), *placements[1:])
    with pytest.raises(chain._UnsupportedChain):
        order_issue_placements(graph, broken)


def test_independent_issuers_are_not_serialized_into_one_role():
    region = _region(independent_first=True)
    graph = ContractionGraph(region)
    placements = _placements(region)
    second_role = object()
    placements = tuple(
        replace(item, role=second_role)
        if item.issue.spec is region.contractions[0]
        else item
        for item in placements
    )
    schedules = order_issue_placements(graph, tuple(reversed(placements)))
    assert len(schedules) == 2
    assert {id(item.role) for item in schedules[0]}.isdisjoint(
        {id(item.role) for item in schedules[1]}
    )
    assert sum(map(len, schedules)) == len(placements)


def test_actual_dependency_change_not_a_name_or_ordinal_hint():
    region = _region(independent_first=True)
    graph = ContractionGraph(region)
    first, producer, consumer = region.contractions
    consumer.node.args = (*consumer.node.args[:2], first.node, torch.float32)
    with pytest.raises(chain._UnsupportedChain, match="dependency graph changed"):
        graph.check()
    # Recollection supplies current original specs, never a snapshot refresh.
    recollected = collect_contraction_region(RootGraphInfo(0, region.graph))
    assert recollected is not None
    current = ContractionGraph(recollected)
    (edge,) = current.dependencies
    assert edge.producer.node is first.node and edge.consumer.node is consumer.node
    assert edge.producer.node is not producer.node
    with pytest.raises(
        chain._UnsupportedChain, match="stale original contraction operands"
    ):
        ContractionGraph(region)


def test_order_validates_deep_revision_once_but_public_predecessors_still_checks():
    region = _region(independent_first=True)
    graph = ContractionGraph(region)
    check = ContractionGraph.check
    seen = []

    def observe(self):
        seen.append(self)
        check(self)

    with patch.object(ContractionGraph, "check", observe):
        graph.order()
        assert seen == [graph]
        graph.predecessors(region.contractions[-1])
        assert seen == [graph, graph]


@pytest.mark.parametrize("argument", ("index", "mask", "payload"))
def test_original_store_nested_effect_dependencies_are_retained_not_rescheduled(
    argument,
):
    region = _region(independent_first=True)
    first, _, last = region.contractions
    value = region.live_ins[0]
    args = {
        "index": (value, ((first.node,),), value, value),
        "mask": (value, (value,), value, {"original_mask": first.node}),
        "payload": (value, (value,), first.node, value),
    }[argument]
    with region.graph.inserting_before(tuple(region.graph.nodes)[-1]):
        store = region.graph.call_function(memory_ops.store, args)
    current = collect_contraction_region(RootGraphInfo(0, region.graph))
    assert current is not None
    graph = ContractionGraph(current)
    assert dict(graph.effects)[store] == frozenset({first.node})
    assert dict(graph.effects)[last.node] == frozenset({last.node})
    saved = store.args
    store.args = (value, (value,), value, value)
    try:
        with pytest.raises(chain._UnsupportedChain, match="dependency graph changed"):
            graph.check()
    finally:
        store.args = saved


def _record_schedule(graph, placements, path):
    schedules = order_issue_placements(graph, tuple(reversed(placements)))
    assert schedules == order_issue_placements(graph, tuple(placements))
    path.write_text(
        json.dumps(
            {
                "dependencies": [
                    (
                        edge.producer.node.name,
                        edge.consumer.node.name,
                        edge.kind.value,
                        sorted(node.name for node in edge.path),
                    )
                    for edge in graph.dependencies
                ],
                "roles": [
                    [
                        (
                            item.issue.spec.node.name,
                            item.member,
                            item.issue.begin,
                            item.issue.end,
                            item.issue.initialized,
                        )
                        for item in role
                    ]
                    for role in schedules
                ],
            },
            indent=2,
        )
    )
    return schedules


def test_actual_dv2_specs_derive_two_issuer_schedules(tmp_path):
    from test.test_cute_prepared_epoch_walk import _code

    from helion._compiler.cute import chunk_recurrence_sm100 as original_native
    from helion._compiler.cute import prepared_epoch_issue
    from helion._compiler.cute.prepared_epoch_issue import DescriptorIssue

    original = prepared_epoch_issue.schedule_native_issues
    seen = []

    def observe(graph, actions):
        assert all(isinstance(action, DescriptorIssue) for action in actions)
        result = original(graph, actions)
        assert original(graph, tuple(reversed(actions))) == result
        epoch = actions[0].owner.plan.prepared_epoch
        assert epoch is not None
        placements = tuple(action.binding.placement for action in actions)
        schedules = _record_schedule(graph, placements, tmp_path / "dv2.json")
        assert len(schedules) == 2
        chain_schedule = next(
            role
            for role in schedules
            if role[0].role == original_native.ROLES.tcgen05_mma
        )
        query_schedule = next(
            role
            for role in schedules
            if role[0].role == original_native.ROLES.super_mma
        )
        assert [p.issue.spec for p in chain_schedule] == [
            epoch.projected.spec,
            epoch.projected.spec,
            epoch.update,
            epoch.update,
        ]
        assert [p.issue.spec for p in query_schedule] == list(epoch.continuation.specs)
        assert {
            role: [
                step.issue.spec for step in steps if isinstance(step, DescriptorIssue)
            ]
            for role, steps in result
        } == {role[0].role: [item.issue.spec for item in role] for role in schedules}
        first = actions[0]
        assert first.binding.complete is not None
        probes = (
            (first.binding.placement, "owner", object()),
            (first.binding.placement, "role", object()),
            (first.binding, "inputs", first.binding.inputs[:1]),
            (first.binding, "complete", None),
            (first.binding.inputs[0].ready.wait, "returns_phase", True),
            (first.binding.inputs[0].ready.wait, "owner", object()),
        )
        for target, name, value in probes:
            saved = getattr(target, name)
            object.__setattr__(target, name, value)
            try:
                with pytest.raises(chain._UnsupportedChain):
                    original(graph, actions)
            finally:
                object.__setattr__(target, name, saved)
        assert original(graph, actions) == result
        seen.append(True)
        return result

    with patch.object(prepared_epoch_issue, "schedule_native_issues", observe):
        source = _code()
    assert seen == [True] and "_helion_epoch_kernel" in source


@pytest.mark.parametrize("mode", ("local", "serial64", "overlap64"))
def test_actual_initialized_root_uses_transformed_not_issued_seed(mode, tmp_path):
    from test.test_cute_prepared_state_body import _selected

    from helion._compiler.cute import prepared_continuation

    original = prepared_continuation.root_issue_schedule
    seen = []

    def observe(body):
        scheduled = original(body)
        assert body is not None and body.continuation is not None
        model = body.continuation
        assert model.transformed is not None
        graph = model.graph
        schedules = _record_schedule(graph, scheduled, tmp_path / (mode + ".json"))
        assert len(schedules) == 1
        assert schedules[0] == scheduled
        assert any(
            edge.kind is DependencyKind.TRANSFORMED for edge in graph.dependencies
        )
        assert scheduled[0].issue.spec is model.first
        assert scheduled[-1].issue.spec is model.second
        seen.append(True)
        return scheduled

    with patch.object(prepared_continuation, "root_issue_schedule", observe):
        source = _selected(mode, torch.bfloat16, 32)
    assert seen and "execute_prepared_continuation" in source
