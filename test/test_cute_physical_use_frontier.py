from __future__ import annotations

from dataclasses import replace
from typing import TypedDict
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _input
from .test_cute_chained_preparation_frame import _safe
from .test_cute_chained_preparation_frame import _synthetic
from .test_cute_chained_recurrence_workspace import _safe as _safe_recurrence
from .test_cute_chained_warp_bridge import _code as _warp_code
from .test_cute_prepared_epoch_walk import _code as _epoch_code
from .test_cute_prepared_state_body import _selected as _root_state_code
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import prepared_state_planner as state_planner
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.physical_use_frontier import InvalidPhysicalUse
from helion._compiler.cute.physical_use_frontier import PhysicalPublication
from helion._compiler.cute.physical_use_frontier import PhysicalReadPoint
from helion._compiler.cute.physical_use_frontier import PhysicalUseFrontier


class _ResolveOptions(TypedDict):
    external: frozenset[Node]
    required: frozenset[Node]
    traversable: frozenset[Node]


def _fixture():
    graph = Graph()
    raw = _input(graph, "raw", (2, 2), torch.float32)
    index = _input(graph, "index", (2, 2), torch.int64)
    mask = _input(graph, "mask", (2, 2), torch.bool)
    destination = _input(graph, "destination", (2, 2), torch.float32)
    cached = _call(graph, torch.neg, (raw,), (2, 2), torch.float32)
    result = _call(graph, torch.add, (cached, raw), (2, 2), torch.float32)
    # No execution: every selector/value is an actual FX argument edge.
    store = graph.call_function(
        torch.add,
        (destination, result),
        {"selectors": {"index": [index], "mask": (mask,)}},
    )
    graph.output(store)
    frontier = PhysicalUseFrontier(tuple(graph.nodes))
    scope = object()
    publications = {}
    for node in (raw, index, mask, destination, cached):
        owner = object()
        frontier.register_owner(owner, scope, 0)
        publications[node] = PhysicalPublication(
            node, owner, node.meta["val"].dtype, (2, 2)
        )
    options: _ResolveOptions = {
        "external": frozenset(),
        "required": frozenset(publications),
        "traversable": frozenset(graph.nodes),
    }
    return graph, frontier, scope, publications, options, result, store


def test_publication_stops_value_ancestry_but_keeps_other_real_reads() -> None:
    _, frontier, _, publications, options, result, store = _fixture()
    cached = result.args[0]
    raw = result.args[1]
    assert isinstance(cached, Node) and isinstance(raw, Node)
    assert frontier.resolve((cached,), publications, **options) == (
        publications[cached],
    )
    assert {p.node for p in frontier.resolve((store,), publications, **options)} == set(
        publications
    )
    assert frontier.resolve((cached,), publications, expand=cached, **options) == (
        publications[raw],
    )
    frontier.check()


@pytest.mark.parametrize("change", ["missing", "dtype", "shape", "foreign_owner"])
def test_malformed_full_publication_rejected(change: str) -> None:
    _, frontier, _, publications, options, result, _ = _fixture()
    node = result.args[0]
    assert isinstance(node, Node)
    original = publications[node]
    if change == "missing":
        publications.pop(node)
    elif change == "dtype":
        publications[node] = replace(original, dtype=torch.bfloat16)
    elif change == "shape":
        publications[node] = replace(original, shape=(1, 4))
    else:
        publications[node] = replace(original, owner=object())
    with pytest.raises(InvalidPhysicalUse):
        frontier.resolve((result,), publications, **options)


@pytest.mark.parametrize("field", ["owner", "dtype", "shape"])
def test_same_object_publication_mutation_rejected(field: str) -> None:
    _, frontier, _, publications, options, result, _ = _fixture()
    node = result.args[0]
    assert isinstance(node, Node)
    publication = publications[node]
    frontier.resolve((result,), publications, **options)
    original = getattr(publication, field)
    replacement = {"owner": object(), "dtype": torch.float16, "shape": (1, 4)}[field]
    try:
        object.__setattr__(publication, field, replacement)
        with pytest.raises(InvalidPhysicalUse, match="changed"):
            frontier.resolve((result,), publications, **options)
        with pytest.raises(InvalidPhysicalUse, match="changed"):
            frontier.check()
    finally:
        object.__setattr__(publication, field, original)


@pytest.mark.parametrize("change", ["args", "dtype", "new_user"])
def test_scoped_graph_mutation_rejected(change: str) -> None:
    graph, frontier, _, _, _, result, _ = _fixture()
    if change == "args":
        result.args = tuple(reversed(result.args))
    elif change == "dtype":
        result.meta["val"] = torch.empty((2, 2), dtype=torch.float16)
    else:
        graph.call_function(torch.neg, (result,))
    with pytest.raises(InvalidPhysicalUse, match="graph changed"):
        frontier.check()


def test_original_lifetime_is_consumed_and_scopes_are_not_ordinally_merged() -> None:
    _, frontier, scope, publications, _, _, _ = _fixture()
    owner = next(iter(publications.values())).owner
    assert frontier.through(owner, scope) == 1
    frontier.read(owner, PhysicalReadPoint(scope, 2, 3))
    frontier.read(owner, PhysicalReadPoint(scope, 7, 9))
    frontier.read(owner, PhysicalReadPoint(scope, 3, 4))
    assert frontier.through(owner, scope) == 9
    assert frontier.first_read(owner, scope) == 2
    with pytest.raises(InvalidPhysicalUse, match="ordering"):
        frontier.read(owner, PhysicalReadPoint(object(), 100, 101))
    assert frontier.through(owner, scope) == 9
    with pytest.raises(InvalidPhysicalUse, match="scope"):
        frontier.through(owner, object())


@pytest.mark.parametrize("event,through", [(0, 1), (1, 1), (2, 1)])
def test_read_cannot_precede_publication_or_its_original_end(
    event: int,
    through: int,
) -> None:
    _, frontier, scope, publications, _, _, _ = _fixture()
    owner = next(iter(publications.values())).owner
    with pytest.raises(InvalidPhysicalUse, match="ordering"):
        frontier.read(owner, PhysicalReadPoint(scope, event, through))


def test_materialization_change_invalidates_only_local_resolution_cache() -> None:
    _, frontier, _, publications, options, result, _ = _fixture()
    cached, raw = result.args
    assert isinstance(cached, Node) and isinstance(raw, Node)
    first = dict(publications)
    first.pop(cached)
    uncached: _ResolveOptions = {**options, "required": options["required"] - {cached}}
    assert frontier.resolve((cached,), first, **uncached) == (publications[raw],)
    assert frontier.resolve((cached,), publications, **options) == (
        publications[cached],
    )
    frontier.check()


@pytest.mark.parametrize("kind", ["dot", "collective", "cache"])
def test_frame_real_requests_consume_common_reads_before_packing(kind: str) -> None:
    plan, cut, shapes = _synthetic(kind)
    used = []
    read = PhysicalUseFrontier.read
    check = PhysicalUseFrontier.check

    def observe(self, owner, point):
        read(self, owner, point)
        used.append((owner.name, point.event, point.through))

    checked = []

    def finish(self):
        check(self)
        checked.append(True)

    with (
        patch.object(PhysicalUseFrontier, "read", observe),
        patch.object(PhysicalUseFrontier, "check", finish),
    ):
        frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None and used and checked == [True]
    _safe(frame)
    for name, _, end in used:
        assert frame.layout.region(name).live_until >= end
    for region in frame.layout.regions:
        assert region.live_until == max(
            [
                region.live_from + 1,
                *(end for name, _, end in used if name == region.name),
            ]
        )


def test_missing_frontier_ordering_rejects_frame_before_packing() -> None:
    plan, cut, shapes = _synthetic()
    with patch.object(
        PhysicalUseFrontier,
        "read",
        side_effect=InvalidPhysicalUse("missing actual cut"),
    ):
        assert plan_preparation_frame(plan, cut, shapes) is None


def test_original_live_in_is_present_before_zero_without_ordinary_negative_birth() -> (
    None
):
    _, frontier, scope, _, _, _, _ = _fixture()
    owner = object()
    with pytest.raises(InvalidPhysicalUse, match="original owner"):
        frontier.register_owner(owner, scope, -1)
    frontier.register_live_in(owner, scope)
    assert frontier.through(owner, scope) == 0
    frontier.read(owner, PhysicalReadPoint(scope, 0, 1))
    assert frontier.through(owner, scope) == 1
    with pytest.raises(InvalidPhysicalUse, match="duplicated"):
        frontier.register_live_in(owner, scope)


def test_resident_seed_keeps_order_but_does_not_extend_shared_lease() -> None:
    _, frontier, scope, publications, _, _, _ = _fixture()
    owner = next(iter(publications.values())).owner
    frontier.require_published(owner, PhysicalReadPoint(scope, 5, 6))
    assert frontier.through(owner, scope) == 1
    with pytest.raises(InvalidPhysicalUse, match="ordering"):
        frontier.require_published(owner, PhysicalReadPoint(scope, 0, 1))


@pytest.mark.parametrize("kind", ["dot", "collective", "cache"])
def test_recurrence_requests_consume_original_live_in_and_complete_reads(
    kind: str,
) -> None:
    plan, cut, shapes = _synthetic(kind)
    reads = []
    live_ins = []
    original = PhysicalUseFrontier.read
    register = PhysicalUseFrontier.register_live_in

    def read(self, owner, point):
        original(self, owner, point)
        reads.append((owner, point))

    def live_in(self, owner, scope):
        register(self, owner, scope)
        live_ins.append((owner, self.through(owner, scope)))

    with (
        patch.object(PhysicalUseFrontier, "read", read),
        patch.object(PhysicalUseFrontier, "register_live_in", live_in),
    ):
        workspace = plan_recurrence_workspace(plan, cut, shapes)
    assert workspace is not None and reads
    _safe_recurrence(workspace)
    assert live_ins == [(carry.input, 0) for carry in cut.carries]
    names = dict(workspace.bindings)
    for owner, point in reads:
        region = workspace.layout.region(names[owner])
        assert region.live_from < point.event < point.through <= region.live_until
    assert workspace.layout.regions[0].live_from == -1


def test_complete_exclusive_frontier_rejects_another_value_or_mask_user() -> None:
    _, frontier, _, _, _, result, store = _fixture()
    cached, raw = result.args
    assert isinstance(cached, Node) and isinstance(raw, Node)
    assert frontier.exclusive_use(cached, cached, result)
    # Raw is also consumed directly, outside raw->negate->consumer.
    assert not frontier.exclusive_use(raw, cached, result)
    assert not frontier.exclusive_use(cached, cached, store)
    frontier.check()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scan", [False, True])
def test_actual_warp_bridge_selection_consumes_common_exclusive_use(
    dtype, scan
) -> None:
    selected = []
    original = PhysicalUseFrontier.exclusive_use

    def observe(self, source, operand, consumer):
        result = original(self, source, operand, consumer)
        selected.append((source, operand, consumer, result))
        return result

    with patch.object(PhysicalUseFrontier, "exclusive_use", observe):
        source = _warp_code(dtype=dtype, scan=scan)
    assert any(item[3] for item in selected)
    assert "chain_0_c_ptr =" not in source
    assert "chain_1_a_bridge_values" in source
    with patch.object(PhysicalUseFrontier, "exclusive_use", return_value=False):
        materialized = _warp_code(dtype=dtype, scan=scan)
    assert "chain_0_c_ptr =" in materialized
    assert "chain_1_a_bridge_values" not in materialized


@pytest.mark.parametrize("client", ["descriptor", "fragment_bf16", "fragment_f16"])
def test_real_state_phases_consume_original_view_use_frontiers(client: str) -> None:
    build = state_planner.plan_state_transfers
    calls = []

    def observe(requests, cuts):
        # Equivalent full native views are legal; no new factory-identity rule.
        views = tuple(
            replace(request, source_view=replace(request.source_view))
            for request in reversed(requests)
        )
        original = build(requests, cuts)
        observed = []
        first = PhysicalUseFrontier.first_read

        def record(self, owner, scope):
            result = first(self, owner, scope)
            assert isinstance(owner, state_planner.StateView)
            observed.append((owner, result, self.through(owner, scope) - 1))
            return result

        with patch.object(PhysicalUseFrontier, "first_read", record):
            replay = build(views, cuts)
        assert observed and len(observed) == len(replay.residency)
        assert [
            [(effect.kind, effect.binding) for effect in phase]
            for phase in replay.phases
        ] == [
            [(effect.kind, effect.binding) for effect in phase]
            for phase in original.phases
        ]
        for residency, (view, start, end) in zip(
            replay.residency, observed, strict=True
        ):
            assert residency.view.facts() == view.facts()
            assert (residency.first_interval, residency.last_interval) == (start, end)
            assert any(
                effect.kind == "read" and effect.binding.source is residency.source
                for effect in replay.phases[start]
            )
        replay.check()
        calls.append(len(observed))
        return original

    with patch.object(state_planner, "plan_state_transfers", observe):
        if client == "descriptor":
            source = _epoch_code()
        else:
            source = _root_state_code(
                "overlap64",
                torch.bfloat16 if client == "fragment_bf16" else torch.float16,
                32,
            )
    assert calls and "prepared_tcgen_edge" in source


@pytest.mark.parametrize("client", ["descriptor", "fragment_bf16", "fragment_f16"])
@pytest.mark.parametrize("method", ["register_live_in", "ancestors", "read", "check"])
def test_actual_state_frontier_rejection_preserves_original_boundary(
    client: str, method: str
) -> None:
    build = state_planner.plan_state_transfers
    calls = []

    def observe(requests, cuts):
        original = build(requests, cuts)
        error = InvalidPhysicalUse("unproven original physical use")
        with (
            patch.object(PhysicalUseFrontier, method, side_effect=error),
            pytest.raises(chain._UnsupportedChain, match=str(error)) as rejected,
        ):
            build(requests, cuts)
        assert rejected.value.__cause__ is error
        original.check()
        calls.append(len(requests))
        return original

    with patch.object(state_planner, "plan_state_transfers", observe):
        if client == "descriptor":
            source = _epoch_code()
        else:
            source = _root_state_code(
                "overlap64",
                torch.bfloat16 if client == "fragment_bf16" else torch.float16,
                32,
            )
    assert calls and "prepared_tcgen_edge" in source
