from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_frontier_materialization import completed as completed
from .test_cute_chained_preparation_actions import _capture
from .test_cute_chained_register_binding import _polynomial_loop
from .test_cute_chained_register_runtime import _inputs
from .test_cute_chained_register_runtime import _island_config
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_reads import CompletedPreparationStage
from helion._compiler.cute.chained_preparation_reads import PreparationReadFrontier
from helion._compiler.cute.chained_preparation_reads import _escapes_preparation
from helion._compiler.cute.chained_preparation_reads import (
    emit_completed_preparation_stage,
)
from helion._compiler.cute.chained_preparation_reads import plan_island_export_leases
from helion._compiler.cute.chained_preparation_reads import resolved_preparation_reads
from helion._compiler.cute.chained_preparation_storage import bind_preparation_storage
from helion._compiler.cute.chained_preparation_storage import (
    plan_accepted_preparation_storage,
)
from helion.language.matmul_ops import dot


def _nominal_copy(accepted, actions):
    """Model an older complete walk without the new successful-stage receipts."""
    candidate = replace(accepted, actions=actions)
    return replace(candidate, _selection=candidate._fields())


def test_actual_export_ends_follow_complete_shared_stage_not_issue(completed):
    _, body, bound, _ = completed
    physical, accepted = bound.physical, bound.physical.accepted
    leases = plan_island_export_leases(accepted)
    assert leases is not None and len(leases) == 3
    assert physical.export_leases == leases
    assert accepted.lines == body
    scan = accepted.pipeline.scan_producer
    assert scan is not None
    for lease in leases:
        region = physical.layout.region(lease.export.name)
        assert region.live_from == lease.original_start
        assert region.live_until == lease.stop < lease.original_stop
        assert lease.stop == max(scan.phases[reader.stop] for reader in lease.readers)
        for reader in lease.readers:
            action = next(
                action for action in accepted.actions if action.proof is reader
            )
            assert reader.matches(accepted, action)
            assert reader.stop == reader.first + 2
            assert (
                accepted.lines[
                    reader.body_first : reader.body_first + len(reader.lines)
                ]
                == reader.lines
            )
            assert lease.export.name in reader.reads
    assert physical.layout.allocated_bytes < 63744


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_unrelated_polynomial_records_completion_but_keeps_escaping_exports(dtype):
    _, _, bound = _capture(
        dtype,
        True,
        kernel=_polynomial_loop,
        args=_inputs("cpu", dtype, 3),
        config=_island_config(3, True),
    )
    assert bound is not None
    accepted = bound.physical.accepted
    stages = tuple(
        action
        for action in accepted.actions
        if isinstance(action.proof, CompletedPreparationStage)
    )
    assert stages and all(action.proof.matches(accepted, action) for action in stages)
    assert plan_island_export_leases(accepted) == ()


def test_nominal_span_without_emission_receipt_cannot_shorten(completed):
    accepted = completed[2].physical.accepted
    nominal = _nominal_copy(
        accepted,
        tuple(
            replace(action, proof=None)
            if isinstance(action.proof, CompletedPreparationStage)
            else action
            for action in accepted.actions
        ),
    )
    assert plan_island_export_leases(nominal) == ()


@pytest.mark.parametrize(
    "field",
    ["reads", "roots", "lines", "inputs", "outputs", "body_first", "stop", "execution"],
)
def test_changed_successful_stage_receipt_rejects_repacking(completed, field):
    accepted = completed[2].physical.accepted
    index, action = next(
        (i, action)
        for i, action in enumerate(accepted.actions)
        if isinstance(action.proof, CompletedPreparationStage)
    )
    receipt = action.proof
    assert isinstance(receipt, CompletedPreparationStage)
    value: Any = {
        "body_first": receipt.body_first + 1,
        "stop": receipt.stop - 1,
        "execution": replace(receipt.execution, threads=receipt.execution.threads + 32),
    }.get(field, ())
    changed = replace(receipt, **{field: value})
    actions = tuple(
        replace(item, proof=changed) if i == index else item
        for i, item in enumerate(accepted.actions)
    )
    modified = _nominal_copy(accepted, actions)
    assert plan_island_export_leases(modified) is None
    assert (
        plan_accepted_preparation_storage(
            accepted.revision.plan, accepted.pipeline, modified, capacity_bytes=1 << 20
        )
        is None
    )


def test_unrecorded_later_reader_preserves_original_lease(completed):
    accepted = completed[2].physical.accepted
    leases = plan_island_export_leases(accepted)
    assert leases is not None and leases
    names = tuple(lease.export.name for lease in leases)
    actions = (*accepted.actions[:-1], replace(accepted.actions[-1], reads=names))
    changed = _nominal_copy(accepted, actions)
    assert plan_island_export_leases(changed) == ()


def test_mutated_sealed_segment_and_derived_lease_are_rejected(completed):
    physical = completed[2].physical
    accepted = physical.accepted
    lease = physical.export_leases[0]
    changed = replace(
        physical,
        export_leases=(
            replace(lease, stop=lease.stop - 1),
            *physical.export_leases[1:],
        ),
    )
    assert (
        bind_preparation_storage(accepted.revision.plan, accepted.pipeline, changed)
        is None
    )
    reader = lease.readers[0]
    lines = list(accepted.lines)
    lines[reader.body_first] = "pass"
    changed_body = replace(accepted, lines=tuple(lines))
    changed_body = replace(changed_body, _selection=changed_body._fields())
    assert plan_island_export_leases(changed_body) is None


def test_direct_escape_to_recurrence_is_not_an_export_release(completed):
    accepted = completed[2].physical.accepted
    lease = completed[2].physical.export_leases[0].export
    assert not _escapes_preparation(accepted, lease.node)
    frame = accepted.pipeline.frame
    changed = replace(
        accepted,
        pipeline=replace(
            accepted.pipeline,
            frame=replace(
                frame,
                cut=replace(frame.cut, recurrence=(*frame.cut.recurrence, lease.node)),
            ),
        ),
    )
    assert _escapes_preparation(changed, lease.node)


def test_failed_shared_stage_never_returns_completion_receipt(completed):
    accepted = completed[2].physical.accepted
    receipt = next(
        action.proof
        for action in accepted.actions
        if isinstance(action.proof, CompletedPreparationStage)
    )
    unused: Any = None
    with (
        patch.object(
            chained_tcgen_stage, "emit_stage", side_effect=RuntimeError("stage failed")
        ) as emitter,
        pytest.raises(RuntimeError, match="stage failed"),
    ):
        emit_completed_preparation_stage(
            unused,
            accepted.revision.plan,
            accepted.pipeline,
            receipt.stage,
            dict(receipt.inputs),
            receipt.execution,
            unused,
            unused,
            first=receipt.first,
            body_first=receipt.body_first,
            use_frontier=accepted.read_frontier,
            published=receipt.inputs,
        )
    assert emitter.call_count == 1


def _nested(a, *, payload):
    return a


def _test_reads(pipeline, nodes, boundaries, *, fragments=frozenset()):
    """Typed unit fixtures for the original graph-cut assertions below.

    Fragment coordinates are an explicit mock of the original binder here;
    real completed KDA/island fixtures above exercise its actual authority.
    No production fallback accepts the former name-only fixture.
    """
    graph = nodes[0].graph
    for node in graph.nodes:
        node.meta["val"] = torch.empty((2, 2), dtype=torch.float32)
    aliases = (
        {}
        if pipeline.operand_retention is None
        else {
            f"chain_retained_operand_{index}": item.owner
            for index, item in enumerate(pipeline.operand_retention.candidates)
        }
    )
    owners = {aliases.get(name, name): node for node, name in boundaries.items()}
    buffers = tuple(
        PreparationBuffer(
            item.name, "cache", owners.get(item.name), torch.float32, (2, 2)
        )
        for item in pipeline.frame.buffers
    )
    candidates = tuple(
        SimpleNamespace(
            node=node,
            owner=aliases[name],
            logical_shape=(2, 2),
            full_shape=(2, 2),
            logical_modes=(0, 1),
            row_offset=0,
            dtype=torch.float32,
            view_path=(),
            publication_event=0,
        )
        for node, name in boundaries.items()
        if name in aliases
    )
    typed: Any = SimpleNamespace(
        frame=SimpleNamespace(
            buffers=buffers,
            cut=SimpleNamespace(region=SimpleNamespace(nodes=tuple(graph.nodes))),
            layout=SimpleNamespace(region=lambda name: SimpleNamespace(live_from=0)),
        ),
        operand_retention=SimpleNamespace(candidates=candidates)
        if candidates
        else None,
    )
    bound: Any = (
        None
        if not fragments
        else SimpleNamespace(
            exports=tuple(SimpleNamespace(node=node) for node in fragments),
            plan=None,
            execution=None,
            matches=lambda *args: True,
            coordinates=lambda node, prefix: ("row", "column"),
        )
    )
    return resolved_preparation_reads(
        typed,
        nodes,
        boundaries,
        use_frontier=PreparationReadFrontier(typed, dict.fromkeys(graph.nodes, (2, 2))),
        published=tuple(boundaries.items()),
        fragments=bound,
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fragment", [False, True])
def test_reads_stop_at_original_cached_and_fragment_values(cached, fragment):
    graph = Graph()
    raw = graph.placeholder("raw")
    cached_value = graph.call_function(_nested, (raw,), {"payload": None})
    register_value = graph.call_function(dot, (cached_value, cached_value))
    side = graph.placeholder("side")
    operand = graph.call_function(
        _nested, (register_value,), {"payload": {"side": [side]}}
    )
    pipeline: Any = SimpleNamespace(
        frame=SimpleNamespace(
            buffers=tuple(
                SimpleNamespace(name=name)
                for name in ("raw_owner", "cache_owner", "side_owner")
            )
        ),
        operand_retention=SimpleNamespace(
            candidates=(SimpleNamespace(owner="side_owner"),)
        ),
    )
    boundaries = {raw: "raw_owner", side: "chain_retained_operand_0"}
    if cached:
        boundaries[cached_value] = "cache_owner"
    expected = {"side_owner"}
    if not fragment:
        expected.add("cache_owner" if cached else "raw_owner")
    assert (
        set(
            _test_reads(
                pipeline,
                (operand,),
                boundaries,
                fragments=frozenset((register_value,)) if fragment else frozenset(),
            )
        )
        == expected
    )
    # A second live use is not killed by the register cut on the first branch.
    assert "raw_owner" in _test_reads(
        pipeline, (operand, raw), boundaries, fragments=frozenset((register_value,))
    )


def test_fragment_precedence_matches_expression_and_does_not_hide_other_reads():
    graph = Graph()
    left, right = graph.placeholder("left"), graph.placeholder("right")
    operand = graph.call_function(_nested, (left,), {"payload": [right]})
    pipeline: Any = SimpleNamespace(
        frame=SimpleNamespace(buffers=(SimpleNamespace(name="right_owner"),)),
        operand_retention=None,
    )
    boundaries = {left: "not_a_shared_owner", right: "right_owner"}
    assert _test_reads(
        pipeline, (operand,), boundaries, fragments=frozenset((left,))
    ) == ("right_owner",)
    with pytest.raises(ValueError, match="unowned"):
        _test_reads(pipeline, (operand,), boundaries)
    with pytest.raises(ValueError, match="unowned"):
        _test_reads(
            pipeline,
            (operand,),
            {left: "not_a_shared_owner", right: "also_not_owned"},
            fragments=frozenset((left,)),
        )


def test_nested_kwargs_and_explicit_accumulator_aliases_are_read():
    graph = Graph()
    left, acc = graph.placeholder("left"), graph.placeholder("acc")
    node = graph.call_function(_nested, (left,), {"payload": {"acc": [acc]}})
    seeded_dot = graph.call_function(dot, (left, left), {"acc": acc})
    pipeline: Any = SimpleNamespace(
        frame=SimpleNamespace(
            buffers=(
                SimpleNamespace(name="left_owner"),
                SimpleNamespace(name="acc_owner"),
            )
        ),
        operand_retention=SimpleNamespace(
            candidates=(SimpleNamespace(owner="acc_owner"),)
        ),
    )
    boundaries = {left: "left_owner", acc: "chain_retained_operand_0"}
    assert _test_reads(pipeline, (node,), boundaries) == (
        "acc_owner",
        "left_owner",
    )
    assert _test_reads(pipeline, tuple(seeded_dot.all_input_nodes), boundaries) == (
        "acc_owner",
        "left_owner",
    )
    with pytest.raises(ValueError, match="unowned"):
        _test_reads(pipeline, (node,), {**boundaries, acc: "unknown_alias"})
