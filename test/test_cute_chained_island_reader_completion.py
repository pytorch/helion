from __future__ import annotations

import copy
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from .test_cute_chained_frontier_groups import _frame
from .test_cute_chained_island_publication import _capture
from helion import exc
from helion._compiler.cute import chained_frontier_groups as groups
from helion._compiler.cute import chained_island_publication as publication
from helion._compiler.cute import chained_prepared_values as values
from helion._compiler.cute import chained_warp_stage as warp
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _ancestors
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_preparation_reads import CompletedPreparationStage
from helion._compiler.cute.chained_preparation_reads import (
    PreparationFrontierCompletion,
)
from helion._compiler.cute.chained_preparation_reads import PreparationWarpCompletion
from helion._compiler.cute.chained_vector_stage import VectorStaging


@pytest.mark.parametrize("site", ["consumer", "later_warp", "frontier"])
@pytest.mark.parametrize("mutation", ["payload", "index", "boundary"])
def test_original_return_cannot_be_recertified(site, mutation):
    original_record = publication.IslandConsumerPublication.record
    original_warp = warp.emit_prepared_warp_stage
    original_frontier = values.emit_prepared_value
    observed, changed = [], []

    def record(self, *args, **kwargs):
        original_record(self, *args, **kwargs)
        observed.append(self)

    def corrupt(lines, boundaries):
        if mutation == "boundary":
            node = next(iter(boundaries))
            boundaries[node] = "review_unpublished_input"
        else:
            token = " = " if mutation == "payload" else "["
            replacement = (
                " = review_unapproved_value + " if mutation == "payload" else "[1 + "
            )
            index = next(i for i, line in enumerate(lines) if token in line)
            lines[index] = lines[index].replace(token, replacement, 1)
        changed.append(site)

    def emit_warp(cg, plan, boundaries, prepared, input_prefix):
        lines = original_warp(cg, plan, boundaries, prepared, input_prefix)
        if observed and not changed and site != "frontier":
            item = observed[0].candidate
            result = prepared.result
            assert isinstance(result, warp.WarpMemberResult)
            nodes = tuple(plan.dots[stage] for stage, _, _ in result.members)
            selected = (
                site == "consumer"
                and plan.dots[item.consumer] in nodes
                or site == "later_warp"
                and plan.dots[item.consumer] not in nodes
                and any(item.operand in _ancestors(node) for node in nodes)
            )
            if selected:
                corrupt(lines, boundaries)
        return lines

    def emit_frontier(cg, plan, buffer, boundaries, execution, **kwargs):
        lines = original_frontier(cg, plan, buffer, boundaries, execution, **kwargs)
        if (
            site == "frontier"
            and observed
            and not changed
            and observed[0].candidate.operand in _ancestors(buffer.node)
        ):
            corrupt(lines, boundaries)
        return lines

    with (
        patch.object(publication.IslandConsumerPublication, "record", record),
        patch.object(warp, "emit_prepared_warp_stage", emit_warp),
        patch.object(values, "emit_prepared_value", emit_frontier),
        pytest.raises((exc.BackendUnsupported, exc.InternalError, _UnsupportedChain)),
    ):
        _capture(True)
    assert changed == [site]


@pytest.mark.parametrize("kind", ["warp", "frontier"])
@pytest.mark.parametrize("mutation", ["drop", "copy", "repeat"])
def test_original_completion_slot_cannot_be_dropped_replaced_or_repeated(
    kind, mutation
):
    cls = PreparationWarpCompletion if kind == "warp" else PreparationFrontierCompletion
    original = cls.record
    changed = []

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not changed:
            changed.append(mutation)
            if mutation == "drop":
                self.publication = None
            elif mutation == "copy":
                self.publication = copy.copy(self.publication)
            else:
                original(self, *args, **kwargs)

    with (
        patch.object(cls, "record", record),
        pytest.raises((exc.BackendUnsupported, exc.InternalError, _UnsupportedChain)),
    ):
        _capture(True)
    assert changed == [mutation]


def test_actual_readers_keep_original_warp_and_frontier_completions():
    _, accepted, bound = _capture(True)
    assert bound is not None
    assert len(accepted.island_reads) == 3
    kinds = []
    for reader in accepted.island_reads:
        assert reader.matches(accepted)
        proof = reader.action.proof
        if isinstance(proof, CompletedPreparationStage):
            token = proof.warp_completion
            assert token is not None
            assert token is token.prepared.completion._state.publication
            assert token.prefix == proof.lines
            assert token.boundaries == proof.outputs
            assert proof.original_warp_matches()
            kinds.append("warp")
        else:
            token = reader.frontier
            assert token is not None and token.slot.consumed
            assert token is token.slot.publication is token.slot._publication
            assert token.lines == reader.lines
            kinds.append("frontier")
    assert kinds == ["warp", "warp", "frontier"]


@pytest.fixture(scope="module")
def completed_readers():
    _, accepted, _ = _capture(True)
    return accepted, tuple(
        reader
        for reader in accepted.island_reads
        if isinstance(reader.action.proof, CompletedPreparationStage)
    )


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "mutation", ["axes", "alias_value", "alias_drop", "alias_order"]
)
def test_completed_warp_preserves_original_revision(completed_readers, index, mutation):
    accepted, readers = completed_readers
    reader = readers[index]
    proof = reader.action.proof
    plan = proof.plan
    axes, aliases = plan.axes, tuple(plan.tensor_aliases.items())
    assert proof.original_warp_matches() and reader.matches(accepted)
    try:
        if mutation == "axes":
            first, *rest = axes
            object.__setattr__(plan, "axes", ((first[0] + 1, *first[1:]), *rest))
        else:
            assert proof.warp_aliases
            key = proof.warp_aliases[0][0]
            if mutation == "alias_value":
                plan.tensor_aliases[key] = "review_unpublished_native_owner"
            elif mutation == "alias_drop":
                plan.tensor_aliases.pop(key)
            else:
                value = plan.tensor_aliases.pop(key)
                plan.tensor_aliases["review_later_alias"] = "review_later_owner"
                plan.tensor_aliases[key] = value
        assert not proof.original_warp_matches()
        assert not reader.matches(accepted)
    finally:
        object.__setattr__(plan, "axes", axes)
        plan.tensor_aliases.clear()
        plan.tensor_aliases.update(aliases)
    assert proof.original_warp_matches() and reader.matches(accepted)


@pytest.mark.parametrize("index", [0, 1])
def test_completed_warp_permits_only_appended_aliases(completed_readers, index):
    _, readers = completed_readers
    proof = readers[index].action.proof
    aliases = tuple(proof.plan.tensor_aliases.items())
    try:
        proof.plan.tensor_aliases["review_later_alias"] = "review_later_owner"
        assert proof.original_warp_matches()
    finally:
        proof.plan.tensor_aliases.clear()
        proof.plan.tensor_aliases.update(aliases)


@pytest.mark.parametrize("path", ["scalar", "vector", "group"])
def test_each_original_frontier_return_captures_its_complete_segment(path):
    """Isolate slot plumbing; real typed math is covered by the original suites."""
    plan, frame = _frame()
    cg = Mock()
    cg.device_function.config.config = {}
    execution = ChainedExecution(128, thread="original_thread", sync="original_join()")
    boundaries = {plan.dots[0]: "original_input"}
    initial = tuple(boundaries.items())
    slot = PreparationFrontierCompletion()
    expression = Mock()
    expression.lines = ["original_point = original_input[0]"]
    expression.value.return_value = "original_point"

    def emit(completion):
        if path == "group":
            result = groups.emit_frontier_group(
                cg,
                plan,
                frame,
                2,
                boundaries,
                execution,
                VectorStaging(True, group_enabled=True),
                BoundedProducerUnroll(1),
                scalar_targets=set(),
                completion=completion,
            )
            assert result is not None
            return result[0]
        return values.emit_prepared_value(
            cg,
            plan,
            frame.buffers[0],
            boundaries,
            execution,
            vector=VectorStaging(path == "vector"),
            completion=completion,
        )

    with (
        patch.object(values, "_shape", return_value=frame.buffers[0].shape),
        patch.object(values, "_Expression", return_value=expression),
        patch.object(values, "storage_dtype", return_value="cutlass.BFloat16"),
        patch.object(
            values, "emit_vector_expression", return_value=["original_vector_copy()"]
        ),
        patch.object(
            groups, "emit_vector_group", return_value=["original_group_copy()"]
        ),
    ):
        original = emit(None)
        actual = emit(slot)
    assert actual == original
    assert actual[-1] == execution.sync and actual.count(execution.sync) == 1
    token = slot.consume(
        cg, plan, execution, initial, tuple(boundaries.items()), tuple(actual)
    )
    assert token.lines == tuple(original)
    assert token.buffers == (frame.buffers if path == "group" else frame.buffers[:1])
    assert token.matches(plan, initial, initial, tuple(actual), consumed=True)
    with pytest.raises(_UnsupportedChain, match="changed after return"):
        slot.consume(cg, plan, execution, initial, initial, tuple(actual))
