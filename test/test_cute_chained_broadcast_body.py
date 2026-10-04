from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from .test_cute_chained_broadcast_retention import _args
from .test_cute_chained_broadcast_retention import _broadcast_loop
from .test_cute_chained_broadcast_retention import _capture
from .test_cute_chained_broadcast_retention import _config
from .test_cute_chained_preparation_leaves import _source
from helion import exc
from helion._compiler.cute import chained_scan_producer_emission as scan_emitter
from helion._compiler.cute import chained_vector_group as group_emitter
from helion._compiler.cute import chained_vector_stage as vector_emitter
from helion._compiler.cute.chained_broadcast_retention import BroadcastRetentionAttempt
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_vector_stage import VectorStaging


def _emit(mode: str) -> str:
    config = _config(True, scan=mode == "scan", batching=mode == "scan")
    if mode != "scan":
        config.config.update(
            cute_chained_vector_group=mode == "group",
            cute_chained_leaf_pipeline="legacy",
            cute_chained_leaf_count=1,
        )
    return _source(_broadcast_loop, _args(), config)


def _zero_assignment(lines):
    changed = list(lines)
    for index, block in enumerate(changed):
        statements = block.splitlines()
        for row, line in enumerate(statements):
            if line.strip().startswith("chain_broadcast") and " = " in line:
                statements[row] = line.split(" = ", 1)[0] + " = cutlass.Float32(0)"
                assert statements[row] != line
                changed[index] = "\n".join(statements)
                return changed
    return None


@pytest.mark.parametrize("mode", ["scan", "single", "group"])
def test_original_completion_cannot_recertify_mutated_producer(mode):
    original = BroadcastRetentionAttempt.complete
    seen = []

    def complete(self, emission, boundaries, execution, lines):
        result = original(self, emission, boundaries, execution, lines)
        if not seen:
            changed = _zero_assignment(lines)
            assert changed is not None
            receipt = self.result()[-1]
            assert receipt.lines == tuple(lines)
            lines[:] = changed
            assert receipt.lines != tuple(lines)
            seen.append(receipt)
        return result

    with (
        patch.object(BroadcastRetentionAttempt, "complete", complete),
        pytest.raises((ValueError, exc.BackendUnsupported, exc.InternalError)) as error,
    ):
        _emit(mode)
    assert len(seen) == 1, str(error.value)


@pytest.mark.parametrize("mode", ["scan", "single", "group"])
@pytest.mark.parametrize("change", ["replace", "duplicate", "drop"])
def test_replacement_return_keeps_original_captured_producer(mode, change):
    owner, name = {
        "scan": (scan_emitter, "emit_scan_producer"),
        "single": (vector_emitter, "emit_vector_expression"),
        "group": (group_emitter, "emit_vector_group"),
    }[mode]
    original = getattr(owner, name)
    seen = []

    def emit(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is None or seen:
            return result
        lines = result.lines if mode == "scan" else result
        changed = _zero_assignment(lines)
        if changed is None:
            return result
        if change == "duplicate":
            changed = [*lines, *lines]
        elif change == "drop":
            changed = list(lines[:-1])
        seen.append(tuple(lines))
        return replace(result, lines=tuple(changed)) if mode == "scan" else changed

    with (
        patch.object(owner, name, emit),
        pytest.raises((ValueError, exc.BackendUnsupported, exc.InternalError)),
    ):
        _emit(mode)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "change", ["offset", "indent", "drop", "duplicate", "copied", "foreign", "receipt"]
)
def test_original_structural_placement_is_not_optional(change):
    original_place = BroadcastRetentionAttempt.place
    original_enclose = BroadcastRetentionAttempt.enclose
    seen = []

    def place(self, *args, **kwargs):
        result = original_place(self, *args, **kwargs)
        if result is None or seen or change == "duplicate":
            return result
        seen.append(result)
        if change == "drop":
            return None
        if change == "copied":
            return replace(result, body=replace(result.body))
        if change == "foreign":
            receipt = result.body.receipts[0]
            foreign = BroadcastRetentionAttempt(self.first, self.execution)
            foreign.complete(
                receipt.emission,
                dict(receipt.boundaries),
                receipt.execution,
                receipt.lines,
            )
            return replace(result, body=foreign._bodies[0])
        if change == "receipt":
            self.receipts = (replace(self.receipts[0]), *self.receipts[1:])
            return result
        return replace(
            result,
            **{
                change if change == "indent" else "first": getattr(
                    result, change if change == "indent" else "first"
                )
                + 1
            },
        )

    def enclose(self, first, lines, placements):
        if change == "duplicate" and not seen and any(placements):
            seen.append(tuple(placements))
            placements = (*placements, *placements)
        return original_enclose(self, first, lines, placements)

    with (
        patch.object(BroadcastRetentionAttempt, "place", place),
        patch.object(BroadcastRetentionAttempt, "enclose", enclose),
        pytest.raises((ValueError, exc.BackendUnsupported, exc.InternalError)),
    ):
        _emit("scan")
    assert len(seen) == 1


@pytest.mark.parametrize(
    "change", ["position", "action", "accepted", "scheduled", "child", "copy"]
)
def test_late_body_and_schedule_mutations_reject_before_consumption(change):
    checked = []

    def check(bound):
        accepted = bound.physical.accepted
        action = next(item for item in accepted.actions if item.broadcasts)
        body = action.broadcast_body
        assert body is not None
        schedule = bound.physical.scheduled_body
        assert schedule is not None
        assert body.matches(accepted, action)
        if change == "position":
            target, field, value = body, "body_first", body.body_first + 1
        elif change == "action":
            target, field, value = action, "broadcast_body", None
        elif change == "accepted":
            target, field, value = accepted, "lines", (*accepted.lines, "unproved()")
        elif change == "scheduled":
            target, field, value = schedule, "lines", (*schedule.lines, "unproved()")
        elif change == "copy":
            target, field, value = action, "broadcast_body", replace(body)
        else:
            target, field, value = body.body.children[0], "first", -1
        previous = getattr(target, field)
        try:
            object.__setattr__(target, field, value)
            assert not bound.matches(accepted.revision.plan, accepted.pipeline)
            with pytest.raises(ValueError, match="unconsumed final allocation"):
                bound.consume_body(
                    bound._state.finalized,
                    VectorStaging(
                        accepted.vector_state[0], group_enabled=accepted.vector_state[2]
                    ),
                    BoundedProducerUnroll(accepted.unroll_state[0]),
                )
            assert not bound._state.consumed
            checked.append(change)
        finally:
            object.__setattr__(target, field, previous)
        assert bound.matches(accepted.revision.plan, accepted.pipeline)

    _capture(batching=True, check=check)
    assert checked == [change]
