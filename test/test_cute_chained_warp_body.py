from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_warp_bridge import _code as _bridge_code
from .test_cute_chained_warp_stage import _code as _singleton_code
from helion import exc
from helion._compiler.cute import chained_body_program as body_program
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_warp_stage as roots
from helion._compiler.cute import chained_warp_bridge as bridge
from helion._compiler.cute import chained_warp_stage as shared
from helion._compiler.cute import native_matmul_metadata


def _code(route, dtype=torch.bfloat16):
    return (
        _singleton_code(dtype=dtype)
        if route == "singleton"
        else _bridge_code(dtype=dtype, scan=route == "scan")
    )


@pytest.mark.parametrize("route", ("singleton", "bridge", "scan"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_complete_original_warp_schedule_uses_common_body(route, dtype):
    original = body_program.emit_body_program
    accepted = []
    joined = []
    validate = bridge.RootWarpBridgeSequence.validate

    def observe(*args, **kwargs):
        body = kwargs.get("root_warp")
        assert isinstance(body, body_program.RootWarpBody)
        owner, prefix = body.owner, body.prefix
        actions = body.program.actions
        if route == "singleton":
            assert len(actions) == 1 and actions[0] is owner
        else:
            assert len(actions) == 2
            for stage, action in enumerate(actions):
                assert isinstance(action, bridge.RootWarpBridgeAction)
                assert action.stage == stage and action.sequence is owner
        lines = original(*args, **kwargs)
        assert lines is prefix is body.prefix
        assert body.consumed and body.pending is None
        assert body.cursor == len(actions)
        assert body.publication is not None
        assert body.publication.prefix == tuple(lines)
        assert body.publication.matches(args[0], args[1], body.boundaries, lines)
        if isinstance(owner, bridge.RootWarpBridgeSequence):
            fragment = owner._state.fragment
            assert fragment is not None and fragment._consumed
            first = fragment.prepared.completion._state.publication
            assert first is not None
            assert body.plan.dots[0] not in dict(first.boundaries)
            assert body.plan.dots[0] not in body.boundaries
            assert "chain_0_c_ptr =" not in "\n".join(lines)
        accepted.append(body)
        return lines

    def validate_join(self, cg, plan, boundaries, staged, prefix):
        assert accepted and accepted[-1].consumed
        assert self._body.body is accepted[-1]
        assert prefix is accepted[-1].prefix
        validate(self, cg, plan, boundaries, staged, prefix)
        joined.append(True)

    with (
        patch.object(body_program, "emit_body_program", observe),
        patch.object(bridge.RootWarpBridgeSequence, "validate", validate_join),
    ):
        source = _code(route, dtype)
    assert "chain_0_mma" in source
    assert len(accepted) == 1
    assert len(joined) == (route != "singleton")


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize(
    "change", ("drop", "duplicate", "replace", "clone_body", "lost_binding")
)
def test_program_mutation_rejects_before_original_builders(route, change):
    original = body_program.emit_body_program

    def changed(*args, **kwargs):
        body = kwargs["root_warp"]
        if change == "clone_body":
            kwargs["root_warp"] = replace(body)
        elif change == "lost_binding":
            body.owner._body.body = None
        else:
            actions = body.program.actions
            altered = (
                actions[:-1]
                if change == "drop"
                else (*actions, actions[0])
                if change == "duplicate"
                else tuple(replace(a) for a in actions)
            )
            object.__setattr__(body.program, "actions", altered)
        with patch.object(
            roots,
            "emit_original_warp_operands",
            side_effect=AssertionError("producer reached"),
        ):
            return original(*args, **kwargs)

    with (
        patch.object(body_program, "emit_body_program", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(route)


@pytest.mark.parametrize("change", ("reverse", "stage", "sequence"))
def test_bridge_action_order_and_owner_reject_before_effects(change):
    original = body_program.emit_body_program

    def changed(*args, **kwargs):
        body = kwargs["root_warp"]
        first, last = body.program.actions
        assert isinstance(first, bridge.RootWarpBridgeAction)
        if change == "reverse":
            object.__setattr__(body.program, "actions", (last, first))
        elif change == "stage":
            object.__setattr__(first, "stage", 1)
        else:
            object.__setattr__(first, "sequence", replace(first.sequence))
        with patch.object(
            roots,
            "emit_original_warp_operands",
            side_effect=AssertionError("producer reached"),
        ):
            return original(*args, **kwargs)

    with (
        patch.object(body_program, "emit_body_program", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code("bridge")


@pytest.mark.parametrize(
    "route,stage", (("singleton", 0), ("bridge", 0), ("bridge", 1))
)
@pytest.mark.parametrize("change", ("owner", "map", "staged", "scratch", "prefix"))
def test_rejected_actual_call_preserves_attempt_and_original_retry(
    route, stage, change
):
    owner_type = (
        roots.RootWarpStageAction
        if route == "singleton"
        else bridge.RootWarpBridgeSequence
    )
    original = owner_type.emit
    checked = []

    def checked_emit(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal):
        current = ordinal[0] if ordinal else 0
        if current == stage:
            body = self._body.body
            assert body is not None
            before = (
                body.cursor,
                body.pending,
                body.publication,
                dict(boundaries),
                tuple(prefix),
            )
            wrong = [self, cg, plan, boundaries, staged, scratch, prefix, *ordinal]
            index = {"owner": 0, "map": 3, "staged": 4, "scratch": 5, "prefix": 6}[
                change
            ]
            wrong[index] = (
                dict(boundaries)
                if change == "map"
                else list(staged)
                if change == "staged"
                else list(prefix)
                if change == "prefix"
                else replace(wrong[index])
            )
            with (
                patch.object(
                    roots,
                    "emit_original_warp_operands",
                    side_effect=AssertionError("producer reached"),
                ),
                pytest.raises(chain._UnsupportedChain),
            ):
                original(*wrong)
            assert (
                body.cursor,
                body.pending,
                body.publication,
                dict(boundaries),
                tuple(prefix),
            ) == before
            checked.append(True)
        return original(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal)

    with patch.object(owner_type, "emit", checked_emit):
        _code(route)
    assert checked == [True]


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize(
    "change", ("missing", "drop", "append", "prefix", "alias", "receipt")
)
def test_returned_stage_must_match_original_successful_publication(route, change):
    owner_type = (
        roots.RootWarpStageAction
        if route == "singleton"
        else bridge.RootWarpBridgeSequence
    )
    original = owner_type.emit

    def changed(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal):
        if change == "missing":
            return []
        result = original(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal)
        if change == "drop":
            result.pop()
        elif change == "append":
            result.append("unapproved source")
        elif change == "prefix":
            prefix[0] = "unapproved prefix"
        elif change == "alias":
            assert plan.tensor_aliases
            plan.tensor_aliases[next(iter(plan.tensor_aliases))] = "unapproved alias"
        else:
            if isinstance(self, roots.RootWarpStageAction):
                assert self._completion is not None
                state = self._completion._state
            else:
                state = self._state
            assert state.publication is not None
            state.publication = replace(state.publication)
        return result

    with (
        patch.object(owner_type, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(route)


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize("change", ("prepared", "result", "input", "execution"))
def test_equal_valued_prepared_replacements_do_not_gain_authority(route, change):
    original = shared.emit_prepared_warp_stage

    def changed(cg, plan, boundaries, prepared, prefix):
        if change == "prepared":
            prepared = replace(prepared)
        elif change == "result":
            object.__setattr__(prepared, "result", replace(prepared.result))
        elif change == "input":
            object.__setattr__(prepared, "completion", replace(prepared.completion))
        else:
            object.__setattr__(
                prepared.completion, "execution", replace(prepared.completion.execution)
            )
        with patch(
            "helion._compiler.cute.chained_warp_mma.emit_warp_mma",
            side_effect=AssertionError("MMA reached"),
        ):
            return original(cg, plan, boundaries, prepared, prefix)

    with (
        patch.object(shared, "emit_prepared_warp_stage", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(route)


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize("change", ("source", "aliases"))
def test_native_recorder_cannot_recertify_changed_publication(route, change):
    original = native_matmul_metadata.record_native_stage

    def changed(cg, plan, dots, kind, lines):
        original(cg, plan, dots, kind, lines)
        if kind == "warp":
            if change == "source":
                lines.append("unapproved after recorder")
            else:
                plan.tensor_aliases["unapproved"] = "unapproved"

    with (
        patch.object(native_matmul_metadata, "record_native_stage", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(route)


@pytest.mark.parametrize(
    "route,stage", (("singleton", 0), ("bridge", 0), ("bridge", 1))
)
def test_accepted_emitter_failure_closes_attempt(route, stage):
    owner_type = (
        roots.RootWarpStageAction
        if route == "singleton"
        else bridge.RootWarpBridgeSequence
    )
    original = owner_type.emit
    builder = roots.emit_original_warp_operands
    checked = []

    def checked_emit(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal):
        current = ordinal[0] if ordinal else 0
        if current != stage:
            return original(
                self, cg, plan, boundaries, staged, scratch, prefix, *ordinal
            )
        body = self._body.body
        assert body is not None

        def failed(*args, **kwargs):
            builder(*args, **kwargs)
            raise chain._UnsupportedChain("accepted original producer failed")

        with (
            patch.object(roots, "emit_original_warp_operands", failed),
            pytest.raises(chain._UnsupportedChain, match="producer failed"),
        ):
            original(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal)
        assert body.cursor == stage and body.pending is body.program.actions[stage]
        with (
            patch.object(
                roots,
                "emit_original_warp_operands",
                side_effect=AssertionError("producer retried"),
            ),
            pytest.raises(chain._UnsupportedChain),
        ):
            original(self, cg, plan, boundaries, staged, scratch, prefix, *ordinal)
        checked.append(True)
        raise chain._UnsupportedChain("accepted original producer failed")

    with (
        patch.object(owner_type, "emit", checked_emit),
        pytest.raises(exc.BackendUnsupported, match="producer failed"),
    ):
        _code(route)
    assert checked == [True]


@pytest.mark.parametrize("change", ("map", "prefix", "staged", "owner"))
def test_final_bridge_validation_retains_original_object_identities(change):
    original = bridge.RootWarpBridgeSequence.validate

    def changed(self, cg, plan, boundaries, staged, prefix):
        return original(
            replace(self) if change == "owner" else self,
            cg,
            plan,
            dict(boundaries) if change == "map" else boundaries,
            list(staged) if change == "staged" else staged,
            list(prefix) if change == "prefix" else prefix,
        )

    with (
        patch.object(bridge.RootWarpBridgeSequence, "validate", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code("scan")


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize("change", ("copy", "prefix", "execution", "cached_source"))
def test_completed_body_return_keeps_exact_prefix_and_original_facts(route, change):
    original = body_program.emit_body_program

    def changed(*args, **kwargs):
        lines = original(*args, **kwargs)
        body = kwargs["root_warp"]
        if change == "copy":
            return list(lines)
        if change == "prefix":
            lines[0] = "unapproved late prefix"
        elif change == "execution":
            object.__setattr__(
                body.publication.prepared.completion.execution,
                "sync",
                "unapproved join",
            )
        elif isinstance(body.owner, bridge.RootWarpBridgeSequence):
            object.__setattr__(
                body.owner.bridge, "lines", ("unapproved cached bridge",)
            )
        else:
            object.__setattr__(
                body.publication, "prefix", ("unapproved cached source",)
            )
        return lines

    with (
        patch.object(body_program, "emit_body_program", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(route)
