from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import sympy
import torch

from ._prepared_root_source_inverse import original_root_call
from ._prepared_root_source_inverse import original_root_source
from ._prepared_root_source_inverse import private_false_source
from .test_cute_chained_root_input_readiness import _source as initialized_source
from .test_cute_chained_root_weighted_pair import _code
from helion import exc
from helion._compiler.cute import chained_body_program as bodies
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_stage as actions
from helion._compiler.cute import chained_tcgen05 as root
from helion._compiler.cute import chained_tcgen_stage as stages


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group", [False, True])
def test_existing_pair_actions_enter_shared_body_exactly(dtype, group):
    with patch.object(bodies, "plan_root_action_body", return_value=None):
        original = _code(dtype=dtype, group=group)
    with (
        patch.object(
            bodies, "emit_body_program", wraps=bodies.emit_body_program
        ) as body,
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as stage,
    ):
        source = _code(dtype=dtype, group=group)
    assert source == original
    assert body.call_count == 1
    selected = body.call_args.kwargs["root_actions"]
    assert selected.consumed and selected.cursor == selected.sequence.next_stage == 2
    assert selected.program.actions == selected.actions
    assert [call.args[3] for call in stage.call_args_list] == [0, 1]
    assert all(
        call.kwargs["root_actions"] is selected.actions[i]
        for i, call in enumerate(stage.call_args_list)
    )
    assert "chain_0_c = cute.make_tensor" not in source


@pytest.mark.parametrize("mode", ["local", "upfront", "deferred"])
def test_initialized_paths_keep_original_source_in_shared_body(mode):
    with patch.object(bodies, "plan_root_action_body", return_value=None):
        original = initialized_source(mode)
    with patch.object(
        bodies, "emit_body_program", wraps=bodies.emit_body_program
    ) as body:
        source = initialized_source(mode)
    assert private_false_source(lambda: initialized_source(mode)) == original
    assert ast.dump(ast.parse(original_root_source(source))) == ast.dump(
        ast.parse(original)
    )
    selected = original_root_call(body.call_args_list).kwargs["root_actions"]
    assert selected.consumed and selected.cursor == 2
    assert selected.plan.dots[0] not in selected.boundaries


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "ordinal",
        "config",
        "geometry",
        "execution",
        "boundaries",
        "aliases",
        "selection",
        "cursor",
    ],
)
def test_bound_action_program_rejects_before_any_shared_stage(mutation):
    original = bodies.plan_root_action_body
    accepted = []

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        assert value is not None
        accepted.append(value)
        if mutation == "order":
            object.__setattr__(value.program, "actions", value.program.actions[::-1])
        elif mutation == "ordinal":
            object.__setattr__(value.actions[0], "stage", 1)
        elif mutation == "config":
            value.codegen.device_function.config.config[
                "cute_chained_pointwise_unroll"
            ] = 2
        elif mutation == "geometry":
            object.__setattr__(value.sequence.geometries[0], "native_rows", 64)
        elif mutation == "execution":
            object.__setattr__(value.execution, "thread", "wrong_thread")
        elif mutation == "boundaries":
            value.boundaries[value.plan.dots[0]] = "unapproved"
        elif mutation == "aliases":
            value.plan.tensor_aliases["unknown"] = "unapproved"
        elif mutation == "selection":
            value.sequence.pair_selection = None
        else:
            value.cursor = 1
        return value

    with (
        patch.object(bodies, "plan_root_action_body", capture),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as stage,
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert accepted and not accepted[0].consumed and stage.call_count == 0


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize(
    "mutation", ["drop", "scope", "append_alias", "execution", "duplicate"]
)
def test_actual_completion_body_and_aliases_are_sealed(ordinal, mutation):
    original = stages.emit_stage
    seen = []

    def intercept(cg, plan, boundaries, stage, geometry, *args, **kwargs):
        result = original(cg, plan, boundaries, stage, geometry, *args, **kwargs)
        if stage == ordinal:
            body = kwargs["root_action_body"]
            seen.append(body)
            if mutation == "drop":
                body.completion = None
            elif mutation == "scope":
                result[-1] = "    " + result[-1]
            elif mutation == "append_alias":
                plan.tensor_aliases["unknown"] = "unapproved"
            elif mutation == "execution":
                object.__setattr__(body.execution, "sync", "wrong_sync")
            else:
                body.record_completion(
                    cg,
                    plan,
                    boundaries,
                    kwargs["root_actions"],
                    stage,
                    geometry,
                    kwargs["execution"],
                    result,
                )
        return result

    with (
        patch.object(stages, "emit_stage", intercept),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert seen and not seen[0].consumed


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("phase", ["before_record", "before_accept"])
@pytest.mark.parametrize("mutation", ["drop", "stale", "wrong_stage", "aliases"])
def test_original_transition_token_is_required_through_body_accept(
    ordinal, phase, mutation
):
    original_complete = actions.RootStageAction.complete
    original_accept = bodies.RootActionBody.accept
    seen = []

    def mutate(action):
        token = action._completion
        assert token is not None and token.matches(action)
        seen.append(action)
        if mutation == "drop":
            object.__setattr__(action, "_completion", None)
        elif mutation == "stale":
            object.__setattr__(action, "_completion", replace(token, progress=()))
        elif mutation == "wrong_stage":
            object.__setattr__(
                action, "_completion", replace(token, stage=action.stage + 1)
            )
        else:
            action.sequence.plan.tensor_aliases["unapproved"] = "unapproved"

    def complete(action):
        result = original_complete(action)
        assert result is None
        if action.stage == ordinal and phase == "before_record":
            mutate(action)

    def accept(body, action, lines):
        if action.stage == ordinal and phase == "before_accept":
            mutate(action)
        return original_accept(body, action, lines)

    with (
        patch.object(actions.RootStageAction, "complete", complete),
        patch.object(bodies.RootActionBody, "accept", accept),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert len(seen) == 1


@pytest.mark.parametrize("ordinal", [0, 1])
def test_original_complete_none_return_and_private_once_only_token(ordinal):
    original = actions.RootStageAction.complete
    seen = []

    def complete(action):
        assert action._completion is None
        assert original(action) is None
        token = action._completion
        assert token is not None and token.matches(action)
        assert replace(action) == action
        assert not token.matches(replace(action))
        if action.stage == ordinal:
            seen.append(action)
            with pytest.raises(chain._UnsupportedChain, match="out of order"):
                original(action)
            stage = action.sequence.next_stage
            try:
                action.sequence.next_stage = action.stage
                with pytest.raises(chain._UnsupportedChain, match="already completed"):
                    original(action)
            finally:
                action.sequence.next_stage = stage
            assert action._completion is token and token.matches(action)

    with patch.object(actions.RootStageAction, "complete", complete):
        _code()
    assert len(seen) == 1


@pytest.mark.parametrize("ordinal", [0, 1])
def test_failed_original_transition_never_creates_token(ordinal):
    original = actions.RootStageAction.complete
    seen = []

    def complete(action):
        if action.stage == ordinal:
            action.sequence.input_waited = False
            seen.append(action)
        return original(action)

    with (
        patch.object(actions.RootStageAction, "complete", complete),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert len(seen) == 1 and seen[0]._completion is None


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("mutation", ["codegen", "map", "action", "geometry"])
def test_actual_stage_call_arguments_are_checked(ordinal, mutation):
    original = stages.emit_stage
    seen = []

    def intercept(cg, plan, boundaries, stage, geometry, *args, **kwargs):
        if stage == ordinal:
            seen.append(kwargs["root_action_body"])
            if mutation == "codegen":
                cg = object()
            elif mutation == "map":
                boundaries = dict(boundaries)
            elif mutation == "action":
                kwargs["root_actions"] = kwargs["root_actions"].sequence.action(stage)
            else:
                geometry = replace(geometry, transpose=True)
        return original(cg, plan, boundaries, stage, geometry, *args, **kwargs)

    with (
        patch.object(stages, "emit_stage", intercept),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert seen and not seen[0].consumed


def test_failed_second_producer_never_falls_back_or_installs_body():
    original = root._stage
    calls = []

    def fail(cg, plan, boundaries, scans, stage, role, *args, **kwargs):
        calls.append((stage, role))
        if stage == 1:
            raise chain._UnsupportedChain("injected root action failure")
        return original(cg, plan, boundaries, scans, stage, role, *args, **kwargs)

    with (
        patch.object(root, "_stage", fail),
        pytest.raises(Exception, match="injected root action failure"),
    ):
        _code()
    assert calls == [(0, "a"), (0, "b"), (1, "b")]


@pytest.mark.parametrize("mutation", ["pair_count", "bridge", "input_wait", "staged"])
def test_completed_readiness_and_staged_leaf_inventory_cannot_change(mutation):
    original = stages.emit_stage
    seen = []

    def intercept(cg, plan, boundaries, stage, geometry, *args, **kwargs):
        result = original(cg, plan, boundaries, stage, geometry, *args, **kwargs)
        if stage == 1:
            body = kwargs["root_action_body"]
            seen.append(body)
            if mutation == "pair_count":
                body.sequence.pair_completed = 0
            elif mutation == "bridge":
                body.sequence.pair_bridged = False
            elif mutation == "input_wait":
                body.sequence.input_waited = False
            else:
                body.sequence.staged.append(
                    chain._StagedInput(
                        plan.dots[0],
                        plan.dots[0],
                        "unapproved",
                        "a",
                        (128, 128),
                        indices=(),
                        coordinates=(sympy.Symbol("row"), sympy.Symbol("column")),
                        inverse_axes=((0, 0), (1, 0)),
                    )
                )
        return result

    with (
        patch.object(stages, "emit_stage", intercept),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert seen and not seen[0].consumed
