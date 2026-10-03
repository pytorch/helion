from __future__ import annotations

from dataclasses import fields
from dataclasses import replace
import inspect
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_accumulator import _initialized_args
from test.test_cute_chained_accumulator import _initialized_code
from test.test_cute_chained_seeded_body import _code
from test.test_cute_prepared_state_body import _selected

from helion import exc
from helion._compiler.cute import chained_body_program as body
from helion._compiler.cute import chained_tcgen_stage as stage
from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute import prepared_state_planner as state


def test_private_constructor_defaults_are_not_changed():
    for function, names in (
        (
            chunk_recurrence._plan_chunk_recurrence,
            ("prepared_edge", "prepared_continuation", "prepared_epoch"),
        ),
        (body.plan_root_action_body, ("prepared_continuation",)),
    ):
        parameters = inspect.signature(function).parameters
        assert all(parameters[name].default is False for name in names)
    assert [
        field.name
        for field in fields(chunk_recurrence.CuteChunkRecurrencePlan)
        if not field.compare
    ] == ["prepared_projection", "prepared_continuation", "prepared_epoch"]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_original_public_initialization_optout_remains_exact(dtype):
    args = _initialized_args(dtype, 128)
    with patch.object(
        state, "plan_state_transfers", wraps=state.plan_state_transfers
    ) as plan:
        absent = _initialized_code(args, "missing")
        disabled = _initialized_code(args, False)
    assert absent == disabled
    assert "prepared_tcgen_edge.execute_prepared_read" not in absent
    assert plan.call_count == 0


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("local", "serial64", "overlap64"))
def test_normal_root_matches_explicit_private_selected_source(mode, dtype):
    original = body.plan_root_action_body
    choices = []

    def observe(*args, **kwargs):
        selected = kwargs.get("prepared_continuation", False)
        result = original(*args, **kwargs)
        assert result is not None
        assert (result.continuation is not None) is selected
        assert emitted.call_count == 0
        choices.append(selected)
        return result

    with (
        patch.object(body, "plan_root_action_body", observe),
        patch.object(stage, "emit_stage", wraps=stage.emit_stage) as emitted,
    ):
        actual = _code(mode, dtype, 32)
    assert choices == [False, True]
    assert "prepared_tcgen_edge.execute_prepared_read" in actual
    assert actual == _selected(mode, dtype, 32)


@pytest.mark.parametrize("mutation", ("drop", "config", "decline", "selected-owner"))
def test_accepted_root_binding_never_falls_back(mutation):
    original = body.plan_root_action_body
    calls = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        selected = kwargs.get("prepared_continuation", False)
        calls.append(selected)
        if not selected:
            if mutation == "drop":
                object.__setattr__(
                    result.program, "actions", result.program.actions[1:]
                )
            elif mutation == "config":
                result.codegen.device_function.config.config["num_warps"] = 8
        elif mutation == "decline":
            return None
        elif mutation == "selected-owner":
            result.plan = replace(result.plan)
        return result

    with (
        patch.object(body, "plan_root_action_body", mutate),
        patch.object(stage, "emit_stage", wraps=stage.emit_stage) as emitted,
        pytest.raises(exc.BackendUnsupported),
    ):
        _code("local", torch.bfloat16, 32)
    assert calls == ([False] if mutation in ("drop", "config") else [False, True])
    assert emitted.call_count == 0
