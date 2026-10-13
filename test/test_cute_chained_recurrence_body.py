from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_completed_store import _source as _completed_source
from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_loop_tmem_transport import _inputs
from .test_cute_chained_loop_tmem_transport import _packed_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_pipeline import _args
from .test_cute_chained_preparation_pipeline import _config
from helion._compiler.cute import chained_body_program as bodies
from helion._compiler.cute import chained_recurrence_body as recurrence
from helion._compiler.cute import chained_tcgen_stage as stages


def _code(kind="packed", dtype=torch.bfloat16, steps=3):
    if kind == "completed":
        return _completed_source(dtype, torch.float32, enabled=True, steps=steps)[0]
    if kind == "typed":
        # The original third operand must stay BF16: the fixture explicitly
        # narrows its other final-dot input to BF16.
        original_args = _args(steps=steps)
        left, right = original_args[:2]
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        args = (
            left.to(dtype),
            right.to(dtype),
            *original_args[2:],
        )
        return _source(_typed_sequence, args, _config(pipeline=True))
    kernel = _packed_sequence if kind == "packed" else _resident_carry_sequence
    return _source(
        kernel,
        _inputs("cpu", dtype, steps=steps),
        _config(16, pipeline=True, consumer_warps=8),
    )


@contextmanager
def _observe():
    captured, calls = [], []
    original_bind, original_stage = recurrence.bind_recurrence_body, stages.emit_stage

    def bind(*args, **kwargs):
        body = original_bind(*args, **kwargs)
        captured.append(body)
        return body

    def emit(*args, **kwargs):
        if captured and args[1] is captured[-1].plan:
            body = captured[-1]
            action = body.recurrence.stages[body._cursor]
            assert body._pending is action
            assert args[2] is body.boundaries
            assert args[6] is action.group
            assert args[3:6] == (
                action.group.stages[0],
                action.group.geometries[0],
                "chain_iteration & 1",
            )
            assert kwargs["execution"] is body.execution
            if body.selected_transports is not None:
                selected = body.selected_transports[body._cursor]
                for key in (
                    "residency",
                    "prepared_operand",
                    "prepared_group",
                    "tmem_input",
                    "tmem_output",
                    "tmem_accumulator",
                    "tmem_carry",
                ):
                    assert kwargs[key] is getattr(selected, key)
            calls.append(action)
        return original_stage(*args, **kwargs)

    with (
        patch.object(recurrence, "bind_recurrence_body", bind),
        patch.object(stages, "emit_stage", emit),
    ):
        yield captured, calls


@pytest.mark.parametrize("kind", ("typed", "packed", "carry", "completed"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("steps", (0, 3))
def test_real_recurrence_actions_keep_original_stage_arguments(kind, dtype, steps):
    initial = torch.cuda.is_initialized()
    with _observe() as (captured, calls):
        source = _code(kind, dtype, steps)
    assert len(captured) == 1
    body = captured[0]
    assert body.program.actions == body.recurrence.stages
    assert calls == list(body.recurrence.stages)
    assert body._pending is None and body._cursor == len(calls)
    assert body._finished == tuple(body._lines)
    assert "chain_store_0" in source and "chain_generation" in source
    assert (body.completed_action is not None) == (kind == "completed")
    if body.completed_action is not None:
        body.completed_action.validate_consumed()
    assert torch.cuda.is_initialized() == initial


def _mutate(body, change):
    if change == "boundary_owner":
        body.boundaries = dict(body.boundaries)
    elif change == "boundary":
        body.boundaries[next(iter(body.boundaries))] = "wrong_boundary"
    elif change == "alias_owner":
        object.__setattr__(body.plan, "tensor_aliases", dict(body.plan.tensor_aliases))
    elif change == "alias":
        body.plan.tensor_aliases["unexpected_alias"] = "unpublished"
    elif change == "config":
        body.codegen.device_function.config.config["num_warps"] += 4
    elif change == "execution":
        object.__setattr__(body.execution, "thread", "wrong_participant")
    elif change == "geometry":
        geometry = body.recurrence.stages[0].group.geometries[0]
        object.__setattr__(
            geometry, "logical", (geometry.logical[0], 999, geometry.logical[2])
        )
    elif change == "graph":
        node = body.plan.dots[-1]
        node.kwargs = {**node.kwargs, "review_changed": True}
    elif change == "dtype":
        body.plan.dots[-1].meta["val"] = (
            body.plan.dots[-1].meta["val"].to(torch.float16)
        )
    elif change == "region":
        region = body.recurrence.layout.regions[0]
        object.__setattr__(region, "byte_offset", region.byte_offset + 128)
    elif change == "raw":
        object.__setattr__(body.plan, "prepared_widenings", object())
    elif change == "drop":
        object.__setattr__(body.program, "actions", body.program.actions[:-1])
    elif change == "duplicate":
        object.__setattr__(
            body.program, "actions", (body.program.actions[0], *body.program.actions)
        )
    elif change == "reorder":
        object.__setattr__(
            body.program, "actions", tuple(reversed(body.program.actions))
        )
    elif change == "transport":
        assert body.selected_transports is not None
        object.__setattr__(
            body.selected_transports[0], "tmem_accumulator", "wrong_owner"
        )
    else:
        raise AssertionError(change)


@pytest.mark.parametrize("point", ("before", "between", "return"))
@pytest.mark.parametrize(
    "change",
    (
        "boundary_owner",
        "boundary",
        "alias_owner",
        "alias",
        "config",
        "execution",
        "geometry",
        "graph",
        "dtype",
        "region",
        "raw",
        "drop",
        "duplicate",
        "reorder",
    ),
)
def test_deep_revision_rejects_before_between_and_after_actions(point, change):
    bind, accept, emit = (
        recurrence.bind_recurrence_body,
        recurrence.RecurrenceBody.accept,
        bodies.emit_body_program,
    )
    touched = []

    def mutate(body):
        if not touched:
            _mutate(body, change)
            touched.append(body)

    def at_bind(*args, **kwargs):
        body = bind(*args, **kwargs)
        if point == "before":
            mutate(body)
        return body

    def at_accept(self, *args, **kwargs):
        result = accept(self, *args, **kwargs)
        if point == "between" and self._cursor == 1:
            mutate(self)
        return result

    def at_return(*args, **kwargs):
        result = emit(*args, **kwargs)
        if point == "return" and kwargs.get("recurrence_body") is not None:
            mutate(kwargs["recurrence_body"])
        return result

    with (
        patch.object(recurrence, "bind_recurrence_body", at_bind),
        patch.object(recurrence.RecurrenceBody, "accept", at_accept),
        patch.object(bodies, "emit_body_program", at_return),
        pytest.raises(Exception, match="recurrence"),
    ):
        _code()
    assert len(touched) == 1


@pytest.mark.parametrize("change", ("selection", "layout", "store", "span", "reuse"))
def test_finalized_store_and_return_obligations_are_not_generic_receipts(change):
    original = bodies.emit_body_program
    touched = []

    def emit(*args, **kwargs):
        body = kwargs.get("recurrence_body")
        if body is None:
            return original(*args, **kwargs)
        if change == "selection":
            assert body.selected_transports is not None
            body.selected_transports = tuple(reversed(body.selected_transports))
        elif change == "layout":
            assert body.storage is not None
            object.__setattr__(
                body.storage, "recurrence", replace(body.storage.recurrence)
            )
        result = original(*args, **kwargs)
        if change == "store":
            body.completed_action = None
        elif change == "span":
            result.append("changed_completed_span")
        elif change == "reuse":
            return original(*args, **kwargs)
        touched.append(body)
        return result

    with (
        patch.object(bodies, "emit_body_program", emit),
        pytest.raises(Exception, match="recurrence"),
    ):
        _code("completed")
    if change in ("store", "span"):
        assert len(touched) == 1


def test_failed_stage_cannot_replay_the_old_walk_or_reuse_pending_body():
    bind, emit = recurrence.bind_recurrence_body, stages.emit_stage
    captured, entered = [], []

    def at_bind(*args, **kwargs):
        body = bind(*args, **kwargs)
        captured.append(body)
        return body

    def fail(*args, **kwargs):
        result = emit(*args, **kwargs)
        if captured and args[1] is captured[0].plan:
            entered.append(args[3])
            raise RuntimeError("original recurrence stage failed")
        return result

    with (
        patch.object(recurrence, "bind_recurrence_body", at_bind),
        patch.object(stages, "emit_stage", fail),
        pytest.raises(Exception, match="original recurrence stage failed"),
    ):
        _code()
    (body,) = captured
    assert entered == [body.recurrence.stages[0].group.stages[0]]
    assert body._cursor == 0 and body._finished is None
    with pytest.raises(Exception, match="recurrence"):
        body.begin(body.recurrence.stages[0])


@pytest.mark.parametrize("change", ("seed", "transport"))
def test_selected_geometry_and_transport_facts_cannot_change_at_return(change):
    original = bodies.emit_body_program
    touched = []

    def emit(*args, **kwargs):
        result = original(*args, **kwargs)
        body = kwargs.get("recurrence_body")
        if body is not None:
            if change == "seed":
                assert body.seed_tiling is not None
                body.seed_tiling.max_columns = 32
            else:
                _mutate(body, change)
            touched.append(body)
        return result

    with (
        patch.object(bodies, "emit_body_program", emit),
        pytest.raises(Exception, match="recurrence|completed store"),
    ):
        _code("completed")
    assert len(touched) == 1


@pytest.mark.parametrize("change", ("storage", "selections", "layout", "stages"))
def test_binding_rejects_missing_final_owner_before_any_action(change):
    original = recurrence.bind_recurrence_body
    entered = []

    def bind(cg, plan, pipeline, workspace, boundaries, execution, **kwargs):
        if change == "storage":
            kwargs["storage"] = None
        elif change == "selections":
            kwargs["selected_transports"] = None
        elif change == "layout":
            plan = replace(plan, loop_workspace=replace(workspace.layout))
        else:
            workspace = replace(workspace, stages=())
        entered.append(change)
        return original(cg, plan, pipeline, workspace, boundaries, execution, **kwargs)

    with (
        patch.object(recurrence, "bind_recurrence_body", bind),
        patch.object(
            recurrence.RecurrenceBody,
            "begin",
            side_effect=AssertionError("must reject before action"),
        ),
        pytest.raises(
            Exception, match="recurrence body lost final transport selection"
        ),
    ):
        _code("completed")
    assert entered == [change]
