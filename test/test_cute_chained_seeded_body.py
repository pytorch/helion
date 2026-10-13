from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._prepared_root_source_inverse import original_root_call
from ._prepared_root_source_inverse import original_root_source
from ._prepared_root_source_inverse import private_false_source
from .test_cute_chained_accumulator import _cpu
from .test_cute_chained_accumulator import _late_rhs_args
from .test_cute_chained_accumulator import _late_rhs_config
from .test_cute_chained_accumulator import _late_rhs_pair
from .test_cute_chained_root_input_readiness import _source
from .test_cute_chained_root_paired import _source as paired_source
from helion import exc
from helion._compiler.cute import chained_body_program as bodies
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages


def _code(mode="deferred", dtype=torch.bfloat16, columns=32):
    if mode.startswith("paired-"):
        return paired_source(dtype, columns, mode.removeprefix("paired-"))
    if mode in ("serial64", "overlap64"):
        with _cpu():
            return _late_rhs_pair._bind_isolated(
                _late_rhs_args(dtype=dtype, n=128)
            ).to_code(
                _late_rhs_config(
                    cute_chained_k_schedule=mode,
                    cute_chained_seed_tile_columns=columns,
                    cute_chained_pointwise_vectorize=True,
                    cute_chained_pointwise_read_cache=True,
                    cute_chained_pointwise_unroll=8,
                )
            )
    return _source(mode, dtype, columns)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32, 64))
@pytest.mark.parametrize(
    "mode",
    (
        "local",
        "upfront",
        "deferred",
        "serial64",
        "overlap64",
        "paired-serial64",
        "paired-overlap64",
    ),
)
def test_original_source_construction_and_seed_only_publication(mode, dtype, columns):
    with (
        patch.object(bodies, "plan_root_action_body", return_value=None),
        patch.object(legacy, "_stage", wraps=legacy._stage) as old_construct,
    ):
        before = _code(mode, dtype, columns)
    events = []
    original_accept, original_enqueue = (
        bodies.RootActionBody.accept,
        bodies.RootActionBody.enqueue,
    )

    def accept(body, action, lines):
        original_accept(body, action, lines)
        events.append(action.stage)
        assert body.plan.dots[0] not in body.boundaries
        assert action.result == (
            "seed_only" if action.stage == 0 else "logical_fragment"
        )

    def enqueue(body, action, lines):
        assert body.cursor == 1 and body.sequence.next_stage == 1
        original_enqueue(body, action, lines)
        assert body.cursor == 1 and body.sequence.next_stage == 1
        assert body.deferred_span is None
        assert body.sequence._rhs_completion.matches(body.sequence)
        events.append("enqueue")

    with (
        patch.object(bodies.RootActionBody, "accept", accept),
        patch.object(bodies.RootActionBody, "enqueue", enqueue),
        patch.object(
            bodies, "emit_body_program", wraps=bodies.emit_body_program
        ) as emitted,
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as issued,
        patch.object(legacy, "_stage", wraps=legacy._stage) as construct,
    ):
        after = _code(mode, dtype, columns)
    assert private_false_source(lambda: _code(mode, dtype, columns)) == before
    restored = original_root_source(after)
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(before))
    assert [call.args[4:6] for call in construct.call_args_list] == [
        call.args[4:6] for call in old_construct.call_args_list
    ]
    body = original_root_call(emitted.call_args_list).kwargs["root_actions"]
    assert body.consumed and body.cursor == body.sequence.next_stage == 2
    assert events == ([0, 1] if mode in ("local", "upfront") else [0, "enqueue", 1])
    assert body.event == len(body.program.actions)
    assert [call.args[3] for call in issued.call_args_list] == [0, 1]
    assert all(call.kwargs["tmem_input"] is None for call in issued.call_args_list)
    assert all(
        call.kwargs["tmem_accumulator"] == "chain_tptr + 0"
        for call in issued.call_args_list
    )
    assert "chain_0_c =" not in after
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)" in restored


@pytest.mark.parametrize("mutation", ("drop", "duplicate", "reorder", "input"))
def test_queue_program_rejects_before_any_stage(mutation):
    original = bodies.plan_root_action_body

    def plan(*args, **kwargs):
        body = original(*args, **kwargs)
        assert body is not None and body.deferred is not None
        if mutation == "input":
            object.__setattr__(body.deferred, "inputs", replace(body.deferred.inputs))
        else:
            a, queue, b = body.program.actions
            object.__setattr__(
                body.program,
                "actions",
                {
                    "drop": (a, b),
                    "duplicate": (a, queue, queue, b),
                    "reorder": (queue, a, b),
                }[mutation],
            )
        return body

    with (
        patch.object(bodies, "plan_root_action_body", plan),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as stage,
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert stage.call_count == 0


@pytest.mark.parametrize("mode", ("local", "upfront", "deferred"))
def test_original_deferred_program_must_match_selected_mode(mode):
    original = bodies.plan_root_action_body

    def plan(*args, **kwargs):
        kwargs["deferred_rhs"] = ["changed_rhs"]
        return original(*args, **kwargs)

    with (
        patch.object(bodies, "plan_root_action_body", plan),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code(mode)


@pytest.mark.parametrize("phase", ("record", "enqueue"))
@pytest.mark.parametrize("mutation", ("drop", "clone", "seed", "paired", "token"))
def test_original_stage_zero_transition_is_required_for_enqueue(phase, mutation):
    complete, enqueue = roots.RootStageAction.complete, bodies.RootActionBody.enqueue

    def mutate(action):
        sequence = action.sequence
        if mutation == "drop":
            sequence.seed_completion = None
        elif mutation == "clone":
            sequence.seed_completion = replace(sequence.seed_completion)
        elif mutation == "seed":
            object.__setattr__(
                sequence.seed_completion, "result", replace(sequence.initialized)
            )
        elif mutation == "paired":
            sequence.paired_issued = not sequence.paired_issued
        else:
            object.__setattr__(
                action, "_completion", replace(action._completion, progress=())
            )

    def finish(action):
        complete(action)
        if action.stage == 0 and phase == "record":
            mutate(action)

    def queue(body, action, lines):
        if phase == "enqueue":
            mutate(body.actions[0])
        return enqueue(body, action, lines)

    with (
        patch.object(roots.RootStageAction, "complete", finish),
        patch.object(bodies.RootActionBody, "enqueue", queue),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()


@pytest.mark.parametrize(
    "mutation", ("drop", "stale", "queue", "seed", "advance", "wait")
)
def test_original_enqueue_transition_is_required(mutation):
    original = roots.RootStageSequence.enqueue_rhs

    def enqueue(sequence, lines):
        assert original(sequence, lines) is None
        assert sequence._rhs_completion.matches(sequence)
        if mutation == "drop":
            sequence._rhs_completion = None
        elif mutation == "stale":
            sequence._rhs_completion = replace(sequence._rhs_completion, progress=())
        elif mutation == "queue":
            sequence.queued_rhs = replace(sequence.queued_rhs)
        elif mutation == "seed":
            sequence.seed_completion = replace(sequence.seed_completion)
        elif mutation == "advance":
            sequence.next_stage += 1
        else:
            sequence.input_waited = not sequence.input_waited

    with (
        patch.object(roots.RootStageSequence, "enqueue_rhs", enqueue),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()


@pytest.mark.parametrize("mutation", ("seed", "wait", "aliases"))
def test_original_enqueue_cannot_reseal_changed_pre_transition_state(mutation):
    original = roots.RootStageSequence.enqueue_rhs

    def enqueue(sequence, lines):
        if mutation == "seed":
            sequence.seed_completion = replace(sequence.seed_completion)
        elif mutation == "wait":
            sequence.input_waited = not sequence.input_waited
        else:
            name = next(iter(sequence.plan.tensor_aliases))
            sequence.plan.tensor_aliases[name] = "changed_tensor"
        return original(sequence, lines)

    with (
        patch.object(roots.RootStageSequence, "enqueue_rhs", enqueue),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()


@pytest.mark.parametrize("phase", ("before", "after"))
@pytest.mark.parametrize("mutation", ("change", "drop", "append"))
def test_enqueue_requires_previously_accepted_source_prefix(mutation, phase):
    original = bodies.RootActionBody.enqueue

    def enqueue(body, action, lines):
        assert lines and body.last_action is not None
        if phase == "after":
            original(body, action, lines)
        if mutation == "change":
            lines[0] = "changed_source = 0"
        elif mutation == "drop":
            lines.pop()
        else:
            lines.append("changed_source = 0")
        if phase == "before":
            original(body, action, lines)

    with (
        patch.object(bodies.RootActionBody, "enqueue", enqueue),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()


@pytest.mark.parametrize(
    "mutation",
    ("offset", "lines", "span", "queue", "token", "before", "seed", "half", "source"),
)
def test_original_insertion_receipt_required_by_naming_prepass(mutation):
    original = bodies.RootActionBody.deferred_position

    def position(body, lines, start, deferred):
        span = body.deferred_span
        if mutation == "offset":
            object.__setattr__(span, "start", span.start + 1)
        elif mutation == "lines":
            object.__setattr__(span, "lines", (*span.lines, "changed"))
        elif mutation == "span":
            body._deferred_span = None
        elif mutation == "queue":
            body.sequence.queued_rhs = replace(body.sequence.queued_rhs)
        elif mutation == "token":
            body.sequence._rhs_completion = None
        elif mutation == "before":
            object.__setattr__(span.completion, "before", ())
        elif mutation == "seed":
            body.sequence.seed_completion = replace(body.sequence.seed_completion)
        elif mutation == "half":
            object.__setattr__(body.sequence.half_producer, "lines", ("changed",))
        else:
            lines[start + span.start] = "changed"
        return original(body, lines, start, deferred)

    with (
        patch.object(bodies.RootActionBody, "deferred_position", position),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code("serial64")


@pytest.mark.parametrize(
    "mutation",
    (
        "tile",
        "kernel_args",
        "out_name",
        "drop_output",
        "extra",
        "transfer",
        "wrapper",
        "registration",
    ),
)
def test_finalized_paired_descriptor_preserves_original_authority(mutation):
    original = bodies.RootActionBody.deferred_position

    def position(body, lines, start, deferred):
        paired = body.sequence.initialized.paired
        wrapper = paired.transfer.wrapper
        assert "out_name" in wrapper
        if mutation == "tile":
            wrapper["tile"] = (1, 1)
        elif mutation == "kernel_args":
            wrapper["kernel_args"] = ["changed_atom", "changed_tensor"]
        elif mutation == "out_name":
            wrapper["out_name"] = "changed_output"
        elif mutation == "drop_output":
            wrapper.pop("out_name")
        elif mutation == "extra":
            wrapper["changed"] = True
        elif mutation == "transfer":
            object.__setattr__(paired, "transfer", replace(paired.transfer))
        elif mutation == "wrapper":
            object.__setattr__(paired.transfer, "wrapper", dict(wrapper))
        else:
            body.codegen.cute_wrapper_plans.remove(wrapper)
        return original(body, lines, start, deferred)

    with (
        patch.object(bodies.RootActionBody, "deferred_position", position),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code("paired-overlap64")


@pytest.mark.parametrize("mutation", ("change", "drop", "reorder"))
def test_naming_prepass_preserves_installed_alias_prefix(mutation):
    original = bodies.RootActionBody.deferred_position

    def position(body, lines, start, deferred):
        name, value = body.aliases[0]
        assert body.plan.tensor_aliases[name] == value
        if mutation == "change":
            body.plan.tensor_aliases[name] = "changed_tensor"
        else:
            body.plan.tensor_aliases.pop(name)
            if mutation == "reorder":
                body.plan.tensor_aliases[name] = value
        return original(body, lines, start, deferred)

    with (
        patch.object(bodies.RootActionBody, "deferred_position", position),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()


def test_failed_second_stage_exposes_no_complete_body_or_splice():
    original = stages.emit_stage
    seen = []

    def emit(*args, **kwargs):
        body = kwargs["root_action_body"]
        seen.append(body)
        if args[3] == 1:
            raise chain._UnsupportedChain("failed initialized stage one")
        return original(*args, **kwargs)

    with (
        patch.object(stages, "emit_stage", emit),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    assert seen and not seen[-1].consumed and not seen[-1].finished_lines


def test_incomplete_original_selection_keeps_pre_admission_fallback():
    original = bodies.plan_root_action_body
    seen = []

    def plan(*args, **kwargs):
        sequence = args[2]
        assert sequence.initialized is not None and sequence.seeded_inputs is None
        assert sequence.handles(0) and not sequence.handles(1)
        result = original(*args, **kwargs)
        seen.append(sequence)
        assert (
            result is None
            and sequence.next_stage == 0
            and sequence.seed_completion is None
        )
        return result

    # Removing the original optional stage-one receipt cannot create capability.
    with (
        patch.object(roots, "capture_seeded_inputs", return_value=None),
        patch.object(bodies, "plan_root_action_body", plan),
        patch.object(
            bodies, "emit_body_program", wraps=bodies.emit_body_program
        ) as emitted,
    ):
        _code()
    assert len(seen) == 1 and emitted.call_count == 0


@pytest.mark.parametrize("mode", ("local", "upfront", "deferred"))
@pytest.mark.parametrize("ordinal", (0, 1))
@pytest.mark.parametrize("mutation", ("codegen", "map", "geometry", "execution"))
def test_initialized_actual_call_identity_and_execution(mode, ordinal, mutation):
    original = stages.emit_stage

    def emit(cg, plan, boundaries, stage, geometry, *args, **kwargs):
        if stage == ordinal:
            if mutation == "codegen":
                cg = object()
            elif mutation == "map":
                boundaries = dict(boundaries)
            elif mutation == "geometry":
                geometry = replace(geometry, transpose=True)
            else:
                kwargs["execution"] = replace(kwargs["execution"], sync="wrong_sync")
        return original(cg, plan, boundaries, stage, geometry, *args, **kwargs)

    with (
        patch.object(stages, "emit_stage", emit),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code(mode)


@pytest.mark.parametrize("mutation", ("duplicate", "token", "drop_span"))
def test_queue_authority_cannot_change_before_stage_one(mutation):
    original = bodies.RootActionBody.enqueue

    def enqueue(body, action, lines):
        original(body, action, lines)
        if mutation == "duplicate":
            original(body, action, lines)
        elif mutation == "token":
            body.sequence._rhs_completion = None
        else:
            body._deferred_span = None

    with (
        patch.object(bodies.RootActionBody, "enqueue", enqueue),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as stage,
        pytest.raises(exc.BackendUnsupported),
    ):
        _code()
    # Stage-one validation is allowed to run, but no second producer may begin.
    assert stage.call_count <= 2


@pytest.mark.parametrize("mode", ("local", "upfront", "deferred", "paired-overlap64"))
@pytest.mark.parametrize("policy", ("phase", "accumulator", "completed_store"))
def test_rejected_stage_policy_does_not_start_body_attempt(mode, policy):
    original = stages.emit_stage

    def emit(*args, **kwargs):
        body = kwargs["root_action_body"]
        before = (
            body.pending,
            body.cursor,
            body.event,
            body.progress,
            body.completion,
            body.completed_action,
            roots.root_stage_progress(body.sequence),
        )
        changed, options = list(args), dict(kwargs)
        if policy == "phase":
            changed[5] = "1"
        elif policy == "accumulator":
            options["tmem_accumulator"] = "chain_tptr + 64"
        else:
            options["completed_store"] = object()
        with pytest.raises(chain._UnsupportedChain):
            original(*changed, **options)
        assert (
            body.pending,
            body.cursor,
            body.event,
            body.progress,
            body.completion,
            body.completed_action,
            roots.root_stage_progress(body.sequence),
        ) == before
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", emit):
        _code(mode)


def test_actual_producer_failure_after_validation_cannot_fall_back():
    original_plan, original_producer = bodies.plan_root_action_body, legacy._stage
    accepted, failed = [], []

    def plan(*args, **kwargs):
        body = original_plan(*args, **kwargs)
        accepted.append(body)
        return body

    def produce(cg, plan, boundaries, scans, stage, role, *args, **kwargs):
        if stage == 1 and accepted:
            body = accepted[-1]
            assert body.pending is body.actions[1]
            failed.append((stage, role))
            raise chain._UnsupportedChain("accepted initialized producer failed")
        return original_producer(
            cg, plan, boundaries, scans, stage, role, *args, **kwargs
        )

    with (
        patch.object(bodies, "plan_root_action_body", plan),
        patch.object(legacy, "_stage", produce),
        pytest.raises(
            exc.BackendUnsupported, match="accepted initialized producer failed"
        ),
    ):
        _code()
    assert failed == [(1, "a")]
    assert not accepted[-1].consumed and accepted[-1].deferred_span is None
