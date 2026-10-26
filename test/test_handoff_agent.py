from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from helion.autotuner import HandoffAgentResult
from helion.autotuner import HandoffBundle
from helion.autotuner import HandoffCaseResult
from helion.autotuner import HandoffEvaluation
from helion.autotuner.handoff_agent import _noise
from helion.autotuner.handoff_agent import _write_sources
from helion.autotuner.handoff_agent import run_agent_rounds


@pytest.fixture
def bundle(tmp_path: Path) -> HandoffBundle:
    (tmp_path / "kernel.py").write_text("original")
    (tmp_path / "original.py").write_text("original")
    (tmp_path / "prompt.md").write_text("Original source instructions.")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"cases": [{"source": "kernel.py"}]})
    )
    return HandoffBundle(tmp_path)


@pytest.fixture
def session_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0, log=Mock())
    monkeypatch.setattr(
        "helion.autotuner.handoff_agent.time.monotonic", lambda: clock.now
    )

    def record(event, **kwargs):
        # Let each completed round consume one second of the session budget.
        if event == "handoff_round":
            clock.now += 1

    clock.log.record_handoff_event.side_effect = record
    return clock


def evaluation(bundle, perf, noise=0.01, *, ok=True):
    source = bundle.sources["kernel.py"]
    case = HandoffCaseResult(
        0,
        "ok" if ok else "accuracy_error",
        hashlib.sha256(source.encode()).hexdigest(),
        str(bundle.directory / "kernel.py"),
        "wall_clock",
        (perf - noise, perf, perf + noise) if ok else (),
        perf if ok else None,
        noise if ok else None,
        None if ok else "Incorrect output",
    )
    return HandoffEvaluation(
        ok,
        perf if ok else None,
        "ms",
        (case,),
        str(bundle.directory),
        "2026-10-05T00:00:00+00:00",
    )


def test_retains_only_valid_noise_separated_improvements(
    bundle, monkeypatch, session_clock
):
    calls = []

    def measure(workspace, **kwargs):
        source = workspace.sources["kernel.py"]
        calls.append((source, kwargs))
        return {
            "original": evaluation(workspace, 10),
            "faster": evaluation(workspace, 9),
            "noisy": evaluation(workspace, 8.99, 0.02),
            "invalid": evaluation(workspace, 1, ok=False),
        }[source]

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    proposals = iter(["faster", "noisy", "invalid"])
    prompts = []

    def agent(workspace):
        prompts.append(workspace.prompt)
        assert workspace.sources["kernel.py"] == (
            "original" if len(prompts) == 1 else "faster"
        )
        return {"kernel.py": next(proposals)}

    result = bundle.run_agent_rounds(
        agent, budget_seconds=3, repetitions=3, timeout=7, log=session_clock.log
    )
    assert isinstance(result, HandoffAgentResult)
    assert [record.accepted for record in result.rounds] == [True, False, False]
    assert [record.reason for record in result.rounds] == [
        "accepted",
        "regression_or_noise",
        "candidate_failed",
    ]
    assert result.rounds[1].improvement_threshold == pytest.approx(0.06)
    assert result.evaluation.perf == 9
    assert result.stop_reason == "deadline"
    assert bundle.sources == {"kernel.py": "faster"}
    assert [source for source, kwargs in calls] == [
        "original",
        "faster",
        "original",
        "faster",
        "noisy",
        "invalid",
    ]
    assert all(
        kwargs == {"repetitions": 3, "timeout": 7, "deadline": 103}
        for source, kwargs in calls
    )
    expected_prompt = (
        "Original source instructions.\n\nSession deadline (time.monotonic()): 103.0."
    )
    assert prompts == [expected_prompt] * 3
    assert bundle.prompt == "Original source instructions."
    assert (bundle.directory / "original.py").read_text() == "original"
    output = Path(result.directory)
    assert (output / "initial/kernel.py").read_text() == "original"
    assert not stat.S_IMODE((output / "initial/kernel.py").stat().st_mode) & 0o222
    for number, source in enumerate(["faster", "noisy", "invalid"], 1):
        saved = output / f"round_{number:04d}"
        assert (saved / "candidate/kernel.py").read_text() == source
        assert not (saved / "candidate.json").exists()
        assert not (saved / "incumbent.json").exists()
        record = json.loads((saved / "round.json").read_text())
        assert record["round"] == number
        assert {"candidate", "incumbent"} <= record.keys()
    assert len(json.loads((output / "feedback.json").read_text())["history"]) == 3


@pytest.mark.parametrize("failure", ["raise", "unknown_path", "bad_source"])
def test_agent_errors_restore_sources_and_continue(
    bundle, monkeypatch, session_clock, failure
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    rounds = 0

    def agent(workspace):
        nonlocal rounds
        rounds += 1
        assert workspace.sources == {"kernel.py": "original"}
        if rounds == 1:
            (workspace.directory / "kernel.py").write_text("unreturned edit")
            if failure == "raise":
                raise RuntimeError("provider failed")
            if failure == "unknown_path":
                return {"original.py": "bad"}
            return {"kernel.py": 42}
        return {"kernel.py": "faster"}

    result = bundle.run_agent_rounds(agent, budget_seconds=2, log=session_clock.log)
    assert result.rounds[0].reason == "agent_error"
    assert result.rounds[0].error is not None
    assert result.rounds[1].accepted
    assert bundle.sources == {"kernel.py": "faster"}
    assert (bundle.directory / "original.py").read_text() == "original"


@pytest.mark.parametrize("late_stage", ["callback", "evaluation"])
def test_late_results_never_replace_incumbent(bundle, monkeypatch, late_stage):
    clock = [100.0]
    monkeypatch.setattr(
        "helion.autotuner.handoff_agent.time.monotonic", lambda: clock[0]
    )
    calls = []

    def measure(workspace, **kwargs):
        calls.append(kwargs)
        result = evaluation(
            workspace, 9 if workspace.sources["kernel.py"] == "faster" else 10
        )
        if late_stage == "evaluation" and len(calls) == 3:
            clock[0] = 106
        return result

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)

    def agent(workspace):
        assert "Session deadline (time.monotonic()): 105.0." in workspace.prompt
        if late_stage == "callback":
            clock[0] = 106
        return {"kernel.py": "faster"}

    result = bundle.run_agent_rounds(agent, budget_seconds=5)
    assert result.stop_reason == "deadline"
    assert len(result.rounds) == 1
    assert result.rounds[0].reason == "deadline"
    assert not result.rounds[0].accepted
    assert len(calls) == (1 if late_stage == "callback" else 3)
    assert all(call["deadline"] == 105 for call in calls)
    assert bundle.sources == {"kernel.py": "original"}


def test_time_budget_alone_continues_after_agent_errors(bundle, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "helion.autotuner.handoff_agent.time.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        HandoffBundle, "evaluate", lambda workspace, **kwargs: evaluation(workspace, 10)
    )

    def agent(workspace):
        clock[0] += 1
        raise RuntimeError("provider failed")

    result = bundle.run_agent_rounds(agent, budget_seconds=3)
    assert len(result.rounds) == 3
    assert result.stop_reason == "deadline"
    assert all(not record.accepted for record in result.rounds)


def test_baseline_failure_does_not_call_agent(bundle, monkeypatch):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(workspace, 10, ok=False),
    )

    def agent(workspace):
        pytest.fail("An invalid baseline cannot start a source session")

    result = bundle.run_agent_rounds(agent, budget_seconds=30)
    assert result.stop_reason == "baseline_failed"
    assert result.rounds == ()
    assert not result.evaluation.ok
    assert bundle.sources == {"kernel.py": "original"}


@pytest.mark.parametrize(
    "budget_seconds",
    [
        0,
        -1,
        float("inf"),
        float("nan"),
    ],
)
def test_invalid_budget_fails_before_evaluation(bundle, budget_seconds):
    with pytest.raises(ValueError, match="budget_seconds must be finite and positive"):
        bundle.run_agent_rounds(lambda workspace: {}, budget_seconds=budget_seconds)
    assert not (bundle.directory / "agent_runs").exists()


@pytest.mark.parametrize("use_bundle_method", [False, True])
def test_budget_is_required_before_evaluation(bundle, use_bundle_method):
    with pytest.raises(TypeError, match="budget_seconds"):
        if use_bundle_method:
            bundle.run_agent_rounds(lambda workspace: {})
        else:
            run_agent_rounds(bundle, lambda workspace: {})
    assert not (bundle.directory / "agent_runs").exists()


def test_noise_uses_objective_units_for_multiple_shapes():
    cases = (
        HandoffCaseResult(0, "ok", "a", "a.py", "cuda_graph", (10, 11, 12), 11, 1),
        HandoffCaseResult(1, "ok", "b", "b.py", "cuda_graph", (40, 50, 60), 50, 10),
    )
    measured = HandoffEvaluation(True, 0.5, "ratio", cases, "out", "now")
    assert _noise(measured) == pytest.approx(0.1)


def test_failed_incumbent_and_evaluator_exception_cannot_promote(
    bundle, monkeypatch, session_clock
):
    calls = 0

    def measure(workspace, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("evaluation unavailable")
        return evaluation(workspace, 9 if calls == 2 else 10, ok=calls != 3)

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    result = bundle.run_agent_rounds(
        lambda workspace: {"kernel.py": "faster"},
        budget_seconds=2,
        log=session_clock.log,
    )
    assert all(not record.accepted for record in result.rounds)
    assert all(record.reason == "incumbent_failed" for record in result.rounds)
    assert result.rounds[1].error == "RuntimeError: evaluation unavailable"
    assert result.evaluation.ok and result.evaluation.perf == 10
    assert bundle.sources == {"kernel.py": "original"}


def test_unchanged_source_cannot_win_from_measurement_noise(
    bundle, monkeypatch, session_clock
):
    calls = 0

    def measure(workspace, **kwargs):
        nonlocal calls
        calls += 1
        return evaluation(workspace, 10 if calls == 1 else 1)

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    result = bundle.run_agent_rounds(
        lambda workspace: workspace.sources, budget_seconds=1, log=session_clock.log
    )
    assert calls == 1
    assert not result.rounds[0].accepted
    assert "did not change" in result.rounds[0].error


def test_new_session_has_fresh_feedback_and_current_starting_source(
    bundle, monkeypatch, session_clock
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    first = bundle.run_agent_rounds(
        lambda workspace: {"kernel.py": "faster"},
        budget_seconds=1,
        log=session_clock.log,
    )

    def agent(workspace):
        assert workspace.prompt == (
            "Original source instructions.\n\n"
            "Session deadline (time.monotonic()): 102.0."
        )
        assert first.directory not in workspace.prompt
        assert workspace.sources == {"kernel.py": "faster"}
        return workspace.sources

    second = bundle.run_agent_rounds(agent, budget_seconds=1, log=session_clock.log)
    assert second.directory != first.directory
    assert second.baseline.perf == 9
    assert (Path(second.directory) / "initial/kernel.py").read_text() == "faster"
    assert (bundle.directory / "original.py").read_text() == "original"


@pytest.mark.parametrize("kind", ["absolute", "parent", "symlink"])
def test_source_snapshots_cannot_write_or_chmod_outside_directory(tmp_path, kind):
    outside = tmp_path / "outside.py"
    outside.write_text("outside source")
    mode = stat.S_IMODE(outside.stat().st_mode)
    directory = tmp_path / "snapshot"
    directory.mkdir()
    if kind == "absolute":
        source = str(outside)
    elif kind == "parent":
        source = "../outside.py"
    else:
        source = "kernel.py"
        (directory / source).symlink_to(outside)
    with pytest.raises(ValueError, match="remain inside the bundle"):
        _write_sources(directory, {"safe.py": "safe", source: "changed"}, readonly=True)
    assert not (directory / "safe.py").exists()
    assert outside.read_text() == "outside source"
    assert stat.S_IMODE(outside.stat().st_mode) == mode


@pytest.fixture
def continuous(monkeypatch):
    state = SimpleNamespace(
        instances=[],
        actions=iter(()),
        calls=0,
        responses=[],
        closed=[],
        stop_reason="agent_completed",
        on_response=None,
    )

    class Adapter:
        def __init__(self, directory, *, deadline, log):
            self.directory = directory
            self.deadline = deadline
            self.log = log
            self.stop_reason = None
            state.instances.append(self)

        def __call__(self, workspace):
            # Every completed submission must be answered before waiting again.
            assert len(state.responses) == state.calls
            state.calls += 1
            try:
                action = next(state.actions)
            except StopIteration:
                self.stop_reason = state.stop_reason
                raise
            if isinstance(action, BaseException):
                raise action
            if callable(action):
                return action(self, workspace)
            return {"kernel.py": action}

        def complete_round(self, retained_sources, feedback, *, done):
            state.responses.append((dict(retained_sources), deepcopy(feedback), done))
            if state.on_response is not None:
                state.on_response(retained_sources, feedback, done=done)

        def close(self, *, reason):
            state.closed.append(reason)

    monkeypatch.setattr("helion.autotuner.handoff_agent.CLISourceAgent", Adapter)
    state.adapter_type = Adapter
    return state


@pytest.mark.parametrize("use_bundle_method", [False, True])
def test_default_adapter_gets_session_deadline_and_logs_rounds_immediately(
    bundle, monkeypatch, continuous, use_bundle_method
):
    monkeypatch.setattr("helion.autotuner.handoff_agent.time.monotonic", lambda: 100)
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace,
            {"original": 10, "faster": 9, "slower": 12}[workspace.sources["kernel.py"]],
        ),
    )
    log = Mock()
    continuous.actions = iter(["faster", "slower"])

    def answered(retained_sources, feedback, *, done):
        assert bundle.sources == retained_sources == {"kernel.py": "faster"}
        latest = feedback["history"][-1]
        saved = Path(latest["directory"]) / "round.json"
        assert json.loads(saved.read_text()) == json.loads(json.dumps(latest))
        event = log.record_handoff_event.call_args_list[-1]
        assert event.args == ("handoff_round",)
        assert event.kwargs["round"] == latest
        assert done is False

    continuous.on_response = answered
    options = {"budget_seconds": 30, "log": log}
    result = (
        bundle.run_agent_rounds(**options)
        if use_bundle_method
        else run_agent_rounds(bundle, **options)
    )
    (adapter,) = continuous.instances
    assert (adapter.directory, adapter.deadline, adapter.log) == (
        Path(result.directory) / "calls",
        130,
        log,
    )
    assert continuous.calls == 3
    assert [response[2] for response in continuous.responses] == [False, False]
    assert continuous.closed == ["agent_completed"]
    assert [record.accepted for record in result.rounds] == [True, False]
    assert result.rounds[1].reason == "regression_or_noise"
    assert result.evaluation.perf == 9
    assert [call.args[0] for call in log.record_handoff_event.call_args_list] == [
        "handoff_baseline",
        "handoff_round",
        "handoff_round",
        "handoff_completed",
    ]
    assert all(
        "perf_ms" not in call.kwargs for call in log.record_handoff_event.call_args_list
    )
    assert any(
        "Native source baseline: 10 ms" in call.args[0] for call in log.call_args_list
    )


def test_default_adapter_unavailable_fails_before_evaluation(bundle, monkeypatch):
    monkeypatch.setattr("helion.autotuner.handoff_cli.shutil.which", lambda name: None)
    measure = Mock()
    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    with pytest.raises(FileNotFoundError, match="Codex CLI"):
        bundle.run_agent_rounds(budget_seconds=30)
    measure.assert_not_called()
    assert bundle.sources == {"kernel.py": "original"}


@pytest.mark.parametrize(
    ("adapter_deadline", "budget_seconds", "expected"),
    [(None, 30, 130), (150, 30, 130), (110, 30, 110)],
)
def test_explicit_adapter_and_evaluations_share_earliest_deadline(
    bundle, monkeypatch, continuous, adapter_deadline, budget_seconds, expected
):
    monkeypatch.setattr("helion.autotuner.handoff_agent.time.monotonic", lambda: 100)
    calls = []

    def measure(workspace, **kwargs):
        calls.append(kwargs["deadline"])
        return evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        )

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    adapter = continuous.adapter_type(
        bundle.directory / "explicit", deadline=adapter_deadline, log=None
    )
    continuous.actions = iter(["faster"])
    result = bundle.run_agent_rounds(adapter, budget_seconds=budget_seconds)
    assert continuous.instances == [adapter]
    assert adapter.deadline == expected
    assert calls == [expected, expected, expected]
    assert result.rounds[0].accepted
    session = json.loads((Path(result.directory) / "session.json").read_text())
    assert session["deadline_monotonic"] == expected
    assert continuous.responses[0][1]["deadline_monotonic"] == expected
    assert continuous.closed == ["agent_completed"]


@pytest.mark.parametrize("submissions", [0, 1])
def test_continuous_agent_completion_does_not_create_extra_round(
    bundle, monkeypatch, continuous, submissions
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    continuous.actions = iter(["faster"] * submissions)
    result = bundle.run_agent_rounds(budget_seconds=30)
    assert result.stop_reason == "agent_completed"
    assert len(result.rounds) == submissions
    assert len(list(Path(result.directory).glob("round_*"))) == submissions
    assert len(continuous.instances) == 1
    assert continuous.calls == submissions + 1
    assert len(continuous.responses) == submissions
    assert continuous.closed == ["agent_completed"]
    assert bundle.sources == {"kernel.py": "faster" if submissions else "original"}
    assert bundle.prompt == "Original source instructions."


def test_continuous_submissions_have_no_count_limit_while_time_remains(
    bundle, monkeypatch, continuous
):
    monkeypatch.setattr("helion.autotuner.handoff_agent.time.monotonic", lambda: 100)

    def measure(workspace, **kwargs):
        source = workspace.sources["kernel.py"]
        return evaluation(workspace, 100 if source == "original" else 100 - int(source))

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    continuous.actions = iter(str(number) for number in range(1, 13))
    result = bundle.run_agent_rounds(budget_seconds=30)
    assert result.stop_reason == "agent_completed"
    assert len(result.rounds) == 12
    assert all(record.accepted for record in result.rounds)
    assert len(continuous.instances) == 1
    assert continuous.calls == 13
    assert continuous.closed == ["agent_completed"]
    assert bundle.sources == {"kernel.py": "12"}
    assert all(not done for retained, feedback, done in continuous.responses)
    assert all(
        feedback["remaining_seconds"] == 30 and "max_rounds" not in feedback
        for retained, feedback, done in continuous.responses
    )
    session = json.loads((Path(result.directory) / "session.json").read_text())
    assert "max_rounds" not in session
    assert session["budget_seconds"] == 30


@pytest.mark.parametrize("submissions", [0, 1])
def test_continuous_deadline_while_waiting_has_no_extra_round(
    bundle, monkeypatch, continuous, submissions
):
    clock = [100.0]
    monkeypatch.setattr(
        "helion.autotuner.handoff_agent.time.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )

    def expire(adapter, workspace):
        clock[0] = 130.0
        adapter.stop_reason = "deadline"
        raise StopIteration

    continuous.actions = iter([*["faster"] * submissions, expire])
    result = bundle.run_agent_rounds(budget_seconds=30)
    assert result.stop_reason == "deadline"
    assert len(result.rounds) == submissions
    assert len(list(Path(result.directory).glob("round_*"))) == submissions
    assert continuous.calls == submissions + 1
    assert len(continuous.responses) == submissions
    assert continuous.closed == ["deadline"]
    assert bundle.sources == {"kernel.py": "faster" if submissions else "original"}


@pytest.mark.parametrize("submissions", [0, 1])
def test_continuous_bridge_failure_closes_without_retry_or_extra_round(
    bundle, monkeypatch, continuous, submissions
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    continuous.actions = iter(
        [*["faster"] * submissions, RuntimeError("submission bridge unavailable")]
    )
    result = bundle.run_agent_rounds(budget_seconds=30)
    assert result.stop_reason == "agent_error"
    assert len(result.rounds) == submissions
    assert len(list(Path(result.directory).glob("round_*"))) == submissions
    assert len(continuous.instances) == 1
    assert continuous.calls == submissions + 1
    assert len(continuous.responses) == submissions
    assert continuous.closed == ["agent_error"]
    assert bundle.sources == {"kernel.py": "faster" if submissions else "original"}
    assert bundle.prompt == "Original source instructions."


def test_continuous_feedback_failure_retains_recorded_winner(
    bundle, monkeypatch, continuous
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    continuous.actions = iter(["faster"])

    def disconnected(retained_sources, feedback, *, done):
        assert bundle.sources == retained_sources == {"kernel.py": "faster"}
        latest = feedback["history"][-1]
        assert (Path(latest["directory"]) / "round.json").exists()
        raise RuntimeError("agent disconnected before feedback")

    continuous.on_response = disconnected
    result = bundle.run_agent_rounds(budget_seconds=30)
    assert result.stop_reason == "agent_error"
    assert len(result.rounds) == 1 and result.rounds[0].accepted
    assert continuous.calls == 1
    assert len(continuous.responses) == 1
    assert continuous.closed == ["agent_error"]
    assert result.evaluation.perf == 9
    assert bundle.sources == {"kernel.py": "faster"}


def test_continuous_late_evaluation_replies_with_rollback_and_done(
    bundle, monkeypatch, continuous
):
    clock = [100.0]
    monkeypatch.setattr(
        "helion.autotuner.handoff_agent.time.monotonic", lambda: clock[0]
    )

    def measure(workspace, **kwargs):
        assert kwargs["deadline"] == 130
        faster = workspace.sources["kernel.py"] == "faster"
        if faster:
            clock[0] = 130.0
        return evaluation(workspace, 9 if faster else 10)

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    continuous.actions = iter(["faster"])
    result = bundle.run_agent_rounds(budget_seconds=30)
    assert result.stop_reason == "deadline"
    assert len(result.rounds) == 1
    assert not result.rounds[0].accepted
    assert result.rounds[0].reason == "deadline"
    (response,) = continuous.responses
    retained, feedback, done = response
    assert done is True
    assert retained == bundle.sources == {"kernel.py": "original"}
    assert feedback["incumbent_evaluation"]["perf"] == 10
    assert feedback["history"][-1]["reason"] == "deadline"
    assert continuous.calls == 1
    assert continuous.closed == ["deadline"]


@pytest.mark.parametrize("stage", ["waiting", "evaluation"])
def test_continuous_keyboard_interrupt_closes_and_restores_sources(
    bundle, monkeypatch, continuous, stage
):
    def measure(workspace, **kwargs):
        if workspace.sources["kernel.py"] == "faster":
            raise KeyboardInterrupt
        return evaluation(workspace, 10)

    def interrupt(adapter, workspace):
        (workspace.directory / "kernel.py").write_text("unreturned edit")
        raise KeyboardInterrupt

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    continuous.actions = iter([interrupt if stage == "waiting" else "faster"])
    with pytest.raises(KeyboardInterrupt):
        bundle.run_agent_rounds(budget_seconds=30)
    assert len(continuous.instances) == 1
    assert continuous.calls == 1
    assert continuous.responses == []
    assert continuous.closed == ["interrupted"]
    assert bundle.sources == {"kernel.py": "original"}
    assert bundle.prompt == "Original source instructions."
