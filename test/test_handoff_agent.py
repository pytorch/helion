from __future__ import annotations

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
from helion.autotuner import HandoffRequest
from helion.autotuner.handoff_agent import MAX_CONSECUTIVE_INVALID_PROPOSALS
from helion.autotuner.handoff_agent import _noise
from helion.autotuner.handoff_agent import _write_sources
from helion.autotuner.handoff_agent import run_agent_rounds

pytestmark = pytest.mark.cpu_only


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
    requests = []

    def agent(request):
        assert isinstance(request, HandoffRequest)
        requests.append(request)
        assert request.deadline == 103
        assert request.sources == {
            "kernel.py": "original" if len(requests) == 1 else "faster"
        }
        # The bundle holds the retained source while the agent works.
        assert bundle.sources == request.sources
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
    assert [request.prompt for request in requests] == [expected_prompt] * 3
    assert [
        None if request.feedback is None else request.feedback["reason"]
        for request in requests
    ] == [None, "accepted", "regression_or_noise"]
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
        feedback = json.loads((saved / "feedback.json").read_text())
        assert feedback["round"] == number
    assert (output / "prompt.md").read_text() == expected_prompt


@pytest.mark.parametrize("failure", ["unknown_path", "bad_source", "not_mapping"])
def test_invalid_proposals_restore_sources_and_continue(
    bundle, monkeypatch, session_clock, failure
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    requests = []

    def agent(request):
        requests.append(request)
        assert request.sources == {"kernel.py": "original"}
        if len(requests) == 1:
            # An edit the agent made in place, without returning it.
            (bundle.directory / "kernel.py").write_text("unreturned edit")
            if failure == "unknown_path":
                return {"original.py": "bad"}
            if failure == "bad_source":
                return {"kernel.py": 42}
            return ["kernel.py"]
        return {"kernel.py": "faster"}

    result = bundle.run_agent_rounds(agent, budget_seconds=2, log=session_clock.log)
    assert result.rounds[0].reason == "invalid_proposal"
    assert result.rounds[0].error is not None
    assert requests[1].feedback["reason"] == "invalid_proposal"
    assert requests[1].feedback["error"] == result.rounds[0].error
    assert result.rounds[1].accepted
    assert bundle.sources == {"kernel.py": "faster"}
    assert (bundle.directory / "original.py").read_text() == "original"


@pytest.mark.parametrize("submissions", [0, 1])
def test_agent_exception_ends_session(bundle, monkeypatch, submissions):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    calls = 0
    log = Mock()

    def agent(request):
        nonlocal calls
        calls += 1
        (bundle.directory / "kernel.py").write_text("unreturned edit")
        if calls > submissions:
            raise RuntimeError("provider failed")
        return {"kernel.py": "faster"}

    result = bundle.run_agent_rounds(agent, budget_seconds=30, log=log)
    assert result.stop_reason == "agent_error"
    assert calls == submissions + 1
    assert len(result.rounds) == submissions
    assert len(list(Path(result.directory).glob("round_*"))) == submissions
    assert bundle.sources == {"kernel.py": "faster" if submissions else "original"}
    # A warning with the traceback, not an INFO line.
    (warning,) = log.warning.call_args_list
    assert "RuntimeError: provider failed" in warning.args[0]
    assert warning.kwargs["exc_info"] is True


@pytest.mark.parametrize("submissions", [0, 1])
def test_agent_returning_none_ends_session_without_extra_round(
    bundle, monkeypatch, submissions
):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    proposals = iter(["faster"] * submissions)

    def agent(request):
        source = next(proposals, None)
        return None if source is None else {"kernel.py": source}

    result = bundle.run_agent_rounds(agent, budget_seconds=30)
    assert result.stop_reason == "agent_stopped"
    assert len(result.rounds) == submissions
    assert len(list(Path(result.directory).glob("round_*"))) == submissions
    assert bundle.sources == {"kernel.py": "faster" if submissions else "original"}
    assert bundle.prompt == "Original source instructions."


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
    requests = []

    def agent(request):
        requests.append(request)
        assert "Session deadline (time.monotonic()): 105.0." in request.prompt
        if late_stage == "callback":
            clock[0] = 106
        return {"kernel.py": "faster"}

    result = bundle.run_agent_rounds(agent, budget_seconds=5)
    assert result.stop_reason == "deadline"
    assert len(requests) == 1
    assert len(result.rounds) == 1
    assert result.rounds[0].reason == "deadline"
    assert not result.rounds[0].accepted
    assert len(calls) == (1 if late_stage == "callback" else 3)
    assert all(call["deadline"] == 105 for call in calls)
    assert bundle.sources == {"kernel.py": "original"}


def test_deadline_while_agent_works_has_no_extra_round(bundle, monkeypatch):
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
    calls = 0

    def agent(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"kernel.py": "faster"}
        # An agent that waits for its proposal until the deadline gives up.
        clock[0] = request.deadline
        return None

    result = bundle.run_agent_rounds(agent, budget_seconds=30)
    assert result.stop_reason == "deadline"
    assert calls == 2
    assert len(result.rounds) == 1 and result.rounds[0].accepted
    assert bundle.sources == {"kernel.py": "faster"}


def test_baseline_failure_does_not_call_agent(bundle, monkeypatch):
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(workspace, 10, ok=False),
    )

    def agent(request):
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
        bundle.run_agent_rounds(lambda request: {}, budget_seconds=budget_seconds)
    assert not (bundle.directory / "agent_runs").exists()


@pytest.mark.parametrize("use_bundle_method", [False, True])
def test_agent_and_budget_are_required_before_evaluation(bundle, use_bundle_method):
    with pytest.raises(TypeError, match="budget_seconds"):
        if use_bundle_method:
            bundle.run_agent_rounds(lambda request: {})
        else:
            run_agent_rounds(bundle, lambda request: {})
    with pytest.raises(TypeError, match="agent"):
        if use_bundle_method:
            bundle.run_agent_rounds(budget_seconds=30)
        else:
            run_agent_rounds(bundle, budget_seconds=30)
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
        lambda request: {"kernel.py": "faster"},
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
        lambda request: request.sources, budget_seconds=1, log=session_clock.log
    )
    assert calls == 1
    assert not result.rounds[0].accepted
    assert result.rounds[0].reason == "invalid_proposal"
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
        lambda request: {"kernel.py": "faster"},
        budget_seconds=1,
        log=session_clock.log,
    )

    def agent(request):
        assert request.prompt == (
            "Original source instructions.\n\n"
            "Session deadline (time.monotonic()): 102.0."
        )
        assert first.directory not in request.prompt
        assert request.feedback is None
        assert request.sources == {"kernel.py": "faster"}
        return request.sources

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


@pytest.mark.parametrize("use_bundle_method", [False, True])
def test_feedback_reports_each_round_after_it_is_recorded(
    bundle, monkeypatch, use_bundle_method
):
    clock = [100.0]
    monkeypatch.setattr(
        "helion.autotuner.handoff_agent.time.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace,
            {"original": 10, "faster": 9, "slower": 12}[workspace.sources["kernel.py"]],
        ),
    )
    log = Mock()
    proposals = iter(["faster", "slower"])
    requests = []

    def agent(request):
        requests.append(request)
        if request.feedback is not None:
            # The round is saved and logged before the agent hears of it.
            event = log.record_handoff_event.call_args_list[-1]
            assert event.args == ("handoff_round",)
            assert event.kwargs["round"]["round"] == request.feedback["round"]
            saved = Path(event.kwargs["round"]["directory"]) / "round.json"
            assert json.loads(saved.read_text()) == json.loads(
                json.dumps(event.kwargs["round"])
            )
        clock[0] += 1
        source = next(proposals, None)
        return None if source is None else {"kernel.py": source}

    result = (
        bundle.run_agent_rounds(agent, budget_seconds=30, log=log)
        if use_bundle_method
        else run_agent_rounds(bundle, agent, budget_seconds=30, log=log)
    )
    assert result.stop_reason == "agent_stopped"
    assert [record.accepted for record in result.rounds] == [True, False]
    assert result.evaluation.perf == 9
    first, second, third = (request.feedback for request in requests)
    assert first is None
    summary = {
        "ok": True,
        "unit": "ms",
        "cases": [{"index": 0, "status": "ok", "noise": 0.01}],
    }

    def measured(perf):
        return {
            **summary,
            "perf": perf,
            "cases": [{**summary["cases"][0], "perf": perf}],
        }

    assert second == {
        "round": 1,
        "accepted": True,
        "reason": "accepted",
        "current": measured(9),
        "candidate": measured(9),
        "incumbent": measured(10),
        "improvement_threshold": pytest.approx(0.03),
        "remaining_seconds": 29,
    }
    assert third["round"] == 2
    assert third["reason"] == "regression_or_noise"
    assert third["current"] == measured(9)
    assert third["candidate"] == measured(12)
    assert third["remaining_seconds"] == 28
    assert "error" not in third
    assert [request.sources for request in requests] == [
        {"kernel.py": "original"},
        {"kernel.py": "faster"},
        {"kernel.py": "faster"},
    ]
    assert [call.args[0] for call in log.record_handoff_event.call_args_list] == [
        "handoff_baseline",
        "handoff_round",
        "handoff_round",
        "handoff_completed",
    ]
    assert any(
        "Native source baseline: 10 ms" in call.args[0] for call in log.call_args_list
    )


@pytest.mark.parametrize("unchanged", [False, True])
def test_consecutive_invalid_proposals_end_session(bundle, monkeypatch, unchanged):
    monkeypatch.setattr("helion.autotuner.handoff_agent.time.monotonic", lambda: 100)
    monkeypatch.setattr(
        HandoffBundle,
        "evaluate",
        lambda workspace, **kwargs: evaluation(
            workspace, 10 if workspace.sources["kernel.py"] == "original" else 9
        ),
    )
    calls = 0

    def agent(request):
        nonlocal calls
        calls += 1
        if calls == 5:
            # One valid proposal resets the count of invalid ones.
            return {"kernel.py": "faster"}
        return dict(request.sources) if unchanged else {}

    result = bundle.run_agent_rounds(agent, budget_seconds=30)
    assert result.stop_reason == "invalid_proposals"
    assert calls == 5 + MAX_CONSECUTIVE_INVALID_PROPOSALS
    assert [r.reason for r in result.rounds[3:6]] == [
        "invalid_proposal",
        "accepted",
        "invalid_proposal",
    ]
    assert bundle.sources == {"kernel.py": "faster"}


def test_proposals_have_no_count_limit_while_time_remains(bundle, monkeypatch):
    monkeypatch.setattr("helion.autotuner.handoff_agent.time.monotonic", lambda: 100)

    def measure(workspace, **kwargs):
        source = workspace.sources["kernel.py"]
        return evaluation(workspace, 100 if source == "original" else 100 - int(source))

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    proposals = iter(str(number) for number in range(1, 13))

    def agent(request):
        source = next(proposals, None)
        return None if source is None else {"kernel.py": source}

    result = bundle.run_agent_rounds(agent, budget_seconds=30)
    assert result.stop_reason == "agent_stopped"
    assert len(result.rounds) == 12
    assert all(record.accepted for record in result.rounds)
    assert bundle.sources == {"kernel.py": "12"}
    session = json.loads((Path(result.directory) / "session.json").read_text())
    assert "max_rounds" not in session
    assert session["budget_seconds"] == 30


@pytest.mark.parametrize("stage", ["waiting", "evaluation"])
def test_keyboard_interrupt_restores_sources(bundle, monkeypatch, stage):
    def measure(workspace, **kwargs):
        if workspace.sources["kernel.py"] == "faster":
            raise KeyboardInterrupt
        return evaluation(workspace, 10)

    def agent(request):
        (bundle.directory / "kernel.py").write_text("unreturned edit")
        if stage == "waiting":
            raise KeyboardInterrupt
        return {"kernel.py": "faster"}

    monkeypatch.setattr(HandoffBundle, "evaluate", measure)
    with pytest.raises(KeyboardInterrupt):
        bundle.run_agent_rounds(agent, budget_seconds=30)
    assert bundle.sources == {"kernel.py": "original"}
    assert bundle.prompt == "Original source instructions."
