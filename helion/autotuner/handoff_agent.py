"""Evaluate a caller-supplied source agent's proposals against a handoff bundle."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import datetime
import json
import math
import time
from typing import TYPE_CHECKING
from typing import Any
import uuid

from .handoff_evaluation import _artifact_path
from .handoff_prompt import _evaluation_summary

if TYPE_CHECKING:
    from pathlib import Path

    from .handoff_bundle import HandoffBundle
    from .handoff_evaluation import HandoffEvaluation
    from .logger import AutotuningLogger


@dataclasses.dataclass(frozen=True)
class HandoffRequest:
    """A request for the next proposal, passed to a :data:`SourceAgent`.

    The agent answers with complete replacement sources for one or more of
    ``sources``, keyed by the same paths, or ``None`` to end the session.
    Helion only validates and measures what the agent returns: how the agent
    produces it (a model API, a coding-agent CLI, a person), where it runs,
    and what it may access are the agent's concern. Calls are synchronous and
    Helion does not interrupt them, so the agent must return by ``deadline``.
    Every hand-off session (one per autotuned specialization) calls the same
    agent; a session's first request has ``feedback`` None, and Helion does
    not signal a session's end.
    """

    # The task and its contracts (the bundle's prompt.md), and the deadline.
    prompt: str
    # The retained native sources, keyed by bundle-relative path.
    sources: Mapping[str, str]
    # The outcome of the previous proposal (see run_agent_rounds); None at first.
    feedback: Mapping[str, Any] | None
    # Absolute time.monotonic() deadline of the session.
    deadline: float


# Returns replacement sources to evaluate, or None to end the session.
SourceAgent = Callable[[HandoffRequest], Mapping[str, str] | None]

# Invalid proposals cost no evaluation, so an agent that keeps returning them
# would otherwise spin through rounds until the deadline.
MAX_CONSECUTIVE_INVALID_PROPOSALS = 20


@dataclasses.dataclass(frozen=True)
class HandoffAgentRound:
    """One proposal, its nearby comparison, and the retention decision."""

    round: int
    accepted: bool
    reason: str
    candidate: HandoffEvaluation | None
    incumbent: HandoffEvaluation | None
    improvement_threshold: float | None
    error: str | None
    directory: str
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class HandoffAgentResult:
    """A source session; the workspace contains the returned evaluation's source."""

    baseline: HandoffEvaluation
    evaluation: HandoffEvaluation
    rounds: tuple[HandoffAgentRound, ...]
    stop_reason: str
    directory: str
    elapsed_seconds: float


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_sources(
    directory: Path, sources: Mapping[str, str], *, readonly: bool = False
) -> None:
    directory = directory.resolve()
    writes = [
        (_artifact_path(directory, relative), source)
        for relative, source in sources.items()
    ]
    for path, source in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        if readonly:
            path.chmod(0o444)


def _noise(evaluation: HandoffEvaluation) -> float:
    """Conservative spread in objective units, including ratios/multiple shapes."""
    assert evaluation.ok and evaluation.perf is not None
    relative_noise = 0.0
    for case in evaluation.cases:
        assert case.noise is not None and case.perf is not None
        relative_noise = max(relative_noise, case.noise / case.perf)
    # Both max and geometric mean are monotonic and scale with their inputs.
    # The largest relative case MAD bounds their corresponding relative spread.
    return evaluation.perf * relative_noise


def _feedback(
    record: HandoffAgentRound, incumbent: HandoffEvaluation, deadline: float
) -> dict[str, Any]:
    """The outcome of one round, as the next request reports it to the agent."""
    return {
        "round": record.round,
        "accepted": record.accepted,
        "reason": record.reason,
        # The retained source's latest measurement.
        "current": _evaluation_summary(dataclasses.asdict(incumbent)),
        **{
            name: _evaluation_summary(dataclasses.asdict(evaluation))
            for name, evaluation in (
                ("candidate", record.candidate),
                ("incumbent", record.incumbent),
            )
            if evaluation is not None
        },
        **({"error": record.error} if record.error is not None else {}),
        **(
            {"improvement_threshold": record.improvement_threshold}
            if record.improvement_threshold is not None
            else {}
        ),
        "remaining_seconds": max(0.0, deadline - time.monotonic()),
    }


def run_agent_rounds(
    bundle: HandoffBundle,  # Native sources, inputs, and references to optimize.
    agent: SourceAgent,  # Proposes replacement sources; see HandoffRequest.
    *,
    budget_seconds: float,  # Total session budget, in seconds.
    repetitions: int = 5,  # Timing samples per case for median and noise estimates.
    timeout: float = 120,  # Per-case evaluation seconds, capped by session budget.
    log: AutotuningLogger | None = None,  # Optional progress and evaluation logger.
) -> HandoffAgentResult:
    """Retain validated improvements, restoring the incumbent after every failure.

    Each round calls ``agent`` with a :class:`HandoffRequest`: the bundle's
    prompt with the session deadline, the retained sources, and the previous
    round's feedback (its decision and reason, summaries of the candidate's
    and the incumbent's measurements and of the retained source's latest one,
    any error, the improvement threshold, and the seconds remaining). The
    proposal is measured against the incumbent and retained only if it is
    faster by more than the noise. The session ends at the deadline, when
    the agent returns ``None`` (``agent_stopped``), when it raises
    (``agent_error``), or after ``MAX_CONSECUTIVE_INVALID_PROPOSALS`` invalid
    proposals in a row (``invalid_proposals``). ``budget_seconds`` includes baseline evaluation, agent
    calls, and comparisons. Calls are synchronous: the agent must return by
    the request's deadline, and proposals completed late are not accepted.
    Sources, feedback, and decisions are saved under ``agent_runs/``.
    """
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")

    started = time.monotonic()
    deadline = started + budget_seconds
    directory = bundle.directory / "agent_runs" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    prompt = bundle.prompt + f"\n\nSession deadline (time.monotonic()): {deadline}."
    incumbent_sources = bundle.sources
    _write_sources(directory / "initial", incumbent_sources, readonly=True)
    _write_json(
        directory / "session.json",
        {
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "budget_seconds": budget_seconds,
            "deadline_monotonic": deadline,
            "repetitions": repetitions,
            "timeout": timeout,
        },
    )
    (directory / "prompt.md").write_text(prompt)
    rounds: list[HandoffAgentRound] = []
    feedback: dict[str, Any] | None = None
    invalid_streak = 0

    def expired() -> bool:
        return time.monotonic() >= deadline

    def evaluate(sources: Mapping[str, str]) -> HandoffEvaluation:
        if expired():
            raise TimeoutError("Source-agent deadline reached")
        _write_sources(bundle.directory, sources)
        return bundle.evaluate(
            repetitions=repetitions, timeout=timeout, deadline=deadline
        )

    stop_reason = "interrupted"
    try:
        baseline = bundle.evaluate(
            repetitions=repetitions, timeout=timeout, deadline=deadline
        )
        _write_json(directory / "baseline.json", dataclasses.asdict(baseline))
        if log is not None:
            log.record_handoff_event(
                "handoff_baseline", evaluation=dataclasses.asdict(baseline)
            )
            log(
                f"Native source baseline: {baseline.perf} {baseline.unit}"
                if baseline.ok
                else "Native source baseline failed validation or measurement"
            )
        incumbent = baseline
        while baseline.ok and not expired():
            number = len(rounds) + 1
            if log is not None:
                log(f"Native source round {number}: requesting proposal")
            request = HandoffRequest(
                prompt, dict(incumbent_sources), feedback, deadline
            )
            try:
                replacements = agent(request)
            except Exception as exception:
                stop_reason = "agent_error"
                if log is not None:
                    log.warning(
                        "Native source agent failed: "
                        f"{type(exception).__qualname__}: {exception}",
                        exc_info=True,
                    )
                break
            if replacements is None:
                stop_reason = "deadline" if expired() else "agent_stopped"
                break
            round_directory = directory / f"round_{number:04d}"
            round_directory.mkdir()
            _write_sources(
                round_directory / "incumbent", incumbent_sources, readonly=True
            )
            candidate = comparison = None
            accepted = False
            threshold = None
            error = None
            reason = "invalid_proposal"
            try:
                if (
                    not isinstance(replacements, Mapping)
                    or not replacements
                    or not replacements.keys() <= incumbent_sources.keys()
                ):
                    raise ValueError(
                        "Agent must replace one or more existing native source files"
                    )
                if any(not isinstance(source, str) for source in replacements.values()):
                    raise TypeError(
                        "Agent replacements must contain complete source strings"
                    )
                proposed = {**incumbent_sources, **replacements}
                _write_sources(round_directory / "candidate", proposed, readonly=True)
                if proposed == incumbent_sources:
                    raise ValueError("Agent proposal did not change native source")
                if log is not None:
                    log(f"Native source round {number}: evaluating proposal")
                order = (
                    ("candidate", "incumbent")
                    if number % 2
                    else ("incumbent", "candidate")
                )
                for name in order:
                    reason = f"{name}_failed"
                    measured = evaluate(
                        proposed if name == "candidate" else incumbent_sources
                    )
                    if name == "candidate":
                        candidate = measured
                    else:
                        comparison = measured
                    if not measured.ok:
                        break
                if (
                    candidate is not None
                    and candidate.ok
                    and comparison is not None
                    and comparison.ok
                ):
                    assert candidate.perf is not None and comparison.perf is not None
                    threshold = max(
                        3 * max(_noise(candidate), _noise(comparison)),
                        0.001 * comparison.perf,
                    )
                    accepted = (
                        not expired() and comparison.perf - candidate.perf > threshold
                    )
                    reason = "accepted" if accepted else "regression_or_noise"
                if expired():
                    accepted = False
                    reason = "deadline"
                if comparison is not None and comparison.ok:
                    incumbent = comparison
                if accepted:
                    assert candidate is not None
                    incumbent_sources = proposed
                    incumbent = candidate
            except Exception as exception:
                error = f"{type(exception).__qualname__}: {exception}"
                if expired():
                    reason = "deadline"
            finally:
                _write_sources(bundle.directory, incumbent_sources)
            record = HandoffAgentRound(
                number,
                accepted,
                reason,
                candidate,
                comparison,
                threshold,
                error,
                str(round_directory),
                time.monotonic() - started,
            )
            rounds.append(record)
            _write_json(round_directory / "round.json", dataclasses.asdict(record))
            feedback = _feedback(record, incumbent, deadline)
            _write_json(round_directory / "feedback.json", feedback)
            if log is not None:
                log.record_handoff_event(
                    "handoff_round", round=dataclasses.asdict(record)
                )
                log(f"Native source round {number}: {reason}")
            invalid_streak = invalid_streak + 1 if reason == "invalid_proposal" else 0
            if invalid_streak >= MAX_CONSECUTIVE_INVALID_PROPOSALS:
                stop_reason = "invalid_proposals"
                if log is not None:
                    log.warning(
                        f"Native source agent made {invalid_streak} invalid "
                        "proposals in a row; ending the session"
                    )
                break
        else:
            if not baseline.ok:
                stop_reason = "baseline_failed"
            else:
                stop_reason = "deadline"
        result = HandoffAgentResult(
            baseline,
            incumbent,
            tuple(rounds),
            stop_reason,
            str(directory),
            time.monotonic() - started,
        )
        _write_json(directory / "result.json", dataclasses.asdict(result))
        if log is not None:
            log.record_handoff_event(
                "handoff_completed", result=dataclasses.asdict(result)
            )
            log(f"Native source session completed: {stop_reason}, {len(rounds)} rounds")
        return result
    finally:
        _write_sources(bundle.directory, incumbent_sources)
