"""Run provider-independent native-source proposals against a handoff bundle."""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import time
from typing import TYPE_CHECKING
import uuid

from .handoff_cli import CLISourceAgent
from .handoff_evaluation import _artifact_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from pathlib import Path

    from .handoff_bundle import HandoffBundle
    from .handoff_evaluation import HandoffEvaluation
    from .logger import AutotuningLogger


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


def run_agent_rounds(
    bundle: HandoffBundle,  # Native sources, inputs, and references to optimize.
    agent: Callable[[HandoffBundle], Mapping[str, str]]
    | None = None,  # None uses the CLI selected by HELION_HANDOFF_AGENT.
    *,
    budget_seconds: float,  # Total session budget, in seconds.
    repetitions: int = 5,  # Timing samples per case for median and noise estimates.
    timeout: float = 120,  # Per-case evaluation seconds, capped by session budget.
    log: AutotuningLogger | None = None,  # Optional progress and evaluation logger.
) -> HandoffAgentResult:
    """Retain validated improvements, restoring the incumbent after every failure.

    ``budget_seconds`` includes baseline evaluation, callbacks, and comparisons.
    Rounds track progress without limiting the number of submissions. Callbacks
    are synchronous and must enforce their own timeout using the absolute
    ``time.monotonic()`` deadline in the prompt. Late proposals are never
    accepted. Each invocation starts a separate log directory, without resuming
    earlier agent context. By default, :class:`CLISourceAgent` runs one
    continuous CLI session; each explicit candidate submission records a
    round and returns evaluation feedback to that session. Explicit callbacks
    retain control of provider and context management.
    """
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")

    started = time.monotonic()
    deadline = started + budget_seconds
    directory = bundle.directory / "agent_runs" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    source_agent = None
    if agent is None:
        source_agent = CLISourceAgent(directory / "calls", deadline=deadline, log=log)
        agent = source_agent
    elif isinstance(agent, CLISourceAgent):
        source_agent = agent
        if source_agent.deadline is not None:
            deadline = min(deadline, source_agent.deadline)
        source_agent.deadline = deadline
    prompt_path = bundle.directory / "prompt.md"
    original_prompt = prompt_path.read_text()
    prompt = original_prompt + f"\n\nSession deadline (time.monotonic()): {deadline}."
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
    rounds: list[HandoffAgentRound] = []

    def expired() -> bool:
        return time.monotonic() >= deadline

    def evaluate(sources: Mapping[str, str]) -> HandoffEvaluation:
        if expired():
            raise TimeoutError("Source-agent deadline reached")
        _write_sources(bundle.directory, sources)
        return bundle.evaluate(
            repetitions=repetitions, timeout=timeout, deadline=deadline
        )

    def feedback(evaluation: HandoffEvaluation) -> dict[str, object]:
        return {
            "round": len(rounds) + 1,
            "deadline_monotonic": deadline,
            "remaining_seconds": max(0.0, deadline - time.monotonic()),
            "incumbent_evaluation": dataclasses.asdict(evaluation),
            "history": [dataclasses.asdict(record) for record in rounds],
            "log_directory": str(directory),
        }

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
        _write_json(directory / "feedback.json", feedback(incumbent))
        (directory / "prompt.md").write_text(prompt)
        while baseline.ok and not expired():
            number = len(rounds) + 1
            prompt_path.write_text(prompt)
            replacements = None
            if source_agent is not None:
                # Waiting for a submission is not itself an optimization round.
                try:
                    if log is not None:
                        log("Native source session: waiting for candidate submission")
                    replacements = source_agent(bundle)
                except StopIteration:
                    assert source_agent.stop_reason is not None
                    stop_reason = source_agent.stop_reason
                    break
                except Exception as exception:
                    stop_reason = "agent_error"
                    if log is not None:
                        log(f"Native source session failed: {exception}")
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
            reason = "agent_error"
            try:
                if source_agent is None:
                    if log is not None:
                        log(f"Native source round {number}: requesting proposal")
                    replacements = agent(bundle)
                elif log is not None:
                    log(f"Native source round {number}: evaluating submission")
                if (
                    not replacements
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
            if log is not None:
                log.record_handoff_event(
                    "handoff_round", round=dataclasses.asdict(record)
                )
                log(f"Native source round {number}: {reason}")
            context = feedback(incumbent)
            _write_json(directory / "feedback.json", context)
            if source_agent is not None:
                try:
                    source_agent.complete_round(
                        incumbent_sources,
                        context,
                        done=expired(),
                    )
                except Exception as exception:
                    stop_reason = "agent_error"
                    if log is not None:
                        log(f"Native source feedback failed: {exception}")
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
        try:
            if source_agent is not None:
                source_agent.close(reason=stop_reason)
        finally:
            _write_sources(bundle.directory, incumbent_sources)
            prompt_path.write_text(original_prompt)
