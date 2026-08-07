"""Dependency-light observations for one RAG-assisted autotuning attempt.

The collector deliberately performs no logging and imports no experiment-harness
code.  Core search paths can record evaluations and phase transitions here; the
terminal adapter copies the frozen records into its single canonical event.
"""

from __future__ import annotations

import dataclasses
import math
from threading import Lock
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import Phase


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Result and incumbent state after one ordered candidate evaluation."""

    sequence: int
    trajectory_sequence: int
    config_id: str | None
    config_repr: str
    candidate_source: str
    candidate_category: str
    compatibility_status: str
    compilation_status: str
    compilation_seconds: float | None
    correctness_status: str
    benchmark_status: str
    timeout_status: str
    performance: float | None
    elapsed_seconds: float
    incumbent_best_perf: float | None


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseSnapshot:
    """Elapsed and incumbent state at one ordered phase transition."""

    trajectory_sequence: int
    phase: Phase
    elapsed_seconds: float
    phase_seconds: float | None
    incumbent_best_perf: float | None


def _finite_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _finite_duration(value: float | None) -> float | None:
    result = _finite_float(value)
    return result if result is not None and result >= 0.0 else None


class InstrumentationCollector:
    """Collect ordered, JSON-safe observations for a single attempt.

    Lower performance is better.  Only a finite performance from a successfully
    compiled, correct, and benchmarked candidate can update the incumbent.
    """

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        started_at = _finite_float(clock())
        self._started_at = started_at
        self._last_elapsed_seconds = 0.0
        self._incumbent_best_perf: float | None = None
        self._trajectory_sequence = 0
        self._evaluations: list[EvaluationRecord] = []
        self._phase_snapshots: list[PhaseSnapshot] = []
        self._lock = Lock()

    @property
    def evaluations(self) -> tuple[EvaluationRecord, ...]:
        """Return an immutable snapshot in recording order."""
        with self._lock:
            return tuple(self._evaluations)

    @property
    def phase_snapshots(self) -> tuple[PhaseSnapshot, ...]:
        """Return an immutable snapshot in transition order."""
        with self._lock:
            return tuple(self._phase_snapshots)

    @property
    def incumbent_best_perf(self) -> float | None:
        """Return the best finite validated performance observed so far."""
        with self._lock:
            return self._incumbent_best_perf

    def _elapsed_seconds(self) -> float:
        now = _finite_float(self._clock())
        if now is None:
            return self._last_elapsed_seconds
        if self._started_at is None:
            self._started_at = now
            return self._last_elapsed_seconds
        elapsed = now - self._started_at
        if elapsed < self._last_elapsed_seconds:
            return self._last_elapsed_seconds
        self._last_elapsed_seconds = elapsed
        return elapsed

    def record_evaluation(
        self,
        *,
        config_id: str | None,
        config_repr: str,
        candidate_source: str,
        candidate_category: str,
        compatibility_status: str,
        compilation_status: str,
        compilation_seconds: float | None,
        correctness_status: str,
        benchmark_status: str,
        timeout_status: str,
        performance: float | None,
    ) -> EvaluationRecord:
        """Record one completed evaluation and update its validated incumbent."""
        with self._lock:
            validated_performance = _finite_float(performance)
            if not (
                compilation_status == "ok"
                and correctness_status == "ok"
                and benchmark_status == "ok"
                and timeout_status == "not_timed_out"
            ):
                validated_performance = None
            if validated_performance is not None:
                if self._incumbent_best_perf is None:
                    self._incumbent_best_perf = validated_performance
                else:
                    self._incumbent_best_perf = min(
                        self._incumbent_best_perf, validated_performance
                    )
            self._trajectory_sequence += 1
            record = EvaluationRecord(
                sequence=len(self._evaluations) + 1,
                trajectory_sequence=self._trajectory_sequence,
                config_id=config_id,
                config_repr=config_repr,
                candidate_source=candidate_source,
                candidate_category=candidate_category,
                compatibility_status=compatibility_status,
                compilation_status=compilation_status,
                compilation_seconds=_finite_duration(compilation_seconds),
                correctness_status=correctness_status,
                benchmark_status=benchmark_status,
                timeout_status=timeout_status,
                performance=validated_performance,
                elapsed_seconds=self._elapsed_seconds(),
                incumbent_best_perf=self._incumbent_best_perf,
            )
            self._evaluations.append(record)
            return record

    def record_phase_transition(
        self,
        phase: Phase,
        *,
        phase_seconds: float | None,
    ) -> PhaseSnapshot:
        """Record one transition, preserving unavailable phase time as ``None``."""
        with self._lock:
            self._trajectory_sequence += 1
            snapshot = PhaseSnapshot(
                trajectory_sequence=self._trajectory_sequence,
                phase=phase,
                elapsed_seconds=self._elapsed_seconds(),
                phase_seconds=_finite_duration(phase_seconds),
                incumbent_best_perf=self._incumbent_best_perf,
            )
            self._phase_snapshots.append(snapshot)
            return snapshot
